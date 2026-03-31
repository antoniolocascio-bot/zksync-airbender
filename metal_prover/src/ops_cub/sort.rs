///! Parallel radix sort on Metal.
///!
///! Replaces CUB's `DeviceRadixSort::SortKeys` and `SortPairs`.
///!
///! The Metal implementation uses separate histogram and scatter kernels:
///! - `ab_radix_sort_histogram_kernel`: counts per-block per-digit frequencies
///! - `ab_radix_sort_scatter_kernel`: scatters keys to output using prefix sums
///!
///! The host computes the global prefix sums between the two kernel dispatches
///! (Apple Silicon unified memory makes CPU↔GPU data exchange zero-copy).

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_structures::MetalBuffer;
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;

type BF = BaseField;

/// Matches the Metal shader constants.
const RADIX_BITS: u32 = 4;
const RADIX_SIZE: usize = 1 << RADIX_BITS; // 16
const SORT_BLOCK_SIZE: u32 = 256;

fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
}

/// Sort direction.
#[derive(Copy, Clone, Debug)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Radix sort of u32 keys (ascending only).
///
/// Sorts `keys_in` over bits `[begin_bit, end_bit)` and writes to `keys_out`.
/// Uses a multi-pass LSD radix sort: RADIX_BITS bits per pass, one histogram
/// kernel + CPU prefix-sum + one scatter kernel per pass.
pub fn sort_keys_u32(
    keys_in: &MetalBuffer<u32>,
    keys_out: &mut MetalBuffer<u32>,
    order: SortOrder,
    begin_bit: u32,
    end_bit: u32,
    ctx: &MetalProverContext,
) {
    assert_eq!(keys_in.len(), keys_out.len());
    let num_items = keys_in.len() as u32;

    if let SortOrder::Descending = order {
        todo!("sort_keys_u32: Descending sort not yet implemented");
    }

    let num_passes = (end_bit - begin_bit + RADIX_BITS - 1) / RADIX_BITS;
    if num_passes == 0 || num_items == 0 {
        keys_out.load_from_host(keys_in.as_slice());
        return;
    }

    let num_threadgroups =
        ((num_items + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE).max(1) as usize;

    // Scratch buffers
    let mut temp = MetalBuffer::<u32>::new(&ctx.device, keys_in.len());
    let mut hist_buf =
        MetalBuffer::<u32>::new(&ctx.device, num_threadgroups * RADIX_SIZE);
    let mut prefix_buf =
        MetalBuffer::<u32>::new(&ctx.device, num_threadgroups * RADIX_SIZE);

    // Histogram kernel needs RADIX_SIZE u32 slots.
    // Scatter kernel needs (RADIX_SIZE + SORT_BLOCK_SIZE) u32 slots for stable rank computation.
    let hist_threadgroup_mem = (RADIX_SIZE * std::mem::size_of::<u32>()) as u64;
    let scatter_threadgroup_mem =
        ((RADIX_SIZE + SORT_BLOCK_SIZE as usize) * std::mem::size_of::<u32>()) as u64;

    let block = MTLSize::new(SORT_BLOCK_SIZE as u64, 1, 1);
    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);

    let hist_pipeline = ctx.get_pipeline("ab_radix_sort_histogram_kernel");
    let scatter_pipeline = ctx.get_pipeline("ab_radix_sort_scatter_kernel");

    // Ping-pong between keys_out (state 0) and temp (state 1).
    // Pass 0: src = keys_in → dst = state 1 (temp), current_state becomes 1
    // Pass 1: src = state 1 (temp) → dst = state 0 (keys_out), current_state becomes 0
    // ...
    // After num_passes: current_state = num_passes % 2
    // If current_state == 1 (odd num_passes): result is in temp → copy to keys_out.
    let mut current_state: usize = 0;

    for pass in 0..num_passes {
        let current_bit = (begin_bit + pass * RADIX_BITS) as i32;
        let next_state = 1 - current_state;

        // Cloned Metal buffer handles (cheap: just increments ObjC ref-count).
        let src_metal = if pass == 0 {
            keys_in.metal_buffer().clone()
        } else if current_state == 0 {
            keys_out.metal_buffer().clone()
        } else {
            temp.metal_buffer().clone()
        };

        let dst_metal = if next_state == 0 {
            keys_out.metal_buffer().clone()
        } else {
            temp.metal_buffer().clone()
        };

        // Step 1: histogram – each threadgroup counts per-digit frequencies.
        {
            let cb = ctx.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&hist_pipeline);
            enc.set_buffer(0, Some(&src_metal), 0);
            enc.set_buffer(1, Some(hist_buf.metal_buffer()), 0);
            set_bytes(&enc, &num_items, 2); // constant unsigned &num_items
            set_bytes(&enc, &current_bit, 3); // constant int &current_bit
            enc.set_threadgroup_memory_length(0, hist_threadgroup_mem);
            enc.dispatch_thread_groups(grid, block);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Step 2: CPU-side global prefix sum (unified memory → zero-copy).
        //
        // hist_buf layout: hist[bid * RADIX_SIZE + d] = count of digit d in block bid.
        // prefix_buf layout: prefix[bid * RADIX_SIZE + d] = starting output position
        //   for elements with digit d that come from block bid.
        {
            let hist = hist_buf.as_slice();
            let prefix = prefix_buf.as_mut_slice();

            // 1. Compute total count per digit across all blocks.
            let mut digit_total = [0u32; RADIX_SIZE];
            for bid in 0..num_threadgroups {
                for d in 0..RADIX_SIZE {
                    digit_total[d] = digit_total[d].wrapping_add(hist[bid * RADIX_SIZE + d]);
                }
            }

            // 2. Exclusive prefix sum of digit_total → global start offset per digit.
            let mut global_prefix = [0u32; RADIX_SIZE];
            let mut running = 0u32;
            for d in 0..RADIX_SIZE {
                global_prefix[d] = running;
                running = running.wrapping_add(digit_total[d]);
            }

            // 3. For each block, compute the per-(block, digit) starting offset:
            //    prefix[bid][d] = global_prefix[d] + sum_{bid' < bid} hist[bid'][d]
            let mut per_block_cumsum = [0u32; RADIX_SIZE];
            for bid in 0..num_threadgroups {
                for d in 0..RADIX_SIZE {
                    prefix[bid * RADIX_SIZE + d] =
                        global_prefix[d].wrapping_add(per_block_cumsum[d]);
                    per_block_cumsum[d] =
                        per_block_cumsum[d].wrapping_add(hist[bid * RADIX_SIZE + d]);
                }
            }
        }

        // Step 3: scatter – reorder keys from src to dst using prefix sums.
        {
            let cb = ctx.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&scatter_pipeline);
            enc.set_buffer(0, Some(&src_metal), 0);
            enc.set_buffer(1, Some(&dst_metal), 0);
            enc.set_buffer(2, Some(prefix_buf.metal_buffer()), 0);
            set_bytes(&enc, &num_items, 3); // constant unsigned &num_items
            set_bytes(&enc, &current_bit, 4); // constant int &current_bit
            enc.set_threadgroup_memory_length(0, scatter_threadgroup_mem);
            enc.dispatch_thread_groups(grid, block);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        current_state = next_state;
    }

    // If the result ended up in temp (odd number of passes), copy to keys_out.
    if current_state == 1 {
        keys_out.load_from_host(temp.as_slice());
    }
}

/// Sort key-value pairs: sorts `keys_in` and applies the same permutation to `values_in`.
pub fn sort_pairs_u32(
    keys_in: &MetalBuffer<u32>,
    keys_out: &mut MetalBuffer<u32>,
    values_in: &MetalBuffer<u32>,
    values_out: &mut MetalBuffer<u32>,
    order: SortOrder,
    begin_bit: u32,
    end_bit: u32,
    ctx: &MetalProverContext,
) {
    assert_eq!(keys_in.len(), keys_out.len());
    assert_eq!(keys_in.len(), values_in.len());
    assert_eq!(keys_in.len(), values_out.len());
    let num_items = keys_in.len() as u32;

    if let SortOrder::Descending = order {
        todo!("sort_pairs_u32: Descending sort not yet implemented");
    }

    let num_passes = (end_bit - begin_bit + RADIX_BITS - 1) / RADIX_BITS;
    if num_passes == 0 || num_items == 0 {
        keys_out.load_from_host(keys_in.as_slice());
        values_out.load_from_host(values_in.as_slice());
        return;
    }

    let num_threadgroups =
        ((num_items + SORT_BLOCK_SIZE - 1) / SORT_BLOCK_SIZE).max(1) as usize;

    let mut temp_keys = MetalBuffer::<u32>::new(&ctx.device, keys_in.len());
    let mut temp_vals = MetalBuffer::<u32>::new(&ctx.device, values_in.len());
    let mut hist_buf =
        MetalBuffer::<u32>::new(&ctx.device, num_threadgroups * RADIX_SIZE);
    let mut prefix_buf =
        MetalBuffer::<u32>::new(&ctx.device, num_threadgroups * RADIX_SIZE);

    let hist_threadgroup_mem_p = (RADIX_SIZE * std::mem::size_of::<u32>()) as u64;
    let scatter_threadgroup_mem_p =
        ((RADIX_SIZE + SORT_BLOCK_SIZE as usize) * std::mem::size_of::<u32>()) as u64;
    let block = MTLSize::new(SORT_BLOCK_SIZE as u64, 1, 1);
    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);

    let hist_pipeline = ctx.get_pipeline("ab_radix_sort_histogram_kernel");
    let scatter_pipeline = ctx.get_pipeline("ab_radix_sort_scatter_pairs_kernel");

    let mut current_state: usize = 0;

    for pass in 0..num_passes {
        let current_bit = (begin_bit + pass * RADIX_BITS) as i32;
        let next_state = 1 - current_state;

        let src_keys = if pass == 0 {
            keys_in.metal_buffer().clone()
        } else if current_state == 0 {
            keys_out.metal_buffer().clone()
        } else {
            temp_keys.metal_buffer().clone()
        };
        let src_vals = if pass == 0 {
            values_in.metal_buffer().clone()
        } else if current_state == 0 {
            values_out.metal_buffer().clone()
        } else {
            temp_vals.metal_buffer().clone()
        };
        let dst_keys = if next_state == 0 {
            keys_out.metal_buffer().clone()
        } else {
            temp_keys.metal_buffer().clone()
        };
        let dst_vals = if next_state == 0 {
            values_out.metal_buffer().clone()
        } else {
            temp_vals.metal_buffer().clone()
        };

        // Histogram pass (keys only).
        {
            let cb = ctx.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&hist_pipeline);
            enc.set_buffer(0, Some(&src_keys), 0);
            enc.set_buffer(1, Some(hist_buf.metal_buffer()), 0);
            set_bytes(&enc, &num_items, 2);
            set_bytes(&enc, &current_bit, 3);
            enc.set_threadgroup_memory_length(0, hist_threadgroup_mem_p);
            enc.dispatch_thread_groups(grid, block);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // CPU prefix sum.
        {
            let hist = hist_buf.as_slice();
            let prefix = prefix_buf.as_mut_slice();
            let mut digit_total = [0u32; RADIX_SIZE];
            for bid in 0..num_threadgroups {
                for d in 0..RADIX_SIZE {
                    digit_total[d] = digit_total[d].wrapping_add(hist[bid * RADIX_SIZE + d]);
                }
            }
            let mut global_prefix = [0u32; RADIX_SIZE];
            let mut running = 0u32;
            for d in 0..RADIX_SIZE {
                global_prefix[d] = running;
                running = running.wrapping_add(digit_total[d]);
            }
            let mut per_block_cumsum = [0u32; RADIX_SIZE];
            for bid in 0..num_threadgroups {
                for d in 0..RADIX_SIZE {
                    prefix[bid * RADIX_SIZE + d] =
                        global_prefix[d].wrapping_add(per_block_cumsum[d]);
                    per_block_cumsum[d] =
                        per_block_cumsum[d].wrapping_add(hist[bid * RADIX_SIZE + d]);
                }
            }
        }

        // Scatter pairs.
        {
            let cb = ctx.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&scatter_pipeline);
            enc.set_buffer(0, Some(&src_keys), 0);
            enc.set_buffer(1, Some(&dst_keys), 0);
            enc.set_buffer(2, Some(&src_vals), 0);
            enc.set_buffer(3, Some(&dst_vals), 0);
            enc.set_buffer(4, Some(prefix_buf.metal_buffer()), 0);
            set_bytes(&enc, &num_items, 5);
            set_bytes(&enc, &current_bit, 6);
            enc.set_threadgroup_memory_length(0, scatter_threadgroup_mem_p);
            enc.dispatch_thread_groups(grid, block);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        current_state = next_state;
    }

    if current_state == 1 {
        keys_out.load_from_host(temp_keys.as_slice());
        values_out.load_from_host(temp_vals.as_slice());
    }
}

/// Run-length encoding: finds unique keys and their counts.
///
/// Given a sorted input, produces:
/// - `unique_out`: the distinct keys
/// - `counts_out`: the count of each key
/// - `num_runs_out`: the total number of distinct keys
pub fn run_length_encode_u32(
    input: &MetalBuffer<u32>,
    unique_out: &mut MetalBuffer<u32>,
    counts_out: &mut MetalBuffer<u32>,
    num_runs_out: &mut MetalBuffer<u32>,
    ctx: &MetalProverContext,
) {
    let num_items = input.len() as i32;

    // Zero the run count before the atomic-increment kernel.
    num_runs_out.zero_fill();

    let pipeline = ctx.get_pipeline("ab_rle_simple_kernel");
    let (grid, block) = {
        let threads = num_items as u32;
        let block_size = 256u32;
        let num_groups = (threads + block_size - 1) / block_size;
        (
            MTLSize::new(num_groups.max(1) as u64, 1, 1),
            MTLSize::new(block_size as u64, 1, 1),
        )
    };

    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(input.metal_buffer()), 0);
    enc.set_buffer(1, Some(unique_out.metal_buffer()), 0);
    enc.set_buffer(2, Some(counts_out.metal_buffer()), 0);
    enc.set_buffer(3, Some(num_runs_out.metal_buffer()), 0);
    set_bytes(&enc, &num_items, 4);
    enc.dispatch_thread_groups(grid, block);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}
