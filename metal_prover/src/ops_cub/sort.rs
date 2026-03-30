///! Parallel radix sort on Metal.
///!
///! Replaces CUB's `DeviceRadixSort::SortKeys` and `SortPairs`.
///! The Metal implementation uses a multi-pass radix sort dispatched
///! as a sequence of compute kernels.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_structures::MetalBuffer;
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;

fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const u8;
    let len = std::mem::size_of::<T>();
    encoder.set_bytes(index, unsafe { std::slice::from_raw_parts(ptr, len) }, len as u64);
}

/// Sort direction.
#[derive(Copy, Clone, Debug)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Radix sort of u32 keys.
///
/// Sorts `keys_in` and writes the sorted result to `keys_out`.
/// `begin_bit` and `end_bit` define the bit range to sort on.
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

    let kernel_name = match order {
        SortOrder::Ascending => "ab_sort_keys_a_u32",
        SortOrder::Descending => "ab_sort_keys_d_u32",
    };

    let pipeline = ctx.get_pipeline(kernel_name);

    // Radix sort processes a few bits per pass.
    // The Metal kernel handles the full sort internally using threadgroup memory.
    let block_size = WARP_SIZE * 4;
    let num_threadgroups = std::cmp::max(1, (num_items + block_size - 1) / block_size);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(keys_in.metal_buffer()), 0);
    encoder.set_buffer(1, Some(keys_out.metal_buffer()), 0);
    set_bytes(&encoder, &num_items, 2);
    set_bytes(&encoder, &begin_bit, 3);
    set_bytes(&encoder, &end_bit, 4);

    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);
    let block = MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_threadgroups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Sort key-value pairs: sorts `keys_in` and applies the same permutation
/// to `values_in`.
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

    let kernel_name = match order {
        SortOrder::Ascending => "ab_sort_pairs_a_u32",
        SortOrder::Descending => "ab_sort_pairs_d_u32",
    };

    let pipeline = ctx.get_pipeline(kernel_name);
    let block_size = WARP_SIZE * 4;
    let num_threadgroups = std::cmp::max(1, (num_items + block_size - 1) / block_size);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(keys_in.metal_buffer()), 0);
    encoder.set_buffer(1, Some(keys_out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(values_in.metal_buffer()), 0);
    encoder.set_buffer(3, Some(values_out.metal_buffer()), 0);
    set_bytes(&encoder, &num_items, 4);
    set_bytes(&encoder, &begin_bit, 5);
    set_bytes(&encoder, &end_bit, 6);

    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);
    let block = MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_threadgroups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
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
    let num_items = input.len() as u32;

    let pipeline = ctx.get_pipeline("ab_run_length_encode_u32");
    let block_size = WARP_SIZE * 4;
    let num_threadgroups = std::cmp::max(1, (num_items + block_size - 1) / block_size);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input.metal_buffer()), 0);
    encoder.set_buffer(1, Some(unique_out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(counts_out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(num_runs_out.metal_buffer()), 0);
    set_bytes(&encoder, &num_items, 4);

    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);
    let block = MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_threadgroups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
