///! Parallel prefix scan (inclusive/exclusive) on Metal.
///!
///! Replaces CUB's `DeviceScan::InclusiveSum`, `ExclusiveSum`,
///! `InclusiveScan`, and `ExclusiveScan`.
///!
///! The Metal shaders use a two-pass approach:
///! - Pass 1 (`*_pass1`): each threadgroup computes a local scan and writes its
///!   block total to `partial_sums`.
///! - Pass 2 (`*_pass2`): after a CPU-side inclusive prefix scan of `partial_sums`,
///!   each block (except block 0) adds `partial_sums[bid-1]` to all its elements.
///!
///! For single-threadgroup inputs pass 2 is skipped.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext4Field};
use crate::prover::context::MetalProverContext;

use field::Field;

type BF = BaseField;
type E4 = Ext4Field;

/// Matches `SCAN_BLOCK_SIZE` in `parallel_scan.metal`.
const SCAN_BLOCK_SIZE: u32 = 256;

fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
}

/// Scan operation type.
#[derive(Copy, Clone, Debug)]
pub enum ScanOperation {
    Sum,
    Product,
}

/// Perform an inclusive prefix scan on a buffer.
///
/// Uses a two-pass approach:
/// - Pass 1: each threadgroup scans locally and writes its block total to `partial_sums`.
/// - Pass 2 (if > 1 threadgroup): CPU computes inclusive prefix scan of `partial_sums`,
///   then GPU adds `partial_sums[bid-1]` to each block's elements.
pub fn inclusive_scan_bf(
    input: &MetalBuffer<BF>,
    output: &mut MetalBuffer<BF>,
    operation: ScanOperation,
    ctx: &MetalProverContext,
) {
    assert_eq!(input.len(), output.len());
    let count = input.len() as u32;
    let num_threadgroups = ((count + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE).max(1);
    let num_items = count as i32;
    let threadgroup_mem = SCAN_BLOCK_SIZE as u64 * std::mem::size_of::<BF>() as u64;

    let (pass1_name, pass2_name) = match operation {
        ScanOperation::Sum => ("ab_scan_i_add_bf_pass1", "ab_scan_i_add_bf_pass2"),
        ScanOperation::Product => {
            panic!("inclusive_scan_bf: ScanOperation::Product not supported (no Metal kernel)");
        }
    };

    let block = MTLSize::new(SCAN_BLOCK_SIZE as u64, 1, 1);
    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);

    // Allocate partial sums buffer (one BF element per threadgroup).
    let mut partial_sums = MetalBuffer::<BF>::new(&ctx.device, num_threadgroups as usize);

    // Pass 1: local scan + per-threadgroup totals written to partial_sums.
    {
        let pipeline = ctx.get_pipeline(pass1_name);
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(input.metal_buffer()), 0);
        enc.set_buffer(1, Some(output.metal_buffer()), 0);
        enc.set_buffer(2, Some(partial_sums.metal_buffer()), 0);
        set_bytes(&enc, &num_items, 3);
        enc.set_threadgroup_memory_length(0, threadgroup_mem);
        enc.dispatch_thread_groups(grid, block);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    if num_threadgroups > 1 {
        // CPU-side inclusive prefix scan of partial_sums (unified memory: directly accessible).
        // After this: partial_sums[i] = sum of original partial_sums[0..=i].
        {
            let ps = partial_sums.as_mut_slice();
            for i in 1..ps.len() {
                let prev = ps[i - 1]; // BF: Copy
                Field::add_assign(&mut ps[i], &prev);
            }
        }

        // Pass 2: for each block bid > 0, add partial_sums[bid-1] to all elements.
        let pipeline = ctx.get_pipeline(pass2_name);
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(output.metal_buffer()), 0);
        enc.set_buffer(1, Some(partial_sums.metal_buffer()), 0);
        set_bytes(&enc, &num_items, 2);
        enc.dispatch_thread_groups(grid, block);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
}

/// Perform an exclusive prefix scan on a buffer.
///
/// Uses a two-pass approach: pass1 computes the exclusive scan locally per threadgroup;
/// pass2 (when needed) adds the inclusive prefix of block totals to each block.
/// The fixup step reuses `ab_scan_i_add_bf_pass2` since the logic is identical.
pub fn exclusive_scan_bf(
    input: &MetalBuffer<BF>,
    output: &mut MetalBuffer<BF>,
    operation: ScanOperation,
    ctx: &MetalProverContext,
) {
    assert_eq!(input.len(), output.len());
    let count = input.len() as u32;
    let num_threadgroups = ((count + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE).max(1);
    let num_items = count as i32;
    let threadgroup_mem = SCAN_BLOCK_SIZE as u64 * std::mem::size_of::<BF>() as u64;

    let pass1_name = match operation {
        ScanOperation::Sum => "ab_scan_e_add_bf_pass1",
        ScanOperation::Product => {
            panic!("exclusive_scan_bf: ScanOperation::Product not supported (no Metal kernel)");
        }
    };
    // The fixup pass is the same as for inclusive scan (add block prefix to each block).
    let pass2_name = "ab_scan_i_add_bf_pass2";

    let block = MTLSize::new(SCAN_BLOCK_SIZE as u64, 1, 1);
    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);
    let mut partial_sums = MetalBuffer::<BF>::new(&ctx.device, num_threadgroups as usize);

    {
        let pipeline = ctx.get_pipeline(pass1_name);
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(input.metal_buffer()), 0);
        enc.set_buffer(1, Some(output.metal_buffer()), 0);
        enc.set_buffer(2, Some(partial_sums.metal_buffer()), 0);
        set_bytes(&enc, &num_items, 3);
        enc.set_threadgroup_memory_length(0, threadgroup_mem);
        enc.dispatch_thread_groups(grid, block);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    if num_threadgroups > 1 {
        {
            let ps = partial_sums.as_mut_slice();
            for i in 1..ps.len() {
                let prev = ps[i - 1];
                Field::add_assign(&mut ps[i], &prev);
            }
        }

        let pipeline = ctx.get_pipeline(pass2_name);
        let cb = ctx.command_queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(output.metal_buffer()), 0);
        enc.set_buffer(1, Some(partial_sums.metal_buffer()), 0);
        set_bytes(&enc, &num_items, 2);
        enc.dispatch_thread_groups(grid, block);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
}

/// Inclusive prefix-product scan on E4 elements.
///
/// Only single-threadgroup inputs are supported (no pass2 kernel exists for E4).
pub fn inclusive_scan_mul_e4(
    input: &MetalBuffer<E4>,
    output: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    assert_eq!(input.len(), output.len());
    let count = input.len() as u32;
    let num_threadgroups = ((count + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE).max(1);
    let num_items = count as i32;
    let threadgroup_mem = SCAN_BLOCK_SIZE as u64 * std::mem::size_of::<E4>() as u64;

    assert!(
        num_threadgroups == 1,
        "inclusive_scan_mul_e4: only single-threadgroup inputs supported (no pass2 kernel for E4); \
         got {} elements (max {})",
        count,
        SCAN_BLOCK_SIZE
    );

    // The pass1 kernel requires a partial_sums buffer even for single-block inputs.
    let partial_sums = MetalBuffer::<E4>::new(&ctx.device, 1);

    let pipeline = ctx.get_pipeline("ab_scan_i_mul_e4_pass1");
    let block = MTLSize::new(SCAN_BLOCK_SIZE as u64, 1, 1);
    let grid = MTLSize::new(1, 1, 1);

    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(input.metal_buffer()), 0);
    enc.set_buffer(1, Some(output.metal_buffer()), 0);
    enc.set_buffer(2, Some(partial_sums.metal_buffer()), 0);
    set_bytes(&enc, &num_items, 3);
    enc.set_threadgroup_memory_length(0, threadgroup_mem);
    enc.dispatch_thread_groups(grid, block);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}
