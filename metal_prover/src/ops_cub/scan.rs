///! Parallel prefix scan (inclusive/exclusive) on Metal.
///!
///! Replaces CUB's `DeviceScan::InclusiveSum`, `ExclusiveSum`,
///! `InclusiveScan`, and `ExclusiveScan`.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext4Field};
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;
type E4 = Ext4Field;

fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const u8;
    let len = std::mem::size_of::<T>();
    encoder.set_bytes(index, unsafe { std::slice::from_raw_parts(ptr, len) }, len as u64);
}

/// Scan operation type.
#[derive(Copy, Clone, Debug)]
pub enum ScanOperation {
    Sum,
    Product,
}

/// Perform an inclusive prefix scan on a buffer.
///
/// The scan kernel is a three-pass algorithm:
/// 1. Per-threadgroup local scan producing partial sums
/// 2. Scan of partial sums
/// 3. Add partial sums back to each threadgroup's results
///
/// For small inputs (< 1 threadgroup), a single pass suffices.
pub fn inclusive_scan_bf(
    input: &MetalBuffer<BF>,
    output: &mut MetalBuffer<BF>,
    operation: ScanOperation,
    ctx: &MetalProverContext,
) {
    assert_eq!(input.len(), output.len());
    let count = input.len() as u32;

    let kernel_name = match operation {
        ScanOperation::Sum => "ab_scan_i_add_bf",
        ScanOperation::Product => "ab_scan_i_mul_bf",
    };

    let pipeline = ctx.get_pipeline(kernel_name);
    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input.metal_buffer()), 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Perform an exclusive prefix scan on a buffer.
pub fn exclusive_scan_bf(
    input: &MetalBuffer<BF>,
    output: &mut MetalBuffer<BF>,
    operation: ScanOperation,
    ctx: &MetalProverContext,
) {
    assert_eq!(input.len(), output.len());
    let count = input.len() as u32;

    let kernel_name = match operation {
        ScanOperation::Sum => "ab_scan_e_add_bf",
        ScanOperation::Product => "ab_scan_e_mul_bf",
    };

    let pipeline = ctx.get_pipeline(kernel_name);
    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input.metal_buffer()), 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Inclusive prefix-product scan on E4 elements.
pub fn inclusive_scan_mul_e4(
    input: &MetalBuffer<E4>,
    output: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    assert_eq!(input.len(), output.len());
    let count = input.len() as u32;

    let pipeline = ctx.get_pipeline("ab_scan_i_mul_e4");
    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input.metal_buffer()), 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
