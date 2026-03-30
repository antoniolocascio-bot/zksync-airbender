///! Parallel reduction on Metal.
///!
///! Replaces CUB's `DeviceReduce::Sum`, `DeviceReduce::Reduce`,
///! and segmented reduction variants.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_structures::{MetalBuffer, MetalMatrixChunkImpl, PtrAndStride};
use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const u8;
    let len = std::mem::size_of::<T>();
    encoder.set_bytes(index, unsafe { std::slice::from_raw_parts(ptr, len) }, len as u64);
}

/// Reduce operation type.
#[derive(Copy, Clone, Debug)]
pub enum ReduceOperation {
    Sum,
    Product,
}

/// Reduce an entire buffer to a single value.
pub fn reduce_bf(
    input: &MetalBuffer<BF>,
    output: &mut MetalBuffer<BF>,
    operation: ReduceOperation,
    ctx: &MetalProverContext,
) {
    assert_eq!(output.len(), 1);
    let count = input.len() as u32;

    let kernel_name = match operation {
        ReduceOperation::Sum => "ab_reduce_sum_bf",
        ReduceOperation::Product => "ab_reduce_mul_bf",
    };

    let pipeline = ctx.get_pipeline(kernel_name);

    // Two-pass reduction: first reduce to partial sums per threadgroup,
    // then reduce partial sums to a single value.
    // For simplicity in the initial port, we use a single large threadgroup
    // approach when possible, falling back to multi-pass.
    let block_size = WARP_SIZE * 4;
    let num_threadgroups = std::cmp::max(1, (count + block_size - 1) / block_size);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input.metal_buffer()), 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);
    let block = MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_threadgroups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn reduce_e4(
    input: &MetalBuffer<E4>,
    output: &mut MetalBuffer<E4>,
    operation: ReduceOperation,
    ctx: &MetalProverContext,
) {
    assert_eq!(output.len(), 1);
    let count = input.len() as u32;

    let kernel_name = match operation {
        ReduceOperation::Sum => "ab_reduce_sum_e4",
        ReduceOperation::Product => "ab_reduce_mul_e4",
    };

    let pipeline = ctx.get_pipeline(kernel_name);
    let block_size = WARP_SIZE * 4;
    let num_threadgroups = std::cmp::max(1, (count + block_size - 1) / block_size);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input.metal_buffer()), 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    let grid = MTLSize::new(num_threadgroups as u64, 1, 1);
    let block = MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_threadgroups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Segmented reduction: reduce each column of a matrix independently.
///
/// Given a matrix with `num_segments` columns, each of length `segment_len`,
/// produce one output value per column.
pub fn segmented_reduce_bf(
    input: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    output: &mut MetalBuffer<BF>,
    operation: ReduceOperation,
    ctx: &MetalProverContext,
) {
    let num_segments = input.cols() as u32;
    let segment_len = input.rows() as u32;
    assert_eq!(output.len(), num_segments as usize);

    let kernel_name = match operation {
        ReduceOperation::Sum => "ab_segmented_reduce_sum_bf",
        ReduceOperation::Product => "ab_segmented_reduce_mul_bf",
    };

    let pipeline = ctx.get_pipeline(kernel_name);
    let block_size = WARP_SIZE * 4;
    let threadgroups_per_segment = std::cmp::max(1, (segment_len + block_size - 1) / block_size);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let input_ps = input.as_ptr_and_stride();
    set_bytes(&encoder, &input_ps, 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &num_segments, 2);
    set_bytes(&encoder, &segment_len, 3);

    let grid = MTLSize::new(threadgroups_per_segment as u64, num_segments as u64, 1);
    let block = MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_threadgroups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
