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
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
}

/// Reduce operation type.
#[derive(Copy, Clone, Debug)]
pub enum ReduceOperation {
    Sum,
    Product,
}

/// Reduce an entire buffer to a single value.
///
/// Uses a single threadgroup with 256 threads (matching the Metal kernel's REDUCE_BLOCK_SIZE).
/// The kernel internally loops to accumulate all elements before reducing within the
/// threadgroup, so a single dispatch suffices for any input size.
pub fn reduce_bf(
    input: &MetalBuffer<BF>,
    output: &mut MetalBuffer<BF>,
    operation: ReduceOperation,
    ctx: &MetalProverContext,
) {
    assert_eq!(output.len(), 1);
    let num_items = input.len() as i32; // kernel uses `constant int &num_items`

    let kernel_name = match operation {
        ReduceOperation::Sum => "ab_reduce_add_bf_kernel",
        ReduceOperation::Product => "ab_reduce_mul_bf_kernel",
    };

    let pipeline = ctx.get_pipeline(kernel_name);

    // REDUCE_BLOCK_SIZE = 256 in the Metal shader.
    // One threadgroup with 256 threads accumulates all elements (the kernel loops internally),
    // then tree-reduces to a single value written to d_out[0].
    let block_size = 256u64;
    let threadgroup_mem = block_size * std::mem::size_of::<BF>() as u64;

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(input.metal_buffer()), 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &num_items, 2);
    // threadgroup(0) = shared memory for REDUCE_BLOCK_SIZE bf elements
    encoder.set_threadgroup_memory_length(0, threadgroup_mem);

    let grid = MTLSize::new(1, 1, 1);
    let block = MTLSize::new(block_size, 1, 1);
    encoder.dispatch_thread_groups(grid, block);
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
    encoder.dispatch_thread_groups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Segmented reduction: reduce each column of a matrix independently.
///
/// Given a matrix with `num_segments` columns, each of length `segment_len`,
/// produce one output value per column.
///
/// Only `ReduceOperation::Sum` is supported (only `ab_segmented_reduce_add_bf_kernel` exists).
/// The Metal kernel signature is:
///   `(device const bf *d_in, device bf *d_out, constant int &num_segments,
///     constant int &segment_length, constant size_t &stride, threadgroup bf *shared)`
/// where `d_in + bid * stride` is the start of segment `bid`.
/// On Apple Silicon (unified memory) the CPU pointer from `as_ptr_and_stride()` equals
/// the GPU address and is passed inline via `set_bytes`.
pub fn segmented_reduce_bf(
    input: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    output: &mut MetalBuffer<BF>,
    operation: ReduceOperation,
    ctx: &MetalProverContext,
) {
    let num_segments = input.cols() as i32;
    let segment_len = input.rows() as i32;
    assert_eq!(output.len(), num_segments as usize);

    match operation {
        ReduceOperation::Sum => {}
        ReduceOperation::Product => {
            panic!("segmented_reduce_bf: Product not supported (no Metal kernel)");
        }
    }

    // REDUCE_BLOCK_SIZE = 256 in the Metal shader; one threadgroup per segment.
    let block_size = 256u64;
    let threadgroup_mem = block_size * std::mem::size_of::<BF>() as u64;

    let pipeline = ctx.get_pipeline("ab_segmented_reduce_add_bf_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    // On Apple Silicon, the CPU pointer is the same as the GPU address.
    // Pass the pointer value inline so the kernel can use it as `device const bf *d_in`.
    let input_ps = input.as_ptr_and_stride();
    let d_in_ptr = input_ps.ptr as u64; // raw GPU address (unified memory)
    let stride = input_ps.stride as u64; // stride in elements between segments
    set_bytes(&encoder, &d_in_ptr, 0);
    encoder.set_buffer(1, Some(output.metal_buffer()), 0);
    set_bytes(&encoder, &num_segments, 2);
    set_bytes(&encoder, &segment_len, 3);
    set_bytes(&encoder, &stride, 4);
    encoder.set_threadgroup_memory_length(0, threadgroup_mem);

    // One threadgroup per segment; the kernel loops internally over segment elements.
    let grid = MTLSize::new(num_segments as u64, 1, 1);
    let block = MTLSize::new(block_size, 1, 1);
    encoder.dispatch_thread_groups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
