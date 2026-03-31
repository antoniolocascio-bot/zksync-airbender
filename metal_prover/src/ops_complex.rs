///! Complex operations: batch_inv, transpose, bit_reverse, fold, get_powers.
///!
///! These correspond to the CUDA ops_complex.rs but dispatch via Metal
///! compute command encoders instead of CUDA kernel launches.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_context::CIRCLE_GROUP_LOG_ORDER;
use crate::device_structures::{
    MetalBuffer, MetalMatrixChunkImpl, MetalMatrixChunkMutImpl, MutPtrAndStride, PtrAndStride,
    PtrAndStrideWrappingMatrix, MutPtrAndStrideWrappingMatrix,
};
use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, GetChunksCount, LOG_WARP_SIZE, WARP_SIZE};

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
}

fn get_launch_dims(count: u32) -> (MTLSize, MTLSize) {
    get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count)
}

// ---------------------------------------------------------------------------
// get_powers: compute [base^0, base^1, ..., base^(n-1)]
// ---------------------------------------------------------------------------

/// Compute powers of a base element, optionally in bit-reversed order.
pub fn get_powers_by_val_bf(
    base: BF,
    offset: u32,
    bit_reverse: bool,
    result: &mut MetalBuffer<BF>,
    ctx: &MetalProverContext,
) {
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let pipeline = ctx.get_pipeline("ab_get_powers_by_val_bf_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_bytes(&encoder, &base, 0);
    set_bytes(&encoder, &offset, 1);
    set_bytes(&encoder, &bit_reverse, 2);
    encoder.set_buffer(3, Some(result.metal_buffer()), 0);
    set_bytes(&encoder, &count, 4);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn get_powers_by_val_e2(
    base: E2,
    offset: u32,
    bit_reverse: bool,
    result: &mut MetalBuffer<E2>,
    ctx: &MetalProverContext,
) {
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let pipeline = ctx.get_pipeline("ab_get_powers_by_val_e2_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_bytes(&encoder, &base, 0);
    set_bytes(&encoder, &offset, 1);
    set_bytes(&encoder, &bit_reverse, 2);
    encoder.set_buffer(3, Some(result.metal_buffer()), 0);
    set_bytes(&encoder, &count, 4);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn get_powers_by_val_e4(
    base: E4,
    offset: u32,
    bit_reverse: bool,
    result: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let pipeline = ctx.get_pipeline("ab_get_powers_by_val_e4_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_bytes(&encoder, &base, 0);
    set_bytes(&encoder, &offset, 1);
    set_bytes(&encoder, &bit_reverse, 2);
    encoder.set_buffer(3, Some(result.metal_buffer()), 0);
    set_bytes(&encoder, &count, 4);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------------------------------------------------------------------------
// batch_inv: element-wise inversion of a buffer
// ---------------------------------------------------------------------------

/// Batch inversion using the Montgomery trick.
/// Dispatches the `ab_batch_inv_*_kernel` which internally does
/// prefix-product, single inversion, then suffix-product.
pub fn batch_inv_bf(
    src: &MetalBuffer<BF>,
    dst: &mut MetalBuffer<BF>,
    ctx: &MetalProverContext,
) {
    assert_eq!(src.len(), dst.len());
    let count = dst.len() as u32;
    let block_dim = WARP_SIZE * 4;
    let batch_size: u32 = 20; // matches CUDA batch size for BF
    let grid_dim = count.get_chunks_count(batch_size * block_dim);

    let pipeline = ctx.get_pipeline("ab_batch_inv_bf_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(src.metal_buffer()), 0);
    encoder.set_buffer(1, Some(dst.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    let grid = MTLSize::new(grid_dim as u64, 1, 1);
    let block = MTLSize::new(block_dim as u64, 1, 1);
    encoder.dispatch_thread_groups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn batch_inv_e2(
    src: &MetalBuffer<E2>,
    dst: &mut MetalBuffer<E2>,
    ctx: &MetalProverContext,
) {
    assert_eq!(src.len(), dst.len());
    let count = dst.len() as u32;
    let block_dim = WARP_SIZE * 4;
    let batch_size: u32 = 5;
    let grid_dim = count.get_chunks_count(batch_size * block_dim);

    let pipeline = ctx.get_pipeline("ab_batch_inv_e2_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(src.metal_buffer()), 0);
    encoder.set_buffer(1, Some(dst.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    let grid = MTLSize::new(grid_dim as u64, 1, 1);
    let block = MTLSize::new(block_dim as u64, 1, 1);
    encoder.dispatch_thread_groups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn batch_inv_e4(
    src: &MetalBuffer<E4>,
    dst: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    assert_eq!(src.len(), dst.len());
    let count = dst.len() as u32;
    let block_dim = WARP_SIZE * 4;
    let batch_size: u32 = 3;
    let grid_dim = count.get_chunks_count(batch_size * block_dim);

    let pipeline = ctx.get_pipeline("ab_batch_inv_e4_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(src.metal_buffer()), 0);
    encoder.set_buffer(1, Some(dst.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    let grid = MTLSize::new(grid_dim as u64, 1, 1);
    let block = MTLSize::new(block_dim as u64, 1, 1);
    encoder.dispatch_thread_groups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// In-place batch inversion (src == dst).
pub fn batch_inv_in_place_bf(values: &mut MetalBuffer<BF>, ctx: &MetalProverContext) {
    let count = values.len() as u32;
    let block_dim = WARP_SIZE * 4;
    let batch_size: u32 = 20;
    let grid_dim = count.get_chunks_count(batch_size * block_dim);

    let pipeline = ctx.get_pipeline("ab_batch_inv_bf_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    // Point both src and dst to the same buffer for in-place operation
    encoder.set_buffer(0, Some(values.metal_buffer()), 0);
    encoder.set_buffer(1, Some(values.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    let grid = MTLSize::new(grid_dim as u64, 1, 1);
    let block = MTLSize::new(block_dim as u64, 1, 1);
    encoder.dispatch_thread_groups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

/// Transpose a matrix from row-major to column-major (or vice versa).
pub fn transpose_bf(
    src: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    dst: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    ctx: &MetalProverContext,
) {
    let src_rows = src.rows() as u32;
    let src_cols = src.cols() as u32;
    assert_eq!(src_rows as usize, dst.cols());
    assert_eq!(src_cols as usize, dst.rows());

    let log_tile_size: u32 = 5; // 32x32 tiles for BF
    let tile_size = 1u32 << log_tile_size;
    let log_block_size: u32 = LOG_WARP_SIZE + 2;
    let log_tiles_per_block = log_block_size - log_tile_size;
    let tiles_per_block = 1u32 << log_tiles_per_block;

    let tile_rows = src_rows.get_chunks_count(tile_size);
    let tile_cols = src_cols.get_chunks_count(tile_size);
    let tiles = tile_rows * tile_cols;
    let grid_dim_x = tiles.get_chunks_count(tiles_per_block);

    let pipeline = ctx.get_pipeline("ab_transpose_bf_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let src_ps = src.as_ptr_and_stride();
    let dst_ps = dst.as_mut_ptr_and_stride();
    set_bytes(&encoder, &src_ps, 0);
    set_bytes(&encoder, &dst_ps, 1);
    set_bytes(&encoder, &src_rows, 2);
    set_bytes(&encoder, &src_cols, 3);

    let block = MTLSize::new(tile_size as u64, tiles_per_block as u64, 1);
    let grid = MTLSize::new(grid_dim_x as u64, 1, 1);
    encoder.dispatch_thread_groups(grid, block);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------------------------------------------------------------------------
// bit_reverse
// ---------------------------------------------------------------------------

/// Bit-reverse a flat buffer in-place.
///
/// Metal kernel: `ab_bit_reverse_naive_bf_kernel`
/// Signature: `(device const bf *src, device bf *dst, constant size_t &stride,
///              constant unsigned &log_count, constant unsigned &col, uint gid)`
pub fn bit_reverse_in_place_bf(
    values: &mut MetalBuffer<BF>,
    ctx: &MetalProverContext,
) {
    let n = values.len();
    assert!(n.is_power_of_two(), "bit_reverse: size must be a power of two");
    let log_n = n.trailing_zeros();
    let stride = n as u64; // size_t stride in elements (column stride for 1-column flat buffer)
    let col = 0u32;

    let (grid_dim, block_dim) = get_launch_dims(n as u32);
    let pipeline = ctx.get_pipeline("ab_bit_reverse_naive_bf_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    // In-place: src == dst
    encoder.set_buffer(0, Some(values.metal_buffer()), 0);
    encoder.set_buffer(1, Some(values.metal_buffer()), 0);
    set_bytes(&encoder, &stride, 2);  // constant size_t &stride
    set_bytes(&encoder, &log_n, 3);   // constant unsigned &log_count
    set_bytes(&encoder, &col, 4);     // constant unsigned &col

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn bit_reverse_in_place_e4(
    values: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    let n = values.len();
    assert!(n.is_power_of_two(), "bit_reverse: size must be a power of two");
    let log_n = n.trailing_zeros();
    let stride = n as u64;
    let col = 0u32;

    let (grid_dim, block_dim) = get_launch_dims(n as u32);
    let pipeline = ctx.get_pipeline("ab_bit_reverse_naive_e4_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(values.metal_buffer()), 0);
    encoder.set_buffer(1, Some(values.metal_buffer()), 0);
    set_bytes(&encoder, &stride, 2);
    set_bytes(&encoder, &log_n, 3);
    set_bytes(&encoder, &col, 4);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------------------------------------------------------------------------
// fold
// ---------------------------------------------------------------------------

/// FRI fold: given src of size 2N (pairs of evaluations), produce dst of size N
/// by folding with the challenge over the circle domain.
pub fn fold(
    challenge: &MetalBuffer<E4>,
    src: &MetalBuffer<E4>,
    dst: &mut MetalBuffer<E4>,
    root_offset: usize,
    ctx: &MetalProverContext,
) {
    assert!(src.len().is_power_of_two());
    assert!(dst.len().is_power_of_two());
    let log_count = dst.len().trailing_zeros();
    assert_eq!(src.len().trailing_zeros(), log_count + 1);
    assert!(log_count < 32);
    assert!(root_offset + (1 << log_count) < (1u64 << CIRCLE_GROUP_LOG_ORDER) as usize);
    let root_offset_u32 = root_offset as u32;

    let (grid_dim, block_dim) = get_launch_dims(1 << log_count);
    let pipeline = ctx.get_pipeline("ab_fold_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(challenge.metal_buffer()), 0);
    encoder.set_buffer(1, Some(src.metal_buffer()), 0);
    encoder.set_buffer(2, Some(dst.metal_buffer()), 0);
    set_bytes(&encoder, &root_offset_u32, 3);
    set_bytes(&encoder, &log_count, 4);

    // Pass twiddle factor buffers for the fold operation
    encoder.set_buffer(5, Some(ctx.device_context.powers_of_w_fine.metal_buffer()), 0);
    encoder.set_buffer(6, Some(ctx.device_context.powers_of_w_coarser.metal_buffer()), 0);
    encoder.set_buffer(7, Some(ctx.device_context.powers_of_w_coarsest.metal_buffer()), 0);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
