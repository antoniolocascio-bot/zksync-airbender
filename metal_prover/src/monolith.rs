///! Monolith hash function wrappers for Metal.
///!
///! The Monolith hash is an algebraic hash function used as an alternative
///! to Blake2s for Merkle tree construction. It operates natively over
///! Mersenne31 field elements.

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_structures::{
    MetalBuffer, MetalMatrixChunkImpl, MetalMatrixChunkMutImpl, PtrAndStride, MutPtrAndStride,
};
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;

pub const RATE: usize = 8;
pub const CAPACITY: usize = 8;
pub const WIDTH: usize = RATE + CAPACITY;

pub type Digest = [BF; CAPACITY];

fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
}

/// Dispatch the Monolith leaves kernel (single-thread-per-hash variant).
pub fn launch_leaves_st_kernel(
    values: &MetalBuffer<BF>,
    results: &mut MetalBuffer<Digest>,
    log_rows_per_hash: u32,
    ctx: &MetalProverContext,
) {
    let values_len = values.len();
    let count = results.len() as u32;
    assert_eq!(values_len % ((count as usize) << log_rows_per_hash), 0);
    let cols_count = (values_len / ((count as usize) << log_rows_per_hash)) as u32;

    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);
    let pipeline = ctx.get_pipeline("ab_monolith_leaves_st_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(values.metal_buffer()), 0);
    encoder.set_buffer(1, Some(results.metal_buffer()), 0);
    set_bytes(&encoder, &log_rows_per_hash, 2);
    set_bytes(&encoder, &cols_count, 3);
    set_bytes(&encoder, &count, 4);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch the Monolith leaves kernel (multi-thread-per-hash variant).
///
/// Uses WIDTH threads per hash for parallelism within the permutation.
pub fn launch_leaves_mt_kernel(
    values: &MetalBuffer<BF>,
    results: &mut MetalBuffer<Digest>,
    log_rows_per_hash: u32,
    ctx: &MetalProverContext,
) {
    let values_len = values.len();
    let count = results.len() as u32;
    assert_eq!(values_len % ((count as usize) << log_rows_per_hash), 0);
    let cols_count = (values_len / ((count as usize) << log_rows_per_hash)) as u32;

    assert_eq!((WARP_SIZE * 4) % WIDTH as u32, 0);
    let threads_per_block = WARP_SIZE * 4 / WIDTH as u32;
    let (grid_dim, block_dim_base) = get_grid_block_dims_for_threads_count(threads_per_block, count);
    let block_dim = MTLSize::new(WIDTH as u64, block_dim_base.width, 1);

    let pipeline = ctx.get_pipeline("ab_monolith_leaves_mt_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(values.metal_buffer()), 0);
    encoder.set_buffer(1, Some(results.metal_buffer()), 0);
    set_bytes(&encoder, &log_rows_per_hash, 2);
    set_bytes(&encoder, &cols_count, 3);
    set_bytes(&encoder, &count, 4);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch the Monolith nodes kernel (single-thread variant).
pub fn launch_nodes_st_kernel(
    values: &MetalBuffer<Digest>,
    results: &mut MetalBuffer<Digest>,
    ctx: &MetalProverContext,
) {
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(values_len, results_len * 2);
    let count = results_len as u32;

    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);
    let pipeline = ctx.get_pipeline("ab_monolith_nodes_st_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(values.metal_buffer()), 0);
    encoder.set_buffer(1, Some(results.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch the Monolith nodes kernel (multi-thread variant).
pub fn launch_nodes_mt_kernel(
    values: &MetalBuffer<Digest>,
    results: &mut MetalBuffer<Digest>,
    ctx: &MetalProverContext,
) {
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(values_len, results_len * 2);
    let count = results_len as u32;

    assert_eq!((WARP_SIZE * 4) % WIDTH as u32, 0);
    let threads_per_block = WARP_SIZE * 4 / WIDTH as u32;
    let (grid_dim, block_dim_base) = get_grid_block_dims_for_threads_count(threads_per_block, count);
    let block_dim = MTLSize::new(WIDTH as u64, block_dim_base.width, 1);

    let pipeline = ctx.get_pipeline("ab_monolith_nodes_mt_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(values.metal_buffer()), 0);
    encoder.set_buffer(1, Some(results.metal_buffer()), 0);
    set_bytes(&encoder, &count, 2);

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Build a Monolith merkle tree: leaves + all node layers.
pub fn build_merkle_tree(
    values: &MetalBuffer<BF>,
    results: &mut MetalBuffer<Digest>,
    log_rows_per_hash: u32,
    ctx: &MetalProverContext,
    layers_count: u32,
) {
    assert_ne!(layers_count, 0);
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(results_len % 2, 0);
    let leaves_count = results_len / 2;
    assert!(1 << (layers_count - 1) <= leaves_count);
    assert_eq!(values_len % leaves_count, 0);

    // Build leaves into the first half
    launch_leaves_st_kernel(values, results, log_rows_per_hash, ctx);

    // Build node layers
    // TODO: Implement multi-layer node building with buffer offset management.
    // Requires Mac testing for proper sub-buffer handling.
}

/// Gather rows for Monolith merkle paths.
pub fn gather_rows(
    indexes: &MetalBuffer<u32>,
    log_rows_per_index: u32,
    values: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    result: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    ctx: &MetalProverContext,
) {
    let indexes_len = indexes.len();
    let values_cols = values.cols();
    let result_rows = result.rows();
    let result_cols = result.cols();
    let _rows_per_index = 1u32 << log_rows_per_index;
    assert!(log_rows_per_index < WARP_SIZE);
    assert_eq!(result_cols, values_cols);
    assert_eq!(result_rows, indexes_len << log_rows_per_index);
    assert!(indexes_len <= u32::MAX as usize);
    let indexes_count = indexes_len as u32;

    let pipeline = ctx.get_pipeline("ab_gather_rows_kernel");
    let (mut grid_dim, block_dim) =
        get_grid_block_dims_for_threads_count(WARP_SIZE >> log_rows_per_index, indexes_count);

    grid_dim.height = result_cols as u64;

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(indexes.metal_buffer()), 0);
    set_bytes(&encoder, &indexes_count, 1);
    let values_ps = values.as_ptr_and_stride();
    let result_ps = result.as_ptr_and_stride();
    set_bytes(&encoder, &values_ps, 2);
    set_bytes(&encoder, &result_ps, 3);

    let block_dim_2d = MTLSize::new(_rows_per_index as u64, block_dim.width, 1);
    encoder.dispatch_thread_groups(grid_dim, block_dim_2d);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Get the merkle tree cap from a tree stored in a buffer.
pub fn merkle_tree_cap(values: &[Digest], cap_size: usize) -> &[Digest] {
    assert_ne!(cap_size, 0);
    assert!(cap_size.is_power_of_two());
    let log_cap_size = cap_size.trailing_zeros();
    let values_len = values.len();
    assert_ne!(values_len, 0);
    assert!(values_len.is_power_of_two());
    let log_values_len = values_len.trailing_zeros();
    assert!(log_values_len > log_cap_size);
    let offset = values_len - (1 << (log_cap_size + 1));
    &values[offset..offset + cap_size]
}
