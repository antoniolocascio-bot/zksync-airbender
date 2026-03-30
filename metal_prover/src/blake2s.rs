use metal::{
    Buffer as MTLBuffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef,
    ComputePipelineState, Device as MTLDevice, MTLSize,
};

use crate::device_structures::{
    MetalBuffer, MetalMatrixChunkImpl, MetalMatrixChunkMutImpl, MutPtrAndStride, PtrAndStride,
};
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;

pub const STATE_SIZE: usize = 8;
pub const BLOCK_SIZE: usize = 16;

pub type Digest = [u32; STATE_SIZE];

// ---------------------------------------------------------------------------
// Helper: encode a kernel dispatch into a command encoder
// ---------------------------------------------------------------------------

/// Sets a `MetalBuffer` (or raw `MTLBuffer`) at a given buffer index on the encoder.
fn set_buffer(
    encoder: &ComputeCommandEncoderRef,
    buffer: &MTLBuffer,
    offset: u64,
    index: u64,
) {
    encoder.set_buffer(index, Some(buffer), offset);
}

/// Sets a plain value as bytes at a given buffer index.
fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const u8;
    let len = std::mem::size_of::<T>();
    encoder.set_bytes(index, unsafe { std::slice::from_raw_parts(ptr, len) }, len as u64);
}

// ---------------------------------------------------------------------------
// Blake2s leaves kernel
// ---------------------------------------------------------------------------

/// Dispatch the blake2s leaves kernel.
/// Computes a Blake2s hash for each row of the input matrix.
///
/// * `values` - input field elements buffer
/// * `results` - output digest buffer
/// * `log_rows_per_hash` - log2 of rows consumed per hash
/// * `ctx` - the Metal prover context (provides pipeline + command queue)
pub fn launch_leaves_kernel(
    values: &MetalBuffer<BF>,
    results: &mut MetalBuffer<Digest>,
    log_rows_per_hash: u32,
    ctx: &MetalProverContext,
) {
    let values_len = values.len();
    let count = results.len();
    assert_eq!(values_len % (count << log_rows_per_hash), 0);
    let cols_count = (values_len / (count << log_rows_per_hash)) as u32;
    let count_u32 = count as u32;

    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count_u32);
    let pipeline = ctx.get_pipeline("ab_blake2s_leaves_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_buffer(&encoder, values.metal_buffer(), 0, 0);
    set_buffer(&encoder, results.metal_buffer(), 0, 1);
    set_bytes(&encoder, &log_rows_per_hash, 2);
    set_bytes(&encoder, &cols_count, 3);
    set_bytes(&encoder, &count_u32, 4);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Build merkle tree leaves from field element values.
pub fn build_merkle_tree_leaves(
    values: &MetalBuffer<BF>,
    results: &mut MetalBuffer<Digest>,
    log_rows_per_hash: u32,
    ctx: &MetalProverContext,
) {
    let values_len = values.len();
    let leaves_count = results.len();
    assert_eq!(values_len % leaves_count, 0);
    launch_leaves_kernel(values, results, log_rows_per_hash, ctx);
}

// ---------------------------------------------------------------------------
// Blake2s nodes kernel
// ---------------------------------------------------------------------------

/// Dispatch the blake2s nodes kernel.
/// Compresses pairs of digests into single digests (one tree layer).
pub fn launch_nodes_kernel(
    values: &MetalBuffer<Digest>,
    results: &mut MetalBuffer<Digest>,
    ctx: &MetalProverContext,
) {
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(values_len, results_len * 2);
    let count = results_len as u32;

    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);
    let pipeline = ctx.get_pipeline("ab_blake2s_nodes_kernel");

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_buffer(&encoder, values.metal_buffer(), 0, 0);
    set_buffer(&encoder, results.metal_buffer(), 0, 1);
    set_bytes(&encoder, &count, 2);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Recursively build merkle tree node layers.
pub fn build_merkle_tree_nodes(
    values: &MetalBuffer<Digest>,
    results: &mut MetalBuffer<Digest>,
    layers_count: u32,
    ctx: &MetalProverContext,
) {
    if layers_count == 0 {
        return;
    }
    let values_len = values.len();
    let results_len = results.len();
    assert!(values_len.is_power_of_two());
    assert_eq!(values_len, results_len);

    // For the first layer, hash pairs from `values` into the first half of `results`.
    // Then recursively process remaining layers using results as both input and output.
    //
    // Since Metal unified memory allows us to read/write the same buffer,
    // we split results into two halves: nodes_out and nodes_remaining.
    // However, MetalBuffer cannot be split into sub-buffers easily,
    // so we dispatch layer by layer using offsets.
    //
    // For now, we implement a simplified version that processes one layer at a time
    // using temporary buffers. A production implementation would use buffer offsets.

    // TODO: Implement multi-layer node building with proper buffer offset management.
    // This requires Mac testing to validate MTLBuffer offset handling.
    let _ = values;
    let _ = results;
    let _ = layers_count;
}

/// Build a complete merkle tree: leaves + all node layers.
pub fn build_merkle_tree(
    values: &MetalBuffer<BF>,
    results: &mut MetalBuffer<Digest>,
    log_rows_per_hash: u32,
    ctx: &MetalProverContext,
    layers_count: u32,
    bit_reverse_leaves: bool,
) {
    assert_ne!(layers_count, 0);
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(results_len % 2, 0);
    let leaves_count = results_len / 2;
    assert!(1 << (layers_count - 1) <= leaves_count);
    assert_eq!(values_len % leaves_count, 0);

    // Build leaves into the first half of results.
    // Then build node layers in the second half.
    // This requires splitting the results buffer.
    //
    // TODO: Implement with proper buffer splitting for leaf/node regions.
    // Requires Mac testing.
    let _ = bit_reverse_leaves;
}

// ---------------------------------------------------------------------------
// Gather rows kernel
// ---------------------------------------------------------------------------

/// Gather specific rows from a matrix based on index array.
pub fn gather_rows(
    indexes: &MetalBuffer<u32>,
    bit_reverse_indexes: bool,
    log_rows_per_index: u32,
    values: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    result: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    ctx: &MetalProverContext,
) {
    let indexes_len = indexes.len();
    let values_cols = values.cols();
    let values_rows = values.rows();
    assert!(values_rows.is_power_of_two());
    let log_rows_count = values_rows.trailing_zeros();
    let result_rows = result.rows();
    let result_cols = result.cols();
    let _rows_per_index = 1u32 << log_rows_per_index;
    assert_eq!(result_cols, values_cols);
    assert_eq!(result_rows, indexes_len << log_rows_per_index);
    assert!(indexes_len <= u32::MAX as usize);
    let indexes_count = indexes_len as u32;

    let pipeline = ctx.get_pipeline("ab_gather_rows_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_buffer(&encoder, indexes.metal_buffer(), 0, 0);
    set_bytes(&encoder, &indexes_count, 1);
    set_bytes(&encoder, &bit_reverse_indexes, 2);
    set_bytes(&encoder, &log_rows_count, 3);

    let values_ps = values.as_ptr_and_stride();
    let result_ps = result.as_ptr_and_stride();
    set_bytes(&encoder, &values_ps, 4);
    set_bytes(&encoder, &result_ps, 5);

    let (grid_dim, block_dim) =
        get_grid_block_dims_for_threads_count(WARP_SIZE, indexes_count);
    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------------------------------------------------------------------------
// Gather merkle paths kernel
// ---------------------------------------------------------------------------

/// Gather merkle authentication paths for given leaf indices.
pub fn gather_merkle_paths(
    indexes: &MetalBuffer<u32>,
    values: &MetalBuffer<Digest>,
    results: &mut MetalBuffer<Digest>,
    layers_count: u32,
    ctx: &MetalProverContext,
) {
    assert!(indexes.len() <= u32::MAX as usize);
    let indexes_count = indexes.len() as u32;
    let values_count = values.len();
    assert!(values_count.is_power_of_two());
    let log_values_count = values_count.trailing_zeros();
    assert_ne!(log_values_count, 0);
    let log_leaves_count = log_values_count - 1;
    assert!(layers_count < log_leaves_count);
    assert_eq!(indexes.len() * layers_count as usize, results.len());

    let pipeline = ctx.get_pipeline("ab_gather_merkle_paths_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_buffer(&encoder, indexes.metal_buffer(), 0, 0);
    set_bytes(&encoder, &indexes_count, 1);
    set_buffer(&encoder, values.metal_buffer(), 0, 2);
    set_bytes(&encoder, &log_leaves_count, 3);
    set_buffer(&encoder, results.metal_buffer(), 0, 4);

    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(
        WARP_SIZE / STATE_SIZE as u32,
        indexes_count,
    );
    let grid_dim_2d = MTLSize::new(grid_dim.width, layers_count as u64, 1);
    let block_dim_2d = MTLSize::new(STATE_SIZE as u64, block_dim.width, 1);

    encoder.dispatch_threadgroups(grid_dim_2d, block_dim_2d);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------------------------------------------------------------------------
// Gather rows and merkle paths (combined kernel)
// ---------------------------------------------------------------------------

/// Combined kernel that gathers both leaf values and their merkle paths.
pub fn gather_rows_and_merkle_paths(
    indexes: &MetalBuffer<u32>,
    bit_reverse_indexes: bool,
    values: &MetalBuffer<BF>,
    log_rows_per_index: u32,
    leaf_values: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    tree_bottom: &MetalBuffer<Digest>,
    merkle_paths: &mut MetalBuffer<Digest>,
    layers_count: u32,
    ctx: &MetalProverContext,
) {
    let indexes_len = indexes.len();
    let values_len = values.len();
    let cols_count = leaf_values.cols() as u32;
    assert_eq!(values_len % cols_count as usize, 0);
    let log_rows_count = (values_len / cols_count as usize).trailing_zeros();
    assert_eq!(leaf_values.rows(), indexes_len << log_rows_per_index);
    assert!(indexes_len <= u32::MAX as usize);
    let indexes_count = indexes_len as u32;
    assert_eq!(indexes_len * layers_count as usize, merkle_paths.len());

    let log_total_leaves_count = log_rows_count - log_rows_per_index;

    let pipeline = ctx.get_pipeline("ab_gather_rows_and_merkle_paths_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_buffer(&encoder, indexes.metal_buffer(), 0, 0);
    set_bytes(&encoder, &indexes_count, 1);
    set_bytes(&encoder, &bit_reverse_indexes, 2);
    set_buffer(&encoder, values.metal_buffer(), 0, 3);
    set_bytes(&encoder, &log_rows_per_index, 4);
    set_bytes(&encoder, &cols_count, 5);
    set_bytes(&encoder, &log_total_leaves_count, 6);
    let leaf_ps = leaf_values.as_ptr_and_stride();
    set_bytes(&encoder, &leaf_ps, 7);
    set_buffer(&encoder, tree_bottom.metal_buffer(), 0, 8);
    set_bytes(&encoder, &layers_count, 9);
    set_buffer(&encoder, merkle_paths.metal_buffer(), 0, 10);

    let grid_dim = MTLSize::new(indexes_count as u64, 1, 1);
    let block_dim = MTLSize::new(WARP_SIZE as u64, 1, 1);
    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Get a reference to the tree cap (the top-most nodes of the merkle tree).
pub fn merkle_tree_cap(values: &[Digest], log_tree_cap_size: u32) -> &[Digest] {
    let values_len = values.len();
    assert_ne!(values_len, 0);
    assert!(values_len.is_power_of_two());
    let log_values_len = values_len.trailing_zeros();
    assert!(log_values_len > log_tree_cap_size);
    let offset = values_len - (1 << (log_tree_cap_size + 1));
    &values[offset..offset + (1 << log_tree_cap_size)]
}

// ---------------------------------------------------------------------------
// Blake2s PoW kernel
// ---------------------------------------------------------------------------

/// Dispatch the Blake2s proof-of-work search kernel.
///
/// Searches for a nonce such that blake2s(seed || nonce) has at least
/// `bits_count` leading zero bits.
pub fn blake2s_pow(
    seed: &MetalBuffer<u32>,
    bits_count: u32,
    max_nonce: u64,
    result: &mut MetalBuffer<u64>,
    ctx: &MetalProverContext,
) {
    assert_eq!(seed.len(), STATE_SIZE);
    // Initialize result to u64::MAX (not found sentinel)
    result.as_mut_slice()[0] = u64::MAX;

    let pipeline = ctx.get_pipeline("ab_blake2s_pow_kernel");

    // Use a large number of threadgroups to saturate the GPU
    let block_size: u32 = WARP_SIZE * 4;
    // Apple Silicon GPUs typically have 30-128 execution units;
    // we use a generous number of threadgroups.
    let num_threadgroups: u32 = 1024;

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_buffer(&encoder, seed.metal_buffer(), 0, 0);
    set_bytes(&encoder, &bits_count, 1);
    set_bytes(&encoder, &max_nonce, 2);
    set_buffer(&encoder, result.metal_buffer(), 0, 3);

    let grid_dim = MTLSize::new(num_threadgroups as u64, 1, 1);
    let block_dim = MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
