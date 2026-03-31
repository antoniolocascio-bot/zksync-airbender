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
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
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

    encoder.dispatch_thread_groups(grid_dim, block_dim);
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

    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch one layer of the nodes kernel with explicit byte offsets.
///
/// Reads `2 * count_out` consecutive `Digest` values starting at `values_byte_offset`
/// in `values_buf`, and writes `count_out` parent digests at `results_byte_offset` in
/// `results_buf`.  The two buffers may be the same (valid for in-place tree building).
fn dispatch_nodes_layer(
    values_buf: &MTLBuffer,
    values_byte_offset: u64,
    results_buf: &MTLBuffer,
    results_byte_offset: u64,
    count_out: u32,
    ctx: &MetalProverContext,
) {
    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count_out);
    let pipeline = ctx.get_pipeline("ab_blake2s_nodes_kernel");
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(values_buf), values_byte_offset);
    encoder.set_buffer(1, Some(results_buf), results_byte_offset);
    set_bytes(&encoder, &count_out, 2);
    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Build merkle tree node layers.
///
/// Tree node layout mirrors the CUDA implementation:
///   `values`  (size n) → `results[0..n/2]`       (layer 1)
///   `results[0..n/2]` → `results[n/2..3n/4]`      (layer 2)
///   …
/// The root ends up at the last occupied position in `results`.
///
/// `values` and `results` must both have the same power-of-two length n.
/// Metal `set_buffer` byte-offset support lets both point into the same underlying
/// MTLBuffer (which is required by `build_merkle_tree`).
pub fn build_merkle_tree_nodes(
    values: &MetalBuffer<Digest>,
    results: &mut MetalBuffer<Digest>,
    layers_count: u32,
    ctx: &MetalProverContext,
) {
    if layers_count == 0 {
        return;
    }
    let n = values.len();
    assert!(n.is_power_of_two());
    assert_eq!(n, results.len());

    const DIGEST_BYTES: u64 = std::mem::size_of::<Digest>() as u64;
    let results_buf = results.metal_buffer().clone();
    // Borrow values MTLBuffer for the first layer; subsequent layers read from results.
    let values_buf = values.metal_buffer().clone();

    // Layer 0: hash values[0..n] → results[0..n/2]
    // Layer k>0: src = results at prev dst_off, dst = results at new dst_off
    let mut use_values_buf = true; // first layer sources from `values`, not `results`
    let mut src_off: u64 = 0;
    let mut dst_off: u64 = 0;
    let mut count_in = n;

    for _ in 0..layers_count {
        let count_out = (count_in / 2) as u32;
        let src = if use_values_buf { &values_buf } else { &results_buf };
        dispatch_nodes_layer(src, src_off, &results_buf, dst_off, count_out, ctx);
        use_values_buf = false; // all layers after the first read from results
        src_off = dst_off;
        dst_off += count_out as u64 * DIGEST_BYTES;
        count_in /= 2;
    }
}

/// Build a complete merkle tree: leaves + all node layers.
///
/// `results` layout (size = 2 × leaves_count):
///   `[0, leaves_count)` — leaf hashes (one per `1 << log_rows_per_hash` rows of `values`)
///   `[leaves_count, 2×leaves_count)` — packed node layers (layer 1 first, root last)
///
/// The tree cap can be extracted with `merkle_tree_cap(results.as_slice(), log_cap)`.
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

    const DIGEST_BYTES: u64 = std::mem::size_of::<Digest>() as u64;

    // Build leaf hashes into a temporary buffer then copy to results[0..leaves_count].
    // (Using a temp avoids needing an offset-aware leaves kernel.)
    let mut leaves_tmp = MetalBuffer::<Digest>::new(&ctx.device, leaves_count);
    build_merkle_tree_leaves(values, &mut leaves_tmp, log_rows_per_hash, ctx);

    if bit_reverse_leaves {
        // TODO: implement bit-reversal for Digest buffers when needed.
        // For now this is only called with bit_reverse_leaves=false in the prover.
        unimplemented!("bit_reverse_leaves for Digest buffers not yet implemented");
    }

    // Copy leaf hashes into the first half of results (unified memory = plain memcpy).
    let digest_size = std::mem::size_of::<Digest>();
    unsafe {
        std::ptr::copy_nonoverlapping(
            leaves_tmp.as_ptr() as *const u8,
            results.as_mut_ptr() as *mut u8,
            leaves_count * digest_size,
        );
    }
    drop(leaves_tmp);

    // Build node layers into results[leaves_count..2*leaves_count].
    // Mirrors CUDA: build_merkle_tree_nodes(leaves, nodes, layers_count-1)
    // where leaves = results[0..L], nodes = results[L..2L].
    // We implement this as a single iterative loop using byte offsets:
    //   Layer 0: hash results[0..L]  → results[L..3L/2]
    //   Layer 1: hash results[L..3L/2] → results[3L/2..7L/4]
    //   …
    let results_buf = results.metal_buffer().clone();
    let nodes_start_bytes = (leaves_count as u64) * DIGEST_BYTES;

    let mut src_off: u64 = 0; // reads leaves (results[0..L]) for the first node layer
    let mut dst_off: u64 = nodes_start_bytes;
    let mut count_in = leaves_count;

    for _ in 0..(layers_count - 1) {
        let count_out = (count_in / 2) as u32;
        dispatch_nodes_layer(&results_buf, src_off, &results_buf, dst_off, count_out, ctx);
        src_off = dst_off;
        dst_off += count_out as u64 * DIGEST_BYTES;
        count_in /= 2;
    }
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
    encoder.dispatch_thread_groups(grid_dim, block_dim);
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

    encoder.dispatch_thread_groups(grid_dim_2d, block_dim_2d);
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
    encoder.dispatch_thread_groups(grid_dim, block_dim);
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
    encoder.dispatch_thread_groups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
