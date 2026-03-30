///! Individual NTT kernel dispatch functions for Metal.
///!
///! Each function encodes a specific NTT stage kernel into a command buffer
///! and dispatches it. The kernel names correspond to the compiled .metallib
///! function names.

use metal::{ComputeCommandEncoderRef, ComputePipelineState, MTLSize};

use crate::device_structures::{MetalMatrixChunkImpl, MetalMatrixChunkMutImpl, PtrAndStride, MutPtrAndStride};
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;

type BF = BaseField;

/// Helper to set bytes on a compute encoder at a given argument index.
fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const u8;
    let len = std::mem::size_of::<T>();
    encoder.set_bytes(index, unsafe { std::slice::from_raw_parts(ptr, len) }, len as u64);
}

/// Dispatch a single-stage B2N (bitrev-Z to natural) NTT kernel.
///
/// Used for small NTTs (log_n < 16) where the entire transform fits
/// in a single kernel launch.
pub fn dispatch_b2n_one_stage(
    input_ps: &PtrAndStride<BF>,
    output_ps: &PtrAndStride<BF>,
    start_stage: u32,
    log_n: u32,
    blocks_per_ntt: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    pipeline: &ComputePipelineState,
    ctx: &MetalProverContext,
) {
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    set_bytes(&encoder, input_ps, 0);
    set_bytes(&encoder, output_ps, 1);
    set_bytes(&encoder, &start_stage, 2);
    set_bytes(&encoder, &log_n, 3);
    set_bytes(&encoder, &blocks_per_ntt, 4);
    set_bytes(&encoder, &log_extension_degree, 5);
    set_bytes(&encoder, &coset_idx, 6);

    // Pass twiddle factor buffers
    encoder.set_buffer(
        7,
        Some(ctx.device_context.powers_of_w_fine_bitrev_for_ntt.metal_buffer()),
        0,
    );
    encoder.set_buffer(
        8,
        Some(ctx.device_context.powers_of_w_coarse_bitrev_for_ntt.metal_buffer()),
        0,
    );

    let n = 1usize << log_n;
    let block_dim = MTLSize::new(32, 1, 1);
    let grid_dim = MTLSize::new(((n + 31) / 32) as u64, 1, 1);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch a multi-stage B2N NTT kernel.
///
/// Processes `stages_this_launch` butterfly stages in a single kernel,
/// reading from one buffer and writing to another (or the same for in-place).
pub fn dispatch_b2n_multi_stage(
    input_ps: &PtrAndStride<BF>,
    output_ps: &PtrAndStride<BF>,
    start_stage: u32,
    stages_this_launch: u32,
    log_n: u32,
    num_z_cols: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    grid_offset: u32,
    pipeline: &ComputePipelineState,
    threadgroup_mem_size: usize,
    ctx: &MetalProverContext,
) {
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    set_bytes(&encoder, input_ps, 0);
    set_bytes(&encoder, output_ps, 1);
    set_bytes(&encoder, &start_stage, 2);
    set_bytes(&encoder, &stages_this_launch, 3);
    set_bytes(&encoder, &log_n, 4);
    set_bytes(&encoder, &num_z_cols, 5);
    set_bytes(&encoder, &log_extension_degree, 6);
    set_bytes(&encoder, &coset_idx, 7);
    set_bytes(&encoder, &grid_offset, 8);

    // Twiddle buffers
    encoder.set_buffer(
        9,
        Some(ctx.device_context.powers_of_w_fine_bitrev_for_ntt.metal_buffer()),
        0,
    );
    encoder.set_buffer(
        10,
        Some(ctx.device_context.powers_of_w_coarse_bitrev_for_ntt.metal_buffer()),
        0,
    );

    let n = 1usize << log_n;
    let blocks_per_ntt = n / threadgroup_mem_size;
    let total_threadgroups = blocks_per_ntt * num_z_cols as usize;

    let block_dim = MTLSize::new(
        std::cmp::min(threadgroup_mem_size, 1024) as u64,
        1,
        1,
    );
    let grid_dim = MTLSize::new(total_threadgroups as u64, 1, 1);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch a single-stage N2B (natural to bitrev-Z) NTT kernel.
pub fn dispatch_n2b_one_stage(
    input_ps: &PtrAndStride<BF>,
    output_ps: &PtrAndStride<BF>,
    start_stage: u32,
    log_n: u32,
    blocks_per_ntt: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    pipeline: &ComputePipelineState,
    ctx: &MetalProverContext,
) {
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    set_bytes(&encoder, input_ps, 0);
    set_bytes(&encoder, output_ps, 1);
    set_bytes(&encoder, &start_stage, 2);
    set_bytes(&encoder, &log_n, 3);
    set_bytes(&encoder, &blocks_per_ntt, 4);
    set_bytes(&encoder, &log_extension_degree, 5);
    set_bytes(&encoder, &coset_idx, 6);

    // Twiddle buffers (inverse)
    encoder.set_buffer(
        7,
        Some(ctx.device_context.powers_of_w_inv_fine_bitrev_for_ntt.metal_buffer()),
        0,
    );
    encoder.set_buffer(
        8,
        Some(ctx.device_context.powers_of_w_inv_coarse_bitrev_for_ntt.metal_buffer()),
        0,
    );

    let n = 1usize << log_n;
    let block_dim = MTLSize::new(32, 1, 1);
    let grid_dim = MTLSize::new(((n + 31) / 32) as u64, 1, 1);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch a multi-stage N2B NTT kernel (inverse direction).
pub fn dispatch_n2b_multi_stage(
    input_ps: &PtrAndStride<BF>,
    output_ps: &PtrAndStride<BF>,
    start_stage: u32,
    stages_this_launch: u32,
    log_n: u32,
    num_z_cols: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    grid_offset: u32,
    pipeline: &ComputePipelineState,
    threadgroup_mem_size: usize,
    ctx: &MetalProverContext,
) {
    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    set_bytes(&encoder, input_ps, 0);
    set_bytes(&encoder, output_ps, 1);
    set_bytes(&encoder, &start_stage, 2);
    set_bytes(&encoder, &stages_this_launch, 3);
    set_bytes(&encoder, &log_n, 4);
    set_bytes(&encoder, &num_z_cols, 5);
    set_bytes(&encoder, &log_extension_degree, 6);
    set_bytes(&encoder, &coset_idx, 7);
    set_bytes(&encoder, &grid_offset, 8);

    // Twiddle buffers (inverse)
    encoder.set_buffer(
        9,
        Some(ctx.device_context.powers_of_w_inv_fine_bitrev_for_ntt.metal_buffer()),
        0,
    );
    encoder.set_buffer(
        10,
        Some(ctx.device_context.powers_of_w_inv_coarse_bitrev_for_ntt.metal_buffer()),
        0,
    );

    let n = 1usize << log_n;
    let blocks_per_ntt = n / threadgroup_mem_size;
    let total_threadgroups = blocks_per_ntt * num_z_cols as usize;

    let block_dim = MTLSize::new(
        std::cmp::min(threadgroup_mem_size, 1024) as u64,
        1,
        1,
    );
    let grid_dim = MTLSize::new(total_threadgroups as u64, 1, 1);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
