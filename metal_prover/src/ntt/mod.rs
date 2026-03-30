#![allow(non_snake_case)]

pub mod kernels;

pub use kernels::*;

use crate::device_structures::{
    MetalBuffer, MetalMatrixChunkImpl, MetalMatrixChunkMutImpl, MutPtrAndStride, PtrAndStride,
};
use crate::field::{BaseField, Ext2Field};
use crate::prover::context::MetalProverContext;

type BF = BaseField;
type E2 = Ext2Field;

/// NTT kernel plan types, matching the CUDA version's stage plan approach.
/// These describe how to decompose a large NTT into multiple kernel launches.

#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum N2B_LAUNCH {
    FINAL_7_WARP,
    FINAL_8_WARP,
    FINAL_9_TO_12_BLOCK,
    NONFINAL_7_OR_8_BLOCK,
}

#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum B2N_LAUNCH {
    INITIAL_7_WARP,
    INITIAL_8_WARP,
    INITIAL_9_TO_12_BLOCK,
    NONINITIAL_7_OR_8_BLOCK,
}

/// Kernel plans for NTT sizes 2^16..2^24.
#[allow(non_camel_case_types)]
pub type N2B_Plan = [Option<(N2B_LAUNCH, usize, usize)>; 3];

pub const STAGE_PLANS_N2B: [N2B_Plan; 9] = [
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::FINAL_8_WARP, 8, 4 * 256)),
        None,
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::FINAL_9_TO_12_BLOCK, 9, 4096)),
        None,
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::FINAL_9_TO_12_BLOCK, 10, 4096)),
        None,
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::FINAL_9_TO_12_BLOCK, 11, 4096)),
        None,
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::FINAL_9_TO_12_BLOCK, 12, 4096)),
        None,
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 7, 4096)),
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 7, 4096)),
        Some((N2B_LAUNCH::FINAL_7_WARP, 7, 4 * 128)),
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 7, 4096)),
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 7, 4096)),
        Some((N2B_LAUNCH::FINAL_8_WARP, 8, 4 * 256)),
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 7, 4096)),
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::FINAL_8_WARP, 8, 4 * 256)),
    ],
    [
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK, 8, 4096)),
        Some((N2B_LAUNCH::FINAL_8_WARP, 8, 4 * 256)),
    ],
];

#[allow(non_camel_case_types)]
pub type B2N_Plan = [Option<(B2N_LAUNCH, usize, usize)>; 3];

pub const STAGE_PLANS_B2N: [B2N_Plan; 9] = [
    [
        Some((B2N_LAUNCH::INITIAL_8_WARP, 8, 4 * 256)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
        None,
    ],
    [
        Some((B2N_LAUNCH::INITIAL_9_TO_12_BLOCK, 9, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
        None,
    ],
    [
        Some((B2N_LAUNCH::INITIAL_9_TO_12_BLOCK, 10, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
        None,
    ],
    [
        Some((B2N_LAUNCH::INITIAL_9_TO_12_BLOCK, 11, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
        None,
    ],
    [
        Some((B2N_LAUNCH::INITIAL_9_TO_12_BLOCK, 12, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
        None,
    ],
    [
        Some((B2N_LAUNCH::INITIAL_7_WARP, 7, 4 * 128)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 7, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 7, 4096)),
    ],
    [
        Some((B2N_LAUNCH::INITIAL_8_WARP, 8, 4 * 256)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 7, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 7, 4096)),
    ],
    [
        Some((B2N_LAUNCH::INITIAL_8_WARP, 8, 4 * 256)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 7, 4096)),
    ],
    [
        Some((B2N_LAUNCH::INITIAL_8_WARP, 8, 4 * 256)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
        Some((B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK, 8, 4096)),
    ],
];

/// Columns per block for the vectorized NTT layout.
pub const REAL_COLS_PER_BLOCK: usize = 8;
pub const COMPLEX_COLS_PER_BLOCK: usize = 4;

/// Get the kernel launch chain for a given log_n NTT size.
pub fn get_main_to_coset_launch_chain(
    log_n: usize,
) -> (
    Vec<(N2B_LAUNCH, usize, usize)>,
    Vec<(B2N_LAUNCH, usize, usize)>,
) {
    assert!(log_n >= 16);
    let n2b_plan = &STAGE_PLANS_N2B[log_n - 16];
    let b2n_plan = &STAGE_PLANS_B2N[log_n - 16];
    let n2b_launches: Vec<_> = n2b_plan.iter().filter_map(|&x| x).collect();
    let b2n_launches: Vec<_> = b2n_plan.iter().filter_map(|&x| x).collect();
    assert_eq!(n2b_launches.len(), b2n_launches.len());
    (n2b_launches, b2n_launches)
}

/// Perform bitrev-Z to natural-order evaluation domain transform.
///
/// This is the main forward NTT used by the prover: given coefficients in
/// bit-reversed Z-basis, produce evaluations in natural order over a coset.
pub fn bitrev_Z_to_natural_evals(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    ctx: &MetalProverContext,
) {
    assert!(log_n >= 1);
    assert!(num_bf_cols > 0);
    assert!(log_extension_degree <= 2);

    // For small sizes (log_n < 16), use a single-stage kernel.
    // For larger sizes, decompose using the stage plans.
    if log_n < 16 {
        // Single-stage dispatch
        dispatch_b2n_single_stage(inputs, outputs, log_n, num_bf_cols, log_extension_degree, coset_idx, ctx);
    } else {
        // Multi-stage dispatch using launch chain
        dispatch_b2n_multi_stage(inputs, outputs, log_n, num_bf_cols, log_extension_degree, coset_idx, ctx);
    }
}

/// Perform natural-order evaluation to bitrev-Z coefficient transform.
///
/// This is the inverse NTT: given evaluations in natural order over a coset,
/// produce coefficients in bit-reversed Z-basis.
pub fn natural_evals_to_bitrev_Z(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    ctx: &MetalProverContext,
) {
    assert!(log_n >= 1);
    assert!(num_bf_cols > 0);
    assert!(log_extension_degree <= 2);

    if log_n < 16 {
        dispatch_n2b_single_stage(inputs, outputs, log_n, num_bf_cols, log_extension_degree, coset_idx, ctx);
    } else {
        dispatch_n2b_multi_stage(inputs, outputs, log_n, num_bf_cols, log_extension_degree, coset_idx, ctx);
    }
}

// ---------------------------------------------------------------------------
// Internal dispatch helpers
// ---------------------------------------------------------------------------

fn dispatch_b2n_single_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    ctx: &MetalProverContext,
) {
    let pipeline = ctx.get_pipeline("ab_bitrev_Z_to_natural_coset_evals_one_stage");
    let n = 1usize << log_n;

    let input_ps = inputs.as_ptr_and_stride();
    let output_ps = outputs.as_ptr_and_stride();
    let log_n_u32 = log_n as u32;
    let log_ext = log_extension_degree as u32;
    let coset = coset_idx as u32;

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_ntt_buffers(&encoder, &input_ps, &output_ps, log_n_u32, log_ext, coset, ctx);

    let block_dim = metal::MTLSize::new(32, 1, 1);
    let grid_dim = metal::MTLSize::new(
        ((n + 31) / 32) as u64,
        num_bf_cols as u64,
        1,
    );
    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

fn dispatch_b2n_multi_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    ctx: &MetalProverContext,
) {
    let (_n2b_launches, b2n_launches) = get_main_to_coset_launch_chain(log_n);

    // Process each stage in sequence, using the stage plan to select kernels
    // and compute grid/block dimensions.
    let mut start_stage = 0u32;
    for (i, &(launch_type, stages_this_launch, threadgroup_mem)) in b2n_launches.iter().enumerate() {
        let kernel_name = match launch_type {
            B2N_LAUNCH::INITIAL_7_WARP => "ab_bitrev_Z_to_natural_coset_evals_initial_7_stages_warp",
            B2N_LAUNCH::INITIAL_8_WARP => "ab_bitrev_Z_to_natural_coset_evals_initial_8_stages_warp",
            B2N_LAUNCH::INITIAL_9_TO_12_BLOCK => "ab_bitrev_Z_to_natural_coset_evals_initial_9_to_12_stages_block",
            B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK => "ab_bitrev_Z_to_natural_coset_evals_noninitial_7_or_8_stages_block",
        };

        let pipeline = ctx.get_pipeline(kernel_name);

        let num_z_cols = ((num_bf_cols + 1) / 2) as u32; // complex columns
        let n = 1usize << log_n;
        let blocks_per_ntt = n / threadgroup_mem;
        let total_threadgroups = blocks_per_ntt * num_z_cols as usize;

        let input_ps = inputs.as_ptr_and_stride();
        let output_ps = outputs.as_ptr_and_stride();

        let command_buffer = ctx.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);

        set_ntt_buffers(
            &encoder,
            &input_ps,
            &output_ps,
            log_n as u32,
            log_extension_degree as u32,
            coset_idx as u32,
            ctx,
        );

        let stages_u32 = stages_this_launch as u32;
        set_bytes_on_encoder(&encoder, &start_stage, 6);
        set_bytes_on_encoder(&encoder, &stages_u32, 7);
        set_bytes_on_encoder(&encoder, &num_z_cols, 8);

        let block_dim = metal::MTLSize::new(
            std::cmp::min(threadgroup_mem, 1024) as u64,
            1,
            1,
        );
        let grid_dim = metal::MTLSize::new(total_threadgroups as u64, 1, 1);

        encoder.dispatch_threadgroups(grid_dim, block_dim);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        start_stage += stages_this_launch as u32;
    }
}

fn dispatch_n2b_single_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    _num_bf_cols: usize,
    _log_extension_degree: usize,
    _coset_idx: usize,
    ctx: &MetalProverContext,
) {
    // TODO: Implement single-stage N2B kernel dispatch.
    // This is the inverse direction (natural evals -> bitrev Z).
    let _ = (inputs, outputs, log_n, ctx);
}

fn dispatch_n2b_multi_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    ctx: &MetalProverContext,
) {
    let (n2b_launches, _b2n_launches) = get_main_to_coset_launch_chain(log_n);

    let mut start_stage = 0u32;
    for (i, &(launch_type, stages_this_launch, threadgroup_mem)) in n2b_launches.iter().enumerate() {
        let kernel_name = match launch_type {
            N2B_LAUNCH::FINAL_7_WARP => "ab_natural_coset_evals_to_bitrev_Z_final_7_stages_warp",
            N2B_LAUNCH::FINAL_8_WARP => "ab_natural_coset_evals_to_bitrev_Z_final_8_stages_warp",
            N2B_LAUNCH::FINAL_9_TO_12_BLOCK => "ab_natural_coset_evals_to_bitrev_Z_final_9_to_12_stages_block",
            N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK => "ab_natural_coset_evals_to_bitrev_Z_nonfinal_7_or_8_stages_block",
        };

        let pipeline = ctx.get_pipeline(kernel_name);

        let num_z_cols = ((num_bf_cols + 1) / 2) as u32;
        let n = 1usize << log_n;
        let blocks_per_ntt = n / threadgroup_mem;
        let total_threadgroups = blocks_per_ntt * num_z_cols as usize;

        let input_ps = inputs.as_ptr_and_stride();
        let output_ps = outputs.as_ptr_and_stride();

        let command_buffer = ctx.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);

        set_ntt_buffers(
            &encoder,
            &input_ps,
            &output_ps,
            log_n as u32,
            log_extension_degree as u32,
            coset_idx as u32,
            ctx,
        );

        let stages_u32 = stages_this_launch as u32;
        set_bytes_on_encoder(&encoder, &start_stage, 6);
        set_bytes_on_encoder(&encoder, &stages_u32, 7);
        set_bytes_on_encoder(&encoder, &num_z_cols, 8);

        let block_dim = metal::MTLSize::new(
            std::cmp::min(threadgroup_mem, 1024) as u64,
            1,
            1,
        );
        let grid_dim = metal::MTLSize::new(total_threadgroups as u64, 1, 1);

        encoder.dispatch_threadgroups(grid_dim, block_dim);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        start_stage += stages_this_launch as u32;
    }
}

// ---------------------------------------------------------------------------
// Encoder helpers
// ---------------------------------------------------------------------------

fn set_ntt_buffers(
    encoder: &metal::ComputeCommandEncoderRef,
    input_ps: &PtrAndStride<BF>,
    output_ps: &PtrAndStride<BF>,
    log_n: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    ctx: &MetalProverContext,
) {
    set_bytes_on_encoder(encoder, input_ps, 0);
    set_bytes_on_encoder(encoder, output_ps, 1);
    set_bytes_on_encoder(encoder, &log_n, 2);
    set_bytes_on_encoder(encoder, &log_extension_degree, 3);
    set_bytes_on_encoder(encoder, &coset_idx, 4);

    // Pass twiddle factor buffers
    encoder.set_buffer(
        5,
        Some(ctx.device_context.powers_of_w_fine_bitrev_for_ntt.metal_buffer()),
        0,
    );
}

fn set_bytes_on_encoder<T: Sized>(encoder: &metal::ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const u8;
    let len = std::mem::size_of::<T>();
    encoder.set_bytes(index, unsafe { std::slice::from_raw_parts(ptr, len) }, len as u64);
}
