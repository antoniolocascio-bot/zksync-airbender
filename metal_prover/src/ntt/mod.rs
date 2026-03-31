#![allow(non_snake_case)]

pub mod kernels;

pub use kernels::*;

use metal::MTLSize;

use crate::device_structures::{
    MetalMatrixChunkImpl, MetalMatrixChunkMutImpl,
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

    if log_n < 16 {
        dispatch_b2n_single_stage(inputs, outputs, log_n, num_bf_cols, log_extension_degree, coset_idx, ctx);
    } else {
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

// Kernel sizing constants (must match the Metal shaders).
// VALS_PER_WARP for 8-stage kernel (LOG_VALS_PER_THREAD=3): 32 threads * 8 vals/thread = 256
const VALS_PER_WARP_8STAGE: usize = 256;
// VALS_PER_WARP for 7-stage kernel (LOG_VALS_PER_THREAD=2): 32 threads * 4 vals/thread = 128
const VALS_PER_WARP_7STAGE: usize = 128;
// [[max_total_threads_per_threadgroup(128)]] → max 4 warps per block
const MAX_WARPS_PER_BLOCK: usize = 4;

/// Compute grid/block dimensions for a warp-level NTT kernel.
///
/// The kernel formula: `gmem_offset = vals_per_warp * 4 * block_x + vals_per_warp * warp_id`
/// Each block processes `vals_per_warp * warps_per_block` elements.
///
/// Returns `(threads_per_block, blocks_per_col, col_groups)`.
fn ntt_warp_dims(
    n: usize,
    num_z_cols: usize,
    vals_per_warp: usize,
) -> (usize, usize, usize) {
    let num_warps_needed = (n / vals_per_warp).max(1);
    let warps_per_block = num_warps_needed.min(MAX_WARPS_PER_BLOCK);
    let threads_per_block = warps_per_block * 32;
    let blocks_per_col = ((num_warps_needed + MAX_WARPS_PER_BLOCK - 1) / MAX_WARPS_PER_BLOCK).max(1);
    let col_groups = ((num_z_cols + COMPLEX_COLS_PER_BLOCK - 1) / COMPLEX_COLS_PER_BLOCK).max(1);
    (threads_per_block, blocks_per_col, col_groups)
}

/// Create a zero-copy Metal buffer wrapping existing memory, returning the buffer and the
/// byte offset to use with `set_buffer`.
///
/// Metal's `newBufferWithBytesNoCopy` requires a 4096-byte aligned base pointer.
/// Metal buffers on macOS are only guaranteed 256-byte aligned, so we round down to the
/// enclosing page and return the offset from page start to the actual data start.
///
/// On Apple Silicon (unified memory, `StorageModeShared`) this is zero-copy.
unsafe fn wrap_as_metal_buffer(
    ptr: *const std::ffi::c_void,
    byte_len: usize,
    device: &metal::Device,
) -> (metal::Buffer, u64) {
    const PAGE_SIZE: usize = 4096;
    let addr = ptr as usize;
    let page_base = addr & !(PAGE_SIZE - 1);
    let offset = addr - page_base;
    // Round up total coverage to page boundary (Metal requires page-aligned size too)
    let alloc_size = ((offset + byte_len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)).max(PAGE_SIZE);
    let mtl_buf = device.new_buffer_with_bytes_no_copy(
        page_base as *const std::ffi::c_void,
        alloc_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
        None,
    );
    (mtl_buf, offset as u64)
}

/// Forward NTT (bitrev-Z → natural evals) for log_n < 16.
///
/// Uses the existing warp-level initial-stage kernels:
/// - 8-stage: `ab_bitrev_Z_to_natural_coset_evals_initial_8_stages_warp`  (log_n ≠ 7)
/// - 7-stage: `ab_bitrev_Z_to_natural_coset_evals_initial_7_stages_warp`  (log_n = 7)
///
/// Buffer layout (matches the Metal shader exactly):
///   0: device const bf *gmem_in      — raw CPU/GPU ptr (unified memory)
///   1: device bf *gmem_out            — raw CPU/GPU ptr
///   2: constant size_t &in_stride
///   3: constant size_t &out_stride
///   4: constant unsigned &start_stage
///   5: constant unsigned &stages_this_launch
///   6: constant unsigned &log_n
///   7: constant unsigned &num_Z_cols
///   8: constant unsigned &log_extension_degree
///   9: constant unsigned &coset_idx
///  10: constant unsigned &grid_offset
///  11: device const e2f *twiddle_fine        (forward bitrev twiddles, fine layer)
///  12: constant unsigned &twiddle_fine_mask
///  13: constant unsigned &twiddle_fine_log
///  14: device const e2f *twiddle_coarse      (forward bitrev twiddles, coarse layer)
///  15: constant unsigned &twiddle_coarse_mask
///  16: constant unsigned &twiddle_coarse_log
///  17: device const e2f *powers_fine         (eval-domain powers, fine)
///  18: constant unsigned &powers_fine_mask
///  19: constant unsigned &powers_fine_log
///  20: device const e2f *powers_coarser      (eval-domain powers, coarser)
///  21: constant unsigned &powers_coarser_mask
///  22: constant unsigned &powers_coarser_log
///  23: device const e2f *powers_coarsest     (eval-domain powers, coarsest)
///  24: constant unsigned &powers_coarsest_mask
fn dispatch_b2n_single_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    ctx: &MetalProverContext,
) {
    assert!(log_n >= 1 && log_n <= 15,
        "dispatch_b2n_single_stage: log_n must be in [1, 15], got {}", log_n);

    let n = 1usize << log_n;
    let num_z_cols = (num_bf_cols + 1) / 2;

    // Choose kernel variant: 7-stage for log_n=7, 8-stage otherwise.
    let (kernel_name, stages_this_launch, vals_per_warp) = if log_n == 7 {
        ("ab_bitrev_Z_to_natural_coset_evals_initial_7_stages_warp", 7u32, VALS_PER_WARP_7STAGE)
    } else {
        ("ab_bitrev_Z_to_natural_coset_evals_initial_8_stages_warp", 8u32, VALS_PER_WARP_8STAGE)
    };

    let pipeline = ctx.get_pipeline(kernel_name);

    let (threads_per_block, blocks_per_col, col_groups) = ntt_warp_dims(n, num_z_cols, vals_per_warp);
    let warps_per_block = threads_per_block / 32;
    let smem_bytes = (warps_per_block * vals_per_warp * std::mem::size_of::<E2>()) as u64;

    let grid = MTLSize::new(blocks_per_col as u64, col_groups as u64, 1);
    let block = MTLSize::new(threads_per_block as u64, 1, 1);

    // Create zero-copy Metal buffer wrappers for input and output.
    // The two-for-one kernel may access up to 2 * n * num_z_cols BF elements.
    // We cover the full matrix allocation (total_len) which is sufficient for even num_bf_cols.
    // For odd num_bf_cols the kernel reads one extra column past the end (into page padding,
    // which is zero on fresh Apple Silicon allocations with StorageModeShared).
    let in_byte_len = inputs.total_len() * std::mem::size_of::<BF>();
    let out_byte_len = outputs.total_len() * std::mem::size_of::<BF>();
    let (in_mtl, in_off) = unsafe {
        wrap_as_metal_buffer(inputs.as_ptr() as *const std::ffi::c_void, in_byte_len, &ctx.device)
    };
    let (out_mtl, out_off) = unsafe {
        wrap_as_metal_buffer(outputs.as_ptr() as *const std::ffi::c_void, out_byte_len, &ctx.device)
    };

    let in_stride = inputs.stride() as u64;
    let out_stride = outputs.stride() as u64;

    let start_stage = 0u32;
    let log_n_u32 = log_n as u32;
    let log_ext = log_extension_degree as u32;
    let coset = coset_idx as u32;
    let grid_offset = 0u32;
    let num_z_cols_u32 = num_z_cols as u32;

    let dc = &ctx.device_context;
    let ntt_fwd = dc.get_powers_data_w_bitrev_for_ntt();
    let pw = dc.get_powers_data_w();

    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    // Data buffers via set_buffer (required for device T* kernel arguments).
    // The byte offset accounts for sub-page pointer alignment.
    enc.set_buffer(0, Some(&in_mtl), in_off);
    enc.set_buffer(1, Some(&out_mtl), out_off);
    // Strides
    set_bytes_on_encoder(&enc, &in_stride, 2);
    set_bytes_on_encoder(&enc, &out_stride, 3);
    // Scalar params
    set_bytes_on_encoder(&enc, &start_stage, 4);
    set_bytes_on_encoder(&enc, &stages_this_launch, 5);
    set_bytes_on_encoder(&enc, &log_n_u32, 6);
    set_bytes_on_encoder(&enc, &num_z_cols_u32, 7);
    set_bytes_on_encoder(&enc, &log_ext, 8);
    set_bytes_on_encoder(&enc, &coset, 9);
    set_bytes_on_encoder(&enc, &grid_offset, 10);
    // Forward twiddle buffers (fine + coarse, bit-reversed)
    enc.set_buffer(11, Some(dc.powers_of_w_fine_bitrev_for_ntt.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &ntt_fwd.fine.mask, 12);
    set_bytes_on_encoder(&enc, &ntt_fwd.fine.log_count, 13);
    enc.set_buffer(14, Some(dc.powers_of_w_coarse_bitrev_for_ntt.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &ntt_fwd.coarse.mask, 15);
    set_bytes_on_encoder(&enc, &ntt_fwd.coarse.log_count, 16);
    // Eval-domain powers (3-layer decomposition for LDE scale/shift)
    enc.set_buffer(17, Some(dc.powers_of_w_fine.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &pw.fine.mask, 18);
    set_bytes_on_encoder(&enc, &pw.fine.log_count, 19);
    enc.set_buffer(20, Some(dc.powers_of_w_coarser.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &pw.coarser.mask, 21);
    set_bytes_on_encoder(&enc, &pw.coarser.log_count, 22);
    enc.set_buffer(23, Some(dc.powers_of_w_coarsest.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &pw.coarsest.mask, 24);

    enc.set_threadgroup_memory_length(0, smem_bytes);
    enc.dispatch_thread_groups(grid, block);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

/// Inverse NTT (natural evals → bitrev-Z) for log_n < 16.
///
/// Uses the existing warp-level final-stage kernels:
/// - Main domain (log_extension_degree = 0): `ab_main_domain_evals_to_Z_final_8_stages_warp`
/// - Coset (log_extension_degree > 0):       `ab_coset_evals_to_Z_final_8_stages_warp`
///
/// Buffer layout (matches the Metal shader exactly):
///   0: device const bf *gmem_in      — raw CPU/GPU ptr
///   1: device bf *gmem_out            — raw CPU/GPU ptr
///   2: constant size_t &in_stride
///   3: constant size_t &out_stride
///   4: constant unsigned &start_stage
///   5: constant unsigned &stages_this_launch
///   6: constant unsigned &log_n
///   7: constant unsigned &num_Z_cols
///   8: constant unsigned &grid_offset   (no log_extension_degree/coset_idx here!)
///   9: device const e2f *twiddle_fine   (inverse bitrev twiddles, fine layer)
///  10: constant unsigned &twiddle_fine_mask
///  11: constant unsigned &twiddle_fine_log
///  12: device const e2f *twiddle_coarse (inverse bitrev twiddles, coarse layer)
///  13: constant unsigned &twiddle_coarse_mask
///  14: constant unsigned &twiddle_coarse_log
///  15: device const e2f *powers_fine    (eval-domain powers, fine)
///  16: constant unsigned &powers_fine_mask
///  17: constant unsigned &powers_fine_log
///  18: device const e2f *powers_coarser
///  19: constant unsigned &powers_coarser_mask
///  20: constant unsigned &powers_coarser_log
///  21: device const e2f *powers_coarsest
///  22: constant unsigned &powers_coarsest_mask
///  23: device const bf *inv_sizes
fn dispatch_n2b_single_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    _coset_idx: usize,
    ctx: &MetalProverContext,
) {
    assert!(log_n >= 1 && log_n <= 15,
        "dispatch_n2b_single_stage: log_n must be in [1, 15], got {}", log_n);

    let n = 1usize << log_n;
    let num_z_cols = (num_bf_cols + 1) / 2;

    // N2B kernels encode the coset flag in their name (template specialization).
    // Only 8-stage variants exist; log_n=7 would need extra work but is not tested.
    let (kernel_name, stages_this_launch) = if log_extension_degree == 0 {
        ("ab_main_domain_evals_to_Z_final_8_stages_warp", 8u32)
    } else {
        ("ab_coset_evals_to_Z_final_8_stages_warp", 8u32)
    };

    let vals_per_warp = VALS_PER_WARP_8STAGE;
    let pipeline = ctx.get_pipeline(kernel_name);

    let (threads_per_block, blocks_per_col, col_groups) = ntt_warp_dims(n, num_z_cols, vals_per_warp);
    let warps_per_block = threads_per_block / 32;
    let smem_bytes = (warps_per_block * vals_per_warp * std::mem::size_of::<E2>()) as u64;

    let grid = MTLSize::new(blocks_per_col as u64, col_groups as u64, 1);
    let block = MTLSize::new(threads_per_block as u64, 1, 1);

    let in_byte_len = inputs.total_len() * std::mem::size_of::<BF>();
    let out_byte_len = outputs.total_len() * std::mem::size_of::<BF>();
    let (in_mtl, in_off) = unsafe {
        wrap_as_metal_buffer(inputs.as_ptr() as *const std::ffi::c_void, in_byte_len, &ctx.device)
    };
    let (out_mtl, out_off) = unsafe {
        wrap_as_metal_buffer(outputs.as_ptr() as *const std::ffi::c_void, out_byte_len, &ctx.device)
    };

    let in_stride = inputs.stride() as u64;
    let out_stride = outputs.stride() as u64;

    let start_stage = 0u32;
    let log_n_u32 = log_n as u32;
    let grid_offset = 0u32;
    let num_z_cols_u32 = num_z_cols as u32;

    let dc = &ctx.device_context;
    let ntt_inv = dc.get_powers_data_w_inv_bitrev_for_ntt();
    let pw = dc.get_powers_data_w();

    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    enc.set_buffer(0, Some(&in_mtl), in_off);
    enc.set_buffer(1, Some(&out_mtl), out_off);
    set_bytes_on_encoder(&enc, &in_stride, 2);
    set_bytes_on_encoder(&enc, &out_stride, 3);
    set_bytes_on_encoder(&enc, &start_stage, 4);
    set_bytes_on_encoder(&enc, &stages_this_launch, 5);
    set_bytes_on_encoder(&enc, &log_n_u32, 6);
    set_bytes_on_encoder(&enc, &num_z_cols_u32, 7);
    // NOTE: N2B kernel has no log_extension_degree/coset_idx at buffer 8;
    // instead buffer 8 is grid_offset (coset is baked into the kernel name).
    set_bytes_on_encoder(&enc, &grid_offset, 8);
    // Inverse twiddle buffers
    enc.set_buffer(9, Some(dc.powers_of_w_inv_fine_bitrev_for_ntt.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &ntt_inv.fine.mask, 10);
    set_bytes_on_encoder(&enc, &ntt_inv.fine.log_count, 11);
    enc.set_buffer(12, Some(dc.powers_of_w_inv_coarse_bitrev_for_ntt.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &ntt_inv.coarse.mask, 13);
    set_bytes_on_encoder(&enc, &ntt_inv.coarse.log_count, 14);
    // Eval-domain powers (needed for coset kernel; unused but must be bound for main domain)
    enc.set_buffer(15, Some(dc.powers_of_w_fine.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &pw.fine.mask, 16);
    set_bytes_on_encoder(&enc, &pw.fine.log_count, 17);
    enc.set_buffer(18, Some(dc.powers_of_w_coarser.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &pw.coarser.mask, 19);
    set_bytes_on_encoder(&enc, &pw.coarser.log_count, 20);
    enc.set_buffer(21, Some(dc.powers_of_w_coarsest.metal_buffer()), 0);
    set_bytes_on_encoder(&enc, &pw.coarsest.mask, 22);
    // Inverse sizes (2^{-log_n} scaling factor)
    enc.set_buffer(23, Some(dc.inv_sizes.metal_buffer()), 0);

    enc.set_threadgroup_memory_length(0, smem_bytes);
    enc.dispatch_thread_groups(grid, block);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

/// Forward NTT for log_n >= 16 using multi-pass block kernels.
///
/// Pass layout (from STAGE_PLANS_B2N):
///   Pass 0 (INITIAL_8_WARP / INITIAL_9_TO_12_BLOCK / INITIAL_7_WARP):
///     Reads `inputs`, writes to `outputs`.
///   Pass k > 0 (NONINITIAL_7_OR_8_BLOCK):
///     Reads `outputs` in-place, writes back to `outputs`.
///
/// All block kernel dispatches use `ab_bitrev_Z_to_natural_coset_evals_noninitial_7_or_8_stages_block`.
fn dispatch_b2n_multi_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    coset_idx: usize,
    ctx: &MetalProverContext,
) {
    let (_n2b_chain, b2n_chain) = get_main_to_coset_launch_chain(log_n);
    let n = 1usize << log_n;
    let num_z_cols = (num_bf_cols + 1) / 2;

    let in_byte_len = inputs.total_len() * std::mem::size_of::<BF>();
    let out_byte_len = outputs.total_len() * std::mem::size_of::<BF>();

    let (in_mtl, in_off) = unsafe {
        wrap_as_metal_buffer(inputs.as_ptr() as *const std::ffi::c_void, in_byte_len, &ctx.device)
    };
    let (out_mtl, out_off) = unsafe {
        wrap_as_metal_buffer(outputs.as_ptr() as *const std::ffi::c_void, out_byte_len, &ctx.device)
    };

    let in_stride  = inputs.stride() as u64;
    let out_stride = outputs.stride() as u64;
    let log_n_u32  = log_n as u32;
    let num_z_u32  = num_z_cols as u32;
    let log_ext_u32 = log_extension_degree as u32;
    let coset_u32   = coset_idx as u32;

    let dc      = &ctx.device_context;
    let ntt_fwd = dc.get_powers_data_w_bitrev_for_ntt();
    let pw      = dc.get_powers_data_w();

    let mut start_stage: u32 = 0;

    for (pass_idx, &(launch_type, stages, n_per_col)) in b2n_chain.iter().enumerate() {
        let stages_u32  = stages as u32;
        let grid_offset = 0u32;
        let blocks_per_col = (n / n_per_col).max(1);
        let col_groups     = ((num_z_cols + COMPLEX_COLS_PER_BLOCK - 1) / COMPLEX_COLS_PER_BLOCK).max(1);

        match launch_type {
            B2N_LAUNCH::INITIAL_8_WARP | B2N_LAUNCH::INITIAL_7_WARP => {
                // Warp-level initial kernel: same as dispatch_b2n_single_stage.
                // Reads from inputs (pass 0), writes to outputs.
                assert_eq!(pass_idx, 0, "INITIAL warp must be first pass");
                let vals_per_warp = if launch_type == B2N_LAUNCH::INITIAL_7_WARP {
                    VALS_PER_WARP_7STAGE
                } else {
                    VALS_PER_WARP_8STAGE
                };
                let kernel_name = if launch_type == B2N_LAUNCH::INITIAL_7_WARP {
                    "ab_bitrev_Z_to_natural_coset_evals_initial_7_stages_warp"
                } else {
                    "ab_bitrev_Z_to_natural_coset_evals_initial_8_stages_warp"
                };
                let pipeline = ctx.get_pipeline(kernel_name);
                let (threads_per_block, bpc, cg) = ntt_warp_dims(n, num_z_cols, vals_per_warp);
                let warps_per_block = threads_per_block / 32;
                let smem_bytes = (warps_per_block * vals_per_warp * std::mem::size_of::<E2>()) as u64;
                let grid  = MTLSize::new(bpc as u64, cg as u64, 1);
                let block = MTLSize::new(threads_per_block as u64, 1, 1);

                let cb  = ctx.command_queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&in_mtl), in_off);
                enc.set_buffer(1, Some(&out_mtl), out_off);
                set_bytes_on_encoder(&enc, &in_stride, 2);
                set_bytes_on_encoder(&enc, &out_stride, 3);
                set_bytes_on_encoder(&enc, &start_stage, 4);
                set_bytes_on_encoder(&enc, &stages_u32, 5);
                set_bytes_on_encoder(&enc, &log_n_u32, 6);
                set_bytes_on_encoder(&enc, &num_z_u32, 7);
                set_bytes_on_encoder(&enc, &log_ext_u32, 8);
                set_bytes_on_encoder(&enc, &coset_u32, 9);
                set_bytes_on_encoder(&enc, &grid_offset, 10);
                enc.set_buffer(11, Some(dc.powers_of_w_fine_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_fwd.fine.mask, 12);
                set_bytes_on_encoder(&enc, &ntt_fwd.fine.log_count, 13);
                enc.set_buffer(14, Some(dc.powers_of_w_coarse_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_fwd.coarse.mask, 15);
                set_bytes_on_encoder(&enc, &ntt_fwd.coarse.log_count, 16);
                enc.set_buffer(17, Some(dc.powers_of_w_fine.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &pw.fine.mask, 18);
                set_bytes_on_encoder(&enc, &pw.fine.log_count, 19);
                enc.set_buffer(20, Some(dc.powers_of_w_coarser.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &pw.coarser.mask, 21);
                set_bytes_on_encoder(&enc, &pw.coarser.log_count, 22);
                enc.set_buffer(23, Some(dc.powers_of_w_coarsest.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &pw.coarsest.mask, 24);
                enc.set_threadgroup_memory_length(0, smem_bytes);
                enc.dispatch_thread_groups(grid, block);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            B2N_LAUNCH::NONINITIAL_7_OR_8_BLOCK => {
                // Block-level non-initial kernel.
                // Reads from outputs (in-place after pass 0), writes back to outputs.
                let pipeline   = ctx.get_pipeline("ab_bitrev_Z_to_natural_coset_evals_noninitial_7_or_8_stages_block");
                let threads_per_block: u64 = 512;
                let smem_bytes: u64 = 4096 * std::mem::size_of::<E2>() as u64; // 32768
                let grid  = MTLSize::new(blocks_per_col as u64, col_groups as u64, 1);
                let block = MTLSize::new(threads_per_block, 1, 1);

                let cb  = ctx.command_queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                // B2N non-initial reads/writes outputs (in-place).
                enc.set_buffer(0, Some(&out_mtl), out_off);
                enc.set_buffer(1, Some(&out_mtl), out_off);
                set_bytes_on_encoder(&enc, &out_stride, 2);
                set_bytes_on_encoder(&enc, &out_stride, 3);
                set_bytes_on_encoder(&enc, &start_stage, 4);
                set_bytes_on_encoder(&enc, &stages_u32, 5);
                set_bytes_on_encoder(&enc, &log_n_u32, 6);
                set_bytes_on_encoder(&enc, &num_z_u32, 7);
                set_bytes_on_encoder(&enc, &grid_offset, 8);
                enc.set_buffer(9,  Some(dc.powers_of_w_fine_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_fwd.fine.mask, 10);
                set_bytes_on_encoder(&enc, &ntt_fwd.fine.log_count, 11);
                enc.set_buffer(12, Some(dc.powers_of_w_coarse_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_fwd.coarse.mask, 13);
                set_bytes_on_encoder(&enc, &ntt_fwd.coarse.log_count, 14);
                enc.set_threadgroup_memory_length(0, smem_bytes);
                enc.dispatch_thread_groups(grid, block);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            B2N_LAUNCH::INITIAL_9_TO_12_BLOCK => {
                unimplemented!(
                    "B2N INITIAL_9_TO_12_BLOCK not yet implemented in Metal (log_n={})",
                    log_n
                );
            }
        }

        start_stage += stages_u32;
    }
}

/// Inverse NTT for log_n >= 16 using multi-pass block kernels.
///
/// Pass layout (from STAGE_PLANS_N2B):
///   Pass 0 (NONFINAL_7_OR_8_BLOCK):
///     Reads `inputs`, writes to `outputs`.
///   Pass k > 0 (NONFINAL_7_OR_8_BLOCK or FINAL_8_WARP / FINAL_9_TO_12_BLOCK):
///     Reads `outputs` in-place, writes back.
fn dispatch_n2b_multi_stage(
    inputs: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    outputs: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    num_bf_cols: usize,
    log_extension_degree: usize,
    _coset_idx: usize,
    ctx: &MetalProverContext,
) {
    let (n2b_chain, _b2n_chain) = get_main_to_coset_launch_chain(log_n);
    let n = 1usize << log_n;
    let num_z_cols = (num_bf_cols + 1) / 2;

    let in_byte_len  = inputs.total_len()  * std::mem::size_of::<BF>();
    let out_byte_len = outputs.total_len() * std::mem::size_of::<BF>();

    let (in_mtl, in_off) = unsafe {
        wrap_as_metal_buffer(inputs.as_ptr() as *const std::ffi::c_void, in_byte_len, &ctx.device)
    };
    let (out_mtl, out_off) = unsafe {
        wrap_as_metal_buffer(outputs.as_ptr() as *const std::ffi::c_void, out_byte_len, &ctx.device)
    };

    let in_stride   = inputs.stride() as u64;
    let out_stride  = outputs.stride() as u64;
    let log_n_u32   = log_n as u32;
    let num_z_u32   = num_z_cols as u32;

    let dc      = &ctx.device_context;
    let ntt_inv = dc.get_powers_data_w_inv_bitrev_for_ntt();
    let pw      = dc.get_powers_data_w();

    let mut start_stage: u32 = 0;

    for (pass_idx, &(launch_type, stages, n_per_col)) in n2b_chain.iter().enumerate() {
        let stages_u32   = stages as u32;
        let grid_offset  = 0u32;
        let blocks_per_col = (n / n_per_col).max(1);
        let col_groups     = ((num_z_cols + COMPLEX_COLS_PER_BLOCK - 1) / COMPLEX_COLS_PER_BLOCK).max(1);

        match launch_type {
            N2B_LAUNCH::FINAL_8_WARP | N2B_LAUNCH::FINAL_7_WARP => {
                // Warp-level final kernel.
                // For pass 0: reads from inputs, writes to outputs.
                // For pass > 0: reads/writes outputs in-place.
                let kernel_name = if log_extension_degree == 0 {
                    "ab_main_domain_evals_to_Z_final_8_stages_warp"
                } else {
                    "ab_coset_evals_to_Z_final_8_stages_warp"
                };
                let vals_per_warp = VALS_PER_WARP_8STAGE;
                let pipeline = ctx.get_pipeline(kernel_name);
                let (threads_per_block, bpc, cg) = ntt_warp_dims(n, num_z_cols, vals_per_warp);
                let warps_per_block = threads_per_block / 32;
                let smem_bytes = (warps_per_block * vals_per_warp * std::mem::size_of::<E2>()) as u64;
                let grid  = MTLSize::new(bpc as u64, cg as u64, 1);
                let block = MTLSize::new(threads_per_block as u64, 1, 1);

                // Decide src_mtl/src_off: first pass reads inputs, subsequent read outputs.
                let (src_mtl_ref, src_off_val, src_stride) = if pass_idx == 0 {
                    (&in_mtl, in_off, in_stride)
                } else {
                    (&out_mtl, out_off, out_stride)
                };

                let cb  = ctx.command_queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(src_mtl_ref), src_off_val);
                enc.set_buffer(1, Some(&out_mtl), out_off);
                set_bytes_on_encoder(&enc, &src_stride, 2);
                set_bytes_on_encoder(&enc, &out_stride, 3);
                set_bytes_on_encoder(&enc, &start_stage, 4);
                set_bytes_on_encoder(&enc, &stages_u32, 5);
                set_bytes_on_encoder(&enc, &log_n_u32, 6);
                set_bytes_on_encoder(&enc, &num_z_u32, 7);
                set_bytes_on_encoder(&enc, &grid_offset, 8);
                enc.set_buffer(9,  Some(dc.powers_of_w_inv_fine_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_inv.fine.mask, 10);
                set_bytes_on_encoder(&enc, &ntt_inv.fine.log_count, 11);
                enc.set_buffer(12, Some(dc.powers_of_w_inv_coarse_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_inv.coarse.mask, 13);
                set_bytes_on_encoder(&enc, &ntt_inv.coarse.log_count, 14);
                enc.set_buffer(15, Some(dc.powers_of_w_fine.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &pw.fine.mask, 16);
                set_bytes_on_encoder(&enc, &pw.fine.log_count, 17);
                enc.set_buffer(18, Some(dc.powers_of_w_coarser.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &pw.coarser.mask, 19);
                set_bytes_on_encoder(&enc, &pw.coarser.log_count, 20);
                enc.set_buffer(21, Some(dc.powers_of_w_coarsest.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &pw.coarsest.mask, 22);
                enc.set_buffer(23, Some(dc.inv_sizes.metal_buffer()), 0);
                enc.set_threadgroup_memory_length(0, smem_bytes);
                enc.dispatch_thread_groups(grid, block);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            N2B_LAUNCH::NONFINAL_7_OR_8_BLOCK => {
                // Block-level non-final kernel.
                // First pass reads inputs → outputs; subsequent passes in-place on outputs.
                let pipeline   = ctx.get_pipeline("ab_evals_to_Z_nonfinal_7_or_8_stages_block");
                let threads_per_block: u64 = 512;
                let smem_bytes: u64 = 4096 * std::mem::size_of::<E2>() as u64;
                let grid  = MTLSize::new(blocks_per_col as u64, col_groups as u64, 1);
                let block = MTLSize::new(threads_per_block, 1, 1);

                let (src_mtl_ref, src_off_val, src_stride_val) = if pass_idx == 0 {
                    (&in_mtl, in_off, in_stride)
                } else {
                    (&out_mtl, out_off, out_stride)
                };

                let cb  = ctx.command_queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(src_mtl_ref), src_off_val);
                enc.set_buffer(1, Some(&out_mtl), out_off);
                set_bytes_on_encoder(&enc, &src_stride_val, 2);
                set_bytes_on_encoder(&enc, &out_stride, 3);
                set_bytes_on_encoder(&enc, &start_stage, 4);
                set_bytes_on_encoder(&enc, &stages_u32, 5);
                set_bytes_on_encoder(&enc, &log_n_u32, 6);
                set_bytes_on_encoder(&enc, &num_z_u32, 7);
                set_bytes_on_encoder(&enc, &grid_offset, 8);
                enc.set_buffer(9,  Some(dc.powers_of_w_inv_fine_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_inv.fine.mask, 10);
                set_bytes_on_encoder(&enc, &ntt_inv.fine.log_count, 11);
                enc.set_buffer(12, Some(dc.powers_of_w_inv_coarse_bitrev_for_ntt.metal_buffer()), 0);
                set_bytes_on_encoder(&enc, &ntt_inv.coarse.mask, 13);
                set_bytes_on_encoder(&enc, &ntt_inv.coarse.log_count, 14);
                enc.set_threadgroup_memory_length(0, smem_bytes);
                enc.dispatch_thread_groups(grid, block);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }

            N2B_LAUNCH::FINAL_9_TO_12_BLOCK => {
                unimplemented!(
                    "N2B FINAL_9_TO_12_BLOCK not yet implemented in Metal (log_n={})",
                    log_n
                );
            }
        }

        start_stage += stages_u32;
    }
}

// ---------------------------------------------------------------------------
// Encoder helpers
// ---------------------------------------------------------------------------

fn set_bytes_on_encoder<T: Sized>(encoder: &metal::ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
}
