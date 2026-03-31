#include <metal_stdlib>
using namespace metal;

#include "ntt.metal"

using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;
using namespace airbender::ntt;

namespace airbender {
namespace ntt {

// Inverse NTT: natural evals -> bitrev Z
// This is a DIT (Decimation In Time) butterfly NTT.

template <unsigned LOG_VALS_PER_THREAD, bool evals_are_coset, bool evals_are_compressed>
DEVICE_FORCEINLINE void evals_to_Z_final_stages_warp(
    device const bf *gmem_in_ptr, const size_t in_stride,
    device bf *gmem_out_ptr, const size_t out_stride,
    const unsigned log_n, const unsigned num_Z_cols,
    const unsigned grid_offset,
    // Inverse twiddle data
    const device e2f *twiddle_fine_values, const unsigned twiddle_fine_mask, const unsigned twiddle_fine_log_count,
    const device e2f *twiddle_coarse_values, const unsigned twiddle_coarse_mask, const unsigned twiddle_coarse_log_count,
    // Powers data for LDE unscaling
    const device e2f *powers_fine_values, const unsigned powers_fine_mask, const unsigned powers_fine_log_count,
    const device e2f *powers_coarser_values, const unsigned powers_coarser_mask, const unsigned powers_coarser_log_count,
    const device e2f *powers_coarsest_values, const unsigned powers_coarsest_mask,
    // Inv sizes
    const device bf *inv_sizes,
    // Shared memory and thread info
    threadgroup e2f *smem,
    const unsigned lane_id, const unsigned warp_id,
    const unsigned effective_block_idx_x, const unsigned blockIdx_y) {

  constexpr unsigned COL_PAIRS_PER_BLOCK = COLS_PER_BLOCK<e2f>::VAL;
  constexpr unsigned VALS_PER_THREAD = 1u << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned VALS_PER_BLOCK = VALS_PER_WARP * 4; // 4 warps

  const unsigned gmem_offset = VALS_PER_BLOCK * effective_block_idx_x + VALS_PER_WARP * warp_id;

  // Build powers_data_3_layer on thread
  powers_data_3_layer powers_w;
  powers_w.fine.values = powers_fine_values;
  powers_w.fine.mask = powers_fine_mask;
  powers_w.fine.log_count = powers_fine_log_count;
  powers_w.coarser.values = powers_coarser_values;
  powers_w.coarser.mask = powers_coarser_mask;
  powers_w.coarser.log_count = powers_coarser_log_count;
  powers_w.coarsest.values = powers_coarsest_values;
  powers_w.coarsest.mask = powers_coarsest_mask;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  // Load inverse twiddles cooperatively
  {
    threadgroup e2f *tc = twiddle_cache;
    unsigned num_twiddles = VALS_PER_WARP >> 1;
    unsigned exchg_region_offset = gmem_offset >> 1;
    for (unsigned stage = 0; stage < LOG_VALS_PER_THREAD; stage++) {
      for (unsigned i = lane_id; i < num_twiddles; i += 32)
        tc[i] = get_twiddle<true>(twiddle_fine_values, twiddle_fine_mask, twiddle_fine_log_count,
                                   twiddle_coarse_values, twiddle_coarse_mask, twiddle_coarse_log_count, i + exchg_region_offset);
      tc += num_twiddles;
      num_twiddles >>= 1;
      exchg_region_offset >>= 1;
    }
    if (lane_id > 0) {
      const unsigned lz = clz(lane_id);
      const unsigned stage_offset = 5 - (32 - lz);
      const unsigned mask = (1u << (32 - lz)) - 1;
      unsigned ero = exchg_region_offset >> stage_offset;
      tc[lane_id ^ 31] = get_twiddle<true>(twiddle_fine_values, twiddle_fine_mask, twiddle_fine_log_count,
                                             twiddle_coarse_values, twiddle_coarse_mask, twiddle_coarse_log_count, (lane_id ^ mask) + ero);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  const unsigned bound = min(COL_PAIRS_PER_BLOCK, num_Z_cols - COL_PAIRS_PER_BLOCK * blockIdx_y);
  for (unsigned ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
    const unsigned col_offset = COL_PAIRS_PER_BLOCK * blockIdx_y + ntt_idx;
    e2f vals[VALS_PER_THREAD];

    // Load: different pattern from forward NTT
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const unsigned row0 = gmem_offset + 64 * i + lane_id;
      const unsigned row1 = gmem_offset + 64 * i + lane_id + 32;
      vals[2 * i] = e2f{*(gmem_in_ptr + row0 + col_offset * 2 * in_stride),
                         *(gmem_in_ptr + row0 + (col_offset * 2 + 1) * in_stride)};
      vals[2 * i + 1] = e2f{*(gmem_in_ptr + row1 + col_offset * 2 * in_stride),
                              *(gmem_in_ptr + row1 + (col_offset * 2 + 1) * in_stride)};
    }

    // Thread-local DIT stages (reverse of DIF)
    threadgroup e2f *twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
    unsigned num_twiddles_this_stage = 1;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
      for (unsigned j = 0; j < (1u << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = twiddles_this_stage[j];
        for (unsigned k = 0; k < half_exchg_tile_sz; k++)
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
      }
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    // Warp-level DIT stages
    unsigned lane_mask = 16;
    for (unsigned stage = 0, s = 5; stage < 6; stage++, s--) {
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> s];
        exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
      }
      if (stage < 5) {
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++)
          shfl_xor_e2f(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    // Multiply by inverse size
    const bf inv_size = inv_sizes[log_n];
    for (unsigned i = 0; i < VALS_PER_THREAD; i++)
      vals[i] = e2f::mul(vals[i], inv_size);

    // Apply coset unscaling if needed
    if (evals_are_coset) {
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const unsigned mem_idx = gmem_offset + 64 * i + 2 * lane_id;
        const unsigned idx0 = bitrev(mem_idx, log_n);
        const unsigned idx1 = bitrev(mem_idx + 1, log_n);
        if (evals_are_compressed) {
          vals[2 * i] = lde_scale_and_shift(powers_w, vals[2 * i], idx0, 1, 1, log_n, true);
          vals[2 * i + 1] = lde_scale_and_shift(powers_w, vals[2 * i + 1], idx1, 1, 1, log_n, true);
        } else {
          vals[2 * i] = lde_scale(powers_w, vals[2 * i], idx0, 1, 1, log_n, true);
          vals[2 * i + 1] = lde_scale(powers_w, vals[2 * i + 1], idx1, 1, 1, log_n, true);
        }
      }
    }

    // Store: two-for-one pattern
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const unsigned row = gmem_offset + 64 * i + 2 * lane_id;
      device bf *p = gmem_out_ptr + row;
      *(p + col_offset * 2 * out_stride) = vals[2 * i][0];
      *(p + (col_offset * 2 + 1) * out_stride) = vals[2 * i][1];
      *(p + 1 + col_offset * 2 * out_stride) = vals[2 * i + 1][0];
      *(p + 1 + (col_offset * 2 + 1) * out_stride) = vals[2 * i + 1][1];
    }
  }
}

// Main domain 8-stage final warp kernel
[[max_total_threads_per_threadgroup(128)]]
kernel void ab_main_domain_evals_to_Z_final_8_stages_warp(
    device const bf *gmem_in [[buffer(0)]],
    device bf *gmem_out [[buffer(1)]],
    constant size_t &in_stride [[buffer(2)]],
    constant size_t &out_stride [[buffer(3)]],
    constant unsigned &start_stage [[buffer(4)]],
    constant unsigned &stages_this_launch [[buffer(5)]],
    constant unsigned &log_n [[buffer(6)]],
    constant unsigned &num_Z_cols [[buffer(7)]],
    constant unsigned &grid_offset [[buffer(8)]],
    device const e2f *twiddle_fine [[buffer(9)]],
    constant unsigned &twiddle_fine_mask [[buffer(10)]],
    constant unsigned &twiddle_fine_log [[buffer(11)]],
    device const e2f *twiddle_coarse [[buffer(12)]],
    constant unsigned &twiddle_coarse_mask [[buffer(13)]],
    constant unsigned &twiddle_coarse_log [[buffer(14)]],
    device const e2f *powers_fine [[buffer(15)]],
    constant unsigned &powers_fine_mask [[buffer(16)]],
    constant unsigned &powers_fine_log [[buffer(17)]],
    device const e2f *powers_coarser [[buffer(18)]],
    constant unsigned &powers_coarser_mask [[buffer(19)]],
    constant unsigned &powers_coarser_log [[buffer(20)]],
    device const e2f *powers_coarsest [[buffer(21)]],
    constant unsigned &powers_coarsest_mask [[buffer(22)]],
    device const bf *inv_sizes [[buffer(23)]],
    threadgroup e2f *smem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 gid_2d [[threadgroup_position_in_grid]]) {

  const unsigned lane_id = tid & 31;
  const unsigned warp_id = tid >> 5;
  evals_to_Z_final_stages_warp<3, false, false>(
      gmem_in, in_stride, gmem_out, out_stride,
      log_n, num_Z_cols, grid_offset,
      twiddle_fine, twiddle_fine_mask, twiddle_fine_log,
      twiddle_coarse, twiddle_coarse_mask, twiddle_coarse_log,
      powers_fine, powers_fine_mask, powers_fine_log,
      powers_coarser, powers_coarser_mask, powers_coarser_log,
      powers_coarsest, powers_coarsest_mask,
      inv_sizes, smem, lane_id, warp_id, gid_2d.x + grid_offset, gid_2d.y);
}

// Coset 8-stage final warp kernel
[[max_total_threads_per_threadgroup(128)]]
kernel void ab_coset_evals_to_Z_final_8_stages_warp(
    device const bf *gmem_in [[buffer(0)]],
    device bf *gmem_out [[buffer(1)]],
    constant size_t &in_stride [[buffer(2)]],
    constant size_t &out_stride [[buffer(3)]],
    constant unsigned &start_stage [[buffer(4)]],
    constant unsigned &stages_this_launch [[buffer(5)]],
    constant unsigned &log_n [[buffer(6)]],
    constant unsigned &num_Z_cols [[buffer(7)]],
    constant unsigned &grid_offset [[buffer(8)]],
    device const e2f *twiddle_fine [[buffer(9)]],
    constant unsigned &twiddle_fine_mask [[buffer(10)]],
    constant unsigned &twiddle_fine_log [[buffer(11)]],
    device const e2f *twiddle_coarse [[buffer(12)]],
    constant unsigned &twiddle_coarse_mask [[buffer(13)]],
    constant unsigned &twiddle_coarse_log [[buffer(14)]],
    device const e2f *powers_fine [[buffer(15)]],
    constant unsigned &powers_fine_mask [[buffer(16)]],
    constant unsigned &powers_fine_log [[buffer(17)]],
    device const e2f *powers_coarser [[buffer(18)]],
    constant unsigned &powers_coarser_mask [[buffer(19)]],
    constant unsigned &powers_coarser_log [[buffer(20)]],
    device const e2f *powers_coarsest [[buffer(21)]],
    constant unsigned &powers_coarsest_mask [[buffer(22)]],
    device const bf *inv_sizes [[buffer(23)]],
    threadgroup e2f *smem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 gid_2d [[threadgroup_position_in_grid]]) {

  const unsigned lane_id = tid & 31;
  const unsigned warp_id = tid >> 5;
  evals_to_Z_final_stages_warp<3, true, false>(
      gmem_in, in_stride, gmem_out, out_stride,
      log_n, num_Z_cols, grid_offset,
      twiddle_fine, twiddle_fine_mask, twiddle_fine_log,
      twiddle_coarse, twiddle_coarse_mask, twiddle_coarse_log,
      powers_fine, powers_fine_mask, powers_fine_log,
      powers_coarser, powers_coarser_mask, powers_coarser_log,
      powers_coarsest, powers_coarsest_mask,
      inv_sizes, smem, lane_id, warp_id, gid_2d.x + grid_offset, gid_2d.y);
}

} // namespace ntt
} // namespace airbender
