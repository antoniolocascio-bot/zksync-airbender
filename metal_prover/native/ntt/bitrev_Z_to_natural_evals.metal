#include <metal_stdlib>
using namespace metal;

#include "ntt.metal"

using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;
using namespace airbender::ntt;

namespace airbender {
namespace ntt {

// Forward NTT: bitrev_Z -> natural evals
// This is a DIF (Decimation In Frequency) butterfly NTT.

// Initial stages (warp-level, 7 or 8 stages)
// LOG_VALS_PER_THREAD=3 -> 8 stages, LOG_VALS_PER_THREAD=2 -> 7 stages
template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void bitrev_Z_to_natural_coset_evals_initial_stages_warp(
    device const bf *gmem_in_ptr, const size_t in_stride,
    device bf *gmem_out_ptr, const size_t out_stride,
    const unsigned start_stage, const unsigned stages_this_launch,
    const unsigned log_n, const unsigned num_Z_cols,
    const unsigned log_extension_degree, const unsigned coset_idx,
    const unsigned grid_offset,
    // Twiddle factor data (passed as buffers instead of constant memory)
    const device e2f *twiddle_fine_values, const unsigned twiddle_fine_mask, const unsigned twiddle_fine_log_count,
    const device e2f *twiddle_coarse_values, const unsigned twiddle_coarse_mask, const unsigned twiddle_coarse_log_count,
    // Powers data for LDE scale
    const device e2f *powers_fine_values, const unsigned powers_fine_mask, const unsigned powers_fine_log_count,
    const device e2f *powers_coarser_values, const unsigned powers_coarser_mask, const unsigned powers_coarser_log_count,
    const device e2f *powers_coarsest_values, const unsigned powers_coarsest_mask,
    // Thread info
    threadgroup e2f *smem,
    const unsigned lane_id, const unsigned warp_id,
    const unsigned effective_block_idx_x, const unsigned blockIdx_y) {

  constexpr unsigned COL_PAIRS_PER_BLOCK = COLS_PER_BLOCK<e2f>::VAL;
  constexpr unsigned VALS_PER_THREAD = 1u << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;

  const unsigned gmem_offset = VALS_PER_WARP * 4 * effective_block_idx_x + VALS_PER_WARP * warp_id;

  // Build powers_data_3_layer on thread for LDE
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

  // Load twiddles cooperatively
  unsigned num_twiddles = VALS_PER_WARP >> 1;
  unsigned exchg_region_offset = gmem_offset >> 1;
  for (unsigned stage = 0; stage < LOG_VALS_PER_THREAD; stage++) {
    for (unsigned i = lane_id; i < num_twiddles; i += 32)
      twiddle_cache[i] = get_twiddle<false>(twiddle_fine_values, twiddle_fine_mask, twiddle_fine_log_count,
                                             twiddle_coarse_values, twiddle_coarse_mask, twiddle_coarse_log_count, i + exchg_region_offset);
    twiddle_cache += num_twiddles;
    num_twiddles >>= 1;
    exchg_region_offset >>= 1;
  }
  // Load final 31 twiddles
  if (lane_id > 0) {
    const unsigned lz = clz(lane_id);
    const unsigned stage_offset = 5 - (32 - lz);
    const unsigned mask = (1u << (32 - lz)) - 1;
    unsigned ero = exchg_region_offset >> stage_offset;
    twiddle_cache[lane_id ^ 31] = get_twiddle<false>(twiddle_fine_values, twiddle_fine_mask, twiddle_fine_log_count,
                                                      twiddle_coarse_values, twiddle_coarse_mask, twiddle_coarse_log_count, (lane_id ^ mask) + ero);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  // Reset twiddle_cache pointer
  twiddle_cache = smem + VALS_PER_WARP * warp_id;

  const unsigned bound = min(COL_PAIRS_PER_BLOCK, num_Z_cols - COL_PAIRS_PER_BLOCK * blockIdx_y);
  for (unsigned ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
    const unsigned col_offset = (COL_PAIRS_PER_BLOCK * blockIdx_y + ntt_idx);
    e2f vals[VALS_PER_THREAD];

    // Load values (two-for-one pattern)
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const unsigned row = gmem_offset + 64 * i + 2 * lane_id;
      const device bf *p = gmem_in_ptr + row;
      const bf c0_0 = *(p + col_offset * 2 * in_stride);
      const bf c0_1 = *(p + 1 + col_offset * 2 * in_stride);
      const bf c1_0 = *(p + (col_offset * 2 + 1) * in_stride);
      const bf c1_1 = *(p + 1 + (col_offset * 2 + 1) * in_stride);
      vals[2 * i] = e2f{c0_0, c1_0};
      vals[2 * i + 1] = e2f{c0_1, c1_1};
    }

    // Apply LDE scale and shift
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const unsigned mem_idx = gmem_offset + 64 * i + 2 * lane_id;
      const unsigned idx0 = bitrev(mem_idx, log_n);
      const unsigned idx1 = bitrev(mem_idx + 1, log_n);
      vals[2 * i] = lde_scale_and_shift(powers_w, vals[2 * i], idx0, log_extension_degree, coset_idx, log_n);
      vals[2 * i + 1] = lde_scale_and_shift(powers_w, vals[2 * i + 1], idx1, log_extension_degree, coset_idx, log_n);
    }

    // Warp-level butterfly stages
    unsigned lane_mask = 1;
    threadgroup e2f *twiddles_this_stage = twiddle_cache;
    unsigned num_twiddles_this_stage = VALS_PER_WARP >> 1;
    for (unsigned stage = 0; stage < 6; stage++) {
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> stage];
        exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
        if (stage < 5)
          shfl_xor_e2f(vals, i, lane_id, lane_mask);
      }
      lane_mask <<= 1;
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

    // Thread-local stages
    for (unsigned i = 1; i < LOG_VALS_PER_THREAD; i++) {
      for (unsigned j = 0; j < (PAIRS_PER_THREAD >> i); j++) {
        const unsigned exchg_tile_sz = 2u << i;
        const unsigned half_exchg_tile_sz = 1u << i;
        const auto twiddle = twiddles_this_stage[j];
        for (unsigned k = 0; k < half_exchg_tile_sz; k++)
          exchg_dif(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
      }
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

    // Store results
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const unsigned row = gmem_offset + 64 * i + lane_id;
      device bf *p = gmem_out_ptr + row;
      *(p + col_offset * 2 * out_stride) = vals[2 * i][0];
      *(p + (col_offset * 2 + 1) * out_stride) = vals[2 * i][1];
      const unsigned row2 = gmem_offset + 64 * i + lane_id + 32;
      device bf *p2 = gmem_out_ptr + row2;
      *(p2 + col_offset * 2 * out_stride) = vals[2 * i + 1][0];
      *(p2 + (col_offset * 2 + 1) * out_stride) = vals[2 * i + 1][1];
    }
  }
}

// 8-stage warp kernel
[[max_total_threads_per_threadgroup(128)]]
kernel void ab_bitrev_Z_to_natural_coset_evals_initial_8_stages_warp(
    device const bf *gmem_in [[buffer(0)]],
    device bf *gmem_out [[buffer(1)]],
    constant size_t &in_stride [[buffer(2)]],
    constant size_t &out_stride [[buffer(3)]],
    constant unsigned &start_stage [[buffer(4)]],
    constant unsigned &stages_this_launch [[buffer(5)]],
    constant unsigned &log_n [[buffer(6)]],
    constant unsigned &num_Z_cols [[buffer(7)]],
    constant unsigned &log_extension_degree [[buffer(8)]],
    constant unsigned &coset_idx [[buffer(9)]],
    constant unsigned &grid_offset [[buffer(10)]],
    // Twiddle buffers
    device const e2f *twiddle_fine [[buffer(11)]],
    constant unsigned &twiddle_fine_mask [[buffer(12)]],
    constant unsigned &twiddle_fine_log [[buffer(13)]],
    device const e2f *twiddle_coarse [[buffer(14)]],
    constant unsigned &twiddle_coarse_mask [[buffer(15)]],
    constant unsigned &twiddle_coarse_log [[buffer(16)]],
    // Powers buffers
    device const e2f *powers_fine [[buffer(17)]],
    constant unsigned &powers_fine_mask [[buffer(18)]],
    constant unsigned &powers_fine_log [[buffer(19)]],
    device const e2f *powers_coarser [[buffer(20)]],
    constant unsigned &powers_coarser_mask [[buffer(21)]],
    constant unsigned &powers_coarser_log [[buffer(22)]],
    device const e2f *powers_coarsest [[buffer(23)]],
    constant unsigned &powers_coarsest_mask [[buffer(24)]],
    // Threadgroup memory and thread indices
    threadgroup e2f *smem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 gid_2d [[threadgroup_position_in_grid]]) {

  const unsigned lane_id = tid & 31;
  const unsigned warp_id = tid >> 5;
  bitrev_Z_to_natural_coset_evals_initial_stages_warp<3>(
      gmem_in, in_stride, gmem_out, out_stride,
      start_stage, stages_this_launch, log_n, num_Z_cols,
      log_extension_degree, coset_idx, grid_offset,
      twiddle_fine, twiddle_fine_mask, twiddle_fine_log,
      twiddle_coarse, twiddle_coarse_mask, twiddle_coarse_log,
      powers_fine, powers_fine_mask, powers_fine_log,
      powers_coarser, powers_coarser_mask, powers_coarser_log,
      powers_coarsest, powers_coarsest_mask,
      smem, lane_id, warp_id, gid_2d.x + grid_offset, gid_2d.y);
}

// 7-stage warp kernel
[[max_total_threads_per_threadgroup(128)]]
kernel void ab_bitrev_Z_to_natural_coset_evals_initial_7_stages_warp(
    device const bf *gmem_in [[buffer(0)]],
    device bf *gmem_out [[buffer(1)]],
    constant size_t &in_stride [[buffer(2)]],
    constant size_t &out_stride [[buffer(3)]],
    constant unsigned &start_stage [[buffer(4)]],
    constant unsigned &stages_this_launch [[buffer(5)]],
    constant unsigned &log_n [[buffer(6)]],
    constant unsigned &num_Z_cols [[buffer(7)]],
    constant unsigned &log_extension_degree [[buffer(8)]],
    constant unsigned &coset_idx [[buffer(9)]],
    constant unsigned &grid_offset [[buffer(10)]],
    device const e2f *twiddle_fine [[buffer(11)]],
    constant unsigned &twiddle_fine_mask [[buffer(12)]],
    constant unsigned &twiddle_fine_log [[buffer(13)]],
    device const e2f *twiddle_coarse [[buffer(14)]],
    constant unsigned &twiddle_coarse_mask [[buffer(15)]],
    constant unsigned &twiddle_coarse_log [[buffer(16)]],
    device const e2f *powers_fine [[buffer(17)]],
    constant unsigned &powers_fine_mask [[buffer(18)]],
    constant unsigned &powers_fine_log [[buffer(19)]],
    device const e2f *powers_coarser [[buffer(20)]],
    constant unsigned &powers_coarser_mask [[buffer(21)]],
    constant unsigned &powers_coarser_log [[buffer(22)]],
    device const e2f *powers_coarsest [[buffer(23)]],
    constant unsigned &powers_coarsest_mask [[buffer(24)]],
    threadgroup e2f *smem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 gid_2d [[threadgroup_position_in_grid]]) {

  const unsigned lane_id = tid & 31;
  const unsigned warp_id = tid >> 5;
  bitrev_Z_to_natural_coset_evals_initial_stages_warp<2>(
      gmem_in, in_stride, gmem_out, out_stride,
      start_stage, stages_this_launch, log_n, num_Z_cols,
      log_extension_degree, coset_idx, grid_offset,
      twiddle_fine, twiddle_fine_mask, twiddle_fine_log,
      twiddle_coarse, twiddle_coarse_mask, twiddle_coarse_log,
      powers_fine, powers_fine_mask, powers_fine_log,
      powers_coarser, powers_coarser_mask, powers_coarser_log,
      powers_coarsest, powers_coarsest_mask,
      smem, lane_id, warp_id, gid_2d.x + grid_offset, gid_2d.y);
}

} // namespace ntt
} // namespace airbender
