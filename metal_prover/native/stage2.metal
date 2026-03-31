#include <metal_stdlib>
using namespace metal;

#include "arg_utils.metal"
#include "field.metal"
#include "memory.metal"
#include "vectorized.metal"

using namespace airbender::arg_utils;
using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;

namespace airbender {
namespace stage2 {

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

[[max_total_threads_per_threadgroup(128)]]
kernel void ab_zero_stage_2_last_row_kernel(device bf *stage_2_bf_cols [[buffer(0)]],
                                             constant size_t &bf_stride [[buffer(1)]],
                                             device bf *stage_2_e4_cols [[buffer(2)]],
                                             constant size_t &e4_stride [[buffer(3)]],
                                             constant unsigned &num_stage_2_bf_cols [[buffer(4)]],
                                             constant unsigned &num_stage_2_e4_cols [[buffer(5)]],
                                             constant unsigned &log_n [[buffer(6)]],
                                             uint gid [[thread_position_in_grid]]) {
  const unsigned n = 1u << log_n;

  if (gid < num_stage_2_bf_cols) {
    stage_2_bf_cols[(n - 1) + gid * bf_stride] = bf::zero();
  }

  if (gid < num_stage_2_e4_cols) {
    // Zero all 4 base field components of e4 at column gid, row n-1
    for (unsigned c = 0; c < 4; c++)
      stage_2_e4_cols[(n - 1) + (gid * 4 + c) * e4_stride] = bf::zero();
  }
}

// Range check aggregated entry inverses and multiplicities argument
[[max_total_threads_per_threadgroup(128)]]
kernel void ab_range_check_aggregated_entry_invs_and_multiplicities_arg_kernel(
    device const LookupChallenges *challenges [[buffer(0)]],
    device const bf *witness_cols [[buffer(1)]],
    constant size_t &witness_stride [[buffer(2)]],
    device const bf *setup_cols [[buffer(3)]],
    constant size_t &setup_stride [[buffer(4)]],
    device bf *stage_2_e4_cols [[buffer(5)]],
    constant size_t &e4_stride [[buffer(6)]],
    device e4 *aggregated_entry_invs [[buffer(7)]],
    constant unsigned &start_col_in_setup [[buffer(8)]],
    constant unsigned &multiplicities_src_cols_start [[buffer(9)]],
    constant unsigned &multiplicities_dst_cols_start [[buffer(10)]],
    constant unsigned &num_multiplicities_cols [[buffer(11)]],
    constant unsigned &num_table_rows_tail [[buffer(12)]],
    constant unsigned &log_n [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  const unsigned n = 1u << log_n;
  if (gid >= n - 1)
    return;

  const e4 gamma = challenges->gamma;

  for (unsigned i = 0; i < num_multiplicities_cols; i++) {
    if (i == num_multiplicities_cols - 1 && gid >= num_table_rows_tail) {
      // Write e4::zero() to stage_2_e4_cols at (multiplicities_dst_cols_start + i), row gid
      for (unsigned c = 0; c < 4; c++)
        stage_2_e4_cols[gid + (multiplicities_dst_cols_start + i) * 4 * e4_stride + c * e4_stride] = bf::zero();
      return;
    }

    // For range check (width=1), the value is just the row index
    bf val = bf{gid};
    e4 denom = e4::add(gamma, val);
    const e4 denom_inv = e4::inv(denom);

    // Read multiplicity
    const bf multiplicity = witness_cols[gid + (multiplicities_src_cols_start + i) * witness_stride];

    // Write m * denom_inv to stage_2_e4_cols
    const e4 result = e4::mul(denom_inv, multiplicity);
    // Write e4 as 4 bf components
    for (unsigned c = 0; c < 4; c++)
      stage_2_e4_cols[gid + ((multiplicities_dst_cols_start + i) * 4 + c) * e4_stride] = result.base_coefficient_from_flat_idx(c);

    // Write denom_inv to aggregated_entry_invs
    aggregated_entry_invs[gid + i * (n - 1)] = denom_inv;
  }
}

} // namespace stage2
} // namespace airbender
