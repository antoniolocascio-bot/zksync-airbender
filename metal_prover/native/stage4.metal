#include <metal_stdlib>
using namespace metal;

#include "arg_utils.metal"
#include "context.metal"
#include "field.metal"
#include "memory.metal"
#include "vectorized.metal"

using namespace airbender::arg_utils;
using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;

namespace airbender {
namespace stage4 {

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

constexpr unsigned MAX_MEMORY_COLS = 256;
constexpr unsigned DOES_NOT_NEED_Z_OMEGA = 0xFFFFFFFF;

struct ColIdxsToChallengeIdxsMap {
  unsigned map[MAX_MEMORY_COLS];
};

struct ChallengesTimesEvalsSums {
  e4 at_z_sum_neg;
  e4 at_z_omega_sum_neg;
};

// DEEP denominator at z kernel
kernel void ab_deep_denom_at_z_kernel(device e4 *denom_at_z [[buffer(0)]],
                                       device const e4 *z_ref [[buffer(1)]],
                                       constant unsigned &log_n [[buffer(2)]],
                                       constant bool &bit_reversed [[buffer(3)]],
                                       // Powers data for get_power_of_w
                                       device const e2 *fine_values [[buffer(4)]],
                                       constant unsigned &fine_mask [[buffer(5)]],
                                       constant unsigned &fine_log_count [[buffer(6)]],
                                       device const e2 *coarser_values [[buffer(7)]],
                                       constant unsigned &coarser_mask [[buffer(8)]],
                                       constant unsigned &coarser_log_count [[buffer(9)]],
                                       device const e2 *coarsest_values [[buffer(10)]],
                                       constant unsigned &coarsest_mask [[buffer(11)]],
                                       uint gid [[thread_position_in_grid]],
                                       uint grid_size [[threads_per_grid]])
  [[max_total_threads_per_threadgroup(128)]] {
  constexpr unsigned INV_BATCH = 3; // InvBatch<e4>::INV_BATCH

  const unsigned n = 1u << log_n;
  if (gid >= n)
    return;

  // Build powers data on thread
  powers_data_3_layer powers_w;
  powers_w.fine.values = fine_values;
  powers_w.fine.mask = fine_mask;
  powers_w.fine.log_count = fine_log_count;
  powers_w.coarser.values = coarser_values;
  powers_w.coarser.mask = coarser_mask;
  powers_w.coarser.log_count = coarser_log_count;
  powers_w.coarsest.values = coarsest_values;
  powers_w.coarsest.mask = coarsest_mask;

  const e4 z = *z_ref;
  const unsigned log_shift = CIRCLE_GROUP_LOG_ORDER - log_n;

  e4 per_elem_factor_invs[INV_BATCH];
  int runtime_batch_size = 0;

  for (unsigned i = 0, g = gid; i < INV_BATCH; i++, g += grid_size)
    if (g < n) {
      const unsigned k = (bit_reversed ? reverse_bits(g) >> (32 - log_n) : g) << log_shift;
      const auto x = get_power_of_w(powers_w, k, false);
      per_elem_factor_invs[i] = e4::sub(x, z);
      runtime_batch_size++;
    }

  // Batch inversion
  e4 per_elem_factors[INV_BATCH];
  {
    e4 running_prod = e4::one();
    for (int i = 0; i < INV_BATCH; i++)
      if (i < runtime_batch_size) {
        per_elem_factors[i] = running_prod;
        running_prod = e4::mul(running_prod, per_elem_factor_invs[i]);
      }
    e4 inv = e4::inv(running_prod);
    for (int i = INV_BATCH - 1; i >= 0; i--)
      if (i < runtime_batch_size) {
        per_elem_factors[i] = e4::mul(per_elem_factors[i], inv);
        if (i > 0)
          inv = e4::mul(inv, per_elem_factor_invs[i]);
      }
  }

  for (unsigned i = 0, g = gid; i < INV_BATCH; i++, g += grid_size)
    if (g < n)
      denom_at_z[g] = per_elem_factors[i];
}

// DEEP quotient kernel
kernel void ab_deep_quotient_kernel(
    device const bf *setup_cols [[buffer(0)]],
    constant size_t &setup_stride [[buffer(1)]],
    device const bf *witness_cols [[buffer(2)]],
    constant size_t &witness_stride [[buffer(3)]],
    device const bf *memory_cols [[buffer(4)]],
    constant size_t &memory_stride [[buffer(5)]],
    device const bf *stage_2_bf_cols [[buffer(6)]],
    constant size_t &stage_2_bf_stride [[buffer(7)]],
    device const bf *stage_2_e4_cols [[buffer(8)]],
    constant size_t &stage_2_e4_stride [[buffer(9)]],
    device const bf *composition_col [[buffer(10)]],
    constant size_t &composition_stride [[buffer(11)]],
    device const e4 *denom_at_z [[buffer(12)]],
    device const e4 *setup_challenges_at_z [[buffer(13)]],
    device const e4 *witness_challenges_at_z [[buffer(14)]],
    device const e4 *memory_challenges_at_z [[buffer(15)]],
    device const e4 *stage_2_bf_challenges_at_z [[buffer(16)]],
    device const e4 *stage_2_e4_challenges_at_z [[buffer(17)]],
    device const e4 *composition_challenge_at_z [[buffer(18)]],
    constant StateLinkageConstraints &state_linkage_constraints [[buffer(19)]],
    constant ColIdxsToChallengeIdxsMap &memory_cols_to_challenges_map [[buffer(20)]],
    device const e4 *witness_challenges_at_z_omega [[buffer(21)]],
    device const e4 *memory_challenges_at_z_omega [[buffer(22)]],
    device const e4 *grand_product_challenge_at_z_omega [[buffer(23)]],
    device const ChallengesTimesEvalsSums *sums_ref [[buffer(24)]],
    device bf *quotient [[buffer(25)]],
    constant size_t &quotient_stride [[buffer(26)]],
    constant unsigned &num_setup_cols [[buffer(27)]],
    constant unsigned &num_witness_cols [[buffer(28)]],
    constant unsigned &num_memory_cols [[buffer(29)]],
    constant unsigned &num_stage_2_bf_cols [[buffer(30)]],
    constant unsigned &num_stage_2_e4_cols [[buffer(31)]],
    // Use a separate constant buffer for the remaining params
    // to avoid exceeding the buffer limit
    constant unsigned &stage_2_memory_grand_product_offset [[buffer(32)]],
    constant unsigned &log_n [[buffer(33)]],
    constant bool &bit_reversed [[buffer(34)]],
    uint gid [[thread_position_in_grid]])
  [[max_total_threads_per_threadgroup(512)]] {
  const unsigned n = 1u << log_n;
  if (gid >= n)
    return;

  e4 acc_z = e4::zero();
  e4 acc_z_omega = e4::zero();

  // Setup terms at z
  for (unsigned i = 0; i < num_setup_cols; i++) {
    const bf val = setup_cols[gid + i * setup_stride];
    const e4 challenge = setup_challenges_at_z[i];
    acc_z = e4::add(acc_z, e4::mul(challenge, val));
  }

  // Witness terms at z
  for (unsigned i = 0; i < num_witness_cols; i++) {
    const bf val = witness_cols[gid + i * witness_stride];
    const e4 challenge = witness_challenges_at_z[i];
    acc_z = e4::add(acc_z, e4::mul(challenge, val));
  }

  // Witness terms at z * omega (state linkage)
  for (unsigned i = 0; i < state_linkage_constraints.num_constraints; i++) {
    const bf val = witness_cols[gid + state_linkage_constraints.dsts[i] * witness_stride];
    const e4 challenge = witness_challenges_at_z_omega[i];
    acc_z_omega = e4::add(acc_z_omega, e4::mul(challenge, val));
  }

  // Memory terms at z and z * omega
  {
    unsigned challenge_at_z_omega_idx = 0;
    for (unsigned i = 0; i < num_memory_cols; i++) {
      const bf val = memory_cols[gid + i * memory_stride];
      const e4 challenge = memory_challenges_at_z[i];
      acc_z = e4::add(acc_z, e4::mul(challenge, val));
      const unsigned maybe_idx = memory_cols_to_challenges_map.map[i];
      if (maybe_idx != DOES_NOT_NEED_Z_OMEGA) {
        const e4 ch = memory_challenges_at_z_omega[challenge_at_z_omega_idx++];
        acc_z_omega = e4::add(acc_z_omega, e4::mul(ch, val));
      }
    }
  }

  // Stage 2 bf terms at z
  for (unsigned i = 0; i < num_stage_2_bf_cols; i++) {
    const bf val = stage_2_bf_cols[gid + i * stage_2_bf_stride];
    const e4 challenge = stage_2_bf_challenges_at_z[i];
    acc_z = e4::add(acc_z, e4::mul(challenge, val));
  }

  // Stage 2 e4 terms at z and z * omega
  for (unsigned i = 0; i < num_stage_2_e4_cols; i++) {
    // Read e4 from 4 bf columns
    bf coeffs[4];
    for (unsigned c = 0; c < 4; c++)
      coeffs[c] = stage_2_e4_cols[gid + (i * 4 + c) * stage_2_e4_stride];
    const e4 val(coeffs);
    const e4 challenge = stage_2_e4_challenges_at_z[i];
    acc_z = e4::add(acc_z, e4::mul(challenge, val));
    if (i == stage_2_memory_grand_product_offset) {
      const e4 ch = grand_product_challenge_at_z_omega[0];
      acc_z_omega = e4::add(acc_z_omega, e4::mul(ch, val));
    }
  }

  // Composition term at z (read e4 from 4 bf columns)
  {
    bf coeffs[4];
    for (unsigned c = 0; c < 4; c++)
      coeffs[c] = composition_col[gid + c * composition_stride];
    const e4 val(coeffs);
    const e4 challenge = composition_challenge_at_z[0];
    acc_z = e4::add(acc_z, e4::mul(challenge, val));
  }

  const e4 denom_z = denom_at_z[gid];
  const unsigned raw_row = bit_reversed ? reverse_bits(gid) >> (32 - log_n) : gid;
  const unsigned row_shift = n - 1;
  const unsigned raw_shifted_row = (raw_row + row_shift >= n) ? raw_row + row_shift - n : raw_row + row_shift;
  const unsigned shifted_row = bit_reversed ? reverse_bits(raw_shifted_row) >> (32 - log_n) : raw_shifted_row;
  const e4 denom_z_omega = denom_at_z[shifted_row];

  acc_z = e4::add(acc_z, sums_ref->at_z_sum_neg);
  acc_z_omega = e4::add(acc_z_omega, sums_ref->at_z_omega_sum_neg);
  acc_z = e4::mul(acc_z, denom_z);
  acc_z_omega = e4::mul(acc_z_omega, denom_z_omega);

  const e4 result = e4::add(acc_z, acc_z_omega);
  for (unsigned c = 0; c < 4; c++)
    quotient[gid + c * quotient_stride] = result.base_coefficient_from_flat_idx(c);
}

} // namespace stage4
} // namespace airbender
