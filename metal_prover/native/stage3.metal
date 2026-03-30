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
namespace stage3 {

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

constexpr unsigned MAX_NON_BOOLEAN_CONSTRAINTS = 192;
constexpr unsigned MAX_TERMS = 2208;
constexpr unsigned MAX_EXPLICIT_COEFFS = 928;
constexpr unsigned MAX_FLAT_COL_IDXS = 4192;
constexpr uint8_t COEFF_IS_ONE = 0x00;
constexpr uint8_t COEFF_IS_MINUS_ONE = 0x01;

// Passed as constant buffer argument instead of __grid_constant__
struct FlattenedGenericConstraintsMetadata {
  uint8_t coeffs_info[MAX_TERMS];
  bf explicit_coeffs[MAX_EXPLICIT_COEFFS];
  uint16_t col_idxs[MAX_FLAT_COL_IDXS];
  // MSL doesn't have uchar2; use two separate arrays or a struct
  uint8_t num_linear_terms[MAX_NON_BOOLEAN_CONSTRAINTS];
  uint8_t num_quadratic_terms[MAX_NON_BOOLEAN_CONSTRAINTS];
  e2 decompression_factor;
  e2 decompression_factor_squared;
  e2 every_row_zerofier;
  e2 omega_inv;
  unsigned current_flat_col_idx;
  unsigned current_flat_term_idx;
  unsigned num_boolean_constraints;
  unsigned num_non_boolean_quadratic_constraints;
  unsigned num_non_boolean_constraints;
};

DEVICE_FORCEINLINE void maybe_apply_coeff(constant const FlattenedGenericConstraintsMetadata &metadata,
                                           const unsigned coeff_idx, thread unsigned &explicit_coeff_idx, thread bf &val) {
  switch (metadata.coeffs_info[coeff_idx]) {
  case COEFF_IS_ONE:
    break;
  case COEFF_IS_MINUS_ONE:
    val = bf::neg(val);
    break;
  default:
    val = bf::mul(val, metadata.explicit_coeffs[explicit_coeff_idx++]);
  }
}

kernel void ab_generic_constraints_kernel(
    constant FlattenedGenericConstraintsMetadata &metadata [[buffer(0)]],
    device const bf *witness_cols [[buffer(1)]],
    constant size_t &witness_stride [[buffer(2)]],
    device const bf *memory_cols [[buffer(3)]],
    constant size_t &memory_stride [[buffer(4)]],
    device const e4 *alphas [[buffer(5)]],
    device bf *quotient [[buffer(6)]],
    constant size_t &quotient_stride [[buffer(7)]],
    constant unsigned &log_n [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
  [[max_total_threads_per_threadgroup(128)]] {
  const unsigned n = 1 << log_n;
  if (gid >= n)
    return;

  // Point to this thread's row
  device const bf *w_row = witness_cols + gid;
  device const bf *m_row = memory_cols + gid;

  e4 acc_linear = e4::zero();
  e4 acc_quadratic = e4::zero();
  unsigned alpha_idx = 0;

  // Boolean constraints
  for (unsigned constraint = 0; constraint < metadata.num_boolean_constraints; constraint++) {
    const bf val_neg = bf::neg(w_row[metadata.col_idxs[constraint] * witness_stride]);
    const bf val_squared = bf::mul(val_neg, val_neg);
    const e4 alpha_power = alphas[alpha_idx++];
    acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha_power, val_squared));
    acc_linear = e4::add(acc_linear, e4::mul(alpha_power, val_neg));
  }

  unsigned flat_term_idx = 0;
  unsigned flat_col_idx = metadata.num_boolean_constraints;
  unsigned explicit_coeff_idx = 0;

  // Non-boolean quadratic constraints
  for (unsigned constraint = 0; constraint < metadata.num_non_boolean_quadratic_constraints; constraint++) {
    const unsigned num_quadratic_terms = metadata.num_quadratic_terms[constraint];
    const unsigned num_linear_terms = metadata.num_linear_terms[constraint];

    bf quadratic_contribution = bf::zero();
    unsigned lim = flat_term_idx + num_quadratic_terms;
    for (; flat_term_idx < lim; flat_term_idx++) {
      const unsigned col0 = metadata.col_idxs[flat_col_idx++];
      const unsigned col1 = metadata.col_idxs[flat_col_idx++];
      bf val0 = (col0 & COL_TYPE_MEMORY) ? m_row[(col0 & COL_IDX_MASK) * memory_stride] : w_row[col0 * witness_stride];
      bf val1 = (col1 & COL_TYPE_MEMORY) ? m_row[(col1 & COL_IDX_MASK) * memory_stride] : w_row[col1 * witness_stride];
      bf val = bf::mul(val0, val1);
      maybe_apply_coeff(metadata, flat_term_idx, explicit_coeff_idx, val);
      quadratic_contribution = bf::add(quadratic_contribution, val);
    }
    const e4 alpha_power = alphas[alpha_idx++];
    acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha_power, quadratic_contribution));

    if (num_linear_terms > 0) {
      bf linear_contribution = bf::zero();
      lim = flat_term_idx + num_linear_terms;
      for (; flat_term_idx < lim; flat_term_idx++) {
        const unsigned col = metadata.col_idxs[flat_col_idx++];
        bf val = (col & COL_TYPE_MEMORY) ? m_row[(col & COL_IDX_MASK) * memory_stride] : w_row[col * witness_stride];
        maybe_apply_coeff(metadata, flat_term_idx, explicit_coeff_idx, val);
        linear_contribution = bf::add(linear_contribution, val);
      }
      acc_linear = e4::add(acc_linear, e4::mul(alpha_power, linear_contribution));
    }
  }

  // Linear-only constraints
  for (unsigned constraint = metadata.num_non_boolean_quadratic_constraints; constraint < metadata.num_non_boolean_constraints; constraint++) {
    const unsigned num_linear_terms = metadata.num_linear_terms[constraint];
    bf linear_contribution = bf::zero();
    const unsigned lim = flat_term_idx + num_linear_terms;
    for (; flat_term_idx < lim; flat_term_idx++) {
      const unsigned col = metadata.col_idxs[flat_col_idx++];
      bf val = (col & COL_TYPE_MEMORY) ? m_row[(col & COL_IDX_MASK) * memory_stride] : w_row[col * witness_stride];
      maybe_apply_coeff(metadata, flat_term_idx, explicit_coeff_idx, val);
      linear_contribution = bf::add(linear_contribution, val);
    }
    const e4 alpha_power = alphas[alpha_idx++];
    acc_linear = e4::add(acc_linear, e4::mul(alpha_power, linear_contribution));
  }

  acc_quadratic = e4::mul(acc_quadratic, metadata.decompression_factor_squared);
  acc_linear = e4::mul(acc_linear, metadata.decompression_factor);
  e4 acc = e4::add(acc_quadratic, acc_linear);

  // Write e4 result to quotient as 4 bf components
  for (unsigned c = 0; c < 4; c++)
    quotient[gid + c * quotient_stride] = acc.base_coefficient_from_flat_idx(c);
}

} // namespace stage3
} // namespace airbender
