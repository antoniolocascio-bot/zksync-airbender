#pragma once
#include <metal_stdlib>
using namespace metal;

#include "../context.metal"
#include "../field.metal"
#include "../memory.metal"
#include "../vectorized.metal"

using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;

namespace airbender {
namespace ntt {

using bf = base_field;
using e2f = ext2_field;

DEVICE_FORCEINLINE unsigned bitrev(const unsigned idx, const unsigned log_n) { return reverse_bits(idx) >> (32 - log_n); }

// Load-store helpers for TWO-FOR-ONE FFT ALGORITHM
DEVICE_FORCEINLINE e2f load_one_maybe_vectorized(const device bf *gmem_in, const unsigned stride, bool is_odd_tail) {
  const auto c0 = *gmem_in;
  const auto c1 = is_odd_tail ? bf{0} : *(gmem_in + stride);
  return e2f{c0, c1};
}

DEVICE_FORCEINLINE e2f load_one_maybe_vectorized(const device e2f *gmem_in, const unsigned stride, bool is_odd_tail) {
  return *gmem_in;
}

DEVICE_FORCEINLINE void load_two_vectorized_complex(const device bf *gmem_in, thread e2f &val0, thread e2f &val1, const unsigned stride, bool is_odd_tail) {
  const auto c0_0 = *gmem_in;
  const auto c0_1 = *(gmem_in + 1);
  bf c1_0{0}, c1_1{0};
  if (!is_odd_tail) {
    c1_0 = *(gmem_in + stride);
    c1_1 = *(gmem_in + stride + 1);
  }
  val0 = e2f{c0_0, c1_0};
  val1 = e2f{c0_1, c1_1};
}

DEVICE_FORCEINLINE void store_one_maybe_vectorized(device bf *gmem_out, const e2f val, const unsigned stride, bool is_odd_tail) {
  *gmem_out = val[0];
  if (!is_odd_tail)
    *(gmem_out + stride) = val[1];
}

DEVICE_FORCEINLINE void store_one_maybe_vectorized(device e2f *gmem_out, const e2f val, const unsigned stride, bool is_odd_tail) {
  *gmem_out = val;
}

DEVICE_FORCEINLINE void store_two_vectorized_complex(device bf *gmem_out, const e2f val0, const e2f val1, const unsigned stride, bool is_odd_tail) {
  *gmem_out = val0[0];
  *(gmem_out + 1) = val1[0];
  if (!is_odd_tail) {
    *(gmem_out + stride) = val0[1];
    *(gmem_out + stride + 1) = val1[1];
  }
}

DEVICE_FORCEINLINE void exchg_dit(thread e2f &a, thread e2f &b, thread const e2f &twiddle) {
  b = e2f::mul(b, twiddle);
  const auto a_tmp = a;
  a = e2f::add(a_tmp, b);
  b = e2f::sub(a_tmp, b);
}

DEVICE_FORCEINLINE void exchg_dif(thread e2f &a, thread e2f &b, thread const e2f &twiddle) {
  const auto a_tmp = a;
  a = e2f::add(a_tmp, b);
  b = e2f::sub(a_tmp, b);
  b = e2f::mul(b, twiddle);
}

// Twiddle factor lookup
// In Metal, we pass the powers_data_2_layer as a parameter instead of using global constant memory.
template <bool inverse>
DEVICE_FORCEINLINE e2f get_twiddle(const device e2f *fine_values, const unsigned fine_mask, const unsigned fine_log_count,
                                    const device e2f *coarse_values, const unsigned coarse_mask, const unsigned coarse_log_count,
                                    const unsigned i) {
  unsigned fine_idx = (i >> coarse_log_count) & fine_mask;
  unsigned coarse_idx = i & coarse_mask;
  auto coarse = coarse_values[coarse_idx];
  if (fine_idx == 0)
    return coarse;
  auto fine = fine_values[fine_idx];
  return e2f::mul(fine, coarse);
}

// Shuffle ext2_field values across SIMD lanes
DEVICE_FORCEINLINE void shfl_xor_e2f(thread e2f *vals, const unsigned i, const unsigned lane_id, const unsigned lane_mask) {
  e2f tmp{};
  if (lane_id & lane_mask)
    tmp = vals[2 * i];
  else
    tmp = vals[2 * i + 1];
  tmp[0].limb = simd_shuffle_xor(tmp[0].limb, static_cast<ushort>(lane_mask));
  tmp[1].limb = simd_shuffle_xor(tmp[1].limb, static_cast<ushort>(lane_mask));
  if (lane_id & lane_mask)
    vals[2 * i] = tmp;
  else
    vals[2 * i + 1] = tmp;
}

// LDE scale and shift
DEVICE_FORCEINLINE e2f get_lde_scale_and_shift_factor(const thread powers_data_3_layer &powers_w,
                                                       const unsigned k, const unsigned log_extension_degree,
                                                       const unsigned coset_idx, const unsigned log_n,
                                                       const bool inverse = false) {
  const unsigned tau_power_of_w = coset_idx << (CIRCLE_GROUP_LOG_ORDER - log_n - log_extension_degree);
  const unsigned H_over_two = 1u << (log_n - 1);
  const unsigned power_of_w = k >= H_over_two
    ? tau_power_of_w * (k - H_over_two)
    : (1u << CIRCLE_GROUP_LOG_ORDER) - tau_power_of_w * (H_over_two - k);
  return get_power_of_w(powers_w, power_of_w, inverse);
}

DEVICE_FORCEINLINE e2f lde_scale_and_shift(const thread powers_data_3_layer &powers_w,
                                            const e2f Zk, const unsigned k,
                                            const unsigned log_extension_degree, const unsigned coset_idx,
                                            const unsigned log_n, const bool inverse = false) {
  if (coset_idx == 0) return Zk;
  const auto gauged_shift_factor = get_lde_scale_and_shift_factor(powers_w, k, log_extension_degree, coset_idx, log_n, inverse);
  return e2f::mul(Zk, gauged_shift_factor);
}

DEVICE_FORCEINLINE e2f lde_scale(const thread powers_data_3_layer &powers_w,
                                  const e2f Zk, const unsigned k,
                                  const unsigned log_extension_degree, const unsigned coset_idx,
                                  const unsigned log_n, const bool inverse = false) {
  if (coset_idx == 0) return Zk;
  const unsigned tau_power_of_w = coset_idx << (CIRCLE_GROUP_LOG_ORDER - log_n - log_extension_degree);
  const auto scale_factor = get_power_of_w(powers_w, k * tau_power_of_w, inverse);
  return e2f::mul(Zk, scale_factor);
}

template <typename T> struct COLS_PER_BLOCK {};
template <> struct COLS_PER_BLOCK<bf> { enum : unsigned { VAL = 8 }; };
template <> struct COLS_PER_BLOCK<e2f> { enum : unsigned { VAL = 4 }; };

template <typename T> struct COLS_INC {};
template <> struct COLS_INC<bf> { enum : unsigned { VAL = 2 }; };
template <> struct COLS_INC<e2f> { enum : unsigned { VAL = 1 }; };

} // namespace ntt
} // namespace airbender
