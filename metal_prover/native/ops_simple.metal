#include <metal_stdlib>
using namespace metal;

#include "field.metal"
#include "memory.metal"

using namespace airbender::field;
using namespace airbender::memory;

namespace airbender {
namespace ops_simple {

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

template <typename T> struct value_getter {
  using value_type = T;
  T value;
  DEVICE_FORCEINLINE T get(const unsigned, const unsigned) const { return value; }
};

using bf_value_getter = value_getter<bf>;
using bf_getter = wrapping_matrix_getter<matrix_getter<bf>>;
using bf_setter = wrapping_matrix_setter<matrix_setter<bf>>;

using e2_value_getter = value_getter<e2>;
using e2_getter = wrapping_matrix_getter<matrix_getter<e2>>;
using e2_setter = wrapping_matrix_setter<matrix_setter<e2>>;

using e4_value_getter = value_getter<e4>;
using e4_getter = wrapping_matrix_getter<matrix_getter<e4>>;
using e4_setter = wrapping_matrix_setter<matrix_setter<e4>>;

using u32_value_getter = value_getter<uint32_t>;
using u32_getter = wrapping_matrix_getter<matrix_getter<uint32_t>>;
using u32_setter = wrapping_matrix_setter<matrix_setter<uint32_t>>;

using u64_value_getter = value_getter<uint64_t>;
using u64_getter = wrapping_matrix_getter<matrix_getter<uint64_t>>;
using u64_setter = wrapping_matrix_setter<matrix_setter<uint64_t>>;

// Helper operation functions
DEVICE_FORCEINLINE bf add(const bf x, const bf y) { return bf::add(x, y); }
DEVICE_FORCEINLINE e2 add(const bf x, const e2 y) { return e2::add(x, y); }
DEVICE_FORCEINLINE e2 add(const e2 x, const bf y) { return e2::add(x, y); }
DEVICE_FORCEINLINE e2 add(const e2 x, const e2 y) { return e2::add(x, y); }
DEVICE_FORCEINLINE e4 add(const bf x, const e4 y) { return e4::add(x, y); }
DEVICE_FORCEINLINE e4 add(const e2 x, const e4 y) { return e4::add(x, y); }
DEVICE_FORCEINLINE e4 add(const e4 x, const bf y) { return e4::add(x, y); }
DEVICE_FORCEINLINE e4 add(const e4 x, const e2 y) { return e4::add(x, y); }
DEVICE_FORCEINLINE e4 add(const e4 x, const e4 y) { return e4::add(x, y); }

DEVICE_FORCEINLINE bf mul(const bf x, const bf y) { return bf::mul(x, y); }
DEVICE_FORCEINLINE e2 mul(const bf x, const e2 y) { return e2::mul(x, y); }
DEVICE_FORCEINLINE e2 mul(const e2 x, const bf y) { return e2::mul(x, y); }
DEVICE_FORCEINLINE e2 mul(const e2 x, const e2 y) { return e2::mul(x, y); }
DEVICE_FORCEINLINE e4 mul(const bf x, const e4 y) { return e4::mul(x, y); }
DEVICE_FORCEINLINE e4 mul(const e2 x, const e4 y) { return e4::mul(x, y); }
DEVICE_FORCEINLINE e4 mul(const e4 x, const bf y) { return e4::mul(x, y); }
DEVICE_FORCEINLINE e4 mul(const e4 x, const e2 y) { return e4::mul(x, y); }
DEVICE_FORCEINLINE e4 mul(const e4 x, const e4 y) { return e4::mul(x, y); }

DEVICE_FORCEINLINE bf sub(const bf x, const bf y) { return bf::sub(x, y); }
DEVICE_FORCEINLINE e2 sub(const bf x, const e2 y) { return e2::sub(x, y); }
DEVICE_FORCEINLINE e2 sub(const e2 x, const bf y) { return e2::sub(x, y); }
DEVICE_FORCEINLINE e2 sub(const e2 x, const e2 y) { return e2::sub(x, y); }
DEVICE_FORCEINLINE e4 sub(const bf x, const e4 y) { return e4::sub(x, y); }
DEVICE_FORCEINLINE e4 sub(const e2 x, const e4 y) { return e4::sub(x, y); }
DEVICE_FORCEINLINE e4 sub(const e4 x, const bf y) { return e4::sub(x, y); }
DEVICE_FORCEINLINE e4 sub(const e4 x, const e2 y) { return e4::sub(x, y); }
DEVICE_FORCEINLINE e4 sub(const e4 x, const e4 y) { return e4::sub(x, y); }

// Kernel templates for element-wise operations
// In Metal, we define individual kernels rather than using macro instantiation.
// The host code dispatches the appropriate kernel.

// Set by value kernels
kernel void ab_set_by_val_bf_kernel(constant bf &value [[buffer(0)]],
                                     device bf *result [[buffer(1)]],
                                     constant unsigned &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  result[gid] = value;
}

kernel void ab_set_by_val_e2_kernel(constant e2 &value [[buffer(0)]],
                                     device e2 *result [[buffer(1)]],
                                     constant unsigned &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  result[gid] = value;
}

kernel void ab_set_by_val_e4_kernel(constant e4 &value [[buffer(0)]],
                                     device e4 *result [[buffer(1)]],
                                     constant unsigned &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  result[gid] = value;
}

kernel void ab_set_by_val_u32_kernel(constant uint32_t &value [[buffer(0)]],
                                      device uint32_t *result [[buffer(1)]],
                                      constant unsigned &count [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  result[gid] = value;
}

kernel void ab_set_by_val_u64_kernel(constant uint64_t &value [[buffer(0)]],
                                      device uint64_t *result [[buffer(1)]],
                                      constant unsigned &count [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  result[gid] = value;
}

// Set by ref kernels
kernel void ab_set_by_ref_bf_kernel(device const bf *src [[buffer(0)]],
                                     device bf *dst [[buffer(1)]],
                                     constant unsigned &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = src[gid];
}

kernel void ab_set_by_ref_e2_kernel(device const e2 *src [[buffer(0)]],
                                     device e2 *dst [[buffer(1)]],
                                     constant unsigned &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = src[gid];
}

kernel void ab_set_by_ref_e4_kernel(device const e4 *src [[buffer(0)]],
                                     device e4 *dst [[buffer(1)]],
                                     constant unsigned &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = src[gid];
}

// Unary operation kernels
kernel void ab_neg_bf_kernel(device const bf *src [[buffer(0)]],
                              device bf *dst [[buffer(1)]],
                              constant unsigned &count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::neg(src[gid]);
}

kernel void ab_neg_e2_kernel(device const e2 *src [[buffer(0)]],
                              device e2 *dst [[buffer(1)]],
                              constant unsigned &count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = e2::neg(src[gid]);
}

kernel void ab_neg_e4_kernel(device const e4 *src [[buffer(0)]],
                              device e4 *dst [[buffer(1)]],
                              constant unsigned &count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = e4::neg(src[gid]);
}

kernel void ab_dbl_bf_kernel(device const bf *src [[buffer(0)]],
                              device bf *dst [[buffer(1)]],
                              constant unsigned &count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::dbl(src[gid]);
}

kernel void ab_inv_bf_kernel(device const bf *src [[buffer(0)]],
                              device bf *dst [[buffer(1)]],
                              constant unsigned &count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::inv(src[gid]);
}

kernel void ab_sqr_bf_kernel(device const bf *src [[buffer(0)]],
                              device bf *dst [[buffer(1)]],
                              constant unsigned &count [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::sqr(src[gid]);
}

// Binary operation kernels
kernel void ab_add_bf_bf_kernel(device const bf *a [[buffer(0)]],
                                 device const bf *b [[buffer(1)]],
                                 device bf *dst [[buffer(2)]],
                                 constant unsigned &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::add(a[gid], b[gid]);
}

kernel void ab_mul_bf_bf_kernel(device const bf *a [[buffer(0)]],
                                 device const bf *b [[buffer(1)]],
                                 device bf *dst [[buffer(2)]],
                                 constant unsigned &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::mul(a[gid], b[gid]);
}

kernel void ab_sub_bf_bf_kernel(device const bf *a [[buffer(0)]],
                                 device const bf *b [[buffer(1)]],
                                 device bf *dst [[buffer(2)]],
                                 constant unsigned &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::sub(a[gid], b[gid]);
}

kernel void ab_add_e4_e4_kernel(device const e4 *a [[buffer(0)]],
                                 device const e4 *b [[buffer(1)]],
                                 device e4 *dst [[buffer(2)]],
                                 constant unsigned &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = e4::add(a[gid], b[gid]);
}

kernel void ab_mul_e4_e4_kernel(device const e4 *a [[buffer(0)]],
                                 device const e4 *b [[buffer(1)]],
                                 device e4 *dst [[buffer(2)]],
                                 constant unsigned &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = e4::mul(a[gid], b[gid]);
}

kernel void ab_sub_e4_e4_kernel(device const e4 *a [[buffer(0)]],
                                 device const e4 *b [[buffer(1)]],
                                 device e4 *dst [[buffer(2)]],
                                 constant unsigned &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = e4::sub(a[gid], b[gid]);
}

kernel void ab_mul_bf_e4_kernel(device const bf *a [[buffer(0)]],
                                 device const e4 *b [[buffer(1)]],
                                 device e4 *dst [[buffer(2)]],
                                 constant unsigned &count [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = e4::mul(a[gid], b[gid]);
}

// Parametrized kernels (pow, shl, shr)
kernel void ab_pow_bf_kernel(device const bf *src [[buffer(0)]],
                              constant uint32_t &parameter [[buffer(1)]],
                              device bf *dst [[buffer(2)]],
                              constant unsigned &count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::pow(src[gid], parameter);
}

kernel void ab_shl_bf_kernel(device const bf *src [[buffer(0)]],
                              constant uint32_t &parameter [[buffer(1)]],
                              device bf *dst [[buffer(2)]],
                              constant unsigned &count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::shl(src[gid], parameter);
}

kernel void ab_shr_bf_kernel(device const bf *src [[buffer(0)]],
                              constant uint32_t &parameter [[buffer(1)]],
                              device bf *dst [[buffer(2)]],
                              constant unsigned &count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::shr(src[gid], parameter);
}

// Ternary kernels (mul_add, mul_sub)
kernel void ab_mul_add_bf_bf_bf_kernel(device const bf *a [[buffer(0)]],
                                        device const bf *b [[buffer(1)]],
                                        device const bf *c [[buffer(2)]],
                                        device bf *dst [[buffer(3)]],
                                        constant unsigned &count [[buffer(4)]],
                                        uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::add(bf::mul(a[gid], b[gid]), c[gid]);
}

kernel void ab_mul_sub_bf_bf_bf_kernel(device const bf *a [[buffer(0)]],
                                        device const bf *b [[buffer(1)]],
                                        device const bf *c [[buffer(2)]],
                                        device bf *dst [[buffer(3)]],
                                        constant unsigned &count [[buffer(4)]],
                                        uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = bf::sub(bf::mul(a[gid], b[gid]), c[gid]);
}

kernel void ab_mul_add_e4_e4_e4_kernel(device const e4 *a [[buffer(0)]],
                                        device const e4 *b [[buffer(1)]],
                                        device const e4 *c [[buffer(2)]],
                                        device e4 *dst [[buffer(3)]],
                                        constant unsigned &count [[buffer(4)]],
                                        uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  dst[gid] = e4::add(e4::mul(a[gid], b[gid]), c[gid]);
}

} // namespace ops_simple
} // namespace airbender
