#include <metal_stdlib>
using namespace metal;

#include "context.metal"
#include "field.metal"
#include "memory.metal"
#include "vectorized.metal"

using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;

namespace airbender {
namespace ops_complex {

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

// Batch inversion using Montgomery's trick
template <typename T, int INV_BATCH, bool batch_is_full>
DEVICE_FORCEINLINE void batch_inv_registers(const thread T *inputs, thread T *fwd_scan_and_outputs, int runtime_batch_size) {
  T running_prod = T::one();
  for (int i = 0; i < INV_BATCH; i++)
    if (batch_is_full || i < runtime_batch_size) {
      fwd_scan_and_outputs[i] = running_prod;
      running_prod = T::mul(running_prod, inputs[i]);
    }

  T inv = T::inv(running_prod);

  for (int i = INV_BATCH - 1; i >= 0; i--) {
    if (batch_is_full || i < runtime_batch_size) {
      const auto input = inputs[i];
      fwd_scan_and_outputs[i] = T::mul(fwd_scan_and_outputs[i], inv);
      if (i > 0)
        inv = T::mul(inv, input);
    }
  }
}

template <typename T> struct InvBatch {};
template <> struct InvBatch<bf> { enum : unsigned { INV_BATCH = 20 }; };
template <> struct InvBatch<e2> { enum : unsigned { INV_BATCH = 5 }; };
template <> struct InvBatch<e4> { enum : unsigned { INV_BATCH = 3 }; };

// Get powers kernels
kernel void ab_get_powers_by_val_bf_kernel(constant bf &base [[buffer(0)]],
                                            constant unsigned &offset [[buffer(1)]],
                                            constant bool &bit_reverse [[buffer(2)]],
                                            device bf *result [[buffer(3)]],
                                            constant unsigned &count [[buffer(4)]],
                                            uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  const unsigned power = (bit_reverse ? reverse_bits(gid) : gid) + offset;
  result[gid] = bf::pow(base, power);
}

kernel void ab_get_powers_by_val_e2_kernel(constant e2 &base [[buffer(0)]],
                                            constant unsigned &offset [[buffer(1)]],
                                            constant bool &bit_reverse [[buffer(2)]],
                                            device e2 *result [[buffer(3)]],
                                            constant unsigned &count [[buffer(4)]],
                                            uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  const unsigned power = (bit_reverse ? reverse_bits(gid) : gid) + offset;
  result[gid] = e2::pow(base, power);
}

kernel void ab_get_powers_by_val_e4_kernel(constant e4 &base [[buffer(0)]],
                                            constant unsigned &offset [[buffer(1)]],
                                            constant bool &bit_reverse [[buffer(2)]],
                                            device e4 *result [[buffer(3)]],
                                            constant unsigned &count [[buffer(4)]],
                                            uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  const unsigned power = (bit_reverse ? reverse_bits(gid) : gid) + offset;
  result[gid] = e4::pow(base, power);
}

// Batch inverse kernels
kernel void ab_batch_inv_bf_kernel(device const bf *src [[buffer(0)]],
                                    device bf *dst [[buffer(1)]],
                                    constant unsigned &count [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]],
                                    uint grid_size [[threads_per_grid]]) {
  constexpr unsigned INV_BATCH = InvBatch<bf>::INV_BATCH;
  if (gid >= count) return;

  bf inputs[INV_BATCH];
  bf outputs[INV_BATCH];

  int runtime_batch_size = 0;
  uint g = gid;
  for (int i = 0; i < (int)INV_BATCH; i++, g += grid_size)
    if (g < count) {
      inputs[i] = src[g];
      runtime_batch_size++;
    }

  if (runtime_batch_size < (int)INV_BATCH)
    batch_inv_registers<bf, INV_BATCH, false>(inputs, outputs, runtime_batch_size);
  else
    batch_inv_registers<bf, INV_BATCH, true>(inputs, outputs, runtime_batch_size);

  g -= grid_size;
  for (int i = INV_BATCH - 1; i >= 0; --i, g -= grid_size)
    if (i < runtime_batch_size)
      dst[g] = outputs[i];
}

kernel void ab_batch_inv_e2_kernel(device const e2 *src [[buffer(0)]],
                                    device e2 *dst [[buffer(1)]],
                                    constant unsigned &count [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]],
                                    uint grid_size [[threads_per_grid]]) {
  constexpr unsigned INV_BATCH = InvBatch<e2>::INV_BATCH;
  if (gid >= count) return;

  e2 inputs[INV_BATCH];
  e2 outputs[INV_BATCH];

  int runtime_batch_size = 0;
  uint g = gid;
  for (int i = 0; i < (int)INV_BATCH; i++, g += grid_size)
    if (g < count) {
      inputs[i] = src[g];
      runtime_batch_size++;
    }

  if (runtime_batch_size < (int)INV_BATCH)
    batch_inv_registers<e2, INV_BATCH, false>(inputs, outputs, runtime_batch_size);
  else
    batch_inv_registers<e2, INV_BATCH, true>(inputs, outputs, runtime_batch_size);

  g -= grid_size;
  for (int i = INV_BATCH - 1; i >= 0; --i, g -= grid_size)
    if (i < runtime_batch_size)
      dst[g] = outputs[i];
}

kernel void ab_batch_inv_e4_kernel(device const e4 *src [[buffer(0)]],
                                    device e4 *dst [[buffer(1)]],
                                    constant unsigned &count [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]],
                                    uint grid_size [[threads_per_grid]]) {
  constexpr unsigned INV_BATCH = InvBatch<e4>::INV_BATCH;
  if (gid >= count) return;

  e4 inputs[INV_BATCH];
  e4 outputs[INV_BATCH];

  int runtime_batch_size = 0;
  uint g = gid;
  for (int i = 0; i < (int)INV_BATCH; i++, g += grid_size)
    if (g < count) {
      inputs[i] = src[g];
      runtime_batch_size++;
    }

  if (runtime_batch_size < (int)INV_BATCH)
    batch_inv_registers<e4, INV_BATCH, false>(inputs, outputs, runtime_batch_size);
  else
    batch_inv_registers<e4, INV_BATCH, true>(inputs, outputs, runtime_batch_size);

  g -= grid_size;
  for (int i = INV_BATCH - 1; i >= 0; --i, g -= grid_size)
    if (i < runtime_batch_size)
      dst[g] = outputs[i];
}

// Transpose kernel using threadgroup shared memory
kernel void ab_transpose_bf_kernel(device const bf *src [[buffer(0)]],
                                    device bf *dst [[buffer(1)]],
                                    constant size_t &src_stride [[buffer(2)]],
                                    constant size_t &dst_stride [[buffer(3)]],
                                    constant unsigned &src_rows [[buffer(4)]],
                                    constant unsigned &src_cols [[buffer(5)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint bid [[threadgroup_position_in_grid]],
                                    threadgroup bf *tile [[threadgroup(0)]]) {
  constexpr unsigned TILE_DIM = 32;
  const unsigned src_tiles_per_row = (src_cols + TILE_DIM - 1) / TILE_DIM;
  const unsigned src_tile_row_offset = bid / src_tiles_per_row * TILE_DIM;
  const unsigned src_tile_col_offset = bid % src_tiles_per_row * TILE_DIM;
  const unsigned dst_tile_row_offset = src_tile_col_offset;
  const unsigned dst_tile_col_offset = src_tile_row_offset;
  const unsigned src_row = src_tile_row_offset + tid;
  const unsigned dst_row = dst_tile_row_offset + tid;

  for (unsigned i = 0; i < TILE_DIM; i++) {
    const unsigned src_col = src_tile_col_offset + i;
    const unsigned row_swizzled = tid ^ i;
    if (src_row < src_rows && src_col < src_cols)
      tile[i * TILE_DIM + row_swizzled] = src[src_row + src_col * src_stride];
  }

  if (TILE_DIM <= 32)
    simdgroup_barrier(mem_flags::mem_threadgroup);
  else
    threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned i = 0; i < TILE_DIM; i++) {
    const unsigned dst_col = dst_tile_col_offset + i;
    const unsigned row_swizzled = tid ^ i;
    if (dst_row < src_cols && dst_col < src_rows)
      dst[dst_row + dst_col * dst_stride] = tile[tid * TILE_DIM + row_swizzled];
  }
}

// Bit reverse naive kernels
kernel void ab_bit_reverse_naive_bf_kernel(device const bf *src [[buffer(0)]],
                                            device bf *dst [[buffer(1)]],
                                            constant size_t &stride [[buffer(2)]],
                                            constant unsigned &log_count [[buffer(3)]],
                                            constant unsigned &col [[buffer(4)]],
                                            uint gid [[thread_position_in_grid]]) {
  if (gid >= 1u << log_count) return;
  const unsigned l_index = gid;
  const unsigned r_index = reverse_bits(l_index) >> (32 - log_count);
  if (l_index > r_index) return;
  const bf l_value = src[l_index + col * stride];
  const bf r_value = src[r_index + col * stride];
  dst[l_index + col * stride] = r_value;
  dst[r_index + col * stride] = l_value;
}

kernel void ab_bit_reverse_naive_e4_kernel(device const e4 *src [[buffer(0)]],
                                            device e4 *dst [[buffer(1)]],
                                            constant size_t &stride [[buffer(2)]],
                                            constant unsigned &log_count [[buffer(3)]],
                                            constant unsigned &col [[buffer(4)]],
                                            uint gid [[thread_position_in_grid]]) {
  if (gid >= 1u << log_count) return;
  const unsigned l_index = gid;
  const unsigned r_index = reverse_bits(l_index) >> (32 - log_count);
  if (l_index > r_index) return;
  const e4 l_value = src[l_index + col * stride];
  const e4 r_value = src[r_index + col * stride];
  dst[l_index + col * stride] = r_value;
  dst[r_index + col * stride] = l_value;
}

// Fold kernel
kernel void ab_fold_kernel(device const e4 *challenge [[buffer(0)]],
                            device const e4 *src [[buffer(1)]],
                            device e4 *dst [[buffer(2)]],
                            constant unsigned &root_offset [[buffer(3)]],
                            constant unsigned &log_count [[buffer(4)]],
                            // Powers data for get_power_of_w
                            device const e2 *fine_values [[buffer(5)]],
                            constant unsigned &fine_mask [[buffer(6)]],
                            constant unsigned &fine_log_count [[buffer(7)]],
                            device const e2 *coarser_values [[buffer(8)]],
                            constant unsigned &coarser_mask [[buffer(9)]],
                            constant unsigned &coarser_log_count [[buffer(10)]],
                            device const e2 *coarsest_values [[buffer(11)]],
                            constant unsigned &coarsest_mask [[buffer(12)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= 1u << log_count)
    return;
  const e4 even = src[2 * gid];
  const e4 odd = src[2 * gid + 1];
  const e4 sum = e4::add(even, odd);
  e4 diff = e4::sub(even, odd);

  // Construct powers_data_3_layer on thread
  powers_data_3_layer powers_w;
  powers_w.fine.values = fine_values;
  powers_w.fine.mask = fine_mask;
  powers_w.fine.log_count = fine_log_count;
  powers_w.coarser.values = coarser_values;
  powers_w.coarser.mask = coarser_mask;
  powers_w.coarser.log_count = coarser_log_count;
  powers_w.coarsest.values = coarsest_values;
  powers_w.coarsest.mask = coarsest_mask;

  const unsigned root_index = reverse_bits(gid + root_offset) >> (32 - CIRCLE_GROUP_LOG_ORDER + 1);
  const e2 root = get_power_of_w(powers_w, root_index, true);
  diff *= root;
  diff *= *challenge;
  dst[gid] = e4::add(sum, diff);
}

} // namespace ops_complex
} // namespace airbender
