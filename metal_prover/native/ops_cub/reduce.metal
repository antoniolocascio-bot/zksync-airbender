#include <metal_stdlib>
using namespace metal;

#include "../field.metal"

using namespace airbender::field;

namespace airbender {
namespace ops_cub {

typedef uint32_t u32;
typedef base_field bf;
typedef ext2_field e2;
typedef ext4_field e4;

// Tree-based reduction using threadgroup shared memory.
// Two-pass approach: per-threadgroup reduce, then reduce partial results.

constexpr unsigned REDUCE_BLOCK_SIZE = 256;

// Per-threadgroup reduction template
template <typename T, typename Op>
DEVICE_FORCEINLINE T threadgroup_reduce(thread T &val, threadgroup T *shared, const unsigned tid, const unsigned count) {
  Op op;
  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned stride = count >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] = op(shared[tid], shared[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return shared[0];
}

// Reduce add bf
kernel void ab_reduce_add_bf_kernel(device const bf *d_in [[buffer(0)]],
                                     device bf *d_out [[buffer(1)]],
                                     constant int &num_items [[buffer(2)]],
                                     threadgroup bf *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  bf val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : bf::zero();

  // Accumulate multiple elements per thread if grid is smaller than input
  for (unsigned idx = global_idx + threads_per_group * gridDim; idx < (unsigned)num_items; idx += threads_per_group * gridDim)
    val = bf::add(val, d_in[idx]);

  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] = bf::add(shared[tid], shared[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0)
    d_out[bid] = shared[0];
}

// Reduce add e2
kernel void ab_reduce_add_e2_kernel(device const e2 *d_in [[buffer(0)]],
                                     device e2 *d_out [[buffer(1)]],
                                     constant int &num_items [[buffer(2)]],
                                     threadgroup e2 *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  e2 val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : e2::zero();

  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] = e2::add(shared[tid], shared[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0)
    d_out[bid] = shared[0];
}

// Reduce add e4
kernel void ab_reduce_add_e4_kernel(device const e4 *d_in [[buffer(0)]],
                                     device e4 *d_out [[buffer(1)]],
                                     constant int &num_items [[buffer(2)]],
                                     threadgroup e4 *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  e4 val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : e4::zero();

  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] = e4::add(shared[tid], shared[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0)
    d_out[bid] = shared[0];
}

// Reduce mul bf
kernel void ab_reduce_mul_bf_kernel(device const bf *d_in [[buffer(0)]],
                                     device bf *d_out [[buffer(1)]],
                                     constant int &num_items [[buffer(2)]],
                                     threadgroup bf *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  bf val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : bf::one();

  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] = bf::mul(shared[tid], shared[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0)
    d_out[bid] = shared[0];
}

// Reduce mul e4
kernel void ab_reduce_mul_e4_kernel(device const e4 *d_in [[buffer(0)]],
                                     device e4 *d_out [[buffer(1)]],
                                     constant int &num_items [[buffer(2)]],
                                     threadgroup e4 *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  e4 val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : e4::one();

  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] = e4::mul(shared[tid], shared[tid + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0)
    d_out[bid] = shared[0];
}

// Segmented reduce: reduce each segment (column) of a matrix independently.
// Each threadgroup handles one segment. Multiple dispatches may be needed for large segments.
kernel void ab_segmented_reduce_add_bf_kernel(device const bf *d_in [[buffer(0)]],
                                               device bf *d_out [[buffer(1)]],
                                               constant int &num_segments [[buffer(2)]],
                                               constant int &segment_length [[buffer(3)]],
                                               constant size_t &stride [[buffer(4)]],
                                               threadgroup bf *shared [[threadgroup(0)]],
                                               uint tid [[thread_index_in_threadgroup]],
                                               uint bid [[threadgroup_position_in_grid]],
                                               uint threads_per_group [[threads_per_threadgroup]]) {
  if ((int)bid >= num_segments) return;
  device const bf *segment = d_in + bid * stride;

  bf val = bf::zero();
  for (unsigned i = tid; i < (unsigned)segment_length; i += threads_per_group)
    val = bf::add(val, segment[i]);

  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (unsigned s = threads_per_group >> 1; s > 0; s >>= 1) {
    if (tid < s)
      shared[tid] = bf::add(shared[tid], shared[tid + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0)
    d_out[bid] = shared[0];
}

} // namespace ops_cub
} // namespace airbender
