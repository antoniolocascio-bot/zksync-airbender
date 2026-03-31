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

// Binary operations for scan (pass by value to avoid address space qualifier issues)
template <typename T> struct add_op {
  DEVICE_FORCEINLINE T operator()(T a, T b) const { return T::add(a, b); }
  static DEVICE_FORCEINLINE T identity() { return T::zero(); }
};
template <> struct add_op<u32> {
  DEVICE_FORCEINLINE u32 operator()(u32 a, u32 b) const { return a + b; }
  static DEVICE_FORCEINLINE u32 identity() { return 0; }
};
template <typename T> struct mul_op {
  DEVICE_FORCEINLINE T operator()(T a, T b) const { return T::mul(a, b); }
  static DEVICE_FORCEINLINE T identity() { return T::one(); }
};
template <> struct mul_op<u32> {
  DEVICE_FORCEINLINE u32 operator()(u32 a, u32 b) const { return a * b; }
  static DEVICE_FORCEINLINE u32 identity() { return 1; }
};

// Two-pass parallel scan implementation:
// Pass 1: Each threadgroup computes its local scan and writes the threadgroup total to partial_sums.
// Pass 2: A fixup kernel adds the scanned partial sums to each threadgroup's results.

constant constexpr unsigned SCAN_BLOCK_SIZE = 256;

// Per-threadgroup inclusive scan using shared memory (Blelloch-style up-sweep/down-sweep)
template <typename T, typename Op>
DEVICE_FORCEINLINE T threadgroup_inclusive_scan(thread T &val, threadgroup T *shared, const unsigned tid, const unsigned count) {
  Op op;
  shared[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Up-sweep (reduce)
  for (unsigned stride = 1; stride < count; stride <<= 1) {
    unsigned idx = (tid + 1) * (stride << 1) - 1;
    if (idx < count)
      shared[idx] = op(shared[idx - stride], shared[idx]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Down-sweep for inclusive scan
  for (unsigned stride = count >> 1; stride > 0; stride >>= 1) {
    unsigned idx = (tid + 1) * (stride << 1) - 1;
    if (idx + stride < count)
      shared[idx + stride] = op(shared[idx], shared[idx + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  val = shared[tid];
  T total = shared[count - 1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// Pass 1: Inclusive scan within each threadgroup, write threadgroup totals to partial_sums
kernel void ab_scan_i_add_bf_pass1(device const bf *d_in [[buffer(0)]],
                                     device bf *d_out [[buffer(1)]],
                                     device bf *partial_sums [[buffer(2)]],
                                     constant int &num_items [[buffer(3)]],
                                     threadgroup bf *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  bf val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : bf::zero();

  bf total = threadgroup_inclusive_scan<bf, add_op<bf>>(val, shared, tid, threads_per_group);

  if (global_idx < (unsigned)num_items)
    d_out[global_idx] = val;

  if (tid == threads_per_group - 1)
    partial_sums[bid] = total;
}

// Pass 2: Fixup - add scanned partial sums to each threadgroup's results
kernel void ab_scan_i_add_bf_pass2(device bf *d_out [[buffer(0)]],
                                     device const bf *scanned_partials [[buffer(1)]],
                                     constant int &num_items [[buffer(2)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  if (bid == 0) return; // First block doesn't need fixup
  const unsigned global_idx = bid * threads_per_group + tid;
  if (global_idx >= (unsigned)num_items) return;
  d_out[global_idx] = bf::add(d_out[global_idx], scanned_partials[bid - 1]);
}

// Exclusive scan for bf with add
kernel void ab_scan_e_add_bf_pass1(device const bf *d_in [[buffer(0)]],
                                     device bf *d_out [[buffer(1)]],
                                     device bf *partial_sums [[buffer(2)]],
                                     constant int &num_items [[buffer(3)]],
                                     threadgroup bf *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  bf val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : bf::zero();

  bf total = threadgroup_inclusive_scan<bf, add_op<bf>>(val, shared, tid, threads_per_group);

  // Convert inclusive to exclusive: shift right and insert identity
  if (global_idx < (unsigned)num_items) {
    bf exclusive_val = (tid == 0) ? bf::zero() : shared[tid - 1];
    d_out[global_idx] = exclusive_val;
  }

  if (tid == threads_per_group - 1)
    partial_sums[bid] = total;
}

// Inclusive scan for e4 with add
kernel void ab_scan_i_add_e4_pass1(device const e4 *d_in [[buffer(0)]],
                                     device e4 *d_out [[buffer(1)]],
                                     device e4 *partial_sums [[buffer(2)]],
                                     constant int &num_items [[buffer(3)]],
                                     threadgroup e4 *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  e4 val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : e4::zero();

  e4 total = threadgroup_inclusive_scan<e4, add_op<e4>>(val, shared, tid, threads_per_group);

  if (global_idx < (unsigned)num_items)
    d_out[global_idx] = val;

  if (tid == threads_per_group - 1)
    partial_sums[bid] = total;
}

// Inclusive scan for e4 with mul
kernel void ab_scan_i_mul_e4_pass1(device const e4 *d_in [[buffer(0)]],
                                     device e4 *d_out [[buffer(1)]],
                                     device e4 *partial_sums [[buffer(2)]],
                                     constant int &num_items [[buffer(3)]],
                                     threadgroup e4 *shared [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint bid [[threadgroup_position_in_grid]],
                                     uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  e4 val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : e4::one();

  e4 total = threadgroup_inclusive_scan<e4, mul_op<e4>>(val, shared, tid, threads_per_group);

  if (global_idx < (unsigned)num_items)
    d_out[global_idx] = val;

  if (tid == threads_per_group - 1)
    partial_sums[bid] = total;
}

// Inclusive scan for u32 with add
kernel void ab_scan_i_add_u32_pass1(device const u32 *d_in [[buffer(0)]],
                                      device u32 *d_out [[buffer(1)]],
                                      device u32 *partial_sums [[buffer(2)]],
                                      constant int &num_items [[buffer(3)]],
                                      threadgroup u32 *shared [[threadgroup(0)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint bid [[threadgroup_position_in_grid]],
                                      uint threads_per_group [[threads_per_threadgroup]]) {
  const unsigned global_idx = bid * threads_per_group + tid;
  u32 val = (global_idx < (unsigned)num_items) ? d_in[global_idx] : 0;

  u32 total = threadgroup_inclusive_scan<u32, add_op<u32>>(val, shared, tid, threads_per_group);

  if (global_idx < (unsigned)num_items)
    d_out[global_idx] = val;

  if (tid == threads_per_group - 1)
    partial_sums[bid] = total;
}

} // namespace ops_cub
} // namespace airbender
