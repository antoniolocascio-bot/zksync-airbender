#include <metal_stdlib>
using namespace metal;

namespace airbender {
namespace ops_cub {

typedef uint32_t u32;

// Radix sort implementation for Metal.
// Uses a 4-bit-per-pass approach (16 buckets per pass).
// Supports begin_bit/end_bit parameters for partial key sorting.
//
// Algorithm:
// 1. For each 4-bit pass:
//    a. Each threadgroup counts occurrences of each digit (histogram)
//    b. Global prefix sum on histograms
//    c. Scatter keys (and values) to output based on prefix sums

constexpr unsigned RADIX_BITS = 4;
constexpr unsigned RADIX_SIZE = 1u << RADIX_BITS; // 16
constexpr unsigned SORT_BLOCK_SIZE = 256;

// Local histogram kernel: each threadgroup counts digit frequencies
kernel void ab_radix_sort_histogram_kernel(device const u32 *keys_in [[buffer(0)]],
                                            device u32 *histograms [[buffer(1)]],
                                            constant unsigned &num_items [[buffer(2)]],
                                            constant int &current_bit [[buffer(3)]],
                                            threadgroup u32 *shared_hist [[threadgroup(0)]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint bid [[threadgroup_position_in_grid]],
                                            uint threads_per_group [[threads_per_threadgroup]],
                                            uint num_threadgroups [[threadgroups_per_grid]]) {
  // Initialize shared histogram to zero
  if (tid < RADIX_SIZE)
    shared_hist[tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Count digits for this threadgroup's chunk
  const unsigned start = bid * SORT_BLOCK_SIZE;
  const unsigned end = min(start + SORT_BLOCK_SIZE, num_items);
  for (unsigned i = start + tid; i < end; i += threads_per_group) {
    const u32 key = keys_in[i];
    const u32 digit = (key >> current_bit) & (RADIX_SIZE - 1);
    // Atomic increment in threadgroup memory
    atomic_fetch_add_explicit((threadgroup atomic_uint *)&shared_hist[digit], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write histogram for this block
  if (tid < RADIX_SIZE)
    histograms[bid * RADIX_SIZE + tid] = shared_hist[tid];
}

// Scatter kernel: reorder keys based on prefix sums
kernel void ab_radix_sort_scatter_kernel(device const u32 *keys_in [[buffer(0)]],
                                          device u32 *keys_out [[buffer(1)]],
                                          device const u32 *prefix_sums [[buffer(2)]],
                                          constant unsigned &num_items [[buffer(3)]],
                                          constant int &current_bit [[buffer(4)]],
                                          threadgroup u32 *shared_offsets [[threadgroup(0)]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint bid [[threadgroup_position_in_grid]],
                                          uint threads_per_group [[threads_per_threadgroup]]) {
  // Load prefix sums for this block into shared memory
  if (tid < RADIX_SIZE)
    shared_offsets[tid] = prefix_sums[bid * RADIX_SIZE + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const unsigned start = bid * SORT_BLOCK_SIZE;
  const unsigned end = min(start + SORT_BLOCK_SIZE, num_items);

  // Each thread processes one element at a time
  for (unsigned i = start + tid; i < end; i += threads_per_group) {
    const u32 key = keys_in[i];
    const u32 digit = (key >> current_bit) & (RADIX_SIZE - 1);
    // Atomic increment to get unique output position
    const u32 pos = atomic_fetch_add_explicit((threadgroup atomic_uint *)&shared_offsets[digit], 1u, memory_order_relaxed);
    keys_out[pos] = key;
  }
}

// Key-value scatter kernel
kernel void ab_radix_sort_scatter_pairs_kernel(device const u32 *keys_in [[buffer(0)]],
                                                device u32 *keys_out [[buffer(1)]],
                                                device const u32 *values_in [[buffer(2)]],
                                                device u32 *values_out [[buffer(3)]],
                                                device const u32 *prefix_sums [[buffer(4)]],
                                                constant unsigned &num_items [[buffer(5)]],
                                                constant int &current_bit [[buffer(6)]],
                                                threadgroup u32 *shared_offsets [[threadgroup(0)]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint bid [[threadgroup_position_in_grid]],
                                                uint threads_per_group [[threads_per_threadgroup]]) {
  if (tid < RADIX_SIZE)
    shared_offsets[tid] = prefix_sums[bid * RADIX_SIZE + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const unsigned start = bid * SORT_BLOCK_SIZE;
  const unsigned end = min(start + SORT_BLOCK_SIZE, num_items);

  for (unsigned i = start + tid; i < end; i += threads_per_group) {
    const u32 key = keys_in[i];
    const u32 digit = (key >> current_bit) & (RADIX_SIZE - 1);
    const u32 pos = atomic_fetch_add_explicit((threadgroup atomic_uint *)&shared_offsets[digit], 1u, memory_order_relaxed);
    keys_out[pos] = key;
    values_out[pos] = values_in[i];
  }
}

} // namespace ops_cub
} // namespace airbender
