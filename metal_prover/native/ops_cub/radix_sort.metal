#include <metal_stdlib>
using namespace metal;

namespace airbender {
namespace ops_cub {

typedef uint32_t u32;

// Radix sort implementation for Metal.
// Uses a 4-bit-per-pass approach (16 buckets per pass).
// Supports begin_bit/end_bit parameters for partial key sorting.
//
// Algorithm (per pass):
//   a. Histogram kernel: each threadgroup counts per-digit frequencies.
//   b. Host computes global prefix sums from histograms.
//   c. Scatter kernel: each element is placed at a STABLE output position.
//
// The scatter kernels use a stable rank computation:
//   for each element i (in block-local index order), its output position is
//   prefix_sums[bid * RADIX_SIZE + digit] + (count of earlier elements in this block with the same digit).
// This guarantees that equal-digit elements maintain their original relative order,
// which is required for multi-pass LSD radix sort to be correct.

constant constexpr unsigned RADIX_BITS = 4;
constant constexpr unsigned RADIX_SIZE = 1u << RADIX_BITS; // 16
constant constexpr unsigned SORT_BLOCK_SIZE = 256;

// Threadgroup memory layout for scatter kernels:
//   [0 .. RADIX_SIZE-1]            : per-digit output offsets  (RADIX_SIZE u32)
//   [RADIX_SIZE .. RADIX_SIZE+255] : per-element digits        (SORT_BLOCK_SIZE u32)
// Total: (RADIX_SIZE + SORT_BLOCK_SIZE) * 4 = 1088 bytes
constant constexpr unsigned SCATTER_SMEM_SIZE = RADIX_SIZE + SORT_BLOCK_SIZE;

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
    atomic_fetch_add_explicit((threadgroup atomic_uint *)&shared_hist[digit], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write histogram for this block
  if (tid < RADIX_SIZE)
    histograms[bid * RADIX_SIZE + tid] = shared_hist[tid];
}

// Scatter kernel: stable reorder of keys using prefix sums.
//
// For each element at block-local position `tid`, its output position is:
//   prefix_sums[bid * RADIX_SIZE + digit] + rank
// where rank = number of elements in [0, tid) with the same digit.
// This stable rank ensures the sort is correct for multi-pass LSD radix sort.
//
// Threadgroup memory: SCATTER_SMEM_SIZE u32 words = (RADIX_SIZE + SORT_BLOCK_SIZE) * 4 bytes
kernel void ab_radix_sort_scatter_kernel(device const u32 *keys_in [[buffer(0)]],
                                          device u32 *keys_out [[buffer(1)]],
                                          device const u32 *prefix_sums [[buffer(2)]],
                                          constant unsigned &num_items [[buffer(3)]],
                                          constant int &current_bit [[buffer(4)]],
                                          threadgroup u32 *smem [[threadgroup(0)]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint bid [[threadgroup_position_in_grid]]) {
  threadgroup u32 *shared_offsets = smem;               // [0 .. RADIX_SIZE)
  threadgroup u32 *shared_digits  = smem + RADIX_SIZE;  // [RADIX_SIZE .. RADIX_SIZE + SORT_BLOCK_SIZE)

  const unsigned start = bid * SORT_BLOCK_SIZE;
  const unsigned end   = min(start + SORT_BLOCK_SIZE, num_items);

  // Step 1: Load per-digit offsets from prefix_sums
  if (tid < RADIX_SIZE)
    shared_offsets[tid] = prefix_sums[bid * RADIX_SIZE + tid];

  // Step 2: Load each element's digit into shared memory (preserving element order)
  const unsigned i = start + tid;
  u32 my_key   = 0;
  u32 my_digit = 0;
  if (i < end) {
    my_key   = keys_in[i];
    my_digit = (my_key >> current_bit) & (RADIX_SIZE - 1);
    shared_digits[tid] = my_digit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Step 3: Compute stable rank and scatter.
  // rank = number of elements at positions [0, tid) with the same digit.
  // This ensures elements with equal digits maintain their original relative order.
  if (i < end) {
    u32 rank = 0;
    for (uint j = 0; j < tid; j++) {
      if (shared_digits[j] == my_digit) rank++;
    }
    const u32 pos = shared_offsets[my_digit] + rank;
    keys_out[pos] = my_key;
  }
}

// Key-value scatter kernel (stable, same approach as ab_radix_sort_scatter_kernel)
// Threadgroup memory: SCATTER_SMEM_SIZE u32 words
kernel void ab_radix_sort_scatter_pairs_kernel(device const u32 *keys_in [[buffer(0)]],
                                                device u32 *keys_out [[buffer(1)]],
                                                device const u32 *values_in [[buffer(2)]],
                                                device u32 *values_out [[buffer(3)]],
                                                device const u32 *prefix_sums [[buffer(4)]],
                                                constant unsigned &num_items [[buffer(5)]],
                                                constant int &current_bit [[buffer(6)]],
                                                threadgroup u32 *smem [[threadgroup(0)]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint bid [[threadgroup_position_in_grid]]) {
  threadgroup u32 *shared_offsets = smem;
  threadgroup u32 *shared_digits  = smem + RADIX_SIZE;

  const unsigned start = bid * SORT_BLOCK_SIZE;
  const unsigned end   = min(start + SORT_BLOCK_SIZE, num_items);

  if (tid < RADIX_SIZE)
    shared_offsets[tid] = prefix_sums[bid * RADIX_SIZE + tid];

  const unsigned i = start + tid;
  u32 my_key   = 0;
  u32 my_val   = 0;
  u32 my_digit = 0;
  if (i < end) {
    my_key   = keys_in[i];
    my_val   = values_in[i];
    my_digit = (my_key >> current_bit) & (RADIX_SIZE - 1);
    shared_digits[tid] = my_digit;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (i < end) {
    u32 rank = 0;
    for (uint j = 0; j < tid; j++) {
      if (shared_digits[j] == my_digit) rank++;
    }
    const u32 pos = shared_offsets[my_digit] + rank;
    keys_out[pos]   = my_key;
    values_out[pos] = my_val;
  }
}

} // namespace ops_cub
} // namespace airbender
