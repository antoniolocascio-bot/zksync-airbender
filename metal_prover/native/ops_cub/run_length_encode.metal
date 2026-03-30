#include <metal_stdlib>
using namespace metal;

namespace airbender {
namespace ops_cub {

typedef uint32_t u32;

// Run-length encoding implementation for Metal.
// Algorithm:
// 1. Boundary detection: mark positions where value changes
// 2. Prefix sum on boundary flags to get output positions
// 3. Scatter unique values and counts to output

// Step 1: Detect run boundaries
kernel void ab_rle_detect_boundaries_kernel(device const u32 *d_in [[buffer(0)]],
                                             device u32 *boundaries [[buffer(1)]],
                                             constant int &num_items [[buffer(2)]],
                                             uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= num_items) return;
  // A boundary exists at position 0, or wherever d_in[i] != d_in[i-1]
  if (gid == 0 || d_in[gid] != d_in[gid - 1])
    boundaries[gid] = 1;
  else
    boundaries[gid] = 0;
}

// Step 3: Scatter unique values and compute counts
// After prefix sum on boundaries, boundaries[i] contains the output index for run starting at i.
kernel void ab_rle_scatter_kernel(device const u32 *d_in [[buffer(0)]],
                                   device const u32 *prefix_sum [[buffer(1)]],
                                   device u32 *d_unique_out [[buffer(2)]],
                                   device u32 *d_counts_out [[buffer(3)]],
                                   device u32 *d_num_runs_out [[buffer(4)]],
                                   constant int &num_items [[buffer(5)]],
                                   uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= num_items) return;

  const bool is_boundary = (gid == 0) || (d_in[gid] != d_in[gid - 1]);
  if (!is_boundary) return;

  // Output index for this run (exclusive prefix sum: index = prefix_sum[gid] - 1 if inclusive, or prefix_sum[gid] if exclusive)
  const u32 out_idx = prefix_sum[gid] - 1; // assuming inclusive prefix sum

  d_unique_out[out_idx] = d_in[gid];

  // Count = start of next run - start of this run
  // Find end of this run
  u32 count;
  if ((int)gid == num_items - 1) {
    // Last element: total runs is prefix_sum value at last element
    count = num_items - gid;
    // Also write total number of runs
    d_num_runs_out[0] = prefix_sum[num_items - 1];
  } else {
    // Find where next run starts
    // If prefix_sum[gid+1] > prefix_sum[gid], then gid+1 is a boundary
    // We need to find the run length. Since we only process boundaries,
    // we need to look ahead to the next boundary.
    // Simple approach: scan forward (but this is O(n) worst case per thread)
    // Better: use the difference between consecutive boundary positions.
    u32 next_boundary = gid + 1;
    while ((int)next_boundary < num_items && d_in[next_boundary] == d_in[gid])
      next_boundary++;
    count = next_boundary - gid;
  }
  d_counts_out[out_idx] = count;

  // Write num_runs from the last boundary
  if ((int)gid == num_items - 1 || (is_boundary && gid == 0)) {
    // num_runs will be written by the last element's handler above
  }
}

// Alternative simpler single-pass RLE for small inputs
// Each thread checks if it's a boundary and writes directly using atomics
kernel void ab_rle_simple_kernel(device const u32 *d_in [[buffer(0)]],
                                  device u32 *d_unique_out [[buffer(1)]],
                                  device u32 *d_counts_out [[buffer(2)]],
                                  device atomic_uint *d_num_runs_out [[buffer(3)]],
                                  constant int &num_items [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= num_items) return;

  const bool is_boundary = (gid == 0) || (d_in[gid] != d_in[gid - 1]);
  if (!is_boundary) return;

  // Atomically get output position
  const u32 out_idx = atomic_fetch_add_explicit(d_num_runs_out, 1u, memory_order_relaxed);
  d_unique_out[out_idx] = d_in[gid];

  // Count run length
  u32 next = gid + 1;
  while ((int)next < num_items && d_in[next] == d_in[gid])
    next++;
  d_counts_out[out_idx] = next - gid;
}

} // namespace ops_cub
} // namespace airbender
