#include <metal_stdlib>
using namespace metal;

#include "field.metal"

using namespace airbender::field;

namespace airbender {
namespace blake2s {

typedef uint32_t u32;
typedef uint64_t u64;
typedef base_field bf;

#define LOG_WARP_SIZE 5
constexpr unsigned WARP_SIZE = 1 << LOG_WARP_SIZE;
constexpr unsigned WARP_MASK = WARP_SIZE - 1;

#define ROTR32(x, y) (((x) >> (y)) ^ ((x) << (32 - (y))))

#define G(a, b, c, d, x, y)    \
  v[a] = v[a] + v[b] + (x);    \
  v[d] = ROTR32(v[d] ^ v[a], 16); \
  v[c] = v[c] + v[d];          \
  v[b] = ROTR32(v[b] ^ v[c], 12); \
  v[a] = v[a] + v[b] + (y);    \
  v[d] = ROTR32(v[d] ^ v[a], 8);  \
  v[c] = v[c] + v[d];          \
  v[b] = ROTR32(v[b] ^ v[c], 7);

constexpr bool USE_REDUCED_ROUNDS = true;
constexpr unsigned FULL_ROUNDS = 10;
constexpr unsigned REDUCED_ROUNDS = 7;
constexpr unsigned ROUNDS = USE_REDUCED_ROUNDS ? REDUCED_ROUNDS : FULL_ROUNDS;
constexpr unsigned STATE_SIZE = 8;
constexpr unsigned BLOCK_SIZE = 16;
constexpr u32 IV_0_TWIST = 0x01010000 ^ 32;

constant u32 IV[STATE_SIZE] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

constant unsigned SIGMAS[10][BLOCK_SIZE] = {
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
  {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
  {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
  {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
  {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
  {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
  {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
  {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
};

DEVICE_FORCEINLINE void initialize(thread u32 state[STATE_SIZE]) {
  for (unsigned i = 0; i < STATE_SIZE; i++)
    state[i] = IV[i];
  state[0] ^= IV_0_TWIST;
}

template <bool IS_FINAL_BLOCK>
DEVICE_FORCEINLINE void compress(thread u32 state[STATE_SIZE], thread u32 &t, const thread u32 m[BLOCK_SIZE], const unsigned block_size) {
  u32 v[BLOCK_SIZE];
  for (unsigned i = 0; i < STATE_SIZE; i++) {
    v[i] = state[i];
    v[i + STATE_SIZE] = IV[i];
  }
  t += (IS_FINAL_BLOCK ? block_size : BLOCK_SIZE) * sizeof(u32);
  v[12] ^= t;
  if (IS_FINAL_BLOCK)
    v[14] ^= 0xffffffff;
  for (unsigned i = 0; i < ROUNDS; i++) {
    // Use constant SIGMAS array
    G(0, 4, 8, 12, m[SIGMAS[i][0]], m[SIGMAS[i][1]])
    G(1, 5, 9, 13, m[SIGMAS[i][2]], m[SIGMAS[i][3]])
    G(2, 6, 10, 14, m[SIGMAS[i][4]], m[SIGMAS[i][5]])
    G(3, 7, 11, 15, m[SIGMAS[i][6]], m[SIGMAS[i][7]])
    G(0, 5, 10, 15, m[SIGMAS[i][8]], m[SIGMAS[i][9]])
    G(1, 6, 11, 12, m[SIGMAS[i][10]], m[SIGMAS[i][11]])
    G(2, 7, 8, 13, m[SIGMAS[i][12]], m[SIGMAS[i][13]])
    G(3, 4, 9, 14, m[SIGMAS[i][14]], m[SIGMAS[i][15]])
  }
  for (unsigned i = 0; i < STATE_SIZE; ++i)
    state[i] ^= v[i] ^ v[i + STATE_SIZE];
}

kernel void ab_blake2s_leaves_kernel(device const bf *values [[buffer(0)]],
                                      device u32 *results [[buffer(1)]],
                                      constant unsigned &log_rows_count [[buffer(2)]],
                                      constant unsigned &cols_count [[buffer(3)]],
                                      constant unsigned &count [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  device const bf *vals = values + (gid << log_rows_count);
  device u32 *res = results + gid * STATE_SIZE;
  const unsigned row_mask = (1u << log_rows_count) - 1;

  u32 state[STATE_SIZE];
  u32 block[BLOCK_SIZE];
  initialize(state);
  u32 t = 0;
  const unsigned values_count = cols_count << log_rows_count;
  unsigned offset = 0;
  while (offset < values_count) {
    const unsigned remaining = values_count - offset;
    const bool is_final_block = remaining <= BLOCK_SIZE;
    for (unsigned i = 0; i < BLOCK_SIZE; i++, offset++) {
      const unsigned row = offset & row_mask;
      const unsigned col = offset >> log_rows_count;
      block[i] = col < cols_count ? bf::into_canonical_u32(vals[row + (col * count << log_rows_count)]) : 0;
    }
    if (is_final_block)
      compress<true>(state, t, block, remaining);
    else
      compress<false>(state, t, block, BLOCK_SIZE);
  }
  for (unsigned i = 0; i < STATE_SIZE; i++)
    res[i] = state[i];
}

kernel void ab_blake2s_nodes_kernel(device const u32 *values [[buffer(0)]],
                                     device u32 *results [[buffer(1)]],
                                     constant unsigned &count [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  device const u32 *vals = values + gid * BLOCK_SIZE;
  device u32 *res = results + gid * STATE_SIZE;
  u32 state[STATE_SIZE];
  u32 block[BLOCK_SIZE];
  initialize(state);
  u32 t = 0;
  for (unsigned i = 0; i < BLOCK_SIZE; i++)
    block[i] = vals[i];
  compress<true>(state, t, block, BLOCK_SIZE);
  for (unsigned i = 0; i < STATE_SIZE; i++)
    res[i] = state[i];
}

kernel void ab_gather_rows_kernel(device const unsigned *indexes [[buffer(0)]],
                                   constant unsigned &indexes_count [[buffer(1)]],
                                   constant bool &bit_reverse_indexes [[buffer(2)]],
                                   constant unsigned &log_rows_count [[buffer(3)]],
                                   device const bf *values_ptr [[buffer(4)]],
                                   constant size_t &values_stride [[buffer(5)]],
                                   device bf *results_ptr [[buffer(6)]],
                                   constant size_t &results_stride [[buffer(7)]],
                                   uint2 tid [[thread_position_in_grid]],
                                   uint2 tpg [[threads_per_grid]]) {
  // tid.x = column-within-leaf index, tid.y = leaf index
  const unsigned idx = tid.y;
  if (idx >= indexes_count)
    return;
  const unsigned i = indexes[idx];
  const unsigned index = bit_reverse_indexes ? reverse_bits(i) >> (32 - log_rows_count) : i;
  const unsigned blockDimX = tpg.x; // threads per row
  const unsigned src_row = index * blockDimX + tid.x;
  const unsigned dst_row = idx * blockDimX + tid.x;
  // col is blockIdx.y, handled via 3D dispatch
  // For simplicity, we use 2D grid: x=within-leaf, y=leaf
  const bf value = values_ptr[src_row]; // simplified: single column per dispatch
  const bf result(bf::into_canonical_u32(value));
  results_ptr[dst_row] = result;
}

kernel void ab_gather_merkle_paths_kernel(device const unsigned *indexes [[buffer(0)]],
                                           constant unsigned &indexes_count [[buffer(1)]],
                                           device const u32 *values [[buffer(2)]],
                                           constant unsigned &log_leaves_count [[buffer(3)]],
                                           device u32 *results [[buffer(4)]],
                                           uint2 tid [[thread_position_in_grid]]) {
  const unsigned idx = tid.y;
  if (idx >= indexes_count)
    return;
  const unsigned leaf_index = indexes[idx];
  const unsigned layer_index = tid.x; // one layer per x-thread
  const unsigned layer_offset = ((1u << (log_leaves_count + 1)) - (1u << (log_leaves_count + 1 - layer_index))) * STATE_SIZE;
  const unsigned hash_offset = ((leaf_index >> layer_index) ^ 1) * STATE_SIZE;
  // element_offset would be a third dimension; simplified for demonstration
  for (unsigned element_offset = 0; element_offset < STATE_SIZE; element_offset++) {
    const unsigned src_index = layer_offset + hash_offset + element_offset;
    const unsigned dst_index = layer_index * indexes_count * STATE_SIZE + idx * STATE_SIZE + element_offset;
    results[dst_index] = values[src_index];
  }
}

kernel void ab_gather_rows_and_merkle_paths_kernel(
    device const unsigned *indexes [[buffer(0)]],
    constant unsigned &indexes_count [[buffer(1)]],
    constant bool &bit_reverse_indexes [[buffer(2)]],
    device const bf *values [[buffer(3)]],
    constant unsigned &log_rows_per_leaf [[buffer(4)]],
    constant unsigned &cols_count [[buffer(5)]],
    constant unsigned &log_total_leaves_count [[buffer(6)]],
    device bf *leaf_values_ptr [[buffer(7)]],
    constant size_t &leaf_values_stride [[buffer(8)]],
    device const u32 *tree_bottom [[buffer(9)]],
    constant unsigned &layers_count [[buffer(10)]],
    device u32 *merkle_paths [[buffer(11)]],
    uint lane_idx [[thread_index_in_simdgroup]],
    uint bid [[threadgroup_position_in_grid]]) {

  const unsigned idx = bid;
  const unsigned index_warp = indexes[idx];
  const unsigned index_lane = (index_warp & ~WARP_MASK) | lane_idx;
  const bool is_output_lane = index_warp == index_lane;
  const unsigned leaf_index = bit_reverse_indexes ? reverse_bits(index_lane) >> (32 - log_total_leaves_count) : index_lane;
  device const bf *my_values = values + (leaf_index << log_rows_per_leaf);
  device u32 *my_merkle_paths = merkle_paths + idx * STATE_SIZE;
  const unsigned row_mask = (1u << log_rows_per_leaf) - 1;

  u32 state[STATE_SIZE];
  u32 block[BLOCK_SIZE];
  initialize(state);
  u32 t = 0;
  const unsigned values_count = cols_count << log_rows_per_leaf;
  unsigned offset = 0;
  while (offset < values_count) {
    const unsigned remaining = values_count - offset;
    const bool is_final_block = remaining <= BLOCK_SIZE;
    for (unsigned i = 0; i < BLOCK_SIZE; i++, offset++) {
      const unsigned row = offset & row_mask;
      const unsigned col = offset >> log_rows_per_leaf;
      device const bf *address = my_values + row + (col << (log_rows_per_leaf + log_total_leaves_count));
      const u32 value = col < cols_count ? bf::into_canonical_u32(*address) : 0;
      block[i] = value;
      if (offset < values_count && is_output_lane) {
        // Write leaf value: leaf_values_ptr[idx + col * leaf_values_stride] = bf(value)
        leaf_values_ptr[idx + col * leaf_values_stride] = bf(value);
      }
    }
    if (is_final_block)
      compress<true>(state, t, block, remaining);
    else
      compress<false>(state, t, block, BLOCK_SIZE);
  }

  // Warp-level Merkle tree building using simd_shuffle_xor
  for (unsigned layer = 0; layer < LOG_WARP_SIZE; layer++) {
    u32 other_state[STATE_SIZE];
    const bool take_other_first = lane_idx >> layer & 1;
    for (unsigned i = 0; i < STATE_SIZE; i++) {
      other_state[i] = simd_shuffle_xor(state[i], static_cast<ushort>(1u << layer));
      if (is_output_lane)
        my_merkle_paths[i] = other_state[i];
      if (take_other_first) {
        block[i] = other_state[i];
        block[i + STATE_SIZE] = state[i];
      } else {
        block[i] = state[i];
        block[i + STATE_SIZE] = other_state[i];
      }
    }
    initialize(state);
    t = 0;
    compress<true>(state, t, block, BLOCK_SIZE);
    my_merkle_paths += indexes_count * STATE_SIZE;
  }

  if (lane_idx >= STATE_SIZE)
    return;
  unsigned digest_index = index_warp >> LOG_WARP_SIZE;
  unsigned log_digests_count = log_total_leaves_count - LOG_WARP_SIZE;
  device const u32 *tree_ptr = tree_bottom + lane_idx;
  device u32 *path_ptr = my_merkle_paths + lane_idx;
  for (unsigned layer = LOG_WARP_SIZE; layer < layers_count; layer++) {
    const unsigned other_index = digest_index ^ 1;
    *path_ptr = *(tree_ptr + other_index * STATE_SIZE);
    digest_index >>= 1;
    tree_ptr += (1u << log_digests_count) * STATE_SIZE;
    log_digests_count--;
    path_ptr += indexes_count * STATE_SIZE;
  }
}

kernel void ab_blake2s_pow_kernel(device const u64 *seed [[buffer(0)]],
                                   constant u32 &bits_count [[buffer(1)]],
                                   constant u64 &max_nonce [[buffer(2)]],
                                   device volatile atomic_uint *result_low [[buffer(3)]],
                                   device volatile atomic_uint *result_high [[buffer(4)]],
                                   uint gid [[thread_position_in_grid]],
                                   uint grid_size [[threads_per_grid]]) {
  const u32 digest_mask = 0xffffffff << (32 - bits_count);
  alignas(8) u32 m_u32[BLOCK_SIZE] = {};
  // Copy seed (32 bytes = 4 x u64)
  device const u32 *seed_u32 = reinterpret_cast<device const u32 *>(seed);
  for (unsigned i = 0; i < 8; i++)
    m_u32[i] = seed_u32[i];

  for (u64 nonce = gid; nonce < max_nonce; nonce += grid_size) {
    // Write nonce at position 8-9 (u64 at offset 4 in u64 terms = offset 8 in u32 terms)
    m_u32[8] = static_cast<u32>(nonce);
    m_u32[9] = static_cast<u32>(nonce >> 32);
    u32 state[STATE_SIZE];
    initialize(state);
    u32 t = 0;
    compress<true>(state, t, m_u32, STATE_SIZE + 2);
    if (!(state[0] & digest_mask)) {
      // Attempt to store result using atomic CAS
      u32 nonce_low = static_cast<u32>(nonce);
      u32 nonce_high = static_cast<u32>(nonce >> 32);
      u32 expected_high = 0xFFFFFFFF;
      atomic_compare_exchange_weak_explicit(result_high, &expected_high, nonce_high, memory_order_relaxed, memory_order_relaxed);
      if (expected_high == 0xFFFFFFFF) {
        atomic_store_explicit(result_low, nonce_low, memory_order_relaxed);
      }
      atomic_thread_fence(mem_flags::mem_device);
    }
    // Check if someone already found a result
    u32 current_high = atomic_load_explicit(result_high, memory_order_relaxed);
    if (current_high != 0xFFFFFFFF)
      return;
  }
}

} // namespace blake2s
} // namespace airbender
