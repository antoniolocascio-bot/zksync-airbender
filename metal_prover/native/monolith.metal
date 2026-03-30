#include <metal_stdlib>
using namespace metal;

#include "field.metal"

using namespace airbender::field;

namespace airbender {
namespace monolith {

typedef base_field bf;

constexpr unsigned WARP_SIZE = 32;
constexpr unsigned CAPACITY = 8;
constexpr unsigned RATE = 8;
constexpr unsigned WIDTH = CAPACITY + RATE;
constexpr unsigned NUM_ROUNDS = 6;
constexpr unsigned NUM_FULL_ROUNDS = NUM_ROUNDS - 1;
constexpr unsigned NUM_BARS = 8;

constant uint32_t ROUND_CONSTANTS[NUM_FULL_ROUNDS][WIDTH] = {
    {1821280327, 1805192324, 127749067, 534494027, 504066389, 661859220, 1964605566, 11087311, 1178584041, 412585466, 2078905810, 549234502, 1181028407,
     363220519, 1649192353, 895839514},
    {939676630, 132824540, 1081150345, 1901266162, 1248854474, 722216947, 711899879, 991065584, 872971327, 1747874412, 889258434, 857014393, 1145792277,
     329607215, 1069482641, 1809464251},
    {1792923486, 1071073386, 2086334655, 615259270, 1680936759, 2069228098, 679754665, 598972355, 1448263353, 2102254560, 1676515281, 1529495635, 981915006,
     436108429, 1959227325, 1710180674},
    {814766386, 746021429, 758709057, 1777861169, 1875425297, 1630916709, 180204592, 1301124329, 307222363, 297236795, 866482358, 1784330946, 1841790988,
     1855089478, 2122902104, 1522878966},
    {1132611924, 1823267038, 539457094, 934064219, 561891167, 1325624939, 1683493283, 1582152536, 851185378, 1187215684, 1520269176, 801897118, 741765053,
     1300119213, 1960664069, 1633755961},
};

constant unsigned MDS_MATRIX_INDEXES[WIDTH + 1] = {0, 4, 10, 12, 7, 14, 8, 13, 11, 1, 6, 15, 2, 3, 5, 9, 15};
constant unsigned MDS_MATRIX_SHIFTS[WIDTH + 1] = {4, 2, 4, 2, 4, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0};

// S-box lookup tables stored in threadgroup memory
DEVICE_FORCEINLINE uint8_t rotl_8(const uint8_t x, const unsigned shift) { return x << shift | x >> (8 - shift); }
DEVICE_FORCEINLINE uint8_t rotl_7(const uint8_t x, const unsigned shift) { return (x << shift | x >> (7 - shift)) & 0x7F; }
DEVICE_FORCEINLINE uint8_t s_box_8(const uint8_t x) { return rotl_8(x ^ ~rotl_8(x, 1) & rotl_8(x, 2) & rotl_8(x, 3), 1); }
DEVICE_FORCEINLINE uint8_t s_box_7(const uint8_t x) { return rotl_7(x ^ ~rotl_7(x, 1) & rotl_7(x, 2), 1); }

DEVICE_FORCEINLINE void initialize_lookup(threadgroup uint8_t *bar_lookup, const unsigned tid, const unsigned block_size) {
  for (unsigned i = tid; i < 1u << 8; i += block_size)
    bar_lookup[i] = s_box_8(i);
  for (unsigned i = tid; i < 1u << 7; i += block_size)
    bar_lookup[(1u << 8) + i] = s_box_7(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

DEVICE_FORCEINLINE uint32_t bar(uint32_t limb, const threadgroup uint8_t *bar_lookup) {
  uint32_t result;
  const uint8_t b0 = bar_lookup[limb & 0xFF];
  const uint8_t b1 = bar_lookup[(limb >> 8) & 0xFF];
  const uint8_t b2 = bar_lookup[(limb >> 16) & 0xFF];
  const uint8_t b3 = bar_lookup[256 + ((limb >> 24) & 0x7F)];
  result = static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) | (static_cast<uint32_t>(b2) << 16) | (static_cast<uint32_t>(b3) << 24);
  return result;
}

DEVICE_FORCEINLINE void bars_st(thread bf state[WIDTH], const threadgroup uint8_t *bar_lookup) {
  for (unsigned i = 0; i < NUM_BARS; i++)
    state[i] = bf(bar(state[i].limb, bar_lookup));
}

DEVICE_FORCEINLINE void bricks_st(thread bf state[WIDTH]) {
  for (unsigned i = WIDTH - 1; i > 0; i--)
    state[i] = bf::add(state[i], bf::sqr(state[i - 1]));
}

// Multi-threaded versions using SIMD shuffles
DEVICE_FORCEINLINE void bars_mt(thread bf &state, const unsigned tid, const threadgroup uint8_t *bar_lookup) {
  if (tid < NUM_BARS)
    state = bf(bar(state.limb, bar_lookup));
}

DEVICE_FORCEINLINE void bricks_mt(thread bf &state, const unsigned tid) {
  const uint32_t previous = simd_shuffle_up(state.limb, 1);
  if (tid > 0)
    state = bf::add(state, bf::sqr(bf(previous)));
}

template <unsigned ROUND>
DEVICE_FORCEINLINE void concrete_shl_st(thread bf state[WIDTH]) {
  bf result[WIDTH];
  for (unsigned row = 0; row < WIDTH; row++) {
    uint64_t acc = 0;
    for (unsigned i = 0; i < WIDTH + 1; i++) {
      const unsigned index = MDS_MATRIX_INDEXES[i];
      const unsigned col = (index + row) % WIDTH;
      const uint32_t value = state[col].limb;
      acc = i ? acc + value : value;
      if (const unsigned shift = MDS_MATRIX_SHIFTS[i])
        acc <<= shift;
    }
    acc <<= 2;
    if (ROUND != 0 && ROUND < NUM_ROUNDS)
      acc += ROUND_CONSTANTS[ROUND - 1][row];
    result[row] = bf::from_u62_max_minus_one(acc);
  }
  for (unsigned i = 0; i < WIDTH; i++)
    state[i] = result[i];
}

template <unsigned ROUND>
DEVICE_FORCEINLINE void concrete_shl_mt(thread bf &state, const unsigned tid) {
  uint64_t acc = 0;
  for (unsigned i = 0; i < WIDTH + 1; i++) {
    const unsigned index = MDS_MATRIX_INDEXES[i];
    const int src_lane = static_cast<int>((index + tid) % WIDTH);
    const uint32_t value = simd_shuffle(state.limb, static_cast<ushort>(src_lane));
    acc = i ? acc + value : value;
    if (const unsigned shift = MDS_MATRIX_SHIFTS[i])
      acc <<= shift;
  }
  acc <<= 2;
  if (ROUND != 0 && ROUND < NUM_ROUNDS)
    acc += ROUND_CONSTANTS[ROUND - 1][tid];
  state = bf::from_u62_max_minus_one(acc);
}

template <unsigned ROUND> DEVICE_FORCEINLINE void round_st(thread bf state[WIDTH], const threadgroup uint8_t *bar_lookup) {
  if (ROUND != 0) {
    bars_st(state, bar_lookup);
    bricks_st(state);
  }
  concrete_shl_st<ROUND>(state);
}

template <unsigned ROUND> DEVICE_FORCEINLINE void round_mt(thread bf &state, const unsigned tid, const threadgroup uint8_t *bar_lookup) {
  if (ROUND != 0) {
    bars_mt(state, tid, bar_lookup);
    bricks_mt(state, tid);
  }
  concrete_shl_mt<ROUND>(state, tid);
}

DEVICE_FORCEINLINE void permutation_st(thread bf state[WIDTH], const threadgroup uint8_t *bar_lookup) {
  round_st<0>(state, bar_lookup);
  round_st<1>(state, bar_lookup);
  round_st<2>(state, bar_lookup);
  round_st<3>(state, bar_lookup);
  round_st<4>(state, bar_lookup);
  round_st<5>(state, bar_lookup);
  round_st<6>(state, bar_lookup);
}

DEVICE_FORCEINLINE void permutation_mt(thread bf &state, const unsigned tid, const threadgroup uint8_t *bar_lookup) {
  round_mt<0>(state, tid, bar_lookup);
  round_mt<1>(state, tid, bar_lookup);
  round_mt<2>(state, tid, bar_lookup);
  round_mt<3>(state, tid, bar_lookup);
  round_mt<4>(state, tid, bar_lookup);
  round_mt<5>(state, tid, bar_lookup);
  round_mt<6>(state, tid, bar_lookup);
}

// Single-threaded leaves kernel
kernel void ab_monolith_leaves_st_kernel(device const bf *values [[buffer(0)]],
                                          device bf *results [[buffer(1)]],
                                          constant unsigned &log_rows_count [[buffer(2)]],
                                          constant unsigned &cols_count [[buffer(3)]],
                                          constant unsigned &count [[buffer(4)]],
                                          threadgroup uint8_t *bar_lookup [[threadgroup(0)]],
                                          uint gid [[thread_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint tpg [[threads_per_threadgroup]])
  [[max_total_threads_per_threadgroup(128)]] {
  initialize_lookup(bar_lookup, tid, tpg);
  if (gid >= count) return;

  device const bf *vals = values + (gid << log_rows_count);
  device bf *res = results + gid * CAPACITY;
  const unsigned row_mask = (1u << log_rows_count) - 1;

  bf state[WIDTH];
  unsigned offset = 0;
  for (unsigned i = 0; i < WIDTH; i++, offset++) {
    const unsigned row = offset & row_mask;
    const unsigned col = offset >> log_rows_count;
    state[i] = col < cols_count ? vals[row + ((col * count) << log_rows_count)] : bf::zero();
  }
  permutation_st(state, bar_lookup);

  while (offset < cols_count << log_rows_count) {
    for (unsigned i = 0; i < RATE; i++, offset++) {
      const unsigned row = offset & row_mask;
      const unsigned col = offset >> log_rows_count;
      state[i + CAPACITY] = col < cols_count ? vals[row + ((col * count) << log_rows_count)] : bf::zero();
    }
    permutation_st(state, bar_lookup);
  }

  for (unsigned i = 0; i < CAPACITY; i++)
    res[i] = state[i];
}

// Single-threaded nodes kernel
kernel void ab_monolith_nodes_st_kernel(device const bf *values [[buffer(0)]],
                                         device bf *results [[buffer(1)]],
                                         constant unsigned &count [[buffer(2)]],
                                         threadgroup uint8_t *bar_lookup [[threadgroup(0)]],
                                         uint gid [[thread_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint tpg [[threads_per_threadgroup]])
  [[max_total_threads_per_threadgroup(128)]] {
  initialize_lookup(bar_lookup, tid, tpg);
  if (gid >= count) return;

  device const bf *vals = values + gid * WIDTH;
  device bf *res = results + gid * CAPACITY;
  bf state[WIDTH];
  for (unsigned i = 0; i < WIDTH; i++)
    state[i] = vals[i];
  permutation_st(state, bar_lookup);
  for (unsigned i = 0; i < CAPACITY; i++)
    res[i] = state[i];
}

// Multi-threaded (SIMD-level) nodes kernel
// Each SIMD group of WIDTH threads processes one hash
kernel void ab_monolith_nodes_mt_kernel(device const bf *values [[buffer(0)]],
                                         device bf *results [[buffer(1)]],
                                         constant unsigned &count [[buffer(2)]],
                                         threadgroup uint8_t *bar_lookup [[threadgroup(0)]],
                                         uint simd_lane [[thread_index_in_simdgroup]],
                                         uint simd_group [[simdgroup_index_in_threadgroup]],
                                         uint bid [[threadgroup_position_in_grid]],
                                         uint threads_per_group [[threads_per_threadgroup]])
  [[max_total_threads_per_threadgroup(128)]] {
  const unsigned groups_per_block = threads_per_group / WARP_SIZE;
  // Initialize bar lookup cooperatively
  const unsigned flat_tid = simd_lane + simd_group * WARP_SIZE;
  initialize_lookup(bar_lookup, flat_tid, threads_per_group);

  // 2 hashes per warp (WARP_SIZE / WIDTH = 2)
  const unsigned hashes_per_warp = WARP_SIZE / WIDTH;
  const unsigned hash_in_warp = simd_lane / WIDTH;
  const unsigned tid_in_hash = simd_lane % WIDTH;

  const unsigned gid = bid * groups_per_block * hashes_per_warp + simd_group * hashes_per_warp + hash_in_warp;
  if (gid >= count) return;

  bf state = values[gid * WIDTH + tid_in_hash];
  permutation_mt(state, tid_in_hash, bar_lookup);
  if (tid_in_hash < CAPACITY)
    results[gid * CAPACITY + tid_in_hash] = state;
}

// Gather rows kernel for Monolith
kernel void ab_monolith_gather_rows_kernel(device const unsigned *indexes [[buffer(0)]],
                                            constant unsigned &indexes_count [[buffer(1)]],
                                            device const bf *values_ptr [[buffer(2)]],
                                            constant size_t &values_stride [[buffer(3)]],
                                            device bf *results_ptr [[buffer(4)]],
                                            constant size_t &results_stride [[buffer(5)]],
                                            uint2 tid [[thread_position_in_grid]]) {
  const unsigned idx = tid.y;
  if (idx >= indexes_count) return;
  const unsigned index = indexes[idx];
  // tid.x covers elements within a leaf row (blockDim.x)
  const unsigned src_row = index * CAPACITY + tid.x; // simplified: CAPACITY elements per leaf
  const unsigned dst_row = idx * CAPACITY + tid.x;
  // Single column dispatch
  results_ptr[dst_row] = values_ptr[src_row];
}

// Gather merkle paths kernel for Monolith
kernel void ab_monolith_gather_merkle_paths_kernel(device const unsigned *indexes [[buffer(0)]],
                                                    constant unsigned &indexes_count [[buffer(1)]],
                                                    device const bf *values [[buffer(2)]],
                                                    constant unsigned &log_leaves_count [[buffer(3)]],
                                                    device bf *results [[buffer(4)]],
                                                    uint2 tid [[thread_position_in_grid]]) {
  const unsigned idx = tid.y;
  if (idx >= indexes_count) return;
  const unsigned leaf_index = indexes[idx];
  const unsigned layer_index = tid.x;
  const unsigned layer_offset = ((1u << (log_leaves_count + 1)) - (1u << (log_leaves_count + 1 - layer_index))) * CAPACITY;
  const unsigned hash_offset = ((leaf_index >> layer_index) ^ 1) * CAPACITY;
  for (unsigned element_offset = 0; element_offset < CAPACITY; element_offset++) {
    const unsigned src_index = layer_offset + hash_offset + element_offset;
    const unsigned dst_index = layer_index * indexes_count * CAPACITY + idx * CAPACITY + element_offset;
    results[dst_index] = values[src_index];
  }
}

} // namespace monolith
} // namespace airbender
