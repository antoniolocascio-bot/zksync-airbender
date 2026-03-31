#include <metal_stdlib>
using namespace metal;

#include "ntt.metal"

using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;
using namespace airbender::ntt;

namespace airbender {
namespace ntt {

// ============================================================================
// Block-level NTT kernels (LOG_VALS_PER_THREAD = 3, hardcoded).
//
// Port of CUDA:
//   bitrev_Z_to_natural_coset_evals_noninitial_stages_block<3>
//   evals_to_Z_nonfinal_stages_block<3>
//
// Constants (LOG_VALS_PER_THREAD = 3):
//   VALS_PER_THREAD       = 8
//   PAIRS_PER_THREAD      = 4
//   VALS_PER_WARP         = 256
//   WARPS_PER_BLOCK       = 16
//   VALS_PER_BLOCK        = 4096
//   TILES_PER_WARP        = 16
//   EXCHG_REGIONS_PER_BLK = 128
//   COL_PAIRS_PER_BLOCK   = 4   (for e2f type)
//   MAX_STAGES            = 8   (= 2*(3+5) - 8)
//
// Thread count: 512 = 16 warps × 32 lanes
// Threadgroup memory: 4096 × 8 = 32768 bytes
// ============================================================================

// ----------------------------------------------------------------------------
// Helper: load per-warp twiddles for non-initial / non-final block kernels.
//
// Fills twiddle_cache[0..14] with 15 twiddles needed for the 4 intra-warp
// stages.  Exactly mirrors CUDA load_noninitial_twiddles_warp<3, inverse>.
// ----------------------------------------------------------------------------
template <bool inverse>
DEVICE_FORCEINLINE void load_noninitial_twiddles_warp_block(
    threadgroup e2f *twiddle_cache,       // smem + VALS_PER_WARP * warp_id
    const unsigned lane_id,
    const unsigned warp_id,
    const unsigned block_exchg_region_offset,
    const device e2f *tw_fine,  const unsigned tw_fine_mask,  const unsigned tw_fine_log,
    const device e2f *tw_coarse, const unsigned tw_coarse_mask, const unsigned tw_coarse_log)
{
    constexpr unsigned LOG_VALS_PER_THREAD      = 3;
    constexpr unsigned NUM_INTRAWARP_STAGES     = LOG_VALS_PER_THREAD + 1;  // 4
    constexpr unsigned NUM_TWIDDLES_FIRST_STAGE = 1u << LOG_VALS_PER_THREAD; // 8

    unsigned exchg_region_offset = block_exchg_region_offset + warp_id * NUM_TWIDDLES_FIRST_STAGE;

    // Each lane fills one position; lane_id 0 does nothing (no twiddle at position 0).
    if (lane_id > 0 && lane_id < 2 * NUM_TWIDDLES_FIRST_STAGE) {
        const unsigned lz           = clz(lane_id);
        const unsigned stage_offset = NUM_INTRAWARP_STAGES - (32 - lz);
        const unsigned mask         = (1u << (32 - lz)) - 1;
        unsigned ero = exchg_region_offset >> stage_offset;
        twiddle_cache[lane_id ^ (2 * NUM_TWIDDLES_FIRST_STAGE - 1)] =
            get_twiddle<inverse>(tw_fine, tw_fine_mask, tw_fine_log,
                                 tw_coarse, tw_coarse_mask, tw_coarse_log,
                                 (lane_id ^ mask) + ero);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// ----------------------------------------------------------------------------
// B2N non-initial block: DIF stages for passes that follow the initial warp.
//
// Memory layout (two-for-one, bf layout):
//   element at (row r, e2f column c):
//     real: gmem_ptr[r + 2*c * stride]
//     imag: gmem_ptr[r + (2*c+1) * stride]
// ----------------------------------------------------------------------------
DEVICE_FORCEINLINE void b2n_noninitial_block(
    device const bf *gmem_in_ptr,  const size_t in_stride,
    device       bf *gmem_out_ptr, const size_t out_stride,
    const unsigned start_stage,    const bool skip_first_stage,
    const unsigned log_n,          const unsigned num_Z_cols,
    const unsigned grid_offset,
    const device e2f *tw_fine,  const unsigned tw_fine_mask,  const unsigned tw_fine_log,
    const device e2f *tw_coarse, const unsigned tw_coarse_mask, const unsigned tw_coarse_log,
    threadgroup e2f *smem,
    const unsigned tid,     // thread_index_in_threadgroup
    const unsigned lane_id, // tid & 31
    const unsigned warp_id, // tid >> 5
    const unsigned effective_block_idx_x,
    const unsigned blockIdx_y)
{
    constexpr unsigned VALS_PER_THREAD       = 8;
    constexpr unsigned PAIRS_PER_THREAD      = 4;
    constexpr unsigned VALS_PER_WARP         = 256;
    constexpr unsigned TILES_PER_WARP        = 16;
    constexpr unsigned WARPS_PER_BLOCK       = 16;
    constexpr unsigned EXCHG_REGIONS_PER_BLK = 128;
    constexpr unsigned COL_PAIRS_PER_BLOCK   = 4;

    // Tile stride: for skip_first_stage=false this equals tile_stride = 2^start_stage.
    const unsigned log_tile_stride      = skip_first_stage ? start_stage - 1u : start_stage;
    const unsigned tile_stride          = 1u << log_tile_stride;
    const unsigned log_blocks_per_region = log_tile_stride - 4u;  // tile size = 16

    // Block butterfly region (which large DIF region this block falls in).
    const unsigned block_bfly_region     = effective_block_idx_x >> log_blocks_per_region;
    const unsigned block_exchg_region_offset = block_bfly_region * EXCHG_REGIONS_PER_BLK;
    const unsigned block_bfly_region_size    = 256u * tile_stride;  // TILES_PER_BLOCK * tile_stride
    const unsigned block_bfly_region_start   = block_bfly_region * block_bfly_region_size;
    const unsigned block_start_in_region     = 16u * (effective_block_idx_x & ((1u << log_blocks_per_region) - 1u));
    const unsigned base_row = block_bfly_region_start + block_start_in_region;

    // Scrambled output row offset (same for all columns, computed once).
    const unsigned gmem_out_thread_offset =
        tile_stride * warp_id
        + tile_stride * WARPS_PER_BLOCK * (lane_id >> 4)
        + 2u * (lane_id & 7u)
        + ((lane_id >> 3) & 1u);

    // Per-warp twiddle cache region.
    threadgroup e2f *twiddle_cache = smem + VALS_PER_WARP * warp_id;

    // Load per-warp intra-warp twiddles once; restored from saved tmp on subsequent cols.
    load_noninitial_twiddles_warp_block<false>(
        twiddle_cache, lane_id, warp_id, block_exchg_region_offset,
        tw_fine, tw_fine_mask, tw_fine_log,
        tw_coarse, tw_coarse_mask, tw_coarse_log);

    const unsigned bound =
        min(COL_PAIRS_PER_BLOCK, num_Z_cols - COL_PAIRS_PER_BLOCK * blockIdx_y);

    for (unsigned ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
        const unsigned col_idx = COL_PAIRS_PER_BLOCK * blockIdx_y + ntt_idx;
        e2f vals[VALS_PER_THREAD];

        // ---- Load from global memory ----
        if (skip_first_stage) {
            // Pairs are at alternating offsets (tile_stride apart in one pair).
            unsigned vo = base_row
                + TILES_PER_WARP * tile_stride * warp_id
                + 2u * tile_stride * (lane_id >> 4)
                + 2u * (tid & 7u) + ((lane_id >> 3) & 1u);
            for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                vals[2*i]   = e2f{gmem_in_ptr[vo                 + col_idx*2*in_stride],
                                  gmem_in_ptr[vo                 + (col_idx*2+1)*in_stride]};
                vals[2*i+1] = e2f{gmem_in_ptr[vo + tile_stride   + col_idx*2*in_stride],
                                  gmem_in_ptr[vo + tile_stride   + (col_idx*2+1)*in_stride]};
                vo += 4u * tile_stride;
            }
        } else {
            // Two consecutive rows form a butterfly pair.
            unsigned po = base_row
                + TILES_PER_WARP * tile_stride * warp_id
                + tile_stride * (lane_id >> 3)
                + 2u * (tid & 7u);
            for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                vals[2*i]   = e2f{gmem_in_ptr[po     + col_idx*2*in_stride],
                                  gmem_in_ptr[po     + (col_idx*2+1)*in_stride]};
                vals[2*i+1] = e2f{gmem_in_ptr[po + 1 + col_idx*2*in_stride],
                                  gmem_in_ptr[po + 1 + (col_idx*2+1)*in_stride]};
                po += 4u * tile_stride;
            }
        }

        // ---- Intra-warp DIF stages (twiddle_cache) ----
        // 2 cross-lane stages (s=4,5 in the CUDA notation),
        // followed by 2 thread-local stages.
        unsigned lane_mask = 8u;
        threadgroup e2f *tc = twiddle_cache;
        unsigned ntws = 8u;  // num twiddles this stage = 1 << LOG_VALS_PER_THREAD

        for (unsigned s = 4; s < 6; s++) {  // 2 warp-shuffle DIF stages
            if (!skip_first_stage || s > 4) {
                for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                    const e2f twiddle = tc[(32u * i + lane_id) >> s];
                    shfl_xor_e2f(vals, i, lane_id, lane_mask);
                    exchg_dif(vals[2*i], vals[2*i+1], twiddle);
                }
            }
            lane_mask <<= 1;
            tc   += ntws;
            ntws >>= 1;
        }
        for (unsigned i = 1; i < 3; i++) {  // 2 thread-local DIF stages
            for (unsigned j = 0; j < (PAIRS_PER_THREAD >> i); j++) {
                const unsigned esz  = 2u << i;
                const unsigned hesz = 1u << i;
                const e2f twiddle   = tc[j];
                for (unsigned k = 0; k < hesz; k++)
                    exchg_dif(vals[esz*j+k], vals[esz*j+k+hesz], twiddle);
            }
            tc   += ntws;
            ntws >>= 1;
        }

        // ---- Save twiddle_cache before scatter overwrites it ----
        e2f twiddle_save = e2f{bf{0}, bf{0}};
        if (ntt_idx + 1 < bound)
            twiddle_save = twiddle_cache[lane_id];

        // ---- Cross-warp scatter to threadgroup memory ----
        const unsigned smem_off = 16u * (lane_id >> 4) + 2u * (lane_id & 7u) + ((lane_id >> 3) & 1u);
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
            twiddle_cache[64u*i + smem_off]      = vals[2*i];
            twiddle_cache[64u*i + smem_off + 32] = vals[2*i+1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Gather from threadgroup memory ----
        const unsigned spb = 16u * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2u * (tid & 7u);
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
            const unsigned sa = spb + 4u * VALS_PER_WARP * i;
            vals[2*i]   = smem[sa];
            vals[2*i+1] = smem[sa + 1];
        }

        // ---- Restore twiddle_cache for next column ----
        if (ntt_idx + 1 < bound) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            twiddle_cache[lane_id] = twiddle_save;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- Cross-warp DIF stages (global twiddle lookup) ----
        unsigned lm2 = 8u;
        // block_exchg_region_offset >> (LOG_VALS_PER_THREAD + 1) = >> 4
        unsigned ero2 = (block_exchg_region_offset >> 4u) + (lane_id >> 4);

        for (unsigned s = 0; s < 2; s++) {
            for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                const e2f twiddle = get_twiddle<false>(
                    tw_fine, tw_fine_mask, tw_fine_log,
                    tw_coarse, tw_coarse_mask, tw_coarse_log,
                    ero2 + ((2u * i) >> s));
                shfl_xor_e2f(vals, i, lane_id, lm2);
                exchg_dif(vals[2*i], vals[2*i+1], twiddle);
            }
            lm2 <<= 1;
            ero2 >>= 1;
        }
        for (unsigned i = 1; i < 3; i++) {
            for (unsigned j = 0; j < (PAIRS_PER_THREAD >> i); j++) {
                const unsigned esz  = 2u << i;
                const unsigned hesz = 1u << i;
                const e2f twiddle   = get_twiddle<false>(
                    tw_fine, tw_fine_mask, tw_fine_log,
                    tw_coarse, tw_coarse_mask, tw_coarse_log,
                    ero2 + (j >> (i - 1u)));
                for (unsigned k = 0; k < hesz; k++)
                    exchg_dif(vals[esz*j+k], vals[esz*j+k+hesz], twiddle);
            }
            ero2 >>= 1;
        }

        // ---- Store to global memory (scrambled output pattern) ----
        const unsigned out_row = base_row + gmem_out_thread_offset;
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
            const unsigned r0 = out_row + 4u * i * tile_stride * WARPS_PER_BLOCK;
            const unsigned r1 = out_row + (4u*i + 2u) * tile_stride * WARPS_PER_BLOCK;
            gmem_out_ptr[r0 + col_idx*2*out_stride]     = vals[2*i][0];
            gmem_out_ptr[r0 + (col_idx*2+1)*out_stride] = vals[2*i][1];
            gmem_out_ptr[r1 + col_idx*2*out_stride]     = vals[2*i+1][0];
            gmem_out_ptr[r1 + (col_idx*2+1)*out_stride] = vals[2*i+1][1];
        }
    }
}

// ----------------------------------------------------------------------------
// N2B non-final block: DIT stages for passes that precede the final warp.
// Reverses the b2n_noninitial_block pattern.
// ----------------------------------------------------------------------------
DEVICE_FORCEINLINE void n2b_nonfinal_block(
    device const bf *gmem_in_ptr,  const size_t in_stride,
    device       bf *gmem_out_ptr, const size_t out_stride,
    const unsigned start_stage,    const bool skip_last_stage,
    const unsigned log_n,          const unsigned num_Z_cols,
    const unsigned grid_offset,
    const device e2f *tw_fine,  const unsigned tw_fine_mask,  const unsigned tw_fine_log,
    const device e2f *tw_coarse, const unsigned tw_coarse_mask, const unsigned tw_coarse_log,
    threadgroup e2f *smem,
    const unsigned tid,
    const unsigned lane_id,
    const unsigned warp_id,
    const unsigned effective_block_idx_x,
    const unsigned blockIdx_y)
{
    constexpr unsigned VALS_PER_THREAD       = 8;
    constexpr unsigned PAIRS_PER_THREAD      = 4;
    constexpr unsigned VALS_PER_WARP         = 256;
    constexpr unsigned TILES_PER_WARP        = 16;
    constexpr unsigned WARPS_PER_BLOCK       = 16;
    constexpr unsigned EXCHG_REGIONS_PER_BLK = 128;
    constexpr unsigned COL_PAIRS_PER_BLOCK   = 4;
    constexpr unsigned MAX_STAGES            = 8;  // 2*(3+5) - 8

    // Tile stride for N2B non-final: log_tile_stride = log_n - start_stage - MAX_STAGES
    const unsigned log_tile_stride       = log_n - start_stage - MAX_STAGES;
    const unsigned tile_stride           = 1u << log_tile_stride;
    const unsigned log_blocks_per_region = log_tile_stride - 4u;

    const unsigned block_bfly_region     = effective_block_idx_x >> log_blocks_per_region;
    const unsigned block_bfly_region_size = 256u * tile_stride;
    const unsigned block_bfly_region_start = block_bfly_region * block_bfly_region_size;
    const unsigned block_start_in_region   = 16u * (effective_block_idx_x & ((1u << log_blocks_per_region) - 1u));
    const unsigned base_row = block_bfly_region_start + block_start_in_region;

    // Input row offset: same as B2N non-initial output pattern.
    const unsigned gmem_in_thread_offset =
        tile_stride * warp_id
        + tile_stride * WARPS_PER_BLOCK * (lane_id >> 4)
        + 2u * (lane_id & 7u)
        + ((lane_id >> 3) & 1u);

    threadgroup e2f *twiddle_cache = smem + VALS_PER_WARP * warp_id;
    const unsigned halfwarp_id = lane_id >> 4;

    const unsigned bound =
        min(COL_PAIRS_PER_BLOCK, num_Z_cols - COL_PAIRS_PER_BLOCK * blockIdx_y);

    for (unsigned ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
        const unsigned col_idx = COL_PAIRS_PER_BLOCK * blockIdx_y + ntt_idx;

        // ---- Save twiddle_cache before smem overwrites it (all but first col) ----
        e2f twiddle_save = e2f{bf{0}, bf{0}};
        if (ntt_idx > 0) {
            twiddle_save = twiddle_cache[lane_id];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- Load from global memory (= B2N non-initial output pattern) ----
        const unsigned in_row = base_row + gmem_in_thread_offset;
        e2f vals[VALS_PER_THREAD];
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
            const unsigned r0 = in_row + 4u * i * tile_stride * WARPS_PER_BLOCK;
            const unsigned r1 = in_row + (4u*i + 2u) * tile_stride * WARPS_PER_BLOCK;
            vals[2*i]   = e2f{gmem_in_ptr[r0 + col_idx*2*in_stride],
                              gmem_in_ptr[r0 + (col_idx*2+1)*in_stride]};
            vals[2*i+1] = e2f{gmem_in_ptr[r1 + col_idx*2*in_stride],
                              gmem_in_ptr[r1 + (col_idx*2+1)*in_stride]};
        }

        // ---- Thread-local DIT stages (global twiddle lookup) ----
        // block_exchg_region_offset starts at block_bfly_region, doubles each iter.
        unsigned bero = block_bfly_region;
        for (unsigned i = 0; i < 2; i++) {  // LOG_VALS_PER_THREAD - 1 = 2
            for (unsigned j = 0; j < (1u << i); j++) {
                const unsigned esz  = VALS_PER_THREAD >> i;
                const unsigned hesz = esz >> 1;
                const e2f twiddle   = get_twiddle<true>(
                    tw_fine, tw_fine_mask, tw_fine_log,
                    tw_coarse, tw_coarse_mask, tw_coarse_log,
                    bero + j);
                for (unsigned k = 0; k < hesz; k++)
                    exchg_dit(vals[esz*j+k], vals[esz*j+k+hesz], twiddle);
            }
            bero <<= 1;
        }

        // ---- Cross-warp DIT stages (global twiddle lookup) ----
        unsigned lm = 16u;
        for (unsigned s = 0; s < 2; s++) {
            for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                const e2f twiddle = get_twiddle<true>(
                    tw_fine, tw_fine_mask, tw_fine_log,
                    tw_coarse, tw_coarse_mask, tw_coarse_log,
                    bero + ((2u*i + halfwarp_id) >> (1u - s)));
                exchg_dit(vals[2*i], vals[2*i+1], twiddle);
            }
            for (unsigned i = 0; i < PAIRS_PER_THREAD; i++)
                shfl_xor_e2f(vals, i, lane_id, lm);
            lm >>= 1;
            bero <<= 1;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Scatter to threadgroup memory (= B2N gather pattern) ----
        const unsigned spb = 16u * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2u * (tid & 7u);
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
            const unsigned sa = spb + 4u * VALS_PER_WARP * i;
            smem[sa]     = vals[2*i];
            smem[sa + 1] = vals[2*i+1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Gather from threadgroup memory (= B2N scatter pattern) ----
        const unsigned smem_off = 16u * (lane_id >> 4) + 2u * (lane_id & 7u) + ((lane_id >> 3) & 1u);
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
            vals[2*i]   = twiddle_cache[64u*i + smem_off];
            vals[2*i+1] = twiddle_cache[64u*i + smem_off + 32];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Restore or load intra-warp twiddles ----
        if (ntt_idx > 0) {
            twiddle_cache[lane_id] = twiddle_save;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            // Load fresh inverse twiddles from global memory.
            load_noninitial_twiddles_warp_block<true>(
                twiddle_cache, lane_id, warp_id,
                block_bfly_region * EXCHG_REGIONS_PER_BLK,
                tw_fine, tw_fine_mask, tw_fine_log,
                tw_coarse, tw_coarse_mask, tw_coarse_log);
        }

        // ---- Intra-warp DIT stages (twiddle_cache) ----
        // Traverse twiddle_cache from the end toward the beginning (reverse of DIF).
        threadgroup e2f *tc = twiddle_cache + 2u * VALS_PER_THREAD - 2u;  // offset 14
        unsigned ntws = 1u;
        for (unsigned i = 0; i < 2; i++) {  // LOG_VALS_PER_THREAD - 1 = 2
            for (unsigned j = 0; j < (1u << i); j++) {
                const unsigned esz  = VALS_PER_THREAD >> i;
                const unsigned hesz = esz >> 1;
                const e2f twiddle   = tc[j];
                for (unsigned k = 0; k < hesz; k++)
                    exchg_dit(vals[esz*j+k], vals[esz*j+k+hesz], twiddle);
            }
            ntws <<= 1;
            tc   -= ntws;
        }

        // 2 cross-lane DIT stages
        unsigned lm2 = 16u;
        for (unsigned s = 0; s < 2; s++) {
            if (!skip_last_stage || s < 1) {
                for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                    const e2f twiddle = tc[(2u*i + halfwarp_id) >> (1u - s)];
                    exchg_dit(vals[2*i], vals[2*i+1], twiddle);
                }
                for (unsigned i = 0; i < PAIRS_PER_THREAD; i++)
                    shfl_xor_e2f(vals, i, lane_id, lm2);
                lm2 >>= 1;
                ntws <<= 1;
                tc   -= ntws;
            }
        }

        // ---- Store to global memory (= B2N non-initial input pattern) ----
        if (skip_last_stage) {
            unsigned vo = base_row
                + TILES_PER_WARP * tile_stride * warp_id
                + 2u * tile_stride * (lane_id >> 4)
                + 2u * (tid & 7u) + ((lane_id >> 3) & 1u);
            for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                gmem_out_ptr[vo               + col_idx*2*out_stride]     = vals[2*i][0];
                gmem_out_ptr[vo               + (col_idx*2+1)*out_stride] = vals[2*i][1];
                gmem_out_ptr[vo + tile_stride + col_idx*2*out_stride]     = vals[2*i+1][0];
                gmem_out_ptr[vo + tile_stride + (col_idx*2+1)*out_stride] = vals[2*i+1][1];
                vo += 4u * tile_stride;
            }
        } else {
            unsigned po = base_row
                + TILES_PER_WARP * tile_stride * warp_id
                + tile_stride * (lane_id >> 3)
                + 2u * (tid & 7u);
            for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
                gmem_out_ptr[po     + col_idx*2*out_stride]     = vals[2*i][0];
                gmem_out_ptr[po     + (col_idx*2+1)*out_stride] = vals[2*i][1];
                gmem_out_ptr[po + 1 + col_idx*2*out_stride]     = vals[2*i+1][0];
                gmem_out_ptr[po + 1 + (col_idx*2+1)*out_stride] = vals[2*i+1][1];
                po += 4u * tile_stride;
            }
        }
    }
}

// ============================================================================
// Public kernel entry points
//
// Buffer layout (same for both kernels, 15 buffers):
//   0: device const bf *gmem_in
//   1: device bf *gmem_out
//   2: constant size_t &in_stride
//   3: constant size_t &out_stride
//   4: constant unsigned &start_stage
//   5: constant unsigned &stages_this_launch
//   6: constant unsigned &log_n
//   7: constant unsigned &num_Z_cols
//   8: constant unsigned &grid_offset
//   9: device const e2f *twiddle_fine
//  10: constant unsigned &twiddle_fine_mask
//  11: constant unsigned &twiddle_fine_log
//  12: device const e2f *twiddle_coarse
//  13: constant unsigned &twiddle_coarse_mask
//  14: constant unsigned &twiddle_coarse_log
//  threadgroup(0): e2f *smem  [32768 bytes = VALS_PER_BLOCK * sizeof(e2f)]
// ============================================================================

[[max_total_threads_per_threadgroup(512)]]
kernel void ab_bitrev_Z_to_natural_coset_evals_noninitial_7_or_8_stages_block(
    device const bf         *gmem_in        [[buffer(0)]],
    device       bf         *gmem_out       [[buffer(1)]],
    constant size_t         &in_stride      [[buffer(2)]],
    constant size_t         &out_stride     [[buffer(3)]],
    constant unsigned       &start_stage    [[buffer(4)]],
    constant unsigned       &stages_this_launch [[buffer(5)]],
    constant unsigned       &log_n          [[buffer(6)]],
    constant unsigned       &num_Z_cols     [[buffer(7)]],
    constant unsigned       &grid_offset    [[buffer(8)]],
    device const e2f        *twiddle_fine   [[buffer(9)]],
    constant unsigned       &tw_fine_mask   [[buffer(10)]],
    constant unsigned       &tw_fine_log    [[buffer(11)]],
    device const e2f        *twiddle_coarse [[buffer(12)]],
    constant unsigned       &tw_coarse_mask [[buffer(13)]],
    constant unsigned       &tw_coarse_log  [[buffer(14)]],
    threadgroup e2f         *smem           [[threadgroup(0)]],
    uint                     tid            [[thread_index_in_threadgroup]],
    uint2                    gid_2d         [[threadgroup_position_in_grid]])
{
    const unsigned lane_id = tid & 31u;
    const unsigned warp_id = tid >> 5;
    const unsigned effective_block_idx_x = gid_2d.x + grid_offset;
    const bool skip_first_stage = (stages_this_launch == 7);

    b2n_noninitial_block(
        gmem_in, in_stride, gmem_out, out_stride,
        start_stage, skip_first_stage, log_n, num_Z_cols, grid_offset,
        twiddle_fine, tw_fine_mask, tw_fine_log,
        twiddle_coarse, tw_coarse_mask, tw_coarse_log,
        smem, tid, lane_id, warp_id, effective_block_idx_x, gid_2d.y);
}

[[max_total_threads_per_threadgroup(512)]]
kernel void ab_evals_to_Z_nonfinal_7_or_8_stages_block(
    device const bf         *gmem_in        [[buffer(0)]],
    device       bf         *gmem_out       [[buffer(1)]],
    constant size_t         &in_stride      [[buffer(2)]],
    constant size_t         &out_stride     [[buffer(3)]],
    constant unsigned       &start_stage    [[buffer(4)]],
    constant unsigned       &stages_this_launch [[buffer(5)]],
    constant unsigned       &log_n          [[buffer(6)]],
    constant unsigned       &num_Z_cols     [[buffer(7)]],
    constant unsigned       &grid_offset    [[buffer(8)]],
    device const e2f        *twiddle_fine   [[buffer(9)]],
    constant unsigned       &tw_fine_mask   [[buffer(10)]],
    constant unsigned       &tw_fine_log    [[buffer(11)]],
    device const e2f        *twiddle_coarse [[buffer(12)]],
    constant unsigned       &tw_coarse_mask [[buffer(13)]],
    constant unsigned       &tw_coarse_log  [[buffer(14)]],
    threadgroup e2f         *smem           [[threadgroup(0)]],
    uint                     tid            [[thread_index_in_threadgroup]],
    uint2                    gid_2d         [[threadgroup_position_in_grid]])
{
    const unsigned lane_id = tid & 31u;
    const unsigned warp_id = tid >> 5;
    const unsigned effective_block_idx_x = gid_2d.x + grid_offset;
    const bool skip_last_stage = (stages_this_launch == 7);

    n2b_nonfinal_block(
        gmem_in, in_stride, gmem_out, out_stride,
        start_stage, skip_last_stage, log_n, num_Z_cols, grid_offset,
        twiddle_fine, tw_fine_mask, tw_fine_log,
        twiddle_coarse, tw_coarse_mask, tw_coarse_log,
        smem, tid, lane_id, warp_id, effective_block_idx_x, gid_2d.y);
}

} // namespace ntt
} // namespace airbender
