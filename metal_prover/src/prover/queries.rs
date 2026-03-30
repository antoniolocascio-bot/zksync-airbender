///! Query generation and opening proof utilities.
///!
///! Handles the derivation of query indices from the transcript and
///! the gathering of polynomial evaluations + Merkle paths.

use crate::device_structures::MetalBuffer;
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;

type BF = BaseField;

/// Holds the output of the query phase for a single oracle.
pub struct QueriesOutput {
    /// Query indices (positions in the evaluation domain).
    pub indices: MetalBuffer<u32>,
    /// Number of queries.
    pub num_queries: usize,
}

/// Generate random query indices from a seed.
///
/// The indices are derived by hashing the transcript state and
/// reducing modulo the domain size.
pub fn generate_query_indices(
    seed: &[u32],
    num_queries: usize,
    log_domain_size: u32,
    ctx: &MetalProverContext,
) -> MetalBuffer<u32> {
    let domain_size = 1u32 << log_domain_size;
    let mut indices = MetalBuffer::<u32>::new(&ctx.device, num_queries);

    // Generate indices on the host side (simple hash-based sampling).
    // In a production implementation, this could be done on the GPU.
    let indices_slice = indices.as_mut_slice();
    let mut state = [0u32; 8];
    state[..seed.len().min(8)].copy_from_slice(&seed[..seed.len().min(8)]);

    for i in 0..num_queries {
        // Simple deterministic index generation from seed
        // (production code would use proper Fiat-Shamir)
        let mut h = state[0].wrapping_add(i as u32);
        h = h.wrapping_mul(0x9e3779b9);
        h ^= h >> 16;
        h = h.wrapping_mul(0x85ebca6b);
        h ^= h >> 13;
        indices_slice[i] = h % domain_size;
    }

    indices
}

/// Gather evaluations of multiple polynomial columns at the query indices.
///
/// For each column and each query index, reads the evaluation from the
/// column buffer. With unified memory, this is a direct memory read.
pub fn gather_evaluations_at_queries(
    columns: &[MetalBuffer<BF>],
    indices: &MetalBuffer<u32>,
    domain_size: usize,
) -> Vec<Vec<BF>> {
    let indices_slice = indices.as_slice();
    columns
        .iter()
        .map(|col| {
            let col_slice = col.as_slice();
            indices_slice
                .iter()
                .map(|&idx| col_slice[idx as usize % domain_size])
                .collect()
        })
        .collect()
}
