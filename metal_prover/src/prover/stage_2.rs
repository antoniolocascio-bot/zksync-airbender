///! Stage 2: Memory argument and lookup argument computation.
///!
///! This stage:
///! 1. Computes column sums for the permutation argument
///! 2. Applies the permutation/memory argument using cumulative products
///! 3. Computes lookup argument (generic lookups + range check)
///! 4. Commits to the argument polynomials via Merkle tree

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext4Field};
use crate::prover::context::MetalProverContext;

type BF = BaseField;
type E4 = Ext4Field;

/// Output of stage 2.
pub struct StageTwoOutput {
    /// Permutation argument columns.
    pub permutation_columns: Vec<MetalBuffer<E4>>,
    /// Lookup argument columns.
    pub lookup_columns: Vec<MetalBuffer<E4>>,
    /// Merkle tree over the argument columns.
    pub argument_tree: Option<MetalBuffer<[u32; 8]>>,
}

/// Execute stage 2 of the proving pipeline.
pub fn execute_stage_2(ctx: &mut MetalProverContext) -> StageTwoOutput {
    // TODO: Implement stage 2.
    //
    // The high-level steps are:
    // 1. Compute column partial sums using segmented reduction (ops_cub::reduce)
    // 2. Compute prefix products using scan (ops_cub::scan)
    // 3. Compute the full permutation argument Z(x)
    // 4. Compute lookup argument columns
    // 5. Build Merkle tree over all argument columns
    //
    // Each step dispatches Metal compute kernels through the context.
    // The key operations used are:
    // - segmented_reduce for column sums
    // - inclusive_scan for cumulative products
    // - batch_inv for denominator inversion
    // - NTT for polynomial evaluation
    // - blake2s/monolith for Merkle tree construction

    StageTwoOutput {
        permutation_columns: Vec::new(),
        lookup_columns: Vec::new(),
        argument_tree: None,
    }
}
