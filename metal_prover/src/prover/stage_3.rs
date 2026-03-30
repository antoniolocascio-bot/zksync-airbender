///! Stage 3: Quotient polynomial computation and commitment.
///!
///! This stage:
///! 1. Evaluates all constraint polynomials at LDE points
///! 2. Divides by the vanishing polynomial to get the quotient
///! 3. Splits the quotient into degree-bounded pieces
///! 4. Commits to quotient columns via Merkle tree

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext4Field};
use crate::prover::context::MetalProverContext;

type BF = BaseField;
type E4 = Ext4Field;

/// Output of stage 3.
pub struct StageThreeOutput {
    /// Quotient polynomial columns in E4.
    pub quotient_columns: Vec<MetalBuffer<E4>>,
    /// Merkle tree over quotient columns.
    pub quotient_tree: Option<MetalBuffer<[u32; 8]>>,
}

/// Execute stage 3 of the proving pipeline.
pub fn execute_stage_3(ctx: &mut MetalProverContext) -> StageThreeOutput {
    // TODO: Implement stage 3.
    //
    // The high-level steps are:
    // 1. For each coset in the LDE domain:
    //    a. Apply NTT to get evaluations (bitrev_Z_to_natural_evals)
    //    b. Evaluate all constraints at these points using stage_3_kernels
    //    c. Divide by vanishing polynomial
    // 2. Apply inverse NTT to get quotient coefficients
    //    (natural_evals_to_bitrev_Z)
    // 3. Split quotient into degree-bounded columns
    // 4. Build Merkle tree over quotient columns
    //
    // Key Metal operations:
    // - NTT forward/inverse via ntt module
    // - Custom constraint evaluation kernels (stage_3_kernels)
    // - Element-wise arithmetic (ops_simple)
    // - Merkle tree building (blake2s or monolith)

    StageThreeOutput {
        quotient_columns: Vec::new(),
        quotient_tree: None,
    }
}
