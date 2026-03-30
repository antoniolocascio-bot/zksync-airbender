///! Stage 4: FRI commitment (Interactive Oracle Proof).
///!
///! This stage:
///! 1. Combines all committed polynomials into a single polynomial
///!    using verifier challenges
///! 2. Performs FRI folding rounds: repeatedly halve the polynomial
///!    degree using folding challenges
///! 3. Commits to each FRI layer via Merkle tree
///! 4. Sends the final (constant) polynomial value

use crate::device_structures::MetalBuffer;
use crate::field::Ext4Field;
use crate::prover::context::MetalProverContext;

type E4 = Ext4Field;

/// Output of stage 4.
pub struct StageFourOutput {
    /// FRI layer commitments (Merkle trees).
    pub fri_layer_trees: Vec<MetalBuffer<[u32; 8]>>,
    /// FRI oracle polynomial values at each folding step.
    pub fri_oracles: Vec<MetalBuffer<E4>>,
    /// Final constant polynomial value.
    pub final_value: Option<E4>,
}

/// Execute stage 4 of the proving pipeline.
pub fn execute_stage_4(ctx: &mut MetalProverContext) -> StageFourOutput {
    // TODO: Implement FRI commitment stage.
    //
    // The FRI protocol proceeds as:
    // 1. Combine all committed polynomials using random challenges:
    //    p(x) = sum_i alpha_i * poly_i(x)
    //    This is done via element-wise mul and add (ops_simple).
    //
    // 2. For each folding round:
    //    a. Fold: dst[i] = (src[2i] + src[2i+1]) + challenge *
    //       (src[2i] - src[2i+1]) * twiddle[i]
    //       Uses ops_complex::fold
    //    b. Commit the folded polynomial via Merkle tree
    //       Uses blake2s::build_merkle_tree or monolith::build_merkle_tree
    //
    // 3. Send the final constant value to the transcript.
    //
    // The number of folding rounds depends on the FRI configuration
    // (determined by the security config from circuit_type).

    StageFourOutput {
        fri_layer_trees: Vec::new(),
        fri_oracles: Vec::new(),
        final_value: None,
    }
}
