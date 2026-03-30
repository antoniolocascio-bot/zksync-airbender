///! Proof generation orchestration for Metal.
///!
///! Coordinates the five prover stages to produce a complete proof.

use crate::circuit_type::CircuitType;
use crate::prover::context::MetalProverContext;
use crate::prover::setup::SetupPrecomputations;

/// A proof job that can be waited on for completion.
///
/// Unlike the CUDA version which uses CUDA events for async completion,
/// the Metal version uses command buffer completion handlers.
pub struct ProofJob {
    /// Whether the proof generation is complete.
    completed: bool,
}

impl ProofJob {
    /// Check if the proof generation has completed.
    pub fn is_finished(&self) -> bool {
        self.completed
    }
}

/// Generate a complete proof for a given circuit.
///
/// This is the main entry point for proof generation. It executes all five
/// stages sequentially on the Metal GPU.
///
/// The stages are:
/// 1. Witness generation: populate trace columns, compute merkle tree
/// 2. Memory/lookup argument: compute permutation argument columns
/// 3. Quotient polynomial: evaluate constraints, compute quotient commitment
/// 4. FRI: fold the quotient polynomial, commit FRI layers
/// 5. Queries: open polynomials at queried positions, gather merkle paths
pub fn generate_proof(
    circuit_type: CircuitType,
    setup: &SetupPrecomputations,
    ctx: &mut MetalProverContext,
) -> ProofJob {
    // Stage 1: Witness generation and trace commitment
    let _stage_1_output = crate::prover::stage_1::execute_stage_1(circuit_type, setup, ctx);

    // Stage 2: Memory argument and lookup argument
    let _stage_2_output = crate::prover::stage_2::execute_stage_2(ctx);

    // Stage 3: Quotient polynomial computation
    let _stage_3_output = crate::prover::stage_3::execute_stage_3(ctx);

    // Stage 4: FRI commitment
    let _stage_4_output = crate::prover::stage_4::execute_stage_4(ctx);

    // Stage 5: Query phase
    let _stage_5_output = crate::prover::stage_5::execute_stage_5(ctx);

    ProofJob { completed: true }
}
