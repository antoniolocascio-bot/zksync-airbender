///! Stage 5: Query phase (opening proofs).
///!
///! This stage:
///! 1. Generates query indices from the transcript
///! 2. Opens all committed polynomials at the queried positions
///! 3. Gathers the corresponding Merkle authentication paths
///! 4. Assembles the final proof

use crate::device_structures::MetalBuffer;
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;

type BF = BaseField;

/// Output of stage 5.
pub struct StageFiveOutput {
    /// Queried polynomial evaluations.
    pub query_values: Vec<Vec<BF>>,
    /// Merkle authentication paths for each query.
    pub merkle_paths: Vec<Vec<[u32; 8]>>,
}

/// Execute stage 5 of the proving pipeline.
pub fn execute_stage_5(ctx: &mut MetalProverContext) -> StageFiveOutput {
    // TODO: Implement query phase.
    //
    // Steps:
    // 1. Derive query indices from the Fiat-Shamir transcript.
    //    The indices are random positions in the evaluation domain.
    //
    // 2. For each committed polynomial oracle (trace, argument, quotient, FRI layers):
    //    a. Gather the polynomial values at queried rows
    //       Uses blake2s::gather_rows or gather_rows_and_merkle_paths
    //    b. Gather the Merkle authentication paths
    //       Uses blake2s::gather_merkle_paths
    //
    // 3. Copy gathered data from Metal buffers back to host memory
    //    (trivial with unified memory -- just read the slices).
    //
    // 4. Assemble all gathered data into the proof structure.
    //
    // Note: With Metal's unified memory, steps 2-3 are simpler than CUDA
    // since no explicit D2H copy is needed.

    StageFiveOutput {
        query_values: Vec::new(),
        merkle_paths: Vec::new(),
    }
}
