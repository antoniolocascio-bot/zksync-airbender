///! Stage 1: Witness generation and trace commitment.
///!
///! This stage:
///! 1. Allocates trace holders (MetalBuffers for witness/memory columns)
///! 2. Generates witness values (executes the circuit to fill in columns)
///! 3. Generates memory argument values
///! 4. Computes Merkle tree commitments over the trace columns
///! 5. Produces the stage-1 transcript contribution

use crate::circuit_type::CircuitType;
use crate::device_structures::MetalBuffer;
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;
use crate::prover::setup::SetupPrecomputations;

type BF = BaseField;

/// Output of stage 1.
pub struct StageOneOutput {
    /// Witness trace columns in Metal buffers.
    pub witness_columns: Vec<MetalBuffer<BF>>,
    /// Memory argument columns in Metal buffers.
    pub memory_columns: Vec<MetalBuffer<BF>>,
    /// Merkle tree over the combined trace (leaves + nodes).
    pub trace_tree: Option<MetalBuffer<[u32; 8]>>,
}

/// Execute stage 1 of the proving pipeline.
pub fn execute_stage_1(
    circuit_type: CircuitType,
    setup: &SetupPrecomputations,
    ctx: &mut MetalProverContext,
) -> StageOneOutput {
    let domain_size = circuit_type.get_domain_size();
    let _lde_factor = circuit_type.get_lde_factor();

    // Allocate witness trace columns
    // The number of witness columns depends on the circuit type.
    // For now, we allocate empty buffers as placeholders.
    let witness_columns = Vec::new();
    let memory_columns = Vec::new();

    // TODO: Populate witness columns by executing the circuit.
    // This involves:
    // 1. Reading input witness data from the host
    // 2. Dispatching witness generation kernels
    // 3. Computing memory argument columns
    // 4. Building Merkle tree commitments

    // Build merkle tree over trace columns
    // The tree stores [leaves | internal nodes] in a single buffer.
    let trace_tree = None;

    StageOneOutput {
        witness_columns,
        memory_columns,
        trace_tree,
    }
}
