///! Main execution prover for Metal.
///!
///! This module provides the top-level interface for running the prover
///! on Metal. It is simplified compared to the CUDA version because:
///! - No multi-GPU support needed (Apple Silicon is single GPU)
///! - No explicit H2D memory transfers (unified memory)
///! - No overlapping computation and transfer

use std::sync::Arc;

use crate::circuit_type::CircuitType;
use crate::prover::context::{MetalProverContext, MetalProverContextConfig};
use crate::prover::proof::ProofJob;
use crate::prover::setup::SetupPrecomputations;

/// Configuration for the execution prover.
pub struct ExecutionConfig {
    /// Metal prover context configuration.
    pub context_config: MetalProverContextConfig,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            context_config: MetalProverContextConfig::default(),
        }
    }
}

/// The main Metal execution prover.
///
/// Manages the Metal prover context and provides an interface for
/// submitting proof jobs.
pub struct MetalExecutionProver {
    ctx: MetalProverContext,
}

impl MetalExecutionProver {
    /// Create a new Metal execution prover.
    pub fn new(config: ExecutionConfig) -> Self {
        let ctx = MetalProverContext::new(config.context_config);
        log::info!(
            "Metal prover initialized on device: {}",
            ctx.device_name()
        );
        Self { ctx }
    }

    /// Submit a proof generation job.
    ///
    /// This method blocks until the proof is complete (Metal command buffers
    /// are committed and waited on synchronously in the current implementation).
    pub fn prove(
        &mut self,
        circuit_type: CircuitType,
        setup: &SetupPrecomputations,
    ) -> ProofJob {
        // Reset the allocator for a fresh proof
        self.ctx.reset_allocator();

        // Run the full proving pipeline
        crate::prover::proof::generate_proof(circuit_type, setup, &mut self.ctx)
    }

    /// Get a reference to the prover context.
    pub fn context(&self) -> &MetalProverContext {
        &self.ctx
    }

    /// Get a mutable reference to the prover context.
    pub fn context_mut(&mut self) -> &mut MetalProverContext {
        &mut self.ctx
    }
}
