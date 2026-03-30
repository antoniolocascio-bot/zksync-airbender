///! Precomputations for the Metal prover pipeline.
///!
///! Holds LDE precomputation data and other circuit-specific constants
///! that are computed once and reused across multiple proof generations.

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext2Field};
use crate::prover::context::MetalProverContext;

type BF = BaseField;
type E2 = Ext2Field;

/// LDE precomputations stored in Metal buffers.
pub struct LdePrecomputations {
    /// Twiddle factors for the LDE, stored in Metal buffers for GPU access.
    pub twiddle_factors: Vec<MetalBuffer<E2>>,
    /// Log2 of the domain size.
    pub log_domain_size: u32,
    /// Log2 of the LDE factor.
    pub log_lde_factor: u32,
}

impl LdePrecomputations {
    /// Create LDE precomputations for a given domain size and LDE factor.
    pub fn new(
        log_domain_size: u32,
        log_lde_factor: u32,
        ctx: &MetalProverContext,
    ) -> Self {
        // The twiddle factors for LDE are already part of the DeviceContext.
        // This struct provides a convenient wrapper for the prover stages.
        let twiddle_factors = Vec::new();

        Self {
            twiddle_factors,
            log_domain_size,
            log_lde_factor,
        }
    }

    /// Total LDE domain size.
    pub fn lde_domain_size(&self) -> usize {
        1usize << (self.log_domain_size + self.log_lde_factor)
    }
}
