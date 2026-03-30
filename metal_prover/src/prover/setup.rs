///! Setup precomputations for the Metal prover.
///!
///! Loads the compiled circuit artifact and precomputes setup-dependent
///! data (e.g., permutation polynomials, setup tree) into Metal buffers.

use crate::device_structures::MetalBuffer;
use crate::field::BaseField;
use crate::prover::context::MetalProverContext;

type BF = BaseField;

/// Holds precomputed setup data in Metal buffers.
pub struct SetupPrecomputations {
    /// Setup columns stored as MetalBuffers (column-major).
    pub setup_columns: Vec<MetalBuffer<BF>>,
    /// Log2 of the LDE factor.
    pub log_lde_factor: u32,
    /// Log2 of the tree cap size.
    pub log_tree_cap_size: u32,
}

impl SetupPrecomputations {
    /// Load setup precomputations from a compiled circuit artifact.
    ///
    /// Copies setup polynomial columns into Metal buffers (unified memory).
    pub fn load(
        setup_columns_host: &[Vec<BF>],
        log_lde_factor: u32,
        log_tree_cap_size: u32,
        ctx: &MetalProverContext,
    ) -> Self {
        let mut setup_columns = Vec::with_capacity(setup_columns_host.len());
        for col_host in setup_columns_host {
            let mut buf = MetalBuffer::<BF>::new(&ctx.device, col_host.len());
            buf.copy_from_slice(col_host);
            setup_columns.push(buf);
        }

        Self {
            setup_columns,
            log_lde_factor,
            log_tree_cap_size,
        }
    }

    /// Number of setup columns.
    pub fn num_columns(&self) -> usize {
        self.setup_columns.len()
    }

    /// Domain size (number of rows per column).
    pub fn domain_size(&self) -> usize {
        if self.setup_columns.is_empty() {
            0
        } else {
            self.setup_columns[0].len()
        }
    }
}
