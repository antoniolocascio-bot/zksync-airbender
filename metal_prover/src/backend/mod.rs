///! ProverBackend trait — platform-agnostic GPU abstraction.
///!
///! The MetalBackend (macOS) wraps Arc<MetalProverContext> and delegates to
///! the existing metal_prover functions.  The StubBackend (all platforms)
///! panics at runtime; it exists so orchestration code can be written
///! generically over `B: ProverBackend` and `cargo check`ed on Linux.

use crate::field::BaseField;
use crate::types::Digest;

pub type BF = BaseField;

/// Trait abstracting every GPU operation needed by the prover orchestration.
///
/// Implementations must be `Send + Sync` so they can be shared across threads.
pub trait ProverBackend: Send + Sync {
    // -------------------------------------------------------------------------
    // NTT / LDE
    // -------------------------------------------------------------------------

    /// Forward NTT: bitrev-Z coefficients → natural-order evaluations over a coset.
    ///
    /// `data` is a flat column-major matrix: `stride` BF elements per column,
    /// `num_bf_cols` columns.  `data.len() == stride * num_bf_cols`.
    /// `log_ext` and `coset_idx` control the LDE coset shift.
    fn forward_ntt(
        &self,
        inputs: &[BF],
        outputs: &mut [BF],
        log_n: usize,
        num_bf_cols: usize,
        stride: usize,
        log_ext: usize,
        coset_idx: usize,
    );

    /// Inverse NTT: natural-order evaluations over a coset → bitrev-Z coefficients.
    fn inverse_ntt(
        &self,
        inputs: &[BF],
        outputs: &mut [BF],
        log_n: usize,
        num_bf_cols: usize,
        stride: usize,
        log_ext: usize,
        coset_idx: usize,
    );

    // -------------------------------------------------------------------------
    // Blake2s / Merkle
    // -------------------------------------------------------------------------

    /// Build a Merkle tree from BF leaf data.
    ///
    /// `values` is a flat row-major matrix; each row of `1 << log_rows_per_hash`
    /// BF elements is hashed to produce one leaf digest.
    /// `results` must have room for all tree nodes (leaves + internal).
    fn build_merkle_tree(
        &self,
        values: &[BF],
        results: &mut [Digest],
        log_rows_per_hash: u32,
        layers_count: u32,
    );

    /// Gather rows from a BF matrix by index.
    fn gather_rows(
        &self,
        matrix: &[BF],
        indices: &[u32],
        dst: &mut [BF],
        row_size: usize,
        num_rows: u32,
    );

    /// Gather Merkle authentication paths.
    fn gather_merkle_paths(
        &self,
        tree: &[Digest],
        indices: &[u32],
        dst: &mut [Digest],
        log_size: u32,
        paths_per_query: u32,
    );

    // -------------------------------------------------------------------------
    // Sort / RLE (Stage 2)
    // -------------------------------------------------------------------------

    /// Radix sort u32 key-value pairs in-place.
    fn sort_pairs_u32(
        &self,
        keys_in: &[u32],
        keys_out: &mut [u32],
        vals_in: &[u32],
        vals_out: &mut [u32],
        begin_bit: u32,
        end_bit: u32,
    );

    /// Run-length encode a sorted u32 array.
    /// Returns the number of distinct runs.
    fn run_length_encode_u32(
        &self,
        input: &[u32],
        unique_out: &mut [u32],
        counts_out: &mut [u32],
    ) -> usize;
}

pub mod stub;

#[cfg(all(target_os = "macos", not(no_metal)))]
pub mod metal_backend;

pub use stub::StubBackend;

#[cfg(all(target_os = "macos", not(no_metal)))]
pub use metal_backend::MetalBackend;
