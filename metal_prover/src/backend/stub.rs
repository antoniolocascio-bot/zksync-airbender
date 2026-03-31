///! StubBackend — panics on every method.
///!
///! Exists so orchestration code can be written generically over
///! `B: ProverBackend` and `cargo check`ed on Linux without Metal.

use super::{ProverBackend, BF};
use crate::types::Digest;

pub struct StubBackend;

impl ProverBackend for StubBackend {
    fn forward_ntt(
        &self, _inputs: &[BF], _outputs: &mut [BF],
        _log_n: usize, _num_bf_cols: usize, _stride: usize,
        _log_ext: usize, _coset_idx: usize,
    ) {
        panic!("StubBackend: Metal not available on this platform");
    }

    fn inverse_ntt(
        &self, _inputs: &[BF], _outputs: &mut [BF],
        _log_n: usize, _num_bf_cols: usize, _stride: usize,
        _log_ext: usize, _coset_idx: usize,
    ) {
        panic!("StubBackend: Metal not available on this platform");
    }

    fn build_merkle_tree(
        &self, _values: &[BF], _results: &mut [Digest],
        _log_rows_per_hash: u32, _layers_count: u32,
    ) {
        panic!("StubBackend: Metal not available on this platform");
    }

    fn gather_rows(
        &self, _matrix: &[BF], _indices: &[u32], _dst: &mut [BF],
        _row_size: usize, _num_rows: u32,
    ) {
        panic!("StubBackend: Metal not available on this platform");
    }

    fn gather_merkle_paths(
        &self, _tree: &[Digest], _indices: &[u32], _dst: &mut [Digest],
        _log_size: u32, _paths_per_query: u32,
    ) {
        panic!("StubBackend: Metal not available on this platform");
    }

    fn sort_pairs_u32(
        &self, _keys_in: &[u32], _keys_out: &mut [u32],
        _vals_in: &[u32], _vals_out: &mut [u32],
        _begin_bit: u32, _end_bit: u32,
    ) {
        panic!("StubBackend: Metal not available on this platform");
    }

    fn run_length_encode_u32(
        &self, _input: &[u32], _unique_out: &mut [u32], _counts_out: &mut [u32],
    ) -> usize {
        panic!("StubBackend: Metal not available on this platform");
    }
}
