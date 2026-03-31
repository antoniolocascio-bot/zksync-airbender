///! MetalBackend — ProverBackend implementation for Apple Silicon.

use std::sync::Arc;

use super::{ProverBackend, BF};
use crate::types::Digest;
use crate::device_structures::{MetalBuffer, MetalMatrix, MetalMatrixMut};
use crate::ops_cub::sort::{self, SortOrder};
use crate::ops_cub::sort::run_length_encode_u32 as rle_u32;
use crate::prover::context::MetalProverContext;
use crate::ntt;

pub struct MetalBackend {
    pub ctx: Arc<MetalProverContext>,
}

impl MetalBackend {
    pub fn new(ctx: Arc<MetalProverContext>) -> Self {
        Self { ctx }
    }
}

unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl ProverBackend for MetalBackend {
    fn forward_ntt(
        &self,
        inputs: &[BF],
        outputs: &mut [BF],
        log_n: usize,
        num_bf_cols: usize,
        stride: usize,
        log_ext: usize,
        coset_idx: usize,
    ) {
        assert_eq!(inputs.len(), stride * num_bf_cols);
        assert_eq!(outputs.len(), stride * num_bf_cols);
        let src = MetalMatrix::new(inputs, stride);
        let mut dst = MetalMatrixMut::new(outputs, stride);
        ntt::bitrev_Z_to_natural_evals(
            &src, &mut dst, log_n, num_bf_cols, log_ext, coset_idx, &self.ctx,
        );
    }

    fn inverse_ntt(
        &self,
        inputs: &[BF],
        outputs: &mut [BF],
        log_n: usize,
        num_bf_cols: usize,
        stride: usize,
        log_ext: usize,
        coset_idx: usize,
    ) {
        assert_eq!(inputs.len(), stride * num_bf_cols);
        assert_eq!(outputs.len(), stride * num_bf_cols);
        let src = MetalMatrix::new(inputs, stride);
        let mut dst = MetalMatrixMut::new(outputs, stride);
        ntt::natural_evals_to_bitrev_Z(
            &src, &mut dst, log_n, num_bf_cols, log_ext, coset_idx, &self.ctx,
        );
    }

    fn build_merkle_tree(
        &self,
        values: &[BF],
        results: &mut [Digest],
        log_rows_per_hash: u32,
        layers_count: u32,
    ) {
        let mut values_buf = MetalBuffer::<BF>::new(&self.ctx.device, values.len());
        values_buf.load_from_host(values);
        let mut results_buf = MetalBuffer::<Digest>::new(&self.ctx.device, results.len());
        crate::blake2s::build_merkle_tree(
            &values_buf,
            &mut results_buf,
            log_rows_per_hash,
            &self.ctx,
            layers_count,
            false, // bit_reverse_leaves
        );
        results_buf.store_to_host(results);
    }

    fn gather_rows(
        &self,
        matrix: &[BF],
        indices: &[u32],
        dst: &mut [BF],
        row_size: usize,
        _num_rows: u32,
    ) {
        // row_size = 1 << log_rows_per_index; compute log
        let log_rows_per_index = row_size.trailing_zeros();
        assert_eq!(1usize << log_rows_per_index, row_size, "row_size must be a power of two");

        let num_cols = matrix.len() / row_size; // assuming square or total / row_stride
        // matrix is stored row-major: stride = num_cols
        let stride = num_cols;

        let mut indices_buf = MetalBuffer::<u32>::new(&self.ctx.device, indices.len());
        indices_buf.load_from_host(indices);
        let mat_view = crate::device_structures::MetalMatrix::new(matrix, stride);
        let dst_stride = num_cols;
        let mut dst_view = crate::device_structures::MetalMatrixMut::new(dst, dst_stride);

        crate::blake2s::gather_rows(
            &indices_buf,
            false, // bit_reverse_indexes
            log_rows_per_index,
            &mat_view,
            &mut dst_view,
            &self.ctx,
        );
    }

    fn gather_merkle_paths(
        &self,
        tree: &[Digest],
        indices: &[u32],
        dst: &mut [Digest],
        _log_size: u32,
        paths_per_query: u32, // = layers_count in the blake2s API
    ) {
        let mut tree_buf = MetalBuffer::<Digest>::new(&self.ctx.device, tree.len());
        tree_buf.load_from_host(tree);
        let mut indices_buf = MetalBuffer::<u32>::new(&self.ctx.device, indices.len());
        indices_buf.load_from_host(indices);
        let mut dst_buf = MetalBuffer::<Digest>::new(&self.ctx.device, dst.len());

        crate::blake2s::gather_merkle_paths(
            &indices_buf,
            &tree_buf,
            &mut dst_buf,
            paths_per_query, // layers_count
            &self.ctx,
        );
        dst_buf.store_to_host(dst);
    }

    fn sort_pairs_u32(
        &self,
        keys_in: &[u32],
        keys_out: &mut [u32],
        vals_in: &[u32],
        vals_out: &mut [u32],
        begin_bit: u32,
        end_bit: u32,
    ) {
        let keys_in_buf = {
            let mut b = MetalBuffer::<u32>::new(&self.ctx.device, keys_in.len());
            b.load_from_host(keys_in);
            b
        };
        let mut keys_out_buf = MetalBuffer::<u32>::new(&self.ctx.device, keys_out.len());
        let vals_in_buf = {
            let mut b = MetalBuffer::<u32>::new(&self.ctx.device, vals_in.len());
            b.load_from_host(vals_in);
            b
        };
        let mut vals_out_buf = MetalBuffer::<u32>::new(&self.ctx.device, vals_out.len());

        sort::sort_pairs_u32(
            &keys_in_buf,
            &mut keys_out_buf,
            &vals_in_buf,
            &mut vals_out_buf,
            SortOrder::Ascending,
            begin_bit,
            end_bit,
            &self.ctx,
        );
        keys_out_buf.store_to_host(keys_out);
        vals_out_buf.store_to_host(vals_out);
    }

    fn run_length_encode_u32(
        &self,
        input: &[u32],
        unique_out: &mut [u32],
        counts_out: &mut [u32],
    ) -> usize {
        let input_buf = {
            let mut b = MetalBuffer::<u32>::new(&self.ctx.device, input.len());
            b.load_from_host(input);
            b
        };
        let mut unique_buf = MetalBuffer::<u32>::new(&self.ctx.device, unique_out.len());
        let mut counts_buf = MetalBuffer::<u32>::new(&self.ctx.device, counts_out.len());
        let mut num_runs_buf = MetalBuffer::<u32>::new(&self.ctx.device, 1);

        rle_u32(
            &input_buf,
            &mut unique_buf,
            &mut counts_buf,
            &mut num_runs_buf,
            &self.ctx,
        );

        let num_runs = num_runs_buf.as_slice()[0] as usize;
        unique_buf.store_to_host(unique_out);
        counts_buf.store_to_host(counts_out);
        num_runs
    }
}
