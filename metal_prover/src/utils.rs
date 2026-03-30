use metal::MTLSize;

pub const LOG_WARP_SIZE: u32 = 5;
/// Metal SIMD group width on Apple Silicon is 32 threads.
pub const WARP_SIZE: u32 = 1 << LOG_WARP_SIZE;

pub trait GetChunksCount {
    fn get_chunks_count(self, chunk_size: Self) -> Self;
}

impl GetChunksCount for u32 {
    fn get_chunks_count(self, chunk_size: Self) -> Self {
        self.next_multiple_of(chunk_size) / chunk_size
    }
}

impl GetChunksCount for usize {
    fn get_chunks_count(self, chunk_size: Self) -> Self {
        self.next_multiple_of(chunk_size) / chunk_size
    }
}

/// Compute 1D threadgroup and grid dimensions for a given total thread count.
///
/// Returns `(threadgroups_per_grid, threads_per_threadgroup)` as `MTLSize` values.
/// The grid size represents the number of threadgroups (not total threads),
/// which is what `dispatchThreadgroups:threadsPerThreadgroup:` expects.
pub fn get_grid_block_dims_for_threads_count(
    threads_per_group: u32,
    total_threads: u32,
) -> (MTLSize, MTLSize) {
    let block_dim = std::cmp::min(total_threads, threads_per_group);
    let grid_dim = total_threads.get_chunks_count(block_dim);
    (
        MTLSize::new(grid_dim as u64, 1, 1),
        MTLSize::new(block_dim as u64, 1, 1),
    )
}

/// Compute 2D threadgroup and grid dimensions.
///
/// X dimension covers `rows`, Y dimension covers `cols`.
pub fn get_grid_block_dims_2d(
    threads_per_group: u32,
    rows: u32,
    cols: u32,
) -> (MTLSize, MTLSize) {
    let threadgroup_size = MTLSize::new(threads_per_group as u64, 1, 1);
    let grid_size = MTLSize::new(
        ((rows + threads_per_group - 1) / threads_per_group) as u64,
        cols as u64,
        1,
    );
    (grid_size, threadgroup_size)
}
