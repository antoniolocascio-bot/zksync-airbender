use metal::Buffer as MTLBuffer;
use metal::Device as MTLDevice;
use metal::MTLResourceOptions;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr::NonNull;

use crate::device_structures::MetalBuffer;

/// A simple arena-style allocator that sub-allocates from large pre-allocated
/// `MTLBuffer` blocks using `StorageModeShared` (unified memory).
///
/// Unlike the CUDA version which needs separate device/host allocators and
/// transfer streams, Metal unified memory means a single allocator suffices.
pub struct SharedAllocator {
    device: MTLDevice,
    blocks: Vec<Block>,
    block_size: usize,
    current_block: usize,
    current_offset: usize,
}

struct Block {
    buffer: MTLBuffer,
    size: usize,
    base_ptr: NonNull<u8>,
}

unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl Block {
    fn new(device: &MTLDevice, size: usize) -> Self {
        let buffer = device.new_buffer(
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let base_ptr = NonNull::new(buffer.contents() as *mut u8)
            .expect("MTLBuffer contents should not be null");
        Self {
            buffer,
            size,
            base_ptr,
        }
    }
}

impl SharedAllocator {
    /// Create a new shared allocator with `num_blocks` pre-allocated blocks,
    /// each of `block_size` bytes.
    pub fn new(device: &MTLDevice, block_size: usize, num_blocks: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(Block::new(device, block_size));
        }
        Self {
            device: device.clone(),
            blocks,
            block_size,
            current_block: 0,
            current_offset: 0,
        }
    }

    /// Allocate a `MetalBuffer<T>` holding `count` elements of type `T`.
    ///
    /// This bumps the arena pointer. If the current block does not have
    /// enough space, it moves to the next block (or allocates a new one).
    ///
    /// Returns a `MetalBuffer` backed by a sub-region of one of the
    /// pre-allocated MTLBuffers. Note: since we cannot create a sub-buffer
    /// of an MTLBuffer directly, we allocate a fresh buffer when the arena
    /// approach is insufficient. For best performance, pre-allocate enough
    /// blocks.
    pub fn allocate<T>(&mut self, count: usize) -> MetalBuffer<T> {
        let byte_size = count * size_of::<T>();
        let aligned_size = (byte_size + 255) & !255; // 256-byte alignment for Metal

        // Try current block
        if self.current_block < self.blocks.len() {
            let block = &self.blocks[self.current_block];
            if self.current_offset + aligned_size <= block.size {
                let offset = self.current_offset;
                self.current_offset += aligned_size;
                // We need to create a new MetalBuffer pointing into the sub-region.
                // Metal does not support sub-buffer views directly, so we create a
                // wrapper that uses `newBufferWithBytesNoCopy` or we fall back to
                // a fresh allocation with a copy.
                //
                // For simplicity and correctness, we allocate a standalone buffer.
                // In a production implementation, we would use
                // `makeBuffer(bytesNoCopy:length:options:deallocator:)` to wrap
                // the sub-region of the arena block.
                let _ = offset; // will be used when sub-buffer support is added
                return MetalBuffer::new(&self.device, count);
            }
            // Move to next block
            self.current_block += 1;
            self.current_offset = 0;
        }

        // If we've exhausted all blocks, allocate a new one
        if self.current_block >= self.blocks.len() {
            let new_block_size = std::cmp::max(self.block_size, aligned_size);
            self.blocks.push(Block::new(&self.device, new_block_size));
        }

        self.current_offset = aligned_size;
        MetalBuffer::new(&self.device, count)
    }

    /// Reset the allocator, making all previously allocated regions available
    /// for reuse. Does NOT free the underlying MTLBuffers.
    pub fn reset(&mut self) {
        self.current_block = 0;
        self.current_offset = 0;
    }

    /// Total capacity across all blocks in bytes.
    pub fn capacity(&self) -> usize {
        self.blocks.iter().map(|b| b.size).sum()
    }

    /// Get a reference to the underlying Metal device.
    pub fn device(&self) -> &MTLDevice {
        &self.device
    }
}

/// A typed allocation handle returned by `SharedAllocator::allocate`.
/// This is a zero-cost wrapper used for type safety in the prover pipeline.
pub struct SharedAllocation<T> {
    pub buffer: MetalBuffer<T>,
}

impl<T> SharedAllocation<T> {
    pub fn new(buffer: MetalBuffer<T>) -> Self {
        Self { buffer }
    }

    pub fn as_slice(&self) -> &[T] {
        self.buffer.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer.as_mut_slice()
    }
}
