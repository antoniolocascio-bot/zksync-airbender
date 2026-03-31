use metal::Buffer as MTLBuffer;
use metal::Device as MTLDevice;
use metal::MTLResourceOptions;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::slice;

// ---------------------------------------------------------------------------
// Pointer-and-stride types matching the Metal kernel argument layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PtrAndStride<T> {
    pub ptr: *const T,
    pub stride: usize,
}

impl<T> PtrAndStride<T> {
    pub fn new(ptr: *const T, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MutPtrAndStride<T> {
    pub ptr: *mut T,
    pub stride: usize,
}

impl<T> MutPtrAndStride<T> {
    pub fn new(ptr: *mut T, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PtrAndStrideWrappingMatrix<T> {
    pub ptr_and_stride: PtrAndStride<T>,
    pub rows: u32,
    pub cols: u32,
}

impl<T> PtrAndStrideWrappingMatrix<T> {
    pub fn new(chunk: &(impl MetalMatrixChunkImpl<T> + ?Sized)) -> Self {
        assert!(chunk.rows() <= u32::MAX as usize);
        assert!(chunk.cols() <= u32::MAX as usize);
        Self {
            ptr_and_stride: chunk.as_ptr_and_stride(),
            rows: chunk.rows() as u32,
            cols: chunk.cols() as u32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MutPtrAndStrideWrappingMatrix<T> {
    pub mut_ptr_and_stride: MutPtrAndStride<T>,
    pub rows: u32,
    pub cols: u32,
}

impl<T> MutPtrAndStrideWrappingMatrix<T> {
    pub fn new(chunk: &mut (impl MetalMatrixChunkMutImpl<T> + ?Sized)) -> Self {
        assert!(chunk.rows() <= u32::MAX as usize);
        assert!(chunk.cols() <= u32::MAX as usize);
        Self {
            mut_ptr_and_stride: chunk.as_mut_ptr_and_stride(),
            rows: chunk.rows() as u32,
            cols: chunk.cols() as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// MetalBuffer -- unified-memory buffer wrapping an MTLBuffer
// ---------------------------------------------------------------------------

/// A typed wrapper around an `MTLBuffer` allocated with `StorageModeShared`.
/// Because Apple Silicon uses unified memory, the same pointer is valid for
/// both CPU and GPU access -- no explicit host-to-device copies are needed.
pub struct MetalBuffer<T> {
    buffer: MTLBuffer,
    len: usize,
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for MetalBuffer<T> {}
unsafe impl<T: Sync> Sync for MetalBuffer<T> {}

impl<T> MetalBuffer<T> {
    /// Allocate a new shared-mode buffer that can hold `len` elements of type `T`.
    pub fn new(device: &MTLDevice, len: usize) -> Self {
        let byte_len = len * size_of::<T>();
        let buffer = device.new_buffer(
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            buffer,
            len,
            _phantom: PhantomData,
        }
    }

    /// Create from an existing `MTLBuffer`. The caller must ensure that the buffer
    /// is large enough to hold `len` elements.
    pub fn from_raw(buffer: MTLBuffer, len: usize) -> Self {
        assert!(buffer.length() as usize >= len * size_of::<T>());
        Self {
            buffer,
            len,
            _phantom: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *const T {
        self.buffer.contents() as *const T
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.buffer.contents() as *mut T
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // Access buffer.contents() directly to avoid recursive trait method dispatch:
        // self.as_mut_ptr() from &mut self resolves to MetalMatrixChunkMutImpl::as_mut_ptr
        // which calls as_raw_mut_ptr() which calls as_mut_ptr() → infinite recursion.
        unsafe { slice::from_raw_parts_mut(self.buffer.contents() as *mut T, self.len) }
    }

    pub fn metal_buffer(&self) -> &MTLBuffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy data from a host slice into this buffer.
    pub fn load_from_host(&mut self, src: &[T]) {
        let n = self.len;
        assert_eq!(src.len(), n, "source slice length mismatch");
        let dst = self.buffer.contents() as *mut T;
        let src_ptr = src.as_ptr();
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst, n);
        }
    }

    /// Copy data from this buffer into a host slice.
    pub fn store_to_host(&self, dst: &mut [T]) {
        assert_eq!(dst.len(), self.len, "destination slice length mismatch");
        unsafe {
            std::ptr::copy_nonoverlapping(self.as_ptr(), dst.as_mut_ptr(), self.len);
        }
    }

    /// Copy data from a host slice into this buffer (alias for load_from_host).
    #[allow(dead_code)]
    pub fn copy_from_slice_host(&mut self, src: &[T]) {
        self.load_from_host(src);
    }

    /// Zero-initialize the entire buffer.
    pub fn zero_fill(&mut self) {
        unsafe {
            // Use buffer.contents() directly to avoid recursive trait method dispatch
            std::ptr::write_bytes(self.buffer.contents() as *mut T, 0, self.len);
        }
    }

    /// Split at a given element offset, returning two sub-buffer views.
    /// Note: these are raw pointer views, not separate MTLBuffers.
    pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        let s = self.as_slice();
        s.split_at(mid)
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        let s = self.as_mut_slice();
        s.split_at_mut(mid)
    }
}

impl<T> Deref for MetalBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for MetalBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

// ---------------------------------------------------------------------------
// Trait abstractions for matrix-like access (port of DeviceMatrix* traits)
// ---------------------------------------------------------------------------

/// Read-only access to a contiguous buffer region with stride information.
pub trait MetalMatrixChunkImpl<T> {
    fn as_raw_ptr(&self) -> *const T;
    fn total_len(&self) -> usize;

    fn stride(&self) -> usize {
        self.total_len()
    }

    fn offset(&self) -> usize {
        0
    }

    fn rows(&self) -> usize {
        self.stride()
    }

    fn cols(&self) -> usize {
        self.total_len() / self.stride()
    }

    fn as_ptr(&self) -> *const T {
        unsafe { self.as_raw_ptr().add(self.offset()) }
    }

    fn as_ptr_and_stride(&self) -> PtrAndStride<T> {
        PtrAndStride::new(self.as_ptr(), self.stride())
    }
}

/// Mutable access to a contiguous buffer region with stride information.
pub trait MetalMatrixChunkMutImpl<T>: MetalMatrixChunkImpl<T> {
    fn as_raw_mut_ptr(&mut self) -> *mut T;

    fn as_mut_ptr(&mut self) -> *mut T {
        let offset = self.offset();
        unsafe { self.as_raw_mut_ptr().add(offset) }
    }

    fn as_mut_ptr_and_stride(&mut self) -> MutPtrAndStride<T> {
        MutPtrAndStride::new(self.as_mut_ptr(), self.stride())
    }
}

// ---------------------------------------------------------------------------
// MetalMatrix and MetalMatrixMut -- strided views
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MetalMatrix<'a, T> {
    data: &'a [T],
    stride: usize,
}

impl<'a, T> MetalMatrix<'a, T> {
    pub fn new(data: &'a [T], stride: usize) -> Self {
        assert_eq!(data.len() % stride, 0);
        Self { data, stride }
    }
}

impl<T> MetalMatrixChunkImpl<T> for MetalMatrix<'_, T> {
    fn as_raw_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    fn total_len(&self) -> usize {
        self.data.len()
    }
    fn stride(&self) -> usize {
        self.stride
    }
}

#[derive(Debug)]
pub struct MetalMatrixMut<'a, T> {
    data: &'a mut [T],
    stride: usize,
}

impl<'a, T> MetalMatrixMut<'a, T> {
    pub fn new(data: &'a mut [T], stride: usize) -> Self {
        assert_eq!(data.len() % stride, 0);
        Self { data, stride }
    }
}

impl<T> MetalMatrixChunkImpl<T> for MetalMatrixMut<'_, T> {
    fn as_raw_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    fn total_len(&self) -> usize {
        self.data.len()
    }
    fn stride(&self) -> usize {
        self.stride
    }
}

impl<T> MetalMatrixChunkMutImpl<T> for MetalMatrixMut<'_, T> {
    fn as_raw_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

// ---------------------------------------------------------------------------
// MetalMatrixChunk / MetalMatrixChunkMut -- sub-views
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MetalMatrixChunk<'a, T> {
    data: &'a [T],
    stride: usize,
    offset: usize,
    rows: usize,
}

impl<'a, T> MetalMatrixChunk<'a, T> {
    pub fn new(data: &'a [T], stride: usize, offset: usize, rows: usize) -> Self {
        assert_eq!(data.len() % stride, 0);
        assert!(offset + rows <= stride);
        Self {
            data,
            stride,
            offset,
            rows,
        }
    }
}

impl<T> MetalMatrixChunkImpl<T> for MetalMatrixChunk<'_, T> {
    fn as_raw_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    fn total_len(&self) -> usize {
        self.data.len()
    }
    fn stride(&self) -> usize {
        self.stride
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn rows(&self) -> usize {
        self.rows
    }
}

#[derive(Debug)]
pub struct MetalMatrixChunkMut<'a, T> {
    data: &'a mut [T],
    stride: usize,
    offset: usize,
    rows: usize,
}

impl<'a, T> MetalMatrixChunkMut<'a, T> {
    pub fn new(data: &'a mut [T], stride: usize, offset: usize, rows: usize) -> Self {
        assert_eq!(data.len() % stride, 0);
        assert!(offset + rows <= stride);
        Self {
            data,
            stride,
            offset,
            rows,
        }
    }
}

impl<T> MetalMatrixChunkImpl<T> for MetalMatrixChunkMut<'_, T> {
    fn as_raw_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    fn total_len(&self) -> usize {
        self.data.len()
    }
    fn stride(&self) -> usize {
        self.stride
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn rows(&self) -> usize {
        self.rows
    }
}

impl<T> MetalMatrixChunkMutImpl<T> for MetalMatrixChunkMut<'_, T> {
    fn as_raw_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

// ---------------------------------------------------------------------------
// Impl for MetalBuffer itself
// ---------------------------------------------------------------------------

impl<T> MetalMatrixChunkImpl<T> for MetalBuffer<T> {
    fn as_raw_ptr(&self) -> *const T {
        self.as_ptr()
    }
    fn total_len(&self) -> usize {
        self.len()
    }
}

impl<T> MetalMatrixChunkMutImpl<T> for MetalBuffer<T> {
    fn as_raw_mut_ptr(&mut self) -> *mut T {
        // Access buffer.contents() directly to avoid recursive trait method dispatch:
        // self.as_mut_ptr() from &mut self would resolve to MetalMatrixChunkMutImpl::as_mut_ptr
        // (which takes &mut self, matching exactly) rather than the inherent as_mut_ptr(&self).
        // That trait default calls as_raw_mut_ptr() → infinite recursion.
        self.buffer.contents() as *mut T
    }
}

// ---------------------------------------------------------------------------
// Impl for plain slices (useful for unified memory access)
// ---------------------------------------------------------------------------

impl<T> MetalMatrixChunkImpl<T> for [T] {
    fn as_raw_ptr(&self) -> *const T {
        self.as_ptr()
    }
    fn total_len(&self) -> usize {
        self.len()
    }
}

impl<T> MetalMatrixChunkMutImpl<T> for [T] {
    fn as_raw_mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
}
