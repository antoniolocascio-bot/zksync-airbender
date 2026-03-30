///! CUB-equivalent parallel primitives for Metal.
///!
///! CUDA's CUB library provides device-wide parallel scan, reduce, and sort.
///! On Metal, we implement these as custom compute kernels dispatched through
///! the Metal command queue. The algorithms are functionally equivalent but
///! adapted for the Metal compute model.

pub mod reduce;
pub mod scan;
pub mod sort;

pub use reduce::*;
pub use scan::*;
pub use sort::*;
