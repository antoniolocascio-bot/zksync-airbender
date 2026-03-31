///! Simple element-wise operations: set, add, sub, mul, neg, dbl, sqr, etc.
///!
///! Metal kernels in ops_simple.metal use a flat interface:
///!   (constant T &value/src, device T *result, constant unsigned &count, uint gid)
///! Device memory must be passed via `set_buffer`; scalars via `set_bytes`.

use metal::ComputeCommandEncoderRef;

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

/// Helper to set raw bytes on a compute command encoder (for scalar args).
fn set_bytes_scalar<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>() as u64;
    encoder.set_bytes(index, len, ptr);
}

/// Helper: dispatch a 1D flat kernel over `count` elements.
fn dispatch_1d(
    ctx: &MetalProverContext,
    kernel_name: &str,
    count: u32,
    f: impl FnOnce(&ComputeCommandEncoderRef),
) {
    let pipeline = ctx.get_pipeline(kernel_name);
    let (grid_dim, block_dim) =
        get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);

    let cb = ctx.command_queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    f(&enc);
    enc.dispatch_thread_groups(grid_dim, block_dim);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

// ---------------------------------------------------------------------------
// Zero fill
// ---------------------------------------------------------------------------

/// Zero-fill a MetalBuffer (host-side memset; unified memory).
pub fn set_to_zero<T>(result: &mut MetalBuffer<T>) {
    result.zero_fill();
}

// ---------------------------------------------------------------------------
// Set by value kernels
// (constant T &value, device T *result, constant unsigned &count, gid)
// ---------------------------------------------------------------------------

pub fn set_by_val_bf(value: BF, result: &mut MetalBuffer<BF>, ctx: &MetalProverContext) {
    let count = result.len() as u32;
    dispatch_1d(ctx, "ab_set_by_val_bf_kernel", count, |enc| {
        set_bytes_scalar(enc, &value, 0);
        enc.set_buffer(1, Some(result.metal_buffer()), 0);
        set_bytes_scalar(enc, &count, 2);
    });
}

pub fn set_by_val_e2(value: E2, result: &mut MetalBuffer<E2>, ctx: &MetalProverContext) {
    let count = result.len() as u32;
    dispatch_1d(ctx, "ab_set_by_val_e2_kernel", count, |enc| {
        set_bytes_scalar(enc, &value, 0);
        enc.set_buffer(1, Some(result.metal_buffer()), 0);
        set_bytes_scalar(enc, &count, 2);
    });
}

pub fn set_by_val_e4(value: E4, result: &mut MetalBuffer<E4>, ctx: &MetalProverContext) {
    let count = result.len() as u32;
    dispatch_1d(ctx, "ab_set_by_val_e4_kernel", count, |enc| {
        set_bytes_scalar(enc, &value, 0);
        enc.set_buffer(1, Some(result.metal_buffer()), 0);
        set_bytes_scalar(enc, &count, 2);
    });
}

// ---------------------------------------------------------------------------
// Set by reference / copy kernels
// (device const T *src, device T *dst, constant unsigned &count, gid)
// ---------------------------------------------------------------------------

pub fn set_by_ref_bf(
    src: &MetalBuffer<BF>,
    dst: &mut MetalBuffer<BF>,
    ctx: &MetalProverContext,
) {
    assert_eq!(src.len(), dst.len());
    let count = src.len() as u32;
    dispatch_1d(ctx, "ab_set_by_ref_bf_kernel", count, |enc| {
        enc.set_buffer(0, Some(src.metal_buffer()), 0);
        enc.set_buffer(1, Some(dst.metal_buffer()), 0);
        set_bytes_scalar(enc, &count, 2);
    });
}

pub fn set_by_ref_e2(
    src: &MetalBuffer<E2>,
    dst: &mut MetalBuffer<E2>,
    ctx: &MetalProverContext,
) {
    assert_eq!(src.len(), dst.len());
    let count = src.len() as u32;
    dispatch_1d(ctx, "ab_set_by_ref_e2_kernel", count, |enc| {
        enc.set_buffer(0, Some(src.metal_buffer()), 0);
        enc.set_buffer(1, Some(dst.metal_buffer()), 0);
        set_bytes_scalar(enc, &count, 2);
    });
}

pub fn set_by_ref_e4(
    src: &MetalBuffer<E4>,
    dst: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    assert_eq!(src.len(), dst.len());
    let count = src.len() as u32;
    dispatch_1d(ctx, "ab_set_by_ref_e4_kernel", count, |enc| {
        enc.set_buffer(0, Some(src.metal_buffer()), 0);
        enc.set_buffer(1, Some(dst.metal_buffer()), 0);
        set_bytes_scalar(enc, &count, 2);
    });
}

// ---------------------------------------------------------------------------
// Unary kernels: (device const T *src, device T *dst, count, gid)
// ---------------------------------------------------------------------------

macro_rules! define_unary_op {
    ($fn_name:ident, $fn_name_in_place:ident, $kernel:expr, $ty:ty) => {
        pub fn $fn_name(
            src: &MetalBuffer<$ty>,
            dst: &mut MetalBuffer<$ty>,
            ctx: &MetalProverContext,
        ) {
            assert_eq!(src.len(), dst.len());
            let count = src.len() as u32;
            dispatch_1d(ctx, $kernel, count, |enc| {
                enc.set_buffer(0, Some(src.metal_buffer()), 0);
                enc.set_buffer(1, Some(dst.metal_buffer()), 0);
                set_bytes_scalar(enc, &count, 2);
            });
        }

        pub fn $fn_name_in_place(
            buf: &mut MetalBuffer<$ty>,
            ctx: &MetalProverContext,
        ) {
            let count = buf.len() as u32;
            let metal_buf = buf.metal_buffer().clone();
            dispatch_1d(ctx, $kernel, count, |enc| {
                enc.set_buffer(0, Some(&metal_buf), 0);
                enc.set_buffer(1, Some(&metal_buf), 0);
                set_bytes_scalar(enc, &count, 2);
            });
        }
    };
}

define_unary_op!(neg_bf, neg_bf_in_place, "ab_neg_bf_kernel", BF);
define_unary_op!(neg_e2, neg_e2_in_place, "ab_neg_e2_kernel", E2);
define_unary_op!(neg_e4, neg_e4_in_place, "ab_neg_e4_kernel", E4);

define_unary_op!(dbl_bf, dbl_bf_in_place, "ab_dbl_bf_kernel", BF);
define_unary_op!(sqr_bf, sqr_bf_in_place, "ab_sqr_bf_kernel", BF);
define_unary_op!(inv_bf, inv_bf_in_place, "ab_inv_bf_kernel", BF);

// ---------------------------------------------------------------------------
// Parametrized unary kernels: (src, param, dst, count, gid)
// ---------------------------------------------------------------------------

macro_rules! define_parametrized_op {
    ($fn_name:ident, $fn_name_in_place:ident, $kernel:expr, $ty:ty) => {
        pub fn $fn_name(
            src: &MetalBuffer<$ty>,
            param: u32,
            dst: &mut MetalBuffer<$ty>,
            ctx: &MetalProverContext,
        ) {
            assert_eq!(src.len(), dst.len());
            let count = src.len() as u32;
            dispatch_1d(ctx, $kernel, count, |enc| {
                enc.set_buffer(0, Some(src.metal_buffer()), 0);
                set_bytes_scalar(enc, &param, 1);
                enc.set_buffer(2, Some(dst.metal_buffer()), 0);
                set_bytes_scalar(enc, &count, 3);
            });
        }

        pub fn $fn_name_in_place(
            buf: &mut MetalBuffer<$ty>,
            param: u32,
            ctx: &MetalProverContext,
        ) {
            let count = buf.len() as u32;
            let metal_buf = buf.metal_buffer().clone();
            dispatch_1d(ctx, $kernel, count, |enc| {
                enc.set_buffer(0, Some(&metal_buf), 0);
                set_bytes_scalar(enc, &param, 1);
                enc.set_buffer(2, Some(&metal_buf), 0);
                set_bytes_scalar(enc, &count, 3);
            });
        }
    };
}

define_parametrized_op!(pow_bf, pow_bf_in_place, "ab_pow_bf_kernel", BF);
define_parametrized_op!(shl_bf, shl_bf_in_place, "ab_shl_bf_kernel", BF);
define_parametrized_op!(shr_bf, shr_bf_in_place, "ab_shr_bf_kernel", BF);

// ---------------------------------------------------------------------------
// Binary kernels: (src_a, src_b, dst, count, gid)
// ---------------------------------------------------------------------------

macro_rules! define_binary_op {
    ($fn_name:ident, $kernel:expr, $ta:ty, $tb:ty, $tr:ty) => {
        pub fn $fn_name(
            a: &MetalBuffer<$ta>,
            b: &MetalBuffer<$tb>,
            dst: &mut MetalBuffer<$tr>,
            ctx: &MetalProverContext,
        ) {
            assert_eq!(a.len(), dst.len());
            assert_eq!(b.len(), dst.len());
            let count = dst.len() as u32;
            dispatch_1d(ctx, $kernel, count, |enc| {
                enc.set_buffer(0, Some(a.metal_buffer()), 0);
                enc.set_buffer(1, Some(b.metal_buffer()), 0);
                enc.set_buffer(2, Some(dst.metal_buffer()), 0);
                set_bytes_scalar(enc, &count, 3);
            });
        }
    };
}

// BF x BF -> BF
define_binary_op!(add_bf_bf, "ab_add_bf_bf_kernel", BF, BF, BF);
define_binary_op!(sub_bf_bf, "ab_sub_bf_bf_kernel", BF, BF, BF);
define_binary_op!(mul_bf_bf, "ab_mul_bf_bf_kernel", BF, BF, BF);

// E4 x E4 -> E4
define_binary_op!(add_e4_e4, "ab_add_e4_e4_kernel", E4, E4, E4);
define_binary_op!(sub_e4_e4, "ab_sub_e4_e4_kernel", E4, E4, E4);
define_binary_op!(mul_e4_e4, "ab_mul_e4_e4_kernel", E4, E4, E4);

// Mixed
define_binary_op!(mul_bf_e4, "ab_mul_bf_e4_kernel", BF, E4, E4);
