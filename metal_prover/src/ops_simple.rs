///! Simple element-wise operations: set, add, sub, mul, neg, dbl, sqr, etc.
///!
///! Each operation dispatches a Metal compute kernel that operates on
///! matrix-like buffers (with stride for column-major layout).

use metal::{ComputeCommandEncoderRef, MTLSize};

use crate::device_structures::{
    MetalBuffer, MetalMatrixChunkImpl, MetalMatrixChunkMutImpl, MutPtrAndStride,
    MutPtrAndStrideWrappingMatrix, PtrAndStride, PtrAndStrideWrappingMatrix,
};
use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::prover::context::MetalProverContext;
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

/// Helper to set raw bytes on a compute command encoder.
fn set_bytes<T: Sized>(encoder: &ComputeCommandEncoderRef, value: &T, index: u64) {
    let ptr = value as *const T as *const u8;
    let len = std::mem::size_of::<T>();
    encoder.set_bytes(index, unsafe { std::slice::from_raw_parts(ptr, len) }, len as u64);
}

/// Compute grid/block dims for a 2D matrix dispatch.
fn get_launch_dims(rows: u32, cols: u32) -> (MTLSize, MTLSize) {
    let (mut grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, rows);
    grid_dim.height = cols as u64;
    (grid_dim, block_dim)
}

// ---------------------------------------------------------------------------
// set_to_zero: memset the entire buffer to zero
// ---------------------------------------------------------------------------

/// Zero-fill a MetalBuffer. Since Metal uses unified memory, this is just
/// a host-side memset.
pub fn set_to_zero<T>(result: &mut MetalBuffer<T>) {
    result.zero_fill();
}

// ---------------------------------------------------------------------------
// Generic kernel dispatch helper
// ---------------------------------------------------------------------------

/// Dispatch a simple unary kernel: one input matrix, one output matrix.
fn dispatch_unary_kernel(
    kernel_name: &str,
    values: &PtrAndStrideWrappingMatrix<impl Sized>,
    result: &MutPtrAndStrideWrappingMatrix<impl Sized>,
    ctx: &MetalProverContext,
) {
    let pipeline = ctx.get_pipeline(kernel_name);
    let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_bytes(&encoder, values, 0);
    set_bytes(&encoder, result, 1);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Dispatch a binary kernel: two input matrices, one output matrix.
fn dispatch_binary_kernel(
    kernel_name: &str,
    x: &PtrAndStrideWrappingMatrix<impl Sized>,
    y: &PtrAndStrideWrappingMatrix<impl Sized>,
    result: &MutPtrAndStrideWrappingMatrix<impl Sized>,
    ctx: &MetalProverContext,
) {
    let pipeline = ctx.get_pipeline(kernel_name);
    let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_bytes(&encoder, x, 0);
    set_bytes(&encoder, y, 1);
    set_bytes(&encoder, result, 2);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

// ---------------------------------------------------------------------------
// Set operations
// ---------------------------------------------------------------------------

/// Set all elements of a matrix to a scalar value.
pub fn set_by_val_bf(
    value: BF,
    result: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    ctx: &MetalProverContext,
) {
    let result_wrap = MutPtrAndStrideWrappingMatrix::new(result);
    let pipeline = ctx.get_pipeline("ab_set_by_val_bf_kernel");
    let (grid_dim, block_dim) = get_launch_dims(result_wrap.rows, result_wrap.cols);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_bytes(&encoder, &value, 0);
    set_bytes(&encoder, &result_wrap, 1);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn set_by_val_e2(
    value: E2,
    result: &mut (impl MetalMatrixChunkMutImpl<E2> + ?Sized),
    ctx: &MetalProverContext,
) {
    let result_wrap = MutPtrAndStrideWrappingMatrix::new(result);
    let pipeline = ctx.get_pipeline("ab_set_by_val_e2_kernel");
    let (grid_dim, block_dim) = get_launch_dims(result_wrap.rows, result_wrap.cols);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    set_bytes(&encoder, &value, 0);
    set_bytes(&encoder, &result_wrap, 1);
    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn set_by_val_e4(
    value: E4,
    result: &mut (impl MetalMatrixChunkMutImpl<E4> + ?Sized),
    ctx: &MetalProverContext,
) {
    let result_wrap = MutPtrAndStrideWrappingMatrix::new(result);
    let pipeline = ctx.get_pipeline("ab_set_by_val_e4_kernel");
    let (grid_dim, block_dim) = get_launch_dims(result_wrap.rows, result_wrap.cols);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    set_bytes(&encoder, &value, 0);
    set_bytes(&encoder, &result_wrap, 1);
    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

/// Copy one matrix into another (set by reference).
pub fn set_by_ref<T: Sized>(
    kernel_name: &str,
    values: &(impl MetalMatrixChunkImpl<T> + ?Sized),
    result: &mut (impl MetalMatrixChunkMutImpl<T> + ?Sized),
    ctx: &MetalProverContext,
) {
    let values_wrap = PtrAndStrideWrappingMatrix::new(values);
    let result_wrap = MutPtrAndStrideWrappingMatrix::new(result);
    dispatch_unary_kernel(kernel_name, &values_wrap, &result_wrap, ctx);
}

pub fn set_by_ref_bf(
    values: &(impl MetalMatrixChunkImpl<BF> + ?Sized),
    result: &mut (impl MetalMatrixChunkMutImpl<BF> + ?Sized),
    ctx: &MetalProverContext,
) {
    set_by_ref("ab_set_by_ref_bf_kernel", values, result, ctx);
}

pub fn set_by_ref_e2(
    values: &(impl MetalMatrixChunkImpl<E2> + ?Sized),
    result: &mut (impl MetalMatrixChunkMutImpl<E2> + ?Sized),
    ctx: &MetalProverContext,
) {
    set_by_ref("ab_set_by_ref_e2_kernel", values, result, ctx);
}

pub fn set_by_ref_e4(
    values: &(impl MetalMatrixChunkImpl<E4> + ?Sized),
    result: &mut (impl MetalMatrixChunkMutImpl<E4> + ?Sized),
    ctx: &MetalProverContext,
) {
    set_by_ref("ab_set_by_ref_e4_kernel", values, result, ctx);
}

// ---------------------------------------------------------------------------
// Unary operations: neg, dbl, sqr, inv
// ---------------------------------------------------------------------------

macro_rules! define_unary_op {
    ($fn_name:ident, $fn_name_in_place:ident, $kernel:expr, $ty:ty) => {
        pub fn $fn_name(
            values: &(impl MetalMatrixChunkImpl<$ty> + ?Sized),
            result: &mut (impl MetalMatrixChunkMutImpl<$ty> + ?Sized),
            ctx: &MetalProverContext,
        ) {
            let values_wrap = PtrAndStrideWrappingMatrix::new(values);
            let result_wrap = MutPtrAndStrideWrappingMatrix::new(result);
            dispatch_unary_kernel($kernel, &values_wrap, &result_wrap, ctx);
        }

        pub fn $fn_name_in_place(
            values: &mut (impl MetalMatrixChunkMutImpl<$ty> + ?Sized),
            ctx: &MetalProverContext,
        ) {
            let values_wrap = PtrAndStrideWrappingMatrix::new(values);
            let result_wrap = MutPtrAndStrideWrappingMatrix::new(values);
            dispatch_unary_kernel($kernel, &values_wrap, &result_wrap, ctx);
        }
    };
}

define_unary_op!(neg_bf, neg_bf_in_place, "ab_neg_bf_kernel", BF);
define_unary_op!(neg_e2, neg_e2_in_place, "ab_neg_e2_kernel", E2);
define_unary_op!(neg_e4, neg_e4_in_place, "ab_neg_e4_kernel", E4);

define_unary_op!(dbl_bf, dbl_bf_in_place, "ab_dbl_bf_kernel", BF);
define_unary_op!(dbl_e2, dbl_e2_in_place, "ab_dbl_e2_kernel", E2);
define_unary_op!(dbl_e4, dbl_e4_in_place, "ab_dbl_e4_kernel", E4);

define_unary_op!(sqr_bf, sqr_bf_in_place, "ab_sqr_bf_kernel", BF);
define_unary_op!(sqr_e2, sqr_e2_in_place, "ab_sqr_e2_kernel", E2);
define_unary_op!(sqr_e4, sqr_e4_in_place, "ab_sqr_e4_kernel", E4);

define_unary_op!(inv_bf, inv_bf_in_place, "ab_inv_bf_kernel", BF);
define_unary_op!(inv_e2, inv_e2_in_place, "ab_inv_e2_kernel", E2);
define_unary_op!(inv_e4, inv_e4_in_place, "ab_inv_e4_kernel", E4);

// ---------------------------------------------------------------------------
// Parametrized operations: pow, shl, shr
// ---------------------------------------------------------------------------

fn dispatch_parametrized_kernel(
    kernel_name: &str,
    values: &PtrAndStrideWrappingMatrix<impl Sized>,
    param: u32,
    result: &MutPtrAndStrideWrappingMatrix<impl Sized>,
    ctx: &MetalProverContext,
) {
    let pipeline = ctx.get_pipeline(kernel_name);
    let (grid_dim, block_dim) = get_launch_dims(result.rows, result.cols);

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    set_bytes(&encoder, values, 0);
    set_bytes(&encoder, &param, 1);
    set_bytes(&encoder, result, 2);

    encoder.dispatch_threadgroups(grid_dim, block_dim);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

macro_rules! define_parametrized_op {
    ($fn_name:ident, $fn_name_in_place:ident, $kernel:expr, $ty:ty) => {
        pub fn $fn_name(
            values: &(impl MetalMatrixChunkImpl<$ty> + ?Sized),
            param: u32,
            result: &mut (impl MetalMatrixChunkMutImpl<$ty> + ?Sized),
            ctx: &MetalProverContext,
        ) {
            let values_wrap = PtrAndStrideWrappingMatrix::new(values);
            let result_wrap = MutPtrAndStrideWrappingMatrix::new(result);
            dispatch_parametrized_kernel($kernel, &values_wrap, param, &result_wrap, ctx);
        }

        pub fn $fn_name_in_place(
            values: &mut (impl MetalMatrixChunkMutImpl<$ty> + ?Sized),
            param: u32,
            ctx: &MetalProverContext,
        ) {
            let values_wrap = PtrAndStrideWrappingMatrix::new(values);
            let result_wrap = MutPtrAndStrideWrappingMatrix::new(values);
            dispatch_parametrized_kernel($kernel, &values_wrap, param, &result_wrap, ctx);
        }
    };
}

define_parametrized_op!(pow_bf, pow_bf_in_place, "ab_pow_bf_kernel", BF);
define_parametrized_op!(pow_e2, pow_e2_in_place, "ab_pow_e2_kernel", E2);
define_parametrized_op!(pow_e4, pow_e4_in_place, "ab_pow_e4_kernel", E4);

define_parametrized_op!(shl_bf, shl_bf_in_place, "ab_shl_bf_kernel", BF);
define_parametrized_op!(shl_e2, shl_e2_in_place, "ab_shl_e2_kernel", E2);
define_parametrized_op!(shl_e4, shl_e4_in_place, "ab_shl_e4_kernel", E4);

define_parametrized_op!(shr_bf, shr_bf_in_place, "ab_shr_bf_kernel", BF);
define_parametrized_op!(shr_e2, shr_e2_in_place, "ab_shr_e2_kernel", E2);
define_parametrized_op!(shr_e4, shr_e4_in_place, "ab_shr_e4_kernel", E4);

// ---------------------------------------------------------------------------
// Binary operations: add, sub, mul
// ---------------------------------------------------------------------------

macro_rules! define_binary_op {
    ($fn_name:ident, $kernel:expr, $t0:ty, $t1:ty, $tr:ty) => {
        pub fn $fn_name(
            x: &(impl MetalMatrixChunkImpl<$t0> + ?Sized),
            y: &(impl MetalMatrixChunkImpl<$t1> + ?Sized),
            result: &mut (impl MetalMatrixChunkMutImpl<$tr> + ?Sized),
            ctx: &MetalProverContext,
        ) {
            let x_wrap = PtrAndStrideWrappingMatrix::new(x);
            let y_wrap = PtrAndStrideWrappingMatrix::new(y);
            let result_wrap = MutPtrAndStrideWrappingMatrix::new(result);
            dispatch_binary_kernel($kernel, &x_wrap, &y_wrap, &result_wrap, ctx);
        }
    };
}

// BF x BF -> BF
define_binary_op!(add_bf_bf, "ab_add_bf_bf_kernel", BF, BF, BF);
define_binary_op!(sub_bf_bf, "ab_sub_bf_bf_kernel", BF, BF, BF);
define_binary_op!(mul_bf_bf, "ab_mul_bf_bf_kernel", BF, BF, BF);

// E2 x E2 -> E2
define_binary_op!(add_e2_e2, "ab_add_e2_e2_kernel", E2, E2, E2);
define_binary_op!(sub_e2_e2, "ab_sub_e2_e2_kernel", E2, E2, E2);
define_binary_op!(mul_e2_e2, "ab_mul_e2_e2_kernel", E2, E2, E2);

// E4 x E4 -> E4
define_binary_op!(add_e4_e4, "ab_add_e4_e4_kernel", E4, E4, E4);
define_binary_op!(sub_e4_e4, "ab_sub_e4_e4_kernel", E4, E4, E4);
define_binary_op!(mul_e4_e4, "ab_mul_e4_e4_kernel", E4, E4, E4);

// Mixed: E2 x BF -> E2, E4 x BF -> E4, E4 x E2 -> E4
define_binary_op!(mul_e2_bf, "ab_mul_e2_bf_kernel", E2, BF, E2);
define_binary_op!(mul_e4_bf, "ab_mul_e4_bf_kernel", E4, BF, E4);
define_binary_op!(mul_e4_e2, "ab_mul_e4_e2_kernel", E4, E2, E4);
