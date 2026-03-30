///! Argument utilities for the Metal prover.
///!
///! Helper functions for constructing and manipulating polynomial arguments
///! used in the permutation, memory, and lookup protocols.

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::prover::context::MetalProverContext;

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

/// Compute the product of all elements in a buffer.
///
/// Used to compute Z(omega^n) for the permutation argument grand product.
pub fn compute_grand_product_check(
    column: &MetalBuffer<E4>,
    ctx: &MetalProverContext,
) -> E4 {
    if column.is_empty() {
        return E4::default();
    }
    // With unified memory, we can read the last element directly.
    // The grand product Z polynomial's last value should equal 1
    // if the permutation argument is satisfied.
    column.as_slice()[column.len() - 1]
}

/// Initialize a buffer with the identity permutation sigma(i) = i.
pub fn init_identity_permutation(
    result: &mut MetalBuffer<u32>,
) {
    let slice = result.as_mut_slice();
    for (i, val) in slice.iter_mut().enumerate() {
        *val = i as u32;
    }
}

/// Compute (challenge - sigma(i) * gamma) for the permutation argument.
///
/// This is one half of the grand-product ratio:
///   Z(x) = prod_i (f(i) + beta*sigma(i) + gamma) / (f(i) + beta*i + gamma)
pub fn compute_permutation_numerators(
    values: &MetalBuffer<BF>,
    permutation: &MetalBuffer<u32>,
    beta: E4,
    gamma: E4,
    result: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    // TODO: Dispatch a custom Metal kernel that computes:
    //   result[i] = values[i] + beta * permutation[i] + gamma
    // for each i.
    //
    // This is a fused multiply-add operation over the extension field.
    let _ = (values, permutation, beta, gamma, result, ctx);
}

/// Compute the denominators for the permutation argument ratio.
pub fn compute_permutation_denominators(
    values: &MetalBuffer<BF>,
    beta: E4,
    gamma: E4,
    result: &mut MetalBuffer<E4>,
    ctx: &MetalProverContext,
) {
    // TODO: Dispatch a custom Metal kernel that computes:
    //   result[i] = values[i] + beta * i + gamma
    // for each i.
    let _ = (values, beta, gamma, result, ctx);
}
