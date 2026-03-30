///! Callback mechanism for the Metal prover pipeline.
///!
///! Callbacks are used to communicate proof data back to the caller
///! after GPU computation completes. With Metal's unified memory,
///! callbacks can directly read from Metal buffers without copies.

use crate::device_structures::MetalBuffer;
use crate::field::BaseField;

type BF = BaseField;

/// Trait for receiving proof stage completion notifications.
pub trait ProverCallback {
    /// Called when stage 1 (witness generation) completes.
    fn on_stage_1_complete(&mut self, tree_cap: &[[u32; 8]]);

    /// Called when stage 2 (argument computation) completes.
    fn on_stage_2_complete(&mut self, tree_cap: &[[u32; 8]]);

    /// Called when stage 3 (quotient computation) completes.
    fn on_stage_3_complete(&mut self, tree_cap: &[[u32; 8]]);

    /// Called when stage 4 (FRI) completes.
    fn on_stage_4_complete(&mut self, fri_caps: &[Vec<[u32; 8]>], final_value: &[BF]);

    /// Called when stage 5 (queries) completes with the full proof.
    fn on_proof_complete(&mut self);
}

/// A collection of closures that implement the callback interface.
///
/// Provides a convenient way to hook into the prover pipeline without
/// implementing a full trait.
pub struct Callbacks<'a> {
    /// Optional closure called after each Merkle tree cap is computed.
    pub on_tree_cap: Option<Box<dyn FnMut(&[[u32; 8]]) + 'a>>,
    /// Optional closure called when the full proof is ready.
    pub on_complete: Option<Box<dyn FnMut() + 'a>>,
}

impl<'a> Callbacks<'a> {
    pub fn new() -> Self {
        Self {
            on_tree_cap: None,
            on_complete: None,
        }
    }

    pub fn with_tree_cap_handler(mut self, handler: impl FnMut(&[[u32; 8]]) + 'a) -> Self {
        self.on_tree_cap = Some(Box::new(handler));
        self
    }

    pub fn with_completion_handler(mut self, handler: impl FnMut() + 'a) -> Self {
        self.on_complete = Some(Box::new(handler));
        self
    }

    /// Notify that a tree cap has been computed.
    pub fn notify_tree_cap(&mut self, cap: &[[u32; 8]]) {
        if let Some(ref mut handler) = self.on_tree_cap {
            handler(cap);
        }
    }

    /// Notify that the proof is complete.
    pub fn notify_complete(&mut self) {
        if let Some(ref mut handler) = self.on_complete {
            handler();
        }
    }
}

impl<'a> Default for Callbacks<'a> {
    fn default() -> Self {
        Self::new()
    }
}
