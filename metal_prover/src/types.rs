///! Platform-agnostic type aliases shared between the backend trait and its implementations.

/// Blake2s digest — 8 × u32 words.  Matches `blake2s::STATE_SIZE = 8`.
pub type Digest = [u32; 8];
