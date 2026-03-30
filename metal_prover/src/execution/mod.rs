///! Execution module: orchestrates proof generation on Metal.
///!
///! Simplified from the CUDA version:
///! - Single GPU only (no multi-GPU manager)
///! - No transfer module (unified memory eliminates H2D copies)
///! - No separate CPU worker for H2D overlap

pub mod gpu_worker;
pub mod prover;
