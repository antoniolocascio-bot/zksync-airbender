#![cfg_attr(not(target_os = "macos"), allow(dead_code))]
#![feature(generic_const_exprs)]

// Platform-agnostic modules — always compiled.
pub mod types;
pub mod field;
pub mod backend;

// Metal-only modules.
#[cfg(not(no_metal))]
pub mod allocator;
#[cfg(not(no_metal))]
pub mod blake2s;
#[cfg(not(no_metal))]
pub mod circuit_type;
#[cfg(not(no_metal))]
pub mod device_context;
#[cfg(not(no_metal))]
pub mod device_structures;
#[cfg(not(no_metal))]
pub mod execution;
#[cfg(not(no_metal))]
pub mod machine_type;
#[cfg(not(no_metal))]
pub mod monolith;
#[cfg(not(no_metal))]
pub mod ntt;
#[cfg(not(no_metal))]
pub mod ops_complex;
#[cfg(not(no_metal))]
pub mod ops_cub;
#[cfg(not(no_metal))]
pub mod ops_simple;
#[cfg(not(no_metal))]
pub mod prover;
#[cfg(not(no_metal))]
pub mod utils;
#[cfg(not(no_metal))]
pub mod witness;

#[cfg(not(no_metal))]
pub use device_structures::*;

#[cfg(all(test, not(no_metal)))]
mod tests;
