///! Prover module: orchestrates the multi-stage proving pipeline on Metal.
///!
///! The prover follows the same 5-stage structure as the CUDA gpu_prover:
///! 1. Witness generation + trace commitment
///! 2. Memory argument + lookup argument
///! 3. Quotient polynomial computation + commitment
///! 4. FRI commitment (folding)
///! 5. Query phase (opening proofs)

pub mod arg_utils;
pub mod callbacks;
pub mod context;
pub mod precomputations;
pub mod proof;
pub mod queries;
pub mod setup;
pub mod stage_1;
pub mod stage_2;
pub mod stage_3;
pub mod stage_4;
pub mod stage_5;

use field::{Mersenne31Complex, Mersenne31Field, Mersenne31Quartic};

type BF = Mersenne31Field;
type E2 = Mersenne31Complex;
type E4 = Mersenne31Quartic;
