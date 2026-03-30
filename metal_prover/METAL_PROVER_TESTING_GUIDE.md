# Metal Prover - Testing & Fixing Guide

This document is intended for a Claude instance running on Apple Silicon macOS. The `metal_prover` crate was developed on Linux and has never been compiled or tested on macOS. Your job is to get it compiling, passing tests, and producing correct proofs.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools installed (`xcode-select --install`)
- Rust toolchain (`rustup`)
- The zksync-airbender repo checked out on the `feature/metal-prover` branch

## Architecture Overview

The `metal_prover` crate mirrors the structure of `gpu_prover` (CUDA), but targets Apple Silicon's Metal GPU API. Key differences from the CUDA version:

1. **Unified memory**: Apple Silicon shares memory between CPU and GPU. No H2D/D2H copies needed. `MTLBuffer` with `StorageModeShared` gives both CPU and GPU access to the same pointer.
2. **Single GPU**: Apple Silicon has one GPU. Multi-GPU support from `gpu_prover` is removed.
3. **Metal Shading Language (MSL)**: Kernels are written in `.metal` files (C++14-based) instead of `.cu`/`.cuh`.
4. **No CUB library**: CUB operations (parallel scan, radix sort, reduce, RLE) are reimplemented as custom Metal compute kernels.
5. **Rust bindings**: Uses `metal` crate (metal-rs) instead of `era_cudart`.

### Directory Structure

```
metal_prover/
  build/main.rs          # Compiles .metal -> .air -> .metallib via xcrun
  native/                # Metal Shading Language kernel files
    common.metal         # Macros and common defines
    field.metal          # Mersenne31 field arithmetic (base, ext2, ext4)
    memory.metal         # Memory access patterns (simplified from CUDA)
    vectorized.metal     # Vectorized ext2/ext4 matrix accessors
    context.metal        # Twiddle factor structs and power-of-w lookup
    blake2s.metal        # Blake2s hashing for Merkle trees
    ops_simple.metal     # Basic field operation kernels
    ops_complex.metal    # Batch inverse, transpose, bit reverse, fold
    arg_utils.metal      # Stage argument structures and helpers
    stage2.metal         # Lookup argument computation
    stage3.metal         # Constraint evaluation
    stage4.metal         # DEEP quotient polynomial
    monolith.metal       # Monolith hash
    ntt/
      ntt.metal              # NTT helper functions (butterfly, twiddle loading)
      bitrev_Z_to_natural_evals.metal  # Forward NTT (bitrev Z -> natural evals)
      natural_evals_to_bitrev_Z.metal  # Inverse NTT (natural evals -> bitrev Z)
    ops_cub/
      parallel_scan.metal    # Blelloch scan (replaces CUB)
      reduce.metal           # Tree reduction (replaces CUB)
      radix_sort.metal       # 4-bit radix sort (replaces CUB)
      run_length_encode.metal
  src/                   # Rust integration layer
    lib.rs
    field.rs             # repr(C) field types matching MSL layout
    device_context.rs    # Twiddle factor buffers
    device_structures.rs # MetalBuffer<T>, DeviceMatrix, PtrAndStride
    utils.rs             # Grid/threadgroup dimension helpers
    allocator/           # Arena allocator on shared MTLBuffer
    blake2s.rs           # Blake2s kernel dispatch
    ntt/                 # NTT kernel dispatch
    ops_simple.rs        # Simple op dispatch
    ops_complex.rs       # Complex op dispatch
    ops_cub/             # CUB replacement dispatch (scan, reduce, sort)
    prover/              # 5-stage prover
      context.rs         # MetalProverContext
      stage_1.rs - stage_5.rs
    execution/           # Orchestration
      prover.rs
      gpu_worker.rs
    witness/             # Witness generation
```

## Step-by-Step Testing Plan

### Phase 1: Compilation

1. **Compile the Metal shaders**:
   ```bash
   cd zksync-airbender
   cargo build -p metal_prover 2>&1 | head -100
   ```
   The build.rs compiles all `.metal` files via `xcrun metal` and links them into `airbender.metallib`.

   **Expected issues**:
   - MSL syntax errors in `.metal` files (the port was done without access to the Metal compiler)
   - Missing includes or wrong function signatures
   - Type mismatches between MSL and Rust repr(C) structs
   - `metal` crate API changes (we targeted v0.29, check latest)

   **How to fix**:
   - Run `xcrun -sdk macosx metal -c -frecord-sources -I native/ native/field.metal -o /dev/null` to test individual files
   - Fix MSL syntax: `consteval` -> `constexpr`, check address space qualifiers, verify template syntax
   - The `metal` Rust crate may have slightly different API from what was written - check docs

2. **Compile Rust code**:
   ```bash
   cargo check -p metal_prover
   ```

   **Expected issues**:
   - `metal` crate API differences (method names, types)
   - Missing trait implementations
   - Workspace dependency resolution issues

### Phase 2: Unit Tests - Field Arithmetic

Create a test that verifies field arithmetic on the GPU matches CPU:

```rust
// In metal_prover/src/tests.rs or metal_prover/tests/field_test.rs
#[test]
fn test_field_add_mul_on_gpu() {
    // 1. Create MetalProverContext
    // 2. Allocate input buffers with known Mersenne31 values
    // 3. Dispatch ab_add_bf_bf_kernel
    // 4. Read back results from shared buffer
    // 5. Compare against CPU field::Mersenne31Field operations
}
```

Key invariants to check:
- `add(a, b)` matches CPU
- `mul(a, b)` matches CPU
- `inv(x)` satisfies `mul(x, inv(x)) == 1`
- `neg(x)` satisfies `add(x, neg(x)) == 0`
- Extension field operations (ext2, ext4) are consistent

### Phase 3: Unit Tests - Blake2s

```rust
#[test]
fn test_blake2s_leaves() {
    // 1. Create random base field elements
    // 2. Run ab_blake2s_leaves_kernel on Metal
    // 3. Run CPU Blake2s (from blake2s_u32 crate)
    // 4. Compare digests
}

#[test]
fn test_blake2s_nodes() {
    // Similar: hash pairs of digests, compare against CPU
}
```

### Phase 4: Unit Tests - NTT

```rust
#[test]
fn test_ntt_roundtrip() {
    // 1. Create random polynomial coefficients (base field elements)
    // 2. Run forward NTT (bitrev_Z_to_natural_evals)
    // 3. Run inverse NTT (natural_evals_to_bitrev_Z)
    // 4. Verify output matches input (within field arithmetic)
}
```

This is the most critical test. The NTT kernels are complex with SIMD group shuffles.

### Phase 5: Unit Tests - CUB Replacements

```rust
#[test]
fn test_parallel_scan() {
    // 1. Create array of base field elements
    // 2. Run inclusive scan with add
    // 3. Verify against sequential scan on CPU
}

#[test]
fn test_radix_sort() {
    // 1. Create random u32 array
    // 2. Run radix sort
    // 3. Verify sorted
}

#[test]
fn test_reduce() {
    // 1. Create array of base field elements
    // 2. Run reduce with add
    // 3. Verify against CPU sum
}
```

### Phase 6: Integration - Single Stage

Test each prover stage individually:
1. Generate a witness trace on CPU (using `prover` crate's witness evaluator)
2. Run Stage 1 on Metal
3. Compare Merkle tree caps against CPU prover output

### Phase 7: End-to-End Proof

```bash
# Build the CLI with Metal support
cargo build -p cli --features metal_prover

# Run a small proof
./target/release/cli prove --metal examples/fibonacci/fibonacci.elf
```

Compare the proof against CPU-generated proof for the same input.

## Common Issues and Fixes

### MSL Compilation

1. **`consteval` not supported in MSL**: Replace with `constexpr`
2. **Template specialization syntax**: MSL uses C++14, some C++20 patterns may not work
3. **Address space qualifiers**: Every pointer in a kernel must have an address space (`device`, `constant`, `threadgroup`). Missing qualifiers cause compile errors.
4. **`uint2`/`uint4` initialization**: Use `uint2(x, y)` not `{x, y}` in MSL
5. **`simd_shuffle_xor` signature**: `simd_shuffle_xor(data, lane_mask)` - no mask parameter (unlike CUDA which has `__shfl_xor_sync(mask, data, lane)`)
6. **`reverse_bits`**: Available in `<metal_stdlib>`, works on `uint`
7. **`clz`**: Available in `<metal_stdlib>`, works on `uint`

### Rust-Metal Integration

1. **Buffer creation**:
   ```rust
   let buffer = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
   ```
2. **Kernel dispatch pattern**:
   ```rust
   let pipeline = library.get_function("kernel_name", None)?;
   let pipeline_state = device.new_compute_pipeline_state_with_function(&pipeline)?;
   let command_buffer = command_queue.new_command_buffer();
   let encoder = command_buffer.new_compute_command_encoder();
   encoder.set_compute_pipeline_state(&pipeline_state);
   encoder.set_buffer(0, Some(&buffer), 0);
   encoder.dispatch_threadgroups(grid_size, threadgroup_size);
   encoder.end_encoding();
   command_buffer.commit();
   command_buffer.wait_until_completed();
   ```
3. **Reading back results**: With `StorageModeShared`, just cast the buffer contents pointer:
   ```rust
   let ptr = buffer.contents() as *const T;
   let slice = std::slice::from_raw_parts(ptr, count);
   ```

### Performance Tuning

After correctness is established:
1. Use **Metal System Trace** (in Instruments) to profile GPU utilization
2. Check **threadgroup memory usage** - Apple Silicon M-series has 32KB per threadgroup
3. The NTT kernels use `threadgroup` (shared) memory for twiddle caches - verify this fits
4. Experiment with threadgroup sizes: CUDA uses 128 threads/block; Metal may benefit from different sizes
5. Apple Silicon's SIMD group size is 32 (same as CUDA warp size), so SIMD group algorithms translate directly

### Repr(C) Alignment

The Rust `repr(C)` structs in `field.rs` and `device_structures.rs` must exactly match the MSL struct layouts:
- `base_field`: 4 bytes (u32)
- `ext2_field`: 8 bytes, alignas(8)
- `ext4_field`: 16 bytes, alignas(16)
- `PtrAndStride<T>`: pointer + u64 stride

Use `std::mem::size_of::<T>()` assertions to verify.

## Key Reference Files

When debugging, these CUDA originals are the ground truth:
- `gpu_prover/native/field.cuh` - Field arithmetic reference
- `gpu_prover/native/ntt/ntt.cuh` - NTT algorithm reference
- `gpu_prover/native/blake2s.cu` - Blake2s reference
- `gpu_prover/native/stage2.cu`, `stage3.cu`, `stage4.cu` - Stage kernel references
- `gpu_prover/src/prover/context.rs` - Context setup reference
- `gpu_prover/src/prover/stage_1.rs` through `stage_5.rs` - Rust orchestration reference

## Witness Generation

The generated `.cuh` files in `circuit_defs/*/generated/witness_generation_fn.cuh` contain only macro calls and should be reusable. The macro definitions need an MSL-compatible header (`witness_generation_metal.h`) that redefines:
- `KERNEL` -> `kernel void ... [[thread_position_in_grid]]`
- `DEVICE_FORCEINLINE` -> `inline`
- Address space qualifiers on pointers
- Thread ID computation

This is a Phase 2 task after the core prover stages work.

## Success Criteria

1. `cargo build -p metal_prover` succeeds on macOS Apple Silicon
2. All `.metal` files compile without errors via `xcrun metal`
3. Field arithmetic GPU tests pass
4. NTT roundtrip test passes
5. Blake2s tests pass against CPU reference
6. CUB replacement tests pass (scan, sort, reduce)
7. Full proof generation produces a valid proof verifiable by the existing verifier
8. Proof is deterministic (same inputs -> same proof as CPU prover)
