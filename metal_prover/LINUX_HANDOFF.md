# Linux Agent Handoff — Metal Prover Orchestration

## What has been done (Mac-side)

All GPU primitives are implemented and tested on Apple Silicon:

- **`metal_prover/src/backend/mod.rs`** — `ProverBackend` trait: the complete GPU abstraction
- **`metal_prover/src/backend/metal_backend.rs`** — `MetalBackend`: wraps `Arc<MetalProverContext>`, macOS-only
- **`metal_prover/src/backend/stub.rs`** — `StubBackend`: panics at runtime, compiles everywhere
- **`metal_prover/src/types.rs`** — platform-agnostic `Digest = [u32; 8]`
- **`metal_prover/src/field.rs`** — platform-agnostic `BaseField`, `Ext2Field`, `Ext4Field` type aliases
- **Block-level NTT kernels** (`metal_prover/native/ntt/block_stages.metal`) for log_n ≥ 16
- **All 26 unit tests pass** on macOS (run with `RUSTFLAGS="-C debuginfo=0"`)

## What you need to implement (Linux-side)

The prover stages in `metal_prover/src/prover/stage_*.rs` are currently empty stubs that
return no-op outputs. Your task is to implement real orchestration logic, calling through
the `ProverBackend` trait instead of `MetalProverContext` directly.

### Step 1 — Refactor stage signatures to be generic over `B: ProverBackend`

Current (Metal-specific):
```rust
pub fn execute_stage_1(
    circuit_type: CircuitType,
    setup: &SetupPrecomputations,
    ctx: &mut MetalProverContext,
) -> StageOneOutput
```

Target (backend-agnostic):
```rust
pub fn execute_stage_1<B: ProverBackend>(
    circuit_type: CircuitType,
    setup: &SetupPrecomputations,
    backend: &B,
) -> StageOneOutput
```

Do the same for stages 2–5 and `proof::generate_proof`.

### Step 2 — Implement each stage

Use the CUDA reference implementation in `gpu_prover/src/` as the authoritative guide.
Mirror its logic but call `backend.*` methods instead of CUDA kernel launches.

#### Stage 1 — `metal_prover/src/prover/stage_1.rs`
CUDA reference: `gpu_prover/src/stage_1.rs`

Key operations:
```rust
// Forward NTT / LDE
backend.forward_ntt(inputs, outputs, log_n, num_bf_cols, stride, log_ext, coset_idx);

// Build Merkle tree over trace
backend.build_merkle_tree(values, &mut results, log_rows_per_hash, layers_count);
```

#### Stage 2 — `metal_prover/src/prover/stage_2.rs`
CUDA reference: `gpu_prover/src/stage_2.rs`

Key operations:
```rust
// Sort memory accesses
backend.sort_pairs_u32(keys_in, keys_out, vals_in, vals_out, begin_bit, end_bit);

// Run-length encode for lookup multiplicities
let num_runs = backend.run_length_encode_u32(input, unique_out, counts_out);
```

#### Stage 3 — `metal_prover/src/prover/stage_3.rs`
CUDA reference: `gpu_prover/src/stage_3.rs`

Key operations:
```rust
// Forward NTT for each coset in the LDE
backend.forward_ntt(inputs, outputs, log_n, num_bf_cols, stride, log_ext, coset_idx);

// Inverse NTT to recover quotient coefficients
backend.inverse_ntt(inputs, outputs, log_n, num_bf_cols, stride, log_ext, coset_idx);

// Commit quotient columns
backend.build_merkle_tree(values, &mut results, log_rows_per_hash, layers_count);
```

#### Stage 4 — `metal_prover/src/prover/stage_4.rs`
CUDA reference: `gpu_prover/src/stage_4.rs`

FRI folding. Each fold halves the domain; commit each layer:
```rust
// After each fold step:
backend.build_merkle_tree(folded_values, &mut layer_tree, log_rows_per_hash, layers_count);
```

#### Stage 5 — `metal_prover/src/prover/stage_5.rs`
CUDA reference: `gpu_prover/src/stage_5.rs`

```rust
// Gather evaluations at query positions
backend.gather_rows(matrix, indices, dst, row_size, num_rows);

// Gather Merkle authentication paths
backend.gather_merkle_paths(tree, indices, dst, log_size, paths_per_query);
```

### Step 3 — Wire up `generate_proof`

In `metal_prover/src/prover/proof.rs`:
```rust
pub fn generate_proof<B: ProverBackend>(
    circuit_type: CircuitType,
    setup: &SetupPrecomputations,
    backend: &B,
) -> Proof {
    let stage_1 = stage_1::execute_stage_1(circuit_type, setup, backend);
    let stage_2 = stage_2::execute_stage_2(setup, &stage_1, backend);
    let stage_3 = stage_3::execute_stage_3(setup, &stage_1, &stage_2, backend);
    let stage_4 = stage_4::execute_stage_4(setup, &stage_3, backend);
    let stage_5 = stage_5::execute_stage_5(setup, &stage_1, &stage_2, &stage_3, &stage_4, backend);
    assemble_proof(stage_1, stage_2, stage_3, stage_4, stage_5)
}
```

### Step 4 — Define `Proof` struct

In `metal_prover/src/prover/proof.rs`, define a `Proof` struct matching the verifier's
expected format. Cross-reference `verifier_common` crate for the expected layout.

---

## How to build on Linux

```bash
# cargo check: should compile with 0 errors (StubBackend satisfies the trait)
cargo check -p metal_prover

# cargo test: will compile but StubBackend panics at runtime
# Write unit tests with a MockBackend (see below) to test orchestration logic
cargo test -p metal_prover
```

### MockBackend for testing

Since `StubBackend` panics, write a `MockBackend` in the test module that records
which operations were called and returns deterministic dummy data:

```rust
#[cfg(test)]
struct MockBackend { calls: std::cell::RefCell<Vec<String>> }

#[cfg(test)]
impl ProverBackend for MockBackend {
    fn forward_ntt(&self, inputs: &[BF], outputs: &mut [BF], ..) {
        self.calls.borrow_mut().push("forward_ntt".into());
        outputs.copy_from_slice(inputs); // identity for testing
    }
    // ... etc
}
```

---

## Key files to read before starting

| File | Purpose |
|------|---------|
| `metal_prover/src/backend/mod.rs` | `ProverBackend` trait — every GPU op you can call |
| `metal_prover/src/prover/context.rs` | `MetalProverContext` — ignore internals, used only by `MetalBackend` |
| `metal_prover/src/prover/setup.rs` | `SetupPrecomputations` — precomputed data passed to stages |
| `metal_prover/src/circuit_type.rs` | `CircuitType` — domain sizes, LDE factor, circuit layout |
| `gpu_prover/src/stage_1.rs` … `stage_5.rs` | **CUDA reference** — authoritative stage logic |
| `prover/src/prover.rs` | CPU prover — alternative reference for algorithm structure |

## Important trait method signatures

```rust
pub trait ProverBackend: Send + Sync {
    fn forward_ntt(
        &self, inputs: &[BF], outputs: &mut [BF],
        log_n: usize, num_bf_cols: usize, stride: usize,
        log_ext: usize, coset_idx: usize,
    );
    fn inverse_ntt(
        &self, inputs: &[BF], outputs: &mut [BF],
        log_n: usize, num_bf_cols: usize, stride: usize,
        log_ext: usize, coset_idx: usize,
    );
    fn build_merkle_tree(
        &self, values: &[BF], results: &mut [Digest],
        log_rows_per_hash: u32, layers_count: u32,
    );
    fn gather_rows(
        &self, matrix: &[BF], indices: &[u32], dst: &mut [BF],
        row_size: usize, num_rows: u32,
    );
    fn gather_merkle_paths(
        &self, tree: &[Digest], indices: &[u32], dst: &mut [Digest],
        log_size: u32, paths_per_query: u32,
    );
    fn sort_pairs_u32(
        &self, keys_in: &[u32], keys_out: &mut [u32],
        vals_in: &[u32], vals_out: &mut [u32],
        begin_bit: u32, end_bit: u32,
    );
    fn run_length_encode_u32(
        &self, input: &[u32], unique_out: &mut [u32], counts_out: &mut [u32],
    ) -> usize;
}
```

## Branch

All Mac-side work is on `feature/metal-prover`. Branch from there or continue on it.
