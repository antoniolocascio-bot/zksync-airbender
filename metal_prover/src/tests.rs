///! Metal Prover Unit Tests
///!
///! Tests for each phase described in METAL_PROVER_TESTING_GUIDE.md:
///! - Phase 2: Field arithmetic GPU vs CPU
///! - Phase 3: Blake2s GPU vs CPU
///! - Phase 4: NTT roundtrip
///! - Phase 5: CUB replacements (scan, reduce, sort)

use std::path::Path;

use blake2s_u32::{Blake2sState, BLAKE2S_BLOCK_SIZE_U32_WORDS, BLAKE2S_DIGEST_SIZE_U32_WORDS};

#[test]
fn test_sanity() {
    assert_eq!(1 + 1, 2);
}

#[test]
fn test_metal_device() {
    // Narrowing test: does creating an MTLDevice stack-overflow?
    let device = metal::Device::system_default()
        .expect("No Metal device");
    println!("Device: {}", device.name());
}

#[test]
fn test_metal_library() {
    // Narrowing test: does loading the library stack-overflow?
    let metallib_path = option_env!("METAL_LIB_PATH")
        .expect("METAL_LIB_PATH not set");
    let device = metal::Device::system_default().expect("No Metal device");
    let _lib = device.new_library_with_file(std::path::Path::new(metallib_path))
        .expect("Failed to load metallib");
    println!("Library loaded OK");
}

#[test]
fn test_device_context_create() {
    // Narrowing test: does DeviceContext::create stack-overflow?
    let device = metal::Device::system_default().expect("No Metal device");
    let _ctx = crate::device_context::DeviceContext::create(&device, 12);
    println!("DeviceContext created OK");
}

#[test]
fn test_generate_powers_buf_ext2() {
    // Narrow down: does generate_powers_buf for Ext2Field overflow?
    use fft::field_utils::domain_generator_for_size;
    use crate::field::Ext2Field;
    use crate::device_context::{CIRCLE_GROUP_LOG_ORDER, FINEST_LOG_COUNT};
    let len = 1usize << FINEST_LOG_COUNT; // 2^5 = 32
    println!("Testing Ext2Field powers, len={}", len);
    let base = domain_generator_for_size::<Ext2Field>(1u64 << CIRCLE_GROUP_LOG_ORDER);
    println!("domain_generator_for_size OK");
    let mut powers = vec![Ext2Field::ONE; len];
    fft::field_utils::distribute_powers_serial::<Ext2Field, Ext2Field>(&mut powers, Ext2Field::ONE, base);
    println!("Ext2Field powers generated OK");
}

#[test]
fn test_inv_sizes() {
    // Narrow down: does the inv_sizes path overflow?
    use crate::field::BaseField;
    use crate::device_context::OMEGA_LOG_ORDER;
    println!("Testing inv_sizes computation");
    let two_inv = BaseField::new(2).inverse().expect("must exist");
    let mut inv_sizes_host = vec![BaseField::ONE; (OMEGA_LOG_ORDER + 1) as usize];
    fft::field_utils::distribute_powers_serial::<BaseField, BaseField>(&mut inv_sizes_host, BaseField::ONE, two_inv);
    println!("inv_sizes OK: {} elements", inv_sizes_host.len());
}

#[test]
fn test_metal_buffer_ext2() {
    // Narrow down: does creating a MetalBuffer<Ext2Field> overflow?
    use crate::field::Ext2Field;
    use crate::device_context::{FINEST_LOG_COUNT, OMEGA_LOG_ORDER};
    let device = metal::Device::system_default().expect("No Metal device");
    println!("Testing MetalBuffer<Ext2Field> creation");
    let len = 1usize << FINEST_LOG_COUNT;
    let buf = crate::device_structures::MetalBuffer::<Ext2Field>::new(&device, len);
    println!("MetalBuffer created, len={}", buf.len());
}

#[test]
fn test_load_from_host_ext2() {
    // Test load_from_host
    use crate::field::Ext2Field;
    use crate::device_structures::MetalBuffer;
    use field::Field;
    let device = metal::Device::system_default().expect("No Metal device");
    let v: Vec<Ext2Field> = vec![Ext2Field::ONE; 32];
    let mut buf = MetalBuffer::<Ext2Field>::new(&device, 32);
    buf.load_from_host(&v);
    eprintln!("load_from_host OK");
}


// Test if ANY function call overflows
fn trivial_fn_with_device_ops(buf: &mut crate::device_structures::MetalBuffer<crate::field::Ext2Field>, src: &[crate::field::Ext2Field]) {
    eprintln!("inside trivial fn");
    let _ = src.len();
    let _ = buf.len();
    eprintln!("trivial fn OK");
}

#[test]
fn test_trivial_fn_call() {
    use crate::field::Ext2Field;
    use crate::device_structures::MetalBuffer;
    use field::Field;
    let device = metal::Device::system_default().expect("No Metal device");
    let v: Vec<Ext2Field> = vec![Ext2Field::ONE; 32];
    let mut buf = MetalBuffer::<Ext2Field>::new(&device, 32);
    eprintln!("before trivial fn call");
    trivial_fn_with_device_ops(&mut buf, &v);
    eprintln!("after trivial fn call");
}

#[test]
fn test_device_context_step_by_step() {
    // Narrow down DeviceContext::create step by step
    use fft::field_utils::domain_generator_for_size;
    use fft::bitreverse_enumeration_inplace;
    use field::Field;
    use crate::field::{BaseField, Ext2Field};
    use crate::device_structures::MetalBuffer;
    use crate::device_context::{FINEST_LOG_COUNT, OMEGA_LOG_ORDER, CIRCLE_GROUP_LOG_ORDER};

    let powers_of_w_coarsest_log_count: u32 = 12;
    let device = metal::Device::system_default().expect("No Metal device");

    let length_fine = 1usize << FINEST_LOG_COUNT;
    let length_coarser = 1usize << (OMEGA_LOG_ORDER - powers_of_w_coarsest_log_count);
    let length_coarsest = 1usize << powers_of_w_coarsest_log_count;

    println!("Step 1: domain_generator_for_size (fine)");
    let gen_fine = domain_generator_for_size::<Ext2Field>(1u64 << CIRCLE_GROUP_LOG_ORDER);
    println!("Step 1 OK");

    use std::io::Write;
    macro_rules! epstep {
        ($($arg:tt)*) => {
            eprintln!($($arg)*);
        }
    }

    epstep!("Step 2: compute powers_of_w_fine");
    let mut powers_host = vec![Ext2Field::ONE; length_fine];
    epstep!("Step 2a: vec created");
    fft::field_utils::distribute_powers_serial::<Ext2Field, Ext2Field>(&mut powers_host, Ext2Field::ONE, gen_fine);
    epstep!("Step 2b: distribute_powers_serial OK");
    let mut buf = MetalBuffer::<Ext2Field>::new(&device, length_fine);
    epstep!("Step 2c: MetalBuffer created");
    buf.load_from_host(&powers_host);
    epstep!("Step 2d: MetalBuffer load OK");

    epstep!("Step 3: coarser buffer, len={}", length_coarser);
    let gen_coarser = domain_generator_for_size::<Ext2Field>(1u64 << OMEGA_LOG_ORDER);
    epstep!("Step 3a: gen_coarser OK");
    let mut powers_coarser = vec![Ext2Field::ONE; length_coarser];
    epstep!("Step 3b: vec coarser created");
    fft::field_utils::distribute_powers_serial::<Ext2Field, Ext2Field>(&mut powers_coarser, Ext2Field::ONE, gen_coarser);
    epstep!("Step 3c: distribute OK");
    let mut buf2 = MetalBuffer::<Ext2Field>::new(&device, length_coarser);
    epstep!("Step 3d: MetalBuffer created");
    buf2.load_from_host(&powers_coarser);
    epstep!("Step 3 OK");

    epstep!("Step 4: coarsest buffer, len={}", length_coarsest);
    let gen_coarsest = domain_generator_for_size::<Ext2Field>(length_coarsest as u64);
    let mut powers_coarsest = vec![Ext2Field::ONE; length_coarsest];
    fft::field_utils::distribute_powers_serial::<Ext2Field, Ext2Field>(&mut powers_coarsest, Ext2Field::ONE, gen_coarsest);
    let mut buf3 = MetalBuffer::<Ext2Field>::new(&device, length_coarsest);
    buf3.load_from_host(&powers_coarsest);
    epstep!("Step 4 OK");

    println!("All steps OK");
}
use field::Field;

use crate::blake2s;
use crate::device_structures::{MetalBuffer, MetalMatrix, MetalMatrixMut, PtrAndStrideWrappingMatrix, MutPtrAndStrideWrappingMatrix};
use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::ntt;
use crate::ops_complex;
use crate::ops_cub::reduce::{reduce_bf, ReduceOperation};
use crate::ops_cub::scan::{inclusive_scan_bf, ScanOperation};
use crate::ops_cub::sort::{sort_keys_u32, SortOrder};
use crate::ops_simple;
use crate::prover::context::{MetalProverContext, MetalProverContextConfig};
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};

type BF = BaseField;
type E2 = Ext2Field;

/// Helper to set raw bytes on a compute encoder.
fn sb<T: Sized>(enc: &metal::ComputeCommandEncoderRef, v: &T, idx: u64) {
    let ptr = v as *const T as *const std::ffi::c_void;
    enc.set_bytes(idx, std::mem::size_of::<T>() as u64, ptr);
}

/// Create a test context by loading the precompiled metallib.
fn make_ctx() -> MetalProverContext {
    let metallib_path = option_env!("METAL_LIB_PATH")
        .expect("METAL_LIB_PATH not set - run `cargo build -p metal_prover` first");
    let config = MetalProverContextConfig::default();
    MetalProverContext::new_with_metallib(config, Some(Path::new(metallib_path)))
}

// ---------------------------------------------------------------------------
// Phase 2: Field arithmetic
// ---------------------------------------------------------------------------

#[test]
fn test_field_set_by_val() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 1024usize;
    let mut buf: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let val = BF::new(42);

    ops_simple::set_by_val_bf(val, &mut buf, &ctx);

    for &x in buf.as_slice() {
        assert_eq!(x, val, "set_by_val_bf: all elements should equal 42");
    }
}

#[test]
fn test_field_add_bf_bf() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 1024usize;
    let a_val = BF::new(100);
    let b_val = BF::new(200);
    let expected = {
        let mut r = a_val;
        Field::add_assign(&mut r, &b_val);
        r
    };

    let mut a_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let mut b_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let mut c_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);

    for x in a_buf.as_mut_slice() { *x = a_val; }
    for x in b_buf.as_mut_slice() { *x = b_val; }

    ops_simple::add_bf_bf(&a_buf, &b_buf, &mut c_buf, &ctx);

    for &x in c_buf.as_slice() {
        assert_eq!(x, expected, "add_bf_bf: GPU result should match CPU");
    }
}

#[test]
fn test_field_mul_bf_bf() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 1024usize;
    let a_val = BF::new(123456789 % (BF::ORDER - 1) + 1);
    let b_val = BF::new(987654321 % (BF::ORDER - 1) + 1);
    let expected = {
        let mut r = a_val;
        Field::mul_assign(&mut r, &b_val);
        r
    };

    let mut a_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let mut b_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let mut c_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);

    for x in a_buf.as_mut_slice() { *x = a_val; }
    for x in b_buf.as_mut_slice() { *x = b_val; }

    ops_simple::mul_bf_bf(&a_buf, &b_buf, &mut c_buf, &ctx);

    for &x in c_buf.as_slice() {
        assert_eq!(x, expected, "mul_bf_bf: GPU result should match CPU");
    }
}

#[test]
fn test_batch_inv_bf() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 128usize;
    let mut in_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let mut out_buf: MetalBuffer<BF> = MetalBuffer::new(device, n);

    // Fill with non-zero values
    for (i, x) in in_buf.as_mut_slice().iter_mut().enumerate() {
        *x = BF::new((i + 1) as u32);
    }

    ops_complex::batch_inv_bf(&in_buf, &mut out_buf, &ctx);

    // Verify: x * inv(x) == 1 for all elements
    for i in 0..n {
        let x = in_buf.as_slice()[i];
        let inv_x = out_buf.as_slice()[i];
        let mut product = x;
        Field::mul_assign(&mut product, &inv_x);
        assert_eq!(product, BF::ONE, "batch_inv: x * inv(x) should be 1 at index {}", i);
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Blake2s
// ---------------------------------------------------------------------------

/// Compute CPU blake2s with reduced rounds (7), matching the Metal kernel.
/// The kernel hashes `words` as a single final block.
fn cpu_blake2s_reduced(words: &[u32]) -> [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS] {
    assert!(words.len() <= BLAKE2S_BLOCK_SIZE_U32_WORDS);
    let mut block = [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS];
    block[..words.len()].copy_from_slice(words);
    let mut state = Blake2sState::new();
    let mut dst = [0u32; BLAKE2S_DIGEST_SIZE_U32_WORDS];
    state.absorb_final_block::<true>(&block, words.len(), &mut dst);
    dst
}

/// Compute CPU blake2s node compression (two-to-one) with reduced rounds.
fn cpu_blake2s_compress(left: &[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS], right: &[u32; BLAKE2S_DIGEST_SIZE_U32_WORDS]) -> [u32; BLAKE2S_DIGEST_SIZE_U32_WORDS] {
    let mut block = [0u32; BLAKE2S_BLOCK_SIZE_U32_WORDS];
    block[..8].copy_from_slice(left);
    block[8..].copy_from_slice(right);
    let mut dst = [0u32; BLAKE2S_DIGEST_SIZE_U32_WORDS];
    Blake2sState::compress_two_to_one::<true>(&block, &mut dst);
    dst
}

#[test]
fn test_blake2s_leaves() {
    let ctx = make_ctx();
    let device = &ctx.device;

    // 256 leaves, each with 4 field elements (1 row per leaf, 4 columns)
    let n_leaves = 256usize;
    let n_cols = 4usize;
    let n_total = n_leaves * n_cols; // column-major: stride = n_leaves

    let mut values_buf: MetalBuffer<BF> = MetalBuffer::new(device, n_total);
    let mut gpu_results: MetalBuffer<blake2s::Digest> = MetalBuffer::new(device, n_leaves);

    // Fill with deterministic values
    for (i, x) in values_buf.as_mut_slice().iter_mut().enumerate() {
        *x = BF::new((i as u32 * 7).wrapping_add(3) % ((1u32 << 31) - 1));
    }

    // GPU: log_rows_per_hash=0 means 1 row per hash
    blake2s::launch_leaves_kernel(&values_buf, &mut gpu_results, 0, &ctx);

    // CPU reference
    // Column-major layout: values_buf[leaf + col * n_leaves]
    let cpu_results: Vec<blake2s::Digest> = (0..n_leaves).map(|leaf_idx| {
        let words: Vec<u32> = (0..n_cols).map(|col| {
            values_buf.as_slice()[leaf_idx + col * n_leaves].0
        }).collect();
        cpu_blake2s_reduced(&words)
    }).collect();

    for i in 0..n_leaves {
        assert_eq!(
            gpu_results.as_slice()[i],
            cpu_results[i],
            "blake2s_leaves: GPU and CPU digests differ at leaf {}",
            i
        );
    }
}

#[test]
fn test_blake2s_nodes() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n_pairs = 64usize;

    let mut leaves: MetalBuffer<blake2s::Digest> = MetalBuffer::new(device, n_pairs * 2);
    let mut gpu_nodes: MetalBuffer<blake2s::Digest> = MetalBuffer::new(device, n_pairs);

    // Fill leaves with deterministic data
    for (i, d) in leaves.as_mut_slice().iter_mut().enumerate() {
        for (j, w) in d.iter_mut().enumerate() {
            *w = (i * 8 + j) as u32;
        }
    }

    blake2s::launch_nodes_kernel(&leaves, &mut gpu_nodes, &ctx);

    // CPU reference using two-to-one compression
    let cpu_nodes: Vec<blake2s::Digest> = (0..n_pairs).map(|i| {
        let left: &[u32; 8] = &leaves.as_slice()[2 * i];
        let right: &[u32; 8] = &leaves.as_slice()[2 * i + 1];
        cpu_blake2s_compress(left, right)
    }).collect();

    for i in 0..n_pairs {
        assert_eq!(
            gpu_nodes.as_slice()[i],
            cpu_nodes[i],
            "blake2s_nodes: GPU and CPU digests differ at node {}",
            i
        );
    }
}

/// Build a Merkle tree in Rust (CPU), matching the Metal layout:
/// results[0..L] = leaves, results[L..2L] = packed node layers.
fn cpu_build_merkle_tree(leaves: &[[u32; 8]]) -> Vec<[u32; 8]> {
    let l = leaves.len();
    assert!(l.is_power_of_two() && l > 0);
    let mut tree = vec![[0u32; 8]; 2 * l];
    tree[..l].copy_from_slice(leaves);
    // Build node layers packed into tree[l..2l]
    let mut src_start = 0usize;  // element index within full tree
    let mut dst_start = l;
    let mut count = l;
    while count > 1 {
        let count_out = count / 2;
        for i in 0..count_out {
            tree[dst_start + i] = cpu_blake2s_compress(&tree[src_start + 2*i], &tree[src_start + 2*i + 1]);
        }
        src_start = dst_start;
        dst_start += count_out;
        count = count_out;
    }
    tree
}

#[test]
fn test_build_merkle_tree_nodes() {
    let ctx = make_ctx();
    let device = &ctx.device;

    // Build with 8 leaves (3 node layers)
    let n_leaves = 8usize;
    let layers_count = 3u32; // log2(8) = 3

    let mut values: MetalBuffer<blake2s::Digest> = MetalBuffer::new(device, n_leaves);
    let mut results: MetalBuffer<blake2s::Digest> = MetalBuffer::new(device, n_leaves);

    // Fill leaves with deterministic data
    for (i, d) in values.as_mut_slice().iter_mut().enumerate() {
        for (j, w) in d.iter_mut().enumerate() {
            *w = ((i * 8 + j) as u32).wrapping_mul(0x9e3779b9);
        }
    }

    blake2s::build_merkle_tree_nodes(&values, &mut results, layers_count, &ctx);

    // CPU reference: build_merkle_tree_nodes(values, results, L)
    // Layer 0: hash values[0..8] → results[0..4]
    // Layer 1: hash results[0..4] → results[4..6]
    // Layer 2: hash results[4..6] → results[6..7]
    let v = values.as_slice();
    let mut cpu = vec![[0u32; 8]; n_leaves];
    // Layer 0
    for i in 0..4 { cpu[i] = cpu_blake2s_compress(&v[2*i], &v[2*i+1]); }
    // Layer 1
    for i in 0..2 { cpu[4+i] = cpu_blake2s_compress(&cpu[2*i], &cpu[2*i+1]); }
    // Layer 2
    cpu[6] = cpu_blake2s_compress(&cpu[4], &cpu[5]);

    let gpu = results.as_slice();
    for i in 0..7 {
        assert_eq!(gpu[i], cpu[i], "build_merkle_tree_nodes mismatch at index {}", i);
    }
}

#[test]
fn test_build_merkle_tree() {
    let ctx = make_ctx();
    let device = &ctx.device;

    // 16 leaves, 4 cols per leaf (1 row per leaf)
    // layers_count = log2(n_leaves) + 1 - log_rows_per_hash - log_tree_cap_size = 4+1-0-0 = 5
    let n_leaves = 16usize;
    let n_cols = 4usize;
    let layers_count = 5u32;

    let mut values: MetalBuffer<BF> = MetalBuffer::new(device, n_leaves * n_cols);
    let mut results: MetalBuffer<blake2s::Digest> = MetalBuffer::new(device, n_leaves * 2);

    for (i, x) in values.as_mut_slice().iter_mut().enumerate() {
        *x = BF::new((i as u32).wrapping_mul(0x9e3779b9) % ((1u32 << 31) - 1));
    }

    blake2s::build_merkle_tree(&values, &mut results, 0, &ctx, layers_count, false);

    // CPU: first compute leaves via the same formula as launch_leaves_kernel
    // (column-major: values[leaf + col * n_leaves])
    let cpu_leaves: Vec<blake2s::Digest> = (0..n_leaves).map(|leaf| {
        let words: Vec<u32> = (0..n_cols).map(|col| values.as_slice()[leaf + col * n_leaves].0).collect();
        cpu_blake2s_reduced(&words)
    }).collect();
    let cpu_tree = cpu_build_merkle_tree(&cpu_leaves);

    let gpu_tree = results.as_slice();
    for i in 0..cpu_tree.len() {
        assert_eq!(gpu_tree[i], cpu_tree[i], "build_merkle_tree mismatch at index {}", i);
    }

    // Verify cap extraction (single root)
    let cap = blake2s::merkle_tree_cap(results.as_slice(), 0);
    assert_eq!(cap.len(), 1);
    assert_eq!(cap[0], cpu_tree[2 * n_leaves - 2], "tree cap should be the root");
}

// ---------------------------------------------------------------------------
// Phase 4: NTT roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_ntt_roundtrip_log8() {
    ntt_roundtrip_inner(8, 1);
}

#[test]
fn test_ntt_roundtrip_log8_multicol() {
    ntt_roundtrip_inner(8, 4);
}

#[test]
fn test_ntt_roundtrip_log16() {
    ntt_roundtrip_inner(16, 2);
}

fn ntt_roundtrip_inner(log_n: usize, num_cols: usize) {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 1usize << log_n;
    // Layout: column-major with stride n. num_cols columns.
    let total = n * num_cols;

    let mut input_buf: MetalBuffer<BF> = MetalBuffer::new(device, total);
    let mut mid_buf: MetalBuffer<BF> = MetalBuffer::new(device, total);
    let mut output_buf: MetalBuffer<BF> = MetalBuffer::new(device, total);

    // Fill with pseudo-random field values (LCG)
    let mut seed = 0xdeadbeef_u32;
    for x in input_buf.as_mut_slice().iter_mut() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *x = BF::new(seed >> 1); // keep < 2^31
    }

    let original: Vec<BF> = input_buf.as_slice().to_vec();

    let in_mat = MetalMatrix::new(input_buf.as_slice(), n);
    let mut mid_mat = MetalMatrixMut::new(mid_buf.as_mut_slice(), n);

    // Forward NTT: bitrev_Z -> natural evals (no extension, coset 0)
    ntt::bitrev_Z_to_natural_evals(&in_mat, &mut mid_mat, log_n, num_cols, 0, 0, &ctx);

    let mid_mat_ro = MetalMatrix::new(mid_buf.as_slice(), n);
    let mut out_mat = MetalMatrixMut::new(output_buf.as_mut_slice(), n);

    // Inverse NTT: natural evals -> bitrev_Z
    ntt::natural_evals_to_bitrev_Z(&mid_mat_ro, &mut out_mat, log_n, num_cols, 0, 0, &ctx);

    // After forward + inverse NTT, output should equal input
    for i in 0..total {
        assert_eq!(
            output_buf.as_slice()[i],
            original[i],
            "NTT roundtrip failed at element {} (log_n={}, num_cols={})",
            i, log_n, num_cols
        );
    }
}

// ---------------------------------------------------------------------------
// Phase 5: CUB replacements
// ---------------------------------------------------------------------------

#[test]
fn test_reduce_bf_sum() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 512usize;
    let mut input: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let mut output: MetalBuffer<BF> = MetalBuffer::new(device, 1);

    // Fill with values 1..n
    let mut cpu_sum = BF::ZERO;
    for (i, x) in input.as_mut_slice().iter_mut().enumerate() {
        *x = BF::new((i + 1) as u32);
        Field::add_assign(&mut cpu_sum, x);
    }

    reduce_bf(&input, &mut output, ReduceOperation::Sum, &ctx);

    assert_eq!(
        output.as_slice()[0],
        cpu_sum,
        "reduce_bf sum: GPU result should match CPU sum"
    );
}

#[test]
fn test_inclusive_scan_bf_sum() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 256usize;
    let mut input: MetalBuffer<BF> = MetalBuffer::new(device, n);
    let mut output: MetalBuffer<BF> = MetalBuffer::new(device, n);

    // Fill with small values
    for (i, x) in input.as_mut_slice().iter_mut().enumerate() {
        *x = BF::new((i % 100 + 1) as u32);
    }

    // CPU inclusive prefix sum
    let cpu_scan: Vec<BF> = {
        let mut acc = BF::ZERO;
        input.as_slice().iter().map(|&x| {
            Field::add_assign(&mut acc, &x);
            acc
        }).collect()
    };

    inclusive_scan_bf(&input, &mut output, ScanOperation::Sum, &ctx);

    for i in 0..n {
        assert_eq!(
            output.as_slice()[i],
            cpu_scan[i],
            "inclusive_scan_bf: mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_sort_keys_u32_ascending() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 256usize;
    let mut keys_in: MetalBuffer<u32> = MetalBuffer::new(device, n);
    let mut keys_out: MetalBuffer<u32> = MetalBuffer::new(device, n);

    // Fill with reverse order
    for (i, k) in keys_in.as_mut_slice().iter_mut().enumerate() {
        *k = (n - 1 - i) as u32;
    }

    sort_keys_u32(&keys_in, &mut keys_out, SortOrder::Ascending, 0, 32, &ctx);

    // Verify sorted ascending
    let sorted = keys_out.as_slice();
    for i in 0..n {
        assert_eq!(sorted[i], i as u32, "sort_keys_u32: element at index {} should be {}", i, i);
    }
}

#[test]
fn test_sort_keys_u32_random() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let n = 1024usize;
    let mut keys_in: MetalBuffer<u32> = MetalBuffer::new(device, n);
    let mut keys_out: MetalBuffer<u32> = MetalBuffer::new(device, n);

    // Pseudo-random values using LCG
    let mut v = 12345u32;
    for k in keys_in.as_mut_slice().iter_mut() {
        v = v.wrapping_mul(1664525).wrapping_add(1013904223);
        *k = v;
    }

    let mut cpu_sorted: Vec<u32> = keys_in.as_slice().to_vec();
    cpu_sorted.sort_unstable();

    sort_keys_u32(&keys_in, &mut keys_out, SortOrder::Ascending, 0, 32, &ctx);

    assert_eq!(
        keys_out.as_slice(),
        cpu_sorted.as_slice(),
        "sort_keys_u32: GPU sort should match CPU sort"
    );
}

#[test]
fn test_bit_reverse_bf() {
    let ctx = make_ctx();
    let device = &ctx.device;

    let log_n = 10usize;
    let n = 1usize << log_n;
    let mut buf: MetalBuffer<BF> = MetalBuffer::new(device, n);

    // Fill with index values
    for (i, x) in buf.as_mut_slice().iter_mut().enumerate() {
        *x = BF::new(i as u32);
    }

    let original: Vec<BF> = buf.as_slice().to_vec();

    ops_complex::bit_reverse_in_place_bf(&mut buf, &ctx);

    // Verify: buf[i] == original[bitrev(i, log_n)]
    for i in 0..n {
        let bitrev_i = (i as u32).reverse_bits() >> (32 - log_n as u32);
        assert_eq!(
            buf.as_slice()[i],
            original[bitrev_i as usize],
            "bit_reverse: mismatch at index {}",
            i
        );
    }
}
