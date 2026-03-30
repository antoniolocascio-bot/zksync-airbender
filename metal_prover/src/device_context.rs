use fft::bitreverse_enumeration_inplace;
use fft::field_utils::{distribute_powers_serial, domain_generator_for_size};
use field::Field;
use metal::Device as MTLDevice;

use crate::device_structures::MetalBuffer;
use crate::field::{BaseField, Ext2Field};

pub const OMEGA_LOG_ORDER: u32 = 26;
pub const CIRCLE_GROUP_LOG_ORDER: u32 = 31;
pub const FINEST_LOG_COUNT: u32 = CIRCLE_GROUP_LOG_ORDER - OMEGA_LOG_ORDER;

/// Metadata describing a single layer of precomputed powers.
/// Passed as a kernel argument (not in __constant__ memory, since Metal
/// does not have that concept in the same way as CUDA).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PowersLayerData {
    /// Byte offset into the powers buffer (relative to buffer start).
    pub values_offset: u64,
    pub mask: u32,
    pub log_count: u32,
}

impl PowersLayerData {
    pub fn new(values_offset: u64, log_count: u32) -> Self {
        let mask = (1u32 << log_count) - 1;
        Self {
            values_offset,
            mask,
            log_count,
        }
    }
}

/// Two-layer powers metadata (used for NTT twiddles).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PowersData2Layer {
    pub fine: PowersLayerData,
    pub coarse: PowersLayerData,
}

/// Three-layer powers metadata (used for evaluation domain powers).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PowersData3Layer {
    pub fine: PowersLayerData,
    pub coarser: PowersLayerData,
    pub coarsest: PowersLayerData,
}

/// Holds all precomputed twiddle factors and inverse sizes in Metal buffers.
/// These are passed as kernel arguments rather than stored in CUDA-style
/// __constant__ memory.
pub struct DeviceContext {
    pub powers_of_w_fine: MetalBuffer<Ext2Field>,
    pub powers_of_w_coarser: MetalBuffer<Ext2Field>,
    pub powers_of_w_coarsest: MetalBuffer<Ext2Field>,
    pub powers_of_w_fine_bitrev_for_ntt: MetalBuffer<Ext2Field>,
    pub powers_of_w_coarse_bitrev_for_ntt: MetalBuffer<Ext2Field>,
    pub powers_of_w_inv_fine_bitrev_for_ntt: MetalBuffer<Ext2Field>,
    pub powers_of_w_inv_coarse_bitrev_for_ntt: MetalBuffer<Ext2Field>,
    pub inv_sizes: MetalBuffer<BaseField>,
    pub powers_of_w_coarsest_log_count: u32,
}

/// Generate powers of `base` on the host, optionally bit-reverse, then copy
/// into a MetalBuffer (unified memory, so the copy is just a memcpy).
fn generate_powers_buf<F: Field>(
    device: &MTLDevice,
    base: F,
    len: usize,
    bit_reverse: bool,
) -> MetalBuffer<F> {
    let mut powers_host = vec![F::ONE; len];
    distribute_powers_serial::<F, F>(&mut powers_host, F::ONE, base);
    if bit_reverse {
        bitreverse_enumeration_inplace(&mut powers_host);
    }
    let mut buf = MetalBuffer::<F>::new(device, len);
    buf.copy_from_slice(&powers_host);
    buf
}

impl DeviceContext {
    /// Create a new device context with precomputed twiddle factors.
    ///
    /// `powers_of_w_coarsest_log_count` controls the granularity of the
    /// three-layer decomposition of the evaluation domain.
    pub fn create(device: &MTLDevice, powers_of_w_coarsest_log_count: u32) -> Self {
        assert!(powers_of_w_coarsest_log_count <= OMEGA_LOG_ORDER);

        // Three layers of evaluation-domain powers
        let length_fine = 1usize << FINEST_LOG_COUNT;
        let length_coarser = 1usize << (OMEGA_LOG_ORDER - powers_of_w_coarsest_log_count);
        let length_coarsest = 1usize << powers_of_w_coarsest_log_count;

        let powers_of_w_fine = generate_powers_buf(
            device,
            domain_generator_for_size::<Ext2Field>(1u64 << CIRCLE_GROUP_LOG_ORDER),
            length_fine,
            false,
        );
        let powers_of_w_coarser = generate_powers_buf(
            device,
            domain_generator_for_size::<Ext2Field>(1u64 << OMEGA_LOG_ORDER),
            length_coarser,
            false,
        );
        let powers_of_w_coarsest = generate_powers_buf(
            device,
            domain_generator_for_size::<Ext2Field>(length_coarsest as u64),
            length_coarsest,
            false,
        );

        // Two layers of NTT twiddles (bit-reversed) and their inverses.
        // The fine layer covers half the range (hence -1).
        let ntt_fine_len = 1usize << (OMEGA_LOG_ORDER - powers_of_w_coarsest_log_count - 1);
        let ntt_coarse_len = 1usize << powers_of_w_coarsest_log_count;

        let powers_of_w_fine_bitrev_for_ntt = generate_powers_buf(
            device,
            domain_generator_for_size::<Ext2Field>(1u64 << OMEGA_LOG_ORDER),
            ntt_fine_len,
            true,
        );
        let powers_of_w_coarse_bitrev_for_ntt = generate_powers_buf(
            device,
            domain_generator_for_size::<Ext2Field>((ntt_coarse_len * 2) as u64),
            ntt_coarse_len,
            true,
        );
        let powers_of_w_inv_fine_bitrev_for_ntt = generate_powers_buf(
            device,
            domain_generator_for_size::<Ext2Field>(1u64 << OMEGA_LOG_ORDER)
                .inverse()
                .expect("must exist"),
            ntt_fine_len,
            true,
        );
        let powers_of_w_inv_coarse_bitrev_for_ntt = generate_powers_buf(
            device,
            domain_generator_for_size::<Ext2Field>((ntt_coarse_len * 2) as u64)
                .inverse()
                .expect("must exist"),
            ntt_coarse_len,
            true,
        );

        // Inverse sizes: inv_sizes[i] = 2^{-i} mod p
        let two_inv = BaseField::new(2).inverse().expect("must exist");
        let mut inv_sizes_host = vec![BaseField::ONE; (OMEGA_LOG_ORDER + 1) as usize];
        distribute_powers_serial(&mut inv_sizes_host, BaseField::ONE, two_inv);
        let mut inv_sizes = MetalBuffer::<BaseField>::new(device, inv_sizes_host.len());
        inv_sizes.copy_from_slice(&inv_sizes_host);

        Self {
            powers_of_w_fine,
            powers_of_w_coarser,
            powers_of_w_coarsest,
            powers_of_w_fine_bitrev_for_ntt,
            powers_of_w_coarse_bitrev_for_ntt,
            powers_of_w_inv_fine_bitrev_for_ntt,
            powers_of_w_inv_coarse_bitrev_for_ntt,
            inv_sizes,
            powers_of_w_coarsest_log_count,
        }
    }

    /// Build the 3-layer powers metadata.
    /// The offsets are byte-offsets into the individual MetalBuffers
    /// (always 0 since each layer has its own buffer).
    pub fn get_powers_data_w(&self) -> PowersData3Layer {
        let coarser_log_count = OMEGA_LOG_ORDER - self.powers_of_w_coarsest_log_count;
        PowersData3Layer {
            fine: PowersLayerData::new(0, FINEST_LOG_COUNT),
            coarser: PowersLayerData::new(0, coarser_log_count),
            coarsest: PowersLayerData::new(0, self.powers_of_w_coarsest_log_count),
        }
    }

    /// Build the 2-layer NTT twiddle metadata (forward).
    pub fn get_powers_data_w_bitrev_for_ntt(&self) -> PowersData2Layer {
        let fine_log_count = OMEGA_LOG_ORDER - self.powers_of_w_coarsest_log_count - 1;
        PowersData2Layer {
            fine: PowersLayerData::new(0, fine_log_count),
            coarse: PowersLayerData::new(0, self.powers_of_w_coarsest_log_count),
        }
    }

    /// Build the 2-layer NTT twiddle metadata (inverse).
    pub fn get_powers_data_w_inv_bitrev_for_ntt(&self) -> PowersData2Layer {
        let fine_log_count = OMEGA_LOG_ORDER - self.powers_of_w_coarsest_log_count - 1;
        PowersData2Layer {
            fine: PowersLayerData::new(0, fine_log_count),
            coarse: PowersLayerData::new(0, self.powers_of_w_coarsest_log_count),
        }
    }
}
