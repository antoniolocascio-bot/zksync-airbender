use common_constants::TimestampScalar;

use crate::vm::{RamPeek, RAM};

/// Lightweight RAM implementation that stores only values (no timestamps).
///
/// Uses `Vec<u32>` (4 bytes/word) instead of `Vec<Register>` (16 bytes/word),
/// and benefits from `alloc_zeroed` for lazy zero-page allocation.
/// Suitable for non-proving runs where timestamps are not needed.
pub struct SimpleRam<const ROM_BOUND_SECOND_WORD_BITS: usize> {
    pub(crate) backing: Vec<u32>,
}

impl<const ROM_BOUND_SECOND_WORD_BITS: usize> SimpleRam<ROM_BOUND_SECOND_WORD_BITS> {
    pub fn from_rom_content(content: &[u32], total_size_bytes: usize) -> Self {
        assert!(total_size_bytes.is_power_of_two());
        let rom_bytes = 1 << (16 + ROM_BOUND_SECOND_WORD_BITS);
        assert!(total_size_bytes > rom_bytes);
        let num_rom_words = rom_bytes / core::mem::size_of::<u32>();

        assert!(content.len() <= num_rom_words);
        let ram_words = total_size_bytes / core::mem::size_of::<u32>();

        let mut backing = vec![0u32; ram_words];
        backing[..content.len()].copy_from_slice(content);

        Self { backing }
    }
}

impl<const ROM_BOUND_SECOND_WORD_BITS: usize> RamPeek for SimpleRam<ROM_BOUND_SECOND_WORD_BITS> {
    #[inline(always)]
    fn peek_word(&self, address: u32) -> u32 {
        debug_assert_eq!(address % 4, 0);
        unsafe {
            let word_idx = (address / 4) as usize;
            debug_assert!(word_idx < self.backing.len());
            *self.backing.get_unchecked(word_idx)
        }
    }
}

impl<const ROM_BOUND_SECOND_WORD_BITS: usize> RAM for SimpleRam<ROM_BOUND_SECOND_WORD_BITS> {
    #[inline(always)]
    fn mask_read_for_witness(&self, _address: &mut u32, _value: &mut u32) {}

    #[inline(always)]
    fn read_word(&mut self, address: u32, _timestamp: TimestampScalar) -> (TimestampScalar, u32) {
        debug_assert_eq!(address % 4, 0);
        unsafe {
            let word_idx = (address / 4) as usize;
            debug_assert!(word_idx < self.backing.len());
            let value = *self.backing.get_unchecked(word_idx);
            (0, value)
        }
    }

    #[inline(always)]
    fn skip_if_replaying(&mut self, _num_snapshots: usize) {
        panic!("must not be used in replayer");
    }

    #[inline(always)]
    fn write_word(
        &mut self,
        address: u32,
        word: u32,
        _timestamp: TimestampScalar,
    ) -> (TimestampScalar, u32) {
        debug_assert_eq!(address % 4, 0);
        unsafe {
            let word_idx = (address / 4) as usize;
            debug_assert!(word_idx < self.backing.len());
            if word_idx < (1 << (16 + ROM_BOUND_SECOND_WORD_BITS)) / core::mem::size_of::<u32>() {
                panic!("attempt to write into ROM range");
            }
            let slot = self.backing.get_unchecked_mut(word_idx);
            let old_value = *slot;
            *slot = word;
            (0, old_value)
        }
    }
}
