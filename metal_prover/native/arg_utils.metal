#pragma once
#include <metal_stdlib>
using namespace metal;

#include "field.metal"

using namespace airbender::field;

namespace airbender {
namespace arg_utils {

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef base_field bf;
typedef ext2_field e2;
typedef ext4_field e4;

constant constexpr unsigned NUM_DELEGATION_ARGUMENT_KEY_PARTS = 4;

struct DelegationChallenges {
  e4 linearization_challenges[NUM_DELEGATION_ARGUMENT_KEY_PARTS - 1];
  e4 gamma;
};

constant constexpr unsigned NUM_MACHINE_STATE_LINEARIZATION_CHALLENGES = 3;

struct MachineStateChallenges {
  e4 linearization_challenges[NUM_MACHINE_STATE_LINEARIZATION_CHALLENGES];
  e4 additive_term;
};

struct DelegationRequestMetadata {
  unsigned multiplicity_col;
  unsigned timestamp_col;
  bf memory_timestamp_high_from_circuit_idx;
  unsigned delegation_type_col;
  bf in_cycle_write_idx;
  unsigned abi_mem_offset_high_col;
  bool has_abi_mem_offset_high;
};

struct DelegationProcessingMetadata {
  unsigned multiplicity_col;
  bf delegation_type;
  unsigned write_timestamp_col;
  unsigned abi_mem_offset_high_col;
  bool has_abi_mem_offset_high;
};

constant constexpr unsigned NUM_LOOKUP_ARGUMENT_KEY_PARTS = 4;

struct LookupChallenges {
  e4 linearization_challenges[NUM_LOOKUP_ARGUMENT_KEY_PARTS - 1];
  e4 gamma;
};

constant constexpr unsigned REGISTER_SIZE = 2;
constant constexpr unsigned EXECUTOR_FAMILY_CIRCUIT_DECODER_TABLE_WIDTH = 2 + 1 + 1 + 1 + 1 + REGISTER_SIZE + 1 + 1;
constant constexpr unsigned EXECUTOR_FAMILY_CIRCUIT_DECODER_TABLE_LINEARIZATION_CHALLENGES = EXECUTOR_FAMILY_CIRCUIT_DECODER_TABLE_WIDTH - 1;

struct DecoderTableChallenges {
  e4 linearization_challenges[EXECUTOR_FAMILY_CIRCUIT_DECODER_TABLE_LINEARIZATION_CHALLENGES];
  e4 gamma;
};

struct IntermediateStateLookupLayout {
  unsigned execute;
  unsigned pc;
  unsigned rs1_index;
  unsigned rs2_index;
  unsigned rd_index;
  unsigned rd_is_zero;
  unsigned imm;
  unsigned funct3;
  unsigned circuit_family_extra_mask;
  unsigned intermediate_poly;
  bool has_decoder;
};

struct RangeCheckArgsLayout {
  unsigned num_dst_cols;
  unsigned src_cols_start;
  unsigned bf_args_start;
  unsigned e4_args_start;
};

constant constexpr unsigned NUM_STATE_LINKAGE_CONSTRAINTS = 2;

struct StateLinkageConstraints {
  unsigned srcs[NUM_STATE_LINKAGE_CONSTRAINTS];
  unsigned dsts[NUM_STATE_LINKAGE_CONSTRAINTS];
  unsigned num_constraints;
};

struct MemoryChallenges {
  e4 address_low_challenge;
  e4 address_high_challenge;
  e4 timestamp_low_challenge;
  e4 timestamp_high_challenge;
  e4 value_low_challenge;
  e4 value_high_challenge;
  e4 gamma;
};

constant constexpr unsigned MAX_EXPRESSION_PAIRS = 84;
constant constexpr unsigned MAX_EXPRESSIONS = 2 * MAX_EXPRESSION_PAIRS;
constant constexpr unsigned MAX_TERMS_PER_EXPRESSION = 4;
constant constexpr unsigned MAX_EXPRESSION_TERMS = MAX_TERMS_PER_EXPRESSION * MAX_EXPRESSIONS;

struct TEMPORARYFlattenedLookupExpressionsLayout {
  unsigned coeffs[MAX_EXPRESSION_TERMS];
  u16 col_idxs[MAX_EXPRESSION_TERMS];
  bf constant_terms[MAX_EXPRESSIONS];
  u8 num_terms_per_expression[MAX_EXPRESSIONS];
  u8 bf_dst_cols[MAX_EXPRESSION_PAIRS];
  u8 e4_dst_cols[MAX_EXPRESSION_PAIRS];
  unsigned num_expression_pairs;
  bool constant_terms_are_zero;
};

struct FlattenedLookupExpressionsLayout {
  unsigned coeffs[MAX_EXPRESSION_TERMS];
  u16 col_idxs[MAX_EXPRESSION_TERMS];
  bf constant_terms[MAX_EXPRESSIONS];
  u8 num_terms_per_expression[MAX_EXPRESSIONS];
  u8 bf_dst_cols[MAX_EXPRESSION_PAIRS];
  u8 e4_dst_cols[MAX_EXPRESSION_PAIRS];
  unsigned num_range_check_16_expression_pairs;
  unsigned num_timestamp_expression_pairs;
  bool range_check_16_constant_terms_are_zero;
  bool timestamp_constant_terms_are_zero;
};

constant constexpr unsigned MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM = 4;
constant constexpr unsigned MAX_EXPRESSIONS_FOR_SHUFFLE_RAM = 2 * MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM;
constant constexpr unsigned MAX_EXPRESSION_TERMS_FOR_SHUFFLE_RAM = MAX_TERMS_PER_EXPRESSION * MAX_EXPRESSIONS_FOR_SHUFFLE_RAM;

struct FlattenedLookupExpressionsForShuffleRamLayout {
  unsigned coeffs[MAX_EXPRESSION_TERMS_FOR_SHUFFLE_RAM];
  u16 col_idxs[MAX_EXPRESSION_TERMS_FOR_SHUFFLE_RAM];
  bf constant_terms[MAX_EXPRESSIONS_FOR_SHUFFLE_RAM];
  u8 num_terms_per_expression[MAX_EXPRESSIONS_FOR_SHUFFLE_RAM];
  u8 bf_dst_cols[MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM];
  u8 e4_dst_cols[MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM];
  unsigned num_expression_pairs;
};

// Column type encoding in u16 col index
constant constexpr unsigned COL_TYPE_MASK = 3 << 14;
constant constexpr unsigned COL_IDX_MASK = (1 << 14) - 1;
constant constexpr unsigned COL_TYPE_WITNESS = 0;
constant constexpr unsigned COL_TYPE_MEMORY = 1 << 14;
constant constexpr unsigned COL_TYPE_SETUP = 1 << 15;

DEVICE_FORCEINLINE bf get_witness_or_memory(const unsigned col_idx,
                                             const device bf *witness_ptr, const size_t witness_stride,
                                             const device bf *memory_ptr, const size_t memory_stride) {
  if (col_idx & COL_TYPE_MEMORY)
    return memory_ptr[(col_idx & COL_IDX_MASK) * memory_stride];
  return witness_ptr[col_idx * witness_stride];
}

DEVICE_FORCEINLINE bf get_witness_memory_or_setup(const unsigned col_idx,
                                                    const device bf *witness_ptr, const size_t witness_stride,
                                                    const device bf *memory_ptr, const size_t memory_stride,
                                                    const device bf *setup_ptr, const size_t setup_stride) {
  const unsigned col_type = col_idx & COL_TYPE_MASK;
  switch (col_type) {
  case COL_TYPE_WITNESS:
    return witness_ptr[(col_idx & COL_IDX_MASK) * witness_stride];
  case COL_TYPE_MEMORY:
    return memory_ptr[(col_idx & COL_IDX_MASK) * memory_stride];
  case COL_TYPE_SETUP:
    return setup_ptr[(col_idx & COL_IDX_MASK) * setup_stride];
  default:
    return bf::zero();
  }
}

DEVICE_FORCEINLINE void apply_coeff(const unsigned coeff, thread bf &val) {
  switch (coeff) {
  case 1:
    break;
  case bf::MINUS_ONE:
    val = bf::neg(val);
    break;
  default:
    val = bf::mul(val, bf{coeff});
  }
}

struct LazyInitTeardownLayout {
  unsigned init_address_start;
  unsigned teardown_value_start;
  unsigned teardown_timestamp_start;
  unsigned init_address_aux_low;
  unsigned init_address_aux_high;
  unsigned init_address_intermediate_borrow;
  unsigned init_address_final_borrow;
  unsigned bf_arg_col;
  unsigned e4_arg_col;
};

constant constexpr unsigned MAX_LAZY_INIT_TEARDOWN_SETS = 16;

struct LazyInitTeardownLayouts {
  LazyInitTeardownLayout layouts[MAX_LAZY_INIT_TEARDOWN_SETS];
  unsigned num_init_teardown_sets;
  unsigned grand_product_contributions_start;
  bool process_shuffle_ram_init;
};

struct MachineStateLayout {
  unsigned initial_pc_start;
  unsigned initial_timestamp_start;
  unsigned final_pc_start;
  unsigned final_timestamp_start;
  unsigned arg_col;
  bool process_machine_state;
};

struct MaskArgLayout {
  unsigned arg_col;
  unsigned execute_col;
  bool process_mask;
};

constant constexpr unsigned MAX_SHUFFLE_RAM_ACCESSES = 3;

struct ShuffleRamAccess {
  unsigned address_start;
  unsigned read_timestamp_start;
  unsigned read_value_start;
  unsigned maybe_write_value_start;
  unsigned maybe_is_register_start;
  bool is_write;
  bool is_register_only;
};

struct ShuffleRamAccesses {
  ShuffleRamAccess accesses[MAX_SHUFFLE_RAM_ACCESSES];
  unsigned num_accesses;
  unsigned write_timestamp_start;
};

struct RegisterAccess {
  e4 gamma_plus_one_plus_address_low_contribution;
  unsigned read_timestamp_col;
  unsigned read_value_col;
  unsigned maybe_write_value_col;
  bool is_write;
};

struct IndirectAccess {
  unsigned read_timestamp_col;
  unsigned read_value_col;
  unsigned maybe_write_value_col;
  unsigned maybe_address_derivation_carry_bit_col;
  unsigned maybe_variable_dependent_coeff;
  unsigned maybe_variable_dependent_col;
  unsigned offset_constant;
  bool has_address_derivation_carry_bit;
  bool has_variable_dependent;
  bool has_write;
};

constant constexpr unsigned MAX_REGISTER_ACCESSES = 4;
constant constexpr unsigned MAX_INDIRECT_ACCESSES = 40;

struct RegisterAndIndirectAccesses {
  RegisterAccess register_accesses[MAX_REGISTER_ACCESSES];
  IndirectAccess indirect_accesses[MAX_INDIRECT_ACCESSES];
  unsigned indirect_accesses_per_register_access[MAX_REGISTER_ACCESSES];
  unsigned num_register_accesses;
  unsigned write_timestamp_col;
};

} // namespace arg_utils
} // namespace airbender
