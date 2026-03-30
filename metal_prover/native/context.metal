#include <metal_stdlib>
using namespace metal;

#include "field.metal"
#include "memory.metal"

using namespace airbender::field;
using namespace airbender::memory;

namespace airbender {
namespace field {

static constexpr unsigned OMEGA_LOG_ORDER = 26;
static constexpr unsigned CIRCLE_GROUP_LOG_ORDER = 31;

struct powers_layer_data {
  device const ext2_field *values;
  unsigned mask;
  unsigned log_count;
};

struct powers_data_2_layer {
  powers_layer_data fine;
  powers_layer_data coarse;
};

struct powers_data_3_layer {
  powers_layer_data fine;
  powers_layer_data coarser;
  powers_layer_data coarsest;
};

// In Metal, we pass these as constant buffer arguments rather than using __constant__ globals.
// The host side will bind them as buffer arguments to each kernel that needs them.

DEVICE_FORCEINLINE ext2_field get_power(const thread powers_data_3_layer &data, const unsigned index, const bool inverse) {
  const unsigned idx = inverse ? (1u << CIRCLE_GROUP_LOG_ORDER) - index : index;

  const unsigned coarsest_idx = (idx >> (data.fine.log_count + data.coarser.log_count)) & data.coarsest.mask;
  ext2_field val = data.coarsest.values[coarsest_idx];

  const unsigned coarser_idx = (idx >> data.fine.log_count) & data.coarser.mask;
  if (coarser_idx != 0)
    val = ext2_field::mul(val, data.coarser.values[coarser_idx]);

  const unsigned fine_idx = idx & data.fine.mask;
  if (fine_idx != 0)
    val = ext2_field::mul(val, data.fine.values[fine_idx]);

  return val;
}

DEVICE_FORCEINLINE ext2_field get_power_of_w(const thread powers_data_3_layer &powers_data_w, const unsigned index, const bool inverse) {
  return get_power(powers_data_w, index, inverse);
}

// Twiddle factor lookup for NTT (takes powers_data_2_layer by reference)
template <bool inverse>
DEVICE_FORCEINLINE ext2_field get_twiddle(const thread powers_data_2_layer &data, const unsigned i) {
  unsigned fine_idx = (i >> data.coarse.log_count) & data.fine.mask;
  unsigned coarse_idx = i & data.coarse.mask;
  auto coarse = data.coarse.values[coarse_idx];
  if (fine_idx == 0)
    return coarse;
  auto fine = data.fine.values[fine_idx];
  return ext2_field::mul(fine, coarse);
}

} // namespace field
} // namespace airbender
