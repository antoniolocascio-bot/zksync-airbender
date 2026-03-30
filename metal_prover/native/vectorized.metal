#include <metal_stdlib>
using namespace metal;

#include "field.metal"
#include "memory.metal"

using namespace airbender::field;
using namespace airbender::memory;

namespace airbender {
namespace vectorized {

template <typename T, unsigned WIDTH> struct vectorized_matrix_accessor {
  T internal;
  DEVICE_FORCEINLINE void add_row(const unsigned offset) { internal.add_row(offset); }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { internal.sub_row(offset); }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { internal.add_col(WIDTH * offset); }
};

template <unsigned WIDTH> struct width_to_value_type;
template <> struct width_to_value_type<2> {
  using VALUE_TYPE = ext2_field;
};
template <> struct width_to_value_type<4> {
  using VALUE_TYPE = ext4_field;
};

template <unsigned WIDTH>
struct vectorized_matrix_getter : vectorized_matrix_accessor<matrix_getter<base_field>, WIDTH> {
  using VALUE_TYPE = typename width_to_value_type<WIDTH>::VALUE_TYPE;

  DEVICE_FORCEINLINE VALUE_TYPE get() const {
    base_field coeffs[WIDTH];
    coeffs[0] = this->internal.get();
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = this->internal.get_at_col(i);
    return VALUE_TYPE(coeffs);
  }

  DEVICE_FORCEINLINE VALUE_TYPE get_at_row(const unsigned row) const {
    base_field coeffs[WIDTH];
    coeffs[0] = this->internal.get_at_row(row);
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = this->internal.get(row, i);
    return VALUE_TYPE(coeffs);
  }

  DEVICE_FORCEINLINE VALUE_TYPE get_at_col(const unsigned col) const {
    base_field coeffs[WIDTH];
    const unsigned bf_col = WIDTH * col;
    coeffs[0] = this->internal.get_at_col(bf_col);
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = this->internal.get_at_col(bf_col + i);
    return VALUE_TYPE(coeffs);
  }
};

template <unsigned WIDTH>
struct vectorized_matrix_setter : vectorized_matrix_accessor<matrix_setter<base_field>, WIDTH> {
  using VALUE_TYPE = typename width_to_value_type<WIDTH>::VALUE_TYPE;

  DEVICE_FORCEINLINE void set(const VALUE_TYPE &value) const {
    this->internal.set(value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      this->internal.set_at_col(i, value.base_coefficient_from_flat_idx(i));
  }

  DEVICE_FORCEINLINE void set_at_row(const unsigned row, const VALUE_TYPE &value) const {
    this->internal.set_at_row(row, value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      this->internal.set(row, i, value.base_coefficient_from_flat_idx(i));
  }

  DEVICE_FORCEINLINE void set_at_col(const unsigned col, const VALUE_TYPE &value) const {
    const unsigned bf_col = WIDTH * col;
    this->internal.set_at_col(bf_col, value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      this->internal.set_at_col(bf_col + i, value.base_coefficient_from_flat_idx(i));
  }
};

template <unsigned WIDTH>
struct vectorized_matrix_getter_setter : vectorized_matrix_accessor<matrix_getter_setter<base_field>, WIDTH> {
  using VALUE_TYPE = typename width_to_value_type<WIDTH>::VALUE_TYPE;

  DEVICE_FORCEINLINE VALUE_TYPE get() const {
    base_field coeffs[WIDTH];
    coeffs[0] = this->internal.get();
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = this->internal.get_at_col(i);
    return VALUE_TYPE(coeffs);
  }

  DEVICE_FORCEINLINE void set(const VALUE_TYPE &value) const {
    this->internal.set(value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      this->internal.set_at_col(i, value.base_coefficient_from_flat_idx(i));
  }
};

using vectorized_e4_matrix_getter = vectorized_matrix_getter<4>;
using vectorized_e4_matrix_setter = vectorized_matrix_setter<4>;
using vectorized_e4_matrix_getter_setter = vectorized_matrix_getter_setter<4>;

struct vectorized_e2_matrix_getter : vectorized_matrix_getter<2> {
  DEVICE_FORCEINLINE void get_two_adjacent(const unsigned row, thread ext2_field &val0, thread ext2_field &val1) const {
    device const base_field *p = this->internal.ptr + row;
    const base_field c0_0 = *p;
    const base_field c0_1 = *(p + 1);
    device const base_field *p1 = p + this->internal.stride;
    const base_field c1_0 = *p1;
    const base_field c1_1 = *(p1 + 1);
    val0 = ext2_field{c0_0, c1_0};
    val1 = ext2_field{c0_1, c1_1};
  }
};

struct vectorized_e2_matrix_setter : vectorized_matrix_setter<2> {
  DEVICE_FORCEINLINE void set_two_adjacent(const unsigned row, const ext2_field val0, const ext2_field val1) const {
    device base_field *p = this->internal.ptr + row;
    *p = val0[0];
    *(p + 1) = val1[0];
    device base_field *p1 = p + this->internal.stride;
    *p1 = val0[1];
    *(p1 + 1) = val1[1];
  }
};

} // namespace vectorized
} // namespace airbender
