#pragma once
#include <metal_stdlib>
using namespace metal;

#include "field.metal"
#include "memory.metal"

using namespace airbender::field;
using namespace airbender::memory;

namespace airbender {
namespace vectorized {

template <unsigned WIDTH> struct width_to_value_type;
template <> struct width_to_value_type<2> {
  using VALUE_TYPE = ext2_field;
};
template <> struct width_to_value_type<4> {
  using VALUE_TYPE = ext4_field;
};

// Metal doesn't support class inheritance, so we use composition and inline the
// vectorized_matrix_accessor methods directly into each struct.

template <unsigned WIDTH>
struct vectorized_matrix_getter {
  using VALUE_TYPE = typename width_to_value_type<WIDTH>::VALUE_TYPE;
  matrix_getter<base_field> internal;

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { internal.add_row(offset); }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { internal.sub_row(offset); }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { internal.add_col(WIDTH * offset); }

  DEVICE_FORCEINLINE VALUE_TYPE get() const {
    base_field coeffs[WIDTH];
    coeffs[0] = internal.get();
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = internal.get_at_col(i);
    return VALUE_TYPE(coeffs);
  }

  DEVICE_FORCEINLINE VALUE_TYPE get_at_row(const unsigned row) const {
    base_field coeffs[WIDTH];
    coeffs[0] = internal.get_at_row(row);
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = internal.get(row, i);
    return VALUE_TYPE(coeffs);
  }

  DEVICE_FORCEINLINE VALUE_TYPE get_at_col(const unsigned col) const {
    base_field coeffs[WIDTH];
    const unsigned bf_col = WIDTH * col;
    coeffs[0] = internal.get_at_col(bf_col);
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = internal.get_at_col(bf_col + i);
    return VALUE_TYPE(coeffs);
  }
};

template <unsigned WIDTH>
struct vectorized_matrix_setter {
  using VALUE_TYPE = typename width_to_value_type<WIDTH>::VALUE_TYPE;
  matrix_setter<base_field> internal;

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { internal.add_row(offset); }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { internal.sub_row(offset); }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { internal.add_col(WIDTH * offset); }

  DEVICE_FORCEINLINE void set(thread const VALUE_TYPE &value) const {
    internal.set(value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      internal.set_at_col(i, value.base_coefficient_from_flat_idx(i));
  }

  DEVICE_FORCEINLINE void set_at_row(const unsigned row, thread const VALUE_TYPE &value) const {
    internal.set_at_row(row, value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      internal.set(row, i, value.base_coefficient_from_flat_idx(i));
  }

  DEVICE_FORCEINLINE void set_at_col(const unsigned col, thread const VALUE_TYPE &value) const {
    const unsigned bf_col = WIDTH * col;
    internal.set_at_col(bf_col, value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      internal.set_at_col(bf_col + i, value.base_coefficient_from_flat_idx(i));
  }
};

template <unsigned WIDTH>
struct vectorized_matrix_getter_setter {
  using VALUE_TYPE = typename width_to_value_type<WIDTH>::VALUE_TYPE;
  matrix_getter_setter<base_field> internal;

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { internal.add_row(offset); }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { internal.sub_row(offset); }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { internal.add_col(WIDTH * offset); }

  DEVICE_FORCEINLINE VALUE_TYPE get() const {
    base_field coeffs[WIDTH];
    coeffs[0] = internal.get();
    for (unsigned i = 1; i < WIDTH; i++)
      coeffs[i] = internal.get_at_col(i);
    return VALUE_TYPE(coeffs);
  }

  DEVICE_FORCEINLINE void set(thread const VALUE_TYPE &value) const {
    internal.set(value.base_coefficient_from_flat_idx(0));
    for (unsigned i = 1; i < WIDTH; i++)
      internal.set_at_col(i, value.base_coefficient_from_flat_idx(i));
  }
};

using vectorized_e4_matrix_getter = vectorized_matrix_getter<4>;
using vectorized_e4_matrix_setter = vectorized_matrix_setter<4>;
using vectorized_e4_matrix_getter_setter = vectorized_matrix_getter_setter<4>;

// Metal doesn't support inheritance, so these embed the <2> template as a member.
struct vectorized_e2_matrix_getter {
  vectorized_matrix_getter<2> base;

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { base.add_row(offset); }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { base.sub_row(offset); }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { base.add_col(offset); }
  DEVICE_FORCEINLINE ext2_field get() const { return base.get(); }
  DEVICE_FORCEINLINE ext2_field get_at_row(const unsigned row) const { return base.get_at_row(row); }
  DEVICE_FORCEINLINE ext2_field get_at_col(const unsigned col) const { return base.get_at_col(col); }

  DEVICE_FORCEINLINE void get_two_adjacent(const unsigned row, thread ext2_field &val0, thread ext2_field &val1) const {
    device const base_field *p = base.internal.ptr + row;
    const base_field c0_0 = *p;
    const base_field c0_1 = *(p + 1);
    device const base_field *p1 = p + base.internal.stride;
    const base_field c1_0 = *p1;
    const base_field c1_1 = *(p1 + 1);
    val0 = ext2_field(c0_0, c1_0);
    val1 = ext2_field(c0_1, c1_1);
  }
};

struct vectorized_e2_matrix_setter {
  vectorized_matrix_setter<2> base;

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { base.add_row(offset); }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { base.sub_row(offset); }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { base.add_col(offset); }
  DEVICE_FORCEINLINE void set(thread const ext2_field &value) const { base.set(value); }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, thread const ext2_field &value) const { base.set_at_row(row, value); }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, thread const ext2_field &value) const { base.set_at_col(col, value); }

  DEVICE_FORCEINLINE void set_two_adjacent(const unsigned row, const ext2_field val0, const ext2_field val1) const {
    device base_field *p = base.internal.ptr + row;
    *p = val0[0];
    *(p + 1) = val1[0];
    device base_field *p1 = p + base.internal.stride;
    *p1 = val0[1];
    *(p1 + 1) = val1[1];
  }
};

} // namespace vectorized
} // namespace airbender
