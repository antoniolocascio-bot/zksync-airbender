#pragma once
#include <metal_stdlib>
using namespace metal;

#include "common.metal"

namespace airbender {
namespace memory {

// Metal manages caches automatically, so no ld_modifier/st_modifier system.
// All loads/stores are plain pointer dereferences.

template <typename T> DEVICE_FORCEINLINE void swap(thread T &a, thread T &b) {
  T temp = a;
  a = b;
  b = temp;
}

template <unsigned STRIDE> DEVICE_FORCEINLINE unsigned swap_index(const unsigned index) {
  const unsigned i1 = index % STRIDE;
  const unsigned i2 = index / STRIDE;
  const unsigned i3 = i2 * STRIDE * 2;
  return i3 + i1;
}

// Metal uses simd_shuffle_xor instead of __shfl_xor_sync
template <typename T> DEVICE_FORCEINLINE T shfl_xor(T var, ushort laneMask) {
  return simd_shuffle_xor(var, laneMask);
}

// Specialization for uint2: shuffle each component separately
DEVICE_FORCEINLINE uint2 shfl_xor_uint2(const uint2 var, const ushort laneMask) {
  uint2 result;
  result.x = simd_shuffle_xor(var.x, laneMask);
  result.y = simd_shuffle_xor(var.y, laneMask);
  return result;
}

// Specialization for uint4: shuffle each component separately
DEVICE_FORCEINLINE uint4 shfl_xor_uint4(const uint4 var, const ushort laneMask) {
  uint4 result;
  result.x = simd_shuffle_xor(var.x, laneMask);
  result.y = simd_shuffle_xor(var.y, laneMask);
  result.z = simd_shuffle_xor(var.z, laneMask);
  result.w = simd_shuffle_xor(var.w, laneMask);
  return result;
}

template <typename T, unsigned STRIDE_ROW, unsigned COUNT_ROW, unsigned STRIDE_COL = STRIDE_ROW, unsigned COUNT_COL = COUNT_ROW>
DEVICE_FORCEINLINE void transpose_tile(thread T *u, const unsigned lane_id) {
  const bool swap_rows = !(lane_id & STRIDE_ROW);
  if (swap_rows) {
    for (unsigned i = 0; i < COUNT_ROW; i++) {
      const unsigned index = swap_index<STRIDE_ROW>(i);
      swap(u[index], u[index + STRIDE_ROW]);
    }
  }
  for (unsigned i = 0; i < COUNT_COL; i++) {
    const unsigned index = swap_index<STRIDE_COL>(i);
    u[index] = simd_shuffle_xor(u[index], static_cast<ushort>(STRIDE_COL));
  }
  if (swap_rows) {
    for (unsigned i = 0; i < COUNT_ROW; i++) {
      const unsigned index = swap_index<STRIDE_ROW>(i);
      swap(u[index], u[index + STRIDE_ROW]);
    }
  }
}

// Simplified load/store: Metal manages caches automatically, just dereference pointers.
template <class T> DEVICE_FORCEINLINE T load(const device T *address) { return *address; }
template <class T> DEVICE_FORCEINLINE T load(const device T *address, const unsigned offset) { return *(address + offset); }
template <class T> DEVICE_FORCEINLINE void store(device T *address, thread const T &value) { *address = value; }
template <class T> DEVICE_FORCEINLINE void store(device T *address, thread const T &value, const unsigned offset) { *(address + offset) = value; }

// Load/store from threadgroup memory
template <class T> DEVICE_FORCEINLINE T load_tg(const threadgroup T *address) { return *address; }
template <class T> DEVICE_FORCEINLINE void store_tg(threadgroup T *address, thread const T &value) { *address = value; }

// Vector accessor pattern (simplified, no modifier templates)
template <typename T> struct vector_accessor {
  using value_type = T;
  device T *ptr;

  DEVICE_FORCEINLINE T get() const { return *ptr; }
  DEVICE_FORCEINLINE T get(const unsigned i) const { return *(ptr + i); }
  DEVICE_FORCEINLINE void set(thread const T &value) const { *ptr = value; }
  DEVICE_FORCEINLINE void set(const unsigned i, thread const T &value) const { *(ptr + i) = value; }
};

template <typename T> struct vector_getter {
  using value_type = T;
  device const T *ptr;

  DEVICE_FORCEINLINE T get() const { return *ptr; }
  DEVICE_FORCEINLINE T get(const unsigned i) const { return *(ptr + i); }

  DEVICE_FORCEINLINE vector_getter operator+(const unsigned offset) const {
    vector_getter result = *this;
    result.ptr += offset;
    return result;
  }
  DEVICE_FORCEINLINE thread vector_getter &operator+=(const unsigned offset) {
    ptr += offset;
    return *this;
  }
  DEVICE_FORCEINLINE thread vector_getter &operator++() {
    ++ptr;
    return *this;
  }
  DEVICE_FORCEINLINE vector_getter operator++(int) {
    vector_getter pre = *this;
    ++ptr;
    return pre;
  }
};

template <typename T> struct vector_setter {
  using value_type = T;
  device T *ptr;

  DEVICE_FORCEINLINE void set(thread const T &value) const { *ptr = value; }
  DEVICE_FORCEINLINE void set(const unsigned i, thread const T &value) const { *(ptr + i) = value; }

  DEVICE_FORCEINLINE vector_setter operator+(const unsigned offset) const {
    vector_setter result = *this;
    result.ptr += offset;
    return result;
  }
  DEVICE_FORCEINLINE thread vector_setter &operator+=(const unsigned offset) {
    ptr += offset;
    return *this;
  }
};

template <typename T> struct matrix_getter {
  using value_type = T;
  device const T *ptr;
  const size_t stride;

  explicit matrix_getter(size_t stride) : ptr(nullptr), stride(stride) {}
  matrix_getter(device const T *ptr, size_t stride) : ptr(ptr), stride(stride) {}

  DEVICE_FORCEINLINE T get() const { return *ptr; }
  DEVICE_FORCEINLINE T get_at_row(const unsigned row) const { return *(ptr + row); }
  DEVICE_FORCEINLINE T get_at_col(const unsigned col) const { return *(ptr + col * stride); }
  DEVICE_FORCEINLINE T get(const unsigned row, const unsigned col) const { return *(ptr + row + col * stride); }

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { ptr += offset; }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { ptr -= offset; }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { ptr += offset * stride; }
  DEVICE_FORCEINLINE void sub_col(const unsigned offset) { ptr -= offset * stride; }
};

template <typename T> struct matrix_setter {
  using value_type = T;
  device T *ptr;
  const size_t stride;

  explicit matrix_setter(size_t stride) : ptr(nullptr), stride(stride) {}
  matrix_setter(device T *ptr, size_t stride) : ptr(ptr), stride(stride) {}

  DEVICE_FORCEINLINE void set(thread const T &value) const { *ptr = value; }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, thread const T &value) const { *(ptr + row) = value; }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, thread const T &value) const { *(ptr + col * stride) = value; }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, thread const T &value) const { *(ptr + row + col * stride) = value; }

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { ptr += offset; }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { ptr -= offset; }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { ptr += offset * stride; }
  DEVICE_FORCEINLINE void sub_col(const unsigned offset) { ptr -= offset * stride; }
};

template <typename T> struct matrix_getter_setter {
  using value_type = T;
  device T *ptr;
  const size_t stride;

  explicit matrix_getter_setter(size_t stride) : ptr(nullptr), stride(stride) {}
  matrix_getter_setter(device T *ptr, size_t stride) : ptr(ptr), stride(stride) {}

  DEVICE_FORCEINLINE T get() const { return *ptr; }
  DEVICE_FORCEINLINE T get_at_row(const unsigned row) const { return *(ptr + row); }
  DEVICE_FORCEINLINE T get_at_col(const unsigned col) const { return *(ptr + col * stride); }
  DEVICE_FORCEINLINE T get(const unsigned row, const unsigned col) const { return *(ptr + row + col * stride); }
  DEVICE_FORCEINLINE void set(thread const T &value) const { *ptr = value; }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, thread const T &value) const { *(ptr + row) = value; }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, thread const T &value) const { *(ptr + col * stride) = value; }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, thread const T &value) const { *(ptr + row + col * stride) = value; }

  DEVICE_FORCEINLINE void add_row(const unsigned offset) { ptr += offset; }
  DEVICE_FORCEINLINE void sub_row(const unsigned offset) { ptr -= offset; }
  DEVICE_FORCEINLINE void add_col(const unsigned offset) { ptr += offset * stride; }
  DEVICE_FORCEINLINE void sub_col(const unsigned offset) { ptr -= offset * stride; }
};

// Wrapping accessors (modular indexing)
template <typename T> struct wrapping_vector_getter {
  using value_type = typename T::value_type;
  T internal;
  unsigned count;

  DEVICE_FORCEINLINE typename T::value_type get() const { return internal.get(); }
  DEVICE_FORCEINLINE typename T::value_type get(const unsigned i) const { return internal.get(i % count); }
};

template <typename T> struct wrapping_vector_setter {
  using value_type = typename T::value_type;
  T internal;
  unsigned count;

  DEVICE_FORCEINLINE void set(thread const typename T::value_type &value) const { internal.set(value); }
  DEVICE_FORCEINLINE void set(const unsigned i, thread const typename T::value_type &value) const { internal.set(i % count, value); }
};

template <typename T> struct wrapping_matrix_getter {
  using value_type = typename T::value_type;
  T internal;
  unsigned rows;
  unsigned cols;

  DEVICE_FORCEINLINE typename T::value_type get() const { return internal.get(); }
  DEVICE_FORCEINLINE typename T::value_type get_at_row(const unsigned row) const { return internal.get_at_row(row % rows); }
  DEVICE_FORCEINLINE typename T::value_type get_at_col(const unsigned col) const { return internal.get_at_col(col % cols); }
  DEVICE_FORCEINLINE typename T::value_type get(const unsigned row, const unsigned col) const { return internal.get(row % rows, col % cols); }
};

template <typename T> struct wrapping_matrix_setter {
  using value_type = typename T::value_type;
  T internal;
  unsigned rows;
  unsigned cols;

  DEVICE_FORCEINLINE void set(thread const typename T::value_type &value) const { internal.set(value); }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, thread const typename T::value_type &value) const { internal.set_at_row(row % rows, value); }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, thread const typename T::value_type &value) const { internal.set_at_col(col % cols, value); }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, thread const typename T::value_type &value) const { internal.set(row % rows, col % cols, value); }
};

template <typename T> struct wrapping_matrix_getter_setter {
  using value_type = typename T::value_type;
  T internal;
  unsigned rows;
  unsigned cols;

  DEVICE_FORCEINLINE typename T::value_type get() const { return internal.get(); }
  DEVICE_FORCEINLINE typename T::value_type get_at_row(const unsigned row) const { return internal.get_at_row(row % rows); }
  DEVICE_FORCEINLINE typename T::value_type get_at_col(const unsigned col) const { return internal.get_at_col(col % cols); }
  DEVICE_FORCEINLINE typename T::value_type get(const unsigned row, const unsigned col) const { return internal.get(row % rows, col % cols); }
  DEVICE_FORCEINLINE void set(thread const typename T::value_type &value) const { internal.set(value); }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, thread const typename T::value_type &value) const { internal.set_at_row(row % rows, value); }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, thread const typename T::value_type &value) const { internal.set_at_col(col % cols, value); }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, thread const typename T::value_type &value) const { internal.set(row % rows, col % cols, value); }
};

} // namespace memory
} // namespace airbender
