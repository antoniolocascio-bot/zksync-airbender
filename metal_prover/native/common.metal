#pragma once

#include <metal_stdlib>
using namespace metal;

#define DEVICE_FORCEINLINE inline __attribute__((always_inline))
#define HOST_DEVICE_FORCEINLINE inline __attribute__((always_inline))

// In Metal, kernel functions are declared with the `kernel` keyword
// No equivalent of CUDA's extern "C" is needed
#define EXTERN

#define likely(x) (x)
#define unlikely(x) (x)
