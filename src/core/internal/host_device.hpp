// src/core/internal/host_device.hpp
//
// The single home of the host/device portability macro (STEPPE_HD) and the
// debug-only facilities (STEPPE_DEBUG_ONLY, STEPPE_ASSERT). CUDA-free and
// CUDA-compilable, so both the host core and the device kernels share one
// definition of each.
//
// Reference: docs/reference/src_core_internal_host_device.hpp.md
#ifndef STEPPE_CORE_INTERNAL_HOST_DEVICE_HPP
#define STEPPE_CORE_INTERNAL_HOST_DEVICE_HPP

#include <cassert>

// STEPPE_HD — the host/device qualifier — reference §2
#ifndef STEPPE_HD
#  if defined(__CUDACC__)
#    define STEPPE_HD __host__ __device__
#  else
#    define STEPPE_HD
#  endif
#endif

// STEPPE_DEBUG_ONLY — debug-only statements — reference §3
#if defined(NDEBUG)
#  define STEPPE_DEBUG_ONLY(...) ((void)0)
#else
#  define STEPPE_DEBUG_ONLY(...) __VA_ARGS__
#endif

// STEPPE_ASSERT — debug-only precondition check — reference §4
#if defined(NDEBUG)
#  define STEPPE_ASSERT(cond, msg) ((void)sizeof(cond), (void)sizeof(msg), (void)0)
#else
#  define STEPPE_ASSERT(cond, msg) assert((cond) && (msg))
#endif

#endif  // STEPPE_CORE_INTERNAL_HOST_DEVICE_HPP
