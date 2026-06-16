// src/core/internal/host_device.hpp
//
// THE single home of the host/device portability macro and the debug-only
// facilities (architecture.md §2 DRY, §7, §8 DRY-internals table; ROADMAP §5).
//
// Three cross-cutting helpers that were previously open-coded in several places:
//
//   STEPPE_HD          — __host__ __device__ under nvcc, nothing otherwise, so a
//                        shared per-element primitive (f2_estimator.hpp,
//                        decode_af.hpp) compiles BOTH into the host-pure `core`
//                        TUs and into the device kernels from ONE definition. It
//                        was previously #define'd byte-identically in
//                        f2_estimator.hpp AND decode_af.hpp with no #undef — a
//                        header-macro leak that is ill-formed the moment the two
//                        definitions drift (architecture.md §2). It now lives
//                        here once; both headers include this.
//
//   STEPPE_DEBUG_ONLY  — expands its argument in debug builds and to nothing
//                        under NDEBUG. The §7 "debug sync attributes a kernel
//                        fault" gate and any other debug-only statement route
//                        through this one macro instead of open-coding
//                        `#if defined(NDEBUG)` per site (architecture.md §7).
//
//   STEPPE_ASSERT      — a debug-only precondition check (architecture.md §2
//                        fail-fast, §7). Compiled out under NDEBUG; in debug it
//                        is the standard `assert`. The §7 debug-assert facility
//                        the cleanup backlog repeatedly names as the carrier for
//                        the cheap "enforce the documented precondition" fixes.
//
// This header is CUDA-FREE-compilable (no CUDA include) and CUDA-compilable
// (pure preprocessor + <cassert>), so it lives in `core/internal/` and is
// consumed via the steppe::core_internal INTERFACE target by both layers
// (architecture.md §4, §8). It is NOT in the public ABI (§16).
#ifndef STEPPE_CORE_INTERNAL_HOST_DEVICE_HPP
#define STEPPE_CORE_INTERNAL_HOST_DEVICE_HPP

#include <cassert>

// ---------------------------------------------------------------------------
// STEPPE_HD — the host/device qualifier.
//
// Under nvcc (__CUDACC__) it is `__host__ __device__` so a function compiles for
// BOTH spaces; under a plain host compiler it expands to nothing so the SAME
// source unit-tests on the CPU. Guarded with #ifndef so co-including a TU that
// already saw it is a harmless no-op rather than a redefinition diagnostic.
// ---------------------------------------------------------------------------
#ifndef STEPPE_HD
#  if defined(__CUDACC__)
#    define STEPPE_HD __host__ __device__
#  else
#    define STEPPE_HD
#  endif
#endif

// ---------------------------------------------------------------------------
// STEPPE_DEBUG_ONLY(...) — expand the argument(s) only in debug builds.
//
// Mirrors the NDEBUG contract of <cassert>: NDEBUG ⇒ release ⇒ the body is
// removed entirely (and, like `assert`, its argument is NOT evaluated, so it
// must be free of needed side effects). The trailing `(void)0` keeps the macro
// usable as a statement (it expects a terminating `;` at the call site).
// ---------------------------------------------------------------------------
#if defined(NDEBUG)
#  define STEPPE_DEBUG_ONLY(...) ((void)0)
#else
#  define STEPPE_DEBUG_ONLY(...) __VA_ARGS__
#endif

// ---------------------------------------------------------------------------
// STEPPE_ASSERT(cond, msg) — debug-only precondition check (§2 fail-fast, §7).
//
// In debug builds it is the standard `assert((cond) && msg)`, so a failure prints
// the message and the call site and aborts under a debugger / compute-sanitizer.
// Under NDEBUG it is removed (and `cond` is NOT evaluated — keep it side-effect
// free). Use for documented-but-unenforced preconditions on the hot paths where a
// throwing check is too costly but a silent wrong answer is unacceptable.
// ---------------------------------------------------------------------------
#if defined(NDEBUG)
#  define STEPPE_ASSERT(cond, msg) ((void)0)
#else
#  define STEPPE_ASSERT(cond, msg) assert((cond) && (msg))
#endif

#endif  // STEPPE_CORE_INTERNAL_HOST_DEVICE_HPP
