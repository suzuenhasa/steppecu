// src/core/internal/nvtx.hpp
//
// THE single home of the optional NVTX phase-boundary annotation macro
// (architecture.md §10 observability; cmake/SteppeOptions.cmake STEPPE_NVTX).
//
// STEPPE_NVTX_RANGE(name) — open a named, RAII-scoped NVTX range that closes at
// the end of the enclosing block. It is a passive HOST-side timeline annotation
// only: no device work, no stream, no sync, no kernel/math — so it never touches
// the single statistic stream (architecture.md §12 parity law is untouched).
//
// TWO-STATE, structurally zero-overhead OFF (the shipping default):
//
//   * STEPPE_NVTX defined (only the `-DSTEPPE_NVTX=ON` profiling build sets it,
//     PRIVATE on steppe_device; see src/device/CMakeLists.txt) ⇒ pull in the
//     header-only NVTX v3 C++ API and expand to an `nvtx3::scoped_range`. NVTX v3
//     is header-only and dependency-free (ships in the CUDA toolkit `nvtx3/`
//     include subdir), so NO `find_package`/link is needed for the header.
//   * STEPPE_NVTX NOT defined (default) ⇒ the macro expands to nothing and no
//     <nvtx3/...> header is included, so the object code is BYTE-IDENTICAL to a
//     build without this header and `nm` shows no nvtx symbols. The option
//     string's "zero-overhead off" promise is structural, not aspirational.
//
// This header is CUDA-FREE-compilable (the OFF path includes no CUDA/NVTX header;
// the ON path includes only the host-side C++ NVTX wrapper, which needs no CUDA
// runtime) and CUDA-compilable, so it lives in `core/internal/` and is consumed
// via the steppe::core_internal INTERFACE target. It is NOT in the public ABI
// (§16): markers live only inside steppe_device TUs.
//
// USAGE: place markers at COARSE phase boundaries only (decode, f2 GEMM batch,
// jackknife, qpAdm fit). Keep them OUT of per-block / per-quartet inner loops so
// that even an ON profiling build does not perturb the relative kernel timeline.
#ifndef STEPPE_CORE_INTERNAL_NVTX_HPP
#define STEPPE_CORE_INTERNAL_NVTX_HPP

#if defined(STEPPE_NVTX)

#include <nvtx3/nvtx3.hpp>

// Two-level concat so the __LINE__ operand is expanded to its value BEFORE the
// `##` paste (a single-level paste would suppress macro expansion and yield the
// literal token `__LINE__`). Gives each range object a unique, line-derived name.
#  define STEPPE_NVTX_CONCAT_IMPL(a, b) a##b
#  define STEPPE_NVTX_CONCAT(a, b) STEPPE_NVTX_CONCAT_IMPL(a, b)

// `name` is a string literal (e.g. "decode"); nvtx3::scoped_range opens on
// construction and closes (Pop) at end of scope — std::lock_guard-style RAII.
#  define STEPPE_NVTX_RANGE(name) \
    ::nvtx3::scoped_range STEPPE_NVTX_CONCAT(steppe_nvtx_range_, __LINE__) { name }

#else  // !STEPPE_NVTX — shipping default: expand to nothing, include no header.

#  define STEPPE_NVTX_RANGE(name)

#endif  // STEPPE_NVTX

#endif  // STEPPE_CORE_INTERNAL_NVTX_HPP
