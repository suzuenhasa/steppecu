// src/core/internal/nvtx.hpp
//
// The one home of steppe's optional NVTX profiler-timeline annotation macro,
// STEPPE_NVTX_RANGE(name). It has two build states: off by default (expands to
// nothing and includes no header, so it adds zero code) and on only under the
// dedicated -DSTEPPE_NVTX profiling build.
//
// Reference: docs/reference/src_core_internal_nvtx.hpp.md
#ifndef STEPPE_CORE_INTERNAL_NVTX_HPP
#define STEPPE_CORE_INTERNAL_NVTX_HPP

#if defined(STEPPE_NVTX)

#include <nvtx3/nvtx3.hpp>

// Unique per-range variable names (two-level token paste) — reference §5
#  define STEPPE_NVTX_CONCAT_IMPL(a, b) a##b
#  define STEPPE_NVTX_CONCAT(a, b) STEPPE_NVTX_CONCAT_IMPL(a, b)

// The STEPPE_NVTX_RANGE(name) macro — reference §2
#  define STEPPE_NVTX_RANGE(name) \
    ::nvtx3::scoped_range STEPPE_NVTX_CONCAT(steppe_nvtx_range_, __LINE__) { name }

#else

#  define STEPPE_NVTX_RANGE(name)

#endif  // STEPPE_NVTX

#endif  // STEPPE_CORE_INTERNAL_NVTX_HPP
