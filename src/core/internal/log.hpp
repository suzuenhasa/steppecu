// src/core/internal/log.hpp
//
// The single logging facade for the steppe library: every diagnostic goes
// through a STEPPE_LOG_* macro instead of a bare printf/cout, so sink, level,
// and async policy stay swappable in one place. Only one level is realized today
// — the STEPPE_LOG_WARN teardown-warning sink — and it compiles away under NDEBUG.
//
// Reference: docs/reference/src_core_internal_log.hpp.md
#ifndef STEPPE_CORE_INTERNAL_LOG_HPP
#define STEPPE_CORE_INTERNAL_LOG_HPP

// STEPPE_LOG_WARN — reference §2
#if defined(NDEBUG)
#  define STEPPE_LOG_WARN(...) ((void)0)
#else
#  include <cstdio>
#  define STEPPE_LOG_WARN(...)                            \
    do {                                                  \
        std::fprintf(stderr, "[steppe][warn] ");          \
        std::fprintf(stderr, __VA_ARGS__);                \
        std::fprintf(stderr, "\n");                       \
    } while (0)
#endif

#endif  // STEPPE_CORE_INTERNAL_LOG_HPP
