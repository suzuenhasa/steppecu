// src/core/internal/log.hpp
//
// THE single logging facade (architecture.md §10 "never printf/cout in library
// code", §8 DRY-internals).
//
// The architecture routes ALL library diagnostics through `STEPPE_LOG_*` so the
// sink/level/async policy is swappable behind the macro (never a bare
// `printf`/`std::cout`). This header is the home of
// that facade. Only the ONE warn level the device-teardown path needs is
// realized — it is the §7 teardown-warning sink that the move-only RAII wrappers
// (`DeviceBuffer`, `Stream`, `Event`, `CublasHandle`) route a nonzero destroy
// status to, so "fail-fast" does not become "fail-silent at teardown" — and it
// replaces the THREE duplicated, already-drifted `fprintf(stderr, ...)` macros
// that were open-coded in device_buffer.cuh / stream.hpp / handles.hpp (the
// `STEPPE_*_WARN_ON_TEARDOWN` placeholders — architecture.md §2 DRY).
// The fuller `STEPPE_LOG_INFO/ERROR/...` levels are reserved for the
// structured-logging backend (§10).
//
// Behavior (UNCHANGED from the placeholders it replaces): in debug builds a warn
// emits one `[steppe][warn] ...` line to stderr; under NDEBUG it is removed
// entirely (a release build is silent at teardown, and — like the macro it
// replaces and like `assert` — does NOT evaluate its arguments, so callers must
// keep the format arguments free of needed side effects). Implemented as a
// printf-style sink (printf-style format, not a `{}` brace style); the format
// string is the single seam a structured-logging backend would swap.
//
// CUDA-FREE-compilable and CUDA-compilable (pure preprocessor + <cstdio> in the
// debug arm), so it lives in `core/internal/` and is consumed via the
// steppe::core_internal INTERFACE target by the device CUDA headers
// (architecture.md §4, §8). NOT in the public ABI (§16).
#ifndef STEPPE_CORE_INTERNAL_LOG_HPP
#define STEPPE_CORE_INTERNAL_LOG_HPP

// ---------------------------------------------------------------------------
// STEPPE_LOG_WARN(fmt, ...) — the one warn sink (printf-style format).
//
// Debug: prefix with `[steppe][warn] ` and emit one line to stderr. `std::fprintf`
// is non-throwing, so this is safe to call from the `noexcept` RAII destructors
// (architecture.md §7 "destructors never throw"). Release: removed entirely,
// matching the NDEBUG-silent teardown contract; args are not evaluated.
//
// Usage: `STEPPE_LOG_WARN("cudaFree at teardown: %s", cudaGetErrorString(e));`
// ---------------------------------------------------------------------------
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
