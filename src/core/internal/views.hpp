// src/core/internal/views.hpp
//
// The Q/V/N contract as a non-owning, column-major [P × M] view (MatView) — the
// shared indexing seam between the CPU reference and the GPU f2 feeder. Host-only
// and CUDA-free: a borrowed double* plus row/column counts, no ownership.
//
// Reference: docs/reference/src_core_internal_views.hpp.md
#ifndef STEPPE_CORE_INTERNAL_VIEWS_HPP
#define STEPPE_CORE_INTERNAL_VIEWS_HPP

namespace steppe::core {

// MatView: non-owning column-major [P × M] double view — reference §4
struct MatView {
    const double* data = nullptr;

    int P = 0;

    long M = 0;

    [[nodiscard]] double element(int i, long s) const noexcept {
        return data[static_cast<long>(i) + P * s];
    }
};

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_VIEWS_HPP
