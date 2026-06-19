// src/core/qpadm/jackknife.hpp
//
// S4 driver (design docs/design/fit-engine.md §1.3): dispatch the weighted
// block-jackknife covariance (Q + Qinv) through the ComputeBackend seam. HOST-
// PURE, CUDA-FREE. OQ-3: the per-block jackknife weight is block_sizes (AT2
// block_lengths), NOT Vpair — the caller passes block_sizes here. The est_to_loo
// → xtau → Q → fudge → invert pipeline lives in the backend (CpuBackend reference).
#ifndef STEPPE_CORE_QPADM_JACKKNIFE_HPP
#define STEPPE_CORE_QPADM_JACKKNIFE_HPP

#include <span>

#include "device/backend.hpp"  // ComputeBackend, F4Blocks, JackknifeCov
#include "steppe/config.hpp"   // Precision

namespace steppe::core::qpadm {

/// S4 — weighted block-jackknife covariance. fudge is the AT2 ridge (OQ-4).
[[nodiscard]] inline JackknifeCov jackknife_cov(ComputeBackend& be, const F4Blocks& x,
                                                std::span<const int> block_sizes,
                                                double fudge, const Precision& precision) {
    return be.jackknife_cov(x, block_sizes, fudge, precision);
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_JACKKNIFE_HPP
