// src/core/qpadm/gls_solve.hpp
//
// S6 driver (design docs/design/fit-engine.md §1.3): dispatch the GLS weight fit
// (the AT2 svd-seed + opt_A/opt_B ALS + constrained weight solve) through the
// ComputeBackend seam. HOST-PURE, CUDA-FREE.
// The ALS algebra lives in the backend (CpuBackend reference; CudaBackend M(fit-4)).
#ifndef STEPPE_CORE_QPADM_GLS_SOLVE_HPP
#define STEPPE_CORE_QPADM_GLS_SOLVE_HPP

#include "device/backend.hpp"  // ComputeBackend, F4Blocks, JackknifeCov, GlsWeights
#include "steppe/config.hpp"   // Precision
#include "steppe/qpadm.hpp"    // QpAdmOptions

namespace steppe::core::qpadm {

/// S6 — GLS weights via the AT2 ALS (OQ-1).
[[nodiscard]] inline GlsWeights gls_weights(ComputeBackend& be, const F4Blocks& x,
                                            const JackknifeCov& cov, int r,
                                            const QpAdmOptions& opts,
                                            const Precision& precision) {
    return be.gls_weights(x, cov, r, opts, precision);
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_GLS_SOLVE_HPP
