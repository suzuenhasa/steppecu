// src/core/qpadm/gls_solve.hpp
//
// S6 driver: the GLS weight fit (AT2 ALS) dispatched through the
// ComputeBackend seam. Host-pure and CUDA-free — the ALS algebra
// itself lives in the backend.
#ifndef STEPPE_CORE_QPADM_GLS_SOLVE_HPP
#define STEPPE_CORE_QPADM_GLS_SOLVE_HPP

#include "device/backend.hpp"
#include "steppe/config.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::core::qpadm {

[[nodiscard]] inline GlsWeights gls_weights(ComputeBackend& be, const F4Blocks& x,
                                            const JackknifeCov& cov, int r,
                                            const QpAdmOptions& opts,
                                            const Precision& precision) {
    return be.gls_weights(x, cov, r, opts, precision);
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_GLS_SOLVE_HPP
