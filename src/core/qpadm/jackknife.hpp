// src/core/qpadm/jackknife.hpp
//
// Host-side entry points for the block-jackknife covariance that error-bars the
// f-statistics and the qpAdm fit. Both functions are one-line forwarders into
// the ComputeBackend seam; this header is host-pure and CUDA-free.
//
// Reference: docs/reference/src_core_qpadm_jackknife.hpp.md
#ifndef STEPPE_CORE_QPADM_JACKKNIFE_HPP
#define STEPPE_CORE_QPADM_JACKKNIFE_HPP

#include <span>

#include "device/backend.hpp"
#include "steppe/config.hpp"

namespace steppe::core::qpadm {

// jackknife_cov — full covariance + inverse — reference §3
[[nodiscard]] inline JackknifeCov jackknife_cov(ComputeBackend& be, const F4Blocks& x,
                                                std::span<const int> block_sizes,
                                                double fudge, const Precision& precision) {
    return be.jackknife_cov(x, block_sizes, fudge, precision);
}

// jackknife_diag — diagonal-only variance — reference §4
[[nodiscard]] inline JackknifeDiag jackknife_diag(ComputeBackend& be, const F4Blocks& x,
                                                  std::span<const int> block_sizes,
                                                  const Precision& precision) {
    return be.jackknife_diag(x, block_sizes, precision);
}

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_JACKKNIFE_HPP
