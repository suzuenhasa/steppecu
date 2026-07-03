// src/core/qpadm/nested_models.hpp
//
// qpAdm weight standard errors: turns mixture weights into per-weight jackknife
// SEs and z-scores. Host-only and CUDA-free — pure orchestration that delegates
// the whole SE reduction to the compute backend's se_from_wmat seam.
//
// Reference: docs/reference/src_core_qpadm_nested_models.hpp.md
#ifndef STEPPE_CORE_QPADM_NESTED_MODELS_HPP
#define STEPPE_CORE_QPADM_NESTED_MODELS_HPP

#include <vector>

#include "device/backend.hpp"
#include "steppe/config.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::core::qpadm {

// SeResult — reference §5
struct SeResult {
    std::vector<double> se;
    std::vector<double> z;
};

// se_from_loo — reference §6
[[nodiscard]] SeResult se_from_loo(ComputeBackend& be, const F4Blocks& x,
                                   const JackknifeCov& cov, int r,
                                   const QpAdmOptions& opts,
                                   const std::vector<double>& weight,
                                   const Precision& precision);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_NESTED_MODELS_HPP
