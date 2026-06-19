// src/core/qpadm/nested_models.hpp
//
// S7 — qpAdm weight standard errors via the leave-one-block-out jackknife
// (design docs/design/fit-engine.md §1.2 S7; AT2 R/qpadm.R get_weights_covariance).
// HOST-PURE, CUDA-FREE orchestration: it re-runs the S6 weight solve (the backend
// gls_weights virtual) once per LOO block, REUSING the full-data Qinv unchanged
// (the AT2 parity pin — AT2 does NOT re-invert per replicate), then forms the
// delete-1 jackknife covariance of the weight replicates → se, z.
//
// In M(fit-1) this is an n_block host loop over gls_weights (correctness first,
// design §2); the batched device S7 is M(fit-3). No new backend virtual is added —
// the batched-capable seam stays clean.
#ifndef STEPPE_CORE_QPADM_NESTED_MODELS_HPP
#define STEPPE_CORE_QPADM_NESTED_MODELS_HPP

#include <vector>

#include "device/backend.hpp"  // steppe::ComputeBackend, F4Blocks, JackknifeCov
#include "steppe/config.hpp"   // steppe::Precision
#include "steppe/qpadm.hpp"    // steppe::QpAdmOptions

namespace steppe::core::qpadm {

/// The S7 result: per-weight jackknife SE and z = weight/se.
struct SeResult {
    std::vector<double> se;
    std::vector<double> z;
};

/// Compute weight SEs over the n_block leave-one-out replicates (AT2
/// get_weights_covariance). For each block b: build the nl×nr xmat from
/// x.x_loo[:,:,b] and re-solve the weights via `be.gls_weights`, reusing
/// `cov.Qinv`. Then scale wmat by (numreps-1)/sqrt(numreps) (the !boot branch)
/// and take se = sqrt(diag(cov(wmat))), z = weight/se. `weight` is the full-data
/// weight vector (for z). Native FP64.
[[nodiscard]] SeResult se_from_loo(ComputeBackend& be, const F4Blocks& x,
                                   const JackknifeCov& cov, int r,
                                   const QpAdmOptions& opts,
                                   const std::vector<double>& weight,
                                   const Precision& precision);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_NESTED_MODELS_HPP
