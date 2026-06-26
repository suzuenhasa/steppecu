// src/core/qpadm/nested_models.hpp
//
// S7 — qpAdm weight standard errors via the leave-one-block-out jackknife
// (design docs/design/fit-engine.md §1.2 S7; AT2 R/qpadm.R get_weights_covariance).
// HOST-PURE, CUDA-FREE orchestration: the WHOLE SE reduction is delegated to the
// single backend seam `se_from_wmat`, which subsumes the per-LOO-block weight re-fits
// (REUSING the full-data Qinv unchanged — the AT2 parity pin, AT2 does NOT re-invert
// per replicate), the AT2 (nb-1)/sqrt(nb) wmat scale, and the delete-1 jackknife
// sample-covariance-diagonal variance reduction into ONE call returning the scaled se.
// This TU forms only z = weight/se from that se (the full-data weight unchanged).
//
// `se_from_wmat` is the backend virtual seam (M7 — the last host-compute move): the
// CUDA backend keeps the resident dWmat and runs the EXISTING on-device SE kernel (no
// dWmat D2H, no host reduction); the CpuBackend overrides it with the long-double
// sample_cov_diag ORACLE. se_from_loo is fully backend-agnostic — it applies neither
// the scale nor the variance reduction on any path.
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
/// get_weights_covariance). Delegates the LOO weight re-fits (reusing `cov.Qinv`) + the
/// AT2 (nb-1)/sqrt(nb) wmat scale + the sample-covariance-diagonal variance reduction to
/// the single backend seam `be.se_from_wmat`, which returns the nl-length scaled se;
/// this function computes only z = weight/se here. `weight` is the full-data weight
/// vector (for z). SE reduction native FP64 (the cancellation carve-out); the underlying
/// re-fits engage `precision` (EmulatedFp64 default, native the carve-out).
[[nodiscard]] SeResult se_from_loo(ComputeBackend& be, const F4Blocks& x,
                                   const JackknifeCov& cov, int r,
                                   const QpAdmOptions& opts,
                                   const std::vector<double>& weight,
                                   const Precision& precision);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_NESTED_MODELS_HPP
