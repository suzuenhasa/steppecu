// src/core/qpadm/qpadm_fit.hpp
//
// Single-model qpAdm orchestrator (design docs/design/fit-engine.md §1.3, §3.1;
// the M(fit-1) FROZEN CONTRACT §3). HOST-PURE, CUDA-FREE: it drives the S3→S4→
// S6→S7 chain entirely through the CUDA-free ComputeBackend seam (no GEMM/SVD/
// Cholesky issued here). The two public run_qpadm overloads (DeviceF2Blocks +
// F2BlockTensor) are thin wrappers that select resources.gpus[0].backend and call
// the shared impl below.
#ifndef STEPPE_CORE_QPADM_QPADM_FIT_HPP
#define STEPPE_CORE_QPADM_QPADM_FIT_HPP

#include <span>
#include <vector>

#include "device/backend.hpp"  // steppe::ComputeBackend, F4Blocks
#include "steppe/qpadm.hpp"     // public QpAdmModel/QpAdmResult/QpAdmOptions + run_qpadm decls

namespace steppe::core::qpadm {

/// Upper-tail chi-squared probability P(X > x | dof) via the regularized upper
/// incomplete gamma Q(dof/2, x/2) — a deterministic special function (OQ-13: p is
/// the loose tier; an own incomplete-gamma is acceptable). dof <= 0 ⇒ NaN.
[[nodiscard]] double pchisq_upper(double x, int dof);

/// [target] ++ model.left (the AT2 left = c(target, sources) convention).
[[nodiscard]] std::vector<int> left_with_target(const QpAdmModel& model);

/// The shared S4→S6→S7 body over an already-assembled S3 F4Blocks (assembled by
/// the caller from a device or host f2). block_sizes is the AT2 block_lengths
/// jackknife weight (OQ-3). Domain outcomes are returned as result.status values.
[[nodiscard]] QpAdmResult run_impl(ComputeBackend& be, F4Blocks&& X,
                                   std::span<const int> block_sizes,
                                   const QpAdmModel& model, const QpAdmOptions& opts);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPADM_FIT_HPP
