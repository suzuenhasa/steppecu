// src/core/qpadm/qpadm_fit.hpp
//
// DIRECTORY-SCOPE POINTER (see architecture.md §4 module map — the qpadm/ MISNOMER-DECISION
// NOTE): this directory (src/core/qpadm/) is the FIT / SEARCH / STATS engine, NOT qpAdm-only.
// It ALSO houses the qpGraph family (qpgraph_*), the standalone f-statistics (f3/f4/f4ratio/
// fstat_sweep), and the model-space search (model_search/nested_models). The qpAdm-centric
// name was DEFERRED (not renamed to fit/ nor split into qpgraph/+fstats/) as a deliberate,
// git-blame-cost-weighed choice — kimiactions A6 / cross-cut G7.
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
#include "steppe/config.hpp"    // Precision, kDefaultMantissaBits (the SAME default f2 uses)
#include "steppe/qpadm.hpp"     // public QpAdmModel/QpAdmResult/QpAdmOptions + run_qpadm decls

namespace steppe::core::qpadm {

/// The UNIFIED default fit precision (= the f2 precompute default): emulated FP64 at
/// the SAME kDefaultMantissaBits the f2 GEMMs use (fit-engine.md §1.4). Single-homed
/// HERE (the header) so EVERY fit-chain TU — the orchestrator (qpadm_fit.cpp) AND the
/// S8 rotation (model_search.cpp) — references ONE (kind, mantissa_bits) source and
/// cannot drift the pair apart ([7.2]/[9.1] dedup). The stages that cannot honor it
/// fall back internally (see the PRECISION POLICY note in run_impl). The VALUE is
/// frozen vs the f2 path (§3.2/§5.8) — this only folds the identical ctors, it does
/// NOT change the policy. constexpr ⇒ implicitly inline: the factory is a compile-time
/// value single-defined in the header (ODR-safe across all including TUs), not a
/// runtime call. (Precision is an aggregate with constexpr defaults.)
[[nodiscard]] constexpr Precision default_fit_precision() {
    return Precision{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
}

/// Upper-tail chi-squared probability P(X > x | dof) via the regularized upper
/// incomplete gamma Q(dof/2, x/2) — a deterministic special function (OQ-13: p is
/// the loose tier; an own incomplete-gamma is acceptable). dof <= 0 ⇒ NaN.
[[nodiscard]] double pchisq_upper(double x, int dof);

/// HONEST precision_tag (architecture.md §9, §12): report what ACTUALLY ran on the
/// covariance SYRK — EmulatedFp64 iff the request is emulated AND the backend can honor
/// it, else native Fp64 (never tag a run EmulatedFp64 that silently ran native). Single-
/// homed so run_impl / run_qpwave_impl / run_f4_impl cannot drift their tag derivation
/// ([7.1] dedup). Declared here so the standalone-f4 TU (f4.cpp) reuses the ONE source.
[[nodiscard]] Precision::Kind honored_tag(const Precision& prec, ComputeBackend& be);

/// [target] ++ model.left (the AT2 left = c(target, sources) convention).
[[nodiscard]] std::vector<int> left_with_target(const QpAdmModel& model);

/// The shared S4→S6→S7 body over an already-assembled S3 F4Blocks (assembled by
/// the caller from a device or host f2). block_sizes is the AT2 block_lengths
/// jackknife weight (OQ-3). Domain outcomes are returned as result.status values.
///
/// `opts.jackknife` (JackknifePolicy) gates the S7 jackknife SE (the host-oracle mirror of the GPU two-pass):
/// All ⇒ always compute (the default / today's behavior, the goldens); FeasibleOnly ⇒
/// compute only when the cheap point estimate is a survivor (feasible weights, optionally
/// AND p>=opts.p_se_threshold); None ⇒ never. A non-survivor leaves res.se/z EMPTY
/// (the sentinel). The point estimate (weights/p/f4rank/feasible/rankdrop/popdrop) is
/// identical regardless of policy — only WHICH models get the SE changes.
[[nodiscard]] QpAdmResult run_impl(ComputeBackend& be, F4Blocks&& X,
                                   std::span<const int> block_sizes,
                                   const QpAdmModel& model, const QpAdmOptions& opts);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPADM_FIT_HPP
