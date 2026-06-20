// src/core/qpadm/ranktest.hpp
//
// M(fit-2) rank test / qpWave HOST ORCHESTRATOR (design docs/design/fit-engine.md
// §3 M(fit-2); the FROZEN CONTRACT §2c, §3). HOST-PURE, CUDA-FREE: it drives the
// rank sweep + the AT2 res$rankdrop nested table + the AT2 res$popdrop
// leave-one-LEFT-SOURCE-out table entirely through the CUDA-free ComputeBackend
// seam (no GEMM/SVD/Cholesky issued here). The sweep + rankdrop come straight from
// `ComputeBackend::rank_sweep` (CpuBackend oracle / CudaBackend deliverable).
//
// THE AT2 popdrop CONTRACT (admixtools::drop_pops, reproduced verbatim): popdrop
// does NOT re-gather f4 or re-jackknife. It takes the ALREADY-COMPUTED full f4_est
// (the nl×nr matrix) and the full FUDGED qinv (the m×m), and for each subset of the
// nl left SOURCES it (a) keeps the corresponding ROWS of f4_est, (b) SUBSETS the
// inverse covariance qinv[ind,ind] (the row-major block for the surviving rows; NOT
// re-inverting a sub-Q), and (c) fits at rank = nrow(subset)-1. This orchestrator
// therefore builds a REDUCED F4Blocks (the kept rows) + a REDUCED JackknifeCov (the
// qinv sub-block) and routes them through the SAME rank_sweep + gls_weights backend
// seam — so the CpuBackend oracle and the CudaBackend deliverable share one path.
//
// The f4rank decision rule lives in the backend's rank_sweep; the popdrop
// `feasible` predicate (all surviving weights in [0,1]) lives here (§3).
#ifndef STEPPE_CORE_QPADM_RANKTEST_HPP
#define STEPPE_CORE_QPADM_RANKTEST_HPP

#include <vector>

#include "core/qpadm/gls_solve.hpp"   // gls_weights (popdrop surviving weights)
#include "device/backend.hpp"         // ComputeBackend, F4Blocks, JackknifeCov, RankSweep, PopDropRow
#include "steppe/config.hpp"          // Precision
#include "steppe/qpadm.hpp"           // QpAdmOptions

namespace steppe::core::qpadm {

/// S5 sweep — thin pass-through to the backend rank_sweep virtual (so a fake
/// backend can drive it later; the CpuBackend is the oracle, the CudaBackend the
/// deliverable). alpha is the rank-decision significance (opts.rank_alpha).
[[nodiscard]] inline RankSweep run_rank_sweep(ComputeBackend& be, const F4Blocks& x,
                                              const JackknifeCov& cov, double alpha,
                                              const QpAdmOptions& opts,
                                              const Precision& precision) {
    return be.rank_sweep(x, cov, alpha, opts, precision);
}

/// AT2 res$popdrop `feasible`: all SURVIVING weights in [0,1] (the
/// allow_negative_weights=false feasibility predicate; NaN slots = dropped sources,
/// skipped). A single-source-surviving model (nl→1, weight==1) is trivially TRUE.
[[nodiscard]] bool popdrop_feasible(const std::vector<double>& weights);

/// AT2 res$popdrop — leave-one-LEFT-SOURCE-out feasibility (§2c, §3; the
/// admixtools::drop_pops contract above). Operates on the ALREADY-COMPUTED full X
/// + cov (NOT a re-gather): for the full model "0..0" plus each single-source drop
/// it builds the reduced F4Blocks (kept rows) + reduced JackknifeCov (the qinv
/// sub-block), routes them through rank_sweep + gls_weights, and records the
/// chosen-rank chisq/dof/p/f4rank + the per-source weights (NaN for the dropped
/// slot) + feasible. Native-FP64 precision carve-out per §4. CUDA-FREE host work
/// (the subset is index arithmetic; the fit routes through the backend).
[[nodiscard]] std::vector<PopDropRow> run_popdrop(ComputeBackend& be, const F4Blocks& x,
                                                  const JackknifeCov& cov,
                                                  const QpAdmOptions& opts,
                                                  const Precision& precision);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_RANKTEST_HPP
