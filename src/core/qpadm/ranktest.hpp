// src/core/qpadm/ranktest.hpp
//
// Host-side orchestrator for the qpAdm/qpWave rank sweep and the popdrop
// leave-one-source-out table. Host-pure and CUDA-free: every heavy numerical
// step routes through the ComputeBackend seam, so the CPU oracle and the CUDA
// deliverable share one path.
//
// Reference: docs/reference/src_core_qpadm_ranktest.hpp.md
#ifndef STEPPE_CORE_QPADM_RANKTEST_HPP
#define STEPPE_CORE_QPADM_RANKTEST_HPP

#include <vector>

#include "core/qpadm/gls_solve.hpp"
#include "device/backend.hpp"
#include "steppe/config.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::core::qpadm {

// The rank sweep entry point — reference §3
[[nodiscard]] inline RankSweep run_rank_sweep(ComputeBackend& be, const F4Blocks& x,
                                              const JackknifeCov& cov, double alpha,
                                              const QpAdmOptions& opts,
                                              const Precision& precision) {
    return be.rank_sweep(x, cov, alpha, opts, precision);
}

// The popdrop admissibility predicate — reference §6
[[nodiscard]] bool popdrop_feasible(const std::vector<double>& weights);

// Leave-one-source-out feasibility — reference §5
[[nodiscard]] std::vector<PopDropRow> run_popdrop(ComputeBackend& be, const F4Blocks& x,
                                                  const JackknifeCov& cov,
                                                  const QpAdmOptions& opts,
                                                  const Precision& precision);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_RANKTEST_HPP
