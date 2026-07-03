// src/core/qpadm/qpadm_fit.hpp
//
// Header for the single-model qpAdm fit orchestrator. Host-pure and CUDA-free: it
// drives the fit through the ComputeBackend seam and single-homes the shared fit
// body (run_impl) that both public run_qpadm entry points funnel into.
//
// Reference: docs/reference/src_core_qpadm_qpadm_fit.hpp.md
#ifndef STEPPE_CORE_QPADM_QPADM_FIT_HPP
#define STEPPE_CORE_QPADM_QPADM_FIT_HPP

#include <span>
#include <vector>

#include "device/backend.hpp"
#include "steppe/config.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::core::qpadm {

// Default fit precision — reference §3
[[nodiscard]] constexpr Precision default_fit_precision() {
    return Precision{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
}

// Chi-squared upper-tail probability — reference §4
[[nodiscard]] double pchisq_upper(double x, int dof);

// Honest precision tag — reference §5
[[nodiscard]] Precision::Kind honored_tag(const Precision& prec, ComputeBackend& be);

// Left-with-target convention — reference §6
[[nodiscard]] std::vector<int> left_with_target(const QpAdmModel& model);

// Shared fit body — reference §7
[[nodiscard]] QpAdmResult run_impl(ComputeBackend& be, F4Blocks&& X,
                                   std::span<const int> block_sizes,
                                   const QpAdmModel& model, const QpAdmOptions& opts);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_QPADM_FIT_HPP
