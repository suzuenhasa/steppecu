// src/core/qpadm/qpadm_fit.cpp — the single-model qpAdm orchestrator + run_qpadm.

#include "core/qpadm/qpadm_fit.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "core/qpadm/f4_matrix.hpp"      // S3 driver
#include "core/qpadm/gls_solve.hpp"      // S6 driver
#include "core/qpadm/jackknife.hpp"      // S4 driver
#include "core/qpadm/nested_models.hpp"  // se_from_loo (S7)
#include "device/backend.hpp"            // ComputeBackend, F4Blocks, JackknifeCov, GlsWeights
#include "device/device_f2_blocks.hpp"   // device::DeviceF2Blocks (S3 device-resident input)
#include "device/resources.hpp"          // device::Resources (the injected backend bundle)
#include "steppe/config.hpp"             // Precision
#include "steppe/error.hpp"              // Status
#include "steppe/fstats.hpp"             // F2BlockTensor

namespace steppe::core::qpadm {

namespace {

/// Regularized lower incomplete gamma P(a, x) by series (good for x < a+1) and the
/// continued-fraction form Q(a, x) (good for x >= a+1). Standard NR recipe; double
/// precision is ample for the loose p tier (OQ-13).
[[nodiscard]] double gammp_series(double a, double x) {
    const int kMaxIter = 1000;
    const double kEps = 1e-15;
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 0; n < kMaxIter; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * kEps) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

[[nodiscard]] double gammq_cf(double a, double x) {
    const int kMaxIter = 1000;
    const double kEps = 1e-15;
    const double kFpMin = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / kFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kMaxIter; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < kFpMin) d = kFpMin;
        c = b + an / c;
        if (std::fabs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < kEps) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

}  // namespace

std::vector<int> left_with_target(const QpAdmModel& model) {
    std::vector<int> left_idx;
    left_idx.reserve(model.left.size() + 1);
    left_idx.push_back(model.target);
    left_idx.insert(left_idx.end(), model.left.begin(), model.left.end());
    return left_idx;
}

QpAdmResult run_impl(ComputeBackend& be, F4Blocks&& X, std::span<const int> block_sizes,
                     const QpAdmModel& model, const QpAdmOptions& opts) {
    QpAdmResult res;
    res.model_index = model.model_index;
    res.precision_tag = Precision::Kind::Fp64;

    const int nl = X.nl;
    const int nr = X.nr;
    const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
    res.est_rank = r;
    res.dof = (nl - r) * (nr - r);

    const Precision prec{Precision::Kind::Fp64};

    // S4 — covariance (OQ-3: weight by block_sizes, NOT vpair).
    JackknifeCov cov = jackknife_cov(be, X, block_sizes, opts.fudge, prec);
    if (cov.status != Status::Ok) {
        res.status = cov.status;  // NonSpdCovariance → value, not throw
        return res;
    }

    // S6 — GLS weights via AT2 ALS.
    GlsWeights gw = gls_weights(be, X, cov, r, opts, prec);
    if (gw.status != Status::Ok) {
        res.status = gw.status;  // RankDeficient → value
        return res;
    }
    res.weight = gw.w;
    res.chisq = gw.chisq;

    // S7 — SE from the n_block LOO weight re-fits (reuse cov.Qinv; AT2 parity pin).
    const SeResult se = se_from_loo(be, X, cov, r, opts, gw.w, prec);
    res.se = se.se;
    res.z = se.z;

    // p of the fitted rank (loose tier, OQ-13).
    res.p = pchisq_upper(res.chisq, res.dof);
    res.rank_p.assign(static_cast<std::size_t>(r) + 1, 0.0);
    if (r >= 0 && static_cast<std::size_t>(r) < res.rank_p.size())
        res.rank_p[static_cast<std::size_t>(r)] = res.p;

    res.status = Status::Ok;
    return res;
}

double pchisq_upper(double x, int dof) {
    if (dof <= 0) return std::nan("");
    if (x <= 0.0) return 1.0;
    const double a = 0.5 * static_cast<double>(dof);
    const double xx = 0.5 * x;
    if (xx < a + 1.0) return 1.0 - gammp_series(a, xx);  // Q = 1 - P
    return gammq_cf(a, xx);
}

}  // namespace steppe::core::qpadm

// ---- Public entry points (include/steppe/qpadm.hpp) -------------------------
namespace steppe {

QpAdmResult run_qpadm(const device::DeviceF2Blocks& f2, const QpAdmModel& model,
                      const QpAdmOptions& opts, device::Resources& resources) {
    ComputeBackend& be = *resources.gpus.at(0).backend;
    const std::vector<int> left_idx = core::qpadm::left_with_target(model);
    const Precision prec{Precision::Kind::Fp64};
    // S3 — device-resident assemble (zero D2H on the CUDA path).
    F4Blocks X = core::qpadm::assemble_f4(be, f2, std::span<const int>(left_idx),
                                          std::span<const int>(model.right), prec);
    return core::qpadm::run_impl(be, std::move(X),
                                 std::span<const int>(f2.block_sizes), model, opts);
}

QpAdmResult run_qpadm(const F2BlockTensor& f2_host, const QpAdmModel& model,
                      const QpAdmOptions& opts, device::Resources& resources) {
    ComputeBackend& be = *resources.gpus.at(0).backend;
    const std::vector<int> left_idx = core::qpadm::left_with_target(model);
    const Precision prec{Precision::Kind::Fp64};
    // S3 — host-oracle assemble (the CpuBackend reads host memory directly).
    F4Blocks X = core::qpadm::assemble_f4(be, f2_host, std::span<const int>(left_idx),
                                          std::span<const int>(model.right), prec);
    return core::qpadm::run_impl(be, std::move(X),
                                 std::span<const int>(f2_host.block_sizes), model, opts);
}

}  // namespace steppe
