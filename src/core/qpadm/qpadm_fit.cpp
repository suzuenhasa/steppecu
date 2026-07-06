// src/core/qpadm/qpadm_fit.cpp — host-side orchestrator for one qpAdm/qpWave model
// fit: drives the fixed stage sequence in run_impl, with no GPU code of its own.
//
// Reference: docs/reference/src_core_qpadm_qpadm_fit.cpp.md

#include "core/qpadm/qpadm_fit.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "core/internal/pchisq.hpp"
#include "core/qpadm/f4_matrix.hpp"
#include "core/qpadm/gls_solve.hpp"
#include "core/qpadm/jackknife.hpp"
#include "core/qpadm/nested_models.hpp"
#include "core/qpadm/ranktest.hpp"
#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"

namespace steppe::core::qpadm {

// qpAdm left rows: target prepended to sources — reference §9
std::vector<int> left_with_target(const QpAdmModel& model) {
    std::vector<int> left_idx;
    left_idx.reserve(model.left.size() + 1);
    left_idx.push_back(model.target);
    left_idx.insert(left_idx.end(), model.left.begin(), model.left.end());
    return left_idx;
}

// The honest precision tag — reference §4
Precision::Kind honored_tag(const Precision& prec, ComputeBackend& be) {
    return (prec.kind == Precision::Kind::EmulatedFp64 &&
            be.capabilities().emulated_fp64_honorable)
               ? Precision::Kind::EmulatedFp64
               : Precision::Kind::Fp64;
}

// The single-model fit pipeline — reference §2
QpAdmResult run_impl(ComputeBackend& be, F4Blocks&& X, std::span<const int> block_sizes,
                     const QpAdmModel& model, const QpAdmOptions& opts) {
    QpAdmResult res;
    res.model_index = model.model_index;

    const int nl = X.nl;
    const int nr = X.nr;
    const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
    res.est_rank = r;
    res.dof = (nl - r) * (nr - r);

    const Precision prec = default_fit_precision();

    res.precision_tag = honored_tag(prec, be);

    JackknifeCov cov = jackknife_cov(be, X, block_sizes, opts.fudge, prec);
    if (cov.status != Status::Ok) {
        res.status = cov.status;
        return res;
    }

    GlsWeights gw = gls_weights(be, X, cov, r, opts, prec);
    if (gw.status != Status::Ok) {
        res.status = gw.status;
        return res;
    }
    res.weight = gw.w;
    res.chisq = gw.chisq;

    res.p = pchisq_upper(res.chisq, res.dof);

    auto weights_feasible = [](const std::vector<double>& w) {
        bool any = false;
        for (const double v : w) {
            if (std::isnan(v)) continue;
            any = true;
            if (v < 0.0 || v > 1.0) return false;
        }
        return any;
    };
    bool compute_se;
    switch (opts.jackknife) {
        case JackknifePolicy::None:
            compute_se = false;
            break;
        case JackknifePolicy::FeasibleOnly:
            compute_se = weights_feasible(gw.w) &&
                         (!opts.se_require_p || res.p >= opts.p_se_threshold);
            break;
        case JackknifePolicy::All:
        default:
            compute_se = true;
            break;
    }

    if (compute_se) {
        const SeResult se = se_from_loo(be, X, cov, r, opts, gw.w, prec);
        res.se = se.se;
        res.z = se.z;
    }

    res.rank_p.assign(idx(r) + 1, 0.0);
    if (r >= 0 && idx(r) < res.rank_p.size())
        res.rank_p[idx(r)] = res.p;

    if (be.provides_rank_sweep()) {
        const RankSweep rs = run_rank_sweep(be, X, cov, opts.rank_alpha, opts, prec);
        res.rank_chisq = rs.chisq;
        res.rank_dof = rs.dof;
        res.f4rank = rs.f4rank;
        res.rankdrop_f4rank = rs.rd_f4rank;
        res.rankdrop_dof = rs.rd_dof;
        res.rankdrop_dofdiff = rs.rd_dofdiff;
        res.rankdrop_chisq = rs.rd_chisq;
        res.rankdrop_p = rs.rd_p;
        res.rankdrop_chisqdiff = rs.rd_chisqdiff;
        res.rankdrop_p_nested = rs.rd_p_nested;

        const std::vector<PopDropRow> pd = run_popdrop(be, X, cov, opts, prec);
        for (const PopDropRow& row : pd) {
            res.popdrop_pat.push_back(row.pat);
            res.popdrop_wt.push_back(row.wt);
            res.popdrop_dof.push_back(row.dof);
            res.popdrop_f4rank.push_back(row.f4rank);
            res.popdrop_chisq.push_back(row.chisq);
            res.popdrop_p.push_back(row.p);
            res.popdrop_feasible.push_back(row.feasible ? char{1} : char{0});
        }
    }

    res.status = (res.dof <= 0) ? Status::ChisqUndefined : Status::Ok;
    return res;
}

namespace {

// qpWave result from a rank sweep — reference §8
QpWaveResult qpwave_from_sweep(const RankSweep& rs, Precision::Kind tag) {
    QpWaveResult qw;
    qw.rank_chisq = rs.chisq;
    qw.rank_dof = rs.dof;
    qw.rank_p = rs.p;
    qw.rankdrop_f4rank = rs.rd_f4rank;
    qw.rankdrop_dof = rs.rd_dof;
    qw.rankdrop_dofdiff = rs.rd_dofdiff;
    qw.rankdrop_chisq = rs.rd_chisq;
    qw.rankdrop_p = rs.rd_p;
    qw.rankdrop_chisqdiff = rs.rd_chisqdiff;
    qw.rankdrop_p_nested = rs.rd_p_nested;
    qw.f4rank = rs.f4rank;
    qw.est_rank = rs.f4rank;
    qw.status = rs.status;
    qw.precision_tag = tag;
    return qw;
}

}  // namespace

// The one shared chi-squared tail function — reference §9
double pchisq_upper(double x, int dof) {
    return core::internal::pchisq_upper(x, dof);
}

}  // namespace steppe::core::qpadm

namespace steppe {

using core::idx;

namespace {

// Primary-GPU backend selection — reference §9
inline constexpr std::size_t kPrimaryGpu = 0;

[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

// Shared qpAdm body — reference §9
template <class F2Src>
QpAdmResult run_qpadm_impl(ComputeBackend& be, const F2Src& f2, const QpAdmModel& model,
                           const QpAdmOptions& opts) {
    const std::vector<int> left_idx = core::qpadm::left_with_target(model);
    const Precision prec = core::qpadm::default_fit_precision();
    F4Blocks X = core::qpadm::assemble_f4(be, f2, std::span<const int>(left_idx),
                                          std::span<const int>(model.right), prec);
    std::span<const int> bs(X.block_sizes);
    return core::qpadm::run_impl(be, std::move(X), bs, model, opts);
}

// Shared qpWave body — reference §8
template <class F2Src>
QpWaveResult run_qpwave_impl(ComputeBackend& be, const F2Src& f2,
                             std::span<const int> left, std::span<const int> right,
                             const QpAdmOptions& opts) {
    const Precision prec = core::qpadm::default_fit_precision();
    F4Blocks X = core::qpadm::assemble_f4(be, f2, left, right, prec);
    const JackknifeCov cov =
        core::qpadm::jackknife_cov(be, X, std::span<const int>(X.block_sizes), opts.fudge, prec);
    const Precision::Kind tag = core::qpadm::honored_tag(prec, be);
    if (cov.status != Status::Ok) {
        QpWaveResult qw; qw.status = cov.status; qw.precision_tag = tag; return qw;
    }
    const RankSweep rs =
        core::qpadm::run_rank_sweep(be, X, cov, opts.rank_alpha, opts, prec);
    return core::qpadm::qpwave_from_sweep(rs, tag);
}

}  // namespace

// Public qpAdm entry points — reference §9
QpAdmResult run_qpadm(const device::DeviceF2Blocks& f2, const QpAdmModel& model,
                      const QpAdmOptions& opts, device::Resources& resources) {
    return run_qpadm_impl(primary_backend(resources), f2, model, opts);
}

QpAdmResult run_qpadm(const F2BlockTensor& f2_host, const QpAdmModel& model,
                      const QpAdmOptions& opts, device::Resources& resources) {
    return run_qpadm_impl(primary_backend(resources), f2_host, model, opts);
}

// Public qpWave entry points — reference §9
QpWaveResult run_qpwave(const device::DeviceF2Blocks& f2, std::span<const int> left,
                        std::span<const int> right, const QpAdmOptions& opts,
                        device::Resources& resources) {
    return run_qpwave_impl(primary_backend(resources), f2, left, right, opts);
}

QpWaveResult run_qpwave(const F2BlockTensor& f2_host, std::span<const int> left,
                        std::span<const int> right, const QpAdmOptions& opts,
                        device::Resources& resources) {
    return run_qpwave_impl(primary_backend(resources), f2_host, left, right, opts);
}

}  // namespace steppe
