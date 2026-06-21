// src/core/qpadm/qpadm_fit.cpp — the single-model qpAdm orchestrator + run_qpadm.

#include "core/qpadm/qpadm_fit.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/internal/pchisq.hpp"     // core::internal::pchisq_upper (the ONE special function)
#include "core/qpadm/f4_matrix.hpp"      // S3 driver
#include "core/qpadm/gls_solve.hpp"      // S6 driver
#include "core/qpadm/jackknife.hpp"      // S4 driver
#include "core/qpadm/nested_models.hpp"  // se_from_loo (S7)
#include "core/qpadm/ranktest.hpp"       // M(fit-2) run_rank_sweep / run_popdrop / run_qpwave_impl
#include "device/backend.hpp"            // ComputeBackend, F4Blocks, JackknifeCov, GlsWeights
#include "device/device_f2_blocks.hpp"   // device::DeviceF2Blocks (S3 device-resident input)
#include "device/resources.hpp"          // device::Resources (the injected backend bundle)
#include "steppe/config.hpp"             // Precision, kDefaultMantissaBits (the SAME default f2 uses)
#include "steppe/error.hpp"              // Status
#include "steppe/fstats.hpp"             // F2BlockTensor

namespace steppe::core::qpadm {

std::vector<int> left_with_target(const QpAdmModel& model) {
    std::vector<int> left_idx;
    left_idx.reserve(model.left.size() + 1);
    left_idx.push_back(model.target);
    left_idx.insert(left_idx.end(), model.left.begin(), model.left.end());
    return left_idx;
}

// default_fit_precision() — the UNIFIED default fit precision — is single-homed in
// qpadm_fit.hpp ([7.2]/[9.1] dedup) so the orchestrator, both run_qpadm/run_qpwave
// forwarders, AND the S8 rotation (model_search.cpp) reference ONE source.

// HONEST precision_tag (architecture.md §9, §12; fit-engine.md §1.4). Report what
// ACTUALLY ran on the covariance SYRK, not what was requested: EmulatedFp64 iff the
// request is emulated AND the backend can honor it (the SAME `emulated_fp64_honorable`
// capability the f2 path consults via `emulation_honorable`), else native Fp64. The
// CpuBackend (the native oracle) reports the all-false default ⇒ Fp64; a CUDA build
// without the fixed-slice tuning degrades to native ⇒ Fp64 — never tag a run
// EmulatedFp64 that silently ran native. Single-homed here so run_impl and
// run_qpwave_impl cannot drift their tag derivation apart ([7.1] dedup, §8).
Precision::Kind honored_tag(const Precision& prec, ComputeBackend& be) {
    return (prec.kind == Precision::Kind::EmulatedFp64 &&
            be.capabilities().emulated_fp64_honorable)
               ? Precision::Kind::EmulatedFp64
               : Precision::Kind::Fp64;
}

QpAdmResult run_impl(ComputeBackend& be, F4Blocks&& X, std::span<const int> block_sizes,
                     const QpAdmModel& model, const QpAdmOptions& opts) {
    QpAdmResult res;
    res.model_index = model.model_index;

    const int nl = X.nl;
    const int nr = X.nr;
    const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
    res.est_rank = r;
    res.dof = (nl - r) * (nr - r);

    // PRECISION POLICY (unified with the f2 precompute; fit-engine.md §1.4). The
    // DEFAULT for all fit stages is now EmulatedFp64{kDefaultMantissaBits} — the
    // SAME default the f2 GEMMs use (Ozaki fixed-slice, ~2.2e-11) — and the stages
    // that cannot honor it fall back internally: the matmul-heavy covariance SYRK
    // ENGAGES this via `emulation_honorable` (auto-native if the build/device cannot
    // honor emulation); the catastrophic-cancellation ops (the f4 4-slab combine,
    // the xtau centering) stay native ALWAYS (emulation faithfully forms a product
    // but cannot recover bits a prior subtraction annihilated); and the ill-
    // conditioned cuSOLVER SPD inverse stays native (the d6d3cbb promotion seam,
    // gated on a future FP64-emulated cuSOLVER mode the toolkit does not yet expose).
    const Precision prec = default_fit_precision();

    // HONEST precision_tag — what ACTUALLY ran on the covariance SYRK (single-homed in
    // honored_tag so run_impl + run_qpwave_impl cannot drift; [7.1] dedup, §9/§12).
    res.precision_tag = honored_tag(prec, be);

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

    // p of the fitted rank (loose tier, OQ-13). Computed BEFORE the SE so the
    // FeasibleOnly p-gate can consult it (a cheap-pass output). For dof<=0 the
    // chi-squared tail-p is undefined and pchisq_upper returns NaN (pchisq.hpp);
    // an over-parameterized model is a DOMAIN OUTCOME, not a fault — flagged
    // below as Status::ChisqUndefined so a consumer filtering on status==Ok does
    // not silently accept a NaN p (architecture.md §10 STEPPE_ERR_CHISQ_UNDEFINED).
    res.p = pchisq_upper(res.chisq, res.dof);

    // ---- S7 SE policy (the host-oracle mirror of the GPU two-pass; fit-engine.md §M(fit-3))
    // The cheap point estimate (weights/chisq/p) is now complete; decide whether THIS
    // model is a survivor that pays the expensive LOO jackknife SE. The criterion is
    // computable from the cheap pass ALONE: feasibility = all fitted weights in [0,1]
    // (at least one) — the same all-in-[0,1] test assemble_result uses on pop_wfull
    // (which == gw.w at the full rank nl-1, the default) — optionally AND p>=threshold.
    // A non-survivor leaves res.se/res.z EMPTY (the sentinel), never a fake 0.
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

    // S7 — SE from the n_block LOO weight re-fits (reuse cov.Qinv; AT2 parity pin).
    // Survivors get the IDENTICAL full FP64 SE; non-survivors leave se/z empty.
    if (compute_se) {
        const SeResult se = se_from_loo(be, X, cov, r, opts, gw.w, prec);
        res.se = se.se;
        res.z = se.z;
    }

    res.rank_p.assign(static_cast<std::size_t>(r) + 1, 0.0);
    if (r >= 0 && static_cast<std::size_t>(r) < res.rank_p.size())
        res.rank_p[static_cast<std::size_t>(r)] = res.p;

    // M(fit-2) — the RANK SWEEP + the AT2 res$rankdrop nested table (the sweep over
    // r=0..rmax + f4rank). Reuses the same X + cov; the precision carve-out (§4)
    // holds the SVD/chisq native. The popdrop is filled by the run_qpadm overloads
    // (it re-gathers a reduced X from the f2 SOURCE, which run_impl does not hold).
    // NOTE (M(fit-2) build order): the CpuBackend (the ORACLE) implements rank_sweep
    // now; the CudaBackend (THE deliverable) implements it in the NEXT phase. Until
    // then a backend without the override throws "not implemented" — caught here so
    // the existing green GPU fit path is not broken (the rankdrop/popdrop fields stay
    // empty on such a backend, a non-breaking absence). The GPU-deliverable phase
    // replaces this with the CudaBackend override and the test asserts the GPU path.
    // NOTE: the popdrop (AT2 res$popdrop) operates on the SAME already-computed X +
    // cov (admixtools::drop_pops subsets rows of f4_est + the qinv block; NO
    // re-gather, NO re-jackknife), so it is filled HERE inside run_impl (it does not
    // need the f2 source). Both rankdrop + popdrop route through rank_sweep, guarded
    // together: a backend without the override (the GPU deliverable phase) leaves
    // these fields empty (non-breaking) rather than throwing out of the fit.
    try {
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

        // res$popdrop (leave-one-LEFT-SOURCE-out), on the same X + cov.
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
    } catch (const std::runtime_error&) {
        // backend has no rank_sweep override yet (the GPU deliverable phase) — the
        // rankdrop/popdrop fields remain empty; the single-rank fit is unaffected.
    }

    // dof<=0 ⇒ ChisqUndefined (architecture.md §10). The fit itself succeeded
    // (weights/chisq/rank-sweep are populated, identical to before), but the
    // tail-p is undefined — surface that as the per-model status VALUE so the
    // rotation/CLI records-and-continues rather than treating a NaN-p model as Ok.
    // Behavior-neutral for normal models (dof>0 ⇒ Ok, goldens unchanged).
    res.status = (res.dof <= 0) ? Status::ChisqUndefined : Status::Ok;
    return res;
}

namespace {

/// Build a QpWaveResult from a RankSweep (qpWave = the sweep WITHOUT a target; the
/// SAME rank_sweep machinery, the orchestrator just gathers X with no target).
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

double pchisq_upper(double x, int dof) {
    return core::internal::pchisq_upper(x, dof);  // the ONE shared special function
}

}  // namespace steppe::core::qpadm

// ---- Public entry points (include/steppe/qpadm.hpp) -------------------------
namespace steppe {

namespace {

/// The single-model entry-point GPU index: select the FIRST backend (the multi-GPU
/// fan-out lives ABOVE this seam; the model-batched rotation drives the others). Names
/// the device-index `0` repeated at the four public wrappers ([7.2] dedup; mirrors
/// gpus[0] = the combine root, resources.hpp). TU-private convention constant, so it
/// is homed here rather than in config.hpp (it is not a cross-TU tunable).
inline constexpr std::size_t kPrimaryGpu = 0;

/// The backend bound to kPrimaryGpu — folds the identical `*resources.gpus.at(0).backend`
/// at every public fit/wave entry into one accessor + names the magic index ([7.2] dedup).
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

/// Shared qpAdm body: prepend the target to `left` (left = c(target, sources)), run the
/// S3 assemble + the S4→S6→S7 chain via run_impl. Templated on the f2 SOURCE so the two
/// public run_qpadm overloads (DeviceF2Blocks vs F2BlockTensor) are thin forwarders —
/// mirroring run_qpwave_impl ([7.1] dedup). assemble_f4 is the cancellation-sensitive
/// 4-slab combine and stays native ALWAYS by carve-out (cuda_backend.cu OQ-5
/// `(void)precision`), exactly like the f2 numerator — passing the emulated default here
/// is the one-policy consistency, not a behavior change for S3. The CpuBackend host-
/// oracle path ignores precision ⇒ always native (the native oracle the GPU path is
/// diffed against). run_impl fills the M(fit-2) rankdrop + popdrop (both on the same
/// X + cov).
template <class F2Src>
QpAdmResult run_qpadm_impl(ComputeBackend& be, const F2Src& f2, const QpAdmModel& model,
                           const QpAdmOptions& opts) {
    const std::vector<int> left_idx = core::qpadm::left_with_target(model);
    const Precision prec = core::qpadm::default_fit_precision();
    F4Blocks X = core::qpadm::assemble_f4(be, f2, std::span<const int>(left_idx),
                                          std::span<const int>(model.right), prec);
    return core::qpadm::run_impl(be, std::move(X),
                                 std::span<const int>(f2.block_sizes), model, opts);
}

/// Shared qpWave body: gather X with `left` as the rows DIRECTLY (NO target
/// prepend; left[0] is the qpWave reference, nl = left.size()-1), run S4 + the
/// rank_sweep, and map to a QpWaveResult. Templated on the f2 SOURCE.
template <class F2Src>
QpWaveResult run_qpwave_impl(ComputeBackend& be, const F2Src& f2,
                             std::span<const int> left, std::span<const int> right,
                             const QpAdmOptions& opts) {
    const Precision prec = core::qpadm::default_fit_precision();
    // qpWave: NO target prepend — `left` IS the rows (left[0] the reference).
    F4Blocks X = core::qpadm::assemble_f4(be, f2, left, right, prec);
    const JackknifeCov cov =
        core::qpadm::jackknife_cov(be, X, std::span<const int>(f2.block_sizes), opts.fudge, prec);
    // HONEST precision_tag — single-homed in honored_tag ([7.1] dedup, §9/§12).
    const Precision::Kind tag = core::qpadm::honored_tag(prec, be);
    if (cov.status != Status::Ok) {
        QpWaveResult qw; qw.status = cov.status; qw.precision_tag = tag; return qw;
    }
    const RankSweep rs =
        core::qpadm::run_rank_sweep(be, X, cov, opts.rank_alpha, opts, prec);
    return core::qpadm::qpwave_from_sweep(rs, tag);
}

}  // namespace

QpAdmResult run_qpadm(const device::DeviceF2Blocks& f2, const QpAdmModel& model,
                      const QpAdmOptions& opts, device::Resources& resources) {
    // S3 — device-resident assemble (zero D2H on the CUDA path).
    return run_qpadm_impl(primary_backend(resources), f2, model, opts);
}

QpAdmResult run_qpadm(const F2BlockTensor& f2_host, const QpAdmModel& model,
                      const QpAdmOptions& opts, device::Resources& resources) {
    // S3 — host-oracle assemble (the CpuBackend reads host memory directly).
    return run_qpadm_impl(primary_backend(resources), f2_host, model, opts);
}

// ---- qpWave (M(fit-2) item (4): the rank sweep WITHOUT a target) -------------

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
