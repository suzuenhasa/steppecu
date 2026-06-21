// include/steppe/qpadm.hpp
//
// PUBLIC, CUDA-FREE qpAdm fit-engine value types + entry points (Phase 2, S3–S8;
// design docs/design/fit-engine.md §1.5 + the M(fit-1) FROZEN CONTRACT §1). This
// is the GPU-FIRST seam: a model references populations by INDEX into the
// device-resident f2_blocks P axis (no strings at the compute seam — name→index
// resolution is an app/binding concern), and the primary entry reads a
// device-resident DeviceF2Blocks (zero D2H on the CUDA path). The const
// F2BlockTensor& overload is the M(fit-1) host-oracle/parity door (the parity test
// stages the golden f2 as a host tensor and calls THIS).
//
// It is deliberately CUDA-FREE and standard-C++ only, so it compiles into core,
// the CLI, and the bindings without dragging in the device toolkit (architecture
// .md §4 layering rule). device::DeviceF2Blocks / device::Resources are forward-
// declared CUDA-free below; the .cpp includes their real headers.
//
// SCOPE: the single-model run_qpadm GLS fit (M(fit-1), design §3 first-milestone
// contract) is the foundation — CpuBackend reference, native FP64, one model, full
// rank r = nl-1, no missing blocks — but the header now ALSO freezes the shapes the
// later milestones grew into: the qpWave / rank-sweep forms (run_qpwave +
// QpWaveResult, the M(fit-2) rank-test fields) and the batched/search + S8 forms
// (run_qpadm_search, JackknifePolicy jackknife). QpAdmModel stays a vector-friendly
// value so the span form is a non-breaking add later.
#ifndef STEPPE_QPADM_HPP
#define STEPPE_QPADM_HPP

#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"  // steppe::Precision
#include "steppe/error.hpp"   // steppe::Status
#include "steppe/fstats.hpp"  // steppe::F2BlockTensor (host-oracle overload input)

namespace steppe {

namespace device {
class DeviceF2Blocks;  // CUDA-free fwd-decl (real decl: src/device/device_f2_blocks.hpp)
struct Resources;      // CUDA-free fwd-decl (real decl: src/device/resources.hpp)
}  // namespace device

// ---- AT2 PARITY CONSTANTS (named; OQ-1/OQ-4 — not magic numbers) -------------
/// SE (jackknife standard-error) policy for the S8 rotation. The int values are the
/// user-facing --jackknife=0/1/2 mapping (frozen). The point estimate
/// (weights/chisq/p/f4rank/feasible/popdrop/rankdrop) is IDENTICAL across all three
/// modes; the policy governs ONLY which models pay the expensive LOO jackknife SE.
/// The single-model run_qpadm / run_qpwave ignore this — only run_qpadm_search reads it.
enum class JackknifePolicy : int {
    None         = 0,  ///< no SE for any model (fastest pure screen); se/z empty-marked.
    FeasibleOnly = 1,  ///< SE only for SURVIVORS passing the criterion; the rest empty-marked.
    All          = 2,  ///< SE for every model = the CURRENT behavior. THE DEFAULT.
};

/// Per-call qpAdm config. The AT2 parity constants (fudge, als_iterations) are
/// NAMED here and recorded in the golden metadata, never bare literals (design
/// §3.3, OQ-1/OQ-4).
struct QpAdmOptions {
    /// AT2 ridge constant. Applied in BOTH the opt_A/opt_B ALS systems AND the
    /// final constrained weight normal-equations AND the Q-inversion
    /// regularization (OQ-4). Matches AT2 `qpadm.R` exactly.
    double fudge = 1e-4;

    /// AT2 opt_A/opt_B default iteration count (OQ-1). The weight fit is an
    /// alternating-least-squares loop, NOT a single Cholesky.
    int als_iterations = 20;

    /// f4rank for the fit. -1 ⇒ default nl-1 (best 2-way rank). M(fit-1) fixes the
    /// single best rank; the full qpWave rank sweep is M(fit-2).
    int rank = -1;

    /// AT2 'constrained' (reserved; false in M(fit-1) — the unconstrained
    /// solve(crossprod(x), crossprod(x,y)) path).
    bool constrained = false;

    /// AT2 does NOT clip; weights may exit [0,1] (an infeasible model is a domain
    /// outcome, not an error).
    bool allow_negative_weights = true;

    /// M(fit-2) rank-decision significance (AT2 res$f4rank default 0.05): f4rank is
    /// the smallest candidate rank r whose model is NOT rejected at this alpha
    /// (p(r) > alpha). Named (not a magic literal), recorded in the golden meta.
    double rank_alpha = 0.05;

    /// S8 ROTATION SE policy (default All = the current behavior; the feature is purely
    /// additive/opt-in). IGNORED by the single-model run_qpadm / run_qpwave (they always
    /// compute the SE — only the rotation search consults it). All ⇒ every model gets the
    /// LOO jackknife SE (today's behavior, the goldens); FeasibleOnly ⇒ only survivors of
    /// the feasibility criterion; None ⇒ no SE for any model.
    JackknifePolicy jackknife = JackknifePolicy::All;

    /// FEASIBLE-ONLY survivor p-gate (consulted only when jackknife==FeasibleOnly AND
    /// se_require_p below). pchisq tail-p threshold; a survivor must have p >= this.
    double p_se_threshold = 0.05;

    /// FEASIBLE-ONLY criterion selector: false (default) ⇒ feasibility ALONE is the
    /// survivor test; true ⇒ feasible AND p >= p_se_threshold. Default false: feasibility
    /// is the canonical, hard qpAdm screen; the p-boundary is statistically noisy, so a
    /// default p-gate would drop the SE on exactly the feasible-but-marginal models a
    /// researcher most needs the SE for. Opt-in for an aggressive first-pass survey.
    bool se_require_p = false;
};

// ---- The model + the result (CUDA-free value shapes) ------------------------
/// A model = target + left sources + right outgroups, as INDICES into the
/// f2_blocks P axis. CUDA-free. Batched-capable: a later S8 dispatches a
/// std::span<const QpAdmModel> (design §1.5; M(fit-6)).
struct QpAdmModel {
    /// L_0 ; target pop index (0..P-1). PREPENDED to left to form the AT2
    /// left = c(target, sources) convention (design §1.2).
    int target;

    /// source pops L_1..L_nl (indices). nl == left.size().
    std::vector<int> left;

    /// outgroups; right[0] == R_0 (the fixed right-ref). nr == right.size() - 1.
    std::vector<int> right;

    /// STABLE identity for the deterministic S8 re-sort (echoed on the result).
    int model_index = -1;
};

/// The single-model qpAdm fit result. Domain outcomes (rank-deficient / non-SPD)
/// are PER-MODEL `status` values, NEVER exceptions (architecture.md §10) — a
/// search of thousands of models must record-and-continue.
struct QpAdmResult {
    std::vector<double> weight;  ///< admixture weights w, len == left.size(), Σw = 1.
    /// jackknife SE per weight (len == weight when computed). EMPTY ⇒ SE not computed
    /// (NONE / a non-survivor under FEASIBLE-ONLY, or a domain-failed model). NEVER a
    /// fake 0/NaN — `se.empty()` is the unambiguous "not computed" sentinel.
    std::vector<double> se;
    std::vector<double> z;       ///< weight / se (EMPTY iff se is empty; same sentinel).
    double p = 0.0;              ///< tail p of the fitted rank.
    double chisq = 0.0;          ///< vec(E)'Qinv vec(E) at the fitted rank.
    int dof = 0;                 ///< (nl - r)*(nr - r).

    /// qpWave nested rank-test p-values (rank 0..min-1). M(fit-1) fills only the
    /// fitted-rank entry; the full sweep is M(fit-2).
    std::vector<double> rank_p;

    int est_rank = 0;  ///< the rank used for the reported weights (== r).

    // ---- M(fit-2) rank test / qpWave (appended; non-breaking) ----------------
    std::vector<double> rank_chisq;  ///< chisq(r) per candidate rank r=0..rmax.
    std::vector<int>    rank_dof;    ///< dof(r) per candidate rank.
    int                 f4rank = 0;  ///< the smallest non-rejected rank (AT2 res$f4rank).
    // rankdrop nested table (mirrors RankSweep.rd_*; AT2 res$rankdrop row order):
    std::vector<int>    rankdrop_f4rank, rankdrop_dof, rankdrop_dofdiff;
    std::vector<double> rankdrop_chisq, rankdrop_p, rankdrop_chisqdiff, rankdrop_p_nested;
    // popdrop table (AT2 res$popdrop):
    std::vector<std::string> popdrop_pat;
    std::vector<int>         popdrop_wt, popdrop_dof, popdrop_f4rank;
    std::vector<double>      popdrop_chisq, popdrop_p;
    std::vector<char>        popdrop_feasible;  ///< char (0/1) to keep the value-type CUDA-free + vector<bool>-free.

    /// PER-MODEL outcome (Ok/RankDeficient/NonSpdCovariance); NEVER an exception
    /// for a domain outcome (architecture.md §10).
    Status status = Status::Ok;

    /// Which arithmetic produced this. M(fit-1) is always Fp64.
    Precision::Kind precision_tag = Precision::Kind::Fp64;

    int model_index = -1;  ///< echoes the input for deterministic ordering.
};

// ---- Entry points (design §1.5; the M(fit-1) host-oracle overload is used by the
//      parity test) -------------------------------------------------------------

/// SINGLE model, reading DEVICE-RESIDENT f2_blocks (the GPU-first primary entry).
/// Routes through resources.gpus[0].backend (the injected ComputeBackend).
[[nodiscard]] QpAdmResult run_qpadm(const device::DeviceF2Blocks& f2,
                                    const QpAdmModel& model,
                                    const QpAdmOptions& opts,
                                    device::Resources& resources);

/// HOST-ORACLE / parity overload: takes a host F2BlockTensor directly (the
/// M(fit-1) path — the parity test uploads the golden f2 as a host tensor and
/// calls THIS, with the CpuBackend reading host memory directly).
[[nodiscard]] QpAdmResult run_qpadm(const F2BlockTensor& f2_host,
                                    const QpAdmModel& model,
                                    const QpAdmOptions& opts,
                                    device::Resources& resources);

// ---- M(fit-6) S8 ROTATION / model-space search ------------------------------
/// S8 ROTATION — fit a POOL of candidate models against the SAME device-resident
/// f2_blocks, BATCHED on the GPU and SHARDED across Resources::gpus, returning a
/// per-model result table in INPUT ORDER (deterministic regardless of GPU count).
/// Each model is fit WHOLLY on one device (zero inter-GPU traffic); domain outcomes
/// (RankDeficient/NonSpdCovariance) are per-model `status`, NEVER exceptions — a
/// search of thousands of models must record-and-continue (design §1.5).
/// results[i].model_index == models[i].model_index (the caller's stable identity);
/// the returned vector is ordered so results[k].model_index resolves the k-th input.
[[nodiscard]] std::vector<QpAdmResult> run_qpadm_search(
    const device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models,
    const QpAdmOptions& opts,
    device::Resources& resources);

/// HOST-ORACLE / parity overload (the CpuBackend reads host memory directly): every
/// model is routed through resources.gpus[0].backend's per-model oracle loop. This is
/// the bit-exact reference the device batched path is diffed against.
[[nodiscard]] std::vector<QpAdmResult> run_qpadm_search(
    const F2BlockTensor& f2_host,
    std::span<const QpAdmModel> models,
    const QpAdmOptions& opts,
    device::Resources& resources);

// ---- M(fit-2) qpWave (rank-sufficiency sweep WITHOUT a target) ---------------
/// qpWave-only: the rank-sufficiency sweep for whether the nl left pops are
/// consistent with rank r (the rankdrop machinery WITHOUT a target — left = all
/// the left pops, no target prepend; left[0] is the qpWave reference row).
/// Returns the per-rank table + f4rank. fit-engine.md §3 M(fit-2) item (4).
struct QpWaveResult {
    std::vector<double> rank_chisq;
    std::vector<int>    rank_dof;
    std::vector<double> rank_p;
    std::vector<int>    rankdrop_f4rank, rankdrop_dof, rankdrop_dofdiff;
    std::vector<double> rankdrop_chisq, rankdrop_p, rankdrop_chisqdiff, rankdrop_p_nested;
    int                 f4rank = 0;
    int                 est_rank = 0;
    Status              status = Status::Ok;
    Precision::Kind     precision_tag = Precision::Kind::Fp64;
};

/// qpWave over DEVICE-RESIDENT f2 (GPU-first). `left` is the FULL left set (no
/// target prepend; left[0] is the qpWave reference); `right` is R0..R_nr.
[[nodiscard]] QpWaveResult run_qpwave(const device::DeviceF2Blocks& f2,
                                      std::span<const int> left,
                                      std::span<const int> right,
                                      const QpAdmOptions& opts,
                                      device::Resources& resources);

/// qpWave host-oracle overload (the CpuBackend reads a host F2BlockTensor).
[[nodiscard]] QpWaveResult run_qpwave(const F2BlockTensor& f2_host,
                                      std::span<const int> left,
                                      std::span<const int> right,
                                      const QpAdmOptions& opts,
                                      device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPADM_HPP
