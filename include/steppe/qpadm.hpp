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
// SCOPE (M(fit-1), design §3 first-milestone contract): single-model qpAdm GLS fit
// on the CpuBackend reference, native FP64, one model, full rank r = nl-1, no
// missing blocks. The batched/search forms (run_qpadm_search, the rank sweep) are
// later milestones; this header freezes only the single-model shapes M(fit-1)
// needs, plus QpAdmModel as a vector-friendly value so the span form is a
// non-breaking add later.
#ifndef STEPPE_QPADM_HPP
#define STEPPE_QPADM_HPP

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
    std::vector<double> se;      ///< jackknife SE per weight (len == weight).
    std::vector<double> z;       ///< weight / se.
    double p = 0.0;              ///< tail p of the fitted rank.
    double chisq = 0.0;          ///< vec(E)'Qinv vec(E) at the fitted rank.
    int dof = 0;                 ///< (nl - r)*(nr - r).

    /// qpWave nested rank-test p-values (rank 0..min-1). M(fit-1) fills only the
    /// fitted-rank entry; the full sweep is M(fit-2).
    std::vector<double> rank_p;

    int est_rank = 0;  ///< the rank used for the reported weights (== r).

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

}  // namespace steppe

#endif  // STEPPE_QPADM_HPP
