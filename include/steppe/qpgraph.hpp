// include/steppe/qpgraph.hpp
//
// PUBLIC, CUDA-FREE qpGraph single-graph-fit value types + entry points (Phase 2,
// the qpGraph milestone; productizes the IDEA-1 optimizer spike + the path-algebra
// prototype, docs/research/qpgraph-gpu-design.md). qpGraph fits a FIXED admixture
// graph to the observed f-statistics: minimize the GLS residual
//   score(theta) = (f3_est - ppwts_2d(theta)*bl)' * ppinv * (f3_est - ppwts_2d(theta)*bl)
// over theta (the admixture mixture weights; the inner drift edge lengths bl are the
// constrained GLS-optimal solve), ppinv = the f3 block-jackknife inverse covariance.
//
// SCOPE: SINGLE-GRAPH FIT of a fixed topology (topology SEARCH is DEFERRED). The
// graph is an admixtools-format edge list (parent->child rows; a node with 2 parents
// is an admixture node). The leaves are the f2 populations; the fit reads an f2 object
// (read_f2 / a DeviceF2Blocks — qpGraph is f2-path, NOT genotype).
//
// GPU-FIRST + GPU-BOUND: the f3 basis (f3_est) + ppinv are assembled ONCE and stay
// device-resident; the IDEA-1 fleet (one thread per restart) runs the WHOLE multistart
// x maxit projected-Newton loop on-device, each objective eval (path-table fill_pwts ->
// the cc design assembly -> the native SPD edge solve -> the GLS quadratic form) entirely
// on the GPU. The host NEVER calls a host objective per optimizer iteration (the AT2
// optim() host-loop trap, designed out). The host launches ONCE per topology and gets
// back only the final {score, theta, edge_lengths}.
//
// Deliberately CUDA-FREE and standard-C++ only (architecture.md §4 layering): it
// compiles into core, the CLI and the bindings without dragging in the device toolkit.
// device::DeviceF2Blocks / device::Resources are forward-declared CUDA-free below.
#ifndef STEPPE_QPGRAPH_HPP
#define STEPPE_QPGRAPH_HPP

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

/// One directed edge of the admixture graph (parent -> child). The leaves (childless
/// nodes) must be f2 population names; internal nodes are arbitrary labels. A node that
/// is the `to` of TWO edges is an admixture node (its two parents are the mixture
/// sources; the single mixture weight theta_j splits mass between them).
struct QpGraphEdge {
    std::string from;
    std::string to;
};

/// Per-call qpGraph config. The AT2 parity constants are NAMED here (never bare
/// literals), matching admixtools::qpgraph() defaults exactly (the golden was generated
/// with these — see goldens/at2/scripts/golden_qpgraph_generate.R).
struct QpGraphOptions {
    /// AT2 `diag` — the trace-scaled ridge fudge on the inner edge-length normal
    /// equations cc (opt_edge_lengths: diag(cc) += fudge*mean(diag(cc))). DEFAULT 1e-4.
    double fudge = 1e-4;

    /// AT2 `diag_f3` — the f3 covariance regularization (diag(f3_var) +=
    /// diag_f3*sum(diag(f3_var)) before inversion to ppinv). DEFAULT 1e-5.
    double diag_f3 = 1e-5;

    /// AT2 `numstart` — the multistart restart count (the fleet's parallel axis; each
    /// restart = one GPU thread). DEFAULT 10 (the golden's value).
    int numstart = 10;

    /// AT2 `constrained` — drift edges >= 0 (the box-constrained active-set edge solve).
    /// DEFAULT TRUE (the golden's mode; AT2 default). false => the unconstrained
    /// normal-equation solve.
    bool constrained = true;

    /// The fleet projected-Newton iteration cap per restart. The objective is cheap and
    /// well-conditioned (the spike converges in << this); a generous cap.
    int maxit = 200;

    /// Fleet convergence tolerance on |dtheta| / |dscore| (the spike's xtol/stol).
    double tol = 1e-9;
};

/// The single-graph qpGraph fit result. Domain outcomes (non-SPD covariance, a
/// degenerate / unparseable graph) are a `status` VALUE, never an exception
/// (architecture.md §10).
struct QpGraphResult {
    /// The fitted admixture mixture weights, one per admixture node (theta_j). Same
    /// order as `admix_from`/`admix_to` below (the discovered admixture nodes, in graph
    /// edge order). EMPTY for a graph with no admixture node (a pure tree).
    std::vector<double> weight;
    /// Restart spread bracket on each weight (min/max across the numstart restarts) —
    /// the identifiability witness (a tight bracket => a unique optimum).
    std::vector<double> weight_lo;
    std::vector<double> weight_hi;
    /// The admixture-node parent labels for each weight: weight[j] is the mass on the
    /// edge admix_from[j] -> admix_to[j] (the FIRST incident parent; the second parent
    /// carries 1 - weight[j]).
    std::vector<std::string> admix_from;
    std::vector<std::string> admix_to;

    /// The fitted drift edge lengths, one per NON-admixture edge, in `edge_from`/
    /// `edge_to` order (the input edge order, admixture edges excluded).
    std::vector<double> edge_length;
    std::vector<std::string> edge_from;
    std::vector<std::string> edge_to;

    /// The GLS fit score (= AT2 fit$score): res' ppinv res at the optimum.
    double score = 0.0;
    /// The restart value spread (max - min across restarts) — the convergence witness.
    double restart_spread = 0.0;

    /// The worst f3-residual z over the basis pairs (|z| max), with its pair labels.
    /// worst_residual_z is signed; the magnitude is the diagnostic.
    double worst_residual_z = 0.0;
    std::string worst_pop2;  ///< the worst-residual pair's pop2 (the base is pop1).
    std::string worst_pop3;  ///< the worst-residual pair's pop3.

    /// The leaf order the fit used (graph leaves; leaf[0] is the f3 base population).
    std::vector<std::string> leaves;

    /// PER-CALL outcome (Ok / NonSpdCovariance / a parse/degeneracy outcome). NEVER an
    /// exception for a domain outcome (architecture.md §10).
    Status status = Status::Ok;

    /// Which arithmetic produced this (the honored precision tag).
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// ---- Entry points -----------------------------------------------------------------
/// SINGLE graph, reading DEVICE-RESIDENT f2_blocks (the GPU-first primary entry). The
/// f3 basis is assembled device-resident, the fleet runs on the GPU. `edges` is the
/// admixture-graph edge list; `leaf_names` maps the graph's leaf labels to the f2
/// P-axis (leaf_names[i] is the name of f2 population index i — the resolver's pops.txt
/// order). Routes through resources.gpus[0].backend.
[[nodiscard]] QpGraphResult run_qpgraph(const device::DeviceF2Blocks& f2,
                                        const std::vector<QpGraphEdge>& edges,
                                        const std::vector<std::string>& leaf_names,
                                        const QpGraphOptions& opts,
                                        device::Resources& resources);

/// HOST-ORACLE / parity overload: takes a host F2BlockTensor directly (the CpuBackend
/// reads host memory). `leaf_names` is the f2 tensor's P-axis pop order.
[[nodiscard]] QpGraphResult run_qpgraph(const F2BlockTensor& f2_host,
                                        const std::vector<QpGraphEdge>& edges,
                                        const std::vector<std::string>& leaf_names,
                                        const QpGraphOptions& opts,
                                        device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPGRAPH_HPP
