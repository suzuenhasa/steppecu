// src/core/qpadm/qpgraph_fit.cpp — the qpGraph single-graph fit entry (run_qpgraph).
//
// Wires the model -> the resident basis -> the fleet -> the result, mirroring f3.cpp /
// qpadm_fit.cpp (HOST-PURE, CUDA-FREE; the GPU is reached ONLY through the ComputeBackend
// seam). The pipeline (qpgraph-gpu-design.md §design):
//   1. MODEL : parse the edge list -> the path-table topology model (qpgraph_model.cpp;
//              the ONE new piece, generalizing the spike's fixed fill_pwts).
//   2. BASIS : assemble the f3 basis over the choose(npop,2) (base; a,b) pairs via the
//              REUSED assemble_f3_triples seam (the three-slab AT2 identity), then the
//              REUSED jackknife_cov seam (fudge=diag_f3 -> the AT2 ppinv). f_obs = X.x_total
//              (the AT2 weighted-jackknife f3 est), Qinv = the inverse covariance. Device-
//              resident on the CUDA path (zero D2H), reused across all restarts.
//   3. FLEET : the productized IDEA-1 fleet (ComputeBackend::qpgraph_fit_fleet) — each
//              restart is one GPU thread running the whole multistart x maxit projected-
//              Newton loop on-device; NO host objective per iteration.
//   4. RESULT: fitted weights (+lo/hi from the restart spread) + edge lengths + score +
//              worst f3-residual z. Domain outcomes are a status VALUE, never an exception.
#include "steppe/qpgraph.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "core/qpadm/f3_triples.hpp"   // assemble_f3_triples (the S3 basis seam, REUSED)
#include "core/qpadm/jackknife.hpp"    // jackknife_cov (the S4 covariance seam, REUSED)
#include "core/qpadm/qpadm_fit.hpp"    // default_fit_precision(), honored_tag()
#include "core/qpadm/qpgraph_model.hpp"  // parse_qpgraph (the NEW topology data model)
#include "device/backend.hpp"          // ComputeBackend, F4Blocks, JackknifeCov, QpGraphTopoArena/Fleet
#include "device/device_f2_blocks.hpp" // device::DeviceF2Blocks
#include "device/resources.hpp"        // device::Resources
#include "steppe/config.hpp"           // Precision
#include "steppe/error.hpp"            // Status

namespace steppe {

namespace {

inline constexpr std::size_t kPrimaryGpu = 0;
[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

/// Fill the CUDA-free arena the fleet virtual consumes from the parsed model + opts.
QpGraphTopoArena make_arena(const core::qpadm::QpGraphModel& m, const QpGraphOptions& opts) {
    QpGraphTopoArena a;
    a.npop = m.npop; a.nedge_norm = m.nedge_norm; a.nadmix = m.nadmix;
    a.npair = m.npair; a.npath = m.npath; a.base_leaf = m.base_leaf;
    a.pwts0 = m.pwts0;
    a.pe_edge = m.pe_edge; a.pe_leaf = m.pe_leaf; a.pe_path = m.pe_path;
    a.pae_path = m.pae_path; a.pae_admixedge = m.pae_admixedge;
    a.cmb1 = m.cmb1; a.cmb2 = m.cmb2;
    a.constrained = opts.constrained;
    a.fudge = opts.fudge;
    return a;
}

/// Shared run_qpgraph body over an f2 SOURCE (DeviceF2Blocks or F2BlockTensor). Templated
/// so the two public overloads are thin forwarders (the f3.cpp/run_f3_impl pattern).
template <class F2Src>
QpGraphResult run_qpgraph_impl(ComputeBackend& be, const F2Src& f2,
                               const std::vector<QpGraphEdge>& edges,
                               const std::vector<std::string>& leaf_names,
                               const QpGraphOptions& opts) {
    QpGraphResult res;

    // ---- 1. MODEL ---------------------------------------------------------------
    const core::qpadm::QpGraphModel m = core::qpadm::parse_qpgraph(edges, leaf_names);
    if (!m.ok()) {
        // A structural / parse problem is a per-call domain outcome (InvalidConfig), a
        // value not an exception (architecture.md §10). The message is surfaced by the
        // CLI/binding; the result carries the leaf echo it managed to build.
        res.status = Status::InvalidConfig;
        return res;
    }
    res.leaves = m.leaves;

    // The f3 basis precision: the matmul-heavy ppwts/cc assembly defaults emulated; the
    // f3 three-slab combine + the SPD solve are native by carve-out (handled inside the
    // seams). One-policy consistency (default_fit_precision()).
    const Precision prec = core::qpadm::default_fit_precision();
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    // ---- 2. BASIS (resident on the CUDA path) -----------------------------------
    // The f3 basis is over the choose(npop,2) pairs (base; leaf_a, leaf_b) with leaf_a <=
    // leaf_b over the NON-base leaves (incl the diagonal a==b => f3(base;i,i)=f2(base,i)).
    // Build the flattened 3*npair triple array {C=base_f2, A=leaf_a_f2, B=leaf_b_f2}.
    const int base_f2 = m.leaf_to_f2[static_cast<std::size_t>(m.base_leaf)];
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(m.npair) * 3);
    for (int k = 0; k < m.npair; ++k) {
        const int la = m.centered_col_to_leaf(m.cmb1[static_cast<std::size_t>(k)]);
        const int lb = m.centered_col_to_leaf(m.cmb2[static_cast<std::size_t>(k)]);
        flat.push_back(base_f2);
        flat.push_back(m.leaf_to_f2[static_cast<std::size_t>(la)]);
        flat.push_back(m.leaf_to_f2[static_cast<std::size_t>(lb)]);
    }
    // S3 — the per-triple f3 X (one assemble; nl=npair, nr=1 => m=npair). REUSED seam.
    F4Blocks X = core::qpadm::assemble_f3_triples(be, f2, std::span<const int>(flat), prec);
    // INVARIANT: X.nl*X.nr == m.npair by the assemble_f3_triples contract. The seam was fed
    // `flat`, which holds exactly m.npair triples (built above), so it returns nl=npair, nr=1
    // => m=npair (== m.npop*(m.npop-1)/2). This local re-derives that count from the returned
    // F4Blocks rather than reusing m.npair so the basis dimension tracks the seam's OWN output
    // (it is the entry count Qinv/f_obs are sized against below); the two must not drift.
    const int npair = X.nl * X.nr;
    if (npair <= 0 || X.n_block <= 0) { res.status = Status::NonSpdCovariance; return res; }

    // S4 — the f3 block-jackknife covariance + inverse (ppinv). AT2's diag_f3 is the
    // jackknife_cov fudge here (ppinv = inverse(Q + diag_f3*tr(Q)*I) == AT2 solve(f3_var
    // with diag += diag_f3*sum(diag))). REUSED seam.
    const JackknifeCov cov =
        core::qpadm::jackknife_cov(be, X, std::span<const int>(X.block_sizes), opts.diag_f3, prec);
    if (cov.status != Status::Ok) { res.status = cov.status; return res; }

    // f_obs (the AT2 f3_est) + Qinv (the AT2 ppinv), resident on the CUDA path.
    std::span<const double> f_obs(X.x_total.data(), static_cast<std::size_t>(npair));
    std::span<const double> qinv(cov.Qinv.data(), static_cast<std::size_t>(npair) * npair);

    // ---- 3. FLEET (GPU-bound; NO host objective per iteration) -------------------
    const QpGraphTopoArena arena = make_arena(m, opts);
    const QpGraphFleet fl =
        be.qpgraph_fit_fleet(arena, f_obs, qinv, opts.numstart, opts.maxit, opts.tol, prec);
    if (fl.status != Status::Ok) { res.status = fl.status; return res; }

    // ---- 4. RESULT --------------------------------------------------------------
    res.score = fl.score;
    res.restart_spread = fl.restart_spread;
    res.weight = fl.theta;
    res.weight_lo = fl.theta_lo;
    res.weight_hi = fl.theta_hi;
    res.admix_from = m.admix_from;
    res.admix_to = m.admix_to;
    res.edge_length = fl.edge_length;
    res.edge_from = m.edge_from;
    res.edge_to = m.edge_to;

    // Worst f3-residual z over the basis pairs: z[k] = (f_obs[k]-f3_fit[k]) / se[k],
    // se[k] = sqrt(Q[k,k]) (the UNFUDGED diagonal). Report the max |z| with labels.
    //
    // L4 — DELIBERATELY HOST-SIDE (ACCEPT + DOCUMENT). This is a single-run, O(npair)
    // bounded diagnostic in the CUDA-FREE driver, run ONCE per qpGraph fit (NOT in any
    // batched/swept/per-restart inner loop — the throughput envelope the host-compute
    // audit targets). Folding the numeric argmax on-device is a NET LOSS, not a win:
    //   (1) its inputs are already host-resident — cov.Q's diagonal (the unfudged
    //       covariance, a CUDA-FREE driver vector) and fl.f3_fit (the L3 device output,
    //       already brought down) — so a device fold would re-ship Q to VRAM and bring
    //       the argmax index back, ADDING round-trips for an O(npair) scan;
    //   (2) the RESULT (worst_pop2/worst_pop3) is a pair of LABEL STRINGS that MUST be
    //       resolved host-side from the model maps (m.leaves / centered_col_to_leaf)
    //       regardless — the device cannot produce them.
    // Unlike the per-block jackknife / decode / reduce / SE / ploidy items the campaign
    // moved on-device (each in a hot per-block or per-model loop), this scan carries no
    // throughput exposure, so it stays on the host as a conscious bounded-single-run
    // diagnostic decision (the CpuBackend oracle path is the SAME host code). L1 (rank_Q)
    // and L3 (the qpGraph edge/f3_fit re-eval) DID move on-device; L4 is accepted here.
    double worst = 0.0;
    int worst_k = -1;
    for (int k = 0; k < npair; ++k) {
        const double qkk = cov.Q[static_cast<std::size_t>(k) + static_cast<std::size_t>(npair) * static_cast<std::size_t>(k)];
        const double se = (qkk > 0.0) ? std::sqrt(qkk) : std::nan("");
        const double z = (X.x_total[static_cast<std::size_t>(k)] - fl.f3_fit[static_cast<std::size_t>(k)]) / se;
        if (std::isfinite(z) && std::fabs(z) > std::fabs(worst)) { worst = z; worst_k = k; }
    }
    res.worst_residual_z = worst;
    if (worst_k >= 0) {
        const int la = m.centered_col_to_leaf(m.cmb1[static_cast<std::size_t>(worst_k)]);
        const int lb = m.centered_col_to_leaf(m.cmb2[static_cast<std::size_t>(worst_k)]);
        res.worst_pop2 = m.leaves[static_cast<std::size_t>(la)];
        res.worst_pop3 = m.leaves[static_cast<std::size_t>(lb)];
    }
    res.status = Status::Ok;
    return res;
}

}  // namespace

// ---- Public entry points (include/steppe/qpgraph.hpp) ---------------------------
QpGraphResult run_qpgraph(const device::DeviceF2Blocks& f2,
                          const std::vector<QpGraphEdge>& edges,
                          const std::vector<std::string>& leaf_names,
                          const QpGraphOptions& opts, device::Resources& resources) {
    return run_qpgraph_impl(primary_backend(resources), f2, edges, leaf_names, opts);
}

QpGraphResult run_qpgraph(const F2BlockTensor& f2_host,
                          const std::vector<QpGraphEdge>& edges,
                          const std::vector<std::string>& leaf_names,
                          const QpGraphOptions& opts, device::Resources& resources) {
    return run_qpgraph_impl(primary_backend(resources), f2_host, edges, leaf_names, opts);
}

}  // namespace steppe
