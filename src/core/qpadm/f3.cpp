// src/core/qpadm/f3.cpp — the standalone f3-statistic entry (run_f3).
//
// run_f3 is the SIBLING of run_f4 (f4.cpp), line-for-line: it REUSES the same two seams —
// assemble_f3_triples (the per-triple THREE-slab AT2 identity, the ONE new math seam vs f4)
// and jackknife_cov (the block-jackknife covariance, REUSED verbatim) — with NO new
// infrastructure. NO ALS / NO rank test: f3 is just the AT2 weighted-jackknife POINT
// ESTIMATE per triple + the jackknife-DIAGONAL SE.
//
// PIPELINE (fit-engine §6, mirroring run_f4): assemble ONE batched X (an F4Blocks reused as
// the generic per-block carrier) whose m axis is the N triples (nl=N, nr=1 ⇒ m=N), run ONE
// jackknife_cov over the whole m-batch with fudge=0 (a bare f3 SE has no GLS matrix to
// ridge-regularize — the qpAdm 1e-4 fudge is a Q-INVERT concern only, and f3 never inverts
// Q), then read est[k] = X.x_total[k] and se[k] = sqrt(Q[k+m*k]) (the UNFUDGED diagonal).
// z = est/se; p = f4_two_sided_p(z) (REUSED — AT2 ztop == erfc(|z|/sqrt2) == f4_two_sided_p,
// the SAME two-sided normal). assemble_f3_triples stays native FP64 by the cancellation
// carve-out (OQ-5), exactly like assemble_f4_quartets. Domain outcomes (non-SPD covariance /
// empty batch) are a per-call status VALUE, never an exception (architecture.md §10).

#include "steppe/f3.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "core/qpadm/f3_triples.hpp"   // run_f3 S3 driver (assemble_f3_triples)
#include "core/qpadm/jackknife.hpp"    // S4 driver (jackknife_cov) — REUSED verbatim
#include "core/qpadm/qpadm_fit.hpp"    // default_fit_precision(), honored_tag()
#include "device/backend.hpp"          // ComputeBackend, F4Blocks, JackknifeCov
#include "device/device_f2_blocks.hpp" // device::DeviceF2Blocks (S3 device-resident input)
#include "device/resources.hpp"        // device::Resources (the injected backend bundle)
#include "steppe/config.hpp"           // Precision
#include "steppe/error.hpp"            // Status
#include "steppe/f4.hpp"               // f4_two_sided_p (REUSED — the SAME z->p convention)

namespace steppe {

namespace {

/// The single-entry primary GPU index (the multi-GPU fan-out lives ABOVE this seam; the
/// model-batched rotation drives the others). Mirrors f4.cpp's kPrimaryGpu — a TU-private
/// convention constant, homed here rather than in config.hpp.
inline constexpr std::size_t kPrimaryGpu = 0;

[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

/// Shared run_f3 body: assemble the N-triple batched X (one assemble), run ONE
/// jackknife_cov over the whole m-batch (fudge=0), read est/se/z/p per triple. Templated
/// on the f2 SOURCE so the two public overloads (DeviceF2Blocks vs F2BlockTensor) are thin
/// forwarders — mirroring run_f4_impl ([7.1] dedup). assemble_f3_triples is the
/// cancellation-sensitive three-slab combine and stays native ALWAYS by carve-out (OQ-5);
/// passing the emulated default here is the one-policy consistency, not a behavior change.
template <class F2Src>
F3Result run_f3_impl(ComputeBackend& be, const F2Src& f2,
                     std::span<const std::array<int, 3>> triples,
                     const QpAdmOptions& opts) {
    // opts is accepted for API symmetry with run_qpadm/run_qpwave/run_f4, but a STANDALONE
    // f3 SE uses fudge=0 ALWAYS (the qpAdm 1e-4 ridge is a GLS Q-invert concern only; an f3
    // SE is the bare UNFUDGED jackknife-diagonal variance), so opts.fudge is deliberately
    // not consulted here. Acknowledge it so -Werror is satisfied.
    (void)opts;
    const Precision prec = core::qpadm::default_fit_precision();

    F3Result res;
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    const int N = static_cast<int>(triples.size());
    if (N <= 0) {
        // An empty batch is a clean, empty Ok result (no rows) — never a fault.
        res.status = Status::Ok;
        return res;
    }

    // Echo the triple P-axis indices onto the result (the emitter/binding label rows).
    res.p1.reserve(static_cast<std::size_t>(N));
    res.p2.reserve(static_cast<std::size_t>(N));
    res.p3.reserve(static_cast<std::size_t>(N));
    // Flatten the (p1,p2,p3) triples into the 3*N index array the seam consumes.
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(N) * 3);
    for (const std::array<int, 3>& t : triples) {
        res.p1.push_back(t[0]);
        res.p2.push_back(t[1]);
        res.p3.push_back(t[2]);
        flat.push_back(t[0]);
        flat.push_back(t[1]);
        flat.push_back(t[2]);
    }

    // S3 — assemble the batched per-triple f3 X (one call; nl=N, nr=1 ⇒ m=N).
    F4Blocks X = core::qpadm::assemble_f3_triples(be, f2, std::span<const int>(flat), prec);
    const int m = X.nl * X.nr;  // == N (the triple m-axis)

    // A degenerate batch (all blocks missing ⇒ m or n_block 0) yields empty est/se but a
    // populated index echo — surface as Ok with NaN rows so a caller filtering on status
    // sees the per-row NaN sentinel (never a crash).
    if (m <= 0 || X.n_block <= 0) {
        res.est.assign(static_cast<std::size_t>(N), std::nan(""));
        res.se.assign(static_cast<std::size_t>(N), std::nan(""));
        res.z.assign(static_cast<std::size_t>(N), std::nan(""));
        res.p.assign(static_cast<std::size_t>(N), std::nan(""));
        res.status = Status::Ok;
        return res;
    }

    // S4 — block-jackknife covariance over the whole m-batch. fudge = 0 (a bare f3 SE; the
    // qpAdm 1e-4 ridge is a GLS Q-invert concern only). The UNFUDGED Q is JackknifeCov.Q;
    // the per-triple variance is its DIAGONAL Q[k + m*k] (Q is column-major m×m, so the
    // diagonal index is layout-agnostic). jackknife_cov fills out.Q UNCONDITIONALLY (before
    // any inversion), so the diagonal is ALWAYS valid — even when the full m×m Q is singular
    // (which it is for a large correlated triple batch: N triples sharing pops produce a
    // rank-deficient Q). A bare f3 SE NEVER inverts Q (only qpAdm's GLS does), so that full-
    // matrix non-invertibility is IRRELEVANT to f3 and is NOT a domain failure — we read the
    // valid diagonal and report Ok. (Identical rationale to run_f4; both have no Qinv consumer.)
    const JackknifeCov cov =
        core::qpadm::jackknife_cov(be, X, std::span<const int>(X.block_sizes), 0.0, prec);

    res.est.assign(static_cast<std::size_t>(N), 0.0);
    res.se.assign(static_cast<std::size_t>(N), 0.0);
    res.z.assign(static_cast<std::size_t>(N), 0.0);
    res.p.assign(static_cast<std::size_t>(N), 0.0);
    const std::size_t m_sz = static_cast<std::size_t>(m);
    for (int k = 0; k < N; ++k) {
        const std::size_t ks = static_cast<std::size_t>(k);
        const double est = X.x_total[ks];
        const double var = cov.Q[ks + m_sz * ks];  // the UNFUDGED diagonal variance.
        const double se = (var > 0.0) ? std::sqrt(var) : std::nan("");
        const double z = est / se;
        res.est[ks] = est;
        res.se[ks] = se;
        res.z[ks] = z;
        res.p[ks] = f4_two_sided_p(z);  // REUSED — the SAME two-sided normal z->p.
    }

    // Ok: the per-triple diagonal SE is valid regardless of the full Q's SPD-ness (f3 has
    // no Qinv consumer). The only true f3 domain failure is a degenerate ASSEMBLE (handled
    // above: m<=0 / n_block<=0 ⇒ NaN rows), not a non-invertible full covariance.
    res.status = Status::Ok;
    return res;
}

}  // namespace

// ---- Public entry points (include/steppe/f3.hpp) -------------------------------------
F3Result run_f3(const device::DeviceF2Blocks& f2,
                std::span<const std::array<int, 3>> triples,
                const QpAdmOptions& opts, device::Resources& resources) {
    // S3 — device-resident assemble (zero D2H on the CUDA path).
    return run_f3_impl(primary_backend(resources), f2, triples, opts);
}

F3Result run_f3(const F2BlockTensor& f2_host,
                std::span<const std::array<int, 3>> triples,
                const QpAdmOptions& opts, device::Resources& resources) {
    // S3 — host-oracle assemble (the CpuBackend reads host memory directly).
    return run_f3_impl(primary_backend(resources), f2_host, triples, opts);
}

}  // namespace steppe
