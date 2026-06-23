// src/core/qpadm/f4ratio.cpp — the standalone f4-RATIO entry (run_f4ratio).
//
// run_f4ratio is the SIBLING of run_f4 (f4.cpp) / run_f3 (f3.cpp): it REUSES the SAME
// assemble seam — assemble_f4_quartets (the per-quartet four-slab AT2 identity) — with ZERO
// new assemble math, and adds the ONE new math seam f4-ratio needs: the per-block-RATIO
// weighted block-jackknife (the AT2 qpf4ratio / jack_mat_stats `$est` of the ratio statistic
// + its xtau variance), HOST-PURE here (NOT a backend virtual; backend.hpp is unchanged).
//
// AT2 qpf4ratio (admixtools 4.3.3): pops is c(p1,p2,p3,p4,p5). alpha = f4(p1,p2;p3,p4) /
// f4(p1,p2;p5,p4). The shared pops across num/den are p1,p2,p4; only the 3rd slot swaps (p3
// in num, p5 in den). Per-block num quartet = {p1,p2,p3,p4}, den quartet = {p1,p2,p5,p4}.
//
// PIPELINE (fit-engine §6, mirroring run_f4/run_f3):
//  1. Build TWO flattened quartet arrays for the N tuples (num k={p1,p2,p3,p4}, den
//     k={p1,p2,p5,p4}) and assemble BOTH in ONE assemble_f4_quartets call over the
//     interleaved 2N-quartet flat array (length 8N). This is ESSENTIAL: it gives ONE shared
//     survivor-block set + one block_sizes for num and den (AT2 forces this via setmiss
//     making num/den NA together; F1/OQ-12 compaction, backend.hpp:113-126). The returned
//     F4Blocks has m=2N: num rows k in [0,N), den rows k in [N,2N) (contiguous halves). It
//     carries x_blocks[k+m*b] (per-block f4), x_loo[k+m*b] (AT2 est_to_loo LOO replicate),
//     and block_sizes (survivor weights) — exactly the est_to_loo output AT2 needs.
//  2. The NEW per-ratio jackknife (the ONLY new math, below): for each tuple k, with
//     num_k=k, den_k=N+k, over the survivor blocks: Rb = x_loo[num_k]/x_loo[den_k];
//     totnum=Σ(x_blocks[num_k]*bl)/Σbl, totden likewise, tot=totnum/totden;
//     est = mean(tot - Rb)*nb + weighted_mean(Rb,bl); h_b=n/bl_b;
//     tau_b = h_b*tot - (h_b-1)*Rb; xtau_b = (tau_b - est)/sqrt(h_b - 1);
//     var = Σ(xtau_b^2)/nb; alpha = est; se = sqrt(var); z = alpha/se. This is the AT2
//     jack_mat_stats(tot = the variance-centering term) — DO NOT route through jackknife_cov
//     (it uses a DIFFERENT xtau decomposition: leading x_total*h minus tot_line_, whereas AT2
//     jack_mat_stats uses tot*h leading and subtracts est; they are not the same — reuse the
//     SEAM only for the est_to_loo replicates, write the ratio xtau explicitly).
//  Near-zero denom (AT2 setmiss thresh=1e-6): per block, if abs(x_blocks[den_k+m*b])<1e-6
//  treat that block's contribution as MISSING (skip from the survivor reduction).
//
// assemble_f4_quartets stays native FP64 by the cancellation carve-out (OQ-5), exactly like
// f4/f3. fudge = 0 always (a bare ratio SE; the qpAdm 1e-4 ridge is a Q-INVERT concern only,
// and f4-ratio never inverts a Q). Domain outcomes (empty batch / all blocks missing) are a
// per-call status VALUE / per-row NaN sentinel, never an exception (architecture.md §10).

#include "steppe/f4ratio.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "core/qpadm/f4_quartets.hpp"  // assemble_f4_quartets (REUSED verbatim — no new math)
#include "core/qpadm/qpadm_fit.hpp"    // default_fit_precision(), honored_tag()
#include "device/backend.hpp"          // ComputeBackend, F4Blocks
#include "device/device_f2_blocks.hpp" // device::DeviceF2Blocks (S3 device-resident input)
#include "device/resources.hpp"        // device::Resources (the injected backend bundle)
#include "steppe/config.hpp"           // Precision
#include "steppe/error.hpp"            // Status

namespace steppe {

namespace {

/// The single-entry primary GPU index (the multi-GPU fan-out lives ABOVE this seam; the
/// model-batched rotation drives the others). Mirrors f4.cpp/f3.cpp's kPrimaryGpu — a
/// TU-private convention constant.
inline constexpr std::size_t kPrimaryGpu = 0;

/// AT2 setmiss near-zero-denominator threshold (qpf4ratio setmiss thresh=1e-6): a per-block
/// numerator/denominator with |x| < this is treated as MISSING.
inline constexpr double kSetmissThresh = 1e-6;

[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

/// The per-ratio weighted block-jackknife for ONE tuple (the AT2 qpf4ratio / jack_mat_stats
/// of the per-block ratio statistic R_b = num_loo_b / den_loo_b). Reads x_blocks (per-block
/// f4, for tot = totnum/totden the variance-centering term) and x_loo (the AT2 est_to_loo
/// replicate, for R_b) from the shared F4Blocks. `num_k` / `den_k` are the m-axis rows of
/// the numerator / denominator quartet for this tuple. Writes alpha (= the jackknife $est),
/// se, z. Near-zero-denom blocks (AT2 setmiss) are dropped from the survivor reduction.
/// All accumulation in long double for the small-cancellation ratio (the §12 numerator rule).
void ratio_jackknife(const F4Blocks& x, std::size_t num_k, std::size_t den_k,
                     double& alpha, double& se, double& z) {
    const std::size_t m = static_cast<std::size_t>(x.nl) * static_cast<std::size_t>(x.nr);
    const int nb_all = x.n_block;

    // Survivor pass over blocks: skip a block whose denominator (per-block f4_den) is
    // near-zero (AT2 setmiss thresh=1e-6 makes that block's ratio missing). The survivor
    // block_sizes total n and survivor count nb_surv drive the jackknife.
    long double n_ld = 0.0L;        // Σ survivor block_sizes
    int nb_surv = 0;                // survivor block count
    long double totnum = 0.0L;      // Σ x_blocks[num]*bl  (over survivors)
    long double totden = 0.0L;      // Σ x_blocks[den]*bl  (over survivors)
    for (int b = 0; b < nb_all; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);
        const double den_blk = x.x_blocks[den_k + m * bb];
        if (std::fabs(den_blk) < kSetmissThresh) continue;  // AT2 setmiss: missing block
        const double bl = static_cast<double>(x.block_sizes[bb]);
        n_ld += static_cast<long double>(bl);
        ++nb_surv;
        totnum += static_cast<long double>(x.x_blocks[num_k + m * bb]) * static_cast<long double>(bl);
        totden += static_cast<long double>(den_blk) * static_cast<long double>(bl);
    }

    if (nb_surv <= 1 || n_ld <= 0.0L || totden == 0.0L) {
        // Too few survivor blocks (no jackknife) / a degenerate total denominator: the honest
        // NaN sentinel, never a fabricated value.
        alpha = std::nan("");
        se = std::nan("");
        z = std::nan("");
        return;
    }

    const double n = static_cast<double>(n_ld);
    const double nb = static_cast<double>(nb_surv);
    const double tot = static_cast<double>(totnum / totden);  // the variance-centering term.

    // est = mean(tot - R_b)*nb + weighted_mean(R_b, bl)  (AT2 jack_mat_stats $est of the
    // per-block ratio). The per-block ratio uses the LOO replicates R_b = num_loo / den_loo.
    long double diffsum = 0.0L;       // Σ (tot - R_b)
    long double wmean_num = 0.0L;     // Σ R_b * bl
    for (int b = 0; b < nb_all; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);
        const double den_blk = x.x_blocks[den_k + m * bb];
        if (std::fabs(den_blk) < kSetmissThresh) continue;  // same survivor mask
        const double bl = static_cast<double>(x.block_sizes[bb]);
        const double num_loo = x.x_loo[num_k + m * bb];
        const double den_loo = x.x_loo[den_k + m * bb];
        const double Rb = num_loo / den_loo;
        diffsum += static_cast<long double>(tot) - static_cast<long double>(Rb);
        wmean_num += static_cast<long double>(Rb) * static_cast<long double>(bl);
    }
    const double term1 = static_cast<double>(diffsum / static_cast<long double>(nb_surv)) * nb;
    const double term2 = static_cast<double>(wmean_num / n_ld);
    const double est = term1 + term2;

    // xtau pseudo-values: h_b = n/bl_b; tau_b = h_b*tot - (h_b - 1)*R_b;
    // xtau_b = (tau_b - est)/sqrt(h_b - 1); var = Σ xtau_b^2 / nb.
    long double var_acc = 0.0L;
    for (int b = 0; b < nb_all; ++b) {
        const std::size_t bb = static_cast<std::size_t>(b);
        const double den_blk = x.x_blocks[den_k + m * bb];
        if (std::fabs(den_blk) < kSetmissThresh) continue;  // same survivor mask
        const double bl = static_cast<double>(x.block_sizes[bb]);
        const double num_loo = x.x_loo[num_k + m * bb];
        const double den_loo = x.x_loo[den_k + m * bb];
        const double Rb = num_loo / den_loo;
        const double h = n / bl;
        const double tau = h * tot - (h - 1.0) * Rb;
        const double xtau = (tau - est) / std::sqrt(h - 1.0);
        var_acc += static_cast<long double>(xtau) * static_cast<long double>(xtau);
    }
    const double var = static_cast<double>(var_acc / static_cast<long double>(nb_surv));

    alpha = est;
    se = (var > 0.0) ? std::sqrt(var) : std::nan("");
    z = alpha / se;
}

/// Shared run_f4ratio body: build the interleaved num/den quartet flat array (length 8N),
/// ONE assemble_f4_quartets call (m=2N: num rows [0,N), den rows [N,2N)), then the per-ratio
/// jackknife per tuple. Templated on the f2 SOURCE so the two public overloads (DeviceF2Blocks
/// vs F2BlockTensor) are thin forwarders — mirroring run_f4_impl / run_f3_impl ([7.1] dedup).
template <class F2Src>
F4RatioResult run_f4ratio_impl(ComputeBackend& be, const F2Src& f2,
                               std::span<const std::array<int, 5>> tuples,
                               const QpAdmOptions& opts) {
    // opts is accepted for API symmetry with run_qpadm/run_qpwave/run_f4/run_f3, but a bare
    // f4-ratio SE uses fudge=0 ALWAYS (no Q is inverted), so opts.fudge is deliberately not
    // consulted here. Acknowledge it so -Werror is satisfied.
    (void)opts;
    const Precision prec = core::qpadm::default_fit_precision();

    F4RatioResult res;
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    const int N = static_cast<int>(tuples.size());
    if (N <= 0) {
        // An empty batch is a clean, empty Ok result (no rows) — never a fault.
        res.status = Status::Ok;
        return res;
    }

    // Echo the 5-tuple P-axis indices onto the result (the emitter/binding label rows). Build
    // ONE interleaved 2N-quartet flat array (length 8N): the num quartets occupy quartets
    // [0,N) = {p1,p2,p3,p4}, the den quartets [N,2N) = {p1,p2,p5,p4}. One shared assemble ⇒
    // one shared survivor-block set + one block_sizes for num and den (the AT2 setmiss pin).
    res.p1.reserve(static_cast<std::size_t>(N));
    res.p2.reserve(static_cast<std::size_t>(N));
    res.p3.reserve(static_cast<std::size_t>(N));
    res.p4.reserve(static_cast<std::size_t>(N));
    res.p5.reserve(static_cast<std::size_t>(N));
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(N) * 8);
    // Numerator quartets first (rows [0,N)): {p1,p2,p3,p4}.
    for (const std::array<int, 5>& t : tuples) {
        res.p1.push_back(t[0]);
        res.p2.push_back(t[1]);
        res.p3.push_back(t[2]);
        res.p4.push_back(t[3]);
        res.p5.push_back(t[4]);
        flat.push_back(t[0]);  // p1
        flat.push_back(t[1]);  // p2
        flat.push_back(t[2]);  // p3 (numerator 3rd slot)
        flat.push_back(t[3]);  // p4
    }
    // Denominator quartets second (rows [N,2N)): {p1,p2,p5,p4}.
    for (const std::array<int, 5>& t : tuples) {
        flat.push_back(t[0]);  // p1
        flat.push_back(t[1]);  // p2
        flat.push_back(t[4]);  // p5 (denominator 3rd slot)
        flat.push_back(t[3]);  // p4
    }

    // S3 — assemble the batched per-quartet f4 X for BOTH num + den in ONE call (nl=2N, nr=1
    // ⇒ m=2N). Native FP64 (OQ-5 cancellation carve-out).
    F4Blocks X = core::qpadm::assemble_f4_quartets(be, f2, std::span<const int>(flat), prec);
    const int m = X.nl * X.nr;  // == 2N

    // A degenerate batch (all blocks missing ⇒ m or n_block 0) yields NaN rows but a populated
    // index echo — surface as Ok with the per-row NaN sentinel (never a crash).
    if (m <= 0 || X.n_block <= 0) {
        res.alpha.assign(static_cast<std::size_t>(N), std::nan(""));
        res.se.assign(static_cast<std::size_t>(N), std::nan(""));
        res.z.assign(static_cast<std::size_t>(N), std::nan(""));
        res.status = Status::Ok;
        return res;
    }

    // S4-equivalent — the per-ratio weighted block-jackknife per tuple (the ONLY new math).
    // num row = k, den row = N + k (contiguous halves of the m=2N axis).
    res.alpha.assign(static_cast<std::size_t>(N), 0.0);
    res.se.assign(static_cast<std::size_t>(N), 0.0);
    res.z.assign(static_cast<std::size_t>(N), 0.0);
    for (int k = 0; k < N; ++k) {
        const std::size_t num_k = static_cast<std::size_t>(k);
        const std::size_t den_k = static_cast<std::size_t>(N + k);
        double alpha = 0.0, se = 0.0, z = 0.0;
        ratio_jackknife(X, num_k, den_k, alpha, se, z);
        res.alpha[num_k] = alpha;
        res.se[num_k] = se;
        res.z[num_k] = z;
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace

// ---- Public entry points (include/steppe/f4ratio.hpp) ---------------------------------
F4RatioResult run_f4ratio(const device::DeviceF2Blocks& f2,
                          std::span<const std::array<int, 5>> tuples,
                          const QpAdmOptions& opts, device::Resources& resources) {
    // S3 — device-resident assemble (zero D2H on the CUDA path).
    return run_f4ratio_impl(primary_backend(resources), f2, tuples, opts);
}

F4RatioResult run_f4ratio(const F2BlockTensor& f2_host,
                          std::span<const std::array<int, 5>> tuples,
                          const QpAdmOptions& opts, device::Resources& resources) {
    // S3 — host-oracle assemble (the CpuBackend reads host memory directly).
    return run_f4ratio_impl(primary_backend(resources), f2_host, tuples, opts);
}

}  // namespace steppe
