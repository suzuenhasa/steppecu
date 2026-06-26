// src/core/qpadm/f4ratio.cpp — the standalone f4-RATIO entry (run_f4ratio).
//
// run_f4ratio is the SIBLING of run_f4 (f4.cpp) / run_f3 (f3.cpp): it REUSES the SAME
// assemble seam — assemble_f4_quartets (the per-quartet four-slab AT2 identity) — with ZERO
// new assemble math, and adds the ONE new math seam f4-ratio needs: the per-block-RATIO
// weighted block-jackknife (the AT2 qpf4ratio / jack_mat_stats `$est` of the ratio statistic
// + its xtau variance). That jackknife is the SHARED on-device RatioBlockJackknife backend
// virtual (device/backend.hpp), reached via f4ratio_blocks_jackknife — NOT a host loop here.
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
//  2. The per-ratio weighted block-jackknife (num row k, den row N+k) is delegated WHOLE to
//     the shared RatioBlockJackknife seam — be.f4ratio_blocks_jackknife(f2, flat, N,
//     kSetmissThresh, prec) — which also performs the AT2 setmiss near-zero-denom per-block
//     skip (|x|<kSetmissThresh=1e-6 treated as MISSING). The detailed jack_mat_stats / xtau
//     derivation lives near that kernel + the long-double oracle (NOT duplicated here).
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

#include "core/qpadm/qpadm_fit.hpp"    // default_fit_precision(), honored_tag()
#include "device/backend.hpp"          // ComputeBackend, RatioBlockJackknife (the shared seam)
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

/// The per-ratio weighted block-jackknife (the AT2 qpf4ratio / jack_mat_stats of the per-block
/// ratio R_b = num_loo_b/den_loo_b) is NO LONGER a host loop here: it is the SHARED on-device
/// ratio_block_jackknife backend virtual (backend.hpp), reached via f4ratio_blocks_jackknife —
/// ONE engine with qpDstat. On the CUDA path the assemble keeps dX/dLoo RESIDENT and feeds the
/// kernel directly (the [m·nb] x_blocks/x_loo D2H is DROPPED); on the CpuBackend the SAME entry
/// assembles then delegates to the long-double oracle (tot_mode=0; the reference math UNCHANGED).
/// Native FP64, AT2 setmiss thresh=1e-6 (kSetmissThresh).

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

    // S3 + S4 FUSED — the device-resident assemble + the SHARED on-device ratio-block-jackknife
    // in ONE backend call (host-compute-audit M1 cure). On the CUDA path the interleaved per-
    // quartet f4 X (num rows [0,N), den rows [N,2N)) stays RESIDENT (dX/dLoo) and feeds the
    // ratio_block_jackknife kernel directly — DROPPING the former [m·nb] x_blocks/x_loo D2H +
    // the host per-tuple ratio_jackknife loop. On the CpuBackend the SAME entry assembles then
    // delegates to the long-double ratio_jackknife oracle (the reference math UNCHANGED). num
    // row = k, den row = N + k (the contiguous halves of the m=2N axis); weight = block_sizes
    // broadcast; tot_mode=0; setmiss_thresh=1e-6. Native FP64 (the §12 cancellation carve-out).
    const RatioBlockJackknife jk =
        be.f4ratio_blocks_jackknife(f2, std::span<const int>(flat), N, kSetmissThresh, prec);
    res.alpha = jk.est;
    res.se = jk.se;
    res.z = jk.z;
    res.status = jk.status;
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
