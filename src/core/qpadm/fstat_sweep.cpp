// src/core/qpadm/fstat_sweep.cpp — the GPU-ONLY f-stat sweep driver (run_f4_sweep / run_f3_sweep).
//
// HOST-PURE, CUDA-FREE: it computes the maxcomb cap (a pure choose() compare, no allocation),
// maps the public SweepRequest -> the backend SweepConfig, dispatches the WHOLE on-device sweep
// to the CUDA backend's f4_sweep / f3_sweep virtual (which enumerates+computes+filters+compacts
// on the device and returns ONLY the survivors), then finishes the small survivor set on the
// host: p = f4_two_sided_p(z), and the optional TopK ranking (host-side, over the already-
// compacted small survivor set — NOT a per-item host loop over the full N).
//
// THE HOST DOES NOT: enumerate (the device unrank kernel does), filter per item (the device
// |z| flag + CUB compaction do), or hold the full N-row table (only survivors leave the device).
// The maxcomb cap guards COMPUTE TIME (every item is computed to test the filter), fired BEFORE
// any device work. MULTI-GPU PARKED: single-GPU (resources.gpus[0]); sharding the index range
// across GPUs is a parked follow-on, not this driver.
#include "steppe/fstat_sweep.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "device/backend.hpp"           // ComputeBackend, SweepConfig, SweepSurvivors
#include "device/device_f2_blocks.hpp"  // device::DeviceF2Blocks
#include "device/resources.hpp"         // device::Resources
#include "core/qpadm/qpadm_fit.hpp"     // default_fit_precision(), honored_tag()
#include "steppe/config.hpp"            // Precision, kFstatMaxComb
#include "steppe/f4.hpp"                // f4_two_sided_p (the AT2 z->p convention)

namespace steppe {

namespace {

inline constexpr std::size_t kPrimaryGpu = 0;

[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

/// C(P, k) as an unsigned-long-long, SATURATING at ULLONG_MAX on overflow so the cap compare
/// is still correct (an overflowing count is certainly > kFstatMaxComb). Pure host arithmetic;
/// no allocation. k is tiny (3/4).
[[nodiscard]] unsigned long long choose_saturating(int P, int k) {
    if (k < 0 || P < k) return 0ULL;
    unsigned long long result = 1ULL;
    // C(P,k) = Π_{i=1..k} (P-k+i)/i, computed so each partial stays integral.
    for (int i = 1; i <= k; ++i) {
        const unsigned long long num = static_cast<unsigned long long>(P - k + i);
        // result = result * num / i, overflow-guarded (saturate).
        if (result > (~0ULL) / (num == 0 ? 1 : num)) return ~0ULL;  // would overflow
        result = result * num / static_cast<unsigned long long>(i);
    }
    return result;
}

/// Shared driver: dispatch the on-device sweep, finish p + the optional TopK ranking on the
/// small survivor set. `k` selects the arity (4 ⇒ f4_sweep, 3 ⇒ f3_sweep).
[[nodiscard]] SweepResult run_sweep_impl(const device::DeviceF2Blocks& f2,
                                         const SweepRequest& req, device::Resources& resources,
                                         int k) {
    SweepResult res;
    const Precision prec = core::qpadm::default_fit_precision();
    ComputeBackend& be = primary_backend(resources);
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    // The sweep range size (subset or whole P) — used for the maxcomb cap (a pure compare).
    const int P = f2.P;
    const int range = req.pop_subset.empty() ? P : static_cast<int>(req.pop_subset.size());
    const unsigned long long enumerated = choose_saturating(range, k);
    // The clamped host estimate is the count REPORTED only on the two early-return paths
    // (the empty `range < k` Ok and the maxcomb `capped` refusal below). On the dispatch
    // path the backend echoes its EXACT survivor-range count, which overwrites res.enumerated
    // after f4_sweep/f3_sweep — so we set it here only in the branches that return early,
    // never as a dead store the compute path immediately replaces.
    const std::size_t enumerated_estimate = static_cast<std::size_t>(
        enumerated > static_cast<unsigned long long>(SIZE_MAX) ? SIZE_MAX : enumerated);

    if (range < k) {  // too few pops to form even one item — a clean empty Ok result.
        res.enumerated = enumerated_estimate;
        res.status = Status::Ok;
        return res;
    }

    // MAXCOMB CAP — fired BEFORE any device work. The filter bounds the OUTPUT (which rows are
    // written); every enumerated item is still computed on the GPU to TEST it, so a sweep over
    // more than kFstatMaxComb items is refused unless `sure` (it guards COMPUTE TIME).
    if (enumerated > kFstatMaxComb && !req.sure) {
        res.enumerated = enumerated_estimate;
        res.capped = true;
        res.status = Status::InvalidConfig;
        return res;
    }

    // Map the public request -> the backend device-config. The backend maintains the top-K ON
    // THE DEVICE (a fixed CAP=O(K) reservoir with a rising tau), so it returns at most top_k rows
    // already sorted by |z| descending — NO unbounded host vector, NO host re-rank.
    SweepConfig cfg;
    cfg.k = k;
    cfg.filter_mode = (req.filter == SweepFilter::MinZ) ? 0 : 1;  // 0 = fixed-tau MinZ; 1 = rising-tau TopK.
    cfg.min_z = req.min_z;
    // top_k is the device reservoir cap. For TopK it is the requested K (the user's intent). For
    // MinZ it is a hard SAFETY CEILING (so even a billions-item min-z sweep cannot OOM the host) —
    // NOT req.top_k (which is meaningless when the filter is MinZ; the public struct defaults it to
    // a small 100). A 0/unset TopK K also falls back to the bounded default so the reservoir is
    // always finite. The ceiling is large (kFstatDefaultSweepTopK = 1e6) so a MinZ sweep returns
    // ALL its |z|>=min_z survivors unless they exceed a million (then it keeps the most extreme K).
    cfg.top_k = (req.filter == SweepFilter::TopK && req.top_k > 0)
                    ? req.top_k
                    : kFstatDefaultSweepTopK;
    cfg.pop_subset = req.pop_subset;
    cfg.sure = req.sure;

    // Dispatch the WHOLE on-device sweep (enumerate+compute+filter+compact) and receive ONLY
    // the survivors. A device fault PROPAGATES as an exception (the app maps it to a fault
    // exit); a domain outcome is the returned status (record-and-continue).
    SweepSurvivors sv = (k == 4) ? be.f4_sweep(f2, cfg, prec) : be.f3_sweep(f2, cfg, prec);
    res.status = sv.status;
    res.enumerated = sv.enumerated;  // the backend's exact count (it knows the survivor range).
    res.capped = sv.capped;
    if (sv.status != Status::Ok) return res;

    // Move the survivor arrays onto the result + compute p on the (small, <=top_k) survivor set.
    // The DEVICE already maintained the top-K (a bounded rising-tau reservoir) and returned the
    // rows sorted by |z| descending — NO host re-rank, NO unbounded host vector. The driver only
    // moves the <=K rows across and computes the two-sided p (a per-survivor O(K) loop, not O(N)).
    const std::size_t n = sv.est.size();
    res.keys = std::move(sv.keys);
    res.est = std::move(sv.est);
    res.se = std::move(sv.se);
    res.z = std::move(sv.z);
    res.p.assign(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) res.p[r] = f4_two_sided_p(res.z[r]);

    res.survivors = res.keys.size();
    return res;
}

}  // namespace

SweepResult run_f4_sweep(const device::DeviceF2Blocks& f2, const SweepRequest& req,
                         device::Resources& resources) {
    return run_sweep_impl(f2, req, resources, 4);
}

SweepResult run_f3_sweep(const device::DeviceF2Blocks& f2, const SweepRequest& req,
                         device::Resources& resources) {
    return run_sweep_impl(f2, req, resources, 3);
}

}  // namespace steppe
