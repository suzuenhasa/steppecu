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
    res.enumerated = static_cast<std::size_t>(
        enumerated > static_cast<unsigned long long>(SIZE_MAX) ? SIZE_MAX : enumerated);

    if (range < k) {  // too few pops to form even one item — a clean empty Ok result.
        res.status = Status::Ok;
        return res;
    }

    // MAXCOMB CAP — fired BEFORE any device work. The filter bounds the OUTPUT (which rows are
    // written); every enumerated item is still computed on the GPU to TEST it, so a sweep over
    // more than kFstatMaxComb items is refused unless `sure` (it guards COMPUTE TIME).
    if (enumerated > kFstatMaxComb && !req.sure) {
        res.capped = true;
        res.status = Status::InvalidConfig;
        return res;
    }

    // Map the public request -> the backend device-config.
    SweepConfig cfg;
    cfg.k = k;
    cfg.filter_mode = (req.filter == SweepFilter::MinZ) ? 0 : 1;  // 1 = keep-all (TopK).
    cfg.min_z = req.min_z;
    cfg.pop_subset = req.pop_subset;
    cfg.sure = req.sure;

    // Dispatch the WHOLE on-device sweep (enumerate+compute+filter+compact) and receive ONLY
    // the survivors. A device fault PROPAGATES as an exception (the app maps it to a fault
    // exit); a domain outcome is the returned status (record-and-continue).
    const SweepSurvivors sv = (k == 4) ? be.f4_sweep(f2, cfg, prec) : be.f3_sweep(f2, cfg, prec);
    res.status = sv.status;
    res.enumerated = sv.enumerated;  // the backend's exact count (it knows the survivor range).
    res.capped = sv.capped;
    if (sv.status != Status::Ok) return res;

    // Move the survivor arrays onto the result + compute p on the (small) survivor set.
    const std::size_t n = sv.est.size();
    res.keys = sv.keys;
    res.est = sv.est;
    res.se = sv.se;
    res.z = sv.z;
    res.p.assign(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) res.p[r] = f4_two_sided_p(sv.z[r]);

    // TopK: the device kept every item (filter_mode 1) and compacted to the survivor set; rank
    // it host-side to the top K by |z| descending. This is a sort over the COMPACTED set (the
    // survivors that fit VRAM/host budget), NOT a per-item host loop over the full N — the
    // device already dropped degenerate rows. For MinZ the device already selected by |z|.
    if (req.filter == SweepFilter::TopK && n > req.top_k) {
        std::vector<std::size_t> order(n);
        for (std::size_t r = 0; r < n; ++r) order[r] = r;
        std::partial_sort(order.begin(),
                          order.begin() + static_cast<std::ptrdiff_t>(req.top_k), order.end(),
                          [&](std::size_t a, std::size_t b) {
                              const double za = std::fabs(res.z[a]);
                              const double zb = std::fabs(res.z[b]);
                              // NaN sorts last (a degenerate item never beats a finite one).
                              if (std::isnan(za)) return false;
                              if (std::isnan(zb)) return true;
                              return za > zb;
                          });
        order.resize(req.top_k);
        SweepResult top;
        top.precision_tag = res.precision_tag;
        top.enumerated = res.enumerated;
        top.status = res.status;
        top.keys.reserve(req.top_k); top.est.reserve(req.top_k); top.se.reserve(req.top_k);
        top.z.reserve(req.top_k);    top.p.reserve(req.top_k);
        for (std::size_t idx : order) {
            top.keys.push_back(res.keys[idx]);
            top.est.push_back(res.est[idx]);
            top.se.push_back(res.se[idx]);
            top.z.push_back(res.z[idx]);
            top.p.push_back(res.p[idx]);
        }
        top.survivors = top.keys.size();
        return top;
    }

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
