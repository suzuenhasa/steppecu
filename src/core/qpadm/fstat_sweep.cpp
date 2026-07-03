// src/core/qpadm/fstat_sweep.cpp — host-side driver for the all-combinations
// f-statistic sweep (run_f4_sweep / run_f3_sweep). Host-pure and CUDA-free:
// enumerate, score, filter, and compact all happen on the GPU behind a backend
// virtual; the host only caps the request, forwards it, and finishes the small
// survivor set.
//
// Reference: docs/reference/src_core_qpadm_fstat_sweep.cpp.md
#include "steppe/fstat_sweep.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "core/qpadm/qpadm_fit.hpp"
#include "steppe/config.hpp"
#include "steppe/f4.hpp"

namespace steppe {

namespace {

inline constexpr std::size_t kPrimaryGpu = 0;

[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

// Overflow-safe combination count — reference §3
[[nodiscard]] unsigned long long choose_saturating(int P, int k) {
    if (k < 0 || P < k) return 0ULL;
    unsigned long long result = 1ULL;
    for (int i = 1; i <= k; ++i) {
        const unsigned long long num = static_cast<unsigned long long>(P - k + i);
        if (result > (~0ULL) / (num == 0 ? 1 : num)) return ~0ULL;
        result = result * num / static_cast<unsigned long long>(i);
    }
    return result;
}

// Shared sweep driver — reference §2
[[nodiscard]] SweepResult run_sweep_impl(const device::DeviceF2Blocks& f2,
                                         const SweepRequest& req, device::Resources& resources,
                                         int k) {
    SweepResult res;
    const Precision prec = core::qpadm::default_fit_precision();
    ComputeBackend& be = primary_backend(resources);
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    const int P = f2.P;
    const int range = req.pop_subset.empty() ? P : static_cast<int>(req.pop_subset.size());
    const unsigned long long enumerated = choose_saturating(range, k);
    const std::size_t enumerated_estimate = static_cast<std::size_t>(
        enumerated > static_cast<unsigned long long>(SIZE_MAX) ? SIZE_MAX : enumerated);

    if (range < k) {
        res.enumerated = enumerated_estimate;
        res.status = Status::Ok;
        return res;
    }

    if (enumerated > kFstatMaxComb && !req.sure) {
        res.enumerated = enumerated_estimate;
        res.capped = true;
        res.status = Status::InvalidConfig;
        return res;
    }

    SweepConfig cfg;
    cfg.k = k;
    cfg.filter_mode = (req.filter == SweepFilter::MinZ) ? kSweepFilterMinZ : kSweepFilterTopK;
    cfg.min_z = req.min_z;
    cfg.top_k = (req.filter == SweepFilter::TopK && req.top_k > 0)
                    ? req.top_k
                    : kFstatDefaultSweepTopK;
    cfg.pop_subset = req.pop_subset;
    cfg.sure = req.sure;

    SweepSurvivors sv = (k == 4) ? be.f4_sweep(f2, cfg, prec) : be.f3_sweep(f2, cfg, prec);
    res.status = sv.status;
    res.enumerated = sv.enumerated;
    res.capped = sv.capped;
    if (sv.status != Status::Ok) return res;

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

// The two entry points — reference §8
SweepResult run_f4_sweep(const device::DeviceF2Blocks& f2, const SweepRequest& req,
                         device::Resources& resources) {
    return run_sweep_impl(f2, req, resources, 4);
}

SweepResult run_f3_sweep(const device::DeviceF2Blocks& f2, const SweepRequest& req,
                         device::Resources& resources) {
    return run_sweep_impl(f2, req, resources, 3);
}

}  // namespace steppe
