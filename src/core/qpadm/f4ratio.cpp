// src/core/qpadm/f4ratio.cpp
//
// run_f4ratio: the standalone f4-ratio statistic (alpha, SE, z) for a batch of
// five-population tuples. Numerator and denominator quartets are assembled in one
// call so they share a single set of surviving genome blocks; the ratio jackknife
// itself is delegated to a shared on-device backend routine.
//
// Reference: docs/reference/src_core_qpadm_f4ratio.cpp.md

#include "steppe/f4ratio.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/qpadm/qpadm_fit.hpp"
#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

using core::idx;

using device::primary_backend;

namespace {

// Named constants — reference §6
inline constexpr double kSetmissThresh = 1e-6;

// Shared implementation body — reference §9
template <class F2Src>
F4RatioResult run_f4ratio_impl(ComputeBackend& be, const F2Src& f2,
                               std::span<const std::array<int, 5>> tuples,
                               const QpAdmOptions& opts) {
    (void)opts;
    const Precision prec = core::qpadm::default_fit_precision();

    F4RatioResult res;
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    const int N = static_cast<int>(tuples.size());
    if (N <= 0) {
        res.status = Status::Ok;
        return res;
    }

    res.p1.reserve(idx(N));
    res.p2.reserve(idx(N));
    res.p3.reserve(idx(N));
    res.p4.reserve(idx(N));
    res.p5.reserve(idx(N));
    std::vector<int> flat;
    flat.reserve(idx(N) * 8);
    for (const std::array<int, 5>& t : tuples) {
        res.p1.push_back(t[0]);
        res.p2.push_back(t[1]);
        res.p3.push_back(t[2]);
        res.p4.push_back(t[3]);
        res.p5.push_back(t[4]);
        flat.push_back(t[0]);
        flat.push_back(t[1]);
        flat.push_back(t[2]);
        flat.push_back(t[3]);
    }
    for (const std::array<int, 5>& t : tuples) {
        flat.push_back(t[0]);
        flat.push_back(t[1]);
        flat.push_back(t[4]);
        flat.push_back(t[3]);
    }

    const RatioBlockJackknife jk =
        be.f4ratio_blocks_jackknife(f2, std::span<const int>(flat), N, kSetmissThresh, prec);
    res.alpha = jk.est;
    res.se = jk.se;
    res.z = jk.z;
    res.status = jk.status;
    return res;
}

}  // namespace

// Public entry points — reference §9
F4RatioResult run_f4ratio(const device::DeviceF2Blocks& f2,
                          std::span<const std::array<int, 5>> tuples,
                          const QpAdmOptions& opts, device::Resources& resources) {
    return run_f4ratio_impl(primary_backend(resources), f2, tuples, opts);
}

F4RatioResult run_f4ratio(const F2BlockTensor& f2_host,
                          std::span<const std::array<int, 5>> tuples,
                          const QpAdmOptions& opts, device::Resources& resources) {
    return run_f4ratio_impl(primary_backend(resources), f2_host, tuples, opts);
}

}  // namespace steppe
