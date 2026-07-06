// src/core/qpadm/f4.cpp — run_f4, the standalone f4-statistic entry point.
//
// f4 is a sibling of qpWave, not a fork of qpAdm: it reuses assemble_f4_quartets and the
// block jackknife with no new math (no ALS, no rank test) — a per-quartet point estimate
// plus a jackknife-diagonal standard error.
//
// Reference: docs/reference/src_core_qpadm_f4.cpp.md

#include "steppe/f4.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "core/qpadm/f4_quartets.hpp"
#include "core/qpadm/jackknife.hpp"
#include "core/qpadm/qpadm_fit.hpp"
#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

using core::idx;

namespace {

// Primary GPU index + backend accessor — reference §8
inline constexpr std::size_t kPrimaryGpu = 0;

[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

// Shared run_f4 body: the assemble/jackknife/read-out pipeline — reference §2
template <class F2Src>
F4Result run_f4_impl(ComputeBackend& be, const F2Src& f2,
                     std::span<const std::array<int, 4>> quartets,
                     const QpAdmOptions& opts) {
    (void)opts;
    const Precision prec = core::qpadm::default_fit_precision();

    F4Result res;
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    const int N = static_cast<int>(quartets.size());
    if (N <= 0) {
        res.status = Status::Ok;
        return res;
    }

    res.p1.reserve(idx(N));
    res.p2.reserve(idx(N));
    res.p3.reserve(idx(N));
    res.p4.reserve(idx(N));
    std::vector<int> flat;
    flat.reserve(idx(N) * 4);
    for (const std::array<int, 4>& q : quartets) {
        res.p1.push_back(q[0]);
        res.p2.push_back(q[1]);
        res.p3.push_back(q[2]);
        res.p4.push_back(q[3]);
        flat.push_back(q[0]);
        flat.push_back(q[1]);
        flat.push_back(q[2]);
        flat.push_back(q[3]);
    }

    F4Blocks X = core::qpadm::assemble_f4_quartets(be, f2, std::span<const int>(flat), prec);
    const int m = X.nl * X.nr;

    if (m <= 0 || X.n_block <= 0) {
        res.est.assign(idx(N), std::nan(""));
        res.se.assign(idx(N), std::nan(""));
        res.z.assign(idx(N), std::nan(""));
        res.p.assign(idx(N), std::nan(""));
        res.status = Status::Ok;
        return res;
    }

    const JackknifeDiag diag =
        core::qpadm::jackknife_diag(be, X, std::span<const int>(X.block_sizes), prec);

    res.est.assign(idx(N), 0.0);
    res.se.assign(idx(N), 0.0);
    res.z.assign(idx(N), 0.0);
    res.p.assign(idx(N), 0.0);
    for (int k = 0; k < N; ++k) {
        const std::size_t ks = idx(k);
        const double est = X.x_total[ks];
        const double var = diag.var[ks];
        const double se = (var > 0.0) ? std::sqrt(var) : std::nan("");
        const double z = est / se;
        res.est[ks] = est;
        res.se[ks] = se;
        res.z[ks] = z;
        res.p[ks] = f4_two_sided_p(z);
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace

// f4_two_sided_p — the z-to-p conversion — reference §6
double f4_two_sided_p(double z) {
    static const double kInvSqrt2 = 1.0 / std::sqrt(2.0);
    return std::erfc(std::fabs(z) * kInvSqrt2);
}

// Public entry points: the device-resident and host-side f2 overloads — reference §8
F4Result run_f4(const device::DeviceF2Blocks& f2,
                std::span<const std::array<int, 4>> quartets,
                const QpAdmOptions& opts, device::Resources& resources) {
    return run_f4_impl(primary_backend(resources), f2, quartets, opts);
}

F4Result run_f4(const F2BlockTensor& f2_host,
                std::span<const std::array<int, 4>> quartets,
                const QpAdmOptions& opts, device::Resources& resources) {
    return run_f4_impl(primary_backend(resources), f2_host, quartets, opts);
}

}  // namespace steppe
