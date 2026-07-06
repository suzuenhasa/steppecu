// src/core/qpadm/f3.cpp — the standalone f3-statistic entry (run_f3): per-triple point
// estimate, jackknife-diagonal standard error, z, and p. A near-exact sibling of f4.cpp
// that reuses the assemble + block-jackknife seams and adds no new infrastructure.
//
// Reference: docs/reference/src_core_qpadm_f3.cpp.md

#include "steppe/f3.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "core/qpadm/f3_triples.hpp"
#include "core/qpadm/jackknife.hpp"
#include "core/qpadm/qpadm_fit.hpp"
#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/f4.hpp"

namespace steppe {

using core::idx;

namespace {

// Primary-GPU seam (kPrimaryGpu, primary_backend) — reference §8
inline constexpr std::size_t kPrimaryGpu = 0;

[[nodiscard]] ComputeBackend& primary_backend(device::Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

// Shared run_f3 body — the pipeline: flatten, assemble once, jackknife once, read out — reference §2
template <class F2Src>
F3Result run_f3_impl(ComputeBackend& be, const F2Src& f2,
                     std::span<const std::array<int, 3>> triples,
                     const QpAdmOptions& opts) {
    (void)opts;
    const Precision prec = core::qpadm::default_fit_precision();

    F3Result res;
    res.precision_tag = core::qpadm::honored_tag(prec, be);

    if (triples.size() > idx(std::numeric_limits<int>::max())) {
        res.status = Status::InvalidConfig;
        return res;
    }
    const int N = static_cast<int>(triples.size());
    if (N <= 0) {
        res.status = Status::Ok;
        return res;
    }

    res.p1.reserve(idx(N));
    res.p2.reserve(idx(N));
    res.p3.reserve(idx(N));
    std::vector<int> flat;
    flat.reserve(idx(N) * 3);
    for (const std::array<int, 3>& t : triples) {
        res.p1.push_back(t[0]);
        res.p2.push_back(t[1]);
        res.p3.push_back(t[2]);
        flat.push_back(t[0]);
        flat.push_back(t[1]);
        flat.push_back(t[2]);
    }

    F4Blocks X = core::qpadm::assemble_f3_triples(be, f2, std::span<const int>(flat), prec);
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

// The two public run_f3 overloads (DeviceF2Blocks + F2BlockTensor forwarders) — reference §8
F3Result run_f3(const device::DeviceF2Blocks& f2,
                std::span<const std::array<int, 3>> triples,
                const QpAdmOptions& opts, device::Resources& resources) {
    return run_f3_impl(primary_backend(resources), f2, triples, opts);
}

F3Result run_f3(const F2BlockTensor& f2_host,
                std::span<const std::array<int, 3>> triples,
                const QpAdmOptions& opts, device::Resources& resources) {
    return run_f3_impl(primary_backend(resources), f2_host, triples, opts);
}

}  // namespace steppe
