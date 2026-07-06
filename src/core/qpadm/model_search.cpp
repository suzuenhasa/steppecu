// src/core/qpadm/model_search.cpp — orchestrates fitting a whole list of qpAdm
// models: one shared per-model fit chain, shard/batch routing, and the multi-GPU
// fan-out. CUDA-free — all device work happens behind ComputeBackend virtuals.
//
// Reference: docs/reference/src_core_qpadm_model_search.cpp.md

#include "core/qpadm/model_search.hpp"

#include <cstddef>
#include <exception>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "core/qpadm/model_search_core.hpp"
#include "core/qpadm/qpadm_bounds.hpp"
#include "core/qpadm/qpadm_fit.hpp"
#include "device/backend.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"

namespace steppe::core::qpadm {

// Single per-model fit chain — reference §2
template <class F2Src>
QpAdmResult fit_one_model(ComputeBackend& be, const F2Src& f2,
                          const QpAdmModel& model, const QpAdmOptions& opts) {
    const Precision prec = default_fit_precision();
    const std::vector<int> left_idx = left_with_target(model);
    F4Blocks X = be.assemble_f4(f2, std::span<const int>(left_idx),
                                std::span<const int>(model.right), prec);
    std::span<const int> bs(X.block_sizes);
    QpAdmResult res = run_impl(be, std::move(X), bs, model, opts);
    res.model_index = model.model_index;
    return res;
}

// Device forwarder onto the fit chain — reference §2
QpAdmResult fit_one_model_device(ComputeBackend& be,
                                 const device::DeviceF2Blocks& f2,
                                 const QpAdmModel& model,
                                 const QpAdmOptions& opts) {
    return fit_one_model(be, f2, model, opts);
}

// Default per-model batch body / large-model tail — reference §3
std::vector<QpAdmResult> fit_models_batched_default(
    ComputeBackend& be, const device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models, const QpAdmOptions& opts) {
    std::vector<QpAdmResult> out;
    out.reserve(models.size());
    for (const QpAdmModel& m : models) {
        out.push_back(fit_one_model_device(be, f2, m, opts));
    }
    return out;
}

// Small-path partition gate — reference §4
bool model_in_small_path(const QpAdmModel& model, const QpAdmOptions& opts) {
    const int nl = static_cast<int>(model.left.size());
    const int nr = static_cast<int>(model.right.size()) - 1;
    const int r = (opts.rank < 0) ? (nl - 1) : opts.rank;
    return model_fits_small_path(nl, nr, r);
}

// Shard result scatter helper — reference §5
void scatter_by_pos(std::vector<QpAdmResult>& out, std::vector<QpAdmResult>& results,
                    const std::vector<std::size_t>& pos) {
    for (std::size_t k = 0; k < results.size() && k < pos.size(); ++k)
        out[pos[k]] = std::move(results[k]);
}

// Per-GPU shard fitter — reference §5
std::vector<QpAdmResult> fit_shard(ComputeBackend& be, const device::DeviceF2Blocks& f2,
                                   std::span<const QpAdmModel> models,
                                   const QpAdmOptions& opts) {
    const Precision prec = default_fit_precision();
    std::vector<QpAdmResult> out(models.size());

    std::vector<QpAdmModel> small, large;
    std::vector<std::size_t> small_pos, large_pos;
    small.reserve(models.size());
    for (std::size_t i = 0; i < models.size(); ++i) {
        if (model_in_small_path(models[i], opts)) {
            small.push_back(models[i]); small_pos.push_back(i);
        } else {
            large.push_back(models[i]); large_pos.push_back(i);
        }
    }

    if (!small.empty()) {
        std::vector<QpAdmResult> small_results =
            be.provides_batched_fit()
                ? be.fit_models_batched(f2, std::span<const QpAdmModel>(small), opts, prec)
                : fit_models_batched_default(be, f2, std::span<const QpAdmModel>(small), opts);
        scatter_by_pos(out, small_results, small_pos);
    }
    if (!large.empty()) {
        std::vector<QpAdmResult> large_results =
            fit_models_batched_default(be, f2, std::span<const QpAdmModel>(large), opts);
        scatter_by_pos(out, large_results, large_pos);
    }
    return out;
}

}  // namespace steppe::core::qpadm

namespace steppe {

using core::idx;

namespace {

// Multi-GPU f2 replication handle — reference §8
struct F2Replication {
    std::vector<const device::DeviceF2Blocks*> per_device;
    std::vector<device::DeviceF2Blocks> owned;
};

// Deterministic re-sort by pre-sized slots — reference §6
void scatter_into_slots(std::vector<QpAdmResult>& results,
                        std::vector<QpAdmResult>& batch, std::size_t n) {
    for (QpAdmResult& r : batch) {
        const int mi = r.model_index;
        if (mi < 0 || idx(mi) >= n)
            throw std::runtime_error("run_qpadm_search: model_index out of range "
                                     "(must be 0..n-1 for the deterministic re-sort)");
        results[idx(mi)] = std::move(r);
    }
}

// One-time f2 replication broadcast — reference §8
F2Replication replicate_f2(const device::DeviceF2Blocks& f2, device::Resources& resources) {
    const std::size_t G = resources.device_count();
    F2Replication rep;
    rep.per_device.assign(G, nullptr);
    std::vector<bool> needs_upload(G, false);
    std::size_t n_owned = 0;
    for (std::size_t g = 0; g < G; ++g) {
        if (resources.gpus[g].device_id != f2.device_id) {
            needs_upload[g] = true;
            ++n_owned;
        }
    }
    F2BlockTensor host;
    if (n_owned > 0) host = f2.to_host();
    rep.owned.reserve(n_owned);
    for (std::size_t g = 0; g < G; ++g) {
        if (needs_upload[g]) {
            rep.owned.push_back(
                device::upload_f2_blocks_to_device(host, resources.gpus[g].device_id));
            rep.per_device[g] = &rep.owned.back();
        } else {
            rep.per_device[g] = &f2;
        }
    }
    return rep;
}

}  // namespace

// Device search entry point — reference §7
std::vector<QpAdmResult> run_qpadm_search(const device::DeviceF2Blocks& f2,
                                          std::span<const QpAdmModel> models,
                                          const QpAdmOptions& opts,
                                          device::Resources& resources) {
    const std::size_t G = resources.device_count();
    const std::size_t n = models.size();
    if (G == 0) throw std::runtime_error("run_qpadm_search: no devices in Resources");

    std::vector<QpAdmResult> results(n);
    if (n == 0) return results;

    if (G == 1) {
        ComputeBackend& be = *resources.gpus[0].backend;
        std::vector<QpAdmResult> batch =
            core::qpadm::fit_shard(be, f2, models, opts);
        scatter_into_slots(results, batch, n);
        return results;
    }

    const F2Replication rep = replicate_f2(f2, resources);
    const std::vector<core::qpadm::ModelShard> shards =
        core::qpadm::plan_model_shards(n, G);

    std::vector<std::exception_ptr> worker_errors(G);
    {
        std::vector<std::jthread> workers;
        workers.reserve(G);
        for (std::size_t g = 0; g < G; ++g) {
            workers.emplace_back([&, g]() {
                try {
                    const core::qpadm::ModelShard& sh = shards[g];
                    if (sh.lo >= sh.hi) return;
                    ComputeBackend& be = *resources.gpus[g].backend;
                    const device::DeviceF2Blocks& f2_g = *rep.per_device[g];
                    const std::span<const QpAdmModel> sub =
                        models.subspan(sh.lo, sh.hi - sh.lo);
                    std::vector<QpAdmResult> batch =
                        core::qpadm::fit_shard(be, f2_g, sub, opts);
                    scatter_into_slots(results, batch, n);
                } catch (...) {
                    worker_errors[g] = std::current_exception();
                }
            });
        }
    }

    for (std::size_t g = 0; g < G; ++g) {
        if (worker_errors[g]) std::rethrow_exception(worker_errors[g]);
    }
    return results;
}

// Host-oracle search entry point — reference §9
std::vector<QpAdmResult> run_qpadm_search(const F2BlockTensor& f2_host,
                                          std::span<const QpAdmModel> models,
                                          const QpAdmOptions& opts,
                                          device::Resources& resources) {
    ComputeBackend& be = *resources.gpus.at(0).backend;
    const std::size_t n = models.size();
    std::vector<QpAdmResult> results(n);
    for (std::size_t i = 0; i < n; ++i) {
        QpAdmResult r = core::qpadm::fit_one_model(be, f2_host, models[i], opts);
        const int mi = r.model_index;
        const std::size_t slot = (mi >= 0 && idx(mi) < n)
                                     ? idx(mi) : i;
        results[slot] = std::move(r);
    }
    return results;
}

}  // namespace steppe
