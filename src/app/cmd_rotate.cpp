// src/app/cmd_rotate.cpp
//
// The `steppe qpadm-rotate` command: enumerate every source-subset of a pool and fit
// the whole model list in one batched GPU call. Plain C++20, app-only — the GPU is
// reached only through the CUDA-free device seams; main() owns stdout/stderr.
//
// Reference: docs/reference/src_app_cmd_rotate.cpp.md
#include "app/cmd_rotate.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Pool-subset enumerator — reference §3
[[nodiscard]] std::vector<QpAdmModel> enumerate_pool_subsets(
    int target_idx, const std::vector<int>& pool_idx, const std::vector<int>& right_idx,
    int lo, int hi) {
    std::vector<QpAdmModel> models;
    const int n = static_cast<int>(pool_idx.size());
    int counter = 0;
    for (int k = lo; k <= hi; ++k) {
        if (k < 1 || k > n) continue;
        std::vector<int> c(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) c[static_cast<std::size_t>(i)] = i;
        while (true) {
            QpAdmModel m;
            m.target = target_idx;
            m.left.reserve(static_cast<std::size_t>(k));
            for (int i = 0; i < k; ++i)
                m.left.push_back(pool_idx[static_cast<std::size_t>(c[static_cast<std::size_t>(i)])]);
            m.right = right_idx;
            m.model_index = counter++;
            models.push_back(std::move(m));

            int i = k - 1;
            while (i >= 0 && c[static_cast<std::size_t>(i)] == n - k + i) --i;
            if (i < 0) break;
            ++c[static_cast<std::size_t>(i)];
            for (int j = i + 1; j < k; ++j)
                c[static_cast<std::size_t>(j)] = c[static_cast<std::size_t>(j - 1)] + 1;
        }
    }
    return models;
}

}  // namespace

// The qpadm-rotate command pipeline — reference §4
int run_qpadm_rotate_command(const cfg::RunConfig& config) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpadm-rotate: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpadm-rotate: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe qpadm-rotate: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    if (config.target().empty()) {
        std::fprintf(stderr, "steppe qpadm-rotate: --target is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.right().empty()) {
        std::fprintf(stderr, "steppe qpadm-rotate: --right needs at least one outgroup population\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.pool().empty()) {
        std::fprintf(stderr, "steppe qpadm-rotate: --pool needs at least one source population\n");
        return cfg::kExitInvalidConfig;
    }

    const ResolveResult t = resolver.resolve(config.target());
    if (!t.ok) { std::fprintf(stderr, "steppe qpadm-rotate: %s\n", t.error.c_str()); return cfg::kExitInvalidConfig; }
    const ResolveListResult rt = resolver.resolve_all(config.right());
    if (!rt.ok) { std::fprintf(stderr, "steppe qpadm-rotate: %s\n", rt.error.c_str()); return cfg::kExitInvalidConfig; }
    const ResolveListResult pl = resolver.resolve_all(config.pool());
    if (!pl.ok) { std::fprintf(stderr, "steppe qpadm-rotate: %s\n", pl.error.c_str()); return cfg::kExitInvalidConfig; }

    const int target_idx = t.index;
    const std::vector<int>& right_idx = rt.indices;
    const std::vector<int>& pool_idx = pl.indices;

    const int pool_n = static_cast<int>(pool_idx.size());
    const int lo = config.min_sources();
    int hi = (config.max_sources() == -1) ? pool_n : config.max_sources();
    if (hi > pool_n) hi = pool_n;
    if (lo > pool_n) {
        std::fprintf(stderr,
                     "steppe qpadm-rotate: --min-sources (%d) exceeds the pool size (%d) "
                     "— no models to fit\n", lo, pool_n);
        return cfg::kExitInvalidConfig;
    }

    const std::vector<QpAdmModel> models =
        enumerate_pool_subsets(target_idx, pool_idx, right_idx, lo, hi);
    if (models.empty()) {
        std::fprintf(stderr,
                     "steppe qpadm-rotate: the [min,max]-sources band over the pool "
                     "enumerated zero models\n");
        return cfg::kExitInvalidConfig;
    }

    if (config.device().devices.size() >= 2) {
        std::fprintf(stderr,
                     "steppe qpadm-rotate: WARNING: %zu devices requested; the rotation "
                     "runs single-GPU-preferred (it is host-bounce-capped on no-P2P "
                     "consumer cards). Use --device 0.\n",
                     config.device().devices.size());
    }

    const QpAdmOptions opts = config.qpadm_options();
    std::vector<QpAdmResult> results;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe qpadm-rotate: no CUDA device available (steppe is a GPU "
                         "product; a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        results = run_qpadm_search(
            dev_f2, std::span<const QpAdmModel>(models), opts, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe qpadm-rotate: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    const std::string target_label = resolver.label_at(target_idx);
    std::vector<std::vector<std::string>> left_labels_per_model(models.size());
    for (std::size_t i = 0; i < models.size(); ++i) {
        left_labels_per_model[i].reserve(models[i].left.size());
        for (int idx : models[i].left)
            left_labels_per_model[i].push_back(resolver.label_at(idx));
    }
    const int right_n = static_cast<int>(right_idx.size()) - 1;

    if (const auto rc = emit_to_destination(
            config, "qpadm-rotate", [&](std::ostream& os, OutputFormat fmt) {
                emit_rotation_table(os, fmt, std::span<const QpAdmResult>(results),
                                    target_label, left_labels_per_model, right_n);
            })) {
        return *rc;
    }

    return cfg::kExitOk;
}

}  // namespace steppe::app
