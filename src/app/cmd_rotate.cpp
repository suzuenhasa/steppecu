// src/app/cmd_rotate.cpp
//
// The `steppe qpadm-rotate` command: cmd_qpadm.cpp with the single-model run swapped
// for a pool-subset enumerator feeding the batched run_qpadm_search. The enumerator is
// the only new logic; f2-dir load, name resolution, and emit follow the qpadm pattern.
//
// App-only, no CUDA header — the GPU is reached only through CUDA-free seams. The
// rotation is record-and-continue: per-model domain outcomes are rows with exit 0; only
// build/upload/run faults return nonzero.
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

#include "app/cmd_emit.hpp"             // emit_to_destination (shared open->write->flush->verify)
#include "app/exit_code_for_caught.hpp" // exit_code_for_caught (device OOM -> the OOM exit code)
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/qpadm.hpp"             // steppe::run_qpadm_search + model/result/options

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Enumerate every k-combination of `pool` (k = lo..hi) in lexicographic order over the
// pool's index order, one QpAdmModel per subset with a dense 0-based model_index. The
// order must match the golden generator (golden_rot_generate.R: all 2-subsets, then all
// 3-subsets, each lexicographic) so model_index lines up with the golden rows. target
// and right are fixed across every model; the engine treats target as the L0 row.
[[nodiscard]] std::vector<QpAdmModel> enumerate_pool_subsets(
    int target_idx, const std::vector<int>& pool_idx, const std::vector<int>& right_idx,
    int lo, int hi) {
    std::vector<QpAdmModel> models;
    const int n = static_cast<int>(pool_idx.size());
    int counter = 0;
    for (int k = lo; k <= hi; ++k) {
        if (k < 1 || k > n) continue;
        // c[0..k-1] is the current combination's positions into pool_idx, ascending.
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

            // Advance to the next lexicographic combination (standard algorithm).
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

int run_qpadm_rotate_command(const cfg::RunConfig& config) {
    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) ----------------------------
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpadm-rotate: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpadm-rotate: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Name -> index resolution against pops.txt -----------------------------
    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe qpadm-rotate: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    // Required fields: target, right, pool (the rotation's fixed parts + the pool).
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

    // ---- 3. Enumerate the pool subsets [lo, hi] -> the model list -----------------
    // lo = min_sources (validated >= 1); hi = max_sources (-1 means the whole pool),
    // clamped to the pool size.
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

    // ---- 4. Multi-GPU warning -----------------------------------------------------
    // The multi-GPU path is correct but throughput-capped by host bouncing; single-GPU
    // is the supported default. This is a warning, not a fault.
    if (config.device().devices.size() >= 2) {
        // TODO(multigpu-host-bounce): single-GPU stays the supported path until the
        // device-resident combine lands.
        std::fprintf(stderr,
                     "steppe qpadm-rotate: WARNING: %zu devices requested; the rotation "
                     "runs single-GPU-preferred (it is host-bounce-capped on no-P2P "
                     "consumer cards). Use --device 0.\n",
                     config.device().devices.size());
    }

    // ---- 5. build_resources -> upload f2 -> run_qpadm_search (the BATCHED engine) --
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
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) — a FAULT,
        // nonzero exit. Per-model domain outcomes never throw; they arrive as row status.
        std::fprintf(stderr, "steppe qpadm-rotate: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    // ---- 6. Resolve labels back onto every row + emit the per-model table ----------
    const std::string target_label = resolver.label_at(target_idx);
    std::vector<std::vector<std::string>> left_labels_per_model(models.size());
    for (std::size_t i = 0; i < models.size(); ++i) {
        left_labels_per_model[i].reserve(models[i].left.size());
        for (int idx : models[i].left)
            left_labels_per_model[i].push_back(resolver.label_at(idx));
    }
    // nr convention: right[0] is R0, so right_n = right.size() - 1.
    const int right_n = static_cast<int>(right_idx.size()) - 1;

    // emit_to_destination does open->write->flush->verify: a torn or short write returns
    // kExitIoError instead of silently exiting 0 with a truncated file. It also parses
    // --format (kExitInvalidConfig on an unknown token, though ConfigBuilder already
    // validated it).
    if (const auto rc = emit_to_destination(
            config, "qpadm-rotate", [&](std::ostream& os, OutputFormat fmt) {
                emit_rotation_table(os, fmt, std::span<const QpAdmResult>(results),
                                    target_label, left_labels_per_model, right_n);
            })) {
        return *rc;
    }

    // RECORD-AND-CONTINUE: a completed rotation emit is exit 0 even when individual rows
    // carry a domain status. Only build/upload/run faults (handled above) are nonzero.
    return cfg::kExitOk;
}

}  // namespace steppe::app
