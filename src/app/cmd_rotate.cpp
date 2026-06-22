// src/app/cmd_rotate.cpp
//
// The `steppe qpadm-rotate` command (M(cli-3); cli-bindings.md §4.1). Structurally this
// is cmd_qpadm.cpp with the single-model resolve/run swapped for a pool-subset
// enumerator + the BATCHED run_qpadm_search. The f2-dir load, name->index resolution,
// build_resources/upload chain, and the output sink are REUSED verbatim from the qpadm
// command pattern; the result_emit format primitives (fmt_double/quote/status/feasible)
// are reused through emit_rotation_table. The ONLY new logic is the subset enumerator.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is
// reached ONLY through the CUDA-FREE seams. main() owns stdout/stderr. The rotation is
// RECORD-AND-CONTINUE: per-model domain outcomes are rows + exit 0; only build/upload/run
// FAULTS return nonzero (cli-bindings.md §1.3, §4.4).
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

// Enumerate every k-combination of `pool` (k = lo..hi) in LEXICOGRAPHIC order over the
// pool's index order, building one QpAdmModel per subset with a dense 0-based
// model_index running counter. This reproduces golden_rot_generate.R's
// `c(combn(pool,2), combn(pool,3))` order EXACTLY (all 2-subsets then all 3-subsets,
// each lexicographic over the pool order), so model_index aligns with the golden rows.
// `target` is prepended-conceptually-as-L0 by the engine; here target/right are the
// fixed parts shared by every model. The classic combn index walk (the same the
// throughput section of test_qpadm_rotation.cu uses).
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
    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) — REUSE cmd_qpadm path -----
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
    // lo = min_sources (validated >= 1). hi = max_sources (-1 ⇒ whole pool), clamped to
    // the pool size. This is the generic subset-of-pool enumerator, which IS exactly
    // what golden_rot_generate.R does (combn) — no AT2-semantics divergence (OQ-6).
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

    // Resolve the output format up-front (defensive — ConfigBuilder already validated it).
    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe qpadm-rotate: unknown --format '%s' (csv|tsv|json)\n",
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    // ---- 4. Multi-GPU warning (cli-bindings.md §4.5 / §379-382) -------------------
    // The engine's G>=2 path is correct but host-bounce throughput-capped; single-GPU is
    // the supported/recommended default. A warning, NOT a fault.
    if (config.device().devices.size() >= 2) {
        std::fprintf(stderr,
                     "steppe qpadm-rotate: WARNING: %zu devices requested; the rotation "
                     "runs single-GPU-preferred (TODO(multigpu-host-bounce): the rotation "
                     "is host-bounce-capped on no-P2P consumer cards). Use --device 0.\n",
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
        return cfg::kExitRuntimeError;
    }

    // ---- 6. Resolve labels back onto every row + emit the per-model table ----------
    const std::string target_label = resolver.label_at(target_idx);
    std::vector<std::vector<std::string>> left_labels_per_model(models.size());
    for (std::size_t i = 0; i < models.size(); ++i) {
        left_labels_per_model[i].reserve(models[i].left.size());
        for (int idx : models[i].left)
            left_labels_per_model[i].push_back(resolver.label_at(idx));
    }
    // nr convention: right[0] == R0, so right_n == right.size()-1 (== metadata.nr).
    const int right_n = static_cast<int>(right_idx.size()) - 1;

    if (config.out_file().empty()) {
        emit_rotation_table(std::cout, fmt, std::span<const QpAdmResult>(results),
                            target_label, left_labels_per_model, right_n);
    } else {
        std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe qpadm-rotate: cannot open --out file: %s\n",
                         config.out_file().c_str());
            return cfg::kExitIoError;
        }
        emit_rotation_table(out, fmt, std::span<const QpAdmResult>(results),
                            target_label, left_labels_per_model, right_n);
    }

    // RECORD-AND-CONTINUE: a completed rotation emit is exit 0 even when individual rows
    // carry a domain status. Only build/upload/run faults (handled above) are nonzero.
    return cfg::kExitOk;
}

}  // namespace steppe::app
