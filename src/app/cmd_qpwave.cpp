// src/app/cmd_qpwave.cpp
//
// The `steppe qpwave` command (M(cli-2); cli-bindings.md §4.1). qpWave = the rank/cladality
// test underlying qpAdm: given left + right pops (NO target; left[0] is the reference) it
// sweeps the minimum f4 rank relating them. The GPU path is the deliverable: read the
// f2_blocks dir -> resolve names->indices via pops.txt -> build_resources(DeviceConfig) ->
// upload_f2_blocks_to_device -> run_qpwave(DeviceF2Blocks, ...) -> emit the rank-sweep table.
//
// Structurally this is cmd_qpadm.cpp with the single difference that qpWave has NO target:
// `left` IS the full left set (left[0] is the reference row), there is no admixture weight
// / popdrop output, and the result is the per-rank rank-sufficiency sweep + rankdrop table.
// The f2-dir load, name->index resolution, build_resources/upload chain, and the output
// sink are REUSED verbatim from cmd_qpadm.cpp; the result_emit format primitives are reused
// through emit_qpwave_result (NO compute or format duplicated).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is
// reached ONLY through the CUDA-FREE seams (resources.hpp / device_f2_blocks.hpp /
// qpadm.hpp). main() owns stdout/stderr (architecture.md §10 — the library never prints).
// A DOMAIN outcome is a row + exit 0 (record-and-continue); only faults return nonzero
// (cli-bindings.md §1.3, §4.4).
#include "app/cmd_qpwave.hpp"

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
#include "core/internal/host_device.hpp"  // STEPPE_ASSERT (debug-only fail-fast; CUDA-free)
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/qpadm.hpp"             // steppe::run_qpwave + QpWaveResult/options

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

}  // namespace

int run_qpwave_command(const cfg::RunConfig& config) {
    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) — REUSE cmd_qpadm path -----
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpwave: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpwave: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Name -> index resolution against pops.txt (NO target, left[0]=ref) ----
    // qpWave has no target argument: `left` IS the full left set and left[0] is the
    // reference row. We only enforce non-empty (mirroring how cmd_qpadm.cpp checks
    // non-empty); the engine gates degenerate cases (e.g. nl<2) as a domain `status`.
    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe qpwave: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }
    if (config.left().empty()) {
        std::fprintf(stderr,
                     "steppe qpwave: --left needs at least the reference + one population\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.right().empty()) {
        std::fprintf(stderr, "steppe qpwave: --right needs at least one outgroup population\n");
        return cfg::kExitInvalidConfig;
    }
    const ResolveListResult l = resolver.resolve_all(config.left());
    if (!l.ok) { std::fprintf(stderr, "steppe qpwave: %s\n", l.error.c_str()); return cfg::kExitInvalidConfig; }
    const ResolveListResult r = resolver.resolve_all(config.right());
    if (!r.ok) { std::fprintf(stderr, "steppe qpwave: %s\n", r.error.c_str()); return cfg::kExitInvalidConfig; }

    const std::vector<int>& left_idx = l.indices;   // left[0] is the reference
    const std::vector<int>& right_idx = r.indices;  // right[0] is R0

    // ---- 3/4. build_resources -> upload f2 to the GPU -> run_qpwave (GPU path) -----
    // The GPU is the deliverable (cli-bindings.md §5.4). All three calls are CUDA-FREE
    // seams; a no-GPU box surfaces a clear fault from build_resources. The DeviceF2Blocks
    // overload is the production GPU path the parity test's CudaBackend block exercises
    // (test_qpwave_parity.cu:332).
    const QpAdmOptions opts = config.qpadm_options();  // fudge / rank_alpha drive qpWave
    QpWaveResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe qpwave: no CUDA device available (steppe is a GPU "
                         "product; a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_qpwave(dev_f2,
                            std::span<const int>(left_idx),
                            std::span<const int>(right_idx),
                            opts, resources);
    } catch (const std::exception& e) {
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) — a FAULT,
        // nonzero exit (cli-bindings.md §1.3). A domain outcome never throws; it arrives
        // as result.status below (record-and-continue, exit 0).
        std::fprintf(stderr, "steppe qpwave: device error: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    // ---- 5. Emit (CSV default / TSV / JSON) to --out or stdout -------------------
    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        // ConfigBuilder::build() already validates --format, so this is defensive.
        std::fprintf(stderr, "steppe qpwave: unknown --format '%s' (csv|tsv|json)\n",
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    // Resolve the left labels back for the output header (left[0] is the reference).
    std::vector<std::string> left_labels;
    left_labels.reserve(left_idx.size());
    for (int idx : left_idx) left_labels.push_back(resolver.label_at(idx));
    // nr convention: right[0] == R0, so right_n == right.size()-1 (== metadata.nr).
    // The subtraction underflows to negative iff right_idx is empty; the R0
    // convention guarantees non-empty here (validated at the config.right() check
    // ~50 lines above, :75). Make that invariant locally self-evident (debug-only).
    STEPPE_ASSERT(!right_idx.empty(),
                  "right_idx non-empty: R0 convention (validated at config.right() check)");
    const int right_n = static_cast<int>(right_idx.size()) - 1;

    if (config.out_file().empty()) {
        emit_qpwave_result(std::cout, fmt, result, left_labels, right_n);
    } else {
        std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe qpwave: cannot open --out file: %s\n",
                         config.out_file().c_str());
            return cfg::kExitIoError;
        }
        emit_qpwave_result(out, fmt, result, left_labels, right_n);
    }

    // A DOMAIN outcome (RankDeficient/NonSpd/ChisqUndefined) is a row + exit 0
    // (record-and-continue, cli-bindings.md §1.3); exit_code_for maps those to kExitOk and
    // only the fault categories to nonzero.
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
