// src/app/cmd_qpwave.cpp
//
// The `steppe qpwave` command. qpWave is the rank/cladality test underlying qpAdm: given
// left + right pops (no target; left[0] is the reference) it sweeps the minimum f4 rank
// relating them. Structurally this is cmd_qpadm.cpp minus the target — `left` is the full
// left set, there is no admixture-weight/popdrop output, and the result is the per-rank
// sufficiency sweep + rankdrop table. The f2-dir load, name->index resolution, resource
// build/upload chain, and output sink are reused from cmd_qpadm.cpp.
//
// Plain C++20, app-only, no CUDA header: the GPU is reached only through the CUDA-free
// seams (resources.hpp / device_f2_blocks.hpp / qpadm.hpp). main() owns stdout/stderr; the
// library never prints. A domain outcome is a row + exit 0 (record-and-continue); only
// faults return nonzero.
#include "app/cmd_qpwave.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "app/cmd_emit.hpp"             // emit_to_destination (shared open->write->flush->verify)
#include "app/exit_code_for_caught.hpp" // exit_code_for_caught (maps caught faults, e.g. device OOM)
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
    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt), same as cmd_qpadm.cpp -----
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpwave: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpwave: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Name -> index resolution against pops.txt (no target, left[0]=ref) ----
    // qpWave has no target argument: `left` is the full left set and left[0] is the
    // reference row. We only enforce non-empty here; the engine gates degenerate cases
    // (e.g. nl<2) as a domain status.
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
    // All three calls are CUDA-free seams; a machine with no GPU surfaces a clear fault
    // from build_resources. The DeviceF2Blocks overload is the production GPU path
    // exercised by tests/reference/test_qpwave_parity.cu.
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
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) are faults:
        // nonzero exit. A domain outcome never throws; it arrives as result.status below
        // (record-and-continue, exit 0).
        std::fprintf(stderr, "steppe qpwave: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    // ---- 5. Emit (CSV default / TSV / JSON) to --out or stdout -------------------
    // Resolve the left labels back for the output header (left[0] is the reference).
    std::vector<std::string> left_labels;
    left_labels.reserve(left_idx.size());
    for (int idx : left_idx) left_labels.push_back(resolver.label_at(idx));
    // right[0] is R0, so right_n = right.size()-1. The subtraction would underflow to
    // negative iff right_idx were empty; the earlier config.right() non-empty check
    // guarantees it isn't. Assert it here so the invariant is locally self-evident.
    STEPPE_ASSERT(!right_idx.empty(),
                  "right_idx non-empty: R0 convention (validated at config.right() check)");
    const int right_n = static_cast<int>(right_idx.size()) - 1;

    // open->write->flush->verify via the shared emit_to_destination: a torn / short write
    // (full disk, closed pipe) returns kExitIoError instead of silently exiting 0 with a
    // truncated file. The helper parses --format (kExitInvalidConfig on an unknown token).
    if (const auto rc = emit_to_destination(
            config, "qpwave", [&](std::ostream& os, OutputFormat fmt) {
                emit_qpwave_result(os, fmt, result, left_labels, right_n);
            })) {
        return *rc;
    }

    // A domain outcome (RankDeficient/NonSpd/ChisqUndefined) is a row + exit 0
    // (record-and-continue); exit_code_for maps those to kExitOk and only the fault
    // categories to nonzero.
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
