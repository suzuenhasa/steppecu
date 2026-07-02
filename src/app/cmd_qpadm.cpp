// src/app/cmd_qpadm.cpp
//
// The `steppe qpadm` command (M(cli-1); cli-bindings.md §4.1). The GPU path is the
// deliverable: read the f2_blocks dir -> resolve names->indices via pops.txt ->
// build_resources(DeviceConfig) -> upload_f2_blocks_to_device -> run_qpadm(
// DeviceF2Blocks, ...) -> emit tidy CSV/JSON.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is
// reached ONLY through the CUDA-FREE seams (resources.hpp / device_f2_blocks.hpp /
// qpadm.hpp). main() owns stdout/stderr (architecture.md §10 — the library never
// prints). A model-level domain outcome is a row + exit 0 (record-and-continue); only
// faults return nonzero (cli-bindings.md §1.3, §4.4).
#include "app/cmd_qpadm.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "app/cmd_emit.hpp"             // emit_to_destination (shared open->write->flush->verify)
#include "app/exit_code_for_caught.hpp" // exit_code_for_caught (5 -> 3 on a real device OOM, B2)
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/qpadm.hpp"             // steppe::run_qpadm + model/result/options

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Resolve the target + left + right NAMES to P-axis INDICES (cli-bindings.md §4.2).
// On any unknown name the resolver fail-fasts naming the offending label; this prints
// the reason to stderr and signals the caller to return kExitInvalidConfig.
[[nodiscard]] bool resolve_model(const PopResolver& resolver,
                                 const std::string& target,
                                 const std::vector<std::string>& left,
                                 const std::vector<std::string>& right,
                                 QpAdmModel& model) {
    if (target.empty()) {
        std::fprintf(stderr, "steppe qpadm: --target is required\n");
        return false;
    }
    if (left.empty()) {
        std::fprintf(stderr, "steppe qpadm: --left needs at least one source population\n");
        return false;
    }
    if (right.empty()) {
        std::fprintf(stderr, "steppe qpadm: --right needs at least one outgroup population\n");
        return false;
    }

    const ResolveResult t = resolver.resolve(target);
    if (!t.ok) { std::fprintf(stderr, "steppe qpadm: %s\n", t.error.c_str()); return false; }
    const ResolveListResult l = resolver.resolve_all(left);
    if (!l.ok) { std::fprintf(stderr, "steppe qpadm: %s\n", l.error.c_str()); return false; }
    const ResolveListResult r = resolver.resolve_all(right);
    if (!r.ok) { std::fprintf(stderr, "steppe qpadm: %s\n", r.error.c_str()); return false; }

    model.target = t.index;
    model.left = l.indices;
    model.right = r.indices;
    model.model_index = 0;
    return true;
}

}  // namespace

int run_qpadm_command(const cfg::RunConfig& config) {
    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) ---------------------------
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpadm: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpadm: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Name -> index resolution against pops.txt ----------------------------
    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe qpadm: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }
    QpAdmModel model;
    if (!resolve_model(resolver, config.target(), config.left(), config.right(), model)) {
        return cfg::kExitInvalidConfig;
    }

    // ---- 3/4. build_resources -> upload f2 to the GPU -> run_qpadm (GPU path) -----
    // The GPU is the deliverable (cli-bindings.md §5.4). All three calls are CUDA-FREE
    // seams; a no-GPU box surfaces a clear fault from build_resources (it enumerates /
    // binds a CUDA device and throws when none is visible). The device ordinal comes
    // from the resolved DeviceConfig (empty ⇒ auto-enumerate; the upload targets the
    // first configured ordinal, the same single-GPU fit path the reference test uses).
    const QpAdmOptions opts = config.qpadm_options();
    QpAdmResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe qpadm: no CUDA device available (steppe is a GPU "
                         "product; a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_qpadm(dev_f2, model, opts, resources);
    } catch (const std::exception& e) {
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) — a
        // FAULT, nonzero exit (cli-bindings.md §1.3). A domain outcome never throws;
        // it arrives as result.status below (record-and-continue, exit 0).
        std::fprintf(stderr, "steppe qpadm: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    // ---- 5. Emit (CSV default / TSV / JSON) to --out or stdout -------------------
    // Resolve the labels back onto the rows so the output is name-readable.
    std::vector<std::string> left_labels;
    left_labels.reserve(model.left.size());
    for (int idx : model.left) left_labels.push_back(resolver.label_at(idx));
    const std::string target_label = resolver.label_at(model.target);

    // open->write->flush->verify via the shared emit_to_destination (B1): a torn / short
    // write (full disk, closed pipe) returns kExitIoError instead of silently exiting 0 with
    // a truncated file. The helper parses --format (kExitInvalidConfig on an unknown token).
    if (const auto rc = emit_to_destination(
            config, "qpadm", [&](std::ostream& os, OutputFormat fmt) {
                emit_qpadm_result(os, fmt, result, target_label, left_labels);
            })) {
        return *rc;
    }

    // A per-model DOMAIN outcome (RankDeficient/NonSpd/ChisqUndefined) is a row +
    // exit 0 (record-and-continue, cli-bindings.md §1.3); exit_code_for maps those to
    // kExitOk and only the fault categories to nonzero.
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
