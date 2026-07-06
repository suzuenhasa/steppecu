// src/app/cmd_qpwave.cpp
//
// The `steppe qpwave` command: the rank/cladality test underlying qpAdm. Structurally
// this is the qpadm command with NO target — `left[0]` is the reference — so it reuses
// the qpadm f2-load / resolve / build-resources / upload / emit plumbing and reaches the
// GPU only through CUDA-free header seams.
//
// Reference: docs/reference/src_app_cmd_qpwave.cpp.md
#include "app/cmd_qpwave.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "core/internal/host_device.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/error.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

}  // namespace

// qpWave command pipeline — reference §2
int run_qpwave_command(const cfg::RunConfig& config) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpwave: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpwave: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

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

    const std::vector<int>& left_idx = l.indices;
    const std::vector<int>& right_idx = r.indices;

    const QpAdmOptions opts = config.qpadm_options();
    QpWaveResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "qpwave")) return cfg::kExitRuntimeError;
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_qpwave(dev_f2,
                            std::span<const int>(left_idx),
                            std::span<const int>(right_idx),
                            opts, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe qpwave: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    std::vector<std::string> left_labels;
    left_labels.reserve(left_idx.size());
    for (int idx : left_idx) left_labels.push_back(resolver.label_at(idx));
    STEPPE_ASSERT(!right_idx.empty(),
                  "right_idx non-empty: R0 convention (validated at config.right() check)");
    const int right_n = static_cast<int>(right_idx.size()) - 1;

    if (const auto rc = emit_to_destination(
            config, "qpwave", [&](std::ostream& os, OutputFormat fmt) {
                emit_qpwave_result(os, fmt, result, left_labels, right_n);
            })) {
        return *rc;
    }

    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
