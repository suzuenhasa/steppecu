// src/app/cmd_qpadm.cpp
//
// The `steppe qpadm` command: read an f2_blocks dir, run the qpAdm fit on the GPU,
// and emit one tidy CSV/TSV/JSON result. Plain C++20 app code that reaches the GPU
// only through CUDA-free seams, so it builds without the CUDA toolchain.
//
// Reference: docs/reference/src_app_cmd_qpadm.cpp.md
#include "app/cmd_qpadm.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/error.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Name resolution and building the model — reference §3
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

// The qpadm command pipeline — reference §2
int run_qpadm_command(const cfg::RunConfig& config) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpadm: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpadm: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe qpadm: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }
    QpAdmModel model;
    if (!resolve_model(resolver, config.target(), config.left(), config.right(), model)) {
        return cfg::kExitInvalidConfig;
    }

    const QpAdmOptions opts = config.qpadm_options();
    QpAdmResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "qpadm")) return cfg::kExitRuntimeError;
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_qpadm(dev_f2, model, opts, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe qpadm: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    std::vector<std::string> left_labels;
    left_labels.reserve(model.left.size());
    for (int idx : model.left) left_labels.push_back(resolver.label_at(idx));
    const std::string target_label = resolver.label_at(model.target);

    if (const auto rc = emit_to_destination(
            config, "qpadm", [&](std::ostream& os, OutputFormat fmt) {
                emit_qpadm_result(os, fmt, result, target_label, left_labels);
            })) {
        return *rc;
    }

    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
