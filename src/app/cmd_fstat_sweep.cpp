// src/app/cmd_fstat_sweep.cpp
//
// The `steppe f4-sweep` / `steppe f3-sweep` commands (GPU-only all-combinations
// f-stat sweep). The full C(P,k) table is never materialized: the device filters
// and compacts, and only the survivors cross back to the host.
//
// Reference: docs/reference/src_app_cmd_fstat_sweep.cpp.md
#include "app/cmd_fstat_sweep.hpp"

#include <array>
#include <cstdio>
#include <exception>
#include <filesystem>
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
#include "steppe/fstat_sweep.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Survivor table emitter — reference §5
void emit_sweep(std::ostream& os, OutputFormat fmt, const SweepResult& r,
                const PopResolver& resolver, int k) {
    const char* status = (r.status == Status::Ok) ? "ok" : "error";
    if (fmt == OutputFormat::Json) {
        os << "{\"status\":\"" << status << "\",\"enumerated\":" << r.enumerated
           << ",\"survivors\":" << r.survivors << ",\"capped\":" << (r.capped ? "true" : "false")
           << ",\"rows\":[";
        for (std::size_t i = 0; i < r.survivors; ++i) {
            if (i) os << ",";
            os << "{";
            for (int c = 0; c < k; ++c)
                os << "\"pop" << (c + 1) << "\":"
                   << json_quote(resolver.label_at(r.keys[i][static_cast<std::size_t>(c)])) << ",";
            os << "\"est\":" << r.est[i] << ",\"se\":" << r.se[i] << ",\"z\":" << r.z[i]
               << ",\"p\":" << r.p[i] << "}";
        }
        os << "]}\n";
        return;
    }
    const char d = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    for (int c = 0; c < k; ++c) os << "pop" << (c + 1) << d;
    os << "est" << d << "se" << d << "z" << d << "p\n";
    for (std::size_t i = 0; i < r.survivors; ++i) {
        for (int c = 0; c < k; ++c)
            os << csv_field(resolver.label_at(r.keys[i][static_cast<std::size_t>(c)]), d) << d;
        os << r.est[i] << d << r.se[i] << d << r.z[i] << d << r.p[i] << "\n";
    }
}

}  // namespace

// Shared sweep body: the pipeline — reference §2
int run_fstat_sweep(const cfg::RunConfig& config, int k, const char* cmd) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe %s: --f2-dir is required\n", cmd);
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe %s: %s\n", cmd, dir.error.c_str());
        return cfg::kExitIoError;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe %s: %s\n", cmd, resolver.error().c_str());
        return cfg::kExitIoError;
    }

    std::vector<int> subset;
    for (const std::string& name : config.pops()) {
        const ResolveResult rr = resolver.resolve(name);
        if (!rr.ok) {
            std::fprintf(stderr, "steppe %s: %s\n", cmd, rr.error.c_str());
            return cfg::kExitInvalidConfig;
        }
        subset.push_back(rr.index);
    }

    SweepRequest req;
    req.pop_subset = subset;
    req.sure = config.sweep_sure();
    req.min_z = config.sweep_min_z();
    if (config.sweep_top_k() > 0) {
        req.filter = SweepFilter::TopK;
        req.top_k = static_cast<std::size_t>(config.sweep_top_k());
    } else if (config.sweep_all_combinations()) {
        req.filter = SweepFilter::TopK;
        req.top_k = steppe::kFstatDefaultSweepTopK;
    } else {
        req.filter = SweepFilter::MinZ;
    }

    SweepResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, cmd)) return cfg::kExitRuntimeError;
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = (k == 4) ? run_f4_sweep(dev_f2, req, resources)
                          : run_f3_sweep(dev_f2, req, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe %s: device error: %s\n", cmd, e.what());
        return exit_code_for_caught(e);
    }

    if (result.capped) {
        std::fprintf(stderr,
                     "steppe %s: refusing to enumerate %zu combinations (> the maxcomb cap). "
                     "Pass --sure to override, or restrict --pops.\n",
                     cmd, result.enumerated);
        return cfg::kExitInvalidConfig;
    }

    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe %s: unknown --format '%s' (csv|tsv|json)\n",
                     cmd, config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    if (!config.shard_dir().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(config.shard_dir(), ec);
        if (ec) {
            std::fprintf(stderr, "steppe %s: cannot create --shard-dir '%s': %s\n",
                         cmd, config.shard_dir().c_str(), ec.message().c_str());
            return cfg::kExitIoError;
        }
        const char* ext = (fmt == OutputFormat::Json) ? "json"
                          : (fmt == OutputFormat::Tsv) ? "tsv" : "csv";
        const std::filesystem::path shard_path =
            std::filesystem::path(config.shard_dir()) /
            (std::string("survivors.") + ext);
        std::ofstream out(shard_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe %s: cannot open shard file: %s\n",
                         cmd, shard_path.string().c_str());
            return cfg::kExitIoError;
        }
        emit_sweep(out, fmt, result, resolver, k);
        if (const auto rc = finish_emit(out, cmd, shard_path.string())) return *rc;
        std::fprintf(stderr, "steppe %s: enumerated %zu, %zu survivors -> %s\n",
                     cmd, result.enumerated, result.survivors, shard_path.string().c_str());
        return cfg::exit_code_for(result.status);
    }

    if (const auto rc = emit_to_destination(
            config, cmd, [&](std::ostream& os, OutputFormat out_fmt) {
                emit_sweep(os, out_fmt, result, resolver, k);
            })) {
        return *rc;
    }

    std::fprintf(stderr, "steppe %s: enumerated %zu, %zu survivors\n",
                 cmd, result.enumerated, result.survivors);
    return cfg::exit_code_for(result.status);
}

// Entry points — reference §7
int run_f4_sweep_command(const cfg::RunConfig& config) {
    return run_fstat_sweep(config, 4, "f4-sweep");
}
int run_f3_sweep_command(const cfg::RunConfig& config) {
    return run_fstat_sweep(config, 3, "f3-sweep");
}

}  // namespace steppe::app
