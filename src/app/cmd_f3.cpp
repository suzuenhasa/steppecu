// src/app/cmd_f3.cpp
//
// The `steppe f3` command: standalone f3 statistic (block-jackknife point estimate
// + jackknife-diagonal SE per triple). Plain C++20, app-only, no CUDA header — the
// GPU is reached only through the CUDA-free resources/device_f2_blocks/f3 seams.
#include "app/cmd_f3.hpp"

#include <array>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/cmd_fstat_sweep.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/error.hpp"
#include "steppe/f3.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

[[nodiscard]] bool build_triple_names(const cfg::RunConfig& config,
                                      std::vector<std::array<std::string, 3>>& triples,
                                      std::string& err) {
    const auto& p1 = config.pop1();
    const auto& p2 = config.pop2();
    const auto& p3 = config.pop3();
    const bool have_cols = !p1.empty() || !p2.empty() || !p3.empty();

    if (have_cols) {
        const std::size_t n = p1.size();
        if (p2.size() != n || p3.size() != n) {
            err = "--pop1/--pop2/--pop3 must be ROW-ALIGNED (same length); got " +
                  std::to_string(p1.size()) + "/" + std::to_string(p2.size()) + "/" +
                  std::to_string(p3.size());
            return false;
        }
        if (n == 0) { err = "--pop1/--pop2/--pop3 are empty"; return false; }
        triples.reserve(n);
        for (std::size_t k = 0; k < n; ++k)
            triples.push_back({p1[k], p2[k], p3[k]});
        return true;
    }

    const auto& pops = config.pops();
    if (pops.empty()) {
        err = "f3 needs triples: either --pop1/--pop2/--pop3 (row-aligned) or "
              "--pops C,A,B[,...] (names in groups of 3)";
        return false;
    }
    if (pops.size() % 3 != 0) {
        err = "--pops for f3 must be a multiple of 3 names (each triple is C,A,B); "
              "got " + std::to_string(pops.size());
        return false;
    }
    const std::size_t n = pops.size() / 3;
    triples.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        triples.push_back({pops[3 * k + 0], pops[3 * k + 1], pops[3 * k + 2]});
    return true;
}

}  // namespace

int run_f3_command(const cfg::RunConfig& config) {
    if (config.sweep_all_combinations()) {
        return run_fstat_sweep(config, /*k=*/3, "f3");
    }

    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe f3: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe f3: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<std::string, 3>> triple_names;
    std::string terr;
    if (!build_triple_names(config, triple_names, terr)) {
        std::fprintf(stderr, "steppe f3: %s\n", terr.c_str());
        return cfg::kExitInvalidConfig;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe f3: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<int, 3>> triples;
    triples.reserve(triple_names.size());
    std::vector<std::string> l1, l2, l3;
    l1.reserve(triple_names.size()); l2.reserve(triple_names.size());
    l3.reserve(triple_names.size());
    for (const std::array<std::string, 3>& t : triple_names) {
        std::array<int, 3> idx{};
        for (int c = 0; c < 3; ++c) {
            const ResolveResult rr = resolver.resolve(t[static_cast<std::size_t>(c)]);
            if (!rr.ok) {
                std::fprintf(stderr, "steppe f3: %s\n", rr.error.c_str());
                return cfg::kExitInvalidConfig;
            }
            idx[static_cast<std::size_t>(c)] = rr.index;
        }
        triples.push_back(idx);
        l1.push_back(resolver.label_at(idx[0]));
        l2.push_back(resolver.label_at(idx[1]));
        l3.push_back(resolver.label_at(idx[2]));
    }

    const QpAdmOptions opts = config.qpadm_options();
    F3Result result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "f3")) return cfg::kExitRuntimeError;
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_f3(dev_f2, std::span<const std::array<int, 3>>(triples), opts,
                        resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe f3: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (const auto rc = emit_to_destination(
            config, "f3", [&](std::ostream& os, OutputFormat fmt) {
                emit_f3_result(os, fmt, result, l1, l2, l3);
            })) {
        return *rc;
    }

    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
