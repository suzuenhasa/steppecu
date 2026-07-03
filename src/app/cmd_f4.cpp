// src/app/cmd_f4.cpp
//
// The `steppe f4` command: standalone f4 statistic (sibling of qpwave, not a qpAdm fork —
// no target, no ALS, no rank). Plain C++20, app-only, no CUDA header; the GPU is reached
// only through the CUDA-free seams (resources.hpp / device_f2_blocks.hpp / f4.hpp).
#include "app/cmd_f4.hpp"

#include <array>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <span>
#include <string>
#include <vector>

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
#include "steppe/f4.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

[[nodiscard]] bool build_quartet_names(const cfg::RunConfig& config,
                                       std::vector<std::array<std::string, 4>>& quartets,
                                       std::string& err) {
    const auto& p1 = config.pop1();
    const auto& p2 = config.pop2();
    const auto& p3 = config.pop3();
    const auto& p4 = config.pop4();
    const bool have_cols = !p1.empty() || !p2.empty() || !p3.empty() || !p4.empty();

    if (have_cols) {
        const std::size_t n = p1.size();
        if (p2.size() != n || p3.size() != n || p4.size() != n) {
            err = "--pop1/--pop2/--pop3/--pop4 must be ROW-ALIGNED (same length); got " +
                  std::to_string(p1.size()) + "/" + std::to_string(p2.size()) + "/" +
                  std::to_string(p3.size()) + "/" + std::to_string(p4.size());
            return false;
        }
        if (n == 0) { err = "--pop1/--pop2/--pop3/--pop4 are empty"; return false; }
        quartets.reserve(n);
        for (std::size_t k = 0; k < n; ++k)
            quartets.push_back({p1[k], p2[k], p3[k], p4[k]});
        return true;
    }

    const auto& pops = config.pops();
    if (pops.empty()) {
        err = "f4 needs quartets: either --pop1/--pop2/--pop3/--pop4 (row-aligned) or "
              "--pops p1,p2,p3,p4[,...] (names in groups of 4)";
        return false;
    }
    if (pops.size() % 4 != 0) {
        err = "--pops for f4 must be a multiple of 4 names (each quartet is p1,p2,p3,p4); "
              "got " + std::to_string(pops.size());
        return false;
    }
    const std::size_t n = pops.size() / 4;
    quartets.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        quartets.push_back({pops[4 * k + 0], pops[4 * k + 1],
                            pops[4 * k + 2], pops[4 * k + 3]});
    return true;
}

}  // namespace

int run_f4_command(const cfg::RunConfig& config) {
    if (config.sweep_all_combinations()) {
        return run_fstat_sweep(config, /*k=*/4, "f4");
    }

    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe f4: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe f4: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<std::string, 4>> quartet_names;
    std::string qerr;
    if (!build_quartet_names(config, quartet_names, qerr)) {
        std::fprintf(stderr, "steppe f4: %s\n", qerr.c_str());
        return cfg::kExitInvalidConfig;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe f4: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<int, 4>> quartets;
    quartets.reserve(quartet_names.size());
    std::vector<std::string> l1, l2, l3, l4;
    l1.reserve(quartet_names.size()); l2.reserve(quartet_names.size());
    l3.reserve(quartet_names.size()); l4.reserve(quartet_names.size());
    for (const std::array<std::string, 4>& q : quartet_names) {
        std::array<int, 4> idx{};
        for (int c = 0; c < 4; ++c) {
            const ResolveResult rr = resolver.resolve(q[static_cast<std::size_t>(c)]);
            if (!rr.ok) {
                std::fprintf(stderr, "steppe f4: %s\n", rr.error.c_str());
                return cfg::kExitInvalidConfig;
            }
            idx[static_cast<std::size_t>(c)] = rr.index;
        }
        quartets.push_back(idx);
        l1.push_back(resolver.label_at(idx[0]));
        l2.push_back(resolver.label_at(idx[1]));
        l3.push_back(resolver.label_at(idx[2]));
        l4.push_back(resolver.label_at(idx[3]));
    }

    const QpAdmOptions opts = config.qpadm_options();
    F4Result result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe f4: no CUDA device available (steppe is a GPU product; a "
                         "CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_f4(dev_f2, std::span<const std::array<int, 4>>(quartets), opts,
                        resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe f4: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (const auto rc = emit_to_destination(
            config, "f4", [&](std::ostream& os, OutputFormat fmt) {
                emit_f4_result(os, fmt, result, l1, l2, l3, l4);
            })) {
        return *rc;
    }

    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
