// src/app/cmd_f4ratio.cpp
//
// The `steppe f4-ratio` command: thin app-layer plumbing that turns options into
// 5-population tuples, resolves names to indices, and runs the f4-ratio statistic on
// the GPU. Plain C++20 — reaches the GPU only through CUDA-free header seams.
//
// Reference: docs/reference/src_app_cmd_f4ratio.cpp.md
#include "app/cmd_f4ratio.hpp"

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
#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/error.hpp"
#include "steppe/f4ratio.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// f4-ratio tuple arity (5) — reference §3
constexpr std::size_t kTupleArity = 5;

// Build the 5-tuple name table from the two input forms — reference §4
[[nodiscard]] bool build_tuple_names(const cfg::RunConfig& config,
                                     std::vector<std::array<std::string, kTupleArity>>& tuples,
                                     std::string& err) {
    const auto& p1 = config.pop1();
    const auto& p2 = config.pop2();
    const auto& p3 = config.pop3();
    const auto& p4 = config.pop4();
    const auto& p5 = config.pop5();
    const bool have_cols =
        !p1.empty() || !p2.empty() || !p3.empty() || !p4.empty() || !p5.empty();

    if (have_cols) {
        const std::size_t n = p1.size();
        if (p2.size() != n || p3.size() != n || p4.size() != n || p5.size() != n) {
            err = "--pop1/--pop2/--pop3/--pop4/--pop5 must be ROW-ALIGNED (same length); got " +
                  std::to_string(p1.size()) + "/" + std::to_string(p2.size()) + "/" +
                  std::to_string(p3.size()) + "/" + std::to_string(p4.size()) + "/" +
                  std::to_string(p5.size());
            return false;
        }
        if (n == 0) { err = "--pop1/--pop2/--pop3/--pop4/--pop5 are empty"; return false; }
        tuples.reserve(n);
        for (std::size_t k = 0; k < n; ++k)
            tuples.push_back({p1[k], p2[k], p3[k], p4[k], p5[k]});
        return true;
    }

    const auto& pops = config.pops();
    if (pops.empty()) {
        err = "f4-ratio needs 5-tuples: either --pop1/--pop2/--pop3/--pop4/--pop5 (row-aligned) "
              "or --pops p1,p2,p3,p4,p5[,...] (names in groups of 5)";
        return false;
    }
    if (pops.size() % kTupleArity != 0) {
        err = "--pops for f4-ratio must be a multiple of 5 names (each tuple is "
              "p1,p2,p3,p4,p5); got " + std::to_string(pops.size());
        return false;
    }
    const std::size_t n = pops.size() / kTupleArity;
    tuples.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        tuples.push_back({pops[kTupleArity * k + 0], pops[kTupleArity * k + 1],
                          pops[kTupleArity * k + 2], pops[kTupleArity * k + 3],
                          pops[kTupleArity * k + 4]});
    return true;
}

}  // namespace

// f4-ratio command pipeline — reference §5
int run_f4ratio_command(const cfg::RunConfig& config) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe f4-ratio: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe f4-ratio: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<std::string, kTupleArity>> tuple_names;
    std::string terr;
    if (!build_tuple_names(config, tuple_names, terr)) {
        std::fprintf(stderr, "steppe f4-ratio: %s\n", terr.c_str());
        return cfg::kExitInvalidConfig;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe f4-ratio: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<int, kTupleArity>> tuples;
    tuples.reserve(tuple_names.size());
    std::vector<std::string> p1_labels, p2_labels, p3_labels, p4_labels, p5_labels;
    p1_labels.reserve(tuple_names.size()); p2_labels.reserve(tuple_names.size());
    p3_labels.reserve(tuple_names.size()); p4_labels.reserve(tuple_names.size());
    p5_labels.reserve(tuple_names.size());
    for (const std::array<std::string, kTupleArity>& t : tuple_names) {
        std::array<int, kTupleArity> idx{};
        for (std::size_t c = 0; c < kTupleArity; ++c) {
            const ResolveResult rr = resolver.resolve(t[c]);
            if (!rr.ok) {
                std::fprintf(stderr, "steppe f4-ratio: %s\n", rr.error.c_str());
                return cfg::kExitInvalidConfig;
            }
            idx[c] = rr.index;
        }
        tuples.push_back(idx);
        p1_labels.push_back(resolver.label_at(idx[0]));
        p2_labels.push_back(resolver.label_at(idx[1]));
        p3_labels.push_back(resolver.label_at(idx[2]));
        p4_labels.push_back(resolver.label_at(idx[3]));
        p5_labels.push_back(resolver.label_at(idx[4]));
    }

    const QpAdmOptions opts = config.qpadm_options();
    F4RatioResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe f4-ratio: no CUDA device available (steppe is a GPU product; a "
                         "CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_f4ratio(dev_f2, std::span<const std::array<int, kTupleArity>>(tuples), opts,
                             resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe f4-ratio: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (const auto rc = emit_to_destination(
            config, "f4-ratio", [&](std::ostream& os, OutputFormat fmt) {
                emit_f4ratio_result(os, fmt, result, p1_labels, p2_labels, p3_labels, p4_labels,
                                    p5_labels);
            })) {
        return *rc;
    }

    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
