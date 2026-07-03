// src/app/cmd_qpdstat.cpp
//
// The `steppe qpdstat` command: a D-statistic / f4 over four populations, with two
// back ends chosen by input — a thin f4 wrapper over an f2_blocks dir (--f2-dir) and
// a genotype-reading normalized-D path (--prefix). App-layer C++ only; no CUDA header.
//
// Reference: docs/reference/src_app_cmd_qpdstat.cpp.md
#include "app/cmd_qpdstat.hpp"

#include <algorithm>
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
#include "steppe/config.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/dstat.hpp"
#include "steppe/error.hpp"
#include "steppe/f4.hpp"

#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/ind_reader.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Build the quartet name table — reference §4
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
        err = "qpdstat needs quartets: either --pop1/--pop2/--pop3/--pop4 (row-aligned) or "
              "--pops p1,p2,p3,p4[,...] (names in groups of 4)";
        return false;
    }
    if (pops.size() % 4 != 0) {
        err = "--pops for qpdstat must be a multiple of 4 names (each quadruple is "
              "p1,p2,p3,p4); got " + std::to_string(pops.size());
        return false;
    }
    const std::size_t n = pops.size() / 4;
    quartets.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        quartets.push_back({pops[4 * k + 0], pops[4 * k + 1],
                            pops[4 * k + 2], pops[4 * k + 3]});
    return true;
}

// Resolve quartet names to population indices — reference §5
[[nodiscard]] bool resolve_quartets(const PopResolver& resolver,
                                    const std::vector<std::array<std::string, 4>>& quartet_names,
                                    std::vector<std::array<int, 4>>& quadruples,
                                    std::vector<std::string>& l1,
                                    std::vector<std::string>& l2,
                                    std::vector<std::string>& l3,
                                    std::vector<std::string>& l4,
                                    std::string& err) {
    quadruples.reserve(quartet_names.size());
    l1.reserve(quartet_names.size()); l2.reserve(quartet_names.size());
    l3.reserve(quartet_names.size()); l4.reserve(quartet_names.size());
    for (const std::array<std::string, 4>& q : quartet_names) {
        std::array<int, 4> idx{};
        for (int c = 0; c < 4; ++c) {
            const ResolveResult rr = resolver.resolve(q[static_cast<std::size_t>(c)]);
            if (!rr.ok) {
                err = rr.error;
                return false;
            }
            idx[static_cast<std::size_t>(c)] = rr.index;
        }
        quadruples.push_back(idx);
        l1.push_back(resolver.label_at(idx[0]));
        l2.push_back(resolver.label_at(idx[1]));
        l3.push_back(resolver.label_at(idx[2]));
        l4.push_back(resolver.label_at(idx[3]));
    }
    return true;
}

// Genotype path (--prefix): normalized D — reference §6
[[nodiscard]] int run_qpdstat_prefix(const cfg::RunConfig& config) {
    const std::string& prefix = config.qpdstat_prefix();
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);
    const std::string& geno = triple.geno;
    const std::string& snp = triple.snp;
    const std::string& ind = triple.ind;

    std::vector<std::array<std::string, 4>> quartet_names;
    std::string qerr;
    if (!build_quartet_names(config, quartet_names, qerr)) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", qerr.c_str());
        return cfg::kExitInvalidConfig;
    }

    std::vector<std::string> pop_union;
    for (const std::array<std::string, 4>& q : quartet_names) {
        for (const std::string& nm : q) {
            if (std::find(pop_union.begin(), pop_union.end(), nm) == pop_union.end())
                pop_union.push_back(nm);
        }
    }

    std::vector<std::string> pop_labels;
    try {
        io::GenoReader reader(geno);
        const io::GenoFormat geno_fmt = reader.header().format;
        const std::size_t n_present = reader.records_present();
        io::PopSelection sel;
        sel.mode = io::PopSelection::Mode::Explicit;
        sel.labels = pop_union;
        const io::IndPartition part = io::read_ind_partition(geno_fmt, ind, sel, n_present);
        pop_labels.reserve(part.groups.size());
        for (const io::PopGroup& g : part.groups) pop_labels.push_back(g.label);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe qpdstat: input error: %s\n", e.what());
        return cfg::kExitIoError;
    }

    const PopResolver resolver(pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<int, 4>> quadruples;
    std::vector<std::string> l1, l2, l3, l4;
    std::string rerr;
    if (!resolve_quartets(resolver, quartet_names, quadruples, l1, l2, l3, l4, rerr)) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", rerr.c_str());
        return cfg::kExitInvalidConfig;
    }

    const double blgsize_morgans = config.blgsize_cm() / kCentimorgansPerMorgan;
    DstatResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe qpdstat: no CUDA device available (steppe is a GPU product; "
                         "a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        result = run_dstat(geno, snp, ind, std::span<const std::string>(pop_union),
                           std::span<const std::array<int, 4>>(quadruples),
                           blgsize_morgans, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe qpdstat: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    F4Result shim;
    shim.p1 = result.p1; shim.p2 = result.p2; shim.p3 = result.p3; shim.p4 = result.p4;
    shim.est = result.est; shim.se = result.se; shim.z = result.z; shim.p = result.p;
    shim.status = result.status; shim.precision_tag = result.precision_tag;

    if (const auto rc = emit_to_destination(
            config, "qpdstat", [&](std::ostream& os, OutputFormat fmt) {
                emit_f4_result(os, fmt, shim, l1, l2, l3, l4);
            })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace

// Dispatch: choosing a mode — reference §2
int run_qpdstat_command(const cfg::RunConfig& config) {
    if (!config.qpdstat_prefix().empty()) {
        return run_qpdstat_prefix(config);
    }

    if (config.sweep_all_combinations()) {
        return run_fstat_sweep(config, /*k=*/4, "qpdstat");
    }

    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpdstat: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<std::string, 4>> quartet_names;
    std::string qerr;
    if (!build_quartet_names(config, quartet_names, qerr)) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", qerr.c_str());
        return cfg::kExitInvalidConfig;
    }

    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    std::vector<std::array<int, 4>> quartets;
    std::vector<std::string> l1, l2, l3, l4;
    std::string rerr;
    if (!resolve_quartets(resolver, quartet_names, quartets, l1, l2, l3, l4, rerr)) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", rerr.c_str());
        return cfg::kExitInvalidConfig;
    }

    const QpAdmOptions opts = config.qpadm_options();
    F4Result result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe qpdstat: no CUDA device available (steppe is a GPU product; "
                         "a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_f4(dev_f2, std::span<const std::array<int, 4>>(quartets), opts,
                        resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe qpdstat: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (const auto rc = emit_to_destination(
            config, "qpdstat", [&](std::ostream& os, OutputFormat fmt) {
                emit_f4_result(os, fmt, result, l1, l2, l3, l4);
            })) {
        return *rc;
    }

    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
