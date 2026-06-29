// src/app/cmd_f3.cpp
//
// The `steppe f3` command (standalone f3 statistic; fit-engine §6). f3 is the SIBLING of
// f4, NOT a fork of qpAdm: NO target, NO ALS, NO rank — it computes the AT2 weighted
// block-jackknife f3 POINT ESTIMATE per triple + the jackknife-DIAGONAL SE. The GPU path
// is the deliverable: read the f2_blocks dir -> resolve names->indices via pops.txt ->
// build_resources -> upload_f2_blocks_to_device -> run_f3(DeviceF2Blocks, triples) -> emit
// the table (pop1,pop2,pop3,est,se,z,p — the golden_fit0_f3_readf2.csv schema).
//
// TRIPLES: EITHER the row-aligned --pop1/--pop2/--pop3 columns (admixtools::f3 comb=FALSE)
// OR the single-triple --pops C,A,B convenience (3 names = one triple, or any multiple of 3
// = several triples). The f2-dir load, name->index resolution, build_resources/upload chain,
// and the output sink are REUSED verbatim from cmd_qpwave.cpp/cmd_f4.cpp; the result_emit
// format primitives are reused through emit_f3_result (NO compute/format dup).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is
// reached ONLY through the CUDA-FREE seams (resources.hpp / device_f2_blocks.hpp / f3.hpp).
// main() owns stdout/stderr (architecture.md §10). A DOMAIN outcome is a row + exit 0
// (record-and-continue); only faults return nonzero (cli-bindings.md §1.3, §4.4).
#include "app/cmd_f3.hpp"

#include <array>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "app/cmd_fstat_sweep.hpp"      // run_fstat_sweep (the GPU sweep, --all-triples mode)
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/f3.hpp"                // steppe::run_f3 + F3Result/options

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

/// Build the triple NAME table (one row per triple, three names each) from the frozen
/// config. Prefers the row-aligned --pop1/--pop2/--pop3 columns; falls back to the --pops
/// 3-tuple convenience (the names are taken in groups of 3). Returns false (with a reason in
/// `err`) on no input or a malformed shape (mismatched columns / non-multiple-of-3 --pops).
/// On success `triples[k]` = {pop1=C, pop2=A, pop3=B} of triple k.
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

    // --pops convenience: names in groups of 3 = triples (3 names = one triple).
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
    // ---- SWEEP MODE (--all-triples): route to the GPU sweep over C(P,3) of the --pops
    // SUBSET (empty ⇒ the whole f2 dir). SEPARATE from the explicit-list path below (the
    // goldens gate it byte-identical). The sweep body lives in cmd_fstat_sweep.cpp
    // (run_fstat_sweep / run_f3_sweep).
    if (config.sweep_all_combinations()) {
        return run_fstat_sweep(config, /*k=*/3, "f3");
    }

    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) — REUSE cmd_qpwave path -----
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe f3: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe f3: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Build the triple name table + resolve names -> indices ----------------
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

    // Resolve each (p1,p2,p3) name triple to a P-axis index triple. Carry the resolved
    // names back for the emitter (label_at — canonical pops.txt spelling).
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

    // ---- 3/4. build_resources -> upload f2 to the GPU -> run_f3 (GPU path) ----------
    // The GPU is the deliverable (cli-bindings.md §5.4). All three calls are CUDA-FREE
    // seams; a no-GPU box surfaces a clear fault from build_resources. fudge defaults to 0
    // for a bare f3 SE inside run_f3 (NOT qpadm's 1e-4) — opts here is the struct default.
    const QpAdmOptions opts = config.qpadm_options();
    F3Result result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe f3: no CUDA device available (steppe is a GPU product; a "
                         "CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        result = run_f3(dev_f2, std::span<const std::array<int, 3>>(triples), opts,
                        resources);
    } catch (const std::exception& e) {
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) — a FAULT,
        // nonzero exit (cli-bindings.md §1.3). A domain outcome never throws; it arrives as
        // result.status below (record-and-continue, exit 0).
        std::fprintf(stderr, "steppe f3: device error: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    // ---- 5. Emit (CSV default / TSV / JSON) to --out or stdout -------------------
    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe f3: unknown --format '%s' (csv|tsv|json)\n",
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    if (config.out_file().empty()) {
        emit_f3_result(std::cout, fmt, result, l1, l2, l3);
    } else {
        std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe f3: cannot open --out file: %s\n",
                         config.out_file().c_str());
            return cfg::kExitIoError;
        }
        emit_f3_result(out, fmt, result, l1, l2, l3);
    }

    // A DOMAIN outcome (NonSpd over the m-batch) is a table + exit 0 (record-and-continue,
    // cli-bindings.md §1.3); exit_code_for maps those to kExitOk, only faults to nonzero.
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
