// src/app/cmd_f4.cpp
//
// The `steppe f4` command (standalone f4 statistic; fit-engine §6). f4 is the SIBLING of
// qpwave, NOT a fork of qpAdm: NO target, NO ALS, NO rank — it computes the AT2 weighted
// block-jackknife f4 POINT ESTIMATE per quartet + the jackknife-DIAGONAL SE. The GPU path
// is the deliverable: read the f2_blocks dir -> resolve names->indices via pops.txt ->
// build_resources -> upload_f2_blocks_to_device -> run_f4(DeviceF2Blocks, quartets) -> emit
// the table (pop1,pop2,pop3,pop4,est,se,z,p — the golden_fit0_f4_readf2.csv schema).
//
// QUARTETS: EITHER the row-aligned --pop1/--pop2/--pop3/--pop4 columns (admixtools::f4
// comb=FALSE) OR the single-quartet --pops A,B,C,D convenience (4 names = one quartet, or
// any multiple of 4 = several quartets). The f2-dir load, name->index resolution,
// build_resources/upload chain, and the output sink are REUSED verbatim from cmd_qpwave.cpp;
// the result_emit format primitives are reused through emit_f4_result (NO compute/format dup).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is
// reached ONLY through the CUDA-FREE seams (resources.hpp / device_f2_blocks.hpp / f4.hpp).
// main() owns stdout/stderr (architecture.md §10). A DOMAIN outcome is a row + exit 0
// (record-and-continue); only faults return nonzero (cli-bindings.md §1.3, §4.4).
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

#include "app/cmd_emit.hpp"             // emit_to_destination (shared open->write->flush->verify)
#include "app/cmd_fstat_sweep.hpp"      // run_fstat_sweep (the GPU sweep, --all-quartets mode)
#include "app/exit_code_for_caught.hpp" // exit_code_for_caught (5 -> 3 on a real device OOM, B2)
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/f4.hpp"                // steppe::run_f4 + F4Result/options

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

/// Build the quartet NAME table (one row per quartet, four names each) from the frozen
/// config. Prefers the row-aligned --pop1/--pop2/--pop3/--pop4 columns; falls back to the
/// --pops 4-tuple convenience (the names are taken in groups of 4). Returns false (with a
/// reason in `err`) on no input or a malformed shape (mismatched columns / non-multiple-of-4
/// --pops). On success `quartets[k]` = {pop1,pop2,pop3,pop4} of quartet k.
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

    // --pops convenience: names in groups of 4 = quartets (4 names = one quartet).
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
    // ---- SWEEP MODE (--all-quartets): route to the GPU sweep over C(P,4) of the --pops
    // SUBSET (empty ⇒ the whole f2 dir). on-device unrank+compute+|z|filter+CUB-compact,
    // survivors only. SEPARATE from the explicit-list path below (which the goldens gate
    // byte-identical). The sweep body lives in cmd_fstat_sweep.cpp (the SAME run_f4_sweep).
    if (config.sweep_all_combinations()) {
        return run_fstat_sweep(config, /*k=*/4, "f4");
    }

    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) — REUSE cmd_qpwave path -----
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe f4: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe f4: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Build the quartet name table + resolve names -> indices ----------------
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

    // Resolve each (p1,p2,p3,p4) name quad to a P-axis index quad. Carry the resolved
    // names back for the emitter (label_at — canonical pops.txt spelling).
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

    // ---- 3/4. build_resources -> upload f2 to the GPU -> run_f4 (GPU path) ----------
    // The GPU is the deliverable (cli-bindings.md §5.4). All three calls are CUDA-FREE
    // seams; a no-GPU box surfaces a clear fault from build_resources. opts comes from the
    // frozen config; fudge defaults to 0 for a bare f4 SE inside run_f4 (NOT qpadm's 1e-4)
    // unless overridden.
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
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) — a FAULT,
        // nonzero exit (cli-bindings.md §1.3). A domain outcome never throws; it arrives as
        // result.status below (record-and-continue, exit 0).
        std::fprintf(stderr, "steppe f4: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    // ---- 5. Emit (CSV default / TSV / JSON) to --out or stdout — open->write->flush->verify
    // via the shared emit_to_destination (B1): a torn / short write (full disk, closed pipe)
    // returns kExitIoError instead of silently exiting 0 with a truncated file.
    if (const auto rc = emit_to_destination(
            config, "f4", [&](std::ostream& os, OutputFormat fmt) {
                emit_f4_result(os, fmt, result, l1, l2, l3, l4);
            })) {
        return *rc;
    }

    // A DOMAIN outcome (NonSpd over the m-batch) is a table + exit 0 (record-and-continue,
    // cli-bindings.md §1.3); exit_code_for maps those to kExitOk, only faults to nonzero.
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
