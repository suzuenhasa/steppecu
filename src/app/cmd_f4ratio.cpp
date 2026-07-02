// src/app/cmd_f4ratio.cpp
//
// The `steppe f4-ratio` command (standalone f4-ratio statistic; fit-engine §6). f4-ratio is
// the SIBLING of f4/f3, NOT a fork of qpAdm: NO target, NO ALS, NO rank — it computes the AT2
// qpf4ratio admixture proportion alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) per 5-tuple + the
// jackknife-of-the-RATIO SE. The GPU path is the deliverable: read the f2_blocks dir ->
// resolve names->indices via pops.txt -> build_resources -> upload_f2_blocks_to_device ->
// run_f4ratio(DeviceF2Blocks, tuples) -> emit the table (pop1,pop2,pop3,pop4,pop5,alpha,se,z
// — the golden_fit0_f4ratio_readf2.csv schema; NO p column).
//
// TUPLES: EITHER the row-aligned --pop1..--pop5 columns (admixtools::qpf4ratio) OR the
// single-tuple --pops p1,p2,p3,p4,p5 convenience (5 names = one tuple, or any multiple of 5 =
// several tuples). The f2-dir load, name->index resolution, build_resources/upload chain, and
// the output sink are REUSED verbatim from cmd_f3.cpp/cmd_f4.cpp; the result_emit format
// primitives are reused through emit_f4ratio_result (NO compute/format dup).
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is
// reached ONLY through the CUDA-FREE seams (resources.hpp / device_f2_blocks.hpp / f4ratio.hpp).
// main() owns stdout/stderr (architecture.md §10). A DOMAIN outcome is a row + exit 0
// (record-and-continue); only faults return nonzero (cli-bindings.md §1.3, §4.4).
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

#include "app/cmd_emit.hpp"             // emit_to_destination (shared open->write->flush->verify)
#include "app/exit_code_for_caught.hpp" // exit_code_for_caught (5 -> 3 on a real device OOM, B2)
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/f4ratio.hpp"           // steppe::run_f4ratio + F4RatioResult/options

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

/// The f4-ratio tuple arity: alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) is computed per
/// 5-tuple (p1,p2,p3,p4,p5). Single-sources the `5` across the column count, the --pops
/// group size / modulus, the 5*k index stride, the per-tuple loop bound, and the
/// std::array arities below. Value 5, parity-frozen (the AT2 qpf4ratio tuple shape); TU-local
/// because the arity is private domain knowledge of this command (§2.5 / §4 unnamed literal).
constexpr std::size_t kTupleArity = 5;

/// Build the 5-tuple NAME table (one row per tuple, five names each) from the frozen config.
/// Prefers the row-aligned --pop1..--pop5 columns; falls back to the --pops 5-tuple
/// convenience (names in groups of 5). Returns false (with a reason in `err`) on no input or a
/// malformed shape (mismatched columns / non-multiple-of-5 --pops). On success `tuples[k]` =
/// {pop1,pop2,pop3,pop4,pop5} of tuple k.
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

    // --pops convenience: names in groups of 5 = tuples (5 names = one tuple).
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

int run_f4ratio_command(const cfg::RunConfig& config) {
    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) — REUSE cmd_f4/cmd_f3 path -----
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe f4-ratio: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe f4-ratio: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Build the 5-tuple name table + resolve names -> indices ----------------
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

    // Resolve each (p1..p5) name 5-tuple to a P-axis index 5-tuple. Carry the resolved names
    // back for the emitter (label_at — canonical pops.txt spelling).
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

    // ---- 3/4. build_resources -> upload f2 to the GPU -> run_f4ratio (GPU path) -------
    // The GPU is the deliverable (cli-bindings.md §5.4). All three calls are CUDA-FREE
    // seams; a no-GPU box surfaces a clear fault from build_resources. fudge defaults to 0
    // for a bare ratio SE inside run_f4ratio (NOT qpadm's 1e-4) — opts here is the default.
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
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) — a FAULT,
        // nonzero exit (cli-bindings.md §1.3). A domain outcome never throws; it arrives as
        // result.status below (record-and-continue, exit 0).
        std::fprintf(stderr, "steppe f4-ratio: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    // ---- 5. Emit (CSV default / TSV / JSON) to --out or stdout — open->write->flush->verify
    // via the shared emit_to_destination (B1): a torn / short write (full disk, closed pipe)
    // returns kExitIoError instead of silently exiting 0 with a truncated file.
    if (const auto rc = emit_to_destination(
            config, "f4-ratio", [&](std::ostream& os, OutputFormat fmt) {
                emit_f4ratio_result(os, fmt, result, p1_labels, p2_labels, p3_labels, p4_labels,
                                    p5_labels);
            })) {
        return *rc;
    }

    // A DOMAIN outcome is a table + exit 0 (record-and-continue, cli-bindings.md §1.3);
    // exit_code_for maps those to kExitOk, only faults to nonzero.
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
