// src/app/cmd_qpdstat.cpp
//
// The `steppe qpdstat` command (D-statistic / f4 over an f2_blocks dir; the qpDstat A+B plan,
// Part A). A THIN wrapper over the EXISTING f4 path (cmd_f4.cpp): the entry body is the
// cmd_f4 body end-to-end (read_f2_dir -> build_quartet_names -> PopResolver -> build_resources
// -> upload_f2_blocks_to_device -> run_f4 -> emit_f4_result), differing ONLY in (a) the
// program-name strings ("steppe qpdstat:"), (b) a --prefix guard (Part B fail-fast), and
// (c) the not-yet-impl message wording.
//
// VERIFIED FACT: admixtools::qpdstat on the f2_data path == f4 (f4mode is a no-op without
// per-SNP genotypes; qpdstat(f2dir,f4mode=TRUE) byte-identical to f4mode=FALSE and to f4). So
// the --f2-dir path is FULL AT2-f2-path parity with ZERO new compute and ZERO new emitter:
// est/se/z/p are f4's, where z = est/se and p = 2*(1-Phi(|z|)) ARE the AT2 D-stat
// sign/Z/p convention. The normalized-D MAGNITUDE (per-SNP genotypes) is Part B (--prefix).
//
// QUARTETS: EITHER the row-aligned --pop1/--pop2/--pop3/--pop4 columns OR the single-/multi-
// quartet --pops convenience (names in groups of 4) = the QUADRUPLE input.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the GPU is reached
// ONLY through the CUDA-FREE seams (resources.hpp / device_f2_blocks.hpp / f4.hpp). main()
// owns stdout/stderr (architecture.md §10). A DOMAIN outcome is a row + exit 0 (record-and-
// continue); only faults return nonzero (cli-bindings.md §1.3, §4.4).
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

#include "app/cmd_fstat_sweep.hpp"      // run_fstat_sweep (the GPU sweep, --all-quartets mode)
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "steppe/config.hpp"            // kCentimorgansPerMorgan (blgsize cM -> Morgans)
#include "device/device_f2_blocks.hpp"  // CUDA-FREE: DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // CUDA-FREE: Resources, build_resources
#include "steppe/dstat.hpp"             // steppe::run_dstat + DstatResult (qpDstat Part B, --prefix)
#include "steppe/error.hpp"             // steppe::Status
#include "steppe/f4.hpp"                // steppe::run_f4 + F4Result/options

#include "io/geno_reader.hpp"           // io::GenoReader (records_present for the --prefix P-axis read)
#include "io/genotype_source.hpp"       // io::resolve_genotype_triple / read_ind_partition (EIGENSTRAT-family vs PLINK)
#include "io/ind_reader.hpp"            // io::read_ind / PopSelection (the --prefix P-axis order)

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

/// Build the quartet NAME table (one row per quartet, four names each) from the frozen
/// config. Prefers the row-aligned --pop1/--pop2/--pop3/--pop4 columns; falls back to the
/// --pops 4-tuple convenience (the names are taken in groups of 4) = the QUADRUPLE input.
/// Returns false (with a reason in `err`) on no input or a malformed shape (mismatched
/// columns / non-multiple-of-4 --pops). On success `quartets[k]` = {pop1,pop2,pop3,pop4} of
/// quartet k. Mirrors cmd_f4.cpp's build_quartet_names (the qpdstat f2-path == f4).
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

/// Resolve each quartet of NAMES to a P-axis index quad against `resolver`, carrying the
/// resolved CANONICAL labels (label_at — the pops.txt spelling) back into l1..l4 for the
/// emitter. Shared by BOTH qpdstat paths (--prefix and --f2-dir): the resolve loop was
/// copy-pasted and is identical. Returns false (with the offending-name reason in `err`) on
/// the first label that does not resolve; on success `quadruples[k]` = the four P-axis
/// indices of quartet k and l1..l4[k] = its canonical labels. l1..l4 are AT2-parity labels
/// (§3.2 — name frozen). The reserve is unconditional here (folds the prefix-path missing
/// reserve into the shared path).
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

/// qpDstat Part B (--prefix): the genotype-path NORMALIZED-D magnitude. Reads the genotype
/// triple PREFIX.{geno,snp,ind} through run_dstat (the extract-f2 decode front-end + the
/// per-SNP D kernel + the num/den block-jackknife), emitting the SAME p1..p4,est,se,z,p table
/// as the f2-path (REUSING emit_f4_result — the D est/se/z/p ARE the AT2 D sign/Z/p
/// convention). The P-axis is read_ind(PopSelection::Explicit{the union of the quadruple
/// pops}), sorted ASC by label — IDENTICAL to run_dstat's internal read, so the resolver
/// indices line up. Forced diploid + allsnps=TRUE + autosomes_only are PINNED inside
/// run_dstat (the AT2 qpdstat_geno parity). blgsize is config.blgsize_cm() -> Morgans.
[[nodiscard]] int run_qpdstat_prefix(const cfg::RunConfig& config) {
    const std::string& prefix = config.qpdstat_prefix();
    // Format-aware --prefix expansion (M-FR PLINK): EIGENSTRAT family -> P.{geno,snp,ind};
    // PLINK -> P.{bed,bim,fam}. The on-disk format is pinned by the GenoReader ctor below.
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);
    const std::string& geno = triple.geno;
    const std::string& snp = triple.snp;
    const std::string& ind = triple.ind;

    // ---- 1. Build the quartet name table (REUSE the existing builder) ----------------
    std::vector<std::array<std::string, 4>> quartet_names;
    std::string qerr;
    if (!build_quartet_names(config, quartet_names, qerr)) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", qerr.c_str());
        return cfg::kExitInvalidConfig;
    }

    // ---- 2. The pop UNION (the AT2 indvec) + resolve names -> indices against it ------
    // run_dstat reads ONLY these populations (read_ind(Explicit{union}), NOT the whole
    // prefix), so a 4-pop D over the giant 27594-ind PA prefix decodes a tiny P. The P axis
    // is that Explicit partition, SORTED ASC by label; the resolver below is built over the
    // SAME read_ind(Explicit{union}) so its indices match run_dstat's decode exactly. The
    // union is the DISTINCT names across every quadruple (any order; read_ind sorts).
    std::vector<std::string> pop_union;
    for (const std::array<std::string, 4>& q : quartet_names) {
        for (const std::string& nm : q) {
            if (std::find(pop_union.begin(), pop_union.end(), nm) == pop_union.end())
                pop_union.push_back(nm);
        }
    }

    std::vector<std::string> pop_labels;  // the SORTED Explicit partition (the P axis order).
    try {
        io::GenoReader reader(geno);
        const io::GenoFormat geno_fmt = reader.header().format;  // .ind vs .fam parser (M-FR PLINK)
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

    // ---- 3/4. build_resources -> run_dstat (the genotype-path D, GPU device-resident) -
    // blgsize is the jackknife block size in MORGANS (AT2 blgsize default 0.05 ⇒ 5 cM).
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
        return cfg::kExitRuntimeError;
    }

    // ---- 5. Emit (CSV/TSV/JSON) — REUSE emit_f4_result via an F4Result shim -----------
    // DstatResult mirrors F4Result (p1..p4,est,se,z,p); shim it across so the emitter +
    // the f2-path output schema are IDENTICAL (the D est/se/z/p ARE the AT2 D convention).
    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe qpdstat: unknown --format '%s' (csv|tsv|json)\n",
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }
    F4Result shim;
    shim.p1 = result.p1; shim.p2 = result.p2; shim.p3 = result.p3; shim.p4 = result.p4;
    shim.est = result.est; shim.se = result.se; shim.z = result.z; shim.p = result.p;
    shim.status = result.status; shim.precision_tag = result.precision_tag;

    if (config.out_file().empty()) {
        emit_f4_result(std::cout, fmt, shim, l1, l2, l3, l4);
    } else {
        std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe qpdstat: cannot open --out file: %s\n",
                         config.out_file().c_str());
            return cfg::kExitIoError;
        }
        emit_f4_result(out, fmt, shim, l1, l2, l3, l4);
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace

int run_qpdstat_command(const cfg::RunConfig& config) {
    // ---- 0. --prefix BRANCH: the genotype-path NORMALIZED-D magnitude (Part B) --------
    // The --f2-dir path reports f4 (the AT2 f2-path convention, proven byte-identical to
    // qpdstat f4mode); --prefix is the genotype-reading normalized-D (D = mean num / mean
    // den over per-SNP allele freqs, block-jackknifed). KEEP --f2-dir Part A unchanged.
    if (!config.qpdstat_prefix().empty()) {
        return run_qpdstat_prefix(config);
    }

    // ---- SWEEP MODE (--all-quartets): the f2-path all-quadruples D scan == the f4 sweep
    // over C(P,4) of the --pops SUBSET (empty ⇒ whole f2 dir). on-device enumerate+compute+
    // |z|filter+compact, survivors only. SEPARATE from the explicit-list f2-path below.
    if (config.sweep_all_combinations()) {
        return run_fstat_sweep(config, /*k=*/4, "qpdstat");
    }

    // ---- 1. Read the f2_blocks dir (f2.bin + pops.txt) — REUSE the f4/qpwave path -----
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe qpdstat: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }

    // ---- 2. Build the quartet name table + resolve names -> indices ----------------
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

    // Resolve each (p1,p2,p3,p4) name quad to a P-axis index quad. Carry the resolved
    // names back for the emitter (label_at — canonical pops.txt spelling).
    std::vector<std::array<int, 4>> quartets;
    std::vector<std::string> l1, l2, l3, l4;
    std::string rerr;
    if (!resolve_quartets(resolver, quartet_names, quartets, l1, l2, l3, l4, rerr)) {
        std::fprintf(stderr, "steppe qpdstat: %s\n", rerr.c_str());
        return cfg::kExitInvalidConfig;
    }

    // ---- 3/4. build_resources -> upload f2 to the GPU -> run_f4 (GPU path) ----------
    // The GPU is the deliverable (cli-bindings.md §5.4). All three calls are CUDA-FREE
    // seams; a no-GPU box surfaces a clear fault from build_resources. fudge defaults to 0
    // for a bare f4 SE inside run_f4 (NOT qpadm's 1e-4) — opts here is the struct default.
    // The qpdstat f2-path == f4: ONE run_f4 over the quadruples (NO new compute).
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
        // build_resources / upload / run faults (no device, OOM, CUDA runtime) — a FAULT,
        // nonzero exit (cli-bindings.md §1.3). A domain outcome never throws; it arrives as
        // result.status below (record-and-continue, exit 0).
        std::fprintf(stderr, "steppe qpdstat: device error: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    // ---- 5. Emit (CSV default / TSV / JSON) to --out or stdout — REUSE emit_f4_result --
    // The D-output convention IS f4's est/se/z/p (the qpdstat f2-path == f4); NO new emitter.
    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe qpdstat: unknown --format '%s' (csv|tsv|json)\n",
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    if (config.out_file().empty()) {
        emit_f4_result(std::cout, fmt, result, l1, l2, l3, l4);
    } else {
        std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe qpdstat: cannot open --out file: %s\n",
                         config.out_file().c_str());
            return cfg::kExitIoError;
        }
        emit_f4_result(out, fmt, result, l1, l2, l3, l4);
    }

    // A DOMAIN outcome (NonSpd over the m-batch) is a table + exit 0 (record-and-continue,
    // cli-bindings.md §1.3); exit_code_for maps those to kExitOk, only faults to nonzero.
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
