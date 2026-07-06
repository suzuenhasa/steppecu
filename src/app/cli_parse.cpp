// src/app/cli_parse.cpp
//
// The command-line front end for the steppe binary: it declares every
// subcommand and flag, merges the parsed flags into a validated run
// configuration, and dispatches to the matching GPU run_*_command. A pure
// host translation unit — CLI11 is named only here and no CUDA header is
// included (a build-time grep gate enforces it).
//
// Flags bind directly to their CliArgs field (each field is a std::optional
// or a std::vector, so CLI11 leaves it untouched when the flag is absent —
// the same "was it set?" sentinel the config merge reads). Every subcommand
// is registered through one shared register_cmd() recipe: its flags bind to a
// CliArgs owned at run_cli scope (so the owner outlives the parse), and on
// parse the callback builds the config and records the run_*_command exit code
// into a shared `code` — no std::exit, so destructors run and run_cli returns
// the code.
//
// Reference: docs/reference/src_app_cli_parse.cpp.md
#include "app/cli_parse.hpp"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "app/cmd_cache.hpp"
#include "app/cmd_extract_f2.hpp"
#include "app/cmd_readv2_concord.hpp"
#include "app/cmd_f3.hpp"
#include "app/cmd_f4.hpp"
#include "app/cmd_f4ratio.hpp"
#include "app/cmd_fstat_sweep.hpp"
#include "app/cmd_qpadm.hpp"
#include "app/cmd_qpgraph.hpp"
#include "app/cmd_dates.hpp"
#include "app/cmd_qpdstat.hpp"
#include "app/cmd_qpfstats.hpp"
#include "app/cmd_qpwave.hpp"
#include "app/cmd_rotate.hpp"
#include "app/cmd_scan.hpp"
#include "core/config/cli_args.hpp"
#include "core/config/config_builder.hpp"
#include "core/config/exit_code.hpp"
#include "core/config/run_config.hpp"

namespace steppe::app {

namespace {

using steppe::config::CliArgs;
using steppe::config::Command;
using steppe::config::ConfigBuilder;
using steppe::config::RunConfig;
namespace cfg = steppe::config;

// The version string — reference §4
#ifndef STEPPE_VERSION
#  define STEPPE_VERSION "0.0.0+unknown"
#endif

// The configuration precedence chain — reference §3
[[nodiscard]] std::optional<RunConfig> build_config(const CliArgs& args) {
    ConfigBuilder builder;
    builder.with_defaults();
    if (args.config_path.has_value() && !args.config_path->empty()) {
        builder.merge_file(*args.config_path);
    }
    builder.merge_env();
    builder.merge_cli(args);

    auto result = builder.build();
    if (!result.has_value()) {
        std::fprintf(stderr, "steppe: invalid configuration: %s\n",
                     builder.error_message().c_str());
        return std::nullopt;
    }
    return std::move(result.value());
}

// Shared flag helpers — reference §5
void add_common_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option("--device", a.device,
                    "CUDA device(s): auto | <ordinal> | <ordinal>,<ordinal> (GPU-only; no 'cpu')");
    sub->add_option("--precision", a.precision,
                    "Matmul precision: emu40 | emu32 | fp64 | tf32 (default emu40)");
    sub->add_option("--config", a.config_path,
                    "TOML config file (reserved; not yet supported — passing one currently errors)");
}

void add_output_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option("--out", a.out_file, "Output FILE (stdout if omitted)");
    sub->add_option("--format", a.format, "Output format: csv | tsv | json (default csv)");
}

void add_qpadm_option_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option("--fudge", a.fudge, "AT2 ridge constant (default 1e-4)");
    sub->add_option("--als-iters", a.als_iterations, "ALS iteration count (default 20)");
    sub->add_option("--rank", a.rank, "f4 rank for the fit (-1 = auto nl-1)");
    sub->add_option("--rank-alpha", a.rank_alpha, "Rank-decision significance (default 0.05)");
    sub->add_flag("--allow-neg,!--no-allow-neg", a.allow_negative_weights,
                  "Allow negative weights (default on)");
    sub->add_option("--jackknife", a.jackknife,
                    "SE policy: 0 none | 1 feasible-only | 2 all (rotate only)");
    sub->add_option("--p-se-threshold", a.p_se_threshold,
                    "Feasible-only survivor p-gate (jackknife=1)");
    sub->add_flag("--se-require-p", a.se_require_p,
                  "Feasible-only also requires p >= --p-se-threshold");
}

void add_qpgraph_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option("--graph", a.graph,
                    "The admixture-graph edge-list file (parent child per line)");
    sub->add_option("--numstart", a.numstart,
                    "Multistart restart count (the fleet axis; default 10)");
    sub->add_option("--diag-f3", a.diag_f3,
                    "f3 covariance regularization (AT2 diag_f3; default 1e-5)");
    sub->add_flag("--constrained,!--no-constrained", a.constrained,
                  "Drift edges >= 0 (AT2 default on)");
}

void add_fudge_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option("--fudge", a.fudge, help);
}

// Comma-delimited string-list option -> target vector; the shared shape behind
// every --pop/--pops/--left/--right flag.
void add_str_list_flag(CLI::App* sub, const char* name, std::vector<std::string>& target,
                       const char* help) {
    sub->add_option(name, target, help)->delimiter(',');
}

void add_f2_dir_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option("--f2-dir", a.f2_dir, help);
}

void add_target_flag(CLI::App* sub, CliArgs& a) {
    sub->add_option("--target", a.target, "Target population label");
}

void add_right_flag(CLI::App* sub, CliArgs& a, const char* help) {
    add_str_list_flag(sub, "--right", a.right, help);
}

void add_left_flag(CLI::App* sub, CliArgs& a, const char* help) {
    add_str_list_flag(sub, "--left", a.left, help);
}

void add_pops_flag(CLI::App* sub, CliArgs& a, const char* help) {
    add_str_list_flag(sub, "--pops", a.pops, help);
}

// Population inputs for the standalone f-statistics — reference §6
void add_f4_quartet_flags(CLI::App* sub, CliArgs& a) {
    add_str_list_flag(sub, "--pop1", a.pop1,
                      "Quartet column 1 (p1, the f4 target/first pop), row-aligned with --pop2/3/4");
    add_str_list_flag(sub, "--pop2", a.pop2, "Quartet column 2 (p2)");
    add_str_list_flag(sub, "--pop3", a.pop3, "Quartet column 3 (p3, the f4 R0)");
    add_str_list_flag(sub, "--pop4", a.pop4, "Quartet column 4 (p4, the f4 R1)");
    add_str_list_flag(sub, "--pops", a.pops,
                      "Quartet(s) as names in groups of 4: p1,p2,p3,p4[,p1,p2,p3,p4,...]");
}

void add_f3_triple_flags(CLI::App* sub, CliArgs& a) {
    add_str_list_flag(sub, "--pop1", a.pop1,
                      "Triple column 1 (C, the f3 apex/outgroup/target), row-aligned with --pop2/3");
    add_str_list_flag(sub, "--pop2", a.pop2, "Triple column 2 (A, the f3 first arg)");
    add_str_list_flag(sub, "--pop3", a.pop3, "Triple column 3 (B, the f3 second arg)");
    add_str_list_flag(sub, "--pops", a.pops,
                      "Triple(s) as names in groups of 3: C,A,B[,C,A,B,...]");
}

void add_f4ratio_flags(CLI::App* sub, CliArgs& a) {
    add_str_list_flag(sub, "--pop1", a.pop1, "5-tuple column 1 (p1), row-aligned with --pop2/3/4/5");
    add_str_list_flag(sub, "--pop2", a.pop2, "5-tuple column 2 (p2)");
    add_str_list_flag(sub, "--pop3", a.pop3, "5-tuple column 3 (p3, the numerator 3rd slot)");
    add_str_list_flag(sub, "--pop4", a.pop4, "5-tuple column 4 (p4, the shared 4th slot)");
    add_str_list_flag(sub, "--pop5", a.pop5, "5-tuple column 5 (p5, the denominator 3rd slot)");
    add_str_list_flag(sub, "--pops", a.pops,
                      "Tuple(s) as names in groups of 5: p1,p2,p3,p4,p5[,...]");
}

// The f-statistic sweep flags — reference §7
void add_sweep_filter_flags(CLI::App* sub, CliArgs& a, const char* min_z_help,
                            const char* top_k_help, const char* sure_help) {
    sub->add_option("--min-z", a.sweep_min_z, min_z_help);
    sub->add_option("--top-k", a.sweep_top_k, top_k_help);
    sub->add_flag("--sure", a.sweep_sure, sure_help);
}

void add_sweep_mode_flags(CLI::App* sub, CliArgs& a, const char* enable_flag,
                          const char* enable_help) {
    sub->add_flag(enable_flag, a.sweep_all_combinations, enable_help);
    add_sweep_filter_flags(
        sub, a,
        "Sweep: keep items with |z| >= this (the on-device filter; default 3.0). Excludes --top-k.",
        "Sweep: keep the K items with the largest |z| (bounded device-side reservoir, ~K resident). Excludes --min-z.",
        "Sweep: lift the maxcomb cap (a sweep over more than the cap refuses without this).");
    sub->add_option("--shard-dir", a.shard_dir,
                    "Sweep: write the survivor table to a CSV under this dir (created if absent; vs stdout/--out).");
}

void add_sweep_flags(CLI::App* sub, CliArgs& a) {
    add_pops_flag(sub, a,
                  "Population SUBSET to sweep all combinations of (names; empty ⇒ the whole f2 dir)");
    add_sweep_filter_flags(
        sub, a,
        "Keep items with |z| >= this (the on-device filter; default 3.0). Excludes --top-k.",
        "Keep the K items with the largest |z| (bounded device-side reservoir, ~K resident). Excludes --min-z.",
        "Lift the maxcomb cap (a sweep over more than the cap refuses without this).");
}

// The shared subcommand recipe — reference §8
//
// `args` MUST be owned by the caller (run_cli) so it outlives CLI11's parse:
// setup() binds this sub-App's options to &args members, and the callback
// captures &args; both are read/written during CLI11_PARSE, after this
// function has returned. On parse the callback records the run_*_command exit
// code into `code` (no std::exit).
using Runner = int (*)(const RunConfig&);

template <typename SetupFn>
void register_cmd(CLI::App& app, const char* name, const char* desc, CliArgs& args,
                  Command cmd, SetupFn&& setup, Runner run, int& code) {
    CLI::App* sub = app.add_subcommand(name, desc);
    args.command = cmd;
    setup(sub, args);
    sub->callback([&args, &code, run]() {
        auto config = build_config(args);
        if (!config) {
            code = cfg::kExitInvalidConfig;
            return;
        }
        code = run(*config);
    });
}

}  // namespace

// How a subcommand runs (dispatch and exit codes) — reference §2
int run_cli(int argc, char** argv) {
    CLI::App app{"steppe — GPU/CUDA qpAdm: f-statistics & model fitting", "steppe"};
    app.set_version_flag("--version", std::string{STEPPE_VERSION});
    app.require_subcommand(0, 1);

    // Each CliArgs is owned here, at run_cli scope, so it lives through
    // CLI11_PARSE below (register_cmd binds each sub-App's options to it).
    CliArgs qpadm_args;
    CliArgs qpgraph_args;
    CliArgs qpgraphsearch_args;
    CliArgs qpwave_args;
    CliArgs rotate_args;
    CliArgs scan_args;
    CliArgs extract_args;
    CliArgs f4_args;
    CliArgs f3_args;
    CliArgs f4ratio_args;
    CliArgs qpdstat_args;
    CliArgs f4sweep_args;
    CliArgs f3sweep_args;
    CliArgs qpfstats_args;
    CliArgs dates_args;

    // The `cache` subcommand is host-only (no GPU, no RunConfig): its args bind
    // to these plain locals, also owned at run_cli scope so they outlive the parse.
    std::string cache_ls_root;
    std::string cache_show_dir;
    std::string cache_verify_dir;
    bool cache_check_sources = false;

    // The host-only READv2 concordance validator args, also owned at run_cli scope.
    Readv2ConcordArgs concord_args;

    int code = cfg::kExitOk;

    // The subcommand catalog — reference §8
    register_cmd(app, "qpadm", "qpAdm fit over an f2_blocks dir", qpadm_args, Command::QpAdm,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory (f2.bin + pops.txt + meta.json)");
            add_target_flag(s, a);
            add_left_flag(s, a, "Left source population labels (comma- or space-separated)");
            add_right_flag(s, a, "Right outgroup labels; right[0] = R0");
            add_qpadm_option_flags(s, a);
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_qpadm_command, code);

    register_cmd(app, "qpgraph", "Single-graph qpGraph fit (--f2-dir + --graph edge-list)",
        qpgraph_args, Command::QpGraph,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory (f2.bin + pops.txt)");
            add_qpgraph_flags(s, a);
            add_fudge_flag(s, a, "AT2 cc edge-solve ridge (diag; default 1e-4)");
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_qpgraph_command, code);

    register_cmd(app, "qpgraph-search",
        "qpGraph topology SEARCH (--f2-dir + --pops bounded leaf set; exhaustive enumeration "
        "+ heterogeneous fleet, deterministic global-best)",
        qpgraphsearch_args, Command::QpGraphSearch,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory (f2.bin + pops.txt)");
            add_pops_flag(s, a,
                          "The bounded leaf pop-set the search enumerates topologies over (>= 3 names)");
            s->add_option("--max-nadmix", a.max_nadmix,
                          "Bounded admixture-node ceiling (v1: 0 or 1; default 1)");
            s->add_option("--numstart", a.numstart,
                          "Per-candidate multistart restart count (the fleet axis; default 10)");
            s->add_option("--diag-f3", a.diag_f3,
                          "f3 covariance regularization (AT2 diag_f3; default 1e-5)");
            add_fudge_flag(s, a, "AT2 cc edge-solve ridge (diag; default 1e-4)");
            s->add_flag("--constrained,!--no-constrained", a.constrained,
                        "Drift edges >= 0 (AT2 default on)");
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_qpgraph_search_command, code);

    register_cmd(app, "dates",
        "Admixture dating: --prefix genotypes + --target + --left{2 sources} -> date + SE",
        dates_args, Command::Dates,
        [](CLI::App* s, CliArgs& a) {
            s->add_option("--prefix", a.qpdstat_prefix,
                          "Genotype triple prefix (reads PREFIX.{geno,snp,ind}; .snp needs a real cM map)");
            add_target_flag(s, a);
            add_left_flag(s, a,
                          "The TWO reference source populations (the ancestral pops; exactly two)");
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_dates_command, code);

    register_cmd(app, "qpwave", "qpWave rank sweep (no target; left[0]=ref)", qpwave_args,
        Command::QpWave,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_left_flag(s, a, "Left population set; left[0] is the reference");
            add_right_flag(s, a, "Right outgroup labels");
            add_qpadm_option_flags(s, a);
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_qpwave_command, code);

    register_cmd(app, "f4", "Standalone f4(p1,p2;p3,p4) statistic (est/se/z/p per quartet)",
        f4_args, Command::F4,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_f4_quartet_flags(s, a);
            add_sweep_mode_flags(s, a, "--all-quartets",
                                 "Sweep ALL quartets C(P,4) over the --pops subset (empty ⇒ whole f2 dir)");
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_f4_command, code);

    register_cmd(app, "qpdstat",
        "D-statistic: --f2-dir reports f4 (the AT2 f2-path convention) OR --prefix "
        "PREFIX.{geno,snp,ind} reports the genotype-path normalized-D magnitude",
        qpdstat_args, Command::Qpdstat,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_f4_quartet_flags(s, a);
            add_sweep_mode_flags(s, a, "--all-quartets",
                                 "Sweep ALL quadruples C(P,4) over the --pops subset (empty ⇒ whole f2 dir)");
            s->add_option("--prefix", a.qpdstat_prefix,
                          "Genotype triple prefix PREFIX.{geno,snp,ind} for the normalized-D magnitude "
                          "(Part B; allsnps=TRUE block-jackknife). Without it, --f2-dir reports f4 "
                          "(the AT2 f2-path convention).");
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_qpdstat_command, code);

    register_cmd(app, "f3", "Standalone f3(C;A,B) statistic (est/se/z/p per triple)", f3_args,
        Command::F3,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_f3_triple_flags(s, a);
            add_sweep_mode_flags(s, a, "--all-triples",
                                 "Sweep ALL triples C(P,3) over the --pops subset (empty ⇒ whole f2 dir)");
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_f3_command, code);

    register_cmd(app, "f4-ratio",
        "Standalone f4-ratio alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) (alpha/se/z per 5-tuple)",
        f4ratio_args, Command::F4Ratio,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_f4ratio_flags(s, a);
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_f4ratio_command, code);

    register_cmd(app, "f4-sweep",
        "GPU-only f4 sweep: every C(P,4) quartet, on-device |z|/top-k filter, survivors only",
        f4sweep_args, Command::F4Sweep,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_sweep_flags(s, a);
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_f4_sweep_command, code);

    register_cmd(app, "f3-sweep",
        "GPU-only f3 sweep: every C(P,3) triple, on-device |z|/top-k filter, survivors only",
        f3sweep_args, Command::F3Sweep,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_sweep_flags(s, a);
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_f3_sweep_command, code);

    register_cmd(app, "qpfstats",
        "Genotype-path joint f2 smoother: --prefix genotypes + --pops -> a smoothed f2 dir",
        qpfstats_args, Command::Qpfstats,
        [](CLI::App* s, CliArgs& a) {
            s->add_option("--prefix", a.qpdstat_prefix,
                          "Genotype triple prefix (reads PREFIX.{geno,snp,ind})");
            add_pops_flag(s, a,
                          "Population set to smooth over (sorted ASC internally = the AT2 dimnames order)");
            s->add_option("--out-dir", a.out_dir,
                          "Output smoothed f2_blocks dir (f2.bin + pops.txt + meta.json)");
            s->add_option("--blgsize", a.blgsize,
                          "Jackknife block size in MORGANS (AT2 convention; default 0.05 = 5 cM)");
            add_common_flags(s, a);
        },
        run_qpfstats_command, code);

    register_cmd(app, "qpadm-rotate", "qpAdm rotation over a source pool", rotate_args,
        Command::QpAdmRotate,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_target_flag(s, a);
            s->add_option("--pool", a.pool, "Source pool to enumerate subsets of")->delimiter(',');
            add_right_flag(s, a, "Right outgroup labels");
            s->add_option("--min-sources", a.min_sources, "Minimum sources per model (default 1)");
            s->add_option("--max-sources", a.max_sources, "Maximum sources per model (-1 = whole pool)");
            add_qpadm_option_flags(s, a);
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_qpadm_rotate_command, code);

    register_cmd(app, "scan", "Proxy/model scanner: gated, best-first ranked qpAdm search",
        scan_args, Command::Scan,
        [](CLI::App* s, CliArgs& a) {
            add_f2_dir_flag(s, a, "The f2_blocks directory");
            add_target_flag(s, a);
            s->add_option("--pool", a.pool, "Source pool to enumerate subsets of")->delimiter(',');
            add_right_flag(s, a, "Right outgroup labels");
            s->add_option("--min-sources", a.min_sources, "Minimum sources per model (default 1)");
            s->add_option("--max-sources", a.max_sources, "Maximum sources per model (-1 = whole pool)");
            s->add_option("--p-min", a.scan_p_min, "Objective hard-gate tail-p cutoff alpha (default 0.05)");
            s->add_flag("--allow-clade,!--no-allow-clade", a.scan_allow_clade,
                        "May a 1-source (clade) model be the winner? (default on; "
                        "--no-allow-clade prefers genuine >=2-source mixtures)");
            s->add_option("--strategy", a.scan_strategy,
                          "Search: greedy | beam | exhaustive (default beam; small pools auto-exhaustive)");
            s->add_option("--beam-width", a.scan_beam_width, "Beam width for --strategy beam (default 3)");
            s->add_option("--base", a.scan_base,
                          "Optional seed model (sources) to grow the guided search from")->delimiter(',');
            s->add_flag("--sure", a.sweep_sure,
                        "Proceed with a huge explicit --strategy exhaustive enumeration (lifts the safety cap)");
            s->add_flag("--prerank", a.scan_prerank,
                        "Rank the pool by mean outgroup-f3 relatedness to the target (over the right set), then exit");
            s->add_flag("--suggest-swaps", a.scan_suggest_swaps,
                        "For models that fail the gate, suggest dropping the least-related source and adding a related one");
            s->add_option("--right-search", a.scan_right_search,
                          "Outgroup admissibility: none | check | add-drop (sources-only qpWave gate; R0 pinned)");
            s->add_option("--right-pool", a.scan_right_pool,
                          "Curated outgroup pool that add-drop may draw from (R0 = --right[0] stays pinned)")->delimiter(',');
            add_qpadm_option_flags(s, a);
            add_output_flags(s, a);
            add_common_flags(s, a);
        },
        run_scan_command, code);

    // extract-f2's genotype filters and defaults — reference §9
    register_cmd(app, "extract-f2", "Precompute the f2_blocks dir from genotypes", extract_args,
        Command::ExtractF2,
        [](CLI::App* s, CliArgs& a) {
            s->add_option("--prefix", a.prefix,
                          "Genotype triple prefix (sets --geno/--snp/--ind = PREFIX.{geno,snp,ind})");
            s->add_option("--geno", a.geno, "Genotype file (overrides --prefix)");
            s->add_option("--snp", a.snp, "SNP file (overrides --prefix)");
            s->add_option("--ind", a.ind, "Individual file (overrides --prefix)");
            s->add_option("--out-dir,--out", a.out_dir, "Output f2_blocks DIRECTORY (f2.bin + pops.txt + meta.json)");
            s->add_option("--pops", a.pops, "Explicit population list")->delimiter(',');
            s->add_option("--auto-top-k", a.auto_top_k, "Keep the K largest pops");
            s->add_option("--min-n", a.min_n, "Keep pops with >= N individuals");
            s->add_option("--blgsize", a.blgsize, "Jackknife block size in MORGANS (AT2 convention; default 0.05 = 5 cM)");
            s->add_option("--maf", a.maf, "Minimum MAF");
            s->add_option("--geno-max-miss", a.geno_max_missing, "Max per-SNP missing fraction");
            s->add_option("--maxmiss", a.geno_max_missing, "AT2 alias for --geno-max-miss");
            s->add_option("--mind-max-miss", a.mind_max_missing, "Max per-sample missing fraction");
            s->add_flag("--auto-only,!--no-auto-only", a.autosomes_only,
                        "Keep only autosomes chr 1-22 (default on; --no-auto-only to disable)");
            s->add_flag("--drop-mono,!--no-drop-mono", a.drop_monomorphic,
                        "Drop monomorphic SNPs (default on, AT2 poly_only parity; --no-drop-mono to keep)");
            s->add_flag("--transversions", a.transversions_only, "Keep only transversions");
            s->add_option("--strand-mode", a.strand_mode,
                          "Strand-ambiguous (A/T, C/G) SNP policy: drop (default; merge-safe) | keep "
                          "(retain, AT2 default) | flip (not-yet-implemented, == keep)");
            // --ploidy keeps its validating lambda so the friendly error message is preserved.
            s->add_option_function<std::string>(
                "--ploidy",
                [&a](const std::string& v) {
                    if (v == "auto") a.ploidy = cfg::PloidyMode::Auto;
                    else if (v == "1") a.ploidy = cfg::PloidyMode::PseudoHaploid;
                    else if (v == "2") a.ploidy = cfg::PloidyMode::Diploid;
                    else throw CLI::ValidationError(
                        "--ploidy", "must be auto, 1 (pseudo-haploid), or 2 (diploid); got '" + v + "'");
                },
                "Ploidy policy: auto (AT2 adjust_pseudohaploid, default) | 1 (pseudo-haploid) | 2 (diploid)");
            s->add_option("--tier", a.tier,
                          "f2_blocks output tier: auto | resident | host | disk (default auto; "
                          "host/disk stream the SNP-tile input so high-P runs that OOM resident complete)");
            s->add_flag("--dry-run", a.dry_run, "Report sizes/tier/precision, no compute");
            s->add_flag("--hash,!--no-hash", a.hash_source,
                        "Compute source-dataset provenance SHA-256 (default OFF; overlapped on a background thread)");
            add_common_flags(s, a);
        },
        run_extract_f2_command, code);

    // The host-only cache inspector — no GPU, no build_config; the native mirror
    // of the steppe-cache Python tool. Nested ls/show/verify dispatch straight to
    // run_cache_* (not register_cmd — there is no RunConfig here) — reference §10.
    CLI::App* cache = app.add_subcommand("cache",
        "Inspect f2 caches (host-only: ls | show | verify)");
    cache->require_subcommand(1);
    {
        CLI::App* c_ls = cache->add_subcommand(
            "ls", "Tabulate STPF2BK1 caches under a root (header-only)");
        c_ls->add_option("root", cache_ls_root, "Root directory to scan (default: current dir)");
        c_ls->callback([&]() { code = run_cache_ls(cache_ls_root); });

        CLI::App* c_show = cache->add_subcommand(
            "show", "Header facts + integrity mark + raw meta.json for one cache");
        c_show->add_option("dir", cache_show_dir, "The f2 cache directory")->required();
        c_show->callback([&]() { code = run_cache_show(cache_show_dir); });

        CLI::App* c_verify = cache->add_subcommand(
            "verify", "Re-hash f2.bin/pops.txt against the stored content-address");
        c_verify->add_option("dir", cache_verify_dir, "The f2 cache directory")->required();
        c_verify->add_flag("--check-sources", cache_check_sources,
                           "(reserved; source-file hashing lives in the steppe-cache Python tool)");
        c_verify->callback([&]() { code = run_cache_verify(cache_verify_dir, cache_check_sources); });
    }

    // The host-only READv2 concordance validator — no GPU, no RunConfig. The Phase-0
    // "ruler": diffs two READv2 output tables (steppe's --a vs the reference tool's --b).
    CLI::App* rc = app.add_subcommand(
        "readv2-concord",
        "Validate a READv2 output table against a reference table (host-only; degree confusion + P0_norm concordance)");
    rc->add_option("--a", concord_args.a_path, "steppe's READv2 output table (CSV/TSV, steppe schema)")->required();
    rc->add_option("--b", concord_args.b_path, "reference READv2 output table (CSV/TSV, steppe schema)")->required();
    rc->add_option("--p0-atol", concord_args.p0_atol, "P0_norm abs tolerance (default 5e-3)");
    rc->add_option("--p0-rtol", concord_args.p0_rtol, "P0_norm rel tolerance (default 1e-2)");
    rc->add_option("--degree-agreement-min", concord_args.degree_agreement_min, "degree-match PASS floor (default 0.95)");
    rc->add_option("--p0-within-tol-min", concord_args.p0_within_tol_min, "P0_norm within-tol PASS floor (default 0.90)");
    rc->add_option("--coverage-min", concord_args.coverage_min, "oracle-pair coverage PASS floor (default 1.0)");
    rc->add_option("--format", concord_args.format, "report format: text | json (default text)");
    rc->add_option("--out", concord_args.out_path, "write report here (default stdout)");
    rc->callback([&]() { code = run_readv2_concord(concord_args); });

    CLI11_PARSE(app, argc, argv);

    if (app.get_subcommands().empty()) {
        std::printf("%s", app.help().c_str());
    }
    return code;
}

}  // namespace steppe::app
