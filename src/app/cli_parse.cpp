// src/app/cli_parse.cpp
//
// The command-line front end for the steppe binary: it declares every
// subcommand and flag, merges the parsed flags into a validated run
// configuration, and dispatches to the matching GPU run_*_command. A pure
// host translation unit — CLI11 is named only here and no CUDA header is
// included (a build-time grep gate enforces it).
//
// Reference: docs/reference/src_app_cli_parse.cpp.md
#include "app/cli_parse.hpp"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "app/cmd_extract_f2.hpp"
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
    sub->add_option_function<std::string>(
        "--device", [&a](const std::string& v) { a.device = v; },
        "CUDA device(s): auto | <ordinal> | <ordinal>,<ordinal> (GPU-only; no 'cpu')");
    sub->add_option_function<std::string>(
        "--precision", [&a](const std::string& v) { a.precision = v; },
        "Matmul precision: emu40 | emu32 | fp64 | tf32 (default emu40)");
    sub->add_option_function<std::string>(
        "--config", [&a](const std::string& v) { a.config_path = v; },
        "TOML config file (reserved; not yet supported — passing one currently errors)");
}

void add_output_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::string>(
        "--out", [&a](const std::string& v) { a.out_file = v; },
        "Output FILE (stdout if omitted)");
    sub->add_option_function<std::string>(
        "--format", [&a](const std::string& v) { a.format = v; },
        "Output format: csv | tsv | json (default csv)");
}

void add_qpadm_option_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<double>("--fudge", [&a](double v) { a.fudge = v; },
                                     "AT2 ridge constant (default 1e-4)");
    sub->add_option_function<int>("--als-iters", [&a](int v) { a.als_iterations = v; },
                                  "ALS iteration count (default 20)");
    sub->add_option_function<int>("--rank", [&a](int v) { a.rank = v; },
                                  "f4 rank for the fit (-1 = auto nl-1)");
    sub->add_option_function<double>("--rank-alpha", [&a](double v) { a.rank_alpha = v; },
                                     "Rank-decision significance (default 0.05)");
    sub->add_flag_function("--allow-neg,!--no-allow-neg",
                           [&a](std::int64_t v) { a.allow_negative_weights = (v >= 0); },
                           "Allow negative weights (default on)");
    sub->add_option_function<int>("--jackknife", [&a](int v) { a.jackknife = v; },
                                  "SE policy: 0 none | 1 feasible-only | 2 all (rotate only)");
    sub->add_option_function<double>("--p-se-threshold", [&a](double v) { a.p_se_threshold = v; },
                                     "Feasible-only survivor p-gate (jackknife=1)");
    sub->add_flag_function("--se-require-p",
                           [&a](std::int64_t) { a.se_require_p = true; },
                           "Feasible-only also requires p >= --p-se-threshold");
}

void add_qpgraph_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::string>("--graph", [&a](const std::string& v) { a.graph = v; },
                                          "The admixture-graph edge-list file (parent child per line)");
    sub->add_option_function<int>("--numstart", [&a](int v) { a.numstart = v; },
                                  "Multistart restart count (the fleet axis; default 10)");
    sub->add_option_function<double>("--diag-f3", [&a](double v) { a.diag_f3 = v; },
                                     "f3 covariance regularization (AT2 diag_f3; default 1e-5)");
    sub->add_flag_function("--constrained,!--no-constrained",
                           [&a](std::int64_t v) { a.constrained = (v >= 0); },
                           "Drift edges >= 0 (AT2 default on)");
}

void add_fudge_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<double>("--fudge", [&a](double v) { a.fudge = v; }, help);
}

void add_f2_dir_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<std::string>(
        "--f2-dir", [&a](const std::string& v) { a.f2_dir = v; }, help);
}

void add_target_flag(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::string>(
        "--target", [&a](const std::string& v) { a.target = v; }, "Target population label");
}

void add_right_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<std::vector<std::string>>(
            "--right", [&a](const std::vector<std::string>& v) { a.right = v; }, help)
        ->delimiter(',');
}

void add_left_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<std::vector<std::string>>(
            "--left", [&a](const std::vector<std::string>& v) { a.left = v; }, help)
        ->delimiter(',');
}

void add_pops_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<std::vector<std::string>>(
            "--pops", [&a](const std::vector<std::string>& v) { a.pops = v; }, help)
        ->delimiter(',');
}

// Population inputs for the standalone f-statistics — reference §6
void add_f4_quartet_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::vector<std::string>>(
            "--pop1", [&a](const std::vector<std::string>& v) { a.pop1 = v; },
            "Quartet column 1 (p1, the f4 target/first pop), row-aligned with --pop2/3/4")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop2", [&a](const std::vector<std::string>& v) { a.pop2 = v; },
            "Quartet column 2 (p2)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop3", [&a](const std::vector<std::string>& v) { a.pop3 = v; },
            "Quartet column 3 (p3, the f4 R0)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop4", [&a](const std::vector<std::string>& v) { a.pop4 = v; },
            "Quartet column 4 (p4, the f4 R1)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pops", [&a](const std::vector<std::string>& v) { a.pops = v; },
            "Quartet(s) as names in groups of 4: p1,p2,p3,p4[,p1,p2,p3,p4,...]")
        ->delimiter(',');
}

void add_f3_triple_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::vector<std::string>>(
            "--pop1", [&a](const std::vector<std::string>& v) { a.pop1 = v; },
            "Triple column 1 (C, the f3 apex/outgroup/target), row-aligned with --pop2/3")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop2", [&a](const std::vector<std::string>& v) { a.pop2 = v; },
            "Triple column 2 (A, the f3 first arg)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop3", [&a](const std::vector<std::string>& v) { a.pop3 = v; },
            "Triple column 3 (B, the f3 second arg)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pops", [&a](const std::vector<std::string>& v) { a.pops = v; },
            "Triple(s) as names in groups of 3: C,A,B[,C,A,B,...]")
        ->delimiter(',');
}

void add_f4ratio_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::vector<std::string>>(
            "--pop1", [&a](const std::vector<std::string>& v) { a.pop1 = v; },
            "5-tuple column 1 (p1), row-aligned with --pop2/3/4/5")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop2", [&a](const std::vector<std::string>& v) { a.pop2 = v; },
            "5-tuple column 2 (p2)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop3", [&a](const std::vector<std::string>& v) { a.pop3 = v; },
            "5-tuple column 3 (p3, the numerator 3rd slot)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop4", [&a](const std::vector<std::string>& v) { a.pop4 = v; },
            "5-tuple column 4 (p4, the shared 4th slot)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pop5", [&a](const std::vector<std::string>& v) { a.pop5 = v; },
            "5-tuple column 5 (p5, the denominator 3rd slot)")
        ->delimiter(',');
    sub->add_option_function<std::vector<std::string>>(
            "--pops", [&a](const std::vector<std::string>& v) { a.pops = v; },
            "Tuple(s) as names in groups of 5: p1,p2,p3,p4,p5[,...]")
        ->delimiter(',');
}

// The f-statistic sweep flags — reference §7
void add_sweep_filter_flags(CLI::App* sub, CliArgs& a, const char* min_z_help,
                            const char* top_k_help, const char* sure_help) {
    sub->add_option_function<double>(
        "--min-z", [&a](double v) { a.sweep_min_z = v; }, min_z_help);
    sub->add_option_function<int>(
        "--top-k", [&a](int v) { a.sweep_top_k = v; }, top_k_help);
    sub->add_flag_function(
        "--sure", [&a](std::int64_t) { a.sweep_sure = true; }, sure_help);
}

void add_sweep_mode_flags(CLI::App* sub, CliArgs& a, const char* enable_flag,
                          const char* enable_help) {
    sub->add_flag_function(enable_flag, [&a](std::int64_t) { a.sweep_all_combinations = true; },
                           enable_help);
    add_sweep_filter_flags(
        sub, a,
        "Sweep: keep items with |z| >= this (the on-device filter; default 3.0). Excludes --top-k.",
        "Sweep: keep the K items with the largest |z| (bounded device-side reservoir, ~K resident). Excludes --min-z.",
        "Sweep: lift the maxcomb cap (a sweep over more than the cap refuses without this).");
    sub->add_option_function<std::string>(
        "--shard-dir", [&a](const std::string& v) { a.shard_dir = v; },
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

}  // namespace

// How a subcommand runs (dispatch and exit codes) — reference §2
int run_cli(int argc, char** argv) {
    CLI::App app{"steppe — GPU/CUDA qpAdm: f-statistics & model fitting", "steppe"};
    app.set_version_flag("--version", std::string{STEPPE_VERSION});
    app.require_subcommand(0, 1);

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

    // The subcommand catalog — reference §8
    {
        CLI::App* sub = app.add_subcommand("qpadm", "qpAdm fit over an f2_blocks dir");
        qpadm_args.command = Command::QpAdm;
        add_f2_dir_flag(sub, qpadm_args,
                        "The f2_blocks directory (f2.bin + pops.txt + meta.json)");
        add_target_flag(sub, qpadm_args);
        add_left_flag(sub, qpadm_args,
                      "Left source population labels (comma- or space-separated)");
        add_right_flag(sub, qpadm_args, "Right outgroup labels; right[0] = R0");
        add_qpadm_option_flags(sub, qpadm_args);
        add_output_flags(sub, qpadm_args);
        add_common_flags(sub, qpadm_args);
        sub->callback([&]() {
            auto config = build_config(qpadm_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_qpadm_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand("qpgraph", "Single-graph qpGraph fit (--f2-dir + --graph edge-list)");
        qpgraph_args.command = Command::QpGraph;
        add_f2_dir_flag(sub, qpgraph_args, "The f2_blocks directory (f2.bin + pops.txt)");
        add_qpgraph_flags(sub, qpgraph_args);
        add_fudge_flag(sub, qpgraph_args, "AT2 cc edge-solve ridge (diag; default 1e-4)");
        add_output_flags(sub, qpgraph_args);
        add_common_flags(sub, qpgraph_args);
        sub->callback([&]() {
            auto config = build_config(qpgraph_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_qpgraph_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "qpgraph-search",
            "qpGraph topology SEARCH (--f2-dir + --pops bounded leaf set; exhaustive enumeration "
            "+ heterogeneous fleet, deterministic global-best)");
        qpgraphsearch_args.command = Command::QpGraphSearch;
        add_f2_dir_flag(sub, qpgraphsearch_args, "The f2_blocks directory (f2.bin + pops.txt)");
        add_pops_flag(sub, qpgraphsearch_args,
                      "The bounded leaf pop-set the search enumerates topologies over (>= 3 names)");
        sub->add_option_function<int>("--max-nadmix",
                                      [&qpgraphsearch_args](int v) { qpgraphsearch_args.max_nadmix = v; },
                                      "Bounded admixture-node ceiling (v1: 0 or 1; default 1)");
        sub->add_option_function<int>("--numstart", [&qpgraphsearch_args](int v) { qpgraphsearch_args.numstart = v; },
                                      "Per-candidate multistart restart count (the fleet axis; default 10)");
        sub->add_option_function<double>("--diag-f3", [&qpgraphsearch_args](double v) { qpgraphsearch_args.diag_f3 = v; },
                                         "f3 covariance regularization (AT2 diag_f3; default 1e-5)");
        add_fudge_flag(sub, qpgraphsearch_args, "AT2 cc edge-solve ridge (diag; default 1e-4)");
        sub->add_flag_function("--constrained,!--no-constrained",
                               [&qpgraphsearch_args](std::int64_t v) { qpgraphsearch_args.constrained = (v >= 0); },
                               "Drift edges >= 0 (AT2 default on)");
        add_output_flags(sub, qpgraphsearch_args);
        add_common_flags(sub, qpgraphsearch_args);
        sub->callback([&]() {
            auto config = build_config(qpgraphsearch_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_qpgraph_search_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "dates",
            "Admixture dating: --prefix genotypes + --target + --left{2 sources} -> date + SE");
        dates_args.command = Command::Dates;
        sub->add_option_function<std::string>(
            "--prefix", [&](const std::string& v) { dates_args.qpdstat_prefix = v; },
            "Genotype triple prefix (reads PREFIX.{geno,snp,ind}; .snp needs a real cM map)");
        add_target_flag(sub, dates_args);
        add_left_flag(sub, dates_args,
                      "The TWO reference source populations (the ancestral pops; exactly two)");
        add_output_flags(sub, dates_args);
        add_common_flags(sub, dates_args);
        sub->callback([&]() {
            auto config = build_config(dates_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_dates_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand("qpwave", "qpWave rank sweep (no target; left[0]=ref)");
        qpwave_args.command = Command::QpWave;
        add_f2_dir_flag(sub, qpwave_args, "The f2_blocks directory");
        add_left_flag(sub, qpwave_args, "Left population set; left[0] is the reference");
        add_right_flag(sub, qpwave_args, "Right outgroup labels");
        add_qpadm_option_flags(sub, qpwave_args);
        add_output_flags(sub, qpwave_args);
        add_common_flags(sub, qpwave_args);
        sub->callback([&]() {
            auto config = build_config(qpwave_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_qpwave_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "f4", "Standalone f4(p1,p2;p3,p4) statistic (est/se/z/p per quartet)");
        f4_args.command = Command::F4;
        add_f2_dir_flag(sub, f4_args, "The f2_blocks directory");
        add_f4_quartet_flags(sub, f4_args);
        add_sweep_mode_flags(sub, f4_args, "--all-quartets",
                             "Sweep ALL quartets C(P,4) over the --pops subset (empty ⇒ whole f2 dir)");
        add_output_flags(sub, f4_args);
        add_common_flags(sub, f4_args);
        sub->callback([&]() {
            auto config = build_config(f4_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_f4_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "qpdstat",
            "D-statistic: --f2-dir reports f4 (the AT2 f2-path convention) OR --prefix "
            "PREFIX.{geno,snp,ind} reports the genotype-path normalized-D magnitude");
        qpdstat_args.command = Command::Qpdstat;
        add_f2_dir_flag(sub, qpdstat_args, "The f2_blocks directory");
        add_f4_quartet_flags(sub, qpdstat_args);
        add_sweep_mode_flags(sub, qpdstat_args, "--all-quartets",
                             "Sweep ALL quadruples C(P,4) over the --pops subset (empty ⇒ whole f2 dir)");
        sub->add_option_function<std::string>(
            "--prefix", [&](const std::string& v) { qpdstat_args.qpdstat_prefix = v; },
            "Genotype triple prefix PREFIX.{geno,snp,ind} for the normalized-D magnitude "
            "(Part B; allsnps=TRUE block-jackknife). Without it, --f2-dir reports f4 "
            "(the AT2 f2-path convention).");
        add_output_flags(sub, qpdstat_args);
        add_common_flags(sub, qpdstat_args);
        sub->callback([&]() {
            auto config = build_config(qpdstat_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_qpdstat_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "f3", "Standalone f3(C;A,B) statistic (est/se/z/p per triple)");
        f3_args.command = Command::F3;
        add_f2_dir_flag(sub, f3_args, "The f2_blocks directory");
        add_f3_triple_flags(sub, f3_args);
        add_sweep_mode_flags(sub, f3_args, "--all-triples",
                             "Sweep ALL triples C(P,3) over the --pops subset (empty ⇒ whole f2 dir)");
        add_output_flags(sub, f3_args);
        add_common_flags(sub, f3_args);
        sub->callback([&]() {
            auto config = build_config(f3_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_f3_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "f4-ratio",
            "Standalone f4-ratio alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) (alpha/se/z per 5-tuple)");
        f4ratio_args.command = Command::F4Ratio;
        add_f2_dir_flag(sub, f4ratio_args, "The f2_blocks directory");
        add_f4ratio_flags(sub, f4ratio_args);
        add_output_flags(sub, f4ratio_args);
        add_common_flags(sub, f4ratio_args);
        sub->callback([&]() {
            auto config = build_config(f4ratio_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_f4ratio_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "f4-sweep",
            "GPU-only f4 sweep: every C(P,4) quartet, on-device |z|/top-k filter, survivors only");
        f4sweep_args.command = Command::F4Sweep;
        add_f2_dir_flag(sub, f4sweep_args, "The f2_blocks directory");
        add_sweep_flags(sub, f4sweep_args);
        add_output_flags(sub, f4sweep_args);
        add_common_flags(sub, f4sweep_args);
        sub->callback([&]() {
            auto config = build_config(f4sweep_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_f4_sweep_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "f3-sweep",
            "GPU-only f3 sweep: every C(P,3) triple, on-device |z|/top-k filter, survivors only");
        f3sweep_args.command = Command::F3Sweep;
        add_f2_dir_flag(sub, f3sweep_args, "The f2_blocks directory");
        add_sweep_flags(sub, f3sweep_args);
        add_output_flags(sub, f3sweep_args);
        add_common_flags(sub, f3sweep_args);
        sub->callback([&]() {
            auto config = build_config(f3sweep_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_f3_sweep_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "qpfstats",
            "Genotype-path joint f2 smoother: --prefix genotypes + --pops -> a smoothed f2 dir");
        qpfstats_args.command = Command::Qpfstats;
        sub->add_option_function<std::string>(
            "--prefix", [&](const std::string& v) { qpfstats_args.qpdstat_prefix = v; },
            "Genotype triple prefix (reads PREFIX.{geno,snp,ind})");
        add_pops_flag(sub, qpfstats_args,
                      "Population set to smooth over (sorted ASC internally = the AT2 dimnames order)");
        sub->add_option_function<std::string>(
            "--out-dir", [&](const std::string& v) { qpfstats_args.out_dir = v; },
            "Output smoothed f2_blocks dir (f2.bin + pops.txt + meta.json)");
        sub->add_option_function<double>(
            "--blgsize", [&](double v) { qpfstats_args.blgsize = v; },
            "Jackknife block size in MORGANS (AT2 convention; default 0.05 = 5 cM)");
        add_common_flags(sub, qpfstats_args);
        sub->callback([&]() {
            auto config = build_config(qpfstats_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_qpfstats_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand("qpadm-rotate", "qpAdm rotation over a source pool");
        rotate_args.command = Command::QpAdmRotate;
        add_f2_dir_flag(sub, rotate_args, "The f2_blocks directory");
        add_target_flag(sub, rotate_args);
        sub->add_option_function<std::vector<std::string>>(
            "--pool", [&](const std::vector<std::string>& v) { rotate_args.pool = v; },
            "Source pool to enumerate subsets of")->delimiter(',');
        add_right_flag(sub, rotate_args, "Right outgroup labels");
        sub->add_option_function<int>("--min-sources", [&](int v) { rotate_args.min_sources = v; },
                                      "Minimum sources per model (default 1)");
        sub->add_option_function<int>("--max-sources", [&](int v) { rotate_args.max_sources = v; },
                                      "Maximum sources per model (-1 = whole pool)");
        add_qpadm_option_flags(sub, rotate_args);
        add_output_flags(sub, rotate_args);
        add_common_flags(sub, rotate_args);
        sub->callback([&]() {
            auto config = build_config(rotate_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_qpadm_rotate_command(*config));
        });
    }

    {
        CLI::App* sub = app.add_subcommand(
            "scan", "Proxy/model scanner: gated, best-first ranked qpAdm search");
        scan_args.command = Command::Scan;
        add_f2_dir_flag(sub, scan_args, "The f2_blocks directory");
        add_target_flag(sub, scan_args);
        sub->add_option_function<std::vector<std::string>>(
            "--pool", [&](const std::vector<std::string>& v) { scan_args.pool = v; },
            "Source pool to enumerate subsets of")->delimiter(',');
        add_right_flag(sub, scan_args, "Right outgroup labels");
        sub->add_option_function<int>("--min-sources", [&](int v) { scan_args.min_sources = v; },
                                      "Minimum sources per model (default 1)");
        sub->add_option_function<int>("--max-sources", [&](int v) { scan_args.max_sources = v; },
                                      "Maximum sources per model (-1 = whole pool)");
        sub->add_option_function<double>("--p-min", [&](double v) { scan_args.scan_p_min = v; },
                                         "Objective hard-gate tail-p cutoff alpha (default 0.05)");
        sub->add_flag_function("--allow-clade,!--no-allow-clade",
                               [&](std::int64_t v) { scan_args.scan_allow_clade = (v >= 0); },
                               "May a 1-source (clade) model be the winner? (default on; "
                               "--no-allow-clade prefers genuine >=2-source mixtures)");
        sub->add_option_function<std::string>(
            "--strategy", [&](const std::string& v) { scan_args.scan_strategy = v; },
            "Search: greedy | beam | exhaustive (default beam; small pools auto-exhaustive)");
        sub->add_option_function<int>("--beam-width", [&](int v) { scan_args.scan_beam_width = v; },
                                      "Beam width for --strategy beam (default 3)");
        sub->add_option_function<std::vector<std::string>>(
            "--base", [&](const std::vector<std::string>& v) { scan_args.scan_base = v; },
            "Optional seed model (sources) to grow the guided search from")->delimiter(',');
        sub->add_flag_function(
            "--sure", [&](std::int64_t) { scan_args.sweep_sure = true; },
            "Proceed with a huge explicit --strategy exhaustive enumeration (lifts the safety cap)");
        sub->add_flag_function(
            "--prerank", [&](std::int64_t) { scan_args.scan_prerank = true; },
            "Rank the pool by mean outgroup-f3 relatedness to the target (over the right set), then exit");
        sub->add_flag_function(
            "--suggest-swaps", [&](std::int64_t) { scan_args.scan_suggest_swaps = true; },
            "For models that fail the gate, suggest dropping the least-related source and adding a related one");
        sub->add_option_function<std::string>(
            "--right-search", [&](const std::string& v) { scan_args.scan_right_search = v; },
            "Outgroup admissibility: none | check | add-drop (sources-only qpWave gate; R0 pinned)");
        sub->add_option_function<std::vector<std::string>>(
            "--right-pool", [&](const std::vector<std::string>& v) { scan_args.scan_right_pool = v; },
            "Curated outgroup pool that add-drop may draw from (R0 = --right[0] stays pinned)")->delimiter(',');
        add_qpadm_option_flags(sub, scan_args);
        add_output_flags(sub, scan_args);
        add_common_flags(sub, scan_args);
        sub->callback([&]() {
            auto config = build_config(scan_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_scan_command(*config));
        });
    }

    // extract-f2's genotype filters and defaults — reference §9
    {
        CLI::App* sub = app.add_subcommand("extract-f2", "Precompute the f2_blocks dir from genotypes");
        extract_args.command = Command::ExtractF2;
        sub->add_option_function<std::string>("--prefix", [&](const std::string& v) { extract_args.prefix = v; },
                                              "Genotype triple prefix (sets --geno/--snp/--ind = PREFIX.{geno,snp,ind})");
        sub->add_option_function<std::string>("--geno", [&](const std::string& v) { extract_args.geno = v; }, "Genotype file (overrides --prefix)");
        sub->add_option_function<std::string>("--snp",  [&](const std::string& v) { extract_args.snp = v; },  "SNP file (overrides --prefix)");
        sub->add_option_function<std::string>("--ind",  [&](const std::string& v) { extract_args.ind = v; },  "Individual file (overrides --prefix)");
        sub->add_option_function<std::string>("--out-dir,--out",  [&](const std::string& v) { extract_args.out_dir = v; }, "Output f2_blocks DIRECTORY (f2.bin + pops.txt + meta.json)");
        sub->add_option_function<std::vector<std::string>>(
            "--pops", [&](const std::vector<std::string>& v) { extract_args.pops = v; },
            "Explicit population list")->delimiter(',');
        sub->add_option_function<int>("--auto-top-k", [&](int v) { extract_args.auto_top_k = v; }, "Keep the K largest pops");
        sub->add_option_function<int>("--min-n", [&](int v) { extract_args.min_n = v; }, "Keep pops with >= N individuals");
        sub->add_option_function<double>("--blgsize", [&](double v) { extract_args.blgsize = v; }, "Jackknife block size in MORGANS (AT2 convention; default 0.05 = 5 cM)");
        sub->add_option_function<double>("--maf", [&](double v) { extract_args.maf = v; }, "Minimum MAF");
        sub->add_option_function<double>("--geno-max-miss", [&](double v) { extract_args.geno_max_missing = v; }, "Max per-SNP missing fraction");
        sub->add_option_function<double>("--maxmiss", [&](double v) { extract_args.geno_max_missing = v; }, "AT2 alias for --geno-max-miss");
        sub->add_option_function<double>("--mind-max-miss", [&](double v) { extract_args.mind_max_missing = v; }, "Max per-sample missing fraction");
        sub->add_flag_function("--auto-only,!--no-auto-only",
                               [&](std::int64_t v) { extract_args.autosomes_only = (v >= 0); },
                               "Keep only autosomes chr 1-22 (default on; --no-auto-only to disable)");
        sub->add_flag_function("--drop-mono,!--no-drop-mono",
                               [&](std::int64_t v) { extract_args.drop_monomorphic = (v >= 0); },
                               "Drop monomorphic SNPs (default on, AT2 poly_only parity; --no-drop-mono to keep)");
        sub->add_flag_function("--transversions", [&](std::int64_t) { extract_args.transversions_only = true; }, "Keep only transversions");
        sub->add_option_function<std::string>(
            "--strand-mode", [&](const std::string& v) { extract_args.strand_mode = v; },
            "Strand-ambiguous (A/T, C/G) SNP policy: drop (default; merge-safe) | keep "
            "(retain, AT2 default) | flip (not-yet-implemented, == keep)");
        sub->add_option_function<std::string>(
            "--ploidy",
            [&](const std::string& v) {
                if (v == "auto") extract_args.ploidy = cfg::PloidyMode::Auto;
                else if (v == "1") extract_args.ploidy = cfg::PloidyMode::PseudoHaploid;
                else if (v == "2") extract_args.ploidy = cfg::PloidyMode::Diploid;
                else throw CLI::ValidationError(
                    "--ploidy", "must be auto, 1 (pseudo-haploid), or 2 (diploid); got '" + v + "'");
            },
            "Ploidy policy: auto (AT2 adjust_pseudohaploid, default) | 1 (pseudo-haploid) | 2 (diploid)");
        sub->add_option_function<std::string>(
            "--tier", [&](const std::string& v) { extract_args.tier = v; },
            "f2_blocks output tier: auto | resident | host | disk (default auto; "
            "host/disk stream the SNP-tile input so high-P runs that OOM resident complete)");
        sub->add_flag_function("--dry-run", [&](std::int64_t) { extract_args.dry_run = true; }, "Report sizes/tier/precision, no compute");
        sub->add_flag_function("--hash,!--no-hash",
                               [&](std::int64_t v) { extract_args.hash_source = (v >= 0); },
                               "Compute source-dataset provenance SHA-256 (default OFF; overlapped on a background thread)");
        add_common_flags(sub, extract_args);
        sub->callback([&]() {
            auto config = build_config(extract_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_extract_f2_command(*config));
        });
    }

    CLI11_PARSE(app, argc, argv);

    std::printf("%s", app.help().c_str());
    return cfg::kExitOk;
}

}  // namespace steppe::app
