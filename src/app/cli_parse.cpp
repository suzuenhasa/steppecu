// src/app/cli_parse.cpp
//
// The CLI11 wiring + subcommand dispatch (M(cli-0) scaffold). Plain CXX, app-only.
// CLI11 is named ONLY here (PRIVATE to the app subtree, §4 layering; cli-bindings.md
// §6.1). NO CUDA header — this is a pure host TU; the arch-grep gate enforces it.
//
// Flow per subcommand (cli-bindings.md §4.5; architecture.md §9):
//   1. CLI11 binds the flags into a steppe::config::CliArgs.
//   2. ConfigBuilder().with_defaults().merge_file(--config).merge_env().merge_cli(args)
//      .build() runs the §9 precedence merge + validation.
//   3. On InvalidConfig, print the builder's reason to stderr, return kExitInvalidConfig.
//   4. Dispatch to the subcommand's run_*_command (the real GPU compute). qpadm, qpwave,
//      qpadm-rotate, and extract-f2 are ALL wired to their GPU compute.
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

// The version string the user sees with `steppe --version`. STEPPE_VERSION is defined
// by the build (the project VERSION); fall back to a literal if the macro is absent so
// the TU compiles standalone.
#ifndef STEPPE_VERSION
#  define STEPPE_VERSION "0.1.0"
#endif

// Build the merged config from the parsed args (the §9 precedence chain). On failure
// it prints the reason to stderr and returns nullopt; the caller maps that to
// kExitInvalidConfig.
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

// Attach the GLOBAL resource/precision/output flags shared by every subcommand
// (cli-bindings.md §4.1). Binds into the per-subcommand CliArgs.
void add_common_flags(CLI::App* sub, CliArgs& a) {
    // --device is a GPU-only ordinal selector ("auto" | "0" | "0,1"); the GPU-only
    // rejection of "cpu" happens in ConfigBuilder::build() (cli-bindings.md §5.4).
    sub->add_option_function<std::string>(
        "--device", [&a](const std::string& v) { a.device = v; },
        "CUDA device(s): auto | <ordinal> | <ordinal>,<ordinal> (GPU-only; no 'cpu')");
    sub->add_option_function<std::string>(
        "--precision", [&a](const std::string& v) { a.precision = v; },
        "Matmul precision: emu40 | emu32 | fp64 | tf32 (default emu40)");
    sub->add_option_function<std::string>(
        "--config", [&a](const std::string& v) { a.config_path = v; },
        "TOML config file (merged below CLI, above env/defaults)");
}

// Attach the output flags (--out / --format) shared by the fit subcommands.
void add_output_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::string>(
        "--out", [&a](const std::string& v) { a.out_file = v; },
        "Output file (stdout if omitted)");
    sub->add_option_function<std::string>(
        "--format", [&a](const std::string& v) { a.format = v; },
        "Output format: csv | tsv | json (default csv)");
}

// Attach the QpAdmOptions overrides shared by qpadm / qpwave / qpadm-rotate. Flag
// names mirror QpAdmOptions so a bare invocation reproduces the goldens (cli-bindings
// §4.1).
void add_qpadm_option_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<double>("--fudge", [&a](double v) { a.fudge = v; },
                                     "AT2 ridge constant (default 1e-4)");
    sub->add_option_function<int>("--als-iters", [&a](int v) { a.als_iterations = v; },
                                  "ALS iteration count (default 20)");
    sub->add_option_function<int>("--rank", [&a](int v) { a.rank = v; },
                                  "f4 rank for the fit (-1 = auto nl-1)");
    sub->add_option_function<double>("--rank-alpha", [&a](double v) { a.rank_alpha = v; },
                                     "Rank-decision significance (default 0.05)");
    // --allow-neg / --no-allow-neg: a paired bool flag (default true; cli-bindings §4.1).
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

// qpgraph-specific flags: --graph (the edge-list file) + --numstart / --diag-f3 /
// --constrained (the AT2 qpgraph defaults). --fudge (the AT2 `diag` cc ridge) is REUSED
// from add_qpadm_option_flags' --fudge (QpAdmOptions::fudge) so the same flag drives both.
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

// Bind --f2-dir (the f2_blocks directory) into `a.f2_dir`. Shared by qpadm / qpwave /
// qpadm-rotate. The help string is PARAMETRIZED because qpadm's help is more verbose
// ("...f2.bin + pops.txt + meta.json") than qpwave/rotate's ("The f2_blocks directory");
// each caller passes its CURRENT exact string so `--help` stays byte-identical
// (behavior-preserving dedup).
void add_f2_dir_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<std::string>(
        "--f2-dir", [&a](const std::string& v) { a.f2_dir = v; }, help);
}

// Bind --target (target population label) into `a.target`. Shared by qpadm + qpadm-rotate
// (NOT qpwave — qpWave has no target). The help is IDENTICAL across both callers
// ("Target population label"), so no parametrization is needed.
void add_target_flag(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::string>(
        "--target", [&a](const std::string& v) { a.target = v; }, "Target population label");
}

// Bind --right (comma- or space-separated outgroup labels) into `a.right`. Shared by
// qpadm / qpwave / qpadm-rotate, with `->delimiter(',')`. The help is PARAMETRIZED because
// qpadm's help ("...right[0] = R0") differs from qpwave/rotate's ("Right outgroup labels");
// each caller passes its CURRENT exact string so `--help` stays byte-identical.
void add_right_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<std::vector<std::string>>(
            "--right", [&a](const std::vector<std::string>& v) { a.right = v; }, help)
        ->delimiter(',');
}

// Bind --left (comma- or space-separated population labels) into `a.left`. Shared by
// qpadm + qpwave, with `->delimiter(',')`. The help is PARAMETRIZED because the two help
// strings are SEMANTICALLY DISTINCT (qpadm "Left source population labels..." vs qpwave
// "Left population set; left[0] is the reference"); each caller passes its CURRENT exact
// string so `--help` stays byte-identical.
void add_left_flag(CLI::App* sub, CliArgs& a, const char* help) {
    sub->add_option_function<std::vector<std::string>>(
            "--left", [&a](const std::vector<std::string>& v) { a.left = v; }, help)
        ->delimiter(',');
}

// Bind the `f4` quartet flags: the ROW-ALIGNED --pop1/--pop2/--pop3/--pop4 columns
// (admixtools::f4 comb=FALSE — quartet k = (pop1[k],pop2[k],pop3[k],pop4[k])) AND the
// single-/multi-quartet --pops convenience (names in groups of 4). All comma/space
// delimited. f4 has NO target/left/right (it is a bare quartet stat), so this is the ONE
// new flag helper the command needs (cli-bindings.md §4.1; the rest is reused).
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

// Bind the `f3` triple flags: the THREE-slab clone of add_f4_quartet_flags (drop --pop4).
// The ROW-ALIGNED --pop1/--pop2/--pop3 columns (triple k = (pop1[k]=C, pop2[k]=A,
// pop3[k]=B)) AND the single-/multi-triple --pops convenience (names in groups of 3). All
// comma/space delimited. f3 has NO target/left/right (it is a bare triple stat), so this is
// the ONE new flag helper the command needs (cli-bindings.md §4.1; the rest is reused).
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

// Bind the `f4-ratio` 5-tuple flags: the FIVE-column clone of add_f4_quartet_flags (add
// --pop5). The ROW-ALIGNED --pop1..--pop5 columns (admixtools::qpf4ratio; tuple k =
// (pop1[k]..pop5[k]), alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4)) AND the single-/multi-tuple
// --pops convenience (names in groups of 5). All comma/space delimited. f4-ratio has NO
// target/left/right (it is a bare ratio stat), so this is the ONE new flag helper the command
// needs (cli-bindings.md §4.1; the rest is reused).
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

// GPU-only f-stat SWEEP MODE flags for the standalone-stat commands (f4 / f3 / qpdstat):
// the --all-quartets / --all-triples ENABLE flag (route to the GPU sweep over the C(P,k)
// of the --pops SUBSET — empty ⇒ the whole f2 dir), the on-device filter (--min-z |
// --top-k), the maxcomb-cap override (--sure), and the survivor destination (--shard-dir).
// It does NOT re-bind --pops (the f4/f3 command already binds it as the quartet/triple
// input; in sweep mode the same --pops is read as the SUBSET). `enable_flag`/`enable_help`
// parametrize the enable flag (f4/qpdstat: --all-quartets; f3: --all-triples) so each
// command's --help stays exact.
void add_sweep_mode_flags(CLI::App* sub, CliArgs& a, const char* enable_flag,
                          const char* enable_help) {
    sub->add_flag_function(enable_flag, [&a](std::int64_t) { a.sweep_all_combinations = true; },
                           enable_help);
    sub->add_option_function<double>(
        "--min-z", [&a](double v) { a.sweep_min_z = v; },
        "Sweep: keep items with |z| >= this (the on-device filter; default 3.0). Excludes --top-k.");
    sub->add_option_function<int>(
        "--top-k", [&a](int v) { a.sweep_top_k = v; },
        "Sweep: keep the K items with the largest |z| (bounded device-side reservoir, ~K resident). Excludes --min-z.");
    sub->add_flag_function(
        "--sure", [&a](std::int64_t) { a.sweep_sure = true; },
        "Sweep: lift the maxcomb cap (a sweep over more than the cap refuses without this).");
    sub->add_option_function<std::string>(
        "--shard-dir", [&a](const std::string& v) { a.shard_dir = v; },
        "Sweep: write the survivor table to a CSV under this dir (created if absent; vs stdout/--out).");
}

// GPU-only f-stat SWEEP flags: an OPTIONAL --pops SUBSET to sweep (empty ⇒ the whole f2 dir),
// the on-device filter (--min-z | --top-k), and the maxcomb-cap override (--sure). Reused by
// both f4-sweep and f3-sweep.
void add_sweep_flags(CLI::App* sub, CliArgs& a) {
    sub->add_option_function<std::vector<std::string>>(
            "--pops", [&a](const std::vector<std::string>& v) { a.pops = v; },
            "Population SUBSET to sweep all combinations of (names; empty ⇒ the whole f2 dir)")
        ->delimiter(',');
    sub->add_option_function<double>(
        "--min-z", [&a](double v) { a.sweep_min_z = v; },
        "Keep items with |z| >= this (the on-device filter; default 3.0). Excludes --top-k.");
    sub->add_option_function<int>(
        "--top-k", [&a](int v) { a.sweep_top_k = v; },
        "Keep the K items with the largest |z| (bounded device-side reservoir, ~K resident). Excludes --min-z.");
    sub->add_flag_function(
        "--sure", [&a](std::int64_t) { a.sweep_sure = true; },
        "Lift the maxcomb cap (a sweep over more than the cap refuses without this).");
}

}  // namespace

int run_cli(int argc, char** argv) {
    CLI::App app{"steppe — GPU/CUDA qpAdm: f-statistics & model fitting", "steppe"};
    app.set_version_flag("--version", std::string{STEPPE_VERSION});
    app.require_subcommand(0, 1);  // 0 ⇒ bare `steppe` prints help; 1 ⇒ one subcommand

    // One CliArgs per subcommand; only the selected subcommand's callback fires, so
    // the chosen args carry the user's input. We hold them in stable storage so the
    // bound lambdas outlive parse.
    CliArgs qpadm_args;
    CliArgs qpgraph_args;
    CliArgs qpwave_args;
    CliArgs rotate_args;
    CliArgs extract_args;
    CliArgs f4_args;
    CliArgs f3_args;
    CliArgs f4ratio_args;
    CliArgs qpdstat_args;
    CliArgs f4sweep_args;
    CliArgs f3sweep_args;
    CliArgs qpfstats_args;
    CliArgs dates_args;

    // ---- qpadm (cli-bindings.md §4.1) — M(cli-1) implements the compute ----------
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
            // The real GPU qpAdm fit (read dir -> resolve -> upload -> run_qpadm ->
            // emit CSV/JSON). qpadm-rotate + extract-f2 are likewise wired; only
            // qpwave (M(cli-2)) remains a scaffold no-op.
            std::exit(run_qpadm_command(*config));
        });
    }

    // ---- qpgraph (single-graph admixture-graph fit) ----------------------------
    {
        CLI::App* sub = app.add_subcommand("qpgraph", "Single-graph qpGraph fit (--f2-dir + --graph edge-list)");
        qpgraph_args.command = Command::QpGraph;
        add_f2_dir_flag(sub, qpgraph_args, "The f2_blocks directory (f2.bin + pops.txt)");
        add_qpgraph_flags(sub, qpgraph_args);
        // --fudge (the AT2 `diag` cc ridge) is shared with qpadm; bind ONLY that one option
        // (not the whole qpadm option set) so qpgraph's --help shows only its flags.
        sub->add_option_function<double>("--fudge", [&qpgraph_args](double v) { qpgraph_args.fudge = v; },
                                         "AT2 cc edge-solve ridge (diag; default 1e-4)");
        add_output_flags(sub, qpgraph_args);
        add_common_flags(sub, qpgraph_args);
        sub->callback([&]() {
            auto config = build_config(qpgraph_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            // The real GPU qpGraph fit (read dir + graph -> resolve leaves -> upload f2
            // RESIDENT -> run_qpgraph (the IDEA-1 fleet on-device) -> emit the edges +
            // admix weights + score). Mirrors how `qpadm` dispatches.
            std::exit(run_qpgraph_command(*config));
        });
    }

    // ---- dates (admixture DATING via the weighted ancestry-covariance decay) -----------
    // The DATES tool: --prefix genotypes + --target (admixed) + --left{2} (the two reference
    // sources) -> the date in generations + the leave-one-chromosome block-jackknife SE, via
    // the cuFFT autocorrelation LD engine (NEVER the f2 cache, NEVER a host O(M²) SNP-pair
    // loop). The .snp MUST carry a real genetic map (cM). Mirrors how qpfstats binds --prefix.
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
            // The real GPU DATES run (decode front-end -> the cuFFT autocorrelation LD engine
            // -> the exp-decay fit + leave-one-chrom jackknife). Mirrors how `qpadm` dispatches.
            std::exit(run_dates_command(*config));
        });
    }

    // ---- qpwave (cli-bindings.md §4.1) — M(cli-2) -------------------------------
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
            // M(cli-2): the real GPU qpWave rank sweep (read dir -> resolve left/right,
            // NO target, left[0]=reference -> upload -> run_qpwave -> emit the rank-sweep
            // table). Mirrors how `qpadm` dispatches.
            std::exit(run_qpwave_command(*config));
        });
    }

    // ---- f4 (standalone f4 statistic; fit-engine §6) ----------------------------
    {
        CLI::App* sub = app.add_subcommand(
            "f4", "Standalone f4(p1,p2;p3,p4) statistic (est/se/z/p per quartet)");
        f4_args.command = Command::F4;
        add_f2_dir_flag(sub, f4_args, "The f2_blocks directory");
        // f4 takes QUARTETS, not target/left/right: the row-aligned --pop1..--pop4 columns
        // OR the --pops 4-tuple convenience (the ONE new flag helper; cli-bindings.md §4.1).
        add_f4_quartet_flags(sub, f4_args);
        // SWEEP MODE: --all-quartets routes to the GPU sweep over C(P,4) of the --pops SUBSET
        // (empty ⇒ the whole f2 dir). The explicit-list path above stays byte-identical when
        // --all-quartets is absent (the goldens). Reuses the same --pops binding as the subset.
        add_sweep_mode_flags(sub, f4_args, "--all-quartets",
                             "Sweep ALL quartets C(P,4) over the --pops subset (empty ⇒ whole f2 dir)");
        add_output_flags(sub, f4_args);
        add_common_flags(sub, f4_args);
        sub->callback([&]() {
            auto config = build_config(f4_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            // The real GPU f4 (read dir -> resolve quartets -> upload -> run_f4 -> emit the
            // pop1,pop2,pop3,pop4,est,se,z,p table). Mirrors how `qpadm`/`qpwave` dispatch.
            std::exit(run_f4_command(*config));
        });
    }

    // ---- qpdstat (D-statistic / f4 over an f2_blocks dir; qpDstat A+B plan, Part A) ----
    // THIN wrapper over the f4 path: --f2-dir reports f4 (the AT2 f2-path convention, proven
    // byte-identical to qpdstat f4mode). The normalized-D MAGNITUDE needs per-SNP genotypes
    // (--prefix = Part B, not yet implemented). REUSES add_f2_dir_flag + add_f4_quartet_flags
    // (the QUADRUPLE input) + add_output_flags + add_common_flags verbatim; the ONLY new flag
    // is --prefix (the Part-B sentinel run_qpdstat_command fails fast on).
    {
        CLI::App* sub = app.add_subcommand(
            "qpdstat",
            "D-statistic / f4 over an f2_blocks dir (--f2-dir reports f4: the AT2 f2-path "
            "convention; normalized-D needs --prefix = Part B, not yet implemented)");
        qpdstat_args.command = Command::Qpdstat;
        add_f2_dir_flag(sub, qpdstat_args, "The f2_blocks directory");
        // qpdstat takes QUADRUPLES, not target/left/right: the row-aligned --pop1..--pop4
        // columns OR the --pops 4-tuple convenience (reused verbatim from f4).
        add_f4_quartet_flags(sub, qpdstat_args);
        // SWEEP MODE: --all-quartets routes to the GPU f4 sweep over C(P,4) of the --pops
        // SUBSET (the D-output-convention all-quartets scan). Explicit-list path unchanged.
        add_sweep_mode_flags(sub, qpdstat_args, "--all-quartets",
                             "Sweep ALL quadruples C(P,4) over the --pops subset (empty ⇒ whole f2 dir)");
        // --prefix: the Part-B genotype prefix for the normalized-D magnitude. Bound to the
        // DEDICATED qpdstat_prefix field (NOT extract's --prefix) so the qpdstat command's
        // fail-fast guard reads it WITHOUT ConfigBuilder expanding it into geno/snp/ind.
        sub->add_option_function<std::string>(
            "--prefix", [&](const std::string& v) { qpdstat_args.qpdstat_prefix = v; },
            "Genotype prefix for normalized-D magnitude (Part B; not yet implemented). "
            "--f2-dir qpdstat reports f4 (the AT2 f2-path convention).");
        add_output_flags(sub, qpdstat_args);
        add_common_flags(sub, qpdstat_args);
        sub->callback([&]() {
            auto config = build_config(qpdstat_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            // Part A: the real GPU f4 over the quadruples (read dir -> resolve -> upload ->
            // run_f4 -> emit the pop1,pop2,pop3,pop4,est,se,z,p table — the D-output
            // convention). --prefix (Part B) fails fast inside run_qpdstat_command.
            std::exit(run_qpdstat_command(*config));
        });
    }

    // ---- f3 (standalone f3 statistic; fit-engine §6) ----------------------------
    {
        CLI::App* sub = app.add_subcommand(
            "f3", "Standalone f3(C;A,B) statistic (est/se/z/p per triple)");
        f3_args.command = Command::F3;
        add_f2_dir_flag(sub, f3_args, "The f2_blocks directory");
        // f3 takes TRIPLES, not target/left/right: the row-aligned --pop1..--pop3 columns
        // OR the --pops 3-tuple convenience (the ONE new flag helper; cli-bindings.md §4.1).
        add_f3_triple_flags(sub, f3_args);
        // SWEEP MODE: --all-triples routes to the GPU sweep over C(P,3) of the --pops SUBSET
        // (empty ⇒ the whole f2 dir). The explicit-list path above stays byte-identical when
        // --all-triples is absent (the goldens). Reuses the same --pops binding as the subset.
        add_sweep_mode_flags(sub, f3_args, "--all-triples",
                             "Sweep ALL triples C(P,3) over the --pops subset (empty ⇒ whole f2 dir)");
        add_output_flags(sub, f3_args);
        add_common_flags(sub, f3_args);
        sub->callback([&]() {
            auto config = build_config(f3_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            // The real GPU f3 (read dir -> resolve triples -> upload -> run_f3 -> emit the
            // pop1,pop2,pop3,est,se,z,p table). Mirrors how `qpadm`/`qpwave`/`f4` dispatch.
            std::exit(run_f3_command(*config));
        });
    }

    // ---- f4-ratio (standalone f4-ratio statistic; fit-engine §6) ----------------
    {
        CLI::App* sub = app.add_subcommand(
            "f4-ratio",
            "Standalone f4-ratio alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4) (alpha/se/z per 5-tuple)");
        f4ratio_args.command = Command::F4Ratio;
        add_f2_dir_flag(sub, f4ratio_args, "The f2_blocks directory");
        // f4-ratio takes 5-TUPLES, not target/left/right: the row-aligned --pop1..--pop5
        // columns OR the --pops 5-tuple convenience (the ONE new flag helper; cli-bindings §4.1).
        add_f4ratio_flags(sub, f4ratio_args);
        add_output_flags(sub, f4ratio_args);
        add_common_flags(sub, f4ratio_args);
        sub->callback([&]() {
            auto config = build_config(f4ratio_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            // The real GPU f4-ratio (read dir -> resolve 5-tuples -> upload -> run_f4ratio ->
            // emit the pop1..pop5,alpha,se,z table). Mirrors how `qpadm`/`f4`/`f3` dispatch.
            std::exit(run_f4ratio_command(*config));
        });
    }

    // ---- f4-sweep (GPU-only all-combinations f4 sweep) --------------------------
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

    // ---- f3-sweep (GPU-only all-combinations f3 sweep) --------------------------
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

    // ---- qpfstats (genotype-path joint f2 SMOOTHER) -----------------------------
    // Reads GENOTYPES (--prefix) + smooths over --pops -> a smoothed f2 DIR (--out-dir)
    // that qpadm/f4/qpGraph consume. REUSES the qpDstat-B genotype seam + the dstat-
    // numerator engine over the full f2/f3/f4 popcomb set + the on-device smoothing solve.
    {
        CLI::App* sub = app.add_subcommand(
            "qpfstats",
            "Genotype-path joint f2 smoother: --prefix genotypes + --pops -> a smoothed f2 dir");
        qpfstats_args.command = Command::Qpfstats;
        sub->add_option_function<std::string>(
            "--prefix", [&](const std::string& v) { qpfstats_args.qpdstat_prefix = v; },
            "Genotype triple prefix (reads PREFIX.{geno,snp,ind})");
        sub->add_option_function<std::vector<std::string>>(
            "--pops", [&](const std::vector<std::string>& v) { qpfstats_args.pops = v; },
            "Population set to smooth over (sorted ASC internally = the AT2 dimnames order)")
            ->delimiter(',');
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

    // ---- qpadm-rotate (cli-bindings.md §4.1) — M(cli-3) -------------------------
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
            // M(cli-3): the real GPU qpAdm ROTATION (read dir -> resolve target/pool/
            // right -> enumerate pool subsets [min,max] -> ONE batched run_qpadm_search
            // -> emit the per-model table). Mirrors how `qpadm` dispatches.
            std::exit(run_qpadm_rotate_command(*config));
        });
    }

    // ---- extract-f2 (cli-bindings.md §4.1) — M(cli-4) ---------------------------
    {
        CLI::App* sub = app.add_subcommand("extract-f2", "Precompute the f2_blocks dir from genotypes");
        extract_args.command = Command::ExtractF2;
        sub->add_option_function<std::string>("--prefix", [&](const std::string& v) { extract_args.prefix = v; },
                                              "Genotype triple prefix (sets --geno/--snp/--ind = PREFIX.{geno,snp,ind})");
        sub->add_option_function<std::string>("--geno", [&](const std::string& v) { extract_args.geno = v; }, "Genotype file (overrides --prefix)");
        sub->add_option_function<std::string>("--snp",  [&](const std::string& v) { extract_args.snp = v; },  "SNP file (overrides --prefix)");
        sub->add_option_function<std::string>("--ind",  [&](const std::string& v) { extract_args.ind = v; },  "Individual file (overrides --prefix)");
        sub->add_option_function<std::string>("--out",  [&](const std::string& v) { extract_args.out_dir = v; }, "Output f2_blocks dir");
        sub->add_option_function<std::vector<std::string>>(
            "--pops", [&](const std::vector<std::string>& v) { extract_args.pops = v; },
            "Explicit population list")->delimiter(',');
        sub->add_option_function<int>("--auto-top-k", [&](int v) { extract_args.auto_top_k = v; }, "Keep the K largest pops");
        sub->add_option_function<int>("--min-n", [&](int v) { extract_args.min_n = v; }, "Keep pops with >= N individuals");
        sub->add_option_function<double>("--blgsize", [&](double v) { extract_args.blgsize = v; }, "Jackknife block size in MORGANS (AT2 convention; default 0.05 = 5 cM)");
        sub->add_option_function<double>("--maf", [&](double v) { extract_args.maf = v; }, "Minimum MAF");
        sub->add_option_function<double>("--geno-max-miss", [&](double v) { extract_args.geno_max_missing = v; }, "Max per-SNP missing fraction");
        // --maxmiss: the AT2-ergonomic alias for --geno-max-miss (AT2 maxmiss == keep
        // iff per-SNP missing frac across selected pops <= maxmiss == geno_max_missing).
        sub->add_option_function<double>("--maxmiss", [&](double v) { extract_args.geno_max_missing = v; }, "AT2 alias for --geno-max-miss");
        sub->add_option_function<double>("--mind-max-miss", [&](double v) { extract_args.mind_max_missing = v; }, "Max per-sample missing fraction");
        // --auto-only / --no-auto-only: extract-f2 defaults autosomes_only ON (AT2
        // extract_f2 default auto_only=TRUE = chr 1-22); --no-auto-only turns it off.
        sub->add_flag_function("--auto-only,!--no-auto-only",
                               [&](std::int64_t v) { extract_args.autosomes_only = (v >= 0); },
                               "Keep only autosomes chr 1-22 (default on; --no-auto-only to disable)");
        // --drop-mono / --no-drop-mono: extract-f2 defaults drop_monomorphic ON (AT2
        // extract_f2 builds f2 on the polymorphic subset — its `poly_only` default);
        // --no-drop-mono keeps monomorphic SNPs.
        sub->add_flag_function("--drop-mono,!--no-drop-mono",
                               [&](std::int64_t v) { extract_args.drop_monomorphic = (v >= 0); },
                               "Drop monomorphic SNPs (default on, AT2 poly_only parity; --no-drop-mono to keep)");
        sub->add_flag_function("--transversions", [&](std::int64_t) { extract_args.transversions_only = true; }, "Keep only transversions");
        // --ploidy auto|1|2: AT2 adjust_pseudohaploid policy (default auto = per-sample
        // detection; the f2 pseudo-haploid fix). 1 = force pseudo-haploid, 2 = force
        // diploid (the legacy hardcoded behavior). An unknown token is InvalidConfig.
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
        // --tier auto|resident|host|disk: the M5 f2_blocks OUTPUT-tier override. auto
        // (default) = the runtime select_output_tier policy; resident = the device-
        // resident path (the small-input path, byte-identical to today); host/disk =
        // the SNP-tile input-streaming tiers that keep the GPU peak independent of M so
        // high-P full-autosome runs that OOM the resident feeder complete. The raw token
        // is carried; ConfigBuilder maps it to DeviceConfig::force_tier (unknown token =
        // InvalidConfig). STEPPE_FORCE_TIER stays the lower-precedence env fallback.
        sub->add_option_function<std::string>(
            "--tier", [&](const std::string& v) { extract_args.tier = v; },
            "f2_blocks output tier: auto | resident | host | disk (default auto; "
            "host/disk stream the SNP-tile input so high-P runs that OOM resident complete)");
        sub->add_flag_function("--dry-run", [&](std::int64_t) { extract_args.dry_run = true; }, "Report sizes/tier/precision, no compute");
        // --hash / --no-hash: source-provenance SHA-256 opt-in. DEFAULT OFF — the whole-
        // .geno SHA is a ~tens-of-seconds whole-file read+compress that dominated extract-f2
        // (a provenance value, not correctness), so it is skipped unless requested. When
        // ON it is overlapped on a background thread with the GPU decode+f2 pipeline.
        // --no-hash is accepted as the explicit form of the default. meta.json records
        // source_hash_computed + empty *_sha256 when skipped (the deliberate-absence marker).
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

    // No subcommand selected (bare `steppe`): CLI11 with require_subcommand(0,1) does
    // not error; print help and exit 0 so a bare invocation is a clean, documented
    // no-op (cli-bindings.md §4; the subcommand callbacks std::exit before reaching
    // here when one is chosen).
    std::printf("%s", app.help().c_str());
    return cfg::kExitOk;
}

}  // namespace steppe::app
