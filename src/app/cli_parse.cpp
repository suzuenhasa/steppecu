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
//   4. M(cli-0): print "not yet implemented" + return 0 (no compute this milestone).
//      M(cli-1) replaces step 4 for `qpadm` with the real GPU fit.
#include "app/cli_parse.hpp"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

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

// M(cli-0) placeholder: a subcommand that parses + validates its config but performs
// no compute yet. Returns kExitOk so the scaffold is a clean, scriptable no-op (the
// compute lands in M(cli-1+)). qpadm is the one M(cli-1) will replace.
[[nodiscard]] int run_not_yet_implemented(const char* name, const RunConfig& /*config*/) {
    std::printf("steppe %s: not yet implemented (M(cli-0) scaffold — config parsed "
                "and validated; compute lands in a later milestone)\n", name);
    return cfg::kExitOk;
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

}  // namespace

int run_cli(int argc, char** argv) {
    CLI::App app{"steppe — GPU/CUDA qpAdm: f-statistics & model fitting", "steppe"};
    app.set_version_flag("--version", std::string{STEPPE_VERSION});
    app.require_subcommand(0, 1);  // 0 ⇒ bare `steppe` prints help; 1 ⇒ one subcommand

    // One CliArgs per subcommand; only the selected subcommand's callback fires, so
    // the chosen args carry the user's input. We hold them in stable storage so the
    // bound lambdas outlive parse.
    CliArgs qpadm_args;
    CliArgs qpwave_args;
    CliArgs rotate_args;
    CliArgs extract_args;

    // ---- qpadm (cli-bindings.md §4.1) — M(cli-1) implements the compute ----------
    {
        CLI::App* sub = app.add_subcommand("qpadm", "qpAdm fit over an f2_blocks dir");
        qpadm_args.command = Command::QpAdm;
        sub->add_option_function<std::string>("--f2-dir", [&](const std::string& v) { qpadm_args.f2_dir = v; },
                                              "The f2_blocks directory (f2.bin + pops.txt + meta.json)");
        sub->add_option_function<std::string>("--target", [&](const std::string& v) { qpadm_args.target = v; },
                                              "Target population label");
        sub->add_option_function<std::vector<std::string>>(
            "--left", [&](const std::vector<std::string>& v) { qpadm_args.left = v; },
            "Left source population labels (comma- or space-separated)")->delimiter(',');
        sub->add_option_function<std::vector<std::string>>(
            "--right", [&](const std::vector<std::string>& v) { qpadm_args.right = v; },
            "Right outgroup labels; right[0] = R0")->delimiter(',');
        add_qpadm_option_flags(sub, qpadm_args);
        add_output_flags(sub, qpadm_args);
        add_common_flags(sub, qpadm_args);
        sub->callback([&]() {
            auto config = build_config(qpadm_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_not_yet_implemented("qpadm", *config));
        });
    }

    // ---- qpwave (cli-bindings.md §4.1) — M(cli-2) -------------------------------
    {
        CLI::App* sub = app.add_subcommand("qpwave", "qpWave rank sweep (no target; left[0]=ref)");
        qpwave_args.command = Command::QpWave;
        sub->add_option_function<std::string>("--f2-dir", [&](const std::string& v) { qpwave_args.f2_dir = v; },
                                              "The f2_blocks directory");
        sub->add_option_function<std::vector<std::string>>(
            "--left", [&](const std::vector<std::string>& v) { qpwave_args.left = v; },
            "Left population set; left[0] is the reference")->delimiter(',');
        sub->add_option_function<std::vector<std::string>>(
            "--right", [&](const std::vector<std::string>& v) { qpwave_args.right = v; },
            "Right outgroup labels")->delimiter(',');
        add_qpadm_option_flags(sub, qpwave_args);
        add_output_flags(sub, qpwave_args);
        add_common_flags(sub, qpwave_args);
        sub->callback([&]() {
            auto config = build_config(qpwave_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_not_yet_implemented("qpwave", *config));
        });
    }

    // ---- qpadm-rotate (cli-bindings.md §4.1) — M(cli-3) -------------------------
    {
        CLI::App* sub = app.add_subcommand("qpadm-rotate", "qpAdm rotation over a source pool");
        rotate_args.command = Command::QpAdmRotate;
        sub->add_option_function<std::string>("--f2-dir", [&](const std::string& v) { rotate_args.f2_dir = v; },
                                              "The f2_blocks directory");
        sub->add_option_function<std::string>("--target", [&](const std::string& v) { rotate_args.target = v; },
                                              "Target population label");
        sub->add_option_function<std::vector<std::string>>(
            "--pool", [&](const std::vector<std::string>& v) { rotate_args.pool = v; },
            "Source pool to enumerate subsets of")->delimiter(',');
        sub->add_option_function<std::vector<std::string>>(
            "--right", [&](const std::vector<std::string>& v) { rotate_args.right = v; },
            "Right outgroup labels")->delimiter(',');
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
            std::exit(run_not_yet_implemented("qpadm-rotate", *config));
        });
    }

    // ---- extract-f2 (cli-bindings.md §4.1) — M(cli-4) ---------------------------
    {
        CLI::App* sub = app.add_subcommand("extract-f2", "Precompute the f2_blocks dir from genotypes");
        extract_args.command = Command::ExtractF2;
        sub->add_option_function<std::string>("--geno", [&](const std::string& v) { extract_args.geno = v; }, "Genotype file");
        sub->add_option_function<std::string>("--snp",  [&](const std::string& v) { extract_args.snp = v; },  "SNP file");
        sub->add_option_function<std::string>("--ind",  [&](const std::string& v) { extract_args.ind = v; },  "Individual file");
        sub->add_option_function<std::string>("--out",  [&](const std::string& v) { extract_args.out_dir = v; }, "Output f2_blocks dir");
        sub->add_option_function<std::vector<std::string>>(
            "--pops", [&](const std::vector<std::string>& v) { extract_args.pops = v; },
            "Explicit population list")->delimiter(',');
        sub->add_option_function<int>("--auto-top-k", [&](int v) { extract_args.auto_top_k = v; }, "Keep the K largest pops");
        sub->add_option_function<int>("--min-n", [&](int v) { extract_args.min_n = v; }, "Keep pops with >= N individuals");
        sub->add_option_function<double>("--blgsize", [&](double v) { extract_args.blgsize = v; }, "Jackknife block size (cM, default 5)");
        sub->add_option_function<double>("--maf", [&](double v) { extract_args.maf = v; }, "Minimum MAF");
        sub->add_option_function<double>("--geno-max-miss", [&](double v) { extract_args.geno_max_missing = v; }, "Max per-SNP missing fraction");
        sub->add_option_function<double>("--mind-max-miss", [&](double v) { extract_args.mind_max_missing = v; }, "Max per-sample missing fraction");
        sub->add_flag_function("--auto-only", [&](std::int64_t) { extract_args.autosomes_only = true; }, "Keep only autosomes (chr 1-22)");
        sub->add_flag_function("--drop-mono", [&](std::int64_t) { extract_args.drop_monomorphic = true; }, "Drop monomorphic SNPs");
        sub->add_flag_function("--transversions", [&](std::int64_t) { extract_args.transversions_only = true; }, "Keep only transversions");
        add_common_flags(sub, extract_args);
        sub->callback([&]() {
            auto config = build_config(extract_args);
            if (!config) std::exit(cfg::kExitInvalidConfig);
            std::exit(run_not_yet_implemented("extract-f2", *config));
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
