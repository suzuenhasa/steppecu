// tests/unit/test_config_builder.cpp
//
// Host-only unit test of the M(cli-0) access-layer config contract
// (src/core/config/{config_builder,run_config,cli_args,exit_code}.hpp; architecture.md
// §9; cli-bindings.md §4.5, §8 M(cli-0) gate). Pure C++ TU, NO GPU, NO CUDA: the
// ConfigBuilder/RunConfig/CliArgs are deliberately CUDA-free, so the very fact this TU
// compiles + links WITHOUT the device layer is itself the §4-layering proof.
//
// VERDICT GATE (cli-bindings.md §8 M(cli-0)): the §9 PRECEDENCE order
// (compiled defaults < TOML < env STEPPE_* < CLI) holds, and build() REJECTS bad
// config as Status::InvalidConfig (fail-fast), while a bare invocation maps onto the
// real-struct defaults that reproduce the goldens. It mirrors the existing test_config
// pattern: a dual GoogleTest / self-checking-main() harness gated on the exit code.
//
// REAL-DATA NOTE: this is the §9 config-CONTRACT unit (the M(cli-0) gate explicitly
// names "unit: precedence order + build() rejects ... reuse the test_config pattern").
// It exercises NO data and NO compute — there is no synthetic statistic anywhere; the
// real-golden CLI gate is M(cli-1) (steppe qpadm reproduces golden_fit0/NRBIG through
// the CLI on the GPU), a separate item.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/config/cli_args.hpp"
#include "core/config/config_builder.hpp"
#include "core/config/exit_code.hpp"
#include "core/config/run_config.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/qpadm.hpp"

namespace {

using steppe::Precision;
using steppe::Status;
using steppe::config::CliArgs;
using steppe::config::Command;
using steppe::config::ConfigBuilder;
using steppe::config::RunConfig;

// Small env-var RAII so the env-precedence cases do not leak STEPPE_* into siblings.
struct ScopedEnv {
    std::string key;
    ScopedEnv(const char* k, const char* v) : key(k) {
#if defined(_WIN32)
        _putenv_s(k, v);
#else
        setenv(k, v, /*overwrite=*/1);
#endif
    }
    ~ScopedEnv() {
#if defined(_WIN32)
        _putenv_s(key.c_str(), "");
#else
        unsetenv(key.c_str());
#endif
    }
};

// ---- defaults: a bare qpadm invocation maps onto the golden-reproducing defaults ---
[[nodiscard]] bool test_defaults_match_struct_defaults() {
    CliArgs a;
    a.command = Command::QpAdm;
    const auto r = ConfigBuilder().with_defaults().merge_env().merge_cli(a).build();
    if (!r) return false;
    const RunConfig& c = *r;
    const steppe::QpAdmOptions def{};
    return c.command() == Command::QpAdm &&
           c.device().devices.empty() &&                              // auto-enumerate (§9)
           c.device().precision.kind == Precision::Kind::EmulatedFp64 &&
           c.device().precision.mantissa_bits == steppe::kDefaultMantissaBits &&
           c.qpadm_options().fudge == def.fudge &&                    // golden-reproducing
           c.qpadm_options().als_iterations == def.als_iterations &&
           c.qpadm_options().rank == def.rank &&
           c.qpadm_options().jackknife == def.jackknife &&
           c.format() == "csv" &&
           c.blgsize_cm() == steppe::kDefaultBlockSizeCm;
}

// ---- precedence: CLI overrides env overrides default (the §9 chain) ----------------
[[nodiscard]] bool test_cli_overrides_env() {
    ScopedEnv e_dev("STEPPE_DEVICE", "0");
    ScopedEnv e_prec("STEPPE_PRECISION", "fp64");
    CliArgs a;
    a.command = Command::QpAdm;
    a.device = std::string{"0,1"};       // CLI wins over the env "0"
    a.precision = std::string{"emu32"};  // CLI wins over the env "fp64"
    const auto r = ConfigBuilder().with_defaults().merge_env().merge_cli(a).build();
    if (!r) return false;
    const RunConfig& c = *r;
    return c.device().devices.size() == 2 && c.device().devices[0] == 0 &&
           c.device().devices[1] == 1 &&
           c.device().precision.kind == Precision::Kind::EmulatedFp64 &&
           c.device().precision.mantissa_bits == 32;
}

// ---- precedence: env overrides the compiled default when the CLI is silent ---------
[[nodiscard]] bool test_env_overrides_default() {
    ScopedEnv e_prec("STEPPE_PRECISION", "fp64");
    ScopedEnv e_fmt("STEPPE_FORMAT", "json");
    CliArgs a;
    a.command = Command::QpAdm;  // no --precision / --format on the CLI
    const auto r = ConfigBuilder().with_defaults().merge_env().merge_cli(a).build();
    if (!r) return false;
    const RunConfig& c = *r;
    return c.device().precision.kind == Precision::Kind::Fp64 && c.format() == "json";
}

// ---- precedence: an UNSET CLI field does NOT clobber the env layer ------------------
[[nodiscard]] bool test_unset_cli_preserves_env() {
    ScopedEnv e_dev("STEPPE_DEVICE", "1");
    CliArgs a;
    a.command = Command::QpAdm;  // device unset on the CLI -> env "1" survives
    const auto r = ConfigBuilder().with_defaults().merge_env().merge_cli(a).build();
    if (!r) return false;
    return r->device().devices.size() == 1 && r->device().devices[0] == 1;
}

// ---- GPU-only: --device cpu is rejected (cli-bindings.md §5.4) ----------------------
[[nodiscard]] bool test_reject_device_cpu() {
    CliArgs a;
    a.command = Command::QpAdm;
    a.device = std::string{"cpu"};
    ConfigBuilder b;
    const auto r = b.with_defaults().merge_cli(a).build();
    return !r && r.error() == Status::InvalidConfig && !b.error_message().empty();
}

// ---- build() rejects: bad device ordinal / duplicate / negative --------------------
[[nodiscard]] bool test_reject_bad_device() {
    const char* bad[] = {"x", "0,0", "-1", "0,abc"};
    for (const char* d : bad) {
        CliArgs a;
        a.command = Command::QpAdm;
        a.device = std::string{d};
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (r || r.error() != Status::InvalidConfig) return false;
    }
    return true;
}

// ---- build() rejects: unknown precision / format tokens ----------------------------
[[nodiscard]] bool test_reject_bad_tokens() {
    {
        CliArgs a; a.command = Command::QpAdm; a.precision = std::string{"emu99"};
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (r || r.error() != Status::InvalidConfig) return false;
    }
    {
        CliArgs a; a.command = Command::QpAdm; a.format = std::string{"parquet"};
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (r || r.error() != Status::InvalidConfig) return false;
    }
    return true;
}

// ---- build() rejects: out-of-range numeric knobs -----------------------------------
[[nodiscard]] bool test_reject_bad_ranges() {
    {
        CliArgs a; a.command = Command::QpAdm; a.jackknife = 3;  // 0..2 only
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (r || r.error() != Status::InvalidConfig) return false;
    }
    {
        CliArgs a; a.command = Command::QpAdm; a.rank_alpha = 1.5;  // (0,1)
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (r || r.error() != Status::InvalidConfig) return false;
    }
    {
        CliArgs a; a.command = Command::ExtractF2; a.maf = 0.9;  // [0,0.5]
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (r || r.error() != Status::InvalidConfig) return false;
    }
    {
        CliArgs a; a.command = Command::QpAdm; a.als_iterations = 0;  // >= 1
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (r || r.error() != Status::InvalidConfig) return false;
    }
    return true;
}

// ---- build() rejects: mutually-exclusive pop-selection modes -----------------------
[[nodiscard]] bool test_reject_conflicting_pop_modes() {
    CliArgs a;
    a.command = Command::ExtractF2;
    a.pops = {"French", "Han"};
    a.auto_top_k = 50;  // conflicts with --pops
    const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
    return !r && r.error() == Status::InvalidConfig;
}

// ---- build() rejects: a TOML config request (no parser compiled in M(cli-0)) -------
[[nodiscard]] bool test_reject_toml_request() {
    CliArgs a;
    a.command = Command::QpAdm;
    const auto r = ConfigBuilder().with_defaults().merge_file("steppe.toml").merge_cli(a).build();
    return !r && r.error() == Status::InvalidConfig;
}

// ---- valid precision/device tokens parse correctly ---------------------------------
[[nodiscard]] bool test_valid_token_mapping() {
    struct { const char* tok; Precision::Kind kind; int bits; } cases[] = {
        {"emu40", Precision::Kind::EmulatedFp64, 40},
        {"emu32", Precision::Kind::EmulatedFp64, 32},
        {"fp64",  Precision::Kind::Fp64,         steppe::kDefaultMantissaBits},
        {"tf32",  Precision::Kind::Tf32,         steppe::kDefaultMantissaBits},
    };
    for (const auto& cse : cases) {
        CliArgs a; a.command = Command::QpAdm; a.precision = std::string{cse.tok};
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (!r) return false;
        if (r->device().precision.kind != cse.kind) return false;
        if (cse.kind == Precision::Kind::EmulatedFp64 &&
            r->device().precision.mantissa_bits != cse.bits) return false;
    }
    // "auto" / "" -> empty devices (auto-enumerate).
    {
        CliArgs a; a.command = Command::QpAdm; a.device = std::string{"auto"};
        const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
        if (!r || !r->device().devices.empty()) return false;
    }
    return true;
}

// ---- the Status -> exit-code map (record-and-continue; cli-bindings.md §1.3/§4.4) ---
[[nodiscard]] bool test_exit_code_map() {
    using steppe::config::exit_code_for;
    using namespace steppe::config;
    return exit_code_for(Status::Ok) == kExitOk &&
           // domain outcomes are rows, not faults -> exit 0
           exit_code_for(Status::RankDeficient) == kExitOk &&
           exit_code_for(Status::NonSpdCovariance) == kExitOk &&
           exit_code_for(Status::ChisqUndefined) == kExitOk &&
           // faults -> nonzero, distinct
           exit_code_for(Status::InvalidConfig) == kExitInvalidConfig &&
           exit_code_for(Status::DeviceOom) == kExitDeviceOom &&
           kExitInvalidConfig != kExitOk && kExitDeviceOom != kExitOk;
}

// ---- QpAdmOptions overrides flow through build() ------------------------------------
[[nodiscard]] bool test_qpadm_option_overrides() {
    CliArgs a;
    a.command = Command::QpAdm;
    a.fudge = 1e-3;
    a.als_iterations = 40;
    a.rank = 2;
    a.jackknife = 1;
    a.allow_negative_weights = false;
    const auto r = ConfigBuilder().with_defaults().merge_cli(a).build();
    if (!r) return false;
    const auto& o = r->qpadm_options();
    return o.fudge == 1e-3 && o.als_iterations == 40 && o.rank == 2 &&
           o.jackknife == steppe::JackknifePolicy::FeasibleOnly &&
           o.allow_negative_weights == false;
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"bare invocation maps onto the golden-reproducing struct defaults", test_defaults_match_struct_defaults},
    {"precedence: CLI overrides env (§9)", test_cli_overrides_env},
    {"precedence: env overrides compiled default (§9)", test_env_overrides_default},
    {"precedence: unset CLI field preserves the env layer (§9)", test_unset_cli_preserves_env},
    {"GPU-only: --device cpu rejected (cli-bindings §5.4)", test_reject_device_cpu},
    {"build() rejects bad device ordinals", test_reject_bad_device},
    {"build() rejects unknown precision/format tokens", test_reject_bad_tokens},
    {"build() rejects out-of-range numeric knobs", test_reject_bad_ranges},
    {"build() rejects conflicting pop-selection modes", test_reject_conflicting_pop_modes},
    {"build() rejects an unsupported TOML config request", test_reject_toml_request},
    {"valid precision/device tokens map correctly", test_valid_token_mapping},
    {"Status -> exit-code map (record-and-continue)", test_exit_code_map},
    {"QpAdmOptions overrides flow through build()", test_qpadm_option_overrides},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(ConfigBuilder, PrecedenceAndValidation) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_config_builder: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_config_builder: all %zu config-contract checks PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif
