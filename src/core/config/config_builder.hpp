// src/core/config/config_builder.hpp
//
// ConfigBuilder — the ONLY mutable config type (architecture.md §9). It accumulates
// the config LAYERS in the §9 precedence order
//   compiled defaults < TOML file < env (STEPPE_*) < CLI
// and build() validates ONCE — fail-fast — and freezes into an immutable RunConfig
// (architecture.md §9; cli-bindings.md §4.5). After build() the config is const.
//
// LAYER API (architecture.md §9 ConfigBuilder sketch):
//   with_defaults()        seed the compiled defaults (the real struct defaults).
//   merge_file(path)       fold a TOML file over the current state (BELOW env+CLI).
//   merge_env()            fold the STEPPE_* environment over the current state.
//   merge_cli(args)        fold the parsed CliArgs (HIGHEST precedence) over it.
//   build()                validate + freeze -> std::expected<RunConfig, Status>.
//
// The merge functions return *this (chainable, the §9 fluent builder). Each higher
// layer overrides ONLY the fields it actually set (the std::optional "was it set?"
// sentinel on CliArgs; a present env var; a present TOML key) — an unset field never
// clobbers a lower layer. build() is where the raw string knobs ("--device 0,1",
// "--precision emu40", "--jackknife 1") are PARSED + RANGE-CHECKED into the typed real
// structs (DeviceConfig / QpAdmOptions / FilterConfig / PopSelection); a bad token,
// an out-of-range value, or a GPU-only-violation ("--device cpu") is rejected as
// Status::InvalidConfig — never silently coerced.
//
// CUDA-FREE BY CONTRACT (architecture.md §4): this builder includes no device header
// and makes no CUDA call. build()'s device/VRAM/precision-HONORABILITY check at the
// LIVE-probe level (architecture.md §9: "a precision the selected backend cannot
// honor ... fall back or error") is the APP's job at run time through the CUDA-free
// build_resources / BackendCapabilities seam (cli-bindings.md §4.5) — this layer does
// the STATIC validation (token legality, ranges, GPU-only, flag conflicts) that needs
// no device, so it is unit-tested with NO GPU (the test_config pattern).
//
// TOML: M(cli-0) does NOT take a TOML dependency (none is in the build yet — the
// architecture's third_party/CPM pins are app-private and CLI11-only for this
// milestone). merge_file() is wired as a documented no-op-when-absent seam: an empty
// path is a no-op; a non-empty path with no TOML parser compiled in is reported as
// Status::InvalidConfig at build() rather than silently ignored. The TOML body lands
// with the parser dependency in a later milestone (cli-bindings.md §9.1).
#ifndef STEPPE_CORE_CONFIG_CONFIG_BUILDER_HPP
#define STEPPE_CORE_CONFIG_CONFIG_BUILDER_HPP

#include <filesystem>
#include <string>

#include "steppe/error.hpp"               // steppe::Status
#include "core/config/build_result.hpp"   // BuildResult<RunConfig> (C++20 expected stand-in)
#include "core/config/cli_args.hpp"       // CliArgs
#include "core/config/run_config.hpp"     // RunConfig

namespace steppe::config {

/// Mutable, layered config accumulator. Construct, fold layers in precedence order,
/// then build() to validate + freeze. Not thread-safe (a single builder is owned by
/// one caller); the frozen RunConfig is const and freely shareable.
class ConfigBuilder {
public:
    ConfigBuilder() = default;

    /// Seed the compiled defaults — the real struct defaults (DeviceConfig{},
    /// QpAdmOptions{}, FilterConfig{}, the cli-bindings.md §4.4 csv format, 5 cM
    /// blgsize). The lowest precedence layer. Idempotent; safe to call once up front.
    ConfigBuilder& with_defaults();

    /// Fold a TOML config file over the current state (BELOW env + CLI; architecture
    /// §9). Empty path ⇒ no-op (no file requested). A non-empty path is RECORDED and
    /// validated at build() (M(cli-0): no TOML parser is compiled in yet, so a
    /// non-empty path is rejected at build() as InvalidConfig rather than silently
    /// dropped — the parser body lands later, cli-bindings.md §9.1).
    ConfigBuilder& merge_file(const std::filesystem::path& path);

    /// Fold the STEPPE_* environment variables over the current state (ABOVE TOML,
    /// BELOW CLI; architecture §9). Reads only the documented STEPPE_* keys (e.g.
    /// STEPPE_DEVICE, STEPPE_PRECISION, STEPPE_FORMAT). An unset var leaves the lower
    /// layer intact; a set var overrides it. Unknown STEPPE_* keys are IGNORED (not an
    /// error — a forward-compat key for a newer steppe is not this build's concern).
    ConfigBuilder& merge_env();

    /// Fold the parsed CLI args over the current state (HIGHEST precedence,
    /// architecture §9). Only the std::optional fields that were SET on the command
    /// line override the lower layers; an unset field is left as-is. The list fields
    /// (left/right/pool/pops) override when non-empty.
    ConfigBuilder& merge_cli(const CliArgs& args);

    /// Validate ONCE + freeze into an immutable RunConfig (architecture §9 fail-fast).
    /// Parses + range-checks the raw string knobs into the typed structs:
    ///   * "--device" -> DeviceConfig::devices  ("auto"/""=auto-enumerate; "0"/"0,1"
    ///     = explicit ordinals; "cpu" REJECTED — GPU-only, cli-bindings.md §5.4;
    ///     negative / duplicate / non-numeric ordinal REJECTED).
    ///   * "--precision" -> Precision  (emu40/emu32 = EmulatedFp64{40|32}; fp64 = Fp64;
    ///     tf32 = Tf32; unknown token REJECTED).
    ///   * "--jackknife" 0|1|2 -> JackknifePolicy (out-of-range REJECTED).
    ///   * "--format" csv|tsv|json (unknown REJECTED).
    ///   * numeric ranges: fudge >= 0, als_iterations >= 1, rank_alpha in (0,1),
    ///     blgsize > 0, the filter fractions in [0,1], min_sources >= 1.
    ///   * a recorded non-empty TOML path with no parser compiled in -> REJECTED.
    /// @returns the frozen RunConfig on success; unexpected(Status::InvalidConfig)
    ///          with a single human-readable reason in `error_message()` on any
    ///          violation. (The architecture §9 build() returns std::expected<RunConfig,
    ///          Error>; the toolchain compiles -std=c++20, where std::expected is
    ///          unavailable, so this returns the CUDA-free C++20 stand-in
    ///          BuildResult<RunConfig> with the same value-or-Status interface
    ///          (build_result.hpp). The public error type available at this layer is
    ///          Status, and the fault category for every static violation here is
    ///          InvalidConfig — §10. The detailed reason is exposed via error_message()
    ///          for the app's stderr line, keeping printf out of the library, §10.)
    [[nodiscard]] BuildResult<RunConfig> build() const;

    /// The human-readable reason the LAST build() failed (empty if it succeeded or has
    /// not run). The app prints this to stderr; the library never prints (§10).
    [[nodiscard]] const std::string& error_message() const noexcept { return error_message_; }

private:
    // The accumulating RAW layers (string knobs not yet parsed). build() parses them.
    // We keep the merged CliArgs-shaped state so the precedence merge is a simple
    // field-wise override; build() does the one parse pass over the merged raw state.
    CliArgs merged_{};
    std::filesystem::path toml_path_{};   // recorded merge_file() path (empty ⇒ none)
    bool toml_requested_ = false;          // a non-empty merge_file() path was given

    mutable std::string error_message_;    // last build() failure reason (app prints it)
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_CONFIG_BUILDER_HPP
