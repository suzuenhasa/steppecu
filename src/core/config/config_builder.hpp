// src/core/config/config_builder.hpp
//
// ConfigBuilder — the only mutable config type: fold the config layers in
// precedence order, then build() validates once and freezes an immutable RunConfig.
// Host-only and CUDA-free by contract, so it unit-tests with no GPU.
//
// Reference: docs/reference/src_core_config_config_builder.hpp.md
#ifndef STEPPE_CORE_CONFIG_CONFIG_BUILDER_HPP
#define STEPPE_CORE_CONFIG_CONFIG_BUILDER_HPP

#include <filesystem>
#include <string>

#include "steppe/error.hpp"
#include "core/config/build_result.hpp"
#include "core/config/cli_args.hpp"
#include "core/config/run_config.hpp"

namespace steppe::config {

// Mutable, layered config accumulator — reference §1
class ConfigBuilder {
public:
    ConfigBuilder() = default;

    // The layer API — fold each source in precedence order — reference §3
    ConfigBuilder& with_defaults();

    ConfigBuilder& merge_file(const std::filesystem::path& path);

    ConfigBuilder& merge_env();

    ConfigBuilder& merge_cli(const CliArgs& args);

    // build() — validate once, parse + range-check the raw knobs, freeze — reference §4
    [[nodiscard]] BuildResult<RunConfig> build();

    // error_message() — last build() failure reason; the app prints it — reference §8
    [[nodiscard]] const std::string& error_message() const noexcept { return error_message_; }

private:
    // Internal accumulating state; not thread-safe — reference §8
    CliArgs merged_{};
    std::filesystem::path toml_path_{};
    bool toml_requested_ = false;

    std::string error_message_;
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_CONFIG_BUILDER_HPP
