// src/app/cli_parse.hpp
//
// CLI11 parser adapter (app-only, plain C++). Builds the `steppe` app's subcommands,
// binds each one's flags into a steppe::config::CliArgs, and runs the ConfigBuilder
// precedence merge + build(). This is the only place CLI11 is named — it stays private
// to the app subtree, so no CUDA header appears here. Each subcommand validates its
// config and dispatches to a run_*_command that does the GPU compute.
#ifndef STEPPE_APP_CLI_PARSE_HPP
#define STEPPE_APP_CLI_PARSE_HPP

namespace steppe::app {

/// Parse argv, dispatch the selected subcommand, and return the process exit code
/// (steppe::config::CliExitCode). Owns all CLI stdout/stderr; the library itself never
/// prints. --version / --help are handled by CLI11 and exit through its own path. A
/// config-validation failure prints the ConfigBuilder reason to stderr and returns
/// kExitInvalidConfig.
[[nodiscard]] int run_cli(int argc, char** argv);

}  // namespace steppe::app

#endif  // STEPPE_APP_CLI_PARSE_HPP
