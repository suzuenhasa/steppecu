// src/app/cli_parse.hpp
//
// The CLI11 parser adapter (app-only, plain CXX). It builds the `steppe` CLI11 app
// (subcommands matching cli-bindings.md §4.1), binds each subcommand's flags into a
// steppe::config::CliArgs, and runs the ConfigBuilder precedence merge + build()
// (architecture.md §9). This is the ONLY place CLI11 is named — it is PRIVATE to the
// app subtree (the §4 layering rule; cli-bindings.md §6.1). NO CUDA header here.
//
// M(cli-0) is the SCAFFOLD: every subcommand parses + validates its config, then the
// fit subcommands (qpadm/qpwave/qpadm-rotate) and extract-f2 print a "not yet
// implemented" notice and exit cleanly — EXCEPT the compute itself, which M(cli-1+)
// fills in. No GPU work happens in this milestone.
#ifndef STEPPE_APP_CLI_PARSE_HPP
#define STEPPE_APP_CLI_PARSE_HPP

namespace steppe::app {

/// Parse argv, dispatch the selected subcommand, and return the process exit code
/// (steppe::config::CliExitCode). Owns all stdout/stderr for the CLI (the library
/// never prints — architecture.md §10). --version / --help are handled by CLI11 and
/// exit through CLI11_PARSE's app.exit() path. A config-validation failure prints the
/// ConfigBuilder reason to stderr and returns kExitInvalidConfig.
[[nodiscard]] int run_cli(int argc, char** argv);

}  // namespace steppe::app

#endif  // STEPPE_APP_CLI_PARSE_HPP
