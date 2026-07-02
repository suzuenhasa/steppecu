// src/app/cli_parse.hpp
//
// The CLI11 parser adapter (app-only, plain CXX). It builds the `steppe` CLI11 app
// (subcommands matching cli-bindings.md §4.1), binds each subcommand's flags into a
// steppe::config::CliArgs, and runs the ConfigBuilder precedence merge + build()
// (architecture.md §9). This is the ONLY place CLI11 is named — it is PRIVATE to the
// app subtree (the §4 layering rule; cli-bindings.md §6.1). NO CUDA header here.
//
// Each subcommand parses + validates its config, then dispatches to the subcommand's
// run_*_command (the real GPU compute) — qpadm / qpwave / qpadm-rotate / extract-f2 and
// the f-stat / qpGraph / dates / qpfstats commands are ALL wired to their GPU compute;
// no subcommand is a not-yet-implemented scaffold (see cli_parse.cpp's dispatch flow).
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
