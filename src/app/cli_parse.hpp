// src/app/cli_parse.hpp
//
// The CLI11 parser adapter for the `steppe` CLI: declares run_cli, which parses argv,
// dispatches the selected subcommand, and returns the process exit code. App-only and
// CUDA-free — the one place CLI11 is named, kept private to the app subtree.
#ifndef STEPPE_APP_CLI_PARSE_HPP
#define STEPPE_APP_CLI_PARSE_HPP

namespace steppe::app {

[[nodiscard]] int run_cli(int argc, char** argv);

}  // namespace steppe::app

#endif  // STEPPE_APP_CLI_PARSE_HPP
