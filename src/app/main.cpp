// src/app/main.cpp — the steppe CLI entry point.
//
// Plain C++20 host target (no CUDA): main() owns the process and stdout/stderr,
// delegates all work to app::run_cli, and maps any escaped exception to a clean
// nonzero exit code.
#include <cstdio>
#include <exception>

#include "app/cli_parse.hpp"
#include "app/exit_code_for_caught.hpp"
#include "core/config/exit_code.hpp"

int main(int argc, char** argv) {
    try {
        return steppe::app::run_cli(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe: fatal: %s\n", e.what());
        return steppe::app::exit_code_for_caught(e);
    } catch (...) {
        std::fprintf(stderr, "steppe: fatal: unknown error\n");
        return steppe::config::kExitRuntimeError;
    }
}
