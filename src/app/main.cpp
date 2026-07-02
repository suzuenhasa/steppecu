// src/app/main.cpp — the steppe CLI entry point.
//
// Plain C++20 host target: no CUDA toolkit header belongs anywhere under src/app.
// main() owns the process and stdout/stderr; the library itself never prints. The
// top-level catch turns any fault (including a thrown io or CUDA-runtime error) into a
// clean nonzero exit rather than an uncaught abort.
#include <cstdio>
#include <exception>

#include "app/cli_parse.hpp"
#include "app/exit_code_for_caught.hpp"  // maps a caught error to an exit code (device OOM -> its own code)
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
