// src/app/main.cpp — the steppe CLI entry point (M(cli-0) scaffold).
//
// PLAIN C++20 host target (architecture.md §4, §6.1; cli-bindings.md §6.1). NO CUDA
// toolkit header here or anywhere in src/app — the arch-grep gate enforces it and a
// leaked CUDA runtime header would hard-fail this host compile. main() owns process +
// stdout/stderr (the library never prints — architecture.md §10); all parsing,
// dispatch, and the Status->exit-code mapping live in app::run_cli / the core config
// layer. A top-level catch turns any unexpected fault into a clean nonzero exit so a
// thrown io/CUDA-runtime error never aborts the process uncaught.
#include <cstdio>
#include <exception>

#include "app/cli_parse.hpp"
#include "app/exit_code_for_caught.hpp"  // exit_code_for_caught (5 -> 3 on a real device OOM, B2)
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
