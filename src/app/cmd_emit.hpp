// src/app/cmd_emit.hpp
//
// The shared emit helper: every `steppe <cmd>` routes its CSV/TSV/JSON output through
// here to a file or stdout, then verifies the write so a torn output fails loudly instead
// of exiting 0. Header-only, app-layer, CUDA-free host stream I/O.
//
// Reference: docs/reference/src_app_cmd_emit.hpp.md
#ifndef STEPPE_APP_CMD_EMIT_HPP
#define STEPPE_APP_CMD_EMIT_HPP

#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>

#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "core/config/run_config.hpp"

namespace steppe::app {

// finish_emit — the post-write integrity gate — reference §3
[[nodiscard]] inline std::optional<int> finish_emit(std::ostream& os, const char* prefix,
                                                    const std::string& dest) {
    namespace cfg = steppe::config;
    os.flush();
    if (!os.good()) {
        std::fprintf(stderr,
                     "steppe %s: write failed (disk full / short write / closed pipe): %s\n",
                     prefix, dest.c_str());
        return cfg::kExitIoError;
    }
    return std::nullopt;
}

// emit_to_destination — routing output to a file or the screen — reference §2
template <typename Write>
[[nodiscard]] std::optional<int> emit_to_destination(const config::RunConfig& config,
                                                     const char* prefix, Write&& write) {
    namespace cfg = steppe::config;
    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe %s: unknown --format '%s' (csv|tsv|json)\n", prefix,
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }
    if (config.out_file().empty()) {
        write(std::cout, fmt);
        return finish_emit(std::cout, prefix, "stdout");
    }
    std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "steppe %s: cannot open --out file: %s\n", prefix,
                     config.out_file().c_str());
        return cfg::kExitIoError;
    }
    write(out, fmt);
    return finish_emit(out, prefix, config.out_file());
}

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_EMIT_HPP
