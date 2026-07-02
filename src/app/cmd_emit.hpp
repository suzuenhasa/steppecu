// src/app/cmd_emit.hpp
//
// The shared emit-to-destination helper: every `steppe <cmd>` routes its CSV/TSV/JSON output
// through this one function. It parses --format once, opens --out (binary+trunc) or selects
// stdout, invokes the command's write(os, fmt) serializer, then flushes and checks good() so a
// torn / short write (full disk, quota, closed pipe) returns kExitIoError instead of exiting 0
// with a truncated file. For a product whose deliverable is files, an exit-0 truncated output
// is a real correctness hole.
//
// Plain host C++20, no CUDA: this is stdout/stream I/O. The command layer owns stdout/stderr;
// the library never prints.
#ifndef STEPPE_APP_CMD_EMIT_HPP
#define STEPPE_APP_CMD_EMIT_HPP

#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>

#include "app/result_emit.hpp"          // OutputFormat / parse_output_format
#include "core/config/exit_code.hpp"    // kExitIoError / kExitInvalidConfig
#include "core/config/run_config.hpp"   // config::RunConfig

namespace steppe::app {

/// Flush `os` and verify its post-write state. On a torn / short write (full disk, quota,
/// closed pipe / EPIPE — the failbit/badbit the ofstream destructor would otherwise swallow)
/// this logs the failure naming `dest` and returns kExitIoError; a clean write returns nullopt.
/// The flush() is load-bearing: a small buffered write has not reached the fd until it flushes,
/// so good() must be checked after the flush, not left to the destructor.
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

/// Route a command's serialized output to the --out file (binary+trunc) or stdout, running the
/// post-write integrity check. Parses config.format() once (kExitInvalidConfig on an unknown
/// token), opens/selects the destination (kExitIoError on open failure), invokes write(os, fmt),
/// then verifies the stream. Returns std::nullopt once cleanly emitted; otherwise the exit code
/// the caller must propagate: `if (auto rc = emit_to_destination(...)) return *rc;`. `prefix` is
/// the "steppe <prefix>:" diagnostic tag; `write` is invoked as write(std::ostream&, OutputFormat).
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
