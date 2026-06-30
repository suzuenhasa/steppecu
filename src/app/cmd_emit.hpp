// src/app/cmd_emit.hpp
//
// The shared emit-to-destination helper (B1 — §12 I/O fault taxonomy). EVERY `steppe <cmd>`
// routes its CSV/TSV/JSON output through this ONE function: parse --format once, open --out
// (binary|trunc) or select stdout, invoke the command's `write(os, fmt)` serializer, then
// FLUSH and check good() so a torn / short write (full disk, quota, closed pipe / EPIPE)
// returns kExitIoError instead of silently exiting 0 with a truncated file. For a product
// whose deliverable IS files, an exit-0 truncated output is a real correctness hole.
//
// Lifted from cmd_qpgraph.cpp's emit_to_destination (already the right shape:
// open->write->route, std::optional<int> = nullopt-ok-else-exit-code) and hardened with the
// post-write verification per docs/kimiactions/01-open-worth-doing.md §B1.
//
// PLAIN C++20, app-only, NO CUDA header (the §4 layering): this is host stdout/stream I/O.
// main()/the command owns stdout/stderr (architecture.md §10 — the library never prints).
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
/// closed pipe / EPIPE — the failbit/badbit the ofstream dtor would otherwise swallow) this
/// logs "steppe <prefix>: write failed ..." naming `dest` and returns kExitIoError; on a
/// clean write it returns std::nullopt. The flush() is load-bearing: a small buffered write
/// has not reached the fd until flush, so good() must be checked AFTER flush (not at dtor).
/// The single post-write integrity gate every emit site shares (B1).
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

/// Route a command's serialized output to the --out file (binary|trunc) or stdout, with the
/// post-write integrity check (finish_emit). Parses config.format() once (kExitInvalidConfig
/// on an unknown token — the existing per-command defensive re-map, kept inside the helper),
/// opens/selects the destination (kExitIoError on open failure), invokes write(os, fmt), then
/// verifies the stream. Returns std::nullopt once cleanly emitted; otherwise the exit code the
/// caller must propagate (`if (auto rc = emit_to_destination(...)) return *rc;`). `prefix` is
/// the "steppe <prefix>:" diagnostic tag; `write` is invoked as write(std::ostream&,
/// OutputFormat).
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
