// src/app/cmd_cache.hpp
//
// Native (host-only, no-GPU) f2-cache inspection — the thin C++ mirror of the
// steppe-cache Python tool. Reads the STPF2BK1 header directly (never the f2/vpair
// payload) and re-hashes f2.bin/pops.txt against the stored content-address. No
// device, no build_config: dispatched straight from the `cache` subcommand.
//
// Reference: docs/reference/src_app_cmd_cache.hpp.md
#ifndef STEPPE_APP_CMD_CACHE_HPP
#define STEPPE_APP_CMD_CACHE_HPP

#include <string>

namespace steppe::app {

// `steppe cache ls [ROOT]` — tabulate STPF2BK1 caches under ROOT (header-only).
[[nodiscard]] int run_cache_ls(const std::string& root);
// `steppe cache show <DIR>` — header facts + integrity mark + raw meta.json.
[[nodiscard]] int run_cache_show(const std::string& dir);
// `steppe cache verify <DIR>` — re-hash f2.bin/pops.txt vs the stored id.
[[nodiscard]] int run_cache_verify(const std::string& dir, bool check_sources);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_CACHE_HPP
