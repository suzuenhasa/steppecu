// src/app/cmd_common.hpp
//
// Shared CLI-handler helpers. `require_first_gpu` is the one no-GPU guard every
// `steppe <cmd>` runs before touching the device: it reports the standard
// "no CUDA device available" diagnostic and never throws (each caller's guard
// sits inside a try{} that would otherwise remap a throw to a different exit
// code). Header-only, app-layer, CUDA-free.
//
// Reference: docs/reference/src_app_cmd_common.hpp.md
#ifndef STEPPE_APP_CMD_COMMON_HPP
#define STEPPE_APP_CMD_COMMON_HPP

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "core/config/run_config.hpp"
#include "device/resources.hpp"

namespace steppe::app {

// require_first_gpu — the standard no-GPU guard shared by every CLI handler.
// Returns true when a CUDA device is present; otherwise prints the canonical
// diagnostic and returns false. Must not throw (callers guard inside a try{}).
[[nodiscard]] inline bool require_first_gpu(const device::Resources& resources,
                                            const char* prefix) {
    if (!resources.gpus.empty()) {
        return true;
    }
    std::fprintf(stderr,
                 "steppe %s: no CUDA device available (steppe is a GPU product; a "
                 "CUDA-capable GPU is required)\n",
                 prefix);
    return false;
}

// build_quartet_names — shared quartet name-table builder for `f4` and `qpdstat`.
// Accepts either row-aligned --pop1..--pop4 columns or a flat --pops list in
// groups of 4, emitting one {p1,p2,p3,p4} row per quartet. `tool` names the CLI
// command and `group_noun` names a group of four ("quartet" vs "quadruple") in
// the "multiple of 4" diagnostic; the two are the only per-command differences.
[[nodiscard]] inline bool build_quartet_names(
    const config::RunConfig& config,
    std::vector<std::array<std::string, 4>>& quartets, std::string& err,
    const char* tool, const char* group_noun) {
    const auto& p1 = config.pop1();
    const auto& p2 = config.pop2();
    const auto& p3 = config.pop3();
    const auto& p4 = config.pop4();
    const bool have_cols = !p1.empty() || !p2.empty() || !p3.empty() || !p4.empty();

    if (have_cols) {
        const std::size_t n = p1.size();
        if (p2.size() != n || p3.size() != n || p4.size() != n) {
            err = "--pop1/--pop2/--pop3/--pop4 must be ROW-ALIGNED (same length); got " +
                  std::to_string(p1.size()) + "/" + std::to_string(p2.size()) + "/" +
                  std::to_string(p3.size()) + "/" + std::to_string(p4.size());
            return false;
        }
        if (n == 0) { err = "--pop1/--pop2/--pop3/--pop4 are empty"; return false; }
        quartets.reserve(n);
        for (std::size_t k = 0; k < n; ++k)
            quartets.push_back({p1[k], p2[k], p3[k], p4[k]});
        return true;
    }

    const auto& pops = config.pops();
    if (pops.empty()) {
        err = std::string(tool) +
              " needs quartets: either --pop1/--pop2/--pop3/--pop4 (row-aligned) or "
              "--pops p1,p2,p3,p4[,...] (names in groups of 4)";
        return false;
    }
    if (pops.size() % 4 != 0) {
        err = "--pops for " + std::string(tool) + " must be a multiple of 4 names (each " +
              group_noun + " is p1,p2,p3,p4); got " + std::to_string(pops.size());
        return false;
    }
    const std::size_t n = pops.size() / 4;
    quartets.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        quartets.push_back({pops[4 * k + 0], pops[4 * k + 1],
                            pops[4 * k + 2], pops[4 * k + 3]});
    return true;
}

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_COMMON_HPP
