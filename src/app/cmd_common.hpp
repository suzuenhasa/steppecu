// src/app/cmd_common.hpp
//
// Shared CLI-handler helpers. `require_first_gpu` is the one no-GPU guard every
// `steppe <cmd>` runs before touching the device: it reports the standard
// "no CUDA device available" diagnostic and never throws (each caller's guard
// sits inside a try{} that would otherwise remap a throw to a different exit
// code). Header-only, app-layer, CUDA-free.
#ifndef STEPPE_APP_CMD_COMMON_HPP
#define STEPPE_APP_CMD_COMMON_HPP

#include <cstdio>

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

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_COMMON_HPP
