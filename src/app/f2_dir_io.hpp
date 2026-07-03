// src/app/f2_dir_io.hpp
//
// The reader for an f2_blocks directory — the on-disk seam that lets steppe
// precompute the f2 statistics once and fit many models against them. A directory
// holds f2.bin (the numeric payload), pops.txt (the P labels), and an optional
// meta.json; this header is the read side only. Plain C++20, no CUDA.
//
// Reference: docs/reference/src_app_f2_dir_io.hpp.md
#ifndef STEPPE_APP_F2_DIR_IO_HPP
#define STEPPE_APP_F2_DIR_IO_HPP

#include <filesystem>
#include <string>
#include <vector>

#include "steppe/error.hpp"
#include "steppe/fstats.hpp"

namespace steppe::app {

// The loaded f2_blocks dir — reference §3
struct F2Dir {
    steppe::F2BlockTensor f2;
    std::vector<std::string> pop_labels;
};

// The read_f2_dir return value — reference §4
struct F2DirResult {
    bool ok = false;
    Status status = Status::Ok;
    std::string error;
    F2Dir dir;
};

// The reader entry point — reference §5
[[nodiscard]] F2DirResult read_f2_dir(const std::filesystem::path& dir);

}  // namespace steppe::app

#endif  // STEPPE_APP_F2_DIR_IO_HPP
