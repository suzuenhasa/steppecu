// src/app/f2_dir_io.hpp
//
// Reader for an f2_blocks directory — the read side of the precompute-once /
// fit-many seam, in an AT2-shaped on-disk interchange. A <dir> holds:
//
//   f2.bin     STPF2BK1 numeric payload (see src/device/f2_disk_format.hpp):
//              header(64B) | f2[P*P*nb] | vpair[P*P*nb] | block_sizes[nb int32]. Its
//              i + P*j + P*P*b layout is byte-identical to F2BlockTensor's, so a
//              whole-file read is a memcpy-equivalent load into the host tensor.
//   pops.txt   the P population labels, one per line, in P-axis index order — the
//              name<->index map the compute seam lacks (that seam is index-only).
//   meta.json  provenance; OPTIONAL and not parsed here (reading needs only f2.bin
//              + pops.txt), so a missing meta.json is not a fault.
//
// Plain C++20, app-only, no CUDA header: reaches only the CUDA-free
// f2_disk_format.hpp for the on-disk header struct.
#ifndef STEPPE_APP_F2_DIR_IO_HPP
#define STEPPE_APP_F2_DIR_IO_HPP

#include <filesystem>
#include <string>
#include <vector>

#include "steppe/error.hpp"   // steppe::Status
#include "steppe/fstats.hpp"  // steppe::F2BlockTensor

namespace steppe::app {

/// The materialized f2_blocks dir: the host f2 tensor + its P pop labels (index
/// order). Returned by read_f2_dir on success.
struct F2Dir {
    steppe::F2BlockTensor f2;            ///< the host tensor read from f2.bin.
    std::vector<std::string> pop_labels; ///< P labels in P-axis index order (pops.txt).
};

/// Result of read_f2_dir: the loaded F2Dir on success, or a fault Status plus a
/// human-readable reason on failure. The app prints the reason to stderr; the
/// library never prints. A malformed dir carries Status::InvalidConfig — a bad
/// cache dir is a config-level fault the user must fix, not an internal error.
struct F2DirResult {
    bool ok = false;
    Status status = Status::Ok;  ///< Ok on success; a fault category on failure.
    std::string error;           ///< empty on success; the reason on failure.
    F2Dir dir;                   ///< valid only when ok.
};

/// Read an f2_blocks dir: parse f2.bin (STPF2BK1) into a host F2BlockTensor and
/// pops.txt into the P pop labels (index order), checking that pops.txt's line count
/// == the tensor's P (the name<->index map must cover the whole P axis). A
/// missing/unreadable f2.bin or pops.txt, a bad magic/version, a truncated payload,
/// or a line count != P is a fault (ok=false with a reason). meta.json is not
/// required.
[[nodiscard]] F2DirResult read_f2_dir(const std::filesystem::path& dir);

}  // namespace steppe::app

#endif  // STEPPE_APP_F2_DIR_IO_HPP
