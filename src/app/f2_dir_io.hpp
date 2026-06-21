// src/app/f2_dir_io.hpp
//
// The f2_blocks DIRECTORY reader (cli-bindings.md §4.3, §3) — the READ side of the
// AT2-shaped interchange the precompute-once / fit-many seam consumes (ADR-0005). A
// <dir> is:
//
//   <dir>/f2.bin     the STPF2BK1 numeric payload (src/device/f2_disk_format.hpp):
//                    header(64B) | f2[P*P*nb] | vpair[P*P*nb] | block_sizes[nb int32],
//                    block-major outer, column-major i+P*j within — BYTE-IDENTICAL to
//                    F2BlockTensor's i + P*j + P*P*b layout (so a whole-file read is a
//                    memcpy-equivalent into the host tensor).
//   <dir>/pops.txt   the P population labels, ONE PER LINE, in P-axis index order —
//                    the name<->index map the compute seam lacks (cli-bindings.md §1:
//                    "the compute seam is INDEX-ONLY; names are an app concern").
//   <dir>/meta.json  provenance (cli-bindings.md §4.3); OPTIONAL for the READ path
//                    (the engine needs only f2.bin + pops.txt). M(cli-1) does not
//                    parse it; M(cli-4) writes it. A missing meta.json is not a fault.
//
// This is the WRITER's READ-only counterpart: the standalone STPF2BK1 write/read
// round-trip + the dir WRITER is M(cli-4) (cli-bindings.md §3 "Honest cache gap",
// §8). M(cli-1) ships ONLY the reader (the fixtures are written into a dir by the
// test). PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate):
// it reaches f2_disk_format.hpp (a CUDA-FREE header) for the on-disk header struct,
// nothing more.
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

/// Result of read_f2_dir: the loaded F2Dir on success, or a fault Status + a
/// human-readable reason on failure (the app prints the reason to stderr; the
/// library never prints — architecture.md §10). The fault is always a FILE/FORMAT
/// fault (Status::IoError mapping is the app's exit_code_for fallthrough →
/// kExitIoError; the carried Status is InvalidConfig for a malformed dir per §10's
/// fault taxonomy — a bad cache dir is a config-level fault the user must fix).
struct F2DirResult {
    bool ok = false;
    Status status = Status::Ok;  ///< Ok on success; a fault category on failure.
    std::string error;           ///< empty on success; the reason on failure.
    F2Dir dir;                   ///< valid only when ok.
};

/// Read an f2_blocks dir: parse <dir>/f2.bin (STPF2BK1) into a host F2BlockTensor and
/// <dir>/pops.txt into the P pop labels (index order), validating that
/// pops.txt's line count == the tensor's P (the name<->index map must cover the
/// whole P axis). A missing/unreadable f2.bin or pops.txt, a bad magic/version, a
/// truncated payload, or a pops.txt line-count != P is a fault (ok=false with the
/// reason). meta.json is NOT required by the read path (M(cli-1)).
[[nodiscard]] F2DirResult read_f2_dir(const std::filesystem::path& dir);

}  // namespace steppe::app

#endif  // STEPPE_APP_F2_DIR_IO_HPP
