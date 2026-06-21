// src/app/f2_dir_io.cpp
//
// The f2_blocks DIRECTORY reader (cli-bindings.md §4.3). PLAIN C++20, app-only, NO
// CUDA header (the §4 layering / arch-grep gate); it reaches the CUDA-FREE
// f2_disk_format.hpp only for the on-disk F2DiskHeader struct + the magic/version
// constants (the single home so the reader cannot drift from the writer's stamps).
#include "app/f2_dir_io.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "device/f2_disk_format.hpp"  // F2DiskHeader, kF2DiskMagic/Version/DtypeFp64, kF2DiskHeaderSize (CUDA-FREE)

namespace steppe::app {

namespace {

// Build the F2DirResult error arm with a fault Status + reason. A malformed cache
// dir is a fault the user must fix (architecture.md §10 fault taxonomy); the app
// maps the carried Status through exit_code_for, but a file/format problem is
// surfaced as InvalidConfig here (the engine never saw it — it is config-level).
[[nodiscard]] F2DirResult fail(Status status, std::string reason) {
    F2DirResult r;
    r.ok = false;
    r.status = status;
    r.error = std::move(reason);
    return r;
}

// Read <dir>/pops.txt: one label per non-empty line, trailing CR stripped (so a
// CRLF-authored sidecar reads cleanly), in file order (== P-axis index order). Blank
// lines are skipped (a trailing newline is not a phantom pop); a leading/trailing
// space is NOT trimmed inside a label (pop names can legitimately... they cannot, but
// we keep the label verbatim minus the line terminator so the map is exact).
[[nodiscard]] bool read_pops_txt(const std::filesystem::path& path,
                                 std::vector<std::string>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open pops.txt: " + path.string();
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF tolerance
        if (line.empty()) continue;                                 // skip blank lines
        out.push_back(line);
    }
    return true;
}

}  // namespace

F2DirResult read_f2_dir(const std::filesystem::path& dir) {
    namespace fs = std::filesystem;

    // The dir itself must exist and be a directory (a clear fault, not a crash).
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
        return fail(Status::InvalidConfig,
                    "--f2-dir is not a directory: " + dir.string());
    }

    const fs::path bin_path = dir / "f2.bin";
    const fs::path pops_path = dir / "pops.txt";

    // ---- f2.bin (STPF2BK1) -> F2BlockTensor --------------------------------------
    std::ifstream bin(bin_path, std::ios::binary);
    if (!bin) {
        return fail(Status::InvalidConfig, "cannot open f2.bin: " + bin_path.string());
    }

    device::F2DiskHeader hdr{};
    bin.read(reinterpret_cast<char*>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
    if (!bin) {
        return fail(Status::InvalidConfig,
                    "f2.bin is truncated (could not read the 64-byte header): " +
                        bin_path.string());
    }
    if (std::memcmp(hdr.magic, device::kF2DiskMagic, sizeof(hdr.magic)) != 0) {
        return fail(Status::InvalidConfig,
                    "f2.bin bad magic (expected STPF2BK1): " + bin_path.string());
    }
    if (hdr.version != device::kF2DiskVersion) {
        return fail(Status::InvalidConfig,
                    "f2.bin unsupported version " + std::to_string(hdr.version) +
                        " (this build reads version " +
                        std::to_string(device::kF2DiskVersion) + ")");
    }
    if (hdr.dtype != device::kF2DiskDtypeFp64) {
        return fail(Status::InvalidConfig,
                    "f2.bin unsupported dtype " + std::to_string(hdr.dtype) +
                        " (only FP64 is supported)");
    }
    if (hdr.P <= 0 || hdr.n_block <= 0) {
        return fail(Status::InvalidConfig,
                    "f2.bin degenerate shape P=" + std::to_string(hdr.P) +
                        " n_block=" + std::to_string(hdr.n_block));
    }

    const std::size_t P = static_cast<std::size_t>(hdr.P);
    const std::size_t nb = static_cast<std::size_t>(hdr.n_block);
    const std::size_t slab_elems = P * P * nb;  // f2 / vpair element counts

    F2Dir result;
    F2BlockTensor& t = result.f2;
    t.P = hdr.P;
    t.n_block = hdr.n_block;

    // f2 region (at hdr.f2_offset). Seek to the recorded offset rather than assuming
    // it follows the header, so a future header growth (reserved bytes) cannot break
    // the read — the offset is authoritative (f2_disk_format.hpp).
    t.f2.resize(slab_elems);
    bin.seekg(static_cast<std::streamoff>(hdr.f2_offset), std::ios::beg);
    bin.read(reinterpret_cast<char*>(t.f2.data()),
             static_cast<std::streamsize>(sizeof(double) * slab_elems));
    if (!bin) {
        return fail(Status::InvalidConfig,
                    "f2.bin truncated reading the f2 region (" +
                        std::to_string(slab_elems) + " doubles): " + bin_path.string());
    }

    // vpair region (at hdr.vpair_offset). The fit reads block_sizes, not vpair (OQ-3),
    // but the device upload copies both slabs, so carry vpair for a faithful round-trip.
    t.vpair.resize(slab_elems);
    bin.seekg(static_cast<std::streamoff>(hdr.vpair_offset), std::ios::beg);
    bin.read(reinterpret_cast<char*>(t.vpair.data()),
             static_cast<std::streamsize>(sizeof(double) * slab_elems));
    if (!bin) {
        return fail(Status::InvalidConfig,
                    "f2.bin truncated reading the vpair region: " + bin_path.string());
    }

    // block_sizes trailer (at hdr.block_sizes_offset; n_block int32).
    std::vector<std::int32_t> sizes32(nb);
    bin.seekg(static_cast<std::streamoff>(hdr.block_sizes_offset), std::ios::beg);
    bin.read(reinterpret_cast<char*>(sizes32.data()),
             static_cast<std::streamsize>(sizeof(std::int32_t) * nb));
    if (!bin) {
        return fail(Status::InvalidConfig,
                    "f2.bin truncated reading the block_sizes trailer: " +
                        bin_path.string());
    }
    t.block_sizes.assign(sizes32.begin(), sizes32.end());

    // ---- pops.txt -> the P labels (the name<->index map) -------------------------
    std::string perr;
    if (!read_pops_txt(pops_path, result.pop_labels, perr)) {
        return fail(Status::InvalidConfig, std::move(perr));
    }
    if (result.pop_labels.size() != P) {
        return fail(Status::InvalidConfig,
                    "pops.txt has " + std::to_string(result.pop_labels.size()) +
                        " labels but f2.bin has P=" + std::to_string(P) +
                        " populations (the name<->index map must cover the whole P axis)");
    }

    F2DirResult ret;
    ret.ok = true;
    ret.status = Status::Ok;
    ret.dir = std::move(result);
    return ret;
}

}  // namespace steppe::app
