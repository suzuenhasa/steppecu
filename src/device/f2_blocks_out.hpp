// src/device/f2_blocks_out.hpp
//
// M5 F2BlocksOut — THE unified, adaptive precompute result (CUDA-FREE seam). The
// result lives in EXACTLY ONE tier (Resident / HostRam / Disk); the fit + the parity
// test consume it ONLY through the tier-agnostic read-back accessors (they never branch
// on `tier` for the numeric read-back — the accessor does). CUDA-FREE like
// device_f2_blocks.hpp: it holds the move-only DeviceF2Blocks by value (TIER 0,
// UNCHANGED), an F2BlockTensor by value (TIER 1), and a CUDA-free DiskF2Blocks
// descriptor (TIER 2). The accessors are defined in cuda/f2_blocks_out.cu (the Resident
// arm needs CUDA for the D2H; HostRam/Disk are plain host/file I/O).
//
// PARITY (architecture.md §12): F2BlocksOut::to_host() is memcmp-bit-identical across
// all tiers and to the device-resident DeviceF2Blocks::to_host() / the single-GPU
// reference. The tier choice is OUT-OF-BAND (it lives here, never on the numeric
// F2BlockTensor) — the same discipline as last_combine_path.
#ifndef STEPPE_DEVICE_F2_BLOCKS_OUT_HPP
#define STEPPE_DEVICE_F2_BLOCKS_OUT_HPP

#include <cstddef>
#include <cstdio>   // std::FILE for the disk descriptor read handle
#include <memory>
#include <string>
#include <vector>

#include "steppe/fstats.hpp"               // F2BlockTensor (TIER 1, host materialization)
#include "device/device_f2_blocks.hpp"     // DeviceF2Blocks (TIER 0, UNCHANGED)
#include "device/tier_select.hpp"          // OutputTier

namespace steppe::device {

/// Per-block f2/vpair slab element count = P² (the within-block [P × P] column-major
/// slab, i + P·j). Single home for the block-major slab shape so the read-back paths
/// (F2BlocksOut::read_block_to_host / to_host) and F2BlocksOut::size() cannot drift
/// from it (DRY; NAMING-STYLE-STANDARD §2.5 single-source; group-5 5.3). Widens `P`
/// to std::size_t BEFORE the multiply so the product never overflows a 32-bit int
/// (P² and P²·n_block reach ~10^10 elements at scale).
[[nodiscard]] inline std::size_t slab_elems(int P) noexcept {
    return static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
}

/// Move-only deleter for the on-disk read handle: closes the FILE* and, on a nonzero
/// std::fclose status, routes ONE teardown-warning line through the §7 sink instead of
/// silently discarding it (the // STEPPE_LOG_WARN teardown promise this unit always
/// carried; group-16 16.1). Stateless/empty -> EBO makes the owning unique_ptr the size
/// of a raw FILE* (no overhead). The std::FILE-pointer null check is defensive (a
/// unique_ptr only invokes the deleter on a non-null managed pointer). Verified vs CUDA
/// 13.x toolchain C++ stdlib: std::fclose (<cstdio>) returns 0 on success / EOF (nonzero)
/// on failure, so `!= 0` is the failure test. The body is in cuda/f2_blocks_out.cu (the
/// debug-arm STEPPE_LOG_WARN pulls <cstdio>; compiles out under NDEBUG, no unused param).
struct FileCloser {
    void operator()(std::FILE* f) const noexcept;
};

/// Binary on-disk f2_blocks cache descriptor (TIER 2). Holds the path + parsed header
/// shape + an open read handle for slab read-back. The fit + parity test pread a block
/// by offset (f2_disk_format.hpp byte layout). Move-only (it owns a FILE* via the
/// move-only unique_ptr below — the deleter supplies the freeing dtor + null-on-move, so
/// all special members are =default; NAMING-STYLE-STANDARD §2.12 RAII / §5 non-goal #9).
/// The read goes through F2BlocksOut::read_block_to_host / to_host, never raw here.
struct DiskF2Blocks {
    DiskF2Blocks() = default;
    ~DiskF2Blocks() = default;
    DiskF2Blocks(DiskF2Blocks&&) noexcept = default;
    DiskF2Blocks& operator=(DiskF2Blocks&&) noexcept = default;
    DiskF2Blocks(const DiskF2Blocks&) = delete;
    DiskF2Blocks& operator=(const DiskF2Blocks&) = delete;

    std::string path;
    int P = 0;
    int n_block = 0;
    std::vector<int> block_sizes;          ///< length n_block (from header trailer).
    /// open read-only handle (DiskSink::finish reopens). PUBLIC POD field — no trailing
    /// underscore (§2.4a); read raw via `read_handle.get()`, truthiness via `if (handle)`.
    std::unique_ptr<std::FILE, FileCloser> read_handle;
};

/// THE unified precompute result — the result lives in EXACTLY ONE tier. Move-only
/// (it owns either move-only DeviceF2Blocks, or vectors, or a FILE*). The fit and the
/// parity test consume it ONLY through the tier-agnostic accessors below.
class F2BlocksOut {
public:
    F2BlocksOut() = default;
    ~F2BlocksOut() = default;
    F2BlocksOut(F2BlocksOut&&) noexcept = default;
    F2BlocksOut& operator=(F2BlocksOut&&) noexcept = default;
    F2BlocksOut(const F2BlocksOut&) = delete;
    F2BlocksOut& operator=(const F2BlocksOut&) = delete;

    OutputTier tier = OutputTier::Resident;
    int P = 0;
    int n_block = 0;
    std::vector<int> block_sizes;

    // EXACTLY ONE of these is engaged, per `tier`:
    DeviceF2Blocks resident;   ///< tier==Resident — the EXISTING handle, UNCHANGED.
    F2BlockTensor  host;       ///< tier==HostRam — full materialized host tensor.
    DiskF2Blocks   disk;       ///< tier==Disk    — on-disk cache descriptor.

    // ---- Tier-agnostic READ-BACK ACCESSORS (the fit + parity test surface) ----

    /// Materialize the WHOLE result to a host F2BlockTensor, BIT-IDENTICAL across all
    /// tiers (architecture.md §12). Resident -> resident.to_host() (the existing ONLY
    /// D2H). HostRam -> a copy of `host`. Disk -> read every block back via pread. THE
    /// parity test calls this and memcmps vs the reference.
    [[nodiscard]] F2BlockTensor to_host() const;

    /// Read ONE block's [P²] f2 slab + [P²] vpair slab (column-major i+P·j) into the
    /// caller's buffers (each length >= P²). Tier-agnostic: Resident -> D2H of the
    /// block's slab; HostRam -> memcpy from host.f2/vpair at P²·b; Disk -> two preads
    /// at the f2_disk_format.hpp byte offsets. This is the FIT's tile reader.
    void read_block_to_host(int b, double* f2_slab_out, double* vpair_slab_out) const;

    [[nodiscard]] std::size_t size() const noexcept {
        return slab_elems(P) * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
    }
};

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_F2_BLOCKS_OUT_HPP
