// src/core/internal/launch_config.hpp
//
// THE single home of launch-grid math (architecture.md §4 line 466, §8
// DRY-internals table line 525 "Launch math | internal/launch_config.hpp";
// ROADMAP §4, §5). Kernel files never recompute grid math.
//
// Contents (the §4/§8 launch-config helpers, moved here verbatim out of
// f2_estimator.hpp where they had de-facto landed while this file was still a
// phantom the spec cited — cleanup X-4):
//
//   cdiv(n, block)    — ceiling division, the launch-grid building block, in an
//                       `int` overload and a `long` overload (the SNP/`M` axis
//                       exceeds 2^31, so it must use the `long` form — views.hpp).
//   kMaxGridX/Y/Z     — the hardware grid-dimension limits (architecture.md §7;
//                       CUDA C++ Programming Guide, Compute Capabilities): x may
//                       reach 2^31−1, but y and z are CAPPED at 65 535 on every
//                       compute capability (incl. Blackwell sm_120). Named here so
//                       the launch wrappers can fail-fast on an over-limit axis
//                       (cleanup X-7/B6) instead of letting the launch return an
//                       opaque cudaErrorInvalidConfiguration.
//   grid_for(n, block, max_grid) — number of `block`-sized tiles to cover `n`
//                       along one axis, defaulting to the project's standard SQUARE
//                       f2 edge `kCdivBlock` and the y/z 65 535 limit. It
//                       debug-asserts the extent fits `max_grid`, so a square f2
//                       [P × P] / [P × s_pad] launch routed through it is checked.
//   grid_z_extent(n)  — the batch (z-axis) extent for the M4 strided-batched
//                       gather/scatter launches, which set `gridDim.z = n_in_group`
//                       DIRECTLY (NOT via `grid_for`, so a `grid_for` clamp alone
//                       would miss them — cleanup X-7/B6). Asserts `1 ≤ n ≤ kMaxGridZ`
//                       and returns the unsigned extent; the backend tiles the batch
//                       so `n_in_group ≤ kMaxGridZ` always holds.
//   kDecodeBlockX/Y   — the decode kernel's 2-D block dims (a NON-square, warp-
//                       justified 32×8), named here per ROADMAP §4 rather than as
//                       bare literals re-picked inside the kernel TU.
//
// These are pure `STEPPE_HD constexpr` integer arithmetic + named constants:
// host-callable, device-callable, and fold at compile time. CUDA-FREE-compilable
// (it pulls in only the host/device macro and config.hpp), so it lives in
// `core/internal/` and is consumed via the steppe::core_internal INTERFACE target
// by both the host-pure `core` and the device kernels (architecture.md §4, §8).
#ifndef STEPPE_CORE_INTERNAL_LAUNCH_CONFIG_HPP
#define STEPPE_CORE_INTERNAL_LAUNCH_CONFIG_HPP

#include "core/internal/host_device.hpp"  // STEPPE_HD, STEPPE_ASSERT
#include "steppe/config.hpp"              // kCdivBlock

namespace steppe::core {

// ===========================================================================
// Hardware grid-dimension limits (architecture.md §7).
//
// A CUDA grid is capped per axis at (x, y, z) = (2^31−1, 65 535, 65 535) on ALL
// compute capabilities, including Blackwell sm_120 (CUDA C++ Programming Guide,
// "Features and Technical Specifications"; verified for CUDA 13.x). Only the x
// axis reaches 2^31−1; y and z are capped at 65 535. A launch that exceeds any
// axis returns cudaErrorInvalidConfiguration — an opaque post-launch failure. The
// launch wrappers route their axis extents through `grid_for` / `grid_z_extent`
// (below) so an over-limit extent fails-fast with a precise message in debug
// (cleanup X-7/B6), and the orientation rule (large extent on x) is enforced by
// construction.
// ===========================================================================

/// Maximum `gridDim.x` (the only axis that reaches 2^31−1). The SNP/M-scale axis
/// MUST ride x (architecture.md §7; the decode + f2-feeder orientation, B6).
inline constexpr unsigned kMaxGridX = 2147483647u;  // 2^31 − 1

/// Maximum `gridDim.y` — hardware-capped at 65 535 on every compute capability.
inline constexpr unsigned kMaxGridY = 65535u;

/// Maximum `gridDim.z` — hardware-capped at 65 535 on every compute capability,
/// the SAME cap as `gridDim.y`. Single-sourced from `kMaxGridY` (NOT a re-typed
/// literal) so the shared y/z cap is stated once: a future per-axis hardware change
/// edits one site, never two in lock-step (DRY; NAMING-STYLE-STANDARD §2.5
/// single-source; group-5 5.3). The M4 strided-batched batch count (`n_in_group`)
/// rides this axis; the backend tiles the batch so it never exceeds this (X-7/B6).
inline constexpr unsigned kMaxGridZ = kMaxGridY;

// ===========================================================================
// Ceiling division — the launch-grid building block.
// ===========================================================================

/// Ceiling division `ceil(n / block)`. `block` must be > 0 (a block dimension)
/// and `n` non-negative (a count/dimension). Replaces the spike's open-coded
/// `(n + block - 1) / block` (architecture.md §2 "one cdiv()").
[[nodiscard]] STEPPE_HD constexpr int cdiv(int n, int block) noexcept {
    return (n + block - 1) / block;
}

/// Long overload of `cdiv` for SNP-count-scale dimensions (`M` can exceed 2^31;
/// views.hpp documents the SNP axis as `long`). The decode/feeder SNP axis MUST
/// route through this overload, never the `int` one.
[[nodiscard]] STEPPE_HD constexpr long cdiv(long n, long block) noexcept {
    return (n + block - 1) / block;
}

// ===========================================================================
// Grid-extent helper for the SQUARE-block f2 launches.
// ===========================================================================

/// Number of `block`-sized tiles needed to cover `n` elements along one axis —
/// the grid extent for a 1-D-per-axis launch. `cdiv` named for the launch-config
/// call site (`grid_for(P, kCdivBlock)` for each of x/y over the [P × P] output),
/// with the hardware grid-axis limit folded in: it debug-asserts the computed
/// extent fits `max_grid` (cleanup X-7/B6) so an over-limit axis fails-fast with a
/// precise message rather than as an opaque cudaErrorInvalidConfiguration at
/// launch. Defaults `block` to the project's standard SQUARE edge and `max_grid`
/// to the y/z 65 535 cap — the conservative limit for any axis other than x. A
/// caller mapping an extent onto `gridDim.x` may pass `kMaxGridX`.
///
/// NOTE: this is for the f2 kernels' square `dim3(kCdivBlock, kCdivBlock)` block
/// only. A kernel with a NON-square block (e.g. the 32×8 decode kernel) must call
/// `cdiv` with its explicit per-axis dims (`kDecodeBlockX`/`kDecodeBlockY`), NOT
/// `grid_for`'s square default — and the SNP axis needs the `cdiv(long,long)`
/// overload, which `grid_for` (int-only) cannot provide. The decode kernel still
/// routes its `gridDim.y` (the P axis) through `grid_for(P, kDecodeBlockY)` so the
/// y-cap assert applies there too.
[[nodiscard]] STEPPE_HD constexpr int grid_for(int n, int block = kCdivBlock,
                                               [[maybe_unused]] unsigned max_grid = kMaxGridY) noexcept {
    // `max_grid` is consumed ONLY by the STEPPE_ASSERT below, which compiles out
    // under NDEBUG (host_device.hpp). `[[maybe_unused]]` (C++17, [dcl.attr.unused])
    // makes the genuinely-conditionally-used parameter honest under -Werror in BOTH
    // build types — Release drops the assert, leaving the bound unread, but it is
    // load-bearing in debug where the fail-fast over-limit check fires.
    const int extent = cdiv(n, block);
    STEPPE_ASSERT(extent >= 0 && static_cast<unsigned>(extent) <= max_grid,
                  "grid extent exceeds the hardware grid-dimension limit "
                  "(architecture.md §7; cleanup X-7/B6) — re-orient the large axis "
                  "onto gridDim.x or tile it");
    return extent;
}

// ===========================================================================
// Batch (z-axis) extent for the M4 strided-batched gather/scatter launches.
// ===========================================================================

/// The `gridDim.z = n_in_group` extent for the M4 strided-batched gather/scatter
/// launches (f2_blocks_kernel.cu). Those wrappers set the z axis to the batch
/// count DIRECTLY (not via `grid_for`), so a `grid_for` clamp alone would not
/// cover them (cleanup X-7/B6 fix-correctness note); this is their dedicated
/// single-home guard. Asserts `1 ≤ n ≤ kMaxGridZ` (a zero z-extent is an invalid
/// launch; an over-limit z-extent is the latent failure on the high-VRAM tier
/// where a bucket's block count can grow) and returns the unsigned extent. The M4
/// backend tiles the batch into chunks of at most `kMaxGridZ` blocks
/// (vram_budget.hpp `max_blocks_per_chunk`), so this precondition always holds at
/// the call site; the assert pins the invariant.
[[nodiscard]] inline unsigned grid_z_extent(int n_in_group) noexcept {
    STEPPE_ASSERT(n_in_group >= 1 && static_cast<unsigned>(n_in_group) <= kMaxGridZ,
                  "M4 batch count (gridDim.z = n_in_group) must be in [1, kMaxGridZ] "
                  "(architecture.md §7; cleanup X-7/B6) — the backend tiles the batch "
                  "so this holds");
    return static_cast<unsigned>(n_in_group);
}

// ===========================================================================
// Decode-kernel block geometry (the GPU S0/S1 decode → allele-freq kernel).
// ===========================================================================

/// SNP-axis (x) edge of the decode kernel's 2-D thread block. 32 is one full
/// warp, so a warp's lanes span 32 contiguous SNPs of one record — keeping the
/// packed-byte reads on the within-record SNP axis contiguous/coalesced
/// (architecture.md §11.3). A half-warp `kCdivBlock = 16` would halve the per-warp
/// coalescing run, so the decode kernel deliberately does NOT reuse the square f2
/// edge; this is the named, warp-alignment-justified home for that choice
/// (ROADMAP §4 — block dims are named, never bare literals re-picked in a kernel).
inline constexpr int kDecodeBlockX = 32;

/// Population-axis (y) edge of the decode kernel's 2-D thread block. 8 keeps the
/// block at 32 × 8 = 256 threads (a standard occupancy-friendly default for a
/// bandwidth-bound kernel) while spending the warp on the SNP axis (kDecodeBlockX).
inline constexpr int kDecodeBlockY = 8;

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_LAUNCH_CONFIG_HPP
