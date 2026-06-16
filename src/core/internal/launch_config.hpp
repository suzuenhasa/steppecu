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
//   cdiv(n, b)        — ceiling division, the launch-grid building block, in an
//                       `int` overload and a `long` overload (the SNP/`M` axis
//                       exceeds 2^31, so it must use the `long` form — views.hpp).
//   grid_for(n, block) — number of `block`-sized tiles to cover `n` along one
//                       axis, defaulting to the project's standard SQUARE f2 edge
//                       `kCdivBlock`. For the f2 [P × P] / [P × s_pad] launches.
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

#include "core/internal/host_device.hpp"  // STEPPE_HD
#include "steppe/config.hpp"              // kCdivBlock

namespace steppe::core {

// ===========================================================================
// Ceiling division — the launch-grid building block.
// ===========================================================================

/// Ceiling division `ceil(n / b)`. `b` must be > 0 (a block dimension) and `n`
/// non-negative (a count/dimension). Replaces the spike's open-coded
/// `(n + b - 1) / b` (architecture.md §2 "one cdiv()").
[[nodiscard]] STEPPE_HD constexpr int cdiv(int n, int b) noexcept {
    return (n + b - 1) / b;
}

/// Long overload of `cdiv` for SNP-count-scale dimensions (`M` can exceed 2^31;
/// views.hpp documents the SNP axis as `long`). The decode/feeder SNP axis MUST
/// route through this overload, never the `int` one.
[[nodiscard]] STEPPE_HD constexpr long cdiv(long n, long b) noexcept {
    return (n + b - 1) / b;
}

// ===========================================================================
// Grid-extent helper for the SQUARE-block f2 launches.
// ===========================================================================

/// Number of `block`-sized tiles needed to cover `n` elements along one axis —
/// the grid extent for a 1-D-per-axis launch. Alias of `cdiv` named for the
/// launch-config call site (`grid_for(P, kCdivBlock)` for each of x/y over the
/// [P × P] output). Defaults `block` to the project's standard SQUARE edge.
///
/// NOTE: this is for the f2 kernels' square `dim3(kCdivBlock, kCdivBlock)` block
/// only. A kernel with a NON-square block (e.g. the 32×8 decode kernel) must call
/// `cdiv` with its explicit per-axis dims (`kDecodeBlockX`/`kDecodeBlockY`), NOT
/// `grid_for`'s square default — and the SNP axis needs the `cdiv(long,long)`
/// overload, which `grid_for` (int-only) cannot provide.
[[nodiscard]] STEPPE_HD constexpr int grid_for(int n, int block = kCdivBlock) noexcept {
    return cdiv(n, block);
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
