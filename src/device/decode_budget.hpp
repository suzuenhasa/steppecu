// src/device/decode_budget.hpp
//
// THE single home for the regime-B DECODE SNP-tile-width budget (the peak-VRAM cure
// for the on-device extract_f2 decode). CUDA-FREE, host-pure `std::size_t` math (the
// established vram_budget.hpp pattern) so the CUDA-free app layer (extract_f2_core.cpp)
// includes it directly and it is unit-testable with NO GPU.
//
// WHY IT EXISTS. CudaBackend::decode_af_compact_filter decodes the tile to THREE
// resident [P × M] FP64 buffers (dQ/dV/dN) BEFORE any filter/tier runs, so peak VRAM
// is O(P × M) UNFILTERED — ~36 GB at 700 pops × 2.14 M SNPs, an OOM on a 32 GB GPU
// (`--tier` cannot help; it governs only the f2 RESULT, decided AFTER the full-M
// decode). The extract_f2 driver SNP-TILES the decode so peak VRAM is O(P × tile_M):
// this helper sizes tile_M against the live free-VRAM probe.
//
// PARITY-NEUTRAL BY CONSTRUCTION (§12). tile_M chooses HOW MANY tiles, never the
// result: per-SNP keep independence + the monotone per-tile scan + the in-file-order
// host append make the tiled Q/V/N/kept-axis BYTE-IDENTICAL to the single-shot path
// for ANY legal tile_M. The ONLY hard constraint is that the tile STEP be a multiple
// of kCodesPerByte (4) so every tile start s_lo stays 2-bit-packing-aligned — enforced
// here (the budget and the env override are both rounded DOWN to a multiple of 4).
#ifndef STEPPE_DEVICE_DECODE_BUDGET_HPP
#define STEPPE_DEVICE_DECODE_BUDGET_HPP

#include <cstddef>
#include <cstdlib>  // std::getenv, std::strtol

#include "core/internal/decode_af.hpp"  // core::kCodesPerByte (the 2-bit packing radix)

namespace steppe::device {

/// Safe fraction of FREE VRAM the regime-B decode working set may occupy per tile
/// (the decode holds ~6 [P × tile_M] FP64 buffers + O(tile_M) metadata + the packed
/// 2-bit slice co-resident). Below vram_budget's f2 utilization fraction because the
/// decode buffers are transient and a lower share leaves headroom for the tier that
/// runs after; parity-neutral (it only sets the tile count).
inline constexpr double kDecodeTileVramFraction = 0.6;

/// The 1-D per-SNP-column device/host metadata byte footprint of the regime-B decode
/// (ref+alt 1B each, chrom 4B int, genpos+physpos 8B each, flags 1B, keep-idx 8B long,
/// and the kept twins chrom 4B / genpos 8B / physpos 8B): 1+1+4+8+8+1+8+4+8+8 = 51.
/// A fixed per-column constant (independent of P and n_individuals).
inline constexpr std::size_t kDecodePerColMetaBytes = 51u;

/// Per-SNP-COLUMN device-memory footprint of the regime-B decode (bytes): the 3
/// resident [P] FP64 decode columns (dQ/dV/dN) + the 3 compacted [P] FP64 columns
/// (out q/v/n, worst-case every SNP kept) = 6·P·8; the 1-D per-SNP metadata
/// (kDecodePerColMetaBytes); and the packed 2-bit slice column ceil(n_individuals/4).
/// The tile width is free_share / this. Clamps a negative P to 0 (well-defined on any
/// input; the caller guards P>0).
[[nodiscard]] inline constexpr std::size_t decode_per_col_bytes(
    int P, std::size_t n_individuals) noexcept {
    const std::size_t p = P > 0 ? static_cast<std::size_t>(P) : 0u;
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t qvn = 6u * p * sizeof(double);      // dQ/dV/dN + compacted q/v/n
    const std::size_t packed = (n_individuals + cpb - 1u) / cpb;  // the 2-bit slice column
    return qvn + kDecodePerColMetaBytes + packed;
}

/// PURE budget: the largest SNP-tile width that keeps the decode working set under
/// `kDecodeTileVramFraction · free_vram`, ROUNDED DOWN to a multiple of kCodesPerByte
/// (so every tile start s_lo = k·tile_M stays 2-bit-aligned) and floored at
/// kCodesPerByte. constexpr — no env, no device call (the env override + the clamp to
/// M live in `decode_tile_snps`). Parity-neutral (§12).
[[nodiscard]] inline constexpr long decode_tile_snps_budget(
    std::size_t free_vram, int P, std::size_t n_individuals) noexcept {
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t per_col = decode_per_col_bytes(P, n_individuals);
    const std::size_t budget =
        static_cast<std::size_t>(kDecodeTileVramFraction * static_cast<double>(free_vram));
    std::size_t tile = (per_col > 0u) ? (budget / per_col) : 0u;
    tile -= tile % cpb;             // round DOWN to a multiple of the packing radix (4)
    if (tile < cpb) tile = cpb;     // at least one full packed byte's worth of SNPs
    return static_cast<long>(tile);
}

/// SNP-tile STEP width for the regime-B decode loop: the pure VRAM budget, with an env
/// override STEPPE_DECODE_TILE_SNPS (>0 ⇒ FORCE that width — lets a test drive many
/// small tiles), then CLAMPED to M. Guarantees the returned step is EITHER a multiple
/// of kCodesPerByte and < M (so every s_lo = k·step is 2-bit-aligned) OR == M (a single
/// tile: s_lo=0 only, so a non-multiple-of-4 M is still aligned). Both the budget and
/// the env value are rounded DOWN to a multiple of 4 for the alignment invariant.
/// Returns 0 when M<=0 (nothing to tile).
[[nodiscard]] inline long decode_tile_snps(std::size_t free_vram, int P,
                                           std::size_t n_individuals, long M) noexcept {
    if (M <= 0) return 0;
    const long cpb = static_cast<long>(core::kCodesPerByte);
    long tile = decode_tile_snps_budget(free_vram, P, n_individuals);
    if (const char* env = std::getenv("STEPPE_DECODE_TILE_SNPS")) {
        const long forced = std::strtol(env, nullptr, 10);
        if (forced > 0) {
            tile = forced - (forced % cpb);   // keep the forced step 2-bit-aligned
            if (tile < cpb) tile = cpb;
        }
    }
    if (tile > M) tile = M;   // one tile covers all SNPs (s_lo=0 only; width M is fine)
    return tile;
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_DECODE_BUDGET_HPP
