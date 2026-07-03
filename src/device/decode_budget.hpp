// src/device/decode_budget.hpp
//
// Sizes the regime-B decode SNP-tile width against live free VRAM so the
// on-device extract_f2 decode stays under peak-VRAM budget. CUDA-free, host-pure
// std::size_t math, so the CUDA-free app layer includes it directly and every
// function is unit-testable with no GPU.
//
// Reference: docs/reference/src_device_decode_budget.hpp.md
#ifndef STEPPE_DEVICE_DECODE_BUDGET_HPP
#define STEPPE_DEVICE_DECODE_BUDGET_HPP

#include <cstddef>
#include <cstdlib>

#include "core/internal/decode_af.hpp"

namespace steppe::device {

// Named constants — reference §4
inline constexpr double kDecodeTileVramFraction = 0.6;

inline constexpr std::size_t kDecodePerColMetaBytes = 51u;

// Per-SNP-column device-memory footprint — reference §5
[[nodiscard]] inline constexpr std::size_t decode_per_col_bytes(
    int P, std::size_t n_individuals) noexcept {
    const std::size_t p = P > 0 ? static_cast<std::size_t>(P) : 0u;
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t qvn = 6u * p * sizeof(double);
    const std::size_t packed = (n_individuals + cpb - 1u) / cpb;
    return qvn + kDecodePerColMetaBytes + packed;
}

// Pure tile-width budget — reference §6
[[nodiscard]] inline constexpr long decode_tile_snps_budget(
    std::size_t free_vram, int P, std::size_t n_individuals) noexcept {
    const std::size_t cpb = static_cast<std::size_t>(core::kCodesPerByte);
    const std::size_t per_col = decode_per_col_bytes(P, n_individuals);
    const std::size_t budget =
        static_cast<std::size_t>(kDecodeTileVramFraction * static_cast<double>(free_vram));
    std::size_t tile = (per_col > 0u) ? (budget / per_col) : 0u;
    tile -= tile % cpb;
    if (tile < cpb) tile = cpb;
    return static_cast<long>(tile);
}

// Final tile step (env override + clamp to M) — reference §7
[[nodiscard]] inline long decode_tile_snps(std::size_t free_vram, int P,
                                           std::size_t n_individuals, long M) noexcept {
    if (M <= 0) return 0;
    const long cpb = static_cast<long>(core::kCodesPerByte);
    long tile = decode_tile_snps_budget(free_vram, P, n_individuals);
    if (const char* env = std::getenv("STEPPE_DECODE_TILE_SNPS")) {
        const long forced = std::strtol(env, nullptr, 10);
        if (forced > 0) {
            tile = forced - (forced % cpb);
            if (tile < cpb) tile = cpb;
        }
    }
    if (tile > M) tile = M;
    return tile;
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_DECODE_BUDGET_HPP
