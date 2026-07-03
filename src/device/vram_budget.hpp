// src/device/vram_budget.hpp
//
// The single home for the batched-f2 VRAM budget math: whether a run fits, and
// how many jackknife blocks pack into one batched chunk. Host-pure and CUDA-free
// integer arithmetic, so the up-front fit check and the runtime chunk sizing share
// one implementation and stay unit-testable on a machine with no GPU.
//
// Reference: docs/reference/src_device_vram_budget.hpp.md
#ifndef STEPPE_DEVICE_VRAM_BUDGET_HPP
#define STEPPE_DEVICE_VRAM_BUDGET_HPP

#include <algorithm>
#include <cstddef>

#include "core/internal/launch_config.hpp"
#include "steppe/config.hpp"

namespace steppe::device {

// Utilization-fraction domain check — reference §4
static_assert(kMaxVramUtilizationFraction > 0.0 && kMaxVramUtilizationFraction <= 1.0,
              "kMaxVramUtilizationFraction must lie in (0, 1] — it is the fraction "
              "of free VRAM the resident working set may occupy (architecture.md §11.1).");

// Shared integer helpers (nonneg, budget_bytes) — reference §5
[[nodiscard]] inline constexpr std::size_t nonneg(int v) noexcept {
    return v < 0 ? 0u : static_cast<std::size_t>(v);
}

[[nodiscard]] inline constexpr std::size_t budget_bytes(double fraction, std::size_t free) noexcept {
    return static_cast<std::size_t>(fraction * static_cast<double>(free));
}

// Resident f2+Vpair tensor footprint — reference §6
[[nodiscard]] inline constexpr std::size_t resident_tensor_bytes(int P, int n_block) noexcept {
    if (P <= 0 || n_block <= 0) return 0;
    const std::size_t p = nonneg(P);
    const std::size_t nb = nonneg(n_block);
    return kResidentTensorCount * p * p * nb * sizeof(double);
}

// Per-block chunk footprint — reference §7
[[nodiscard]] inline constexpr std::size_t per_block_chunk_elems(int P, int s_pad) noexcept {
    const std::size_t p = nonneg(P);
    const std::size_t sp = nonneg(s_pad);
    const std::size_t slab = p * p;
    return kChunkInputStacks * p * sp + kChunkOutputStacks * slab;
}

[[nodiscard]] inline constexpr std::size_t per_block_chunk_bytes(int P, int s_pad) noexcept {
    return per_block_chunk_elems(P, s_pad) * sizeof(double);
}

// VRAM budget for one chunk — reference §8
[[nodiscard]] inline constexpr std::size_t chunk_budget_bytes(std::size_t free_vram, int P,
                                                              int n_block) noexcept {
    const std::size_t reserved = resident_tensor_bytes(P, n_block) + kCublasWorkspaceBytes;
    const std::size_t net = (free_vram > reserved) ? (free_vram - reserved) : 0u;
    return budget_bytes(kMaxVramUtilizationFraction, net);
}

// Blocks-per-chunk gate + grid-z tiling — reference §9
[[nodiscard]] inline constexpr int max_blocks_per_chunk(std::size_t free_vram, int P, int n_block,
                                                        int s_pad, int nb_total) noexcept {
    if (nb_total <= 0) return 0;
    const std::size_t budget = chunk_budget_bytes(free_vram, P, n_block);
    const std::size_t per_block = per_block_chunk_bytes(P, s_pad);
    const std::size_t budget_blocks = (per_block > 0u) ? (budget / per_block) : 0u;
    const std::size_t capped =
        std::min({budget_blocks, static_cast<std::size_t>(nb_total),
                  static_cast<std::size_t>(core::kMaxGridZ)});
    const std::size_t clamped = std::max<std::size_t>(capped, 1u);
    return static_cast<int>(clamped);
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_VRAM_BUDGET_HPP
