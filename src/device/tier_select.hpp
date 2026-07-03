// src/device/tier_select.hpp
//
// The single home of the tier-select policy: where the large [P×P×n_block]
// f2/Vpair result lives — device VRAM, host RAM, or disk — chosen from runtime
// free memory. Host-pure and CUDA-free (plain std::size_t arithmetic), so it is
// host-unit-testable and usable from the CUDA-free orchestrator.
//
// Reference: docs/reference/src_device_tier_select.hpp.md
#ifndef STEPPE_DEVICE_TIER_SELECT_HPP
#define STEPPE_DEVICE_TIER_SELECT_HPP

#include <cstddef>

#include "device/vram_budget.hpp"
#include "steppe/config.hpp"

namespace steppe::device {

// Output tiers — reference §2
enum class OutputTier {
    Resident,
    HostRam,
    Disk
};

// Force-tier token spellings — reference §3
inline constexpr const char* kForceTierTokenResident = "resident";
inline constexpr const char* kForceTierTokenHostRam  = "host";
inline constexpr const char* kForceTierTokenDisk     = "disk";

// Working-set footprint helpers — reference §4
[[nodiscard]] inline std::size_t resident_working_set_bytes(int P, long M) noexcept {
    if (P <= 0 || M <= 0) return 0;
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t feeder_bufs_per_pop = kFeederRawBufsPerPop + kFeederOutBufsPerPop;
    return feeder_bufs_per_pop * pm * sizeof(double) + kCublasWorkspaceBytes;
}

[[nodiscard]] inline std::size_t streamed_working_set_bytes(
        int P, long /*M*/, int max_tile, int max_nb, int max_s_pad) noexcept {
    if (P <= 0) return 0;
    const std::size_t p = nonneg(P);
    const std::size_t max_tile_z = nonneg(max_tile);
    const std::size_t nb = nonneg(max_nb);
    const std::size_t feeder =
        (kFeederRawBufsPerPop * p * max_tile_z + kFeederOutBufsPerPop * p * max_tile_z);
    const std::size_t slabs =
        per_block_chunk_elems(P, max_s_pad) * nb;
    const std::size_t ring =
        static_cast<std::size_t>(kStreamDeviceChunks) * p * p * nb;
    return (feeder + slabs + ring) * sizeof(double) + kCublasWorkspaceBytes;
}

// Free-host-RAM probe — reference §5
[[nodiscard]] std::size_t free_host_ram_bytes() noexcept;

// Tier-select policy — reference §6
[[nodiscard]] inline OutputTier select_output_tier(
        int P, long M, int n_block,
        std::size_t free_vram, std::size_t free_host_ram) noexcept {
    if (P <= 0 || n_block <= 0) return OutputTier::Resident;
    const std::size_t result_bytes = resident_tensor_bytes(P, n_block);
    const std::size_t resident_need = result_bytes + resident_working_set_bytes(P, M);
    const std::size_t vram_budget = budget_bytes(kResidentTierVramFraction, free_vram);
    if (resident_need <= vram_budget) return OutputTier::Resident;
    const std::size_t host_budget = budget_bytes(kHostTierRamFraction, free_host_ram);
    if (result_bytes <= host_budget) return OutputTier::HostRam;
    return OutputTier::Disk;
}

// Effective-tier resolution — reference §7
[[nodiscard]] OutputTier resolve_output_tier(
        DeviceConfig::ForceTier force, const char* env_value,
        int P, long M, int n_block,
        std::size_t free_vram, std::size_t free_host_ram) noexcept;

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_TIER_SELECT_HPP
