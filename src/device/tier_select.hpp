// src/device/tier_select.hpp
//
// M5 ADAPTIVE TIERED f2_blocks OUTPUT — the tier-select policy (host-pure, CUDA-FREE).
//
// The adaptive [P×P×n_block] f2/Vpair result lives in the FASTEST tier it FITS in,
// selected AUTOMATICALLY from runtime free VRAM / free host RAM (or pinned by the
// STEPPE_FORCE_TIER / config.force_tier override for tests). This header is the SINGLE
// home of that policy + the runtime free-RAM probe declaration. It sits beside
// vram_budget.hpp in src/device/ and is CUDA-FREE for the SAME reason (vram_budget.hpp
// :14-18): it is device-path policy expressed as plain std::size_t arithmetic with no
// device call, unit-testable on the host with NO GPU, and reachable by the CUDA-free
// orchestrator (steppe::core) without dragging in the device toolkit. No CUDA header.
//
// PARITY-NEUTRAL (architecture.md §12): the tier choice changes only WHERE/WHEN a slab
// lands, never its bits. TIER 0 (Resident) is the existing device-resident path
// verbatim — when select returns Resident the orchestrator calls compute_f2_blocks_device
// exactly as today (no sink, no streaming). Streaming is OPT-IN-BY-NEED: reached ONLY
// when the result does not fit VRAM.
#ifndef STEPPE_DEVICE_TIER_SELECT_HPP
#define STEPPE_DEVICE_TIER_SELECT_HPP

#include <cstddef>

#include "device/vram_budget.hpp"  // resident_tensor_bytes (the §11.2 footprint home, reused)
#include "steppe/config.hpp"       // kResidentTierVramFraction, kHostTierRamFraction, kCublasWorkspaceBytes, DeviceConfig::ForceTier

namespace steppe::device {

/// WHERE the adaptive [P×P×n_block] f2/Vpair result lives — the FASTEST tier it FITS.
/// Selected automatically by select_output_tier() from runtime free VRAM / free host
/// RAM, or pinned by the STEPPE_FORCE_TIER override (tests). PARITY-NEUTRAL: the tier
/// changes only WHERE/WHEN a slab lands, never its bits (architecture.md §12).
enum class OutputTier {
    Resident,  ///< TIER 0: result + working set fit free VRAM -> device-resident
               ///< (DeviceF2Blocks), the EXISTING path UNCHANGED. NO streaming.
    HostRam,   ///< TIER 1: does not fit VRAM but fits free host RAM -> stream blocks
               ///< into a host F2BlockTensor via the triple-buffered sink.
    Disk       ///< TIER 2: fits neither -> stream blocks to a disk file via a small
               ///< persistent pinned staging buffer (laptop-friendly tiny RAM).
};

/// Non-result transient VRAM the Resident-tier single-GPU compute holds co-resident
/// with the [P×P×n_block] result during run_f2_blocks_resident: the feeder phase
/// (raw inputs 3·P·M + persisted feeder outputs 4·P·M = 7·P·M doubles) + the cuBLAS
/// determinism workspace. The chunk slabs reuse the freed-raw VRAM (cuda_backend.cu
/// :544-560) so they fit under this envelope. Mirrors estimate_peak_vram_bytes in
/// test_f2_multigpu_parity.cu:223-230, MINUS the resident_tensor_bytes term (the
/// caller adds the result separately). std::size_t throughout (no 32-bit wrap).
[[nodiscard]] inline std::size_t resident_working_set_bytes(int P, long M) noexcept {
    if (P <= 0 || M <= 0) return 0;
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    return 7u * pm * sizeof(double) + kCublasWorkspaceBytes;
}

/// Free host RAM in bytes, read at RUNTIME (NEVER hardcoded — vast instances vary).
/// Linux: sysinfo(2) -> (freeram + bufferram) * mem_unit, i.e. RAM the OS can hand
/// back without swapping (a conservative "available" proxy; we do NOT count
/// reclaimable page cache beyond bufferram). Returns 0 if sysinfo fails (forces the
/// caller toward the Disk tier — the safe direction). Defined in host_ram.cpp.
[[nodiscard]] std::size_t free_host_ram_bytes() noexcept;

/// THE TIER-SELECT POLICY (frozen). Pick the FASTEST tier the result FITS in.
///   result_bytes = resident_tensor_bytes(P, n_block) = 2·P²·n_block·8  (vram_budget.hpp).
///   Resident  iff  result_bytes + resident_working_set_bytes(P, M)
///                      <= kResidentTierVramFraction * free_vram   (0.70)
///   else HostRam iff result_bytes <= kHostTierRamFraction * free_host_ram   (0.60)
///   else Disk.
/// All comparisons in std::size_t against the fraction*double (cast the size_t up,
/// floor the product). free_vram / free_host_ram are the RUNTIME probes, never
/// hardcoded. P<=0 / n_block<=0 -> Resident (degenerate empty result, the existing
/// path's no-op; never streams nothing). PARITY-NEUTRAL: the choice moves no bits.
///
/// @param P            populations.
/// @param M            total SNP count (for the working-set term; Resident only).
/// @param n_block      jackknife blocks.
/// @param free_vram    free device VRAM, bytes (resources.gpus[0].caps.free_vram_bytes).
/// @param free_host_ram free host RAM, bytes (free_host_ram_bytes()).
[[nodiscard]] inline OutputTier select_output_tier(
        int P, long M, int n_block,
        std::size_t free_vram, std::size_t free_host_ram) noexcept {
    if (P <= 0 || n_block <= 0) return OutputTier::Resident;  // degenerate empty -> existing no-op
    const std::size_t result_bytes = resident_tensor_bytes(P, n_block);  // 2·P²·n_block·8
    const std::size_t resident_need = result_bytes + resident_working_set_bytes(P, M);
    const std::size_t vram_budget =
        static_cast<std::size_t>(kResidentTierVramFraction * static_cast<double>(free_vram));
    if (resident_need <= vram_budget) return OutputTier::Resident;
    const std::size_t host_budget =
        static_cast<std::size_t>(kHostTierRamFraction * static_cast<double>(free_host_ram));
    if (result_bytes <= host_budget) return OutputTier::HostRam;
    return OutputTier::Disk;
}

/// Resolve the EFFECTIVE tier: config.force_tier wins; else STEPPE_FORCE_TIER env
/// ("resident"/"host"/"disk", case-insensitive); else the automatic select_output_tier.
/// `env_value` is the already-read getenv result (nullptr if unset) — passed in so this
/// helper stays pure/testable (the orchestrator does the getenv, the §2 no-global rule).
[[nodiscard]] OutputTier resolve_output_tier(
        DeviceConfig::ForceTier force, const char* env_value,
        int P, long M, int n_block,
        std::size_t free_vram, std::size_t free_host_ram) noexcept;

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_TIER_SELECT_HPP
