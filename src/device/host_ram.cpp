// src/device/host_ram.cpp
//
// The CUDA-FREE runtime free-host-RAM probe (free_host_ram_bytes) + the effective-tier
// resolver (resolve_output_tier) declared in tier_select.hpp. Linux sysinfo(2) is the
// single home for the RAM probe (there is no existing sysinfo/getenv precedent in the
// tree); resolve_output_tier lives here too (the frozen design's "host_ram.cpp or a
// small tier_select.cpp" choice) so the env-parse + the policy fall-through have ONE
// home. No CUDA header — this links into steppe_device but is device-toolkit-free.
#include "device/tier_select.hpp"

#include <cctype>
#include <sys/sysinfo.h>

namespace steppe::device {

std::size_t free_host_ram_bytes() noexcept {
    struct sysinfo si{};
    if (sysinfo(&si) != 0) return 0;
    // mem_unit==0 is the legacy (pre-2.3.23 kernel) convention where the size fields
    // are already in bytes (no unit field), so the 1u branch is NOT dead — it scales by
    // one on those kernels. On modern kernels the fields are multiples of mem_unit bytes.
    const std::size_t unit = si.mem_unit ? si.mem_unit : 1u;
    // Count the reclaimable buffer cache as free: bufferram is reclaimable under memory
    // pressure, so it is available to the host-RAM tier. Other reclaimable fields
    // (sharedram, the page cache not exposed here) are deliberately omitted — freeram +
    // bufferram is the conservative sysinfo(2) lower bound the tier is gated on.
    return (static_cast<std::size_t>(si.freeram) +
            static_cast<std::size_t>(si.bufferram)) * unit;
}

namespace {
// Case-insensitive exact compare of a (possibly null) C string to a lowercase literal.
[[nodiscard]] bool iequals(const char* s, const char* lower_lit) noexcept {
    if (s == nullptr) return false;
    std::size_t i = 0;
    for (; s[i] != '\0' && lower_lit[i] != '\0'; ++i) {
        const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        if (c != lower_lit[i]) return false;
    }
    return s[i] == '\0' && lower_lit[i] == '\0';
}
}  // namespace

OutputTier resolve_output_tier(
        DeviceConfig::ForceTier force, const char* env_value,
        int P, long M, int n_block,
        std::size_t free_vram, std::size_t free_host_ram) noexcept {
    // Frozen precedence: config field first, then env var, then automatic.
    switch (force) {
        case DeviceConfig::ForceTier::Resident: return OutputTier::Resident;
        case DeviceConfig::ForceTier::HostRam:  return OutputTier::HostRam;
        case DeviceConfig::ForceTier::Disk:     return OutputTier::Disk;
        case DeviceConfig::ForceTier::Auto:     break;  // fall through to env, then auto
    }
    // The STEPPE_FORCE_TIER token → tier mapping as a small {token, tier} table + loop,
    // instead of three near-identical if (iequals(...)) lines that differ only by the
    // (token, enum) pair — one row per tier, so a tier add/rename touches a single place
    // (cleanup group-7 7.1; the tokens are single-homed beside OutputTier in tier_select.hpp).
    struct ForceTierToken { const char* token; OutputTier tier; };
    static constexpr ForceTierToken kForceTierTokens[] = {
        {kForceTierTokenResident, OutputTier::Resident},
        {kForceTierTokenHostRam,  OutputTier::HostRam},
        {kForceTierTokenDisk,     OutputTier::Disk},
    };
    for (const ForceTierToken& entry : kForceTierTokens) {
        if (iequals(env_value, entry.token)) return entry.tier;
    }
    // Any other env value (or unset) ⇒ ignored ⇒ automatic policy.
    return select_output_tier(P, M, n_block, free_vram, free_host_ram);
}

}  // namespace steppe::device
