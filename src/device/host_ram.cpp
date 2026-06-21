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
    const std::size_t unit = si.mem_unit ? si.mem_unit : 1u;
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
    if (iequals(env_value, "resident")) return OutputTier::Resident;
    if (iequals(env_value, "host"))     return OutputTier::HostRam;
    if (iequals(env_value, "disk"))     return OutputTier::Disk;
    // Any other env value (or unset) ⇒ ignored ⇒ automatic policy.
    return select_output_tier(P, M, n_block, free_vram, free_host_ram);
}

}  // namespace steppe::device
