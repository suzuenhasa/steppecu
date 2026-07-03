// src/device/host_ram.cpp
//
// CUDA-free host-RAM probe (free_host_ram_bytes, via Linux sysinfo(2)) plus the
// effective output-tier resolver (resolve_output_tier). No CUDA header: links into
// steppe_device but is device-toolkit-free.
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
    switch (force) {
        case DeviceConfig::ForceTier::Resident: return OutputTier::Resident;
        case DeviceConfig::ForceTier::HostRam:  return OutputTier::HostRam;
        case DeviceConfig::ForceTier::Disk:     return OutputTier::Disk;
        case DeviceConfig::ForceTier::Auto:     break;
    }
    struct ForceTierToken { const char* token; OutputTier tier; };
    static constexpr ForceTierToken kForceTierTokens[] = {
        {kForceTierTokenResident, OutputTier::Resident},
        {kForceTierTokenHostRam,  OutputTier::HostRam},
        {kForceTierTokenDisk,     OutputTier::Disk},
    };
    for (const ForceTierToken& entry : kForceTierTokens) {
        if (iequals(env_value, entry.token)) return entry.tier;
    }
    return select_output_tier(P, M, n_block, free_vram, free_host_ram);
}

}  // namespace steppe::device
