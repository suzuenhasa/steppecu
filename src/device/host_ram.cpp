// src/device/host_ram.cpp
//
// CUDA-free host-RAM probe (free_host_ram_bytes, via Linux sysinfo(2)) plus the
// effective output-tier resolver (resolve_output_tier). No CUDA header: links into
// steppe_device but is device-toolkit-free.
#include "device/tier_select.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/sysinfo.h>

namespace steppe::device {

namespace {

// Read a single unsigned integer from a cgroup pseudo-file. Returns SIZE_MAX for a
// literal "max" (cgroup v2 = unlimited) or on any open/read/parse failure.
std::size_t read_uint_file(const char* path) noexcept {
    std::FILE* f = std::fopen(path, "r");
    if (f == nullptr) return SIZE_MAX;
    char buf[64] = {};
    const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    if (n == 0 || buf[0] == 'm') return SIZE_MAX;  // empty or "max"
    char* end = nullptr;
    const unsigned long long v = std::strtoull(buf, &end, 10);
    return end == buf ? SIZE_MAX : static_cast<std::size_t>(v);
}

// Container memory headroom (limit - current usage) from the cgroup, or SIZE_MAX if
// there is no meaningful cap (uncapped, sentinel-unlimited, or files unavailable).
// Tries the cgroup v2 unified hierarchy first, then falls back to v1.
std::size_t cgroup_free_bytes() noexcept {
    struct { const char* limit; const char* current; } hierarchies[] = {
        {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"},              // v2
        {"/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"},                             // v1
    };
    for (const auto& h : hierarchies) {
        const std::size_t limit = read_uint_file(h.limit);
        // SIZE_MAX = "max"/unreadable; a near-SIZE_MAX value is the v1 "unlimited" sentinel.
        if (limit == SIZE_MAX || limit >= (SIZE_MAX >> 1)) continue;
        const std::size_t current = read_uint_file(h.current);
        if (current == SIZE_MAX || current >= limit) return 0;
        return limit - current;
    }
    return SIZE_MAX;
}

}  // namespace

std::size_t free_host_ram_bytes() noexcept {
    struct sysinfo si{};
    std::size_t sys_free = 0;
    if (sysinfo(&si) == 0) {
        const std::size_t unit = si.mem_unit ? si.mem_unit : 1u;
        sys_free = (static_cast<std::size_t>(si.freeram) +
                    static_cast<std::size_t>(si.bufferram)) * unit;
    }
    // A memory-capped container reports host-wide free RAM through sysinfo(2), which
    // over-budgets the host tier and gets the process OOM-killed. Clamp to the real
    // cgroup headroom so select_output_tier falls through to Disk instead.
    return std::min(sys_free, cgroup_free_bytes());
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
