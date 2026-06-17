// tests/reference/test_resources_build.cu
//
// UNIT I1 OBJECTIVE GATE — build_resources / Resources / PerGpuResources (M4.5
// multi-GPU; architecture.md §9 DeviceConfig/Resources "Dependency injection of
// resources", §11.4 SPMG capability-tiered combine, §12 parity; cleanup
// 00-overview §(2).1 "One capability probe, owned at Resources construction").
//
// WHAT THIS PINS (the verify gate for UNIT I1, the Resources HOME + builder):
//   * build_resources(DeviceConfig{devices={0}}) ⇒ EXACTLY ONE PerGpuResources,
//     bound to device 0 (entry.device_id == 0) with a live backend whose probed
//     caps report a Blackwell+ device (compute_major >= 12). This is the
//     single-GPU lane — the exact current path.
//   * build_resources(DeviceConfig{devices={0,1}}) on this 2-GPU box ⇒ EXACTLY
//     TWO PerGpuResources, bound to devices 0 and 1 IN ORDER (gpus[g].device_id
//     == g, the FIXED g=0..G-1 combine order, §11.4/§12), each with a live backend
//     reporting compute_major >= 12, and — on the rtxbox PRO tier — each reporting
//     can_access_peer == true (the REAL stock-driver P2P this box has, MEASURED
//     55.6 GB/s; the capability-tier law, architecture.md §11.4).
//   * The probed caps live in PerGpuResources::caps (OUT-OF-BAND), never on a
//     numeric tensor (§12; cleanup §(2).2). The probe is the ONE probe, owned at
//     Resources construction (§(2).1) — build_resources stores it.
//   * Auto-enumerate (empty devices) resolves to all visible ordinals 0..count-1
//     in order — the §9 DeviceConfig::devices contract — verified to match the
//     explicit {0,..,count-1} build.
//
// NO real f2 compute is exercised here (the OBJECTIVE TEST asks only for the
// Resources binding + probe), so this TU is DATA-FREE. Every checked field is
// parity-NEUTRAL (§12): the gate asserts the device BINDING + the PROBE, never a
// reported statistic.
//
// PORTABILITY (so the ONE sm_120 build serves BOTH boxes — the §0 / ⚡ box-role
// split, mirroring test_backend_capabilities_probe.cu):
//   * The TIER-INVARIANT facts (1 entry for {0}, the device 0 binding, every
//     entry compute_major >= 12) are asserted on EVERY lane.
//   * The 2-device build + the strict can_access_peer == true assertion are gated
//     on (count >= 2) AND the PRO tier (device name contains "PRO"). On a budget
//     GeForce box (P2P driver-disabled) can_access_peer == false is the EXPECTED
//     tagged degrade to the host-staged combine baseline — a PASS (logged), not a
//     fault; the 2-device build still succeeds and is asserted, only the strict
//     peer assertion is relaxed. On a single visible device the {0,1} build is
//     SKIPPED (its second ordinal does not exist).
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Self-checking main() (not a
// GoogleTest TU); CTest gates on the exit code. Links steppe::device (the builder
// + the factory + capabilities()) + steppe::core_internal (the src/ include root
// resolving "device/resources.hpp") + steppe::api (config.hpp). architecture.md
// §9, §11.4, §12, §13. Run:  ./test_resources_build   (no data needed)
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "steppe/config.hpp"        // steppe::DeviceConfig
#include "device/backend.hpp"       // steppe::BackendCapabilities
#include "device/resources.hpp"     // steppe::device::Resources, build_resources (the unit under test)

using steppe::BackendCapabilities;
using steppe::DeviceConfig;
using steppe::device::build_resources;
using steppe::device::Resources;

namespace {

int g_failures = 0;

// One named PASS/FAIL line; increments the failure counter on a false condition.
void check(const char* label, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

// Is device `device_id` a datacenter / PRO-tier Blackwell (stock-driver P2P), vs a
// consumer GeForce (P2P driver-disabled)? The capability-tier law keys the strict
// peer assertion off this: "PRO" in the device name ⇒ the rtxbox tier that MUST
// report real P2P; otherwise the budget GeForce tier where peer==false is the
// expected tagged degrade (architecture.md §11.4; matches the probe gate's keying).
bool device_is_pro_tier(int device_id) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return false;
    return std::strstr(prop.name, "PRO") != nullptr;
}

}  // namespace

int main() {
    std::printf("test_resources_build (UNIT I1: build_resources / Resources / "
                "PerGpuResources)\n");

    // The process must see at least one CUDA device to build any Resources.
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::fprintf(stderr,
                     "test_resources_build: no CUDA device visible "
                     "(cudaGetDeviceCount); cannot run the I1 Resources gate.\n");
        return 1;
    }
    std::printf("  visible CUDA devices: %d\n", count);

    // ---- LANE A: single-GPU build, DeviceConfig{devices={0}} --------------------
    // EXACTLY ONE PerGpuResources bound to device 0 with a Blackwell+ probe.
    try {
        DeviceConfig cfg_g1;
        cfg_g1.devices = {0};
        Resources r1 = build_resources(cfg_g1);

        check("devices={0} -> device_count() == 1", r1.device_count() == 1);
        if (r1.device_count() == 1) {
            const auto& g0 = r1.gpus[0];
            check("devices={0} -> gpus[0].device_id == 0 (bound to device 0)",
                  g0.device_id == 0);
            check("devices={0} -> gpus[0].backend is non-null (a live bound backend)",
                  g0.backend != nullptr);
            check("devices={0} -> gpus[0].caps.compute_major >= 12 (Blackwell sm_120)",
                  g0.caps.compute_major >= 12);
            check("devices={0} -> gpus[0].caps.total_vram_bytes > 0 (probed device)",
                  g0.caps.total_vram_bytes > 0);
            // The frozen intent levers round-trip into Resources::config (§9).
            check("devices={0} -> config.devices preserved ({0})",
                  r1.config.devices.size() == 1 && r1.config.devices[0] == 0);
            std::printf("    G1: device_id=%d compute=%d.%d total_vram=%zu "
                        "can_access_peer=%s\n",
                        g0.device_id, g0.caps.compute_major, g0.caps.compute_minor,
                        g0.caps.total_vram_bytes,
                        g0.caps.can_access_peer ? "true" : "false");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_resources_build: build_resources({0}) threw: %s\n",
                     e.what());
        ++g_failures;
    }

    // ---- LANE B: auto-enumerate build, DeviceConfig{devices={}} -----------------
    // Empty ⇒ every visible ordinal 0..count-1 in order (§9 DeviceConfig contract).
    try {
        DeviceConfig cfg_auto;  // devices left empty
        Resources r_auto = build_resources(cfg_auto);

        check("devices={} (auto) -> device_count() == cudaGetDeviceCount",
              r_auto.device_count() == static_cast<std::size_t>(count));
        bool order_ok = (r_auto.device_count() == static_cast<std::size_t>(count));
        for (std::size_t g = 0; g < r_auto.device_count(); ++g) {
            if (r_auto.gpus[g].device_id != static_cast<int>(g)) order_ok = false;
        }
        check("devices={} (auto) -> gpus[g].device_id == g for all g (dense, in order)",
              order_ok);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_resources_build: build_resources({} auto) threw: %s\n",
                     e.what());
        ++g_failures;
    }

    // ---- LANE C: 2-GPU build, DeviceConfig{devices={0,1}} (count >= 2 only) ------
    if (count >= 2) {
        try {
            DeviceConfig cfg_g2;
            cfg_g2.devices = {0, 1};
            Resources r2 = build_resources(cfg_g2);

            check("devices={0,1} -> device_count() == 2", r2.device_count() == 2);
            if (r2.device_count() == 2) {
                // Each entry bound to its ordinal IN ORDER (the fixed g=0..G-1
                // combine order, §11.4/§12).
                check("devices={0,1} -> gpus[0].device_id == 0", r2.gpus[0].device_id == 0);
                check("devices={0,1} -> gpus[1].device_id == 1", r2.gpus[1].device_id == 1);
                check("devices={0,1} -> gpus[0].backend non-null", r2.gpus[0].backend != nullptr);
                check("devices={0,1} -> gpus[1].backend non-null", r2.gpus[1].backend != nullptr);
                check("devices={0,1} -> gpus[0].caps.compute_major >= 12",
                      r2.gpus[0].caps.compute_major >= 12);
                check("devices={0,1} -> gpus[1].caps.compute_major >= 12",
                      r2.gpus[1].caps.compute_major >= 12);

                for (std::size_t g = 0; g < 2; ++g) {
                    std::printf("    G2: gpus[%zu].device_id=%d compute=%d.%d "
                                "total_vram=%zu can_access_peer=%s\n",
                                g, r2.gpus[g].device_id, r2.gpus[g].caps.compute_major,
                                r2.gpus[g].caps.compute_minor,
                                r2.gpus[g].caps.total_vram_bytes,
                                r2.gpus[g].caps.can_access_peer ? "true" : "false");
                }

                // Strict peer assertion, KEYED to the tier (the rtxbox verdict).
                if (device_is_pro_tier(0)) {
                    std::printf("  tier: datacenter/PRO Blackwell -> strict P2P "
                                "assertion (both entries)\n");
                    check("devices={0,1} -> gpus[0].caps.can_access_peer == true "
                          "(PRO 6000 stock-driver P2P)", r2.gpus[0].caps.can_access_peer);
                    check("devices={0,1} -> gpus[1].caps.can_access_peer == true "
                          "(PRO 6000 stock-driver P2P)", r2.gpus[1].caps.can_access_peer);
                } else {
                    // Consumer GeForce: P2P driver-DISABLED. can_access_peer == false
                    // is the EXPECTED tagged degrade to the host-staged combine
                    // baseline — the 2-device build still succeeded (asserted above),
                    // which is the PASS (architecture.md §11.4, §12).
                    std::printf("  tier: consumer GeForce -> P2P expected DISABLED "
                                "(host-staged combine baseline); 2-device build still "
                                "succeeds\n");
                    check("devices={0,1} build succeeded on the budget tier "
                          "(host-staged baseline is the expected degrade)", true);
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "test_resources_build: build_resources({0,1}) threw: %s\n",
                         e.what());
            ++g_failures;
        }
    } else {
        std::printf("  single-GPU lane (count == 1): skipping the {0,1} 2-device "
                    "build verdict (device 1 does not exist)\n");
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "test_resources_build: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_resources_build: all Resources-build checks PASS\n");
    return 0;
}
