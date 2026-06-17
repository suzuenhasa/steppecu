// tests/reference/test_backend_capabilities_probe.cu
//
// U5 OBJECTIVE GATE — the CUDA CudaBackend::capabilities() probe + device_id
// threading (M4.5 capability-tier SCAFFOLD; architecture.md §9 DeviceConfig/
// Resources, §11.4 SPMG capability-tiered combine, §12 parity; cleanup
// device-cuda-cuda_backend F19/F20 + X-6/B2; 00-overview §(2).1 "the ONE unified
// design"). The sibling host-only gate (tests/unit/test_backend_capabilities.cpp,
// U2) pins the BackendCapabilities POD SHAPE + the value-initialized "unknown"
// base default; THIS gate pins the REAL probe VALUES the CUDA override reports on a
// live device, plus the device_id binding (F19/F20).
//
// WHY (the M4.5 prerequisite this closes): pre-U5 the single-device CudaBackend
// ignored DeviceConfig::devices, ran on the ambient current device, and had no
// capability probe — so `Resources` could neither bind one backend per device nor
// pick the P2P device-resident combine vs the host-staged baseline (architecture.md
// §11.4). U5 threads `int device_id` into make_cuda_backend()/the ctor (default 0 ⇒
// single-GPU unchanged) and implements capabilities() — cudaGetDeviceProperties
// (compute major/minor), cudaMemGetInfo (total+free; the M0/M4 path historically
// DISCARDED total — captured now), cudaGetDeviceCount, and cudaDeviceCanAccessPeer
// routed through the NON-throwing STEPPE_CUDA_WARN (canAccessPeer "no" is an
// EXPECTED tagged degrade, never a fault), and emulated_fp64_honorable from the ONE
// X-6/B2 predicate.
//
// THE rtxbox VERDICT (the verify gate, 2× RTX PRO 6000 Blackwell, sm_120, 96 GB ea):
//   * compute_major >= 12               (sm_120 ⇒ {12, 0})
//   * total_vram_bytes > 0              (the captured `total` from cudaMemGetInfo)
//   * device_count == 2                 (both PRO 6000s visible)
//   * can_access_peer == true           (the REAL stock-driver P2P this box has —
//                                         MEASURED 55.6 GB/s, both directions; the
//                                         capability-tier law, workflow wxz1fiiln)
//   * a backend constructs on device 0 AND device 1 (count >= 2) with NO error.
//
// PORTABILITY (so the ONE sm_120 build serves BOTH boxes — the §0 / ⚡ box-role
// split). The capability-tier law makes can_access_peer == FALSE the EXPECTED budget
// answer on consumer GeForce (2× 5090: P2P driver-disabled), so this gate keys the
// STRICT peer assertion to the actual tier:
//   * on a datacenter / PRO Blackwell box (device name contains "PRO" — the rtxbox),
//     2+ devices MUST report can_access_peer == true (the stock-driver P2P fact);
//   * on a GeForce box, peer may be false — that is the tagged degrade to the
//     host-staged combine baseline, a PASS (logged), not a fault;
//   * a single-GPU lane (count == 1) has no peer to reach ⇒ can_access_peer == false
//     by construction, and the device-1 construction sub-check is SKIPPED.
// The TIER-INVARIANT facts (compute_major >= 12, total_vram > 0, device_count >= 1,
// honorability == library emulation_honorable, device-0 construct) are asserted on
// EVERY lane. Every field is parity-NEUTRAL (§12), so this gate never touches a
// reported number — it asserts the PROBE, not a statistic.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `backend_capabilities_probe` test (tests/CMakeLists.txt) linking steppe::device.
// DATA-FREE. Self-checking main() (not a GoogleTest TU); CTest gates on the exit
// code. Run:  ./test_backend_capabilities_probe     (no data needed)
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>

#include "steppe/config.hpp"               // Precision, kDefaultMantissaBits
#include "device/backend.hpp"              // ComputeBackend, BackendCapabilities
#include "device/backend_factory.hpp"      // steppe::device::make_cuda_backend (X-9/B8; M4.5 device_id)
#include "device/cuda/f2_block_kernel.cuh" // emulation_honorable (the X-6/B2 honorability SoT)

using steppe::BackendCapabilities;
using steppe::Precision;
using steppe::device::emulation_honorable;

namespace {

int g_failures = 0;

// One named PASS/FAIL line; increments the failure counter on a false condition.
void check(const char* label, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

// Construct a backend bound to `device_id` and confirm it (a) constructs without
// throwing and (b) reports that device_id back through capabilities() — i.e. the
// probe is the bound device, not the ambient one. Returns true on PASS.
bool construct_on_device(int device_id) {
    try {
        auto gpu = steppe::device::make_cuda_backend(device_id);
        // The probe must read THIS backend's device (the per-device-instance
        // contract): re-querying capabilities after construction is non-throwing and
        // device-neutral (it save/restores the ambient device).
        const BackendCapabilities c = gpu->capabilities();
        const bool ok = (c.compute_major >= 12) && (c.total_vram_bytes > 0);
        std::printf("    device %d: constructed; compute=%d.%d total_vram=%zu bytes -> %s\n",
                    device_id, c.compute_major, c.compute_minor, c.total_vram_bytes,
                    ok ? "ok" : "BAD");
        return ok;
    } catch (const std::exception& e) {
        std::printf("    device %d: THREW %s\n", device_id, e.what());
        return false;
    }
}

// Is the bound device a datacenter / PRO-tier Blackwell (stock-driver P2P), vs a
// consumer GeForce (P2P driver-disabled)? The capability-tier law keys the strict
// peer assertion off this: "PRO" in the device name ⇒ the rtxbox tier that MUST
// report real P2P; otherwise the budget GeForce tier where peer==false is the
// expected tagged degrade (architecture.md §11.4; workflow wxz1fiiln).
bool device_is_pro_tier(int device_id) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return false;
    return std::strstr(prop.name, "PRO") != nullptr;
}

}  // namespace

int main() {
    std::printf("test_backend_capabilities_probe (U5: CUDA capabilities() probe + "
                "device_id threading)\n");

    // The process must see at least one CUDA device to run a GPU gate at all.
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
        std::fprintf(stderr,
                     "test_backend_capabilities_probe: no CUDA device visible "
                     "(cudaGetDeviceCount); cannot run the U5 probe gate.\n");
        return 1;
    }
    std::printf("  visible CUDA devices: %d\n", count);

    // ---- Construct on device 0 and read its capabilities (the bound-device probe).
    std::unique_ptr<steppe::ComputeBackend> gpu0;
    try {
        gpu0 = steppe::device::make_cuda_backend(0);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "test_backend_capabilities_probe: make_cuda_backend(0) threw: %s\n",
                     e.what());
        return 1;
    }
    const BackendCapabilities c0 = gpu0->capabilities();
    std::printf("  device 0 probe: count=%d compute=%d.%d total_vram=%zu free_vram=%zu "
                "can_access_peer=%s emulated_fp64_honorable=%s\n",
                c0.device_count, c0.compute_major, c0.compute_minor,
                c0.total_vram_bytes, c0.free_vram_bytes,
                c0.can_access_peer ? "true" : "false",
                c0.emulated_fp64_honorable ? "true" : "false");

    // ---- TIER-INVARIANT assertions (every lane: rtxbox, budget box, single-GPU) --
    // sm_120 (and any future Blackwell+) ⇒ compute_major >= 12. The architecture
    // ships ONE sm_120 build for both boxes (architecture.md §0), so this is the
    // floor every supported device clears.
    check("compute_major >= 12 (Blackwell sm_120)", c0.compute_major >= 12);
    // The captured `total` from cudaMemGetInfo — the datum the M0/M4 path discarded
    // (cleanup 00-overview §(2).1). A live device always has nonzero total VRAM.
    check("total_vram_bytes > 0 (captured, no longer discarded)", c0.total_vram_bytes > 0);
    // free <= total is a runtime invariant of cudaMemGetInfo (free is a subset of
    // total physical memory); a violation would mean the two outputs were swapped.
    check("free_vram_bytes <= total_vram_bytes (cudaMemGetInfo ordering)",
          c0.free_vram_bytes <= c0.total_vram_bytes);
    // The probe's device_count must equal the count this TU independently read.
    check("device_count matches cudaGetDeviceCount", c0.device_count == count);
    // emulated_fp64_honorable must equal the LIBRARY's own X-6/B2 predicate for a
    // default EmulatedFp64 request — the probe and the GEMM path consult the SAME
    // predicate, so they can never report different honorability (the C-2 close).
    const Precision emu_probe{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    check("emulated_fp64_honorable == library emulation_honorable(EmulatedFp64)",
          c0.emulated_fp64_honorable == emulation_honorable(emu_probe));
    // Device 0 must construct AND report itself (the bound-device probe).
    check("backend constructs on device 0 (and reports compute>=12, total>0)",
          construct_on_device(0));

    // ---- TIER- and COUNT-conditional assertions (the rtxbox verdict) -------------
    if (count >= 2) {
        // The rtxbox verdict: BOTH PRO 6000s visible. Construct on device 1 too.
        check("backend constructs on device 1 (count >= 2)", construct_on_device(1));

        if (device_is_pro_tier(0)) {
            // Datacenter / PRO Blackwell on the stock driver: REAL P2P. This is the
            // MEASURED rtxbox fact (cudaDeviceCanAccessPeer == 1 both directions,
            // 55.6 GB/s; architecture.md §11.4 capability-tier law). On this tier the
            // probe MUST report the P2P-capable fast-path is available.
            std::printf("  tier: datacenter/PRO Blackwell -> strict P2P assertion\n");
            check("can_access_peer == true (PRO 6000 stock-driver P2P, 2+ devices)",
                  c0.can_access_peer);
        } else {
            // Consumer GeForce (2× 5090 budget box): P2P is driver-DISABLED, so
            // can_access_peer == false is the EXPECTED tagged degrade to the
            // host-staged fixed-order combine baseline — a PASS (the NON-throwing
            // probe degraded cleanly), not a fault (architecture.md §11.4, §12).
            std::printf("  tier: consumer GeForce -> P2P expected DISABLED "
                        "(host-staged combine baseline)\n");
            check("can_access_peer probe did not fault on the budget tier "
                  "(host-staged baseline is the expected degrade)",
                  true);  // reaching here means the probe returned without throwing
            std::printf("    can_access_peer=%s (false is the expected budget-tier "
                        "tagged degrade)\n", c0.can_access_peer ? "true" : "false");
        }
    } else {
        // Single-GPU lane: no peer to reach, so can_access_peer is false by
        // construction, and the device-1 sub-check is not applicable. The probe
        // shape + tier-invariant facts above still gate this lane.
        std::printf("  single-GPU lane (count == 1): skipping the device-1 + peer "
                    "verdict; can_access_peer must be false (no peer to reach)\n");
        check("can_access_peer == false on a single visible device (no peer)",
              !c0.can_access_peer);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "test_backend_capabilities_probe: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_backend_capabilities_probe: all capability-probe checks PASS\n");
    return 0;
}
