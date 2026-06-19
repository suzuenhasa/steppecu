// tests/reference/test_cuda_check.cu
//
// M4.5 CAP-1/CAP-2 OBJECTIVE GATE — the NON-throwing, tagged-degrade CUDA check
// (cleanup device-cuda-check CAP-1/CAP-2; architecture.md §11.4 capability tiers,
// §10 log taxonomy, §12 parity; TODO M4.5). This is the test the verify gate
// requires: STEPPE_CUDA_WARN must LOG-and-CONTINUE on a recoverable status while
// STEPPE_CUDA_CHECK on the SAME status FAULTS (throws CudaError).
//
// WHY (the foot-gun this pins closed): M4.5 makes the single-device CudaBackend
// multi-GPU-READY with a capability probe. On the budget tier (GeForce, P2P
// driver-disabled) `cudaDeviceCanAccessPeer` returns "cannot" and a subsequent
// `cudaDeviceEnablePeerAccess` returns cudaErrorPeerAccessAlreadyEnabled — these
// are EXPECTED capability-degrade outcomes, NOT faults (re-verified on rtxbox:
// the PRO-6000 tier reports canAccessPeer=1 both ways, byte-exact P2P; the budget
// tier degrades to host-staged fixed-order combine — both bit-identical per §12).
// Routing those probes through the THROWING STEPPE_CUDA_CHECK would convert a
// graceful, tagged degrade into a hard failure. STEPPE_CUDA_WARN is the sanctioned
// non-throwing path: it emits ONE STEPPE_LOG_WARN line and YIELDS the cudaError_t
// so the caller branches and tags the degrade.
//
// WHAT IT PINS (data-free, NO GPU work — pure macro/exception control flow, so it
// runs on every lane and needs neither a CUDA device nor real AADR; the runtime's
// cudaGetErrorName/String are host-side string lookups requiring no context):
//   1. WARN DOES NOT THROW + YIELDS THE STATUS: STEPPE_CUDA_WARN on the CAP-2
//      status cudaErrorPeerAccessAlreadyEnabled returns that exact status without
//      throwing (the capability-degrade caller can branch on it). Same for the
//      CAP-1 surrogate cudaErrorPeerAccessNotEnabled. The return is build-mode
//      INDEPENDENT (only the WARN *line* is NDEBUG-gated, per STEPPE_LOG_WARN).
//   2. WARN ON cudaSuccess IS A QUIET PASS-THROUGH: STEPPE_CUDA_WARN(cudaSuccess)
//      returns cudaSuccess and (by construction) logs nothing — the success path
//      is the hot path and must not warn.
//   3. CHECK ON THE SAME STATUS DOES THROW: STEPPE_CUDA_CHECK on the identical
//      cudaErrorPeerAccessAlreadyEnabled throws a CudaError whose status() ==
//      that status (the contrast that proves WARN and CHECK are DIFFERENT paths,
//      not the same call wearing two names) — and CHECK(cudaSuccess) does NOT.
//   4. SOURCE-LOCATION CAPTURE: the thrown CudaError::what() carries this file's
//      name and the stringized expression (the std::source_location default-arg
//      captures the CALLER — the A-4 nvcc-fragility regression guard).
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `cuda_check_unit` test (tests/CMakeLists.txt) linking steppe::device's device-
// private check.cuh (a .cuh — PRIVATE to steppe_device, architecture.md §4 — so
// this gate is a CUDA TU, not a host-only unit test) + steppe::core_internal (the
// STEPPE_LOG_WARN sink). NO GPU, NO data. Run:  ./test_cuda_check   (no args)
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

#include "device/cuda/check.cuh"  // STEPPE_CUDA_WARN / STEPPE_CUDA_CHECK + CudaError

using steppe::device::CudaError;

namespace {

// (1) STEPPE_CUDA_WARN on a recoverable status must NOT throw and must YIELD the
//     exact status. `status` is one of the EXPECTED M4.5 capability-degrade codes.
//     Returns true on PASS.
bool warn_does_not_throw_and_yields(const char* label, cudaError_t status) {
    try {
        const cudaError_t yielded = STEPPE_CUDA_WARN(status);
        const bool ok = (yielded == status);
        std::printf("  %-44s -> warn yielded %s (%d) -> %s\n",
                    label, cudaGetErrorName(yielded), static_cast<int>(yielded),
                    ok ? "PASS" : "FAIL (yielded wrong status)");
        if (!ok)
            std::fprintf(stderr,
                "  [FAIL] %s: STEPPE_CUDA_WARN yielded %d, expected %d.\n",
                label, static_cast<int>(yielded), static_cast<int>(status));
        return ok;
    } catch (const std::exception& e) {
        std::printf("  %-44s -> THREW -> FAIL\n", label);
        std::fprintf(stderr,
            "  [FAIL] %s: STEPPE_CUDA_WARN THREW on a recoverable status — it must\n"
            "         log-and-continue, NEVER throw (the CAP-1/CAP-2 contract).\n"
            "         what(): %s\n", label, e.what());
        return false;
    }
}

// (3) STEPPE_CUDA_CHECK on the SAME status must THROW a CudaError carrying that
//     status — the contrast that proves WARN and CHECK are distinct paths.
//     Also pins (4): the thrown message carries this file + the stringized expr.
//     Returns true on PASS.
bool check_throws_carrying_status(cudaError_t status) {
    try {
        STEPPE_CUDA_CHECK(status);
        std::printf("  %-44s -> did NOT throw -> FAIL\n", "check throws on the same status");
        std::fprintf(stderr,
            "  [FAIL] STEPPE_CUDA_CHECK did NOT throw on %s — the throwing checker\n"
            "         must FAULT on any non-cudaSuccess status (architecture.md §7).\n",
            cudaGetErrorName(status));
        return false;
    } catch (const CudaError& e) {
        const bool status_ok = (e.status() == status);
        // (4) source_location captured the CALLER (this file) + the #expr.
        const bool names_file = std::strstr(e.what(), "test_cuda_check") != nullptr;
        const bool names_expr = std::strstr(e.what(), "status") != nullptr;
        const bool ok = status_ok && names_file && names_expr;
        std::printf("  %-44s -> threw CudaError(status=%s) -> %s\n",
                    "check throws on the same status",
                    cudaGetErrorName(e.status()), ok ? "PASS" : "FAIL");
        if (!ok)
            std::fprintf(stderr,
                "  [FAIL] CudaError thrown but: status round-trips? %d; what() names\n"
                "         this file? %d; what() names the expr? %d. what(): %s\n",
                static_cast<int>(status_ok), static_cast<int>(names_file),
                static_cast<int>(names_expr), e.what());
        return ok;
    } catch (const std::exception& e) {
        std::printf("  %-44s -> threw non-CudaError -> FAIL\n", "check throws on the same status");
        std::fprintf(stderr,
            "  [FAIL] STEPPE_CUDA_CHECK threw a non-CudaError (expected the typed\n"
            "         CudaError). what(): %s\n", e.what());
        return false;
    }
}

}  // namespace

int main() {
    std::printf("\nM4.5 CAP-1/CAP-2 non-throwing tagged-degrade check (synthetic, no GPU, no data)\n");

    bool ok = true;

    // (1) WARN on the EXPECTED capability-degrade statuses must log-and-continue
    //     and yield the status the probe caller branches on.
    ok = warn_does_not_throw_and_yields(
             "warn on cudaErrorPeerAccessAlreadyEnabled (CAP-2)",
             cudaErrorPeerAccessAlreadyEnabled) && ok;
    ok = warn_does_not_throw_and_yields(
             "warn on cudaErrorPeerAccessNotEnabled (CAP-1)",
             cudaErrorPeerAccessNotEnabled) && ok;

    // (2) WARN on success is a quiet pass-through (no log, returns cudaSuccess).
    ok = warn_does_not_throw_and_yields(
             "warn on cudaSuccess (quiet pass-through)", cudaSuccess) && ok;

    // (3)+(4) CHECK on the SAME status FAULTS (throws CudaError carrying it) —
    //     and CHECK on success does NOT throw (the happy path stays clean).
    ok = check_throws_carrying_status(cudaErrorPeerAccessAlreadyEnabled) && ok;
    try {
        STEPPE_CUDA_CHECK(cudaSuccess);
        std::printf("  %-44s -> did not throw -> PASS\n", "check on cudaSuccess is clean");
    } catch (const std::exception& e) {
        ok = false;
        std::printf("  %-44s -> THREW -> FAIL\n", "check on cudaSuccess is clean");
        std::fprintf(stderr, "  [FAIL] STEPPE_CUDA_CHECK threw on cudaSuccess: %s\n", e.what());
    }

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — STEPPE_CUDA_WARN did not log-and-continue (or did not yield the\n"
            "        status), or STEPPE_CUDA_CHECK did not fault on the same recoverable\n"
            "        status. The capability probe would then either throw where it must\n"
            "        tagged-degrade, or silently swallow a real fault. architecture.md §10,\n"
            "        §11.4, §12; cleanup device-cuda-check CAP-1/CAP-2; TODO M4.5.\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr,
        "RESULT: PASS (STEPPE_CUDA_WARN logs-and-continues yielding the cudaError_t for the\n"
        "        EXPECTED P2P capability-degrade statuses, while STEPPE_CUDA_CHECK faults on\n"
        "        the same status — the two are distinct paths, parity-neutral on both tiers)\n");
    return EXIT_SUCCESS;
}
