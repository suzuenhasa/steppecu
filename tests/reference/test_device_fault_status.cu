// tests/reference/test_device_fault_status.cu
//
// B2 GUARD — the OOM fault-taxonomy translator (architecture.md §10 fault taxonomy;
// cli-bindings.md §4.4 exit-code map). It pins the one behaviour B2 adds: a GENUINE
// device out-of-memory exits kExitDeviceOom (3), not the catch-all kExitRuntimeError
// (5), while every non-allocation fault keeps 5.
//
// WHY (the layering this pins): the typed device exceptions (CudaError / CublasError /
// CusolverError) and their cudaError_t / cublasStatus_t / cusolverStatus_t status
// codes are PRIVATE to steppe_device (cuda/check.cuh is a .cuh — the CUDA-free app
// never sees it; architecture.md §4). So the OOM classification lives in a
// steppe_device TU: device::device_fault_status (declared CUDA-free in
// device/resources.hpp, defined in cuda/cuda_backend.cu). The app's CUDA-free helper
// app::exit_code_for_caught calls it. This test exercises BOTH ends — the device seam
// and the app helper — so neither can silently drift from the §10 taxonomy.
//
// WHAT IT PINS (data-free, NO GPU work — pure exception/RTTI control flow; the typed
// exceptions are host objects whose ctors only format a std::string, so this runs on
// every lane with no CUDA device and no real AADR):
//   1. CudaError(cudaErrorMemoryAllocation)      -> DeviceOom -> exit 3
//   2. CublasError(CUBLAS_STATUS_ALLOC_FAILED)   -> DeviceOom -> exit 3
//   3. CusolverError(CUSOLVER_STATUS_ALLOC_FAILED) -> DeviceOom -> exit 3
//   4. CudaError(cudaErrorInvalidValue)  (a NON-alloc device fault) -> nullopt -> 5
//   5. std::runtime_error (a host fault) -> nullopt -> 5  (host std::bad_alloc, host
//      RAM not device VRAM, takes the same nullopt -> 5 path — correct).
//
// Build: REMOTE sm_120 / CUDA 13 box (NOT locally). Built by CMake/CTest as the
// `device_fault_status_unit` test (tests/CMakeLists.txt), the same CUDA_SEPARABLE_
// COMPILATION pattern as test_cuda_check.cu. Links steppe::device (check.cuh +
// the device_fault_status definition) + steppe::api (steppe::Status). NO GPU, NO data.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <source_location>
#include <stdexcept>

#include "app/exit_code_for_caught.hpp"  // app::exit_code_for_caught (the CUDA-free end-to-end map)
#include "core/config/exit_code.hpp"     // kExitDeviceOom (3), kExitRuntimeError (5), exit_code_for
#include "device/cuda/check.cuh"         // CudaError / CublasError / CusolverError
#include "device/resources.hpp"          // steppe::device::device_fault_status (the CUDA-free seam)
#include "steppe/error.hpp"              // steppe::Status

using steppe::Status;
using steppe::app::exit_code_for_caught;
using steppe::device::CublasError;
using steppe::device::CudaError;
using steppe::device::CusolverError;
using steppe::device::device_fault_status;

namespace cfg = steppe::config;

namespace {

// An alloc-fault exception must classify to DeviceOom on the seam AND exit 3 via the
// app helper. Returns true on PASS.
bool oom_maps_to_three(const char* label, const std::exception& e) {
    const std::optional<Status> s = device_fault_status(e);
    const bool seam_ok = s.has_value() && *s == Status::DeviceOom;
    const int code = exit_code_for_caught(e);
    const bool code_ok = (code == cfg::kExitDeviceOom);  // 3
    const bool ok = seam_ok && code_ok;
    std::printf("  %-46s -> seam=%s exit=%d -> %s\n", label,
                seam_ok ? "DeviceOom" : "(not DeviceOom)", code,
                ok ? "PASS" : "FAIL");
    if (!ok)
        std::fprintf(stderr,
            "  [FAIL] %s: expected device_fault_status->DeviceOom and exit %d; got "
            "seam_has_value=%d, exit=%d.\n",
            label, cfg::kExitDeviceOom, static_cast<int>(s.has_value()), code);
    return ok;
}

// A non-alloc fault (device or host) must NOT classify (nullopt) and must keep the
// catch-all exit 5. Returns true on PASS.
bool nonalloc_maps_to_five(const char* label, const std::exception& e) {
    const std::optional<Status> s = device_fault_status(e);
    const bool seam_ok = !s.has_value();  // nullopt
    const int code = exit_code_for_caught(e);
    const bool code_ok = (code == cfg::kExitRuntimeError);  // 5
    const bool ok = seam_ok && code_ok;
    std::printf("  %-46s -> seam=%s exit=%d -> %s\n", label,
                seam_ok ? "nullopt" : "(classified!)", code,
                ok ? "PASS" : "FAIL");
    if (!ok)
        std::fprintf(stderr,
            "  [FAIL] %s: expected nullopt + catch-all exit %d; got "
            "seam_has_value=%d, exit=%d.\n",
            label, cfg::kExitRuntimeError, static_cast<int>(s.has_value()), code);
    return ok;
}

}  // namespace

int main() {
    std::printf("\nB2 OOM fault-taxonomy translator (synthetic, no GPU, no data)\n");

    const std::source_location loc = std::source_location::current();
    bool ok = true;

    // (1)-(3) The three device allocation faults all reclassify 5 -> 3.
    ok = oom_maps_to_three("CudaError(cudaErrorMemoryAllocation)",
                           CudaError(cudaErrorMemoryAllocation, "alloc", loc)) && ok;
    ok = oom_maps_to_three("CublasError(CUBLAS_STATUS_ALLOC_FAILED)",
                           CublasError(CUBLAS_STATUS_ALLOC_FAILED, "alloc", loc)) && ok;
    ok = oom_maps_to_three("CusolverError(CUSOLVER_STATUS_ALLOC_FAILED)",
                           CusolverError(CUSOLVER_STATUS_ALLOC_FAILED, "alloc", loc)) && ok;

    // (4) A NON-alloc device fault stays the catch-all (not every CUDA error is OOM).
    ok = nonalloc_maps_to_five("CudaError(cudaErrorInvalidValue)",
                               CudaError(cudaErrorInvalidValue, "bad", loc)) && ok;

    // (5) A plain host fault (and, by the same nullopt path, host std::bad_alloc) is 5.
    ok = nonalloc_maps_to_five("std::runtime_error (host fault)",
                               std::runtime_error("host fault")) && ok;

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — the OOM translator did not map a device allocation fault to\n"
            "        DeviceOom/exit 3, or reclassified a non-alloc fault away from the\n"
            "        catch-all exit 5. architecture.md §10; cli-bindings.md §4.4; B2.\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr,
        "RESULT: PASS (device allocation faults map to DeviceOom -> exit 3 via both the\n"
        "        device seam and the app helper; non-alloc faults keep the catch-all 5)\n");
    return EXIT_SUCCESS;
}
