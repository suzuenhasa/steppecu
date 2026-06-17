// tests/unit/test_backend_factory.cpp
//
// B8 / cleanup X-9 verdict gate (host-only, GPU-FREE, CUDA-FREE).
//
// Pins the single-source backend-factory contract that device/backend_factory.hpp
// now owns (architecture.md §8 DRY single-home, §4 layering, §13 testing):
//
//   1. The header is CUDA-FREE. This TU is a plain .cpp compiled by the host
//      compiler and links ONLY steppe::core_internal + steppe::api (NOT
//      steppe::device, which carries the PRIVATE CUDA toolkit + RDC device code).
//      If backend_factory.hpp (or its transitive device/backend.hpp) dragged in a
//      CUDA header, this TU would FAIL TO COMPILE — proving the CUDA-free contract
//      the whole `core`-reaches-GPU-only-through-a-CUDA-free-seam design rests on.
//
//   2. BOTH factories are declared in ONE place and ONE namespace. Before B8 every
//      consumer hand-declared the prototype itself, in two different namespaces
//      (steppe::core for cpu, steppe::device for cuda) — a latent ODR/linkage
//      hazard with no single source (cleanup X-9/B8). This test asserts, at COMPILE
//      TIME, that `steppe::device::make_cpu_backend` and
//      `steppe::device::make_cuda_backend` both name a declaration reachable through
//      the header with the EXACT documented signature
//      `std::unique_ptr<steppe::ComputeBackend>()` (CPU) /
//      `std::unique_ptr<steppe::ComputeBackend>(int)` (CUDA). A regression that
//      re-splits the namespace, drops a factory, or changes a signature breaks this
//      build.
//
//      M4.5 (U5) DEVICE THREADING: `make_cuda_backend` now takes an `int device_id`
//      (default 0) so `Resources` can BIND one GPU backend per device in
//      `DeviceConfig::devices` (the per-device-instance contract, device/backend.hpp;
//      architecture.md §11.4 SPMG). The default argument keeps every zero-arg CALL
//      compiling unchanged, but the function's declared TYPE is now `(*)(int)` —
//      pinned below. `make_cpu_backend` stays `(*)()` (the CPU reference is
//      device-agnostic; it ignores DeviceConfig::devices, §9).
//
// The checks are `decltype`-only: they inspect the declared TYPE of each factory
// and never ODR-use it (no call, no escaping address-of), so this TU needs NO
// definition at link time and therefore does NOT need to link steppe::device or
// device-link its RDC kernels. (The definitions ARE exercised, and the link-through-
// the-header path IS proven end-to-end, by the six reference .cu tests that now
// reach both factories through this same header with no forward declarations.)
//
// Build harness mirrors tests/unit/test_launch_config.cpp (GoogleTest when found,
// else a self-checking main() returning non-zero on failure; CTest gates on the
// exit code either way — architecture.md §13).

#include <memory>
#include <type_traits>

#include "device/backend_factory.hpp"  // the unit under test (CUDA-FREE by contract)
#include "device/backend.hpp"          // steppe::ComputeBackend (the return type)

// --- The single-source contract, checked at COMPILE TIME ---------------------
// Both factories must be reachable as steppe::device::make_*_backend through the
// header (no forward declaration anywhere), in the ONE namespace steppe::device,
// each returning std::unique_ptr<steppe::ComputeBackend> by value. The CPU factory
// takes no device arg; the CUDA factory takes the binding `int device_id` (M4.5/U5).
using CpuFactoryFn  = std::unique_ptr<steppe::ComputeBackend> (*)();
using CudaFactoryFn = std::unique_ptr<steppe::ComputeBackend> (*)(int);

static_assert(
    std::is_same_v<decltype(&steppe::device::make_cpu_backend), CpuFactoryFn>,
    "make_cpu_backend must be declared in steppe::device returning "
    "std::unique_ptr<steppe::ComputeBackend>() (cleanup X-9/B8 single-source)");

static_assert(
    std::is_same_v<decltype(&steppe::device::make_cuda_backend), CudaFactoryFn>,
    "make_cuda_backend must be declared in steppe::device returning "
    "std::unique_ptr<steppe::ComputeBackend>(int device_id) (X-9/B8 single-source; "
    "M4.5/U5 device-id threading, default 0)");

#if defined(STEPPE_TEST_WITH_GTEST)
#include <gtest/gtest.h>

// The static_asserts above are the real gate (they fail the COMPILE). This case
// gives CTest a named, passing assertion and documents the contract at runtime.
TEST(BackendFactory, BothFactoriesSingleSourcedInDeviceNamespace) {
    constexpr bool kCpuOk =
        std::is_same_v<decltype(&steppe::device::make_cpu_backend), CpuFactoryFn>;
    constexpr bool kCudaOk =
        std::is_same_v<decltype(&steppe::device::make_cuda_backend), CudaFactoryFn>;
    EXPECT_TRUE(kCpuOk);
    EXPECT_TRUE(kCudaOk);
}

#else  // self-checking fallback (no GoogleTest)

#include <cstdio>

int main() {
    // If we compiled and linked at all, the CUDA-free + single-source + signature
    // contract held (the static_asserts above are compile-time). Report and pass.
    constexpr bool kCpuOk =
        std::is_same_v<decltype(&steppe::device::make_cpu_backend), CpuFactoryFn>;
    constexpr bool kCudaOk =
        std::is_same_v<decltype(&steppe::device::make_cuda_backend), CudaFactoryFn>;
    if (!kCpuOk || !kCudaOk) {
        std::fprintf(stderr, "[backend_factory] FAIL: factory signature mismatch\n");
        return 1;
    }
    std::printf("[backend_factory] OK: make_cpu_backend + make_cuda_backend "
                "single-sourced in steppe::device through a CUDA-free header\n");
    return 0;
}

#endif
