// tests/unit/test_backend_capabilities.cpp
//
// U2 / M4.5 capability-tier scaffold verdict gate (host-only, GPU-FREE, CUDA-FREE).
//
// Pins the SHAPE + value-initialized "unknown" defaults of the BackendCapabilities
// POD that device/backend.hpp now owns (architecture.md §9 DeviceConfig/Resources,
// §11.4 SPMG capability-tiered combine, §12 parity; cleanup 00-overview §(2).1
// "the ONE unified design"; device-backend §11.1/§11.2). This is the OBJECTIVE
// TEST for U2: a default-constructed BackendCapabilities is a POD/aggregate whose
// every field carries the documented zero/false default. The REAL probe values
// (what the fields actually take on a device) are asserted in the CUDA path's
// probe test (U5), NOT here — this gate owns only the type contract.
//
// Why this TU is CUDA-FREE proof (the §4 layering anchor, exactly like
// tests/unit/test_backend_factory.cpp): backend.hpp is CUDA-free by contract, so
// this plain .cpp compiles with the HOST compiler and links ONLY
// steppe::core_internal (which exposes the src/ include root resolving
// "device/backend.hpp" and "core/internal/views.hpp") + steppe::api (config.hpp /
// fstats.hpp). It links NEITHER steppe::device NOR any CUDA toolkit. If
// BackendCapabilities (or its transitive includes) dragged in a CUDA header this
// TU would FAIL TO COMPILE — proving the capability struct sits on the CUDA-free
// side of the seam the whole DI design rests on.
//
// Dual harness (identical to the sibling host-only tests): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise a self-checking main()
// returns non-zero on the first failure — all CTest needs to gate
// (tests/CMakeLists.txt). No CUDA here.
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

#include "device/backend.hpp"  // steppe::BackendCapabilities, steppe::ComputeBackend (CUDA-FREE)

namespace {

// ---- (1) BackendCapabilities is a POD/aggregate (crosses the §4 CUDA-free seam) -
// "POD struct" per the U2 contract: an aggregate that is trivially copyable and
// standard-layout, so it slots BY VALUE into Resources / a result envelope and is
// memcpy-safe across the seam (architecture.md §4). These are COMPILE-TIME — a
// regression that adds a user-provided ctor, a virtual, or a non-trivial member
// breaks the build, not just the run.
static_assert(std::is_aggregate_v<steppe::BackendCapabilities>,
              "BackendCapabilities must stay an aggregate (POD, value-initializable; "
              "architecture.md §4, §9)");
static_assert(std::is_trivially_copyable_v<steppe::BackendCapabilities>,
              "BackendCapabilities must be trivially copyable (crosses the CUDA-free "
              "seam by value; architecture.md §4)");
static_assert(std::is_standard_layout_v<steppe::BackendCapabilities>,
              "BackendCapabilities must be standard-layout (POD contract; §4)");

// ---- the documented field TYPES (the §9 / overview §(2).1 shape) ---------------
static_assert(std::is_same_v<decltype(steppe::BackendCapabilities::device_count), int>,
              "device_count is int");
static_assert(std::is_same_v<decltype(steppe::BackendCapabilities::compute_major), int>,
              "compute_major is int");
static_assert(std::is_same_v<decltype(steppe::BackendCapabilities::compute_minor), int>,
              "compute_minor is int");
static_assert(std::is_same_v<decltype(steppe::BackendCapabilities::total_vram_bytes), std::size_t>,
              "total_vram_bytes is std::size_t");
static_assert(std::is_same_v<decltype(steppe::BackendCapabilities::free_vram_bytes), std::size_t>,
              "free_vram_bytes is std::size_t");
static_assert(std::is_same_v<decltype(steppe::BackendCapabilities::can_access_peer), bool>,
              "can_access_peer is bool");
static_assert(std::is_same_v<decltype(steppe::BackendCapabilities::emulated_fp64_honorable), bool>,
              "emulated_fp64_honorable is bool");

// ---- (2) the value-initialized "unknown" defaults (all zero / false) -----------
// A value-initialized BackendCapabilities means "nothing probed yet" — every
// numeric field 0, every capability flag false. This is also exactly what the
// ComputeBackend::capabilities() base default returns (checked in (3)), so the
// two must agree. constexpr because the aggregate has only constant initializers.
[[nodiscard]] constexpr bool defaults_all_unknown(const steppe::BackendCapabilities& c) {
    return c.device_count == 0 &&
           c.compute_major == 0 &&
           c.compute_minor == 0 &&
           c.total_vram_bytes == std::size_t{0} &&
           c.free_vram_bytes == std::size_t{0} &&
           c.can_access_peer == false &&
           c.emulated_fp64_honorable == false;
}
static_assert(defaults_all_unknown(steppe::BackendCapabilities{}),
              "default-constructed BackendCapabilities must be all zero/false "
              "(the 'unknown' tier; architecture.md §9, cleanup 00-overview §(2).1)");

[[nodiscard]] bool test_value_initialized_defaults() {
    const steppe::BackendCapabilities c{};  // value-initialization
    return defaults_all_unknown(c);
}

// ---- (3) ComputeBackend::capabilities() has a usable base default --------------
// The base must return the value-initialized "unknown" so CpuBackend need not
// override it (architecture.md §13; the U2 contract). We instantiate a trivial
// concrete backend that overrides ONLY the pure-virtual compute methods and does
// NOT touch capabilities(), then assert the inherited base returns the unknown
// tier. Each compute override is unreachable here (we never call them) and exists
// solely to make the type concrete — no CUDA, no device.
struct DefaultCapsBackend final : steppe::ComputeBackend {
    [[nodiscard]] steppe::F2Result compute_f2(const steppe::core::MatView&,
                                              const steppe::core::MatView&,
                                              const steppe::core::MatView&,
                                              const steppe::Precision&) override {
        return {};
    }
    [[nodiscard]] steppe::F2BlockTensor compute_f2_blocks(const steppe::core::MatView&,
                                                          const steppe::core::MatView&,
                                                          const steppe::core::MatView&,
                                                          const int*, int,
                                                          const steppe::Precision&) override {
        return {};
    }
    [[nodiscard]] steppe::DecodeResult decode_af(const steppe::DecodeTileView&) override {
        return {};
    }
    // NOTE: capabilities() is intentionally NOT overridden — that is the contract.
};

[[nodiscard]] bool test_base_capabilities_default_unknown() {
    const DefaultCapsBackend backend;
    const steppe::ComputeBackend& base = backend;  // through the abstract interface
    return defaults_all_unknown(base.capabilities());
}

// ---- (4) the B3 capability-query virtuals: base default false, consistent with the
//          sentinel throw (docs/kimiactions/01-open-worth-doing.md §B3) --------------
// provides_rank_sweep()/provides_batched_fit() are the EXPLICIT, type-safe replacement
// for the old try/catch-as-capability-detection: the host orchestrator (qpadm_fit.cpp /
// model_search.cpp) branches on these instead of CATCHING the rank_sweep /
// fit_models_batched sentinel std::runtime_error. The base default is false, and a
// backend that does NOT override the underlying virtual MUST throw the sentinel from
// it — so the query and the throw are two views of ONE fact and cannot drift.
// DefaultCapsBackend (above) overrides NEITHER sentinel virtual, so it is exactly that
// "no real implementation" backend. The throw half is exercised through rank_sweep,
// whose args are all header-only CUDA-free PODs; fit_models_batched's DeviceF2Blocks
// arg has its ctor in steppe_device (not linkable into this CUDA-free host TU), so only
// its query half is checked here — the CudaBackend true-path is golden-gated.
[[nodiscard]] bool test_capability_query_base_default_consistent() {
    DefaultCapsBackend backend;  // overrides neither rank_sweep nor fit_models_batched
    const steppe::ComputeBackend& base = backend;
    if (base.provides_rank_sweep() != false) return false;
    if (base.provides_batched_fit() != false) return false;

    // false ⇒ the matching sentinel virtual throws (the consistency the query stands in
    // for). Base rank_sweep body throws std::runtime_error("not implemented...").
    bool rank_sweep_threw = false;
    try {
        (void)backend.rank_sweep(steppe::F4Blocks{}, steppe::JackknifeCov{}, 0.05,
                                 steppe::QpAdmOptions{}, steppe::Precision{});
    } catch (const std::runtime_error&) {
        rank_sweep_threw = true;
    }
    return rank_sweep_threw;
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"BackendCapabilities value-init is all zero/false (unknown tier)",
     test_value_initialized_defaults},
    {"ComputeBackend::capabilities() base default returns unknown (no override needed)",
     test_base_capabilities_default_unknown},
    {"capability-query virtuals: base default false ⇒ rank_sweep sentinel throws (B3)",
     test_capability_query_base_default_consistent},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(BackendCapabilities, PodShapeAndUnknownDefaults) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}

#else  // self-checking fallback (no GoogleTest)

int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_backend_capabilities: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_backend_capabilities: all %zu capability-contract checks PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}

#endif
