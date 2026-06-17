// tests/unit/test_config.cpp
//
// Host-only unit test of the public configuration contract (include/steppe/config.hpp,
// architecture.md §9, §12; ROADMAP §4). Pure C++ TU, NO GPU, NO CUDA: config.hpp is
// deliberately CUDA-free, so the very fact this TU compiles and links WITHOUT the
// device layer is itself the §4-layering proof. It pins the SPEC-MANDATED defaults of
// DeviceConfig — the value object the §12 determinism rules are phrased against — so a
// future edit cannot silently change them.
//
// VERDICT GATE for cleanup B9 (the `deterministic` field add): DeviceConfig must carry
// `bool deterministic` and it must default to TRUE, matching the §9 spec listing
// (architecture.md:578) and gating the §12 stream_count==1 / EmulatedFp64-workspace /
// fixed-order multi-GPU-combine rules the M4.5 parity-recompute path relies on. Those
// rules are inexpressible without this field.
//
// Dual harness (build contract, identical to the sibling host-only tests): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise it is a self-checking main()
// that returns non-zero on the first failure — all CTest needs to gate
// (tests/CMakeLists.txt). No CUDA here.
#include <cstddef>
#include <cstdio>
#include <type_traits>

#include "steppe/config.hpp"  // Precision, DeviceConfig, FilterConfig (CUDA-free)

namespace {

// ---- B9: the `deterministic` field exists, is a bool, and defaults to TRUE -----
// This is the close-out of cleanup B9 / include-config finding 9.1. The §12
// determinism guarantee is GATED on this flag (force stream_count==1 on the
// statistic path; require an explicit cublasSetWorkspace workspace for EmulatedFp64;
// combine the multi-GPU partials in fixed host order) — none of which is expressible
// if the field is absent or defaults off. So we assert BOTH the type and the default.
static_assert(std::is_same_v<decltype(steppe::DeviceConfig::deterministic), bool>,
              "DeviceConfig::deterministic must be a bool (override INTENT, §9/§12)");
static_assert(steppe::DeviceConfig{}.deterministic == true,
              "DeviceConfig::deterministic must default to TRUE (architecture.md §9, §12)");

[[nodiscard]] bool test_deterministic_default_true() {
    const steppe::DeviceConfig cfg{};
    return cfg.deterministic == true;
}

// ---- the §12 sibling defaults the `deterministic` contract is phrased against ---
// `deterministic == true` forces `stream_count == 1` on the statistic path (cuBLAS
// reproducibility does not hold across concurrent streams, §12), so the default pair
// must already be the deterministic-consistent one out of the box: a single statistic
// stream, EmulatedFp64 (whose run-to-run guarantee needs the bound workspace under
// `deterministic`). Pin them so the gated rules start from the documented state.
static_assert(steppe::DeviceConfig{}.stream_count == 1u,
              "default stream_count must be 1 (the bit-stable statistic path, §12)");
static_assert(steppe::DeviceConfig{}.precision.kind == steppe::Precision::Kind::EmulatedFp64,
              "default precision is EmulatedFp64 (architecture.md §9, §12)");

[[nodiscard]] bool test_statistic_path_defaults() {
    const steppe::DeviceConfig cfg{};
    // The default config is internally consistent with `deterministic == true`:
    // single statistic stream + EmulatedFp64 (gated on the bound workspace).
    return cfg.deterministic && cfg.stream_count == 1u &&
           cfg.precision.kind == steppe::Precision::Kind::EmulatedFp64 &&
           cfg.precision.mantissa_bits == steppe::kDefaultMantissaBits;
}

// ---- the rest of the DeviceConfig default surface (regression pins) ------------
[[nodiscard]] bool test_other_device_defaults() {
    const steppe::DeviceConfig cfg{};
    return cfg.devices.empty() &&          // empty ⇒ auto-enumerate (§9)
           cfg.search_streams == 4u &&     // throughput-only search lanes (S8)
           cfg.use_mem_pool == true &&     // pool-backed allocator (§7, §11.2)
           cfg.enable_peer_access == true; // opportunistic P2P (§11.4)
}

struct Case {
    const char* name;
    bool (*fn)();
};

constexpr Case kCases[] = {
    {"DeviceConfig::deterministic defaults to true (B9)", test_deterministic_default_true},
    {"deterministic-consistent statistic-path defaults", test_statistic_path_defaults},
    {"DeviceConfig remaining defaults (devices/search/pool/p2p)", test_other_device_defaults},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(Config, DeviceConfigDefaults) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_config: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_config: all %zu config-contract checks PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif
