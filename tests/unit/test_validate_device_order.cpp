// tests/unit/test_validate_device_order.cpp
//
// B8 / cleanup device-resources C1+T1 verdict gate (host-only, GPU-FREE).
//
// Pins the §9 build()-validation contract — reject duplicate / out-of-range device
// ordinals — that build_resources now enforces, exercised through the PURE,
// CUDA-FREE predicate steppe::device::validate_device_order(span<const int>, int)
// it was factored into (architecture.md §9 "Validation at build() rejects ... any
// device id in DeviceConfig::devices that is absent or duplicated", §2 fail-fast,
// §13 testing; cleanup device-resources C1/T1, B8).
//
// WHY THIS IS HOST-ONLY / GPU-FREE: validate_device_order is pure host arithmetic
// over (order, visible) — it touches NO CUDA, no device, no allocation beyond a
// `seen` bitmap. The old code entangled this validation with a real device probe
// (the only coverage was a GPU `.cu`), so the §13 "exercisable GPU-free" duty went
// unmet and the duplicate-ordinal case was not validated AT ALL (it silently ran
// two lanes on one GPU). This TU runs the validator directly on chosen
// (order, visible) pairs with no GPU.
//
// The validator is out-of-line in steppe_device (resources.cpp), so this TU links
// steppe::device to RESOLVE it — but it calls only validate_device_order, never a
// device-touching symbol, so no GPU is needed at run time (the function makes no
// CUDA call). resources.hpp is reached as "device/resources.hpp" via the src/ root
// on steppe::device's PUBLIC include dir. Dual harness (GoogleTest when present,
// else a self-checking main(); CTest gates on the exit code either way) — mirrors
// tests/unit/test_f2_combine.cpp.

#include <span>
#include <stdexcept>
#include <vector>

#include "device/resources.hpp"  // steppe::device::validate_device_order (the unit under test)

using steppe::device::validate_device_order;

namespace {

// A bad-ordinal/duplicate case is expected to THROW; a valid case must NOT.
bool throws_for(const std::vector<int>& order, int visible) {
    try {
        validate_device_order(std::span<const int>(order.data(), order.size()), visible);
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

}  // namespace

#if defined(STEPPE_TEST_WITH_GTEST)
#include <gtest/gtest.h>

TEST(ValidateDeviceOrder, ValidOrdersPass) {
    // The fixed g=0..G-1 orders build_resources actually produces all pass.
    EXPECT_FALSE(throws_for({0}, 1));            // single-GPU
    EXPECT_FALSE(throws_for({0, 1}, 2));         // dense 2-GPU
    EXPECT_FALSE(throws_for({1, 0}, 2));         // reordered (a non-default but legal pin)
    EXPECT_FALSE(throws_for({0, 1, 2, 3}, 4));   // dense 4-GPU
    EXPECT_FALSE(throws_for({2}, 4));            // a single non-zero device
    EXPECT_FALSE(throws_for({}, 2));             // empty span is a no-op here (build_resources owns the empty fail-fast)
    EXPECT_FALSE(throws_for({}, 0));             // empty + no device: still a no-op here
}

TEST(ValidateDeviceOrder, DuplicateOrdinalThrows) {
    // §9 reject: a repeated member is ill-formed (two lanes on one GPU, §11.4/§12).
    EXPECT_TRUE(throws_for({0, 0}, 2));          // the headline footgun (C1)
    EXPECT_TRUE(throws_for({1, 1}, 2));
    EXPECT_TRUE(throws_for({0, 1, 0}, 2));       // duplicate of an in-range ordinal
    EXPECT_TRUE(throws_for({0, 1, 1, 2}, 4));
}

TEST(ValidateDeviceOrder, OutOfRangeOrdinalThrows) {
    // §9 reject: an ordinal not among the `visible` devices.
    EXPECT_TRUE(throws_for({0, 5}, 2));          // 5 >= visible
    EXPECT_TRUE(throws_for({2}, 2));             // == visible (ordinals are 0-based)
    EXPECT_TRUE(throws_for({-1}, 2));            // negative
    EXPECT_TRUE(throws_for({0}, 0));             // a device on a zero-visible box
    EXPECT_TRUE(throws_for({0, 1}, 1));          // second ordinal out of range
}

TEST(ValidateDeviceOrder, OutOfRangeCheckedBeforeDuplicate) {
    // An out-of-range ordinal must be caught even when a later duplicate exists —
    // the out-of-range test gates the `seen` index, so it cannot index OOB.
    EXPECT_TRUE(throws_for({5, 5}, 2));
}

#else  // self-checking fallback (no GoogleTest)

#include <cstdio>

int main() {
    int failures = 0;
    auto expect = [&](const char* label, bool cond) {
        std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
        if (!cond) ++failures;
    };

    // Valid orders (the g=0..G-1 pins build_resources produces) pass.
    expect("{0}/1 valid", !throws_for({0}, 1));
    expect("{0,1}/2 valid", !throws_for({0, 1}, 2));
    expect("{1,0}/2 reordered valid", !throws_for({1, 0}, 2));
    expect("{0,1,2,3}/4 valid", !throws_for({0, 1, 2, 3}, 4));
    expect("{2}/4 single non-zero valid", !throws_for({2}, 4));
    expect("{}/2 empty no-op", !throws_for({}, 2));
    expect("{}/0 empty+no-device no-op", !throws_for({}, 0));

    // Duplicate ordinals throw (§9; the C1 footgun).
    expect("{0,0}/2 duplicate throws", throws_for({0, 0}, 2));
    expect("{1,1}/2 duplicate throws", throws_for({1, 1}, 2));
    expect("{0,1,0}/2 duplicate throws", throws_for({0, 1, 0}, 2));
    expect("{0,1,1,2}/4 duplicate throws", throws_for({0, 1, 1, 2}, 4));

    // Out-of-range ordinals throw (§9).
    expect("{0,5}/2 out-of-range throws", throws_for({0, 5}, 2));
    expect("{2}/2 ==visible throws", throws_for({2}, 2));
    expect("{-1}/2 negative throws", throws_for({-1}, 2));
    expect("{0}/0 device-on-zero-box throws", throws_for({0}, 0));
    expect("{0,1}/1 second out-of-range throws", throws_for({0, 1}, 1));

    // Out-of-range checked before the duplicate index (no OOB on `seen`).
    expect("{5,5}/2 out-of-range (not OOB) throws", throws_for({5, 5}, 2));

    if (failures != 0) {
        std::fprintf(stderr, "[validate_device_order] %d failure(s)\n", failures);
        return 1;
    }
    std::printf("[validate_device_order] OK: §9 duplicate/out-of-range fail-fast "
                "(B8/C1/T1)\n");
    return 0;
}

#endif
