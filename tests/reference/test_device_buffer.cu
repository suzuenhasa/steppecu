// tests/reference/test_device_buffer.cu
//
// B23 OBJECTIVE GATE — the DeviceBuffer<T> construction-time `n * sizeof(T)`
// overflow guard (cleanup device_buffer 1.1/7.5, B23; architecture.md §2
// fail-fast, §11.2 VRAM budget, §13). This is the test the verdict gate
// requires: constructing DeviceBuffer<T> with `n` near SIZE_MAX/sizeof(T) must
// THROW a typed error — never SILENTLY WRAP and under-allocate.
//
// WHY (the bug this pins closed): the ctor's byte request `n * sizeof(T)` is a
// `std::size_t` product. Unsigned overflow is *defined* (modular), so an
// UNCHECKED multiply does NOT trap — for `n > SIZE_MAX/sizeof(T)` it WRAPS to a
// small value, `cudaMalloc` succeeds with a buffer far smaller than `size_`
// advertises, and every downstream kernel/copy over-runs it: silent heap
// corruption, the exact opposite of §2's "fail-fast, not silent corruption". `n`
// is NOT bounded by hardware at this layer — cuda_backend.cu forms buffer sizes
// as products of three host values BEFORE any §11.2 VRAM-budget check sees them
// — so the single-source allocation owner is the place to make this safe once.
// The B23 fix adds, in the ctor BEFORE the multiply,
//   `if (n > SIZE_MAX/sizeof(T)) throw CudaError(cudaErrorMemoryAllocation, ...)`,
// a typed throw the public API maps to STEPPE_ERR_DEVICE_OOM (architecture.md
// §10): the request is, by definition, larger than any allocatable size. With
// the ctor invariant established, `bytes()` is then always EXACT (it can never
// observe an `n` whose product overflows), which the §11.2 budget relies on.
//
// WHAT IT PINS (data-free, synthetic — a pure control-flow / fail-fast assertion,
// NOT a precision claim, so no real AADR is needed; it runs on every lane):
//   1. OVERFLOW REJECT, no silent wrap: DeviceBuffer<double> with
//      n == SIZE_MAX/sizeof(double) + 1 (the smallest n whose *8 wraps) THROWS,
//      and so does the maximal n == SIZE_MAX (which wraps hard to SIZE_MAX-7).
//      Asserted for sizeof(T)==8 (double) AND sizeof(T)==4 (std::int32_t-shaped),
//      since the wrap point n > SIZE_MAX/sizeof(T) is sizeof-dependent.
//   2. BOUNDARY ALLOWED (tight inequality): n == SIZE_MAX/sizeof(T) EXACTLY is
//      NOT over the limit (its byte product is <= SIZE_MAX), so the B23 guard must
//      NOT reject it — it must proceed to a genuine cudaMalloc that fails with a
//      DIFFERENT, expected DeviceOom (an ~2.3e18-byte request no GPU can satisfy).
//      This proves the guard is `n > SIZE_MAX/sizeof(T)`, not `>=` or a blanket
//      reject. (The boundary case for sizeof(T)==1 is SIZE_MAX itself — no n can
//      overflow a 1-byte element — so for `std::byte` the guard NEVER fires; the
//      maximal request still fails on the real allocation, not on B23.)
//   3. HAPPY PATH UNTOUCHED: a small DeviceBuffer<double>(n) allocates a non-null
//      pointer, reports size()==n and bytes()==n*sizeof(double) EXACTLY, and a
//      zero-size buffer is {nullptr, 0, 0} — the guard did not break construction.
//   4. MOVE SEMANTICS (the move-only RAII shape the §13 sanitizer protects on an
//      otherwise un-exercised path): move-construct NULLS the source, move-assign
//      frees-then-steals, self-move-assign is a no-op — run under the box's
//      compute-sanitizer would catch a double-free regression here.
//
// ORDERING: the boundary controls (case 2) provoke a genuine ~2.3e18-byte
// cudaMalloc OOM whose cudaErrorMemoryAllocation is sticky on the device, so they
// run AFTER the happy-path / move cases (which assert on a clean device). The
// overflow rejects (case 1) and the guard itself allocate nothing (the guard
// fires before cudaMalloc), so they are order-neutral.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `device_buffer_unit` test (tests/CMakeLists.txt) linking steppe::device's
// device-private headers (device_buffer.cuh is a .cuh — PRIVATE to steppe_device,
// architecture.md §4 — so this gate is a CUDA TU, not a host-only unit test). No
// data. Run:  ./test_device_buffer   (no data needed)
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>

#include "device/cuda/check.cuh"          // CudaError (the typed throw)
#include "device/cuda/device_buffer.cuh"  // DeviceBuffer<T> (the unit under test)

using steppe::device::CudaError;
using steppe::device::DeviceBuffer;

namespace {

constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

// Assert that constructing DeviceBuffer<T>(n) FAILS FAST with the typed B23
// overflow error (a CudaError carrying cudaErrorMemoryAllocation whose message
// names the overflow), WITHOUT any silent wrap / under-allocation. The guard
// fires before cudaMalloc, so no device allocation happens. Returns true on PASS.
template <class T>
bool expect_overflow_throw(const char* label, std::size_t n) {
    try {
        DeviceBuffer<T> buf(n);
        // Reached only if the guard is MISSING: the multiply wrapped to a small
        // byte count, cudaMalloc succeeded with an under-sized buffer, and size()
        // now advertises far more than was allocated — the exact silent
        // under-allocation B23 forbids.
        std::printf("  %-34s n=%zu sizeof(T)=%zu -> CONSTRUCTED (size=%zu, bytes=%zu) -> FAIL\n",
                    label, n, sizeof(T), buf.size(), buf.bytes());
        std::fprintf(stderr,
                     "  [FAIL] %s: DeviceBuffer<T>(n) did NOT throw on n*sizeof(T) overflow —\n"
                     "         the construction-time multiply is UNGUARDED (silent under-alloc).\n",
                     label);
        return false;
    } catch (const CudaError& e) {
        // The B23 fix: a typed, descriptive fail-fast that maps to DeviceOom.
        const bool names_overflow = std::strstr(e.what(), "overflow") != nullptr;
        const bool is_oom = (e.status() == cudaErrorMemoryAllocation);
        const bool ok = names_overflow && is_oom;
        std::printf("  %-34s n=%zu sizeof(T)=%zu -> threw CudaError -> %s\n",
                    label, n, sizeof(T), ok ? "PASS" : "FAIL (wrong message/status)");
        if (!ok)
            std::fprintf(stderr,
                         "  [FAIL] %s: threw a CudaError but not the B23 overflow guard\n"
                         "         (status==cudaErrorMemoryAllocation? %d; what(): %s)\n",
                         label, static_cast<int>(is_oom), e.what());
        return ok;
    } catch (const std::exception& e) {
        std::printf("  %-34s n=%zu sizeof(T)=%zu -> threw a non-CudaError -> FAIL\n",
                    label, n, sizeof(T));
        std::fprintf(stderr,
                     "  [FAIL] %s: DeviceBuffer<T>(n) threw a non-CudaError on overflow\n"
                     "         (expected the typed B23 fail-fast). what(): %s\n",
                     label, e.what());
        return false;
    }
}

// Boundary control: n == SIZE_MAX/sizeof(T) EXACTLY is NOT over the limit, so the
// B23 guard must NOT fire. It proceeds to a genuine ~2.3e18-byte cudaMalloc it
// cannot satisfy and fails with a DIFFERENT error (DeviceOom from cudaMalloc, NOT
// the B23 overflow message) — proving the guard is a TIGHT `n > SIZE_MAX/sizeof(T)`,
// not `>=`. Passes iff the construction does NOT throw the B23 overflow message
// (whatever else happens). Returns true on PASS.
template <class T>
bool expect_boundary_not_b23(const char* label) {
    const std::size_t n = kSizeMax / sizeof(T);  // largest n whose *sizeof(T) <= SIZE_MAX
    try {
        DeviceBuffer<T> buf(n);
        // Astronomically unlikely (no GPU has 2.3e18 bytes), but if it ever
        // succeeds the guard correctly did NOT reject the boundary — PASS.
        std::printf("  %-34s n=%zu sizeof(T)=%zu -> constructed (no B23 reject) -> PASS\n",
                    label, n, sizeof(T));
        return true;
    } catch (const std::exception& e) {
        const bool tripped_b23 = std::strstr(e.what(), "overflow") != nullptr;
        std::printf("  %-34s n=%zu sizeof(T)=%zu -> threw (%s) -> %s\n",
                    label, n, sizeof(T),
                    tripped_b23 ? "B23 overflow guard" : "other (e.g. DeviceOom from cudaMalloc)",
                    tripped_b23 ? "FAIL" : "PASS");
        if (tripped_b23)
            std::fprintf(stderr,
                         "  [FAIL] %s: the B23 guard fired at n==SIZE_MAX/sizeof(T) — it must be\n"
                         "         a TIGHT `n > SIZE_MAX/sizeof(T)`, not `>=`. what(): %s\n",
                         label, e.what());
        // Any NON-B23 failure (the expected cudaMalloc DeviceOom) is a PASS for
        // THIS control: it proves the boundary is not B23-rejected.
        return !tripped_b23;
    }
}

// Happy-path control: a small DeviceBuffer<double>(n) allocates a non-null pointer
// with EXACT size()/bytes(), and a zero-size buffer is {nullptr, 0, 0}. Proves the
// B23 guard did not break normal construction (and that bytes() is exact under the
// invariant). Returns true on PASS.
bool expect_happy_path() {
    bool ok = true;
    try {
        const std::size_t n = 1024;
        DeviceBuffer<double> buf(n);
        const bool small_ok = (buf.data() != nullptr) && (buf.size() == n) &&
                              (buf.bytes() == n * sizeof(double));
        std::printf("  %-34s n=%zu -> data=%p size=%zu bytes=%zu -> %s\n",
                    "happy path (small alloc)", n, static_cast<const void*>(buf.data()),
                    buf.size(), buf.bytes(), small_ok ? "PASS" : "FAIL");
        if (!small_ok) {
            std::fprintf(stderr,
                         "  [FAIL] happy path: a valid small DeviceBuffer<double>(n) did not\n"
                         "         allocate a non-null exact-size buffer — the B23 guard broke it.\n");
            ok = false;
        }

        DeviceBuffer<double> empty(0);
        const bool empty_ok = (empty.data() == nullptr) && (empty.size() == 0) &&
                             (empty.bytes() == 0);
        std::printf("  %-34s n=0 -> data=%p size=%zu bytes=%zu -> %s\n",
                    "happy path (zero-size)", static_cast<const void*>(empty.data()),
                    empty.size(), empty.bytes(), empty_ok ? "PASS" : "FAIL");
        if (!empty_ok) {
            std::fprintf(stderr, "  [FAIL] happy path: zero-size buffer was not {nullptr, 0, 0}.\n");
            ok = false;
        }
    } catch (const std::exception& e) {
        std::printf("  %-34s -> THREW -> FAIL\n", "happy path");
        std::fprintf(stderr, "  [FAIL] happy path THREW: %s\n", e.what());
        ok = false;
    }
    return ok;
}

// Move-only RAII control: move-construct nulls the source, move-assign
// frees-then-steals, self-move-assign is a no-op. The §13 compute-sanitizer
// protects this otherwise un-exercised path (a double-free regression would
// surface here). Returns true on PASS.
bool expect_move_semantics() {
    bool ok = true;
    try {
        DeviceBuffer<double> a(256);
        const double* a_ptr = a.data();
        DeviceBuffer<double> b(std::move(a));
        const bool moved =
            (a.data() == nullptr) && (a.size() == 0) && (b.data() == a_ptr) && (b.size() == 256);
        std::printf("  %-34s -> %s\n", "move-construct nulls source", moved ? "PASS" : "FAIL");
        if (!moved) {
            std::fprintf(stderr, "  [FAIL] move-construct did not null the source / steal the pointer.\n");
            ok = false;
        }

        DeviceBuffer<double> c(128);
        c = std::move(b);  // frees c's original, steals b's
        const bool assigned =
            (b.data() == nullptr) && (b.size() == 0) && (c.data() == a_ptr) && (c.size() == 256);
        std::printf("  %-34s -> %s\n", "move-assign frees-then-steals", assigned ? "PASS" : "FAIL");
        if (!assigned) {
            std::fprintf(stderr, "  [FAIL] move-assign did not free-then-steal correctly.\n");
            ok = false;
        }

        // Self-move-assign must be a no-op (the `if (this != &o)` guard), never a
        // free-then-use. Route through a reference alias so the literal
        // `c = std::move(c)` is hidden from -Wself-move (warnings-as-errors) while
        // the SAME runtime self-assignment path is still exercised.
        DeviceBuffer<double>& c_alias = c;
        c = std::move(c_alias);
        const bool self_ok = (c.data() == a_ptr) && (c.size() == 256);
        std::printf("  %-34s -> %s\n", "self-move-assign is a no-op", self_ok ? "PASS" : "FAIL");
        if (!self_ok) {
            std::fprintf(stderr, "  [FAIL] self-move-assign was not a no-op (freed its own buffer).\n");
            ok = false;
        }
    } catch (const std::exception& e) {
        std::printf("  %-34s -> THREW -> FAIL\n", "move semantics");
        std::fprintf(stderr, "  [FAIL] move semantics THREW: %s\n", e.what());
        ok = false;
    }
    return ok;
}

}  // namespace

int main() {
    // Fail fast (not "no GPU PASS") if there is no usable device: the happy-path /
    // move / boundary cases touch the CUDA runtime, and a silent skip would hide a
    // regression. The overflow rejects (the core B23 assertion) need no GPU — the
    // guard fires before cudaMalloc — but the box always has a device.
    int dev_count = 0;
    const cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count < 1) {
        std::fprintf(stderr, "RESULT: FAIL — no CUDA device available (%s); this is a GPU gate.\n",
                     cudaGetErrorString(derr));
        return EXIT_FAILURE;
    }

    std::printf("\nB23 DeviceBuffer<T> n*sizeof(T) overflow guard (synthetic, no data)\n");

    bool ok = true;

    // (3) HAPPY PATH + (4) MOVE first, on a clean device (the boundary controls
    //     below pollute the device's sticky last-error with a 2.3e18-byte
    //     cudaMalloc OOM).
    ok = expect_happy_path() && ok;
    ok = expect_move_semantics() && ok;

    // (1) OVERFLOW REJECT, no silent wrap. The smallest overflowing n is
    //     SIZE_MAX/sizeof(T) + 1; SIZE_MAX itself overflows for any sizeof(T) > 1.
    //     The guard allocates nothing, so these are order-neutral.
    ok = expect_overflow_throw<double>("double: SIZE_MAX/8 + 1",
                                       kSizeMax / sizeof(double) + 1) && ok;
    ok = expect_overflow_throw<double>("double: SIZE_MAX (hard wrap)", kSizeMax) && ok;
    ok = expect_overflow_throw<std::int32_t>("int32: SIZE_MAX/4 + 1",
                                             kSizeMax / sizeof(std::int32_t) + 1) && ok;
    ok = expect_overflow_throw<std::int32_t>("int32: SIZE_MAX (hard wrap)", kSizeMax) && ok;

    // (2) BOUNDARY ALLOWED (tight `>`): n == SIZE_MAX/sizeof(T) EXACTLY is NOT
    //     over the limit; the guard must NOT fire (it fails later on the genuine
    //     cudaMalloc, a DIFFERENT expected DeviceOom). Run LAST: provokes a real
    //     ~2.3e18-byte allocation whose failure is sticky on the device.
    ok = expect_boundary_not_b23<double>("double: boundary n==SIZE_MAX/8") && ok;
    ok = expect_boundary_not_b23<std::int32_t>("int32: boundary n==SIZE_MAX/4") && ok;

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — DeviceBuffer<T>(n) did not FAIL FAST with a typed CudaError on an\n"
            "        n for which n*sizeof(T) overflows size_t (the construction-time multiply\n"
            "        silently wrapped / under-allocated), or the guard broke the boundary /\n"
            "        happy path / move semantics. architecture.md §2 fail-fast, §11.2, §13;\n"
            "        cleanup device_buffer 1.1/7.5, B23.\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr,
                 "RESULT: PASS (DeviceBuffer<T> rejects n*sizeof(T) overflow with a typed\n"
                 "        CudaError; the SIZE_MAX/sizeof(T) boundary, the happy path, and the\n"
                 "        move-only RAII shape are unaffected; bytes() is exact)\n");
    return EXIT_SUCCESS;
}
