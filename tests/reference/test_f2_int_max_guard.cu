// tests/reference/test_f2_int_max_guard.cu
//
// B22 OBJECTIVE GATE — the M0 compute_f2 SNP-count narrowing guard (cleanup
// f2_block_kernel E-1, B22; architecture.md §2 fail-fast, §13). This is the test
// the verdict gate requires: CudaBackend::compute_f2 with an SNP count M > INT_MAX
// must FAIL FAST with a typed exception (std::runtime_error) — never silently
// overflow.
//
// WHY (the bug this pins closed): the M0 WHOLE-matrix path issues the three f2
// GEMMs over ALL M SNPs in one shot via cublasGemmEx, whose contraction extent `k`
// is a signed 32-bit `int` (cuBLAS Library API: cublasGemmEx takes `int m, n, k`;
// only the cublas*Ex_64 variants take int64_t). MatView::M is deliberately `long`
// (views.hpp: large SNP blocks do not overflow 32-bit), and the M0 path does NOT
// tile — it uploads all M. Pre-fix, `const int Mi = static_cast<int>(M)` in
// run_f2_gemms narrowed M into `int` UNCHECKED: for M > INT_MAX the conversion is
// implementation-defined → a wrapped/negative `k` that cuBLAS either rejects
// (CUBLAS_STATUS_INVALID_VALUE) or, worse, SILENTLY contracts over fewer SNPs → a
// wrong-but-plausible f2. The fix adds `if (M > INT_MAX) throw std::runtime_error`
// in compute_f2 (cuda_backend.cu) BEFORE any device allocation or the narrowing,
// and a debug-only STEPPE_ASSERT at the cast site itself (f2_block_kernel.cu).
//
// WHAT IT PINS (data-free, synthetic — a pure control-flow / fail-fast assertion,
// NOT a precision claim, so no real AADR is needed; it runs on every lane):
//   1. compute_f2 with M == (long)INT_MAX + 1 THROWS std::runtime_error — and the
//      guard fires BEFORE touching Q/V/N (data == nullptr on purpose: a too-small/
//      silent narrowing would otherwise have to allocate ~2^31*P*8 bytes, which is
//      itself impossible — the point is the guard never gets there). A clean throw,
//      not a CUDA/cuBLAS error from deep in the runtime and not a silent wrong f2.
//   2. NEGATIVE / POSITIVE CONTROLS that the guard is a TIGHT inequality, not a
//      blanket reject: a tiny VALID block (M well under INT_MAX) still computes a
//      fully populated [P x P] result (the guard did not break the happy path), and
//      M == INT_MAX exactly is NOT rejected by the guard (it is a legal — if
//      astronomically large — extent; it fails later only on the VRAM allocation it
//      genuinely cannot satisfy, which is a DIFFERENT, expected failure, so this
//      control asserts only that it is NOT the B22 message).
//
// ORDERING (important): the happy-path control runs FIRST, on a pristine device.
// The boundary control deliberately provokes an ~86 GB cudaMalloc OOM whose
// cudaErrorMemoryAllocation is surfaced by the NEXT launch's cudaGetLastError(); a
// fresh make_cuda_backend() does NOT reset the device's sticky last-error, so the
// happy path must not follow the OOM-provoking case. The over-limit throws and the
// guard itself allocate nothing (the guard fires first), so they are order-neutral.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `f2_int_max_guard` test (tests/CMakeLists.txt) linking steppe::device. No data.
// Run:  ./test_f2_int_max_guard     (no data needed)
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "steppe/config.hpp"          // Precision, kDefaultMantissaBits
#include "core/internal/views.hpp"    // MatView (Q/V/N contract; M is long)
#include "device/backend.hpp"         // ComputeBackend, F2Result
#include "device/backend_factory.hpp" // steppe::device::make_cuda_backend (X-9/B8)

using steppe::F2Result;
using steppe::Precision;
using steppe::core::MatView;

namespace {

// Assert that compute_f2 with the given M FAILS FAST with std::runtime_error whose
// message names the M0 INT_MAX narrowing (the B22 typed error), WITHOUT touching
// Q/V/N (data == nullptr — the guard must fire before any dereference/allocation).
// Returns true on PASS.
bool expect_int_max_throw(const char* label, long M) {
    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const int P = 5;
    const MatView Q{nullptr, P, M};
    const MatView V{nullptr, P, M};
    const MatView N{nullptr, P, M};
    try {
        auto gpu = steppe::device::make_cuda_backend();
        const F2Result out = gpu->compute_f2(Q, V, N, prec);
        // Reached only if the guard is MISSING: the narrowing wrapped `k` and cuBLAS
        // either silently produced an f2 or the path completed — the exact silent
        // overflow B22 forbids.
        std::printf("  %-30s M=%ld -> RETURNED (f2=%zu) -> FAIL\n", label, M, out.f2.size());
        std::fprintf(stderr,
                     "  [FAIL] %s: compute_f2 did NOT throw on M > INT_MAX — the M0\n"
                     "         static_cast<int>(M) narrowing is UNGUARDED (silent overflow).\n",
                     label);
        return false;
    } catch (const std::runtime_error& e) {
        // The B22 fix: a typed, descriptive fail-fast. Confirm it is the INT_MAX
        // guard (names INT_MAX), not some unrelated runtime_error.
        const bool is_b22 = std::strstr(e.what(), "INT_MAX") != nullptr;
        std::printf("  %-30s M=%ld -> threw runtime_error -> %s\n",
                    label, M, is_b22 ? "PASS" : "FAIL (wrong message)");
        if (!is_b22)
            std::fprintf(stderr,
                         "  [FAIL] %s: threw a runtime_error but not the B22 INT_MAX guard.\n"
                         "         what(): %s\n",
                         label, e.what());
        return is_b22;
    } catch (const std::exception& e) {
        // A CUDA/cuBLAS error from deep in the runtime is NOT the clean typed
        // fail-fast the contract promises.
        std::printf("  %-30s M=%ld -> threw a non-runtime_error -> FAIL\n", label, M);
        std::fprintf(stderr,
                     "  [FAIL] %s: compute_f2 threw a non-std::runtime_error on M > INT_MAX\n"
                     "         (expected the typed B22 fail-fast, not a CUDA/cuBLAS error).\n"
                     "         what(): %s\n",
                     label, e.what());
        return false;
    }
}

// Negative control: M == INT_MAX exactly is NOT the over-limit case, so the B22
// guard must NOT fire. It is an astronomically large extent that fails LATER on the
// genuine VRAM allocation (an EXPECTED, different failure) — so this control passes
// iff compute_f2 does NOT throw the B22 INT_MAX message (whatever else happens). It
// proves the guard is a TIGHT `M > INT_MAX`, not `M >= INT_MAX` or a blanket reject.
// Returns true on PASS.
bool expect_not_b22_at_int_max() {
    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const int P = 5;
    const long M = static_cast<long>(std::numeric_limits<int>::max());  // exactly INT_MAX
    const MatView Q{nullptr, P, M};
    const MatView V{nullptr, P, M};
    const MatView N{nullptr, P, M};
    const char* label = "M==INT_MAX (boundary)";
    try {
        auto gpu = steppe::device::make_cuda_backend();
        const F2Result out = gpu->compute_f2(Q, V, N, prec);
        // Unreachable in practice (the P*M upload cannot fit VRAM), but if it ever
        // does, the guard correctly did NOT reject the boundary — PASS.
        std::printf("  %-30s M=%ld -> returned (no B22 reject) -> PASS\n", label, M);
        return true;
    } catch (const std::exception& e) {
        const bool tripped_b22 = std::strstr(e.what(), "INT_MAX") != nullptr &&
                                 std::strstr(e.what(), "exceeds") != nullptr;
        std::printf("  %-30s M=%ld -> threw (%s) -> %s\n", label, M,
                    tripped_b22 ? "B22 guard" : "other (e.g. DeviceOom on the P*M upload)",
                    tripped_b22 ? "FAIL" : "PASS");
        if (tripped_b22)
            std::fprintf(stderr,
                         "  [FAIL] %s: the B22 guard fired at M==INT_MAX — it must be a\n"
                         "         TIGHT `M > INT_MAX`, not `M >= INT_MAX`. what(): %s\n",
                         label, e.what());
        // Any NON-B22 failure (the expected VRAM/allocation failure) is a PASS for
        // THIS control: it proves the boundary is not B22-rejected.
        return !tripped_b22;
    }
}

// Positive control: a tiny VALID block (M well under INT_MAX) must still compute a
// fully populated result, proving the guard did not break the happy path. Returns
// true on PASS.
bool expect_populated(int P, long M) {
    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    const std::size_t pp = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    std::vector<double> q(pm), v(pm), n(pm);
    for (long j = 0; j < M; ++j)
        for (int i = 0; i < P; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(j);
            q[idx] = 0.05 + 0.9 * static_cast<double>((i * 7 + j * 3) % 97) / 97.0;  // (0,1)
            v[idx] = 1.0;                                                            // all valid
            n[idx] = 2.0 + static_cast<double>((i + j) % 199);                       // n >= 2
        }
    const MatView Q{q.data(), P, M};
    const MatView V{v.data(), P, M};
    const MatView N{n.data(), P, M};
    try {
        auto gpu = steppe::device::make_cuda_backend();
        const F2Result out = gpu->compute_f2(Q, V, N, prec);
        const bool ok = (out.P == P) && (out.f2.size() == pp) && (out.vpair.size() == pp);
        std::printf("  %-30s P=%d M=%ld -> f2=%zu vpair=%zu (expect %zu) -> %s\n",
                    "positive control", P, M, out.f2.size(), out.vpair.size(), pp,
                    ok ? "PASS" : "FAIL");
        if (!ok)
            std::fprintf(stderr,
                         "  [FAIL] positive control: a VALID block produced an empty/wrong-size\n"
                         "         result — the B22 guard broke the happy path.\n");
        return ok;
    } catch (const std::exception& e) {
        std::printf("  %-30s P=%d M=%ld -> THREW -> FAIL\n", "positive control", P, M);
        std::fprintf(stderr, "  [FAIL] positive control THREW: %s\n", e.what());
        return false;
    }
}

}  // namespace

int main() {
    // Fail fast (not "no GPU PASS") if there is no usable device: this is a GPU
    // gate, and a silent skip would hide a regression. The box always has a device.
    int dev_count = 0;
    const cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count < 1) {
        std::fprintf(stderr, "RESULT: FAIL — no CUDA device available (%s); this is a GPU gate.\n",
                     cudaGetErrorString(derr));
        return EXIT_FAILURE;
    }

    std::printf("\nB22 compute_f2 M>INT_MAX narrowing guard (synthetic, no data)\n");

    bool ok = true;
    // (1) HAPPY PATH FIRST, on a pristine device: a tiny VALID block (M well under
    //     INT_MAX) still computes a fully populated result (the guard did not break
    //     it). This MUST precede the OOM-provoking boundary control below — that
    //     control deliberately requests an ~86 GB [P×M] upload whose cudaMalloc
    //     failure (cudaErrorMemoryAllocation) is surfaced by the NEXT launch's
    //     cudaGetLastError(); running the happy path first keeps its assertion on a
    //     clean device (a fresh make_cuda_backend() does not reset the device's
    //     sticky last-error). The guard itself fires before allocation, so its
    //     ordering is immaterial; only this real-allocation control is order-sensitive.
    ok = expect_populated(/*P=*/4, /*M=*/32) && ok;

    // (2) M just over INT_MAX, and a far-over value, must each THROW the typed B22
    //     fail-fast (data == nullptr: the guard fires before any dereference, so no
    //     allocation happens and the device stays clean).
    ok = expect_int_max_throw("M==INT_MAX+1", static_cast<long>(std::numeric_limits<int>::max()) + 1) && ok;
    ok = expect_int_max_throw("M==2^32 (far over)", 1L << 32) && ok;

    // (3) Boundary control LAST: M==INT_MAX is NOT B22-rejected (tight `>`). It
    //     proceeds to a genuine ~86 GB [P×M] allocation it cannot satisfy and fails
    //     with a NON-B22 error (DeviceOom) — an EXPECTED, different failure that
    //     proves the guard is `M > INT_MAX`, not `M >= INT_MAX`. Run last because it
    //     pollutes the device's sticky last-error (see the note above).
    ok = expect_not_b22_at_int_max() && ok;

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — CudaBackend::compute_f2 did not FAIL FAST with a typed\n"
            "        std::runtime_error on an SNP count M > INT_MAX (the M0 cublasGemmEx\n"
            "        `int k` narrowing), or the guard broke the boundary / happy path.\n"
            "        architecture.md §2 fail-fast, §13; cleanup f2_block_kernel E-1, B22.\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS (compute_f2 fails fast with a typed error on M > INT_MAX; "
                         "the INT_MAX boundary and valid blocks are unaffected)\n");
    return EXIT_SUCCESS;
}
