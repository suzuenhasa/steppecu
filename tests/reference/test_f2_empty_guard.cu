// tests/reference/test_f2_empty_guard.cu
//
// B12 OBJECTIVE GATE — the GPU compute_f2 degenerate-extent early-return guard
// (cleanup X-7/E-3, B12; architecture.md §2 fail-fast, §13). This is the test the
// verdict gate requires: CudaBackend::compute_f2 with a zero/negative P or M must
// return an EMPTY F2Result CLEANLY — never throw from deep in the CUDA runtime /
// cuBLAS.
//
// WHY (the bug this pins closed): pre-fix, compute_f2 had no extent guard, while
// its two siblings did (compute_f2_blocks guards `P<=0 || M<=0 || n_block<=0`,
// decode_af guards `P<=0 || M<=0`). So:
//   * M == 0 made core::cdiv(0,16) == 0 — a zero-extent grid the CUDA driver
//     rejects with cudaErrorInvalidConfiguration (thrown by the post-launch
//     STEPPE_CUDA_CHECK_KERNEL), and then Mi==0 as the GEMM contraction extent `k`
//     hit cuBLAS (CUBLAS_STATUS_INVALID_VALUE).
//   * a negative P/M wrapped the size_t extent products to ~1.8e19 ⇒ a huge
//     DeviceBuffer allocation reported as DeviceOom, not the clean empty result the
//     contract intends.
// The fix adds the sibling-consistent `if (P <= 0 || M <= 0) return out;` (out.P
// set, f2/vpair empty) at the top of compute_f2. This is the shape M4.5 multi-GPU
// sharding needs when a device is handed an empty SNP shard.
//
// WHAT IT PINS (data-free, synthetic — a pure control-flow / no-throw assertion,
// NOT a precision claim, so no real AADR is needed; it runs on every lane):
//   1. compute_f2 with M==0, P==0, both-zero, and a NEGATIVE extent each return
//      cleanly (NO exception) an empty F2Result: f2.empty() && vpair.empty(), and
//      out.P carries the given P (matching the compute_f2_blocks/decode_af shape).
//   2. POSITIVE CONTROL: a tiny VALID block (P>0, M>0) still computes a fully
//      populated [P x P] result (f2.size()==vpair.size()==P*P) — the guard did NOT
//      break the happy path.
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `f2_empty_guard` test (tests/CMakeLists.txt) linking steppe::device. No data.
// Run:  ./test_f2_empty_guard     (no data needed)
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

#include "steppe/config.hpp"          // Precision, kDefaultMantissaBits
#include "core/internal/views.hpp"    // MatView (Q/V/N contract)
#include "device/backend.hpp"         // ComputeBackend, F2Result
#include "device/backend_factory.hpp" // steppe::device::make_cuda_backend (X-9/B8)

using steppe::F2Result;
using steppe::Precision;
using steppe::core::MatView;

namespace {

// Drive compute_f2 once with the given (degenerate) extents through the PRODUCTION
// GPU backend, asserting it returns an EMPTY F2Result WITHOUT throwing. `data` may
// be null for the degenerate cases — the guard must early-return before it ever
// dereferences Q/V/N. Returns true on PASS.
bool expect_empty_clean(const char* label, int P, long M, const double* data) {
    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    const MatView Q{data, P, M};
    const MatView V{data, P, M};
    const MatView N{data, P, M};
    try {
        auto gpu = steppe::device::make_cuda_backend();
        const F2Result out = gpu->compute_f2(Q, V, N, prec);
        const bool empty_ok = out.f2.empty() && out.vpair.empty();
        const bool p_ok = (out.P == P);  // out.P carries the given P (sibling shape)
        const bool ok = empty_ok && p_ok;
        std::printf("  %-22s P=%d M=%ld -> returned cleanly; f2=%zu vpair=%zu P=%d -> %s\n",
                    label, P, M, out.f2.size(), out.vpair.size(), out.P,
                    ok ? "PASS" : "FAIL");
        if (!empty_ok)
            std::fprintf(stderr, "  [FAIL] %s: expected EMPTY f2/vpair, got f2=%zu vpair=%zu\n",
                         label, out.f2.size(), out.vpair.size());
        if (!p_ok)
            std::fprintf(stderr, "  [FAIL] %s: expected out.P==%d, got %d\n", label, P, out.P);
        return ok;
    } catch (const std::exception& e) {
        // The pre-fix behavior: a throw from deep in the CUDA runtime / cuBLAS.
        std::printf("  %-22s P=%d M=%ld -> THREW -> FAIL\n", label, P, M);
        std::fprintf(stderr,
                     "  [FAIL] %s: compute_f2 THREW on a degenerate extent instead of\n"
                     "         returning an empty F2Result (the B12 guard is missing).\n"
                     "         what(): %s\n",
                     label, e.what());
        return false;
    }
}

// Positive control: a tiny VALID block must still compute a fully populated result,
// proving the guard did not break the happy path. Returns true on PASS.
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
        std::printf("  %-22s P=%d M=%ld -> f2=%zu vpair=%zu (expect %zu) -> %s\n",
                    "positive control", P, M, out.f2.size(), out.vpair.size(), pp,
                    ok ? "PASS" : "FAIL");
        if (!ok)
            std::fprintf(stderr,
                         "  [FAIL] positive control: a VALID block produced an empty/wrong-size\n"
                         "         result — the B12 guard broke the happy path.\n");
        return ok;
    } catch (const std::exception& e) {
        std::printf("  %-22s P=%d M=%ld -> THREW -> FAIL\n", "positive control", P, M);
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

    std::printf("\nB12 compute_f2 degenerate-extent guard (synthetic, no data)\n");

    bool ok = true;
    // (1) Degenerate extents must return an empty F2Result cleanly (no throw).
    //     data == nullptr on purpose: the guard must early-return before touching it.
    ok = expect_empty_clean("M==0 (zero-grid)", /*P=*/5, /*M=*/0,  nullptr) && ok;
    ok = expect_empty_clean("P==0",             /*P=*/0, /*M=*/64, nullptr) && ok;
    ok = expect_empty_clean("P==0 && M==0",     /*P=*/0, /*M=*/0,  nullptr) && ok;
    ok = expect_empty_clean("M<0 (negative)",   /*P=*/5, /*M=*/-1, nullptr) && ok;
    ok = expect_empty_clean("P<0 (negative)",   /*P=*/-1, /*M=*/64, nullptr) && ok;

    // (2) Positive control: a valid tiny block still produces a populated result.
    ok = expect_populated(/*P=*/4, /*M=*/32) && ok;

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — CudaBackend::compute_f2 did not return an empty F2Result\n"
            "        cleanly on a degenerate (P<=0 || M<=0) extent (or the guard broke the\n"
            "        happy path). It must match its siblings compute_f2_blocks / decode_af\n"
            "        (architecture.md §2 fail-fast; cleanup X-7/E-3, B12).\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS (compute_f2 early-returns an empty F2Result on a "
                         "degenerate extent; valid blocks still compute)\n");
    return EXIT_SUCCESS;
}
