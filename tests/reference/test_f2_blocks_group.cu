// tests/reference/test_f2_blocks_group.cu
//
// B24 OBJECTIVE GATE — the M4 grouped-batch wrappers' empty/degenerate fail-fast
// (cleanup F6/B24; architecture.md §2 fail-fast, §13). This is the test the verdict
// gate requires for B24: the three device-private launch wrappers in
// f2_blocks_kernel.cuh — launch_gather_group / run_f2_gemms_group /
// launch_assemble_blocks_group — must fail FAST on a degenerate batch (n_in_group
// <= 0, a zero gridDim.z = invalid launch) or a zero pad width (s_pad <= 0, a zero
// gridDim.y / a k==0 GEMM), rather than silently issuing an invalid launch or a
// beta-only GEMM and producing a wrong/empty result.
//
// WHY (the shape this pins, F6): the M4 backend always satisfies n_in_group >= 1
// (a bucket has >= 1 block) and s_pad >= 1 (the bucket width is ceil-pow2 of a
// block's SNP count, >= kBlockGroupPadBase), so these never fire from the sole
// caller TODAY. But the wrappers accept bare ints with no contract, and M4.5
// multi-GPU sharding can hand a device an empty SNP shard => an empty bucket =>
// n_in_group == 0. With n_in_group == 0 the gather/assemble launches set gridDim.z
// = 0 (the driver rejects a zero grid dimension with cudaErrorInvalidConfiguration)
// and run_f2_gemms_group passes batchCount == 0 (a degenerate cublasGemmStridedBatchedEx);
// with s_pad == 0 the gather sets gridDim.y == grid_for(0) == 0 (zero y-extent) and
// the GEMM contraction extent k == 0 (a beta-only scale). The B24 fix routes
// n_in_group through grid_z_extent in the two kernel wrappers (its STEPPE_ASSERT
// pins 1 <= n <= kMaxGridZ) and adds explicit STEPPE_ASSERT(s_pad >= 1) to the
// gather wrapper plus STEPPE_ASSERT(n_in_group >= 1 && s_pad >= 1) to the pure-cuBLAS
// run_f2_gemms_group (which issues no <<<>>> and so never reaches grid_z_extent).
//
// WHAT IT PINS (data-free, synthetic — control-flow / launch-validity, NOT a
// precision claim, so no real AADR is needed; it runs on every lane):
//   1. POSITIVE CONTROL: a valid synthetic 2-block, REORDERED block_ids_in_group
//      (= {1, 0}, a genuine permutation, not the identity the single-block==M0 arm
//      exercises) drives all three wrappers end-to-end and the scattered [P×P×n_block]
//      f2 + Vpair tensors match a hand-computed host oracle (the SAME shared
//      assemble_f2_numerator/finalize_f2 the GPU calls), block-for-block. This also
//      exercises the gather's pad-column zero-fill (block sizes 2 and 3, both padded
//      to s_pad=4) and the scatter-to-arbitrary-id slab offset — the M4-specific
//      index math (the B25 gather coverage the B24 gate references, folded in here
//      so this gate is self-standing).
//   2. DEATH CASES (only when STEPPE_ASSERT is active, i.e. !NDEBUG — under NDEBUG
//      the guard is compiled out by contract and these are SKIPPED): each wrapper,
//      called with a degenerate extent, must ABORT (SIGABRT from the fired
//      STEPPE_ASSERT) in a forked child:
//        - launch_gather_group        : n_in_group == 0, n_in_group == -1, s_pad == 0
//        - run_f2_gemms_group         : n_in_group == 0, s_pad == 0
//        - launch_assemble_blocks_group: n_in_group == 0, n_in_group == -1
//
// Build (REMOTE sm_120 / CUDA 13 box; NOT locally). Built by CMake/CTest as the
// `f2_blocks_group` test (tests/CMakeLists.txt) linking steppe::device for the
// device-private wrappers + the RAII handle, and steppe::core_internal for the
// shared host/device f2 primitive (the oracle). No data.
// Run:  ./test_f2_blocks_group     (no data needed)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/internal/f2_estimator.hpp"   // assemble_f2_numerator, finalize_f2 (the oracle)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK
#include "device/cuda/device_buffer.cuh"    // DeviceBuffer<T> (RAII device memory)
#include "device/cuda/f2_blocks_kernel.cuh" // launch_gather_group, run_f2_gemms_group, launch_assemble_blocks_group
#include "device/cuda/handles.hpp"          // CublasHandle (RAII cuBLAS handle)

using steppe::Precision;
using steppe::core::assemble_f2_numerator;
using steppe::core::finalize_f2;
using steppe::device::CublasHandle;
using steppe::device::DeviceBuffer;

namespace {

// ===========================================================================
// Synthetic feeder-output layout (what the gather reads), all column-major:
//   Q_all [P × M]  : masked allele freq q (0 on invalid)
//   V_all [P × M]  : validity mask in {0,1}
//   S_all [2P × M] : top P rows = Qsq (= v·q²), bottom P rows = Hc (het corr)
// The gather selects each block's contiguous columns [off, off+sz) into a padded
// slab; pad columns (c >= sz) are zero-filled. We pick tiny, fully-determined
// values so the host oracle below reproduces every GEMM sum exactly.
// ===========================================================================
constexpr int kP = 2;       // populations
constexpr int kSpad = 4;    // bucket pad width (a power of kBlockGroupPadBase=2)
constexpr int kNblock = 2;  // resident tensor slabs

// Two blocks: block 0 spans SNP columns [0, 2) (size 2), block 1 spans [2, 5)
// (size 3). M = 5 contiguous feeder columns. Both pad to s_pad = 4.
constexpr long kM = 5;
const long kOffsets[kNblock] = {0, 2};
const int kSizes[kNblock] = {2, 3};

// Per (population i, SNP column src): a fully-determined q and validity. We make
// EVERY entry valid (v=1) so Vpair == the block's SNP count (the pad-column-zero
// property is then visible as "Vpair is the REAL size, not s_pad").
double q_at(int i, long src) {
    // deterministic, in (0,1): varies with both i and src.
    return 0.1 + 0.15 * static_cast<double>((i * 3 + static_cast<int>(src) * 2) % 5);
}
double n_at(int /*i*/, long src) {
    return 4.0 + static_cast<double>(src);  // non-missing haploid count, >= 2
}

// Build the three feeder-output host arrays from q_at/n_at.
void build_feeder(std::vector<double>& Qall, std::vector<double>& Vall,
                  std::vector<double>& Sall) {
    const std::size_t pm = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kM);
    Qall.assign(pm, 0.0);
    Vall.assign(pm, 0.0);
    Sall.assign(2u * pm, 0.0);
    for (long s = 0; s < kM; ++s) {
        for (int i = 0; i < kP; ++i) {
            const double q = q_at(i, s);
            const double n = n_at(i, s);
            const double hc = steppe::core::het_correction(q, n, /*valid=*/true);
            const std::size_t qi = static_cast<std::size_t>(i) + static_cast<std::size_t>(kP) * static_cast<std::size_t>(s);
            Qall[qi] = q;        // masked q (valid => q)
            Vall[qi] = 1.0;      // valid
            // S is [2P × M]: row i = Qsq, row P+i = Hc.
            const std::size_t s_qsq = static_cast<std::size_t>(i) + 2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(s);
            const std::size_t s_hc = (static_cast<std::size_t>(kP) + static_cast<std::size_t>(i)) + 2u * static_cast<std::size_t>(kP) * static_cast<std::size_t>(s);
            Sall[s_qsq] = q * q;  // Qsq = v·q² (v=1)
            Sall[s_hc] = hc;      // Hc
        }
    }
}

// Host oracle: f2(i,j) and Vpair(i,j) for one block (columns [off, off+sz)),
// reproducing the three GEMM sums and the shared assemble/finalize primitives over
// the REAL (unpadded) columns. The wrappers' gather pads to s_pad with zeros, which
// must be inert — so this oracle (summing only [off, off+sz)) is the ground truth.
void oracle_block(long off, int sz, int i, int j, double& f2_out, double& vpair_out) {
    double G = 0.0, Vpair = 0.0, sumsq_i = 0.0, sumsq_j = 0.0, hsum_i = 0.0, hsum_j = 0.0;
    for (int c = 0; c < sz; ++c) {
        const long src = off + c;
        const double qi = q_at(i, src), qj = q_at(j, src);
        const double ni = n_at(i, src), nj = n_at(j, src);
        const double hci = steppe::core::het_correction(qi, ni, true);
        const double hcj = steppe::core::het_correction(qj, nj, true);
        G += qi * qj;
        Vpair += 1.0;                 // both always valid
        sumsq_i += (qi * qi) * 1.0;   // Σ Qsq_i · v_j
        sumsq_j += (qj * qj) * 1.0;
        hsum_i += hci * 1.0;          // Σ Hc_i · v_j
        hsum_j += hcj * 1.0;
    }
    const double num = assemble_f2_numerator(sumsq_i, sumsq_j, G, hsum_i, hsum_j);
    f2_out = finalize_f2(num, Vpair);
    vpair_out = Vpair;
}

// ---------------------------------------------------------------------------
// Positive control: drive all three wrappers on the REORDERED 2-block layout and
// diff the scattered tensors against the host oracle. Returns true on PASS.
// ---------------------------------------------------------------------------
bool positive_control() {
    std::vector<double> Qall, Vall, Sall;
    build_feeder(Qall, Vall, Sall);

    // A genuine permutation: slab 0 -> global block 1, slab 1 -> global block 0.
    const int n_in_group = kNblock;
    const std::vector<int> ids_in_group = {1, 0};

    // ---- Upload feeder outputs + per-block metadata -----------------------
    const std::size_t pm = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kM);
    DeviceBuffer<double> dQall(pm), dVall(pm), dSall(2u * pm);
    STEPPE_CUDA_CHECK(cudaMemcpy(dQall.data(), Qall.data(), pm * sizeof(double), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dVall.data(), Vall.data(), pm * sizeof(double), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dSall.data(), Sall.data(), 2u * pm * sizeof(double), cudaMemcpyHostToDevice));

    DeviceBuffer<int> dIds(static_cast<std::size_t>(n_in_group));
    DeviceBuffer<long> dOff(static_cast<std::size_t>(kNblock));
    DeviceBuffer<int> dSizes(static_cast<std::size_t>(kNblock));
    STEPPE_CUDA_CHECK(cudaMemcpy(dIds.data(), ids_in_group.data(),
                                 static_cast<std::size_t>(n_in_group) * sizeof(int), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dOff.data(), kOffsets, sizeof(kOffsets), cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dSizes.data(), kSizes, sizeof(kSizes), cudaMemcpyHostToDevice));

    // ---- Group slabs: dQg/dVg [P×s_pad×n], dSg [2P×s_pad×n] ----------------
    const std::size_t psp_n = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kSpad) * static_cast<std::size_t>(n_in_group);
    DeviceBuffer<double> dQg(psp_n), dVg(psp_n), dSg(2u * psp_n);
    const std::size_t pp = static_cast<std::size_t>(kP) * static_cast<std::size_t>(kP);
    const std::size_t pp_n = pp * static_cast<std::size_t>(n_in_group);
    DeviceBuffer<double> dGg(pp_n), dVpairg(pp_n), dRg(2u * pp_n);

    // ---- Resident [P×P×n_block] f2 + Vpair tensors ------------------------
    const std::size_t total = pp * static_cast<std::size_t>(kNblock);
    DeviceBuffer<double> dF2all(total), dVpairAll(total);
    STEPPE_CUDA_CHECK(cudaMemset(dF2all.data(), 0, total * sizeof(double)));
    STEPPE_CUDA_CHECK(cudaMemset(dVpairAll.data(), 0, total * sizeof(double)));

    // ---- The handle: stream + workspace bound once (the §12 invariant) -----
    // Native Fp64 here, so the emulated-FP64 workspace is not load-bearing, but we
    // bind one anyway to mirror the production handle setup (CublasHandle owns the
    // (stream, workspace) invariant; cleanup X-1/B1).
    CublasHandle blas;
    DeviceBuffer<unsigned char> dWs(64u * 1024u * 1024u);  // a real 64 MiB workspace
    blas.set_workspace(dWs.data(), dWs.bytes());
    blas.set_stream(/*default stream*/ nullptr);

    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};

    // ---- The three wrappers under test ------------------------------------
    steppe::device::launch_gather_group(dQall.data(), dVall.data(), dSall.data(),
                                        dIds.data(), dOff.data(), dSizes.data(),
                                        kP, kSpad, n_in_group,
                                        dQg.data(), dVg.data(), dSg.data(), /*stream=*/nullptr);
    steppe::device::run_f2_gemms_group(blas.get(), prec, kP, kSpad, n_in_group,
                                       dQg.data(), dVg.data(), dSg.data(),
                                       dGg.data(), dVpairg.data(), dRg.data());
    steppe::device::launch_assemble_blocks_group(dGg.data(), dVpairg.data(), dRg.data(),
                                                 dIds.data(), kP, n_in_group,
                                                 dF2all.data(), dVpairAll.data(), /*stream=*/nullptr);
    STEPPE_CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Copy the resident tensors back and diff vs the oracle ------------
    std::vector<double> f2(total), vpair(total);
    STEPPE_CUDA_CHECK(cudaMemcpy(f2.data(), dF2all.data(), total * sizeof(double), cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(vpair.data(), dVpairAll.data(), total * sizeof(double), cudaMemcpyDeviceToHost));

    const double tol = 1e-12;  // tiny exact-arithmetic synthetic; GEMM reorder is far below this
    bool ok = true;
    for (int b = 0; b < kNblock; ++b) {
        for (int j = 0; j < kP; ++j) {
            for (int i = 0; i < kP; ++i) {
                double ef2 = 0.0, evp = 0.0;
                oracle_block(kOffsets[b], kSizes[b], i, j, ef2, evp);
                const std::size_t idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(kP) * static_cast<std::size_t>(j) +
                                        pp * static_cast<std::size_t>(b);
                const double gf2 = f2[idx], gvp = vpair[idx];
                if (std::fabs(gf2 - ef2) > tol || std::fabs(gvp - evp) > tol) {
                    std::fprintf(stderr,
                                 "  [FAIL] block %d (i=%d,j=%d): f2 gpu=%.15g oracle=%.15g | "
                                 "vpair gpu=%.15g oracle=%.15g\n",
                                 b, i, j, gf2, ef2, gvp, evp);
                    ok = false;
                }
                // The pad-column-zero property made concrete: Vpair is the REAL
                // block size (2 or 3), NOT s_pad (4) — a non-inert pad would inflate it.
                if (std::fabs(evp - static_cast<double>(kSizes[b])) > tol) {
                    std::fprintf(stderr, "  [FAIL] oracle Vpair %g != real size %d (test bug)\n",
                                 evp, kSizes[b]);
                    ok = false;
                }
            }
        }
    }
    std::printf("  positive control (reordered ids={1,0}, sizes {2,3} padded to %d) -> %s\n",
                kSpad, ok ? "PASS" : "FAIL");
    return ok;
}

// ===========================================================================
// Death-case drivers — each calls ONE wrapper with a degenerate extent. When
// STEPPE_ASSERT is active each must abort (SIGABRT). They run in a FORKED child.
//
// CRITICAL: a forked child CANNOT reuse the parent's CUDA context, so it must
// NOT touch CUDA before the guard fires — otherwise the abort would come from a
// post-fork cudaErrorInitializationError, NOT the B24 guard (the test would pass
// for the wrong reason and could not detect a removed guard). The B24 STEPPE_ASSERT
// is the FIRST statement in each wrapper, BEFORE any pointer dereference, kernel
// launch, or cuBLAS call (run_f2_gemms_group asserts before f2_compute_type / the
// handle is used). So we pass NULLPTR device pointers and a NULL cuBLAS handle: the
// guard must fire on the scalar EXTENT alone, with no device memory and no live
// CUDA context. If the guard were missing, the wrapper would instead dereference
// null / issue an invalid launch — a different (or no) failure — so this isolates
// the guard precisely.
// ===========================================================================

void death_gather_nz0() {  // gather, n_in_group == 0
    steppe::device::launch_gather_group(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        kP, kSpad, /*n_in_group=*/0,
                                        nullptr, nullptr, nullptr, nullptr);
}
void death_gather_nzneg() {  // gather, n_in_group == -1
    steppe::device::launch_gather_group(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        kP, kSpad, /*n_in_group=*/-1,
                                        nullptr, nullptr, nullptr, nullptr);
}
void death_gather_spad0() {  // gather, s_pad == 0 (zero gridDim.y)
    steppe::device::launch_gather_group(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        kP, /*s_pad=*/0, /*n_in_group=*/1,
                                        nullptr, nullptr, nullptr, nullptr);
}
void death_gemms_nz0() {  // gemms, n_in_group (batchCount) == 0
    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    steppe::device::run_f2_gemms_group(/*handle=*/nullptr, prec, kP, kSpad, /*n_in_group=*/0,
                                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}
void death_gemms_spad0() {  // gemms, s_pad (k) == 0
    const Precision prec{Precision::Kind::Fp64, steppe::kDefaultMantissaBits};
    steppe::device::run_f2_gemms_group(/*handle=*/nullptr, prec, kP, /*s_pad=*/0, /*n_in_group=*/1,
                                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}
void death_assemble_nz0() {  // assemble, n_in_group == 0
    steppe::device::launch_assemble_blocks_group(nullptr, nullptr, nullptr, nullptr,
                                                 kP, /*n_in_group=*/0, nullptr, nullptr, nullptr);
}
void death_assemble_nzneg() {  // assemble, n_in_group == -1
    steppe::device::launch_assemble_blocks_group(nullptr, nullptr, nullptr, nullptr,
                                                 kP, /*n_in_group=*/-1, nullptr, nullptr, nullptr);
}

struct DeathCase {
    const char* name;
    void (*fn)();
};
const DeathCase kDeathCases[] = {
    {"launch_gather_group: n_in_group == 0 -> abort", death_gather_nz0},
    {"launch_gather_group: n_in_group == -1 -> abort", death_gather_nzneg},
    {"launch_gather_group: s_pad == 0 -> abort", death_gather_spad0},
    {"run_f2_gemms_group: n_in_group == 0 -> abort", death_gemms_nz0},
    {"run_f2_gemms_group: s_pad == 0 -> abort", death_gemms_spad0},
    {"launch_assemble_blocks_group: n_in_group == 0 -> abort", death_assemble_nz0},
    {"launch_assemble_blocks_group: n_in_group == -1 -> abort", death_assemble_nzneg},
};

}  // namespace

#include <sys/wait.h>
#include <unistd.h>

namespace {

// Run `fn` in a forked child; return true iff the child died via SIGABRT (what a
// fired STEPPE_ASSERT does). Mirrors tests/unit/test_f2_from_blocks.cpp's harness.
//
// `[[maybe_unused]]` (C++17, [dcl.attr.unused]): the death-case loop in main() that
// calls this is `#ifndef NDEBUG`, so under NDEBUG the helper is defined but never
// called by contract (the asserts compile out) — the attribute keeps the TU clean
// under nvcc `--Werror all-warnings` in BOTH builds without #ifdef'ing the helper
// away from its <sys/wait.h>/<unistd.h> home.
[[maybe_unused]] [[nodiscard]] bool child_aborts(void (*fn)()) {
    const pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork() failed\n");
        return false;
    }
    if (pid == 0) {
        fn();
        _exit(0);  // did NOT abort -> a test failure
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) { /* retry on EINTR */ }
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

}  // namespace

int main() {
    // Fail fast (not a silent skip) if there is no usable device: this is a GPU
    // gate. The box always has a device.
    int dev_count = 0;
    const cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count < 1) {
        std::fprintf(stderr, "RESULT: FAIL — no CUDA device available (%s); this is a GPU gate.\n",
                     cudaGetErrorString(derr));
        return EXIT_FAILURE;
    }

    std::printf("\nB24 M4 grouped-batch wrapper guards + gather/scatter (synthetic, no data)\n");

    bool ok = true;

    // (1) Positive control: the happy path must produce the correct scattered tensors.
    ok = positive_control() && ok;

#ifndef NDEBUG
    // (2) STEPPE_ASSERT active: each degenerate-extent call must abort (SIGABRT)
    //     before issuing the invalid launch / degenerate GEMM.
    for (const auto& c : kDeathCases) {
        const bool aborted = child_aborts(c.fn);
        std::printf("  [%s] %s\n", aborted ? "PASS" : "FAIL", c.name);
        if (!aborted) {
            std::fprintf(stderr,
                         "  [FAIL] %s: the wrapper did NOT fail-fast on the degenerate extent\n"
                         "         (the F6/B24 STEPPE_ASSERT is missing or mis-bounded).\n",
                         c.name);
            ok = false;
        }
    }
#else
    std::printf("  [SKIP] NDEBUG build: STEPPE_ASSERT compiled out, %zu degenerate-extent\n"
                "         death cases skipped by contract (the guard is debug-only).\n",
                sizeof(kDeathCases) / sizeof(kDeathCases[0]));
#endif

    std::printf("\n");
    if (!ok) {
        std::fprintf(stderr,
            "RESULT: FAIL — the M4 grouped-batch wrappers did not produce the correct\n"
            "        scattered tensors on the happy path, or did not fail-fast on a\n"
            "        degenerate (n_in_group <= 0 / s_pad <= 0) extent (architecture.md\n"
            "        §2 fail-fast; cleanup f2_blocks_kernel F6, B24).\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS (grouped gather/GEMM/scatter correct on a reordered "
                         "multi-block layout; the wrappers fail-fast on degenerate extents)\n");
    return EXIT_SUCCESS;
}
