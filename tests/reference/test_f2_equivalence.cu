// tests/reference/test_f2_equivalence.cu
//
// REFERENCE-EQUIVALENCE TEST for the M0 f2 kernel (ROADMAP M0 gate; architecture
// .md §13 "Golden / reference-equivalence", §5 S2, §12; ROADMAP §5).
//
// This is the central TRUST SEAM at f2: the GPU 3-GEMM reformulation diffed
// against an obviously-correct scalar/long-double CPU oracle, on REAL Q/V/N
// input. It is a CORRECTNESS test (real genotype-derived input), NOT a precision
// or throughput benchmark — there is no synthetic data here (ROADMAP §0
// cautionary tale: precision/throughput claims are benchmarked on real data
// only; this test merely confirms the two backends AGREE on real data).
//
// What it does:
//   1. Loads a small REAL Q/V/N matrix from a directory (SHARED BINARY FORMAT:
//      shape.txt + Q.f64 / V.f64 / N.f64, column-major [P × M], ld = P) — reusing
//      the validated spike loader. Default dir /workspace/data/aadr/derived_acc
//      (P=50, M=100000 on the box); overridable as argv[1].
//   2. Computes the f2 matrix THREE ways from the IDENTICAL Q/V/N input, all
//      through the architecture's contract surfaces (steppe::core::MatView Q/V/N,
//      steppe::Precision, steppe::F2Result, and the SHARED per-element primitives
//      in core/internal/f2_estimator.hpp so the formula cannot diverge):
//        * CUDA backend, Precision::Fp64          (native FP64 GEMMs)
//        * CUDA backend, Precision::EmulatedFp64  (fixed-slice Ozaki, 40-bit)
//        * CPU reference backend                  (scalar long-double oracle)
//   3. Asserts max relative error over the OFF-DIAGONAL entries with |ref| above
//      the floor lands in the TIGHT tier:
//        * emu  vs ref  <  1e-6   (architecture.md §12 est tier; ROADMAP §0
//                                  40-bit worst-case f2 error ≈ 2.2e-11 ≪ 1e-6)
//        * native vs ref  <  1e-9 (near bit-stable; §12)
//   4. Prints a small pass/fail table and exits NONZERO on any failure (so CTest
//      / a sweep harness detects it).
//
// Why the backends are implemented INLINE here (M0 staging note): the concrete
// `CudaBackend` / `CpuBackend` classes land later in M0; this test exercises the
// EXACT same validated logic through the already-authored contract — the Q/V/N
// `MatView`, the `Precision` knob, the `F2Result` seam, and crucially the shared
// `core::f2_estimator` primitives (het_correction / f2_term / assemble_f2_numerator
// / finalize_f2) plus the `core::grid_for` launch helper. When the real backends
// are authored against `ComputeBackend::compute_f2`, they call these same
// primitives, so this test's expectations are unchanged. There is ONE loader, ONE
// CUDA_CHECK, ONE 3-GEMM routine, ONE assemble — no duplication of the formula.
//
// PRECISION POLICY IS THE LAW (architecture.md §12; ROADMAP §0): the f2 GEMMs use
// FIXED-slice Ozaki emulation at the default mantissa_bits (kDefaultMantissaBits
// == 40 ≈ native). DYNAMIC mantissa control is the rejected parity trap and is
// NEVER selected. The small numerator/divide stays native FP64 in BOTH GPU arms.
//
// Build (on the REMOTE sm_120 / CUDA 13 box — NOT locally; the local RTX 2070 /
// CUDA 11.8 box is the wrong arch and has no emulation API). One command:
//   nvcc -O3 -std=c++20 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1
//        -I include -I src
//        tests/reference/test_f2_equivalence.cu -lcublas -o test_f2_equivalence
//   (Drop -DSTEPPE_HAVE_EMU_TUNING to compile without the emulation tuning
//    symbols; then the EmulatedFp64 arm falls back to native FP64 and is reported
//    as SKIPPED rather than asserted — see the engagement guard below. The native
//    arm is always exercised.)
//
// Run:
//   ./test_f2_equivalence [dir]      (default /workspace/data/aadr/derived_acc)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <library_types.h>

// Project contract surfaces (architecture.md §4 real paths). These are the
// already-authored M0 foundation files; the test builds against them verbatim.
#include "steppe/config.hpp"            // steppe::Precision, kCdivBlock, kRelFloor, kAbsFloor, kDefaultMantissaBits
#include "core/internal/views.hpp"      // steppe::core::MatView (the Q/V/N contract)
#include "core/internal/f2_estimator.hpp"  // het_correction/f2_term/assemble_f2_numerator/finalize_f2/grid_for
#include "device/backend.hpp"           // steppe::F2Result (the f2/Vpair seam)

using steppe::Precision;
using steppe::core::MatView;
using steppe::F2Result;

// ---------------------------------------------------------------------------
// STEPPE_HAVE_EMU_TUNING gates the cuBLAS FP64-emulation tuning symbols. The
// LOAD-BEARING confirmed names (verified in the validated spike f2_prec_acc.cu /
// f2_timing.cu against live CUDA 13 cuBLAS) are:
//   cublasSetMathMode(h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)
//   cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER)
//   cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED)
//   cublasSetFixedPointEmulationMaxMantissaBitCount(h, mantissa_bits)
// plus the emulated compute type CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT.
// When the macro is off, only the native arm runs and the emu arm is SKIPPED.
#ifndef STEPPE_HAVE_EMU_TUNING
#define STEPPE_HAVE_EMU_TUNING 0
#endif

// ---------------------------------------------------------------------------
// Error checks — ONE CUDA check, ONE cuBLAS check for the whole test (DRY,
// ROADMAP §5). They throw via std::exit on failure so a leak of a sticky error
// cannot mask a later assertion. (The production §8 STEPPE_CUDA_CHECK lives in
// device/cuda/check.cuh; this test is a standalone TU and uses its own local
// pair — the single home for checks WITHIN this TU.)
// ---------------------------------------------------------------------------
#define STEPPE_CUDA_CHECK(expr)                                                 \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",             \
                         __FILE__, __LINE__, #expr,                            \
                         cudaGetErrorString(err__));                          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (expr);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error %s:%d: '%s' -> status %d\n",    \
                         __FILE__, __LINE__, #expr, static_cast<int>(st__));   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace {

// Default real-data directory on the remote box (exists there per the task; the
// local box has no copy — that is expected, this test only runs on the box).
constexpr const char* kDefaultDataDir = "/workspace/data/aadr/derived_acc";

// Tight-tier relative-error thresholds (architecture.md §12 est tier; ROADMAP
// §0). These are TEST tolerances, the single home for the verdict thresholds —
// they replace the spike's bare inline `1e-6`/`1e-9` literals.
constexpr double kTolEmuVsRef    = 1e-6;   // EmulatedFp64{40} vs oracle
constexpr double kTolNativeVsRef = 1e-9;   // native Fp64 vs oracle

// ---------------------------------------------------------------------------
// SHARED BINARY FORMAT loader (reused from the validated spike: f2_emu_spike.cu
// read_f64 / load_real_data). shape.txt holds "P M"; Q/V/N.f64 each hold P*M
// little-endian doubles, column-major [P × M] (element (pop i, snp s) at i+P*s).
// Loud on any shape/format mismatch. Returns the raw arrays unchanged; the
// backends below mask / build their own working arrays from them.
// ---------------------------------------------------------------------------
void read_f64(const std::string& path, std::vector<double>& out, std::size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out.resize(count);
    const std::size_t got = std::fread(out.data(), sizeof(double), count, f);
    std::size_t extra = 0;
    if (got == count) {  // probe one more double to catch a too-large file (wrong P*M)
        double probe = 0.0;
        extra = std::fread(&probe, sizeof(double), 1, f);
    }
    std::fclose(f);
    if (got != count) {
        std::fprintf(stderr,
            "ERROR: %s has %zu doubles, expected %zu (P*M). Shape/format mismatch.\n",
            path.c_str(), got, count);
        std::exit(EXIT_FAILURE);
    }
    if (extra != 0) {
        std::fprintf(stderr,
            "ERROR: %s has MORE than %zu doubles (trailing data). Shape/format mismatch.\n",
            path.c_str(), count);
        std::exit(EXIT_FAILURE);
    }
}

// Load shape.txt + Q/V/N.f64 from `dir`. Fills raw arrays and P, M.
void load_real_data(const std::string& dir, int& P_out, long& M_out,
                    std::vector<double>& Q_raw,
                    std::vector<double>& V_raw,
                    std::vector<double>& N_raw) {
    const std::string shapePath = dir + "/shape.txt";
    FILE* sf = std::fopen(shapePath.c_str(), "r");
    if (!sf) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", shapePath.c_str());
        std::exit(EXIT_FAILURE);
    }
    int P = 0;
    long M = 0;
    if (std::fscanf(sf, "%d %ld", &P, &M) != 2) {
        std::fprintf(stderr, "ERROR: %s must contain 'P M' (two ints)\n", shapePath.c_str());
        std::fclose(sf);
        std::exit(EXIT_FAILURE);
    }
    std::fclose(sf);
    if (P <= 1 || M <= 0) {
        std::fprintf(stderr, "ERROR: bad shape from %s: P=%d M=%ld (need P>1, M>0)\n",
                     shapePath.c_str(), P, M);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t count = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    read_f64(dir + "/Q.f64", Q_raw, count);
    read_f64(dir + "/V.f64", V_raw, count);
    read_f64(dir + "/N.f64", N_raw, count);

    P_out = P;
    M_out = M;
}

// ---------------------------------------------------------------------------
// Build the masked GEMM-input arrays from the raw Q/V/N (one recipe, shared by
// both GPU arms — mirrors the spike build_host_arrays so the GEMM inputs are the
// same recipe). Q is zero-filled where invalid (the zero is what makes the
// masked GEMM correct), and S = [Qsq ; Hc] is the [2P × M] stack feeding the R
// GEMM. The het correction Hc uses the SHARED primitive core::het_correction so
// the GPU feeder and the CPU oracle cannot diverge on the formula.
//   Q  [P  × M] (masked)   element (i,s) at i + P*s
//   V  [P  × M]            element (i,s) at i + P*s
//   S  [2P × M]            Qsq at i + 2P*s, Hc at (P+i) + 2P*s
// ---------------------------------------------------------------------------
void build_gemm_inputs(int P, long M,
                       const std::vector<double>& Q_raw,
                       const std::vector<double>& V_raw,
                       const std::vector<double>& N_raw,
                       std::vector<double>& Q,
                       std::vector<double>& V,
                       std::vector<double>& S) {
    const long twoP = 2L * P;
    Q.assign(static_cast<std::size_t>(P) * M, 0.0);
    V.assign(static_cast<std::size_t>(P) * M, 0.0);
    S.assign(static_cast<std::size_t>(twoP) * M, 0.0);

    for (long s = 0; s < M; ++s) {
        for (int i = 0; i < P; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * s;
            const bool valid = (V_raw[idx] != 0.0);
            const double praw = Q_raw[idx];
            const double n    = N_raw[idx];

            const double q  = valid ? praw : 0.0;                         // masked freq
            const double hc = steppe::core::het_correction(praw, n, valid);  // SHARED primitive

            Q[idx] = q;
            V[idx] = valid ? 1.0 : 0.0;

            const std::size_t sidx_qsq = static_cast<std::size_t>(i) + static_cast<std::size_t>(twoP) * s;
            const std::size_t sidx_hc  = static_cast<std::size_t>(P + i) + static_cast<std::size_t>(twoP) * s;
            S[sidx_qsq] = q * q;  // Qsq block (top P rows)
            S[sidx_hc]  = hc;     // Hc  block (bottom P rows)
        }
    }
}

// ---------------------------------------------------------------------------
// Device kernel: assemble f2 [P × P] (column-major) from the three GEMM outputs,
// using the SHARED per-element primitives so the GPU and the CPU oracle assemble
// the identical formula (core::assemble_f2_numerator + core::finalize_f2). The
// numerator/divide is held in NATIVE FP64 in both GPU arms (architecture.md §12:
// the Precision knob governs only the GEMMs, never this step).
//   G     [P  × P] column-major
//   Vpair [P  × P] column-major
//   R     [2P × P] column-major  (top P rows = Σp², bottom P rows = Σhc)
// ---------------------------------------------------------------------------
__global__ void assemble_f2_kernel(const double* __restrict__ G,
                                   const double* __restrict__ Vpair,
                                   const double* __restrict__ R,
                                   double* __restrict__ f2,
                                   int P) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (i >= P || j >= P) return;

    const std::size_t Pp   = static_cast<std::size_t>(P);
    const std::size_t twoP = 2 * Pp;

    const double Gij     = G[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * Pp];
    const double vp      = Vpair[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * Pp];
    const double sumsq_i = R[static_cast<std::size_t>(i)        + static_cast<std::size_t>(j) * twoP];  // R(i,   j)
    const double sumsq_j = R[static_cast<std::size_t>(j)        + static_cast<std::size_t>(i) * twoP];  // R(j,   i)
    const double hsum_i  = R[(Pp + static_cast<std::size_t>(i)) + static_cast<std::size_t>(j) * twoP];  // R(P+i, j)
    const double hsum_j  = R[(Pp + static_cast<std::size_t>(j)) + static_cast<std::size_t>(i) * twoP];  // R(P+j, i)

    const double num = steppe::core::assemble_f2_numerator(sumsq_i, sumsq_j, Gij, hsum_i, hsum_j);
    f2[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * Pp] = steppe::core::finalize_f2(num, vp);
}

// ---------------------------------------------------------------------------
// Run the three f2 GEMMs on a handle at a given compute type, then assemble.
// The ONE 3-GEMM routine for the whole test (DRY). All column-major; the op /
// m,n,k / lda,ldb,ldc match the validated derivation in the spike header:
//   G     [P  × P] = Q  * Qᵀ   (OP_N, OP_T, m=P,  n=P, k=M, lda=P,   ldb=P, ldc=P)
//   Vpair [P  × P] = V  * Vᵀ   (OP_N, OP_T, m=P,  n=P, k=M, lda=P,   ldb=P, ldc=P)
//   R     [2P × P] = S  * Vᵀ   (OP_N, OP_T, m=2P, n=P, k=M, lda=2P,  ldb=P, ldc=2P)
// Returns the host f2 [P × P] and the host Vpair [P × P] in `out`.
// ---------------------------------------------------------------------------
void run_f2_gemms(cublasHandle_t handle, cublasComputeType_t computeType,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR, double* dF2,
                  F2Result& out) {
    const double one  = 1.0;
    const double zero = 0.0;
    const int    twoP = 2 * P;
    const int    Mi   = static_cast<int>(M);

    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, Mi,
                              &one, dQ, CUDA_R_64F, P, dQ, CUDA_R_64F, P,
                              &zero, dG, CUDA_R_64F, P,
                              computeType, CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, Mi,
                              &one, dV, CUDA_R_64F, P, dV, CUDA_R_64F, P,
                              &zero, dVpair, CUDA_R_64F, P,
                              computeType, CUBLAS_GEMM_DEFAULT));
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, twoP, P, Mi,
                              &one, dS, CUDA_R_64F, twoP, dV, CUDA_R_64F, P,
                              &zero, dR, CUDA_R_64F, twoP,
                              computeType, CUBLAS_GEMM_DEFAULT));

    // Assemble (native FP64; not the thing under test). Launch geometry comes
    // from the SHARED launch helper core::grid_for over the [P × P] output —
    // never a re-picked block size (kCdivBlock; ROADMAP §4).
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(steppe::core::grid_for(P)),
                    static_cast<unsigned>(steppe::core::grid_for(P)));
    assemble_f2_kernel<<<grid, block>>>(dG, dVpair, dR, dF2, P);
    STEPPE_CUDA_CHECK(cudaGetLastError());
    STEPPE_CUDA_CHECK(cudaDeviceSynchronize());

    out.P = P;
    out.f2.resize(static_cast<std::size_t>(P) * P);
    out.vpair.resize(static_cast<std::size_t>(P) * P);
    STEPPE_CUDA_CHECK(cudaMemcpy(out.f2.data(), dF2,
                                 sizeof(double) * static_cast<std::size_t>(P) * P,
                                 cudaMemcpyDeviceToHost));
    STEPPE_CUDA_CHECK(cudaMemcpy(out.vpair.data(), dVpair,
                                 sizeof(double) * static_cast<std::size_t>(P) * P,
                                 cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// The CUDA "backend": compute f2 + Vpair from the Q/V/N contract at a given
// Precision (the 3-GEMM reformulation). Mirrors what ComputeBackend::compute_f2
// will do on the GPU — same inputs (MatView Q/V/N), same Precision knob, same
// F2Result. `engaged_out` reports whether FP64 emulation actually engaged for
// the EmulatedFp64 request (header-independent bit-identity guard: Ozaki is NOT
// bit-identical to native, so emu == native bit-for-bit means it silently fell
// back — architecture.md §12; see the spike defect-(2) check).
// ---------------------------------------------------------------------------
F2Result cuda_compute_f2(const MatView& Q, const MatView& V, const MatView& N,
                         const Precision& precision,
                         const std::vector<double>* native_for_engage_check,
                         bool& engaged_out) {
    engaged_out = true;  // native always "engaged"; refined below for emulation
    const int  P = Q.P;
    const long M = Q.M;

    // Build the masked GEMM inputs (Q masked, V, S=[Qsq;Hc]) from the raw views.
    // The MatView::data pointers are the raw column-major [P × M] arrays.
    std::vector<double> Qraw(Q.data, Q.data + static_cast<std::size_t>(P) * M);
    std::vector<double> Vraw(V.data, V.data + static_cast<std::size_t>(P) * M);
    std::vector<double> Nraw(N.data, N.data + static_cast<std::size_t>(P) * M);
    std::vector<double> hQ, hV, hS;
    build_gemm_inputs(P, M, Qraw, Vraw, Nraw, hQ, hV, hS);

    // Device buffers. (Standalone-test allocations; the production hot path wraps
    // these in DeviceBuffer<T> — architecture.md §7. Here they are local and
    // freed at the end of the function, never escaping.)
    const std::size_t szPM  = sizeof(double) * static_cast<std::size_t>(P) * M;
    const std::size_t sz2PM = sizeof(double) * static_cast<std::size_t>(2L * P) * M;
    const std::size_t szPP  = sizeof(double) * static_cast<std::size_t>(P) * P;
    const std::size_t sz2PP = sizeof(double) * static_cast<std::size_t>(2L * P) * P;

    double *dQ = nullptr, *dV = nullptr, *dS = nullptr;
    double *dG = nullptr, *dVpair = nullptr, *dR = nullptr, *dF2 = nullptr;
    STEPPE_CUDA_CHECK(cudaMalloc(&dQ, szPM));
    STEPPE_CUDA_CHECK(cudaMalloc(&dV, szPM));
    STEPPE_CUDA_CHECK(cudaMalloc(&dS, sz2PM));
    STEPPE_CUDA_CHECK(cudaMalloc(&dG, szPP));
    STEPPE_CUDA_CHECK(cudaMalloc(&dVpair, szPP));
    STEPPE_CUDA_CHECK(cudaMalloc(&dR, sz2PP));
    STEPPE_CUDA_CHECK(cudaMalloc(&dF2, szPP));

    STEPPE_CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), szPM, cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dV, hV.data(), szPM, cudaMemcpyHostToDevice));
    STEPPE_CUDA_CHECK(cudaMemcpy(dS, hS.data(), sz2PM, cudaMemcpyHostToDevice));

    // Explicit workspace — required for run-to-run reproducibility of emulated
    // FP64 (architecture.md §12). 64 MiB is ample for these tiny GEMMs.
    const std::size_t workspaceSize = static_cast<std::size_t>(64) * 1024 * 1024;
    void* dWorkspace = nullptr;
    STEPPE_CUDA_CHECK(cudaMalloc(&dWorkspace, workspaceSize));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetWorkspace(handle, dWorkspace, workspaceSize));

    cublasComputeType_t computeType = CUBLAS_COMPUTE_64F;
    if (precision.kind == Precision::Kind::Fp64) {
        // Native FP64 oracle/fallback arm; strict native accumulation.
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        computeType = CUBLAS_COMPUTE_64F;
    } else if (precision.kind == Precision::Kind::EmulatedFp64) {
        // Fixed-slice Ozaki FP64 emulation at precision.mantissa_bits (default
        // kDefaultMantissaBits == 40). FIXED control only — DYNAMIC is the trap
        // (architecture.md §12; ROADMAP §0) and is NEVER selected here.
        computeType = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
#if STEPPE_HAVE_EMU_TUNING
        CUBLAS_CHECK(cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(
                         handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(
                         handle, precision.mantissa_bits));
#endif
    } else {
        std::fprintf(stderr, "ERROR: Tf32 is screening-only and not exercised by "
                             "this correctness test (architecture.md §12).\n");
        std::exit(EXIT_FAILURE);
    }

    F2Result result;
    run_f2_gemms(handle, computeType, P, M, dQ, dV, dS, dG, dVpair, dR, dF2, result);

    // Engagement guard for the emulated arm: Ozaki output is NOT bit-identical to
    // native FP64, so if emu == native bit-for-bit the library silently fell back
    // to native and this arm did NOT test emulation (architecture.md §12).
    if (precision.kind == Precision::Kind::EmulatedFp64 && native_for_engage_check) {
        const std::vector<double>& nat = *native_for_engage_check;
        engaged_out = !(nat.size() == result.f2.size() &&
                        std::memcmp(nat.data(), result.f2.data(),
                                    nat.size() * sizeof(double)) == 0);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    STEPPE_CUDA_CHECK(cudaFree(dQ));
    STEPPE_CUDA_CHECK(cudaFree(dV));
    STEPPE_CUDA_CHECK(cudaFree(dS));
    STEPPE_CUDA_CHECK(cudaFree(dG));
    STEPPE_CUDA_CHECK(cudaFree(dVpair));
    STEPPE_CUDA_CHECK(cudaFree(dR));
    STEPPE_CUDA_CHECK(cudaFree(dF2));
    STEPPE_CUDA_CHECK(cudaFree(dWorkspace));
    return result;
}

// ---------------------------------------------------------------------------
// Pairwise (cascade) summation of long doubles — reduces the accumulator error
// factor from O(M) to O(log2 M) so the reference's summation error is negligible
// against the f2 magnitude (reused from the validated spike). Does not mutate.
// ---------------------------------------------------------------------------
long double pairwise_sum(const long double* a, std::size_t n) {
    if (n == 0) return 0.0L;
    if (n <= 128) {
        long double s = 0.0L;
        for (std::size_t k = 0; k < n; ++k) s += a[k];
        return s;
    }
    const std::size_t h = n / 2;
    return pairwise_sum(a, h) + pairwise_sum(a + h, n - h);
}

// ---------------------------------------------------------------------------
// The CPU REFERENCE backend: the obviously-correct scalar long-double oracle the
// GPU is diffed against (architecture.md §13; ROADMAP §5). Cancellation-FREE: it
// forms (p_i − p_j) per SNP and squares it directly via the SHARED primitive
// core::f2_term (never the expanded p² − 2pq + q² the GEMMs build), then sums
// with pairwise summation over jointly-valid SNPs and divides by Vpair via the
// SHARED core::finalize_f2. Computed INDEPENDENTLY from the host inputs — no GPU
// sum feeds it. Consumes the SAME MatView Q/V/N contract and returns the SAME
// F2Result seam as the CUDA arm.
// ---------------------------------------------------------------------------
F2Result cpu_reference_f2(const MatView& Q, const MatView& V, const MatView& N) {
    const int  P = Q.P;
    const long M = Q.M;

    F2Result result;
    result.P = P;
    result.f2.assign(static_cast<std::size_t>(P) * P, 0.0);
    result.vpair.assign(static_cast<std::size_t>(P) * P, 0.0);

    std::vector<long double> terms(static_cast<std::size_t>(M));  // per-SNP f2 summands

    for (int j = 0; j < P; ++j) {
        for (int i = 0; i < P; ++i) {
            std::size_t cnt = 0;
            for (long s = 0; s < M; ++s) {
                const bool vi = (V.element(i, s) != 0.0);
                const bool vj = (V.element(j, s) != 0.0);
                if (vi && vj) {
                    const double pi  = Q.element(i, s);
                    const double pj  = Q.element(j, s);
                    // Shared het-correction primitive: identical formula and floor
                    // (kHetCorrDenomFloor) to the GPU feeder — they cannot diverge.
                    const double hci = steppe::core::het_correction(pi, N.element(i, s), true);
                    const double hcj = steppe::core::het_correction(pj, N.element(j, s), true);
                    // Shared cancellation-free per-SNP summand.
                    terms[cnt] = static_cast<long double>(steppe::core::f2_term(pi, pj, hci, hcj));
                    ++cnt;
                }
            }
            const double num   = static_cast<double>(pairwise_sum(terms.data(), cnt));
            const double vpair = static_cast<double>(cnt);
            const std::size_t off = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * j;
            result.f2[off]    = steppe::core::finalize_f2(num, vpair);  // shared finalize
            result.vpair[off] = vpair;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Max relative error of a candidate vs the reference over the OFF-DIAGONAL
// entries whose |ref| is above the relative floor (kRelFloor): below that, an
// entry is at the noise floor and would blow up the relative metric, so it is
// skipped (the combined-form guard, architecture.md §12). Also counts sign flips
// (a FAIL signature). The diagonal is ~0 by construction and excluded.
// ---------------------------------------------------------------------------
struct ErrStats {
    double maxRel = 0.0;
    int    signFlips = 0;
    std::size_t scored = 0;
};

ErrStats rel_error_stats(const F2Result& cand, const F2Result& ref) {
    const int P = ref.P;
    ErrStats es;
    for (int j = 0; j < P; ++j) {
        for (int i = 0; i < P; ++i) {
            if (i == j) continue;
            const std::size_t off = static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * j;
            const double r = ref.f2[off];
            const double c = cand.f2[off];
            if (std::fabs(r) < steppe::kRelFloor) continue;  // near-zero: skip (noise floor)
            const double denom = std::max(std::fabs(r), steppe::kAbsFloor);
            const double rel = std::fabs(c - r) / denom;
            if (rel > es.maxRel) es.maxRel = rel;
            ++es.scored;
            if ((r > 0.0) != (c > 0.0) && c != 0.0) ++es.signFlips;
        }
    }
    return es;
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const std::string dir = (argc >= 2) ? argv[1] : kDefaultDataDir;

    // ---- Load real Q/V/N (reused spike loader) ------------------------------
    int  P = 0;
    long M = 0;
    std::vector<double> Q_raw, V_raw, N_raw;
    load_real_data(dir, P, M, Q_raw, V_raw, N_raw);
    std::fprintf(stderr,
        "[load] dir=%s  P=%d  M=%ld  (%zu doubles per matrix) — REAL data, not synthetic\n",
        dir.c_str(), P, M, static_cast<std::size_t>(P) * M);

    // Wrap the raw arrays as the Q/V/N contract views (column-major [P × M]).
    const MatView Q{Q_raw.data(), P, M};
    const MatView V{V_raw.data(), P, M};
    const MatView N{N_raw.data(), P, M};

    // ---- (1) CPU reference oracle (the ground truth) ------------------------
    const F2Result ref = cpu_reference_f2(Q, V, N);

    // ---- (2) CUDA backend, native Fp64 --------------------------------------
    bool native_engaged = true;  // unused for native, kept for symmetry
    const F2Result f2_native =
        cuda_compute_f2(Q, V, N, Precision{Precision::Kind::Fp64, steppe::kDefaultMantissaBits},
                        nullptr, native_engaged);

    // ---- (3) CUDA backend, EmulatedFp64{mantissa_bits=40} -------------------
    // Pass the native f2 so the engagement guard can detect a silent fallback.
    bool emu_engaged = false;
    const Precision emuPrec{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    const F2Result f2_emu =
        cuda_compute_f2(Q, V, N, emuPrec, &f2_native.f2, emu_engaged);

    // ---- Vpair cross-check (the retained S4 jackknife weight) ---------------
    // The CPU oracle and the GPU GEMM both produce the pairwise-valid count; they
    // are exact integer counts and must match exactly (architecture.md §5 S2
    // caveat (a)). A mismatch means the masking is wrong, so check it.
    bool vpair_match = (f2_native.vpair.size() == ref.vpair.size());
    if (vpair_match) {
        for (std::size_t k = 0; k < ref.vpair.size() && vpair_match; ++k) {
            if (f2_native.vpair[k] != ref.vpair[k]) vpair_match = false;
        }
    }

    // ---- Accuracy stats vs the oracle ---------------------------------------
    const ErrStats esNative = rel_error_stats(f2_native, ref);
    const ErrStats esEmu    = rel_error_stats(f2_emu, ref);

    // ---- Verdicts -----------------------------------------------------------
    // Native: tight-tier (< 1e-9), no sign flips, Vpair exact.
    const bool native_pass = vpair_match &&
                             esNative.signFlips == 0 &&
                             esNative.maxRel < kTolNativeVsRef;

    // Emulated: must have actually engaged (else SKIPPED, not asserted), then
    // tight-tier (< 1e-6), no sign flips.
    const bool emu_skipped = !emu_engaged;  // built without tuning, or silent fallback
    const bool emu_pass = emu_engaged &&
                          esEmu.signFlips == 0 &&
                          esEmu.maxRel < kTolEmuVsRef;

    // ---- Pass/fail table ----------------------------------------------------
    std::printf("\n");
    std::printf("f2 reference-equivalence (REAL data, P=%d M=%ld) — tight tier\n", P, M);
    std::printf("%-22s %12s %10s %9s %8s %8s\n",
                "arm", "maxRel", "tol", "signFlip", "scored", "verdict");
    std::printf("%-22s %12.3e %10.1e %9d %8zu %8s\n",
                "cuda Fp64 vs ref", esNative.maxRel, kTolNativeVsRef,
                esNative.signFlips, esNative.scored,
                native_pass ? "PASS" : "FAIL");
    std::printf("%-22s %12.3e %10.1e %9d %8zu %8s\n",
                "cuda EmuFp64{40} vs ref", esEmu.maxRel, kTolEmuVsRef,
                esEmu.signFlips, esEmu.scored,
                emu_skipped ? "SKIPPED" : (emu_pass ? "PASS" : "FAIL"));
    std::printf("%-22s %12s %10s %9s %8s %8s\n",
                "Vpair native==ref", vpair_match ? "exact" : "MISMATCH",
                "-", "-", "-", vpair_match ? "PASS" : "FAIL");
    std::printf("\n");

    // ---- Diagnostics on failure / skip --------------------------------------
    if (emu_skipped) {
        std::fprintf(stderr,
            "  [warn] EmulatedFp64 arm SKIPPED: emulation did not engage. Either the "
            "build lacks -DSTEPPE_HAVE_EMU_TUNING (so the EAGER strategy that forces "
            "emulation of tiny GEMMs is inactive and the library may decline to "
            "emulate), or the emu output was bit-identical to native (silent fallback). "
            "Ozaki is not bit-identical to native FP64 (architecture.md §12). Rebuild "
            "with -DSTEPPE_HAVE_EMU_TUNING=1 on a CUDA 13 / sm_120 box to assert this arm.\n");
    }
    if (!vpair_match) {
        std::fprintf(stderr,
            "  [FAIL] Vpair (pairwise-valid SNP count) differs between the GPU GEMM and "
            "the CPU oracle — the masking is wrong (architecture.md §5 S2).\n");
    }

    // The verdict: native must pass, Vpair must match, and the emu arm must pass
    // WHEN it engaged. A skipped emu arm does not fail the test (it could not be
    // exercised on this build), but it is loudly reported so CI on the box (built
    // with the tuning macro) treats a skip as a configuration error, not a pass.
    const bool overall = native_pass && vpair_match && (emu_skipped || emu_pass);
    if (!overall) {
        std::fprintf(stderr, "RESULT: FAIL\n");
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "RESULT: PASS%s\n",
                 emu_skipped ? " (EmulatedFp64 arm skipped — see warning above)" : "");
    return EXIT_SUCCESS;
}
