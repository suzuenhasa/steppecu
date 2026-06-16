// src/device/cuda/f2_block_kernel.cu
//
// S2 — the f2 3-GEMM reformulation on the GPU (architecture.md §5 S2, §7, §12;
// ROADMAP §0, M0/M4). The VALIDATED kernel lifted from the spike
// (experiments/f2_emu_spike/f2_emu_spike.cu run_f2_gemms + assemble_f2_kernel;
// f2_timing.cu fixed-bit Ozaki engagement) into the production layout — logic
// preserved, structure rebuilt per the standards (ROADMAP §1).
//
// What this TU owns (architecture.md §5 S2 "Two custom kernels only"):
//   1. launch_f2_feeder      — the FUSED elementwise pre-pass: one sweep over the
//                              decoded Q/V/N tile producing Q(masked), V, and
//                              S = [Qsq ; Hc], never materializing [SNP×pop×pop].
//   2. run_f2_gemms          — the three library GEMMs (G=Q·Qᵀ, Vpair=V·Vᵀ,
//                              R=[Qsq;Hc]·Vᵀ), with the precision policy engaged.
//   3. launch_assemble_f2    — the FUSED numerator+divide, NATIVE FP64.
//
// PRECISION POLICY (MEASURED on real AADR — architecture.md §12, ROADMAP §0; this
// is the law):
//   * The f2 GEMMs default to FIXED-slice Ozaki emulation at
//     `precision.mantissa_bits` (40 ≈ native FP64, 32 = 8.6e-9/faster). MEASURED
//     7–17× over native FP64 on real data; the lead grows with population count.
//   * DYNAMIC mantissa control is the REJECTED trap — it overshoots to ~60 bits on
//     real data's wide dynamic range and collapses to parity (no win). We engage
//     FIXED control explicitly; dynamic is never selected.
//   * Native FP64 (CUBLAS_COMPUTE_64F, PEDANTIC) is the oracle / fallback.
//   * The numerator/divide (assemble) stays NATIVE FP64 in every mode: it is the
//     catastrophic-cancellation step (Σp_i² + Σp_j² − 2Σp_i p_j) that emulation
//     cannot recover bits for; the Precision knob governs only the GEMMs.
//   * Tf32 is screening-only (architecture.md §12).
//
// NOTE on the stale spike comments: the spike concluded "f2 GEMMs must stay
// native FP64" — that PRE-DATES the real-AADR measurement that made fixed-slice
// Ozaki the default (ROADMAP §0, the cautionary tale). Those comments are NOT
// carried here; this file implements the measured policy.
//
// The emulation TUNING calls (strategy / fixed mantissa control / bit count) are
// compiled under -DSTEPPE_HAVE_EMU_TUNING (architecture.md build flag; ROADMAP
// precision policy). Without it, the LOAD-BEARING mechanisms — the emulated
// compute type and the FP64-emulated math mode — still engage emulation; the
// tuning only pins the FIXED slice count. Build on the remote CUDA-13 box.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It includes
// the SHARED host/device f2 primitive so the CPU oracle and this path cannot
// diverge on the formula (architecture.md §13; ROADMAP §5).
#include "device/cuda/f2_block_kernel.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include "core/internal/f2_estimator.hpp"  // het_correction, assemble_f2_numerator,
                                           //   finalize_f2, grid_for, kCdivBlock
#include "device/cuda/check.cuh"           // STEPPE_CUDA_CHECK, CUBLAS_CHECK,
                                           //   STEPPE_CUDA_CHECK_KERNEL

// -----------------------------------------------------------------------------
// STEPPE_HAVE_EMU_TUNING — gates the cuBLAS emulation tuning symbols (strategy,
// FIXED mantissa control, max mantissa-bit count). The two LOAD-BEARING
// mechanisms (the emulated compute type + the FP64-emulated math mode) are
// unconditional; only the FIXED-slice pinning needs the tuning API. Defaults off
// so the TU compiles on a stock toolkit; the remote build passes
// -DSTEPPE_HAVE_EMU_TUNING=1 (architecture.md §6 build flag; ROADMAP precision).
// -----------------------------------------------------------------------------
#ifndef STEPPE_HAVE_EMU_TUNING
#define STEPPE_HAVE_EMU_TUNING 0
#endif

namespace steppe::device {

using core::assemble_f2_numerator;
using core::finalize_f2;
using core::het_correction;

namespace {

// =============================================================================
// Kernel 1: fused elementwise pre-pass (architecture.md §5 S2, §11.3).
//
// One sweep over the decoded tile produces the three GEMM-input matrices. Each
// thread owns one (population i, SNP s) entry. Column-major [P × M] inputs /
// outputs (element (i,s) at i + P·s); the stacked S is [2P × M] (lda = 2P) with
// Qsq in rows 0..P-1 and Hc in rows P..2P-1 — exactly the spike build_host_arrays
// recipe, moved onto the GPU and expressed through the shared `het_correction`
// primitive so the per-element formula is identical to the CPU oracle.
// All native FP64 (the feeder is bandwidth-bound; reduced precision buys nothing,
// architecture.md §12).
// =============================================================================
__global__ void f2_feeder_kernel(const double* __restrict__ Q_raw,
                                 const double* __restrict__ V_raw,
                                 const double* __restrict__ N_raw,
                                 double* __restrict__ Q_masked,
                                 double* __restrict__ V_out,
                                 double* __restrict__ S,
                                 int P, long M) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;  // population
    const long s = static_cast<long>(blockIdx.y) * blockDim.y + threadIdx.y;  // SNP
    if (i >= P || s >= M) return;

    const long Pl = static_cast<long>(P);
    const long idx = i + Pl * s;            // (i,s) in a [P  × M] matrix
    const long sidx = i + (2 * Pl) * s;     // (i,s) in the [2P × M] stacked S

    const bool valid = (V_raw[idx] != 0.0);
    // Zero-fill where invalid: Q²=0 and the cross term G(i,j) vanish there, which
    // is what makes the masked GEMM reproduce the pairwise-complete reference
    // (architecture.md §5 S2; views.hpp Q/V/N contract).
    const double q = valid ? Q_raw[idx] : 0.0;
    // Shared per-element het correction (carries its own validity ⇒ 0 when
    // invalid). N is the non-missing HAPLOID count (Q/V/N contract); sample size
    // is per-SNP `N`, NEVER hardcoded (ROADMAP §4 — fixes the spike's /198.0).
    const double hc = het_correction(Q_raw[idx], N_raw[idx], valid);

    Q_masked[idx] = q;
    V_out[idx] = valid ? 1.0 : 0.0;
    S[sidx] = q * q;             // Qsq block (rows 0..P-1)
    S[Pl + sidx] = hc;           // Hc  block (rows P..2P-1)
}

// =============================================================================
// Kernel 2: fused numerator + divide (architecture.md §5 S2 line 240, §11.3).
//
// Each thread owns one output entry (i,j). Reads the three GEMM outputs and
// assembles f2(i,j) through the SHARED `assemble_f2_numerator` / `finalize_f2`
// primitives — the SAME functions the CPU oracle calls, so the cancellation
// formula cannot diverge (architecture.md §13). NATIVE FP64 in every precision
// mode: this is the catastrophic-cancellation step (architecture.md §12).
//
//   R is [2P × P] column-major (lda = 2P): top P rows = Σp², bottom P rows = Σhc.
//     sumsq_i = R(i,   j)   sumsq_j = R(j,   i)
//     hsum_i  = R(P+i, j)   hsum_j  = R(P+j, i)
//   G, Vpair are [P × P]. `size_t` indexing is mandatory above P≈32k (ROADMAP §4)
//   and free below it.
// =============================================================================
__global__ void assemble_f2_kernel(const double* __restrict__ G,
                                   const double* __restrict__ Vpair,
                                   const double* __restrict__ R,
                                   double* __restrict__ f2,
                                   int P) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (i >= P || j >= P) return;

    const size_t Pp = static_cast<size_t>(P);
    const size_t twoP = 2 * Pp;
    const size_t si = static_cast<size_t>(i);
    const size_t sj = static_cast<size_t>(j);

    const double Gij = G[si + sj * Pp];
    const double vp = Vpair[si + sj * Pp];
    const double sumsq_i = R[si + sj * twoP];          // R(i,   j)
    const double sumsq_j = R[sj + si * twoP];          // R(j,   i)
    const double hsum_i = R[(Pp + si) + sj * twoP];    // R(P+i, j)
    const double hsum_j = R[(Pp + sj) + si * twoP];    // R(P+j, i)

    const double num =
        assemble_f2_numerator(sumsq_i, sumsq_j, Gij, hsum_i, hsum_j);
    f2[si + sj * Pp] = finalize_f2(num, vp);
}

// -----------------------------------------------------------------------------
// Map the typed Precision to a cuBLAS compute type for the f2 GEMMs.
//   EmulatedFp64 -> CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT (Ozaki; default)
//   Fp64         -> CUBLAS_COMPUTE_64F                     (native; oracle)
//   Tf32         -> CUBLAS_COMPUTE_32F_FAST_TF32 path is data-typed FP32 and is
//                   not wired into this FP64-storage GEMM path (screening-only,
//                   architecture.md §12); EmulatedFp64 is the FP64-storage
//                   default. We therefore treat a Tf32 request on this FP64 path
//                   as native FP64 here (the dedicated TF32 screening path is a
//                   later milestone) — never silently downgrade reported numbers.
// -----------------------------------------------------------------------------
cublasComputeType_t compute_type_for(const Precision& precision) {
    switch (precision.kind) {
        case Precision::Kind::EmulatedFp64:
            return CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
        case Precision::Kind::Fp64:
        case Precision::Kind::Tf32:
        default:
            return CUBLAS_COMPUTE_64F;
    }
}

// -----------------------------------------------------------------------------
// Engage the precision policy on the handle (architecture.md §12; spike
// f2_timing.cu:91-99 fixed-bit engagement). For EmulatedFp64 this is the
// load-bearing cublasSet* sequence:
//   cublasSetMathMode(CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)
//   cublasSetEmulationStrategy(EAGER)                    -- force-emulate even tiny GEMMs
//   cublasSetFixedPointEmulationMantissaControl(FIXED)   -- FIXED slices, NOT dynamic
//   cublasSetFixedPointEmulationMaxMantissaBitCount(bits)
// FIXED control is the whole point: dynamic overshoots to ~60 bits and loses the
// win (ROADMAP §0 trap). For Fp64 we set PEDANTIC math (strict native FP64, the
// oracle path). The EAGER/FIXED tuning needs STEPPE_HAVE_EMU_TUNING; without it
// the emulated math mode still engages emulation (just not the pinned slice
// count), which is why the build flag is required for the measured speedup.
// -----------------------------------------------------------------------------
void engage_precision(cublasHandle_t handle, const Precision& precision) {
    if (precision.kind == Precision::Kind::EmulatedFp64) {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
#if STEPPE_HAVE_EMU_TUNING
        CUBLAS_CHECK(cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(
            handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(
            handle, precision.mantissa_bits));
#else
        (void)precision;
#endif
    } else {
        // Native FP64 oracle / fallback: strict, no tensor-core shortcuts.
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    }
}

}  // namespace

// =============================================================================
// Launch wrappers (architecture.md §7 — host code never sees <<<>>>).
// =============================================================================

void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream) {
    // 2-D block over (population, SNP); grid math from the one launch-config home
    // (core/internal/f2_estimator.hpp grid_for / kCdivBlock — replaces the spike's
    // open-coded (n+b-1)/b, ROADMAP §4). `long` grid extent for the SNP axis.
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::cdiv(M, static_cast<long>(steppe::kCdivBlock))));
    f2_feeder_kernel<<<grid, block, 0, stream>>>(dQ_raw, dV_raw, dN_raw,
                                                 dQ_masked, dV_out, dS, P, M);
    STEPPE_CUDA_CHECK_KERNEL();
}

void run_f2_gemms(cublasHandle_t handle, const Precision& precision,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR,
                  cudaStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    engage_precision(handle, precision);

    const cublasComputeType_t ct = compute_type_for(precision);
    const double one = 1.0;
    const double zero = 0.0;
    const int Mi = static_cast<int>(M);   // cublasGemmEx takes int k; M fits a tile
    const int twoP = 2 * P;

    // G[P × P] = Q · Qᵀ.  C(i,j) = Σ_s Q(i,s) Q(j,s).
    //   A=Q [P×M] OP_N (m=P, k=M, lda=P); B=Q [P×M] OP_T (n=P, ldb=P); C=G ldc=P.
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, Mi,
                              &one, dQ, CUDA_R_64F, P, dQ, CUDA_R_64F, P,
                              &zero, dG, CUDA_R_64F, P, ct, CUBLAS_GEMM_DEFAULT));

    // Vpair[P × P] = V · Vᵀ.  Vpair(i,j) = # SNPs valid in both i and j (RETAINED
    // as the S4 jackknife weight, architecture.md §5 S2 caveat (a)).
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, Mi,
                              &one, dV, CUDA_R_64F, P, dV, CUDA_R_64F, P,
                              &zero, dVpair, CUDA_R_64F, P, ct, CUBLAS_GEMM_DEFAULT));

    // R[2P × P] = S · Vᵀ.  S is [2P×M] (lda=2P); Vᵀ is [M×P].
    //   A=S OP_N (m=2P, k=M, lda=2P); B=V OP_T (n=P, ldb=P); C=R ldc=2P.
    // Top P rows = Σp², bottom P rows = Σhc (architecture.md §5 S2 line 238-240).
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, twoP, P, Mi,
                              &one, dS, CUDA_R_64F, twoP, dV, CUDA_R_64F, P,
                              &zero, dR, CUDA_R_64F, twoP, ct, CUBLAS_GEMM_DEFAULT));
}

void launch_assemble_f2(const double* dG, const double* dVpair, const double* dR,
                        double* dF2, int P, cudaStream_t stream) {
    // 2-D block over the [P × P] output; grid math from the one launch-config home
    // (replaces the spike's dim3 block(16,16), ROADMAP §4).
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(P)));
    assemble_f2_kernel<<<grid, block, 0, stream>>>(dG, dVpair, dR, dF2, P);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
