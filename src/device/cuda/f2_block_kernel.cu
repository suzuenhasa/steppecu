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
// precision policy). The FIXED-slice pin is what makes EmulatedFp64 HONORABLE:
// without it cuBLAS engages emulation under its DOCUMENTED DEFAULT mantissa
// control, `CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC` ("Dynamic mantissa control
// represents the cuBLAS library default mantissa control" — cuBLAS Library docs,
// FP64 floating-point emulation), which on real data's wide dynamic range
// overshoots to ~60 bits and collapses to parity with native FP64 (ROADMAP §0;
// architecture.md §12 line 726). That is the EXPLICITLY REJECTED trap — and it
// would still report the `EmulatedFp64` tag. So when the tuning is unavailable we
// do NOT silently run dynamic: `emulation_honorable()` is the ONE predicate that
// gates BOTH the math mode and the compute type (cleanup X-6/B2; architecture.md
// §9 build() "fall back to native Fp64 or error"); an unhonorable EmulatedFp64
// request DOWNGRADES to native Fp64 with an observable capability-tagged log line.
// Build on the remote CUDA-13 box with -DSTEPPE_HAVE_EMU_TUNING=1.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). It includes
// the SHARED host/device f2 primitive so the CPU oracle and this path cannot
// diverge on the formula (architecture.md §13; ROADMAP §5).
#include "device/cuda/f2_block_kernel.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include <atomic>   // std::atomic_flag — one-shot capability-tag emission
#include <limits>   // std::numeric_limits<int>::max — the M0 k-narrowing assert (B22)

#include "core/internal/f2_estimator.hpp"  // het_correction, assemble_f2_numerator,
                                           //   finalize_f2, grid_for, kCdivBlock
#include "core/internal/log.hpp"           // STEPPE_LOG_WARN (the one warn sink, B7/X-4)
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
using core::kF2StackedBlocks;  // the [2P × …] stacked-S row-block count (f2_estimator.hpp)

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
//
// COALESCING (cleanup 20.1/MED — uncoalesced global access): the [P×M] arenas are
// column-major, so the UNIT-STRIDE dimension is the population i, NOT the SNP s.
// The grid keeps M on the only 2^31 axis (gridDim.x) per the X-7/B6 cap fix, so we
// MUST NOT swap the grid. Instead we map threadIdx.x onto the population axis and
// threadIdx.y onto the SNP axis within the (square, 16×16) block: the block tile
// covers SNPs [bx·16, bx·16+16) × pops [by·16, by·16+16) exactly as before, but a
// warp's consecutive lanes (threadIdx.x) now vary i, so the loads Q/V/N_raw[idx]
// and the stores Q_masked/V_out/S[idx] land at CONSECUTIVE addresses i + P·s ⇒
// coalesced 16-double bursts instead of the previous P-strided 32-way scatter.
// This is per-element work (no inter-thread sharing), so the swap needs no shared
// tile / __syncthreads — it is exactly the square-block degenerate of the
// finding's tile-and-transpose, with identical per-element math, identical written
// addresses, and the grid orientation (M on x) preserved (§12 parity unchanged).
// =============================================================================
__global__ void f2_feeder_kernel(const double* __restrict__ Q_raw,
                                 const double* __restrict__ V_raw,
                                 const double* __restrict__ N_raw,
                                 double* __restrict__ Q_masked,
                                 double* __restrict__ V_out,
                                 double* __restrict__ S,
                                 int P, long M) {
    // Population i on threadIdx.x (the [P×M] column-major UNIT-STRIDE axis ⇒
    // coalesced); SNP s on threadIdx.y. The block tiles are still 16-wide on each
    // axis, so gridDim.x (= cdiv(M,16)) keeps covering M and gridDim.y (= cdiv(P,16))
    // covers P — the grid orientation (M on the only 2^31 axis) is unchanged
    // (X-7/B6), only the in-block thread→element map is transposed (20.1/MED).
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.y;  // SNP
    const long i = static_cast<long>(blockIdx.y) * blockDim.y + threadIdx.x;  // population
    if (i >= P || s >= M) return;

    const long Pl = static_cast<long>(P);
    const long idx = i + Pl * s;            // (i,s) in a [P  × M] matrix
    const long stack_idx = i + (kF2StackedBlocks * Pl) * s;  // (i,s) in the [2P × M] stacked S

    const bool valid = (V_raw[idx] != 0.0);
    // Hoist the single raw-Q load (20.3/LOW): Q_raw[idx] feeds BOTH the masked q
    // and the het correction below. With __restrict__ the compiler already CSEs to
    // one LDG; this makes the single fetch explicit. Masking semantics are intact —
    // het_correction re-derives its own validity from `valid`.
    const double qraw = Q_raw[idx];
    // Zero-fill where invalid: Q²=0 and the cross term G(i,j) vanish there, which
    // is what makes the masked GEMM reproduce the pairwise-complete reference
    // (architecture.md §5 S2; views.hpp Q/V/N contract).
    const double q = valid ? qraw : 0.0;
    // Shared per-element het correction (carries its own validity ⇒ 0 when
    // invalid). N is the non-missing HAPLOID count (Q/V/N contract); sample size
    // is per-SNP `N`, NEVER hardcoded (ROADMAP §4 — fixes the spike's /198.0).
    const double hc = het_correction(qraw, N_raw[idx], valid);

    Q_masked[idx] = q;
    V_out[idx] = valid ? 1.0 : 0.0;
    S[stack_idx] = q * q;        // Qsq block (rows 0..P-1)
    S[Pl + stack_idx] = hc;      // Hc  block (rows P..2P-1)
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
//
// COALESCING (cleanup 20.1/LOW — single-block analog of the f2_blocks assemble, same
// ACCEPTED COST): lanes vary si (row i), so the own-orientation reads (column factor
// sj) coalesce, while the MATHEMATICALLY-REQUIRED transposed sumsq_j=R(j,i) /
// hsum_j=R(P+j,i) (column factor si) stride by twoP ⇒ uncoalesced. Same rationale as
// assemble_blocks_group_kernel (f2_batched_kernel.cu): the SMALL [2P×P] R buffer off
// the GEMM-dominated critical path, native-FP64 catastrophic-cancellation assemble,
// per-element math unchanged — accepted, not shared-tiled (the symmetric access would
// need two off-diagonal tiles for a bounded off-critical-path win).
// =============================================================================
__global__ void assemble_f2_kernel(const double* __restrict__ G,
                                   const double* __restrict__ Vpair,
                                   const double* __restrict__ R,
                                   double* __restrict__ f2,
                                   int P) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    if (i >= P || j >= P) return;

    const size_t Pz = static_cast<size_t>(P);
    const size_t twoP = kF2StackedBlocks * Pz;
    const size_t si = static_cast<size_t>(i);
    const size_t sj = static_cast<size_t>(j);

    const double Gij = G[si + sj * Pz];
    const double vp = Vpair[si + sj * Pz];
    const double sumsq_i = R[si + sj * twoP];          // R(i,   j)
    const double sumsq_j = R[sj + si * twoP];          // R(j,   i)
    const double hsum_i = R[(Pz + si) + sj * twoP];    // R(P+i, j)
    const double hsum_j = R[(Pz + sj) + si * twoP];    // R(P+j, i)

    const double num =
        assemble_f2_numerator(sumsq_i, sumsq_j, Gij, hsum_i, hsum_j);
    f2[si + sj * Pz] = finalize_f2(num, vp);
}

}  // namespace

// -----------------------------------------------------------------------------
// emulation_honorable — THE ONE predicate that decides whether an EmulatedFp64
// request can be honored as the FIXED-slice Ozaki path (cleanup X-6/B2).
//
// EmulatedFp64 is honorable ONLY when the build carries the fixed-slice tuning
// API (STEPPE_HAVE_EMU_TUNING). Without it, `cublasSetFixedPointEmulationMantissa
// Control(FIXED)` cannot be called, so cuBLAS would engage emulation under its
// DOCUMENTED DEFAULT `CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC` (cuBLAS Library
// docs, FP64 floating-point emulation) — the rejected ~60-bit parity trap
// (architecture.md §12 line 726; ROADMAP §0). So when tuning is off we report
// EmulatedFp64 as NOT honorable and the caller downgrades to native Fp64.
//
// BOTH the math-mode engagement (`engage_f2_precision`) AND the compute-type
// mapping (`f2_compute_type`) consult THIS predicate, so they can never disagree:
// a request that is downgraded for the math mode is downgraded for the compute
// type in lockstep (the C-2 split is closed). It is the single source of the
// EmulatedFp64-honorability decision (architecture.md §2 DRY).
// -----------------------------------------------------------------------------
[[nodiscard]] bool emulation_honorable(const Precision& precision) noexcept {
    if (precision.kind != Precision::Kind::EmulatedFp64) return false;
#if STEPPE_HAVE_EMU_TUNING
    return true;
#else
    return false;
#endif
}

#if !STEPPE_HAVE_EMU_TUNING
namespace {

// Emit the EmulatedFp64-unavailable capability tag AT MOST ONCE per process so the
// downgrade is OBSERVABLE without spamming the per-call hot path (cleanup X-6/B2,
// T-CAP-1). Compiled ONLY on the no-tuning lane — the only build where an
// EmulatedFp64 request is downgraded — so the default (tuning-ON) build carries no
// unused helper (warnings-as-errors clean). Routes through the ONE warn sink
// (STEPPE_LOG_WARN, core/internal/log.hpp, B7/X-4) now that it exists — satisfying
// the §10 "no printf in library code" rule (this replaces the earlier interim
// `std::fprintf(stderr, …)`). STEPPE_LOG_WARN prefixes `[steppe][warn] ` and is
// debug-only (NDEBUG-silent, matching the sink's teardown-warn contract); the
// std::atomic_flag makes the one-shot guard thread-safe (M4.5 multi-GPU may engage
// from more than one host thread), and the test_and_set runs even under NDEBUG so a
// future verbose sink still fires at most once.
void warn_emulated_fp64_downgraded_once() {
    static std::atomic_flag emitted = ATOMIC_FLAG_INIT;
    if (!emitted.test_and_set(std::memory_order_relaxed)) {
        STEPPE_LOG_WARN(
            "[capability] EmulatedFp64 requested but the FIXED-slice Ozaki tuning is "
            "unavailable (built without -DSTEPPE_HAVE_EMU_TUNING) -> downgraded to "
            "native Fp64 [tag: emu_tuning_unavailable]. The reported precision is "
            "native FP64, NOT emulated (architecture.md §9, §12; cleanup X-6/B2).");
    }
}

}  // namespace
#endif  // !STEPPE_HAVE_EMU_TUNING

// -----------------------------------------------------------------------------
// Map the typed Precision to a cuBLAS compute type for the f2 GEMMs. The
// EmulatedFp64-honorability decision routes through `emulation_honorable` so the
// compute type and the math mode (engage_f2_precision) are ALWAYS derived from the
// same predicate and can never disagree (cleanup X-6/B2 C-2):
//   EmulatedFp64 (honorable)   -> CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT (Ozaki)
//   EmulatedFp64 (unhonorable) -> CUBLAS_COMPUTE_64F  (DOWNGRADED — tuning absent;
//                                  must match engage_f2_precision's PEDANTIC math)
//   Fp64                       -> CUBLAS_COMPUTE_64F                     (native; oracle)
//   Tf32                       -> CUBLAS_COMPUTE_32F_FAST_TF32 path is data-typed FP32
//                                 and is not wired into this FP64-storage GEMM path
//                                 (screening-only, architecture.md §12); treated as
//                                 native FP64 here — never silently downgrade reported
//                                 numbers. The dedicated TF32 path is a later milestone.
// Shared by the single-block (run_f2_gemms) and the M4 batched/grouped
// (run_f2_gemms_group) paths — the single source of the compute-type mapping.
// -----------------------------------------------------------------------------
cublasComputeType_t f2_compute_type(const Precision& precision) {
    if (emulation_honorable(precision)) return CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
    // Fp64, Tf32, AND an unhonorable EmulatedFp64 request all run native FP64.
    return CUBLAS_COMPUTE_64F;
}

// -----------------------------------------------------------------------------
// Engage the precision policy on the handle (architecture.md §12; spike
// f2_timing.cu:91-99 fixed-bit engagement). For an HONORABLE EmulatedFp64 request
// (STEPPE_HAVE_EMU_TUNING on) this is the load-bearing cublasSet* sequence:
//   cublasSetMathMode(CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)
//   cublasSetEmulationStrategy(EAGER)                    -- force-emulate even tiny GEMMs
//   cublasSetFixedPointEmulationMantissaControl(FIXED)   -- FIXED slices, NOT dynamic
//   cublasSetFixedPointEmulationMaxMantissaBitCount(bits)
// FIXED control is the whole point: dynamic overshoots to ~60 bits and loses the
// win (ROADMAP §0 trap).
//
// For Fp64 — AND for an EmulatedFp64 request the build CANNOT honor (tuning absent,
// `emulation_honorable()==false`) — we set PEDANTIC math (strict native FP64, the
// oracle/fallback path) and emit the capability tag ONCE. Critically we do NOT set
// the emulated math mode in the unhonorable case: doing so would engage cuBLAS's
// DYNAMIC-default emulation (the rejected trap) while `f2_compute_type` (consulting
// the SAME predicate) returns native CUBLAS_COMPUTE_64F — the downgrade is coherent
// across both halves (cleanup X-6/B2). Exposed (non-anonymous) so the M4 grouped
// path engages the SAME policy ONCE.
// -----------------------------------------------------------------------------
void engage_f2_precision(cublasHandle_t handle, const Precision& precision) {
    if (emulation_honorable(precision)) {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
#if STEPPE_HAVE_EMU_TUNING
        CUBLAS_CHECK(cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(
            handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(
            handle, precision.mantissa_bits));
#endif
    } else {
        // Native FP64 oracle / fallback: strict, no tensor-core shortcuts.
#if !STEPPE_HAVE_EMU_TUNING
        // On the no-tuning lane this branch ALSO catches a DOWNGRADED EmulatedFp64
        // request (emulation_honorable()==false for it) — surface it observably
        // exactly once (the foot-gun closes with a logged tag, not silently). On
        // the tuning-ON lane EmulatedFp64 is honorable, so it never reaches here
        // and the tag/helper are compiled out entirely.
        if (precision.kind == Precision::Kind::EmulatedFp64)
            warn_emulated_fp64_downgraded_once();
#endif
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    }
}

// =============================================================================
// Launch wrappers (architecture.md §7 — host code never sees <<<>>>).
// =============================================================================

void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream) {
    // 2-D block over (SNP-tile on gridDim.x, population-tile on gridDim.y); grid
    // math from the one launch-config home (core/internal/launch_config.hpp cdiv /
    // grid_for / kCdivBlock — replaces the spike's open-coded (n+b-1)/b, ROADMAP §4).
    // ORIENTATION (cleanup X-7/B6): the SNP count M rides gridDim.x via the
    // `cdiv(long,long)` overload — x is the only axis that reaches 2^31−1, so M
    // (which `MatView::M`'s `long` type permits past 2^31, and which exceeds the
    // 65 535 y/z cap already at M > ~1.05M SNPs) MUST ride it. P rides gridDim.y
    // through `grid_for`, whose y/z-cap assert applies (P ≤ a few thousand ≪ the
    // 65 535 y/z cap, so it is always satisfied). The block is SQUARE (16×16), so
    // the in-block thread→element
    // map is transposed INSIDE the kernel (threadIdx.x→pop, threadIdx.y→SNP) for
    // coalesced column-major access (cleanup 20.1/MED) WITHOUT moving M off gridDim.x
    // — the grid orientation here is unchanged.
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::cdiv(M, static_cast<long>(steppe::kCdivBlock))),
                    static_cast<unsigned>(core::grid_for(P)));
    f2_feeder_kernel<<<grid, block, 0, stream>>>(dQ_raw, dV_raw, dN_raw,
                                                 dQ_masked, dV_out, dS, P, M);
    STEPPE_CUDA_CHECK_KERNEL();
}

void run_f2_gemms(cublasHandle_t handle, const Precision& precision,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR) {
    // No cublasSetStream here: the handle's stream + emulated-FP64 workspace are
    // bound ONCE at backend construction via CublasHandle::set_stream
    // (architecture.md §12; cleanup X-1/B1). A per-call cublasSetStream would
    // "unconditionally reset the cuBLAS library workspace back to the default
    // workspace pool" (cuBLAS §2.4.7), silently discarding the determinism
    // workspace before every GEMM batch — the exact defect B1 fixes.
    engage_f2_precision(handle, precision);

    const cublasComputeType_t ct = f2_compute_type(precision);
    const double one = 1.0;
    const double zero = 0.0;
    // cublasGemmEx's contraction extent `k` is a signed `int` (only the _64
    // variants take int64_t). M is `long` (the M0 path uploads ALL SNPs and does
    // NOT tile), so M > INT_MAX would make this narrowing yield a wrapped/negative
    // k → cuBLAS rejects or silently contracts over fewer SNPs. The sole caller,
    // CudaBackend::compute_f2 (cuda_backend.cu), GUARDS `M <= INT_MAX` and throws a
    // typed error BEFORE this point (cleanup f2_block_kernel E-1, B22); the
    // debug-only assert pins that invariant at the narrowing itself.
    STEPPE_ASSERT(M <= static_cast<long>(std::numeric_limits<int>::max()),
                  "run_f2_gemms: M exceeds INT_MAX; cublasGemmEx k is a 32-bit int "
                  "(guarded by CudaBackend::compute_f2, B22)");
    const int Mi = static_cast<int>(M);   // safe: M <= INT_MAX (guarded upstream)
    const int twoP = kF2StackedBlocks * P;

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
    // (replaces the spike's dim3 block(16,16), ROADMAP §4). P rides BOTH axes via
    // grid_for; as in launch_f2_feeder, grid_for's y/z-cap assert is what enforces
    // `P fits gridDim.y` (P ≤ a few thousand ≪ the 65 535 y/z cap, always satisfied).
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(P)));
    assemble_f2_kernel<<<grid, block, 0, stream>>>(dG, dVpair, dR, dF2, P);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
