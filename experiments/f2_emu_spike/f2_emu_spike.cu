// f2_emu_spike.cu
//
// Empirical test of the steppe S2 policy question (ADR-0010 / architecture.md
// section 12): can the f2 GEMMs run on cuBLAS FP64 fixed-point (Ozaki) emulation
// instead of native FP64, given the catastrophic cancellation inside the
// numerator  Sigma (p_i - p_j)^2 = Sigma p_i^2 - 2 Sigma p_i p_j + Sigma p_j^2 ?
// (MEASURED ANSWER -- this spike is the evidence behind it: YES. On real AADR the
//  f2 GEMMs default to FIXED-slice Ozaki emulation, mantissa_bits=40 (~= native
//  FP64 accuracy, 2.2e-11 worst-case) at 8-17x native FP64 throughput; 32 bits
//  gives 8.6e-9 even faster. See architecture.md sections 5 S2 / 9 / 12 and
//  ROADMAP section 0. The earlier draft conclusion "f2 GEMMs must stay native
//  FP64" PRE-DATES this real-data measurement and is superseded -- only the small
//  numerator/divide stays native FP64. The decisive caveat, also measured here:
//  DYNAMIC mantissa control is the rejected TRAP -- it overshoots to ~60 bits on
//  real data's wide dynamic range and collapses to parity with native (no win);
//  the win comes from FIXED slices.)
// (Bit-sweep evidence lives in the follow-ups, NOT this file: f2_prec_acc.cu =
//  fixed-bit ACCURACY sweep {24,32,40,48,53} vs the long-double reference;
//  f2_timing.cu = fixed-bit THROUGHPUT sweep {32,40,48}. This file uses dynamic
//  mantissa, so read it for the engagement guard + oracle, not the bit frontier.)
//
// Three computations of the same P x P f2 matrix on identical input:
//   (1) NATIVE FP64    : cublasGemmEx with CUBLAS_COMPUTE_64F (native hardware).
//   (2) EMULATED FP64  : cublasGemmEx with CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
//                        and math-mode CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (the
//                        two load-bearing mechanisms; architecture.md line 89).
//                        The tuning hooks (emulation strategy, FIXED mantissa
//                        control, max mantissa-bit count) are compiled under
//                        -DSTEPPE_HAVE_EMU_TUNING. All of these symbols are now
//                        CONFIRMED working in standard cuBLAS 13 -- the f2 kernel
//                        builds clean and the emulation arm engages on the CUDA-13
//                        / sm_120 box (the names were verified against the live
//                        CUDA 13 cublas_api.h; see the engagement note below).
//   (3) REFERENCE      : CPU long double, cancellation-free -- forms (p_i - p_j)
//                        per SNP and squares it directly, with pairwise (cascade)
//                        summation so the accumulator error is ~log2(M)*u_ld, well
//                        below the f2 magnitude. This is the ground-truth oracle,
//                        computed INDEPENDENTLY from the host inputs (never from
//                        any GPU sum).
//
// The final numerator+divide is held in NATIVE FP64 in BOTH candidate arms, so the
// experiment isolates GEMM-accumulation precision -- the actual policy question.
//
// ---------------------------------------------------------------------------
// TWO INPUT MODES
// ---------------------------------------------------------------------------
//   SYNTHETIC : ./f2_emu_spike P M sigma [missing_frac] [seed]
//               generate random data with a tunable cancellation dial (sigma).
//   REAL DATA : ./f2_emu_spike --load <dir>
//               load Q.f64 / V.f64 / N.f64 (+ shape.txt) produced by
//               aadr/02_build_matrix.py, per the SHARED BINARY FORMAT:
//                 shape.txt : one ASCII line "P M" (fscanf-friendly)
//                 Q.f64     : raw little-endian float64, P*M doubles, COLUMN-MAJOR
//                             [P x M] (element (pop i, snp s) at i + P*s), ld = P.
//                             Q = freq of the FIXED reference allele in [0,1].
//                 V.f64     : 1.0 if data present else 0.0, same layout.
//                 N.f64     : non-missing HAPLOID count = 2*non-missing diploids.
//               From (Q_raw, V, N) the loader reconstructs EXACTLY the same host
//               arrays the synthetic path builds before the GEMMs (Q_masked, Hc,
//               Qsq, S) and then runs the IDENTICAL pipeline.
//
// ---------------------------------------------------------------------------
// HOW WE PROVE EMULATION ACTUALLY ENGAGED (defect (2), the load-bearing check)
// ---------------------------------------------------------------------------
// Emulation could silently fall back to native FP64 (unsupported arch, strategy
// declining to emulate tiny GEMMs, an ignored compute type, etc.), which would
// make the emu column secretly measure native FP64 and FALSELY "prove" emulation
// is safe. We guard against this HEADER-INDEPENDENTLY by comparing the emulated
// and native f2 matrices BIT-FOR-BIT:
//   * Emulated FP64 (Ozaki) is accuracy-approximate and NOT bit-identical to
//     native FP64 (architecture.md line 614/738: "NOT bit-identical to native
//     FP64"). So if emu == native bit-for-bit on a non-trivial input, emulation
//     was almost certainly NOT engaged.
//   * We print emuEqNat (yes/NO) and, when STEPPE_HAVE_EMU_TUNING is defined, the
//     mantissa-bit verification hook (mBits >= 0 confirms engagement, -1 = not).
//   * If emulation was requested but emu == native bit-for-bit (and the hook did
//     not positively confirm engagement), we emit a LOUD stderr warning and force
//     the verdict to FAIL_IGNORED -- the experiment did not actually test
//     emulation, so any "PASS" would be meaningless.
//
// Target: sm_120 (RTX 5090), CUDA 13, link with -lcublas.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120 -DSTEPPE_HAVE_EMU_TUNING=1 f2_emu_spike.cu -lcublas
//   (drop -DSTEPPE_HAVE_EMU_TUNING to compile without the unconfirmed tuning
//    symbols; the load-bearing emulated compute type + math mode are used
//    unconditionally either way.)
//
// Run:
//   ./f2_emu_spike P M sigma [missing_frac] [seed]      (synthetic)
//   ./f2_emu_spike --load <dir>                         (real AADR data)
//
// ---------------------------------------------------------------------------
// THE f2 STATISTIC AND ITS GEMM REFORMULATION
// ---------------------------------------------------------------------------
// For two populations i and j, the (unbiased) f2 statistic averaged over SNPs is
//
//   f2(i,j) = mean_s [ (p_{i,s} - p_{j,s})^2 - hc_{i,s} - hc_{j,s} ]
//
// where p_{i,s} is the allele frequency of population i at SNP s and
// hc_{i,s} = p_{i,s}(1 - p_{i,s}) / max(N - 1, 1) is the within-population
// heterozygosity correction (N = haploid sample size). The mean is taken over
// SNPs jointly genotyped (valid) in both i and j.
//
// Expanding (p_i - p_j)^2 = p_i^2 - 2 p_i p_j + p_j^2 lets us express the
// numerator over valid SNPs purely with matrix products. With column-major
// matrices over SNPs s = 0..M-1:
//
//   Q   [P  x M] : allele frequencies          (row = population, col = SNP)
//   V   [P  x M] : validity mask (1.0 / 0.0)
//   Qsq [P  x M] : Q .* Q
//   Hc  [P  x M] : Q .* (1 - Q) ./ max(N-1,1) .* V
//   S   [2P x M] : vertical stack [ Qsq ; Hc ]
//
// Three GEMMs (all column-major):
//   G     [P  x P] = Q * Q^T       -> G(i,j)     = sum_s p_{i,s} p_{j,s}
//   Vpair [P  x P] = V * V^T       -> Vpair(i,j) = # SNPs valid in both i and j
//   R     [2P x P] = S * V^T       -> R(i,j)     = sum_s Qsq_{i,s} V_{j,s}   (top P rows)
//                                     R(P+i,j)   = sum_s Hc_{i,s} V_{j,s}    (bottom P rows)
//
// (Hc already carries its own V_i factor, so R's bottom block restricts Hc_i to
//  SNPs valid in BOTH i and j once multiplied by V_j.)
//
// Per pair, the summed numerator over jointly-valid SNPs is
//
//   num(i,j) = SUMsq_i + SUMsq_j - 2 G(i,j) - Hsum_i - Hsum_j
//
// where, using the R blocks (matching architecture.md line 240):
//   SUMsq_i = R(i,   j)      SUMsq_j = R(j,   i)
//   Hsum_i  = R(P+i, j)      Hsum_j  = R(P+j, i)
//
// and f2(i,j) = num(i,j) / Vpair(i,j).
//
// NOTE on validity masking of Q: so that R(i,j) only sums over SNPs valid in i
// (not merely j), Q is zeroed at invalid entries of population i. Then p_{i,s}^2 = 0
// there, V_{j,s} multiplies it, and the cross term G(i,j) = sum_s p_{i,s} p_{j,s}
// also vanishes wherever either is invalid -- so the GEMM form matches the masked
// reference exactly.
//
// ---------------------------------------------------------------------------
// COLUMN-MAJOR GEMM CORRECTNESS (defect (1)) -- VERIFIED, no bug
// ---------------------------------------------------------------------------
// cublasGemmEx computes C = op(A) * op(B), all column-major; C is m x n,
// op(A) is m x k, op(B) is k x n. Data is laid out [rows x cols] column-major
// with leading dimension = #rows.
//
//   G[P x P] = Q*Q^T,  G(i,j) = sum_s Q(i,s) Q(j,s):
//     A = Q [P x M] (OP_N) -> m x k = P x M  => m=P, k=M
//     B = Q [P x M] (OP_T) -> Q^T is M x P, i.e. k x n = M x P => n=P
//     C = G [P x P], lda=P, ldb=P, ldc=P.
//     => OP_N, OP_T, m=P, n=P, k=M, A=dQ lda=P, B=dQ ldb=P, C=dG ldc=P.  CORRECT.
//
//   Vpair[P x P] = V*V^T: identical shape with V.  CORRECT.
//
//   R[2P x P] = S*V^T,  R(r,j) = sum_s S(r,s) V(j,s):
//     A = S [2P x M] (OP_N) -> m x k = 2P x M => m=2P, k=M, lda=2P
//     B = V [P  x M] (OP_T) -> V^T is M x P, i.e. k x n = M x P => n=P, ldb=P
//     C = R [2P x P], ldc=2P.
//     => OP_N, OP_T, m=2P, n=P, k=M, A=dS lda=2P, B=dV ldb=P, C=dR ldc=2P.  CORRECT.
//
// The op/transpose flags, m/n/k and lda/ldb/ldc below match this derivation
// exactly. (The review brief flagged this as "the most likely bug" -- it is in
// fact correct; the real defect was the missing emulation-engagement check.)
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// STEPPE_HAVE_EMU_TUNING gates the cuBLAS emulation *tuning* symbols whose exact
// names/signatures we could NOT verify against any header available in this
// environment (only CUDA 11.8 is installed, which has no emulation API at all).
// The two LOAD-BEARING mechanisms -- the emulated compute type
// (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT) and the math mode
// (CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH) -- ARE confirmed in architecture.md
// line 89 and are used unconditionally. Everything else is opt-in so the file
// always compiles on a stock CUDA 13 toolkit; flip the macro on once the names
// are confirmed against the live cuBLAS 13 headers.
#ifndef STEPPE_HAVE_EMU_TUNING
#define STEPPE_HAVE_EMU_TUNING 0
#endif

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",             \
                         __FILE__, __LINE__, #expr,                            \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (expr);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error %s:%d: '%s' -> status %d\n",    \
                         __FILE__, __LINE__, #expr, (int)st__);                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Column-major indexing helper: element (row r, col c) of an [rows x cols]
// matrix lives at r + c*rows.  All-size_t to forestall 32-bit index overflow.
// ---------------------------------------------------------------------------
static inline size_t cm(size_t r, size_t c, size_t rows) { return r + c * rows; }

// ---------------------------------------------------------------------------
// Device kernel: assemble f2[P x P] (column-major) from the three GEMM outputs.
//   G     [P  x P] column-major
//   Vpair [P  x P] column-major
//   R     [2P x P] column-major  (top P rows = SUMsq, bottom P rows = Hsum)
// f2(i,j) = ( R(i,j) + R(j,i) - 2*G(i,j) - R(P+i,j) - R(P+j,i) ) / Vpair(i,j)
// Held in native FP64 in both candidate arms (the final subtract is not the
// thing under test; the GEMM accumulation is).
// ---------------------------------------------------------------------------
__global__ void assemble_f2_kernel(const double* __restrict__ G,
                                    const double* __restrict__ Vpair,
                                    const double* __restrict__ R,
                                    double* __restrict__ f2,
                                    int P)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col
    if (i >= P || j >= P) return;

    const size_t Pp   = (size_t)P;
    const size_t twoP = (size_t)2 * Pp;

    double Gij      = G[(size_t)i + (size_t)j * Pp];
    double vp       = Vpair[(size_t)i + (size_t)j * Pp];
    double sumsq_i  = R[(size_t)i        + (size_t)j * twoP]; // R(i,   j)
    double sumsq_j  = R[(size_t)j        + (size_t)i * twoP]; // R(j,   i)
    double hsum_i   = R[(Pp + (size_t)i) + (size_t)j * twoP]; // R(P+i, j)
    double hsum_j   = R[(Pp + (size_t)j) + (size_t)i * twoP]; // R(P+j, i)

    double num = sumsq_i + sumsq_j - 2.0 * Gij - hsum_i - hsum_j;
    double den = (vp > 0.0) ? vp : 1.0;  // Vpair==0 => 0 (num is also 0 then)
    f2[(size_t)i + (size_t)j * Pp] = num / den;
}

// ---------------------------------------------------------------------------
// Run the three GEMMs on a given handle via cublasGemmEx with an explicit
// compute type, then assemble the f2 matrix. Times the three GEMMs together.
// All device buffers are pre-allocated and inputs already uploaded.
//   computeType = CUBLAS_COMPUTE_64F                      -> native FP64
//   computeType = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT  -> Ozaki emulation
// ---------------------------------------------------------------------------
static void run_f2_gemms(cublasHandle_t handle,
                         cublasComputeType_t computeType,
                         int P, int M,
                         const double* dQ,   // [P  x M]
                         const double* dV,   // [P  x M]
                         const double* dS,   // [2P x M]
                         double* dG,         // [P  x P]
                         double* dVpair,     // [P  x P]
                         double* dR,         // [2P x P]
                         double* dF2,        // [P  x P]
                         std::vector<double>& hF2_out,
                         float* gemm_ms_out)
{
    const double one  = 1.0;
    const double zero = 0.0;
    const int    twoP = 2 * P;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0, 0));

    // G[P x P] = Q * Q^T.   C(i,j) = sum_s Q(i,s) Q(j,s).   (see header proof)
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              P, P, M,
                              &one,
                              dQ, CUDA_R_64F, P,
                              dQ, CUDA_R_64F, P,
                              &zero,
                              dG, CUDA_R_64F, P,
                              computeType, CUBLAS_GEMM_DEFAULT));

    // Vpair[P x P] = V * V^T.
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              P, P, M,
                              &one,
                              dV, CUDA_R_64F, P,
                              dV, CUDA_R_64F, P,
                              &zero,
                              dVpair, CUDA_R_64F, P,
                              computeType, CUBLAS_GEMM_DEFAULT));

    // R[2P x P] = S * V^T.  S is [2P x M] (lda=2P); V is [P x M] -> V^T is [M x P].
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_T,
                              twoP, P, M,
                              &one,
                              dS, CUDA_R_64F, twoP,
                              dV, CUDA_R_64F, P,
                              &zero,
                              dR, CUDA_R_64F, twoP,
                              computeType, CUBLAS_GEMM_DEFAULT));

    CUDA_CHECK(cudaEventRecord(ev1, 0));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(gemm_ms_out, ev0, ev1));

    // ---- Assemble f2 (native FP64 in both arms; not timed) ------------------
    dim3 block(16, 16);
    dim3 grid((P + block.x - 1) / block.x, (P + block.y - 1) / block.y);
    assemble_f2_kernel<<<grid, block>>>(dG, dVpair, dR, dF2, P);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    hF2_out.resize((size_t)P * P);
    CUDA_CHECK(cudaMemcpy(hF2_out.data(), dF2,
                          sizeof(double) * (size_t)P * P,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
}

// ---------------------------------------------------------------------------
// Pairwise (cascade) summation of an array of long doubles. Reduces the
// summation error factor from O(M) to O(log2 M), making the reference's
// accumulator error negligible (~log2(M)*u_ld) relative to the f2 magnitude.
// Does not mutate the input.
// ---------------------------------------------------------------------------
static long double pairwise_sum(const long double* a, size_t n)
{
    if (n == 0) return 0.0L;
    if (n <= 128) {            // small-block base case: straight sum
        long double s = 0.0L;
        for (size_t k = 0; k < n; ++k) s += a[k];
        return s;
    }
    size_t h = n / 2;
    return pairwise_sum(a, h) + pairwise_sum(a + h, n - h);
}

// ---------------------------------------------------------------------------
// Reference f2 on CPU in long double. Cancellation-free: forms (p_i - p_j) per
// SNP and squares it directly (never the p^2 - 2pq + q^2 form), accumulating the
// per-SNP summands  (p_i - p_j)^2 - hc_i - hc_j  with pairwise summation. This is
// the trustworthy ground-truth oracle the GEMM reformulations are scored against,
// computed INDEPENDENTLY from the host inputs (defect (3)) -- no GPU sum feeds it.
//   Q, V, Hc are host column-major [P x M].
// Also returns, per pair, the cancellation severity of the EXPANDED (GEMM-form)
// numerator,  kappa = Sigma(p_i^2 + p_j^2) / |Sigma(p_i-p_j)^2|  (the quantity the
// emulated GEMMs accumulate, BEFORE the het correction), so error can be read
// against actual conditioning (printed as kappaMax).
// ---------------------------------------------------------------------------
static void reference_f2(int P, int M,
                         const std::vector<double>& Q,   // [P x M] (masked)
                         const std::vector<double>& V,   // [P x M]
                         const std::vector<double>& Hc,  // [P x M]
                         std::vector<double>& f2_ref,    // out [P x P]
                         double& kappaMax_out)           // out worst cancellation
{
    f2_ref.assign((size_t)P * P, 0.0);
    kappaMax_out = 0.0;
    std::vector<long double> terms((size_t)M);   // per-SNP f2 summands
    std::vector<long double> sq((size_t)M);       // per-SNP (p_i-p_j)^2 (for kappa)
    for (int j = 0; j < P; ++j) {
        for (int i = 0; i < P; ++i) {
            size_t cnt = 0;
            long double bigmag = 0.0L;     // Sigma (p_i^2 + p_j^2) over valid pairs
            for (int s = 0; s < M; ++s) {
                double vi = V[cm(i, s, P)];
                double vj = V[cm(j, s, P)];
                if (vi != 0.0 && vj != 0.0) {
                    long double pi = (long double)Q[cm(i, s, P)];
                    long double pj = (long double)Q[cm(j, s, P)];
                    long double d  = pi - pj;
                    long double hi = (long double)Hc[cm(i, s, P)];
                    long double hj = (long double)Hc[cm(j, s, P)];
                    sq[cnt]      = d * d;
                    terms[cnt]   = d * d - hi - hj;
                    ++cnt;
                    bigmag += pi * pi + pj * pj;
                }
            }
            long double num = pairwise_sum(terms.data(), cnt);
            long double den = (cnt > 0) ? (long double)cnt : 1.0L;
            f2_ref[cm(i, j, P)] = (double)(num / den);

            // Cancellation severity of the EXPANDED numerator the GEMMs build:
            //   sumExpanded = Sigma(p_i-p_j)^2 (the true small value);
            //   bigmag      = Sigma(p_i^2+p_j^2) (the large like-magnitude sums).
            if (i != j) {
                long double sumExpanded = pairwise_sum(sq.data(), cnt);
                long double an = (sumExpanded < 0.0L) ? -sumExpanded : sumExpanded;
                if (an > 0.0L) {
                    double kappa = (double)(bigmag / an);
                    if (kappa > kappaMax_out) kappaMax_out = kappa;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Relative-error statistics of a candidate vs the long-double reference, over
// the P(P-1) off-diagonal entries. Uses a combined-form tolerance (defect (5)):
//   relerr = |c - r| / max(|r|, atol_floor)
// so pairs whose true f2 is at/below the noise floor are scored by absolute
// error and do not blow up the relative metric (divide-by-zero guard). Also
// counts sign flips on pairs whose |ref| is above the floor (a FAIL signature).
// ---------------------------------------------------------------------------
struct ErrStats { double maxRel; double medRel; double meanAbsF2; int signFlips; };

static ErrStats rel_error_stats(const std::vector<double>& cand,
                                const std::vector<double>& ref,
                                int P)
{
    std::vector<double> rels;
    rels.reserve((size_t)P * P);
    long double absAccum = 0.0L;
    size_t absCount = 0;
    int signFlips = 0;

    const double atol_floor = 1e-15; // noise floor for near-zero f2
    for (int j = 0; j < P; ++j) {
        for (int i = 0; i < P; ++i) {
            if (i == j) continue;                 // diagonal is ~0 by construction
            double r = ref[cm(i, j, P)];
            double c = cand[cm(i, j, P)];
            double denom = std::fabs(r);
            if (denom < atol_floor) denom = atol_floor;
            rels.push_back(std::fabs(c - r) / denom);
            absAccum += std::fabs((long double)r);
            ++absCount;
            if (std::fabs(r) > atol_floor && ((r > 0.0) != (c > 0.0)) && c != 0.0)
                ++signFlips;
        }
    }

    ErrStats es{0.0, 0.0, 0.0, signFlips};
    if (!rels.empty()) {
        es.maxRel = *std::max_element(rels.begin(), rels.end());
        std::sort(rels.begin(), rels.end());
        size_t n = rels.size();
        es.medRel = (n & 1) ? rels[n / 2]
                            : 0.5 * (rels[n / 2 - 1] + rels[n / 2]);
    }
    es.meanAbsF2 = (absCount > 0) ? (double)(absAccum / (long double)absCount) : 0.0;
    return es;
}

// ---------------------------------------------------------------------------
// PASS/FAIL verdict tied to the steppe est tolerance tiers (architecture.md
// section 12, "Tolerance policy": est tier rtol ~1e-9 .. 1e-6).
//
// FAIL_IGNORED is the loud failure for defect (2): emulation was requested but
// its output was bit-identical to native (and the mantissa hook did not confirm
// engagement), so the emu column reflects native FP64 -- the experiment did not
// actually test emulation, and a numeric PASS would be meaningless.
//
//   FAIL_IGNORED : emulation requested but emu == native bit-for-bit (& hook
//                  did not confirm engagement)
//   PASS         : max_relerr_emu <= 1e-9  AND no sign flips
//   MARGINAL     : 1e-9 < max_relerr_emu <= 1e-6  AND emu within 10x native
//   FAIL         : max_relerr_emu > 1e-6, OR emu > 10x native, OR sign flips
// ---------------------------------------------------------------------------
static const char* verdict(const ErrStats& emu, const ErrStats& nat,
                           bool emulation_engaged)
{
    if (!emulation_engaged)                                 return "FAIL_IGNORED";
    if (emu.signFlips > 0)                                  return "FAIL";
    if (emu.maxRel > 1e-6)                                  return "FAIL";
    if (emu.maxRel > 10.0 * std::max(nat.maxRel, 1e-300))   return "FAIL";
    if (emu.maxRel <= 1e-9)                                 return "PASS";
    return "MARGINAL";
}

// ---------------------------------------------------------------------------
// Build the host arrays Q_masked, V, Hc, S (column-major [P x M] / [2P x M])
// from raw reference-allele freq Q_raw, validity V_raw, and haploid count N_raw.
// This is the SHARED reconstruction used by BOTH the synthetic and the --load
// paths so the GEMM inputs are byte-for-byte the same recipe:
//   Q_masked = Q_raw * V                       (mask invalid entries to 0)
//   Hc       = Q_raw*(1-Q_raw)/max(N-1,1) * V  (het correction; carries V_i)
//   Qsq      = Q_masked * Q_masked             (matches synthetic: S top block)
//   S        = [ Qsq ; Hc ]   vertically stacked -> [2P x M], lda = 2P
// Q_raw is expected in [0,1] and V in {0,1}; on the synthetic path these hold by
// construction, on the --load path they are asserted by the writer.
// ---------------------------------------------------------------------------
static void build_host_arrays(int P, int M,
                              const std::vector<double>& Q_raw,  // [P x M]
                              const std::vector<double>& V_raw,  // [P x M]
                              const std::vector<double>& N_raw,  // [P x M]
                              std::vector<double>& Q,            // out [P x M] masked
                              std::vector<double>& V,            // out [P x M]
                              std::vector<double>& Hc,           // out [P x M]
                              std::vector<double>& S)            // out [2P x M]
{
    Q.assign((size_t)P * M, 0.0);
    V.assign((size_t)P * M, 0.0);
    Hc.assign((size_t)P * M, 0.0);
    S.assign((size_t)(2 * P) * M, 0.0);

    for (int s = 0; s < M; ++s) {
        for (int i = 0; i < P; ++i) {
            const size_t idx = cm((size_t)i, (size_t)s, (size_t)P);
            double valid = (V_raw[idx] != 0.0) ? 1.0 : 0.0;
            double praw  = Q_raw[idx];
            double n     = N_raw[idx];
            double ncorr = std::max(n - 1.0, 1.0);

            double pq = (valid != 0.0) ? praw : 0.0;                       // Q_masked
            double hc = (valid != 0.0) ? (praw * (1.0 - praw) / ncorr) : 0.0;

            Q [idx] = pq;
            V [idx] = valid;
            Hc[idx] = hc;

            // S = [Qsq ; Hc] stacked vertically -> [2P x M], lda = 2P.
            S[cm((size_t)i,     (size_t)s, (size_t)2 * P)] = pq * pq; // Qsq block
            S[cm((size_t)P + i, (size_t)s, (size_t)2 * P)] = hc;      // Hc  block
        }
    }
}

// ---------------------------------------------------------------------------
// Load the SHARED BINARY FORMAT from <dir>: shape.txt (P M) then Q/V/N.f64
// (each P*M little-endian doubles, column-major [P x M], ld = P). Fails loud.
// ---------------------------------------------------------------------------
static void read_f64(const std::string& path, std::vector<double>& out, size_t count)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out.resize(count);
    size_t got = std::fread(out.data(), sizeof(double), count, f);
    size_t extra = 0;
    if (got == count) {
        // detect a trailing byte mismatch (wrong P*M) by probing one more read
        double probe;
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
            "ERROR: %s has MORE than %zu doubles (trailing data). Shape/format "
            "mismatch.\n", path.c_str(), count);
        std::exit(EXIT_FAILURE);
    }
}

static void load_real_data(const std::string& dir,
                           int& P_out, int& M_out,
                           std::vector<double>& Q_raw,
                           std::vector<double>& V_raw,
                           std::vector<double>& N_raw)
{
    std::string shapePath = dir + "/shape.txt";
    FILE* sf = std::fopen(shapePath.c_str(), "r");
    if (!sf) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", shapePath.c_str());
        std::exit(EXIT_FAILURE);
    }
    int P = 0, M = 0;
    if (std::fscanf(sf, "%d %d", &P, &M) != 2) {
        std::fprintf(stderr, "ERROR: %s must contain 'P M' (two ints)\n",
                     shapePath.c_str());
        std::fclose(sf);
        std::exit(EXIT_FAILURE);
    }
    std::fclose(sf);
    if (P <= 1 || M <= 0) {
        std::fprintf(stderr, "ERROR: bad shape from %s: P=%d M=%d (need P>1, M>0)\n",
                     shapePath.c_str(), P, M);
        std::exit(EXIT_FAILURE);
    }

    const size_t count = (size_t)P * (size_t)M;
    read_f64(dir + "/Q.f64", Q_raw, count);
    read_f64(dir + "/V.f64", V_raw, count);
    read_f64(dir + "/N.f64", N_raw, count);

    // Light validation against the SHARED FORMAT invariants (warn, don't abort,
    // so a slightly-off real dataset can still be inspected).
    size_t badQ = 0, badV = 0, badN = 0, mism = 0;
    for (size_t k = 0; k < count; ++k) {
        if (Q_raw[k] < 0.0 || Q_raw[k] > 1.0) ++badQ;
        if (V_raw[k] != 0.0 && V_raw[k] != 1.0) ++badV;
        if (N_raw[k] < 0.0) ++badN;
        bool vpos = (V_raw[k] != 0.0);
        bool npos = (N_raw[k] > 0.0);
        if (vpos != npos) ++mism;
    }
    if (badQ) std::fprintf(stderr, "  [warn] %zu Q entries outside [0,1]\n", badQ);
    if (badV) std::fprintf(stderr, "  [warn] %zu V entries not in {0,1}\n", badV);
    if (badN) std::fprintf(stderr, "  [warn] %zu N entries < 0\n", badN);
    if (mism) std::fprintf(stderr,
        "  [warn] %zu entries where V!=(N>0) (writer guarantees they match)\n", mism);

    P_out = P;
    M_out = M;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // -----------------------------------------------------------------------
    // Mode selection: --load <dir> (real data) vs synthetic (P M sigma ...).
    // -----------------------------------------------------------------------
    bool loadMode = (argc >= 2 && std::strcmp(argv[1], "--load") == 0);

    if (!loadMode && argc < 4) {
        std::fprintf(stderr,
            "usage:\n"
            "  %s P M sigma [missing_frac] [seed]   (synthetic)\n"
            "  %s --load <dir>                       (real AADR data)\n"
            "\n"
            "  P            number of populations\n"
            "  M            number of SNPs\n"
            "  sigma        per-SNP frequency dispersion (cancellation dial)\n"
            "  missing_frac fraction of (pop,SNP) entries marked invalid [default 0]\n"
            "  seed         RNG seed (recorded for reproducibility) [default 12345]\n"
            "  <dir>        directory with shape.txt + Q.f64/V.f64/N.f64\n"
            "               (SHARED BINARY FORMAT; see aadr/02_build_matrix.py)\n",
            argv[0], argv[0]);
        return EXIT_FAILURE;
    }
    if (loadMode && argc < 3) {
        std::fprintf(stderr, "usage: %s --load <dir>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int    P = 0, M = 0;
    double sigma = 0.0;          // synthetic-only; NaN-printed in load mode
    double missingFrac = 0.0;    // synthetic-only
    unsigned long long seed = 12345ULL;
    std::string srcDir;          // for the table's source label in load mode

    // Host raw inputs (column-major [P x M]) -- the common interface both paths
    // feed into build_host_arrays().
    std::vector<double> Q_raw, V_raw, N_raw;

    if (loadMode) {
        srcDir = argv[2];
        load_real_data(srcDir, P, M, Q_raw, V_raw, N_raw);
        sigma       = std::nan("");   // not meaningful for real data
        missingFrac = std::nan("");
        std::fprintf(stderr,
            "[load] source dir = %s  ->  P=%d  M=%d  (%zu doubles per matrix)\n",
            srcDir.c_str(), P, M, (size_t)P * M);
    } else {
        P           = std::atoi(argv[1]);
        M           = std::atoi(argv[2]);
        sigma       = std::atof(argv[3]);
        missingFrac = (argc >= 5) ? std::atof(argv[4]) : 0.0;
        seed        = (argc >= 6) ? std::strtoull(argv[5], nullptr, 10) : 12345ULL;
        if (P <= 1 || M <= 0) {
            std::fprintf(stderr, "P must be >1 and M must be >0\n");
            return EXIT_FAILURE;
        }
    }

    // ---- Memory budget (defect (4)): confirm no P x M x P materialization ---
    // Only [P x M], [2P x M], [P x P], [2P x P] buffers are ever allocated; the
    // GEMMs reduce straight to the P x P / 2P x P outputs. Largest device object
    // is S = [2P x M] doubles. For P=100, M=1e6: S=1.6 GB, Q=V=0.8 GB each,
    // outputs negligible -> ~4 GB, well within 32 GB. Host reference is O(P^2*M)
    // time, O(M) extra space (two long-double scratch vectors).
    {
        const long double bytesPM  = (long double)sizeof(double) * P * (long double)M;
        const long double bytes2PM = (long double)sizeof(double) * (2.0L * P) * (long double)M;
        const long double devTotal = 2.0L * bytesPM + bytes2PM        // dQ,dV,dS
                                    + 2.0L * sizeof(double) * (long double)P * P  // dG,dVpair
                                    + (long double)sizeof(double) * (2.0L * P) * P // dR
                                    + (long double)sizeof(double) * (long double)P * P // dF2
                                    + 64.0L * 1024.0L * 1024.0L;       // workspace
        std::fprintf(stderr,
            "[mem] device working set ~= %.2f GB (largest single buffer S=[2P x M]=%.2f GB); "
            "host long-double reference is O(P^2*M) time, O(M) scratch.\n",
            (double)(devTotal / (1024.0L * 1024.0L * 1024.0L)),
            (double)(bytes2PM / (1024.0L * 1024.0L * 1024.0L)));
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess &&
            (long double)freeB < devTotal) {
            std::fprintf(stderr,
                "[mem] WARNING: estimated working set exceeds free device memory "
                "(%.2f GB free); allocation may fail.\n",
                (double)freeB / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // ---- Build the GEMM-input host arrays -----------------------------------
    // SYNTHETIC: generate raw freqs/validity/N then run the SAME reconstruction
    //            the load path uses, so both modes share one recipe.
    // LOAD     : Q_raw/V_raw/N_raw already populated from disk.
    std::vector<double> Q, V, Hc, S;        // [P x M], [P x M], [P x M], [2P x M]

    if (!loadMode) {
        const double N        = 100.0;            // haploid sample size
        const double Ncorr    = std::max(N - 1.0, 1.0);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> ancU(0.05, 0.95);
        std::normal_distribution<double>       gauss(0.0, 1.0);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const double lo = 1e-6, hi = 1.0 - 1e-6;

        Q_raw.assign((size_t)P * M, 0.0);
        V_raw.assign((size_t)P * M, 0.0);
        N_raw.assign((size_t)P * M, 0.0);

        for (int s = 0; s < M; ++s) {
            double a = ancU(rng);                 // ancestral allele freq, this SNP
            for (int i = 0; i < P; ++i) {
                double z = gauss(rng);
                double p = a + sigma * z;
                if (p < lo) p = lo;
                if (p > hi) p = hi;

                double valid = 1.0;
                if (missingFrac > 0.0 && u01(rng) < missingFrac) valid = 0.0;

                const size_t idx = cm((size_t)i, (size_t)s, (size_t)P);
                Q_raw[idx] = p;                   // unmasked raw freq (masked in builder)
                V_raw[idx] = valid;
                N_raw[idx] = (valid != 0.0) ? N : 0.0;
                (void)Ncorr;
            }
        }
    }

    // One reconstruction recipe for BOTH modes (Q_masked, Hc, Qsq, S).
    build_host_arrays(P, M, Q_raw, V_raw, N_raw, Q, V, Hc, S);

    // ---- Device allocations -------------------------------------------------
    double *dQ = nullptr, *dV = nullptr, *dS = nullptr;
    double *dG = nullptr, *dVpair = nullptr, *dR = nullptr, *dF2 = nullptr;
    const size_t szPM  = sizeof(double) * (size_t)P * M;
    const size_t sz2PM = sizeof(double) * (size_t)(2 * P) * M;
    const size_t szPP  = sizeof(double) * (size_t)P * P;
    const size_t sz2PP = sizeof(double) * (size_t)(2 * P) * P;

    CUDA_CHECK(cudaMalloc(&dQ,     szPM));
    CUDA_CHECK(cudaMalloc(&dV,     szPM));
    CUDA_CHECK(cudaMalloc(&dS,     sz2PM));
    CUDA_CHECK(cudaMalloc(&dG,     szPP));
    CUDA_CHECK(cudaMalloc(&dVpair, szPP));
    CUDA_CHECK(cudaMalloc(&dR,     sz2PP));
    CUDA_CHECK(cudaMalloc(&dF2,    szPP));

    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), szPM,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), szPM,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS, S.data(), sz2PM, cudaMemcpyHostToDevice));

    // Explicit workspace -> required for run-to-run reproducibility of emulated
    // FP64 (architecture.md section 12). 64 MiB is ample for these tiny GEMMs.
    const size_t workspaceSize = (size_t)64 * 1024 * 1024;
    void* dWorkspace = nullptr;
    CUDA_CHECK(cudaMalloc(&dWorkspace, workspaceSize));

#if STEPPE_HAVE_EMU_TUNING
    // Device int for the emulation verification hook: cuBLAS writes the retained
    // fixed-point mantissa bit count here (>=0 if emulation engaged, -1 if not).
    int* dMantissaBits = nullptr;
    CUDA_CHECK(cudaMalloc(&dMantissaBits, sizeof(int)));
    int initMb = -2;  // sentinel distinct from cuBLAS's -1 "not used"
    CUDA_CHECK(cudaMemcpy(dMantissaBits, &initMb, sizeof(int), cudaMemcpyHostToDevice));
#endif

    // ---- (1) NATIVE FP64 (CUBLAS_COMPUTE_64F) -------------------------------
    cublasHandle_t hNative;
    CUBLAS_CHECK(cublasCreate(&hNative));
    CUBLAS_CHECK(cublasSetWorkspace(hNative, dWorkspace, workspaceSize));
    CUBLAS_CHECK(cublasSetMathMode(hNative, CUBLAS_PEDANTIC_MATH)); // strict native FP64

    std::vector<double> f2_native;
    float t_native_ms = 0.0f;
    {   // warmup (amortizes lazy module load); not timed
        std::vector<double> tmp; float t;
        run_f2_gemms(hNative, CUBLAS_COMPUTE_64F, P, M,
                     dQ, dV, dS, dG, dVpair, dR, dF2, tmp, &t);
    }
    run_f2_gemms(hNative, CUBLAS_COMPUTE_64F, P, M,
                 dQ, dV, dS, dG, dVpair, dR, dF2, f2_native, &t_native_ms);

    // ---- (2) EMULATED FP64 (Ozaki fixed-point) ------------------------------
    // Load-bearing mechanisms (architecture.md line 89): the emulated compute type
    // CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT and the math mode
    // CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH. These are used unconditionally.
    //
    // The following tuning calls are compiled under -DSTEPPE_HAVE_EMU_TUNING. Their
    // symbol names/signatures are CONFIRMED working in standard cuBLAS 13 (verified
    // against the live CUDA 13 cublas_api.h on the sm_120 box; the f2 kernel builds
    // clean and the emulation arm engages there):
    //   - cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER)
    //       Forces emulation even for tiny GEMMs, so the experiment does not
    //       silently measure native FP64 under a "performant/default" strategy
    //       that declines to emulate.
    //   - cublasSetFixedPointEmulationMantissaControl(h,
    //         CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC)
    //       Selects DYNAMIC slice count for THIS measurement cell only -- this is
    //       the arm that DEMONSTRATES the dynamic TRAP (overshoots to ~60 bits on
    //       real data -> parity with native, no win). Production uses FIXED control
    //       (see f2_timing.cu / f2_prec_acc.cu fixed-slice arms; the production
    //       kernel never selects dynamic -- architecture.md section 12, ROADMAP 0).
    //   - cublasSetFixedPointEmulationMaxMantissaBitCount(h, 0)
    //       Bit cap (0 = let the dynamic control choose, for the dynamic arm).
    //   - cublasSetFixedPointEmulationMantissaBitCountPointer(h, dMantissaBits)
    //       Verification hook; >=0 confirms emulation engaged, -1 = not used.
    // Independently of the tuning hooks, the BIT-IDENTITY check below (emu vs
    // native) is the header-independent proof that emulation engaged.
    cublasHandle_t hEmu;
    CUBLAS_CHECK(cublasCreate(&hEmu));
    CUBLAS_CHECK(cublasSetWorkspace(hEmu, dWorkspace, workspaceSize));
    CUBLAS_CHECK(cublasSetMathMode(hEmu, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
#if STEPPE_HAVE_EMU_TUNING
    CUBLAS_CHECK(cublasSetEmulationStrategy(hEmu, CUBLAS_EMULATION_STRATEGY_EAGER));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(
                     hEmu, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(hEmu, 0));
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaBitCountPointer(hEmu, dMantissaBits));
#endif

    std::vector<double> f2_emu;
    float t_emu_ms = 0.0f;
    {   // warmup; not timed
        std::vector<double> tmp; float t;
        run_f2_gemms(hEmu, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, P, M,
                     dQ, dV, dS, dG, dVpair, dR, dF2, tmp, &t);
    }
    run_f2_gemms(hEmu, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, P, M,
                 dQ, dV, dS, dG, dVpair, dR, dF2, f2_emu, &t_emu_ms);

    // Run-to-run determinism check (same workspace): emulated FP64 should be
    // bit-identical across repeats when an explicit workspace is set.
    std::vector<double> f2_emu2; float t_tmp = 0.0f;
    run_f2_gemms(hEmu, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, P, M,
                 dQ, dV, dS, dG, dVpair, dR, dF2, f2_emu2, &t_tmp);
    bool deterministic = (f2_emu.size() == f2_emu2.size()) &&
        std::equal(f2_emu.begin(), f2_emu.end(), f2_emu2.begin());

    // ---- DEFECT (2) CORE CHECK: did emulation actually engage? --------------
    // Emulated FP64 is NOT bit-identical to native FP64 (architecture.md
    // line 614/738). So bit-for-bit equality of the two arms on a non-trivial
    // input is strong evidence emulation was IGNORED / silently fell back to
    // native. This proof is header-INDEPENDENT and is the real fix for the
    // "fail loudly if emulation didn't engage" requirement.
    bool emuEqNative = (f2_emu.size() == f2_native.size()) &&
        std::memcmp(f2_emu.data(), f2_native.data(),
                    f2_emu.size() * sizeof(double)) == 0;

    // The mantissa-bit hook, when available, positively confirms engagement.
    int  mantissaBits     = -2;   // -2 = hook unavailable / not read
    bool hookConfirmedEmu = false;
#if STEPPE_HAVE_EMU_TUNING
    CUDA_CHECK(cudaMemcpy(&mantissaBits, dMantissaBits, sizeof(int),
                          cudaMemcpyDeviceToHost));
    hookConfirmedEmu = (mantissaBits >= 0);
#endif

    // Emulation is considered engaged iff EITHER the hook positively confirms it
    // OR (hook unavailable) the emu output differs from native bit-for-bit.
    // If emu == native AND the hook did not confirm, we treat emulation as NOT
    // engaged and fail loudly -- a numeric "PASS" here would be meaningless.
    const bool emulation_engaged = hookConfirmedEmu || !emuEqNative;

    // ---- (3) REFERENCE (CPU long double, cancellation-free) -----------------
    std::vector<double> f2_ref;
    double kappaMax = 0.0;
    reference_f2(P, M, Q, V, Hc, f2_ref, kappaMax);

    // ---- Accuracy stats vs reference ----------------------------------------
    ErrStats esNative = rel_error_stats(f2_native, f2_ref, P);
    ErrStats esEmu    = rel_error_stats(f2_emu,    f2_ref, P);
    const char* v     = verdict(esEmu, esNative, emulation_engaged);

    // ---- Print results-table row --------------------------------------------
    // In --load mode sigma/miss are not meaningful (the dispersion/missingness
    // come from real genotypes), so they print as NaN and a "src" column carries
    // the source dir basename (or "real"); the synthetic path prints "synth".
    auto basename_of = [](const std::string& p) -> std::string {
        if (p.empty()) return std::string("synth");
        size_t pos = p.find_last_of('/');
        std::string b = (pos == std::string::npos) ? p : p.substr(pos + 1);
        if (b.empty()) b = "real";
        if (b.size() > 12) b = b.substr(b.size() - 12);   // keep column width
        return b;
    };
    std::string srcLabel = loadMode ? basename_of(srcDir) : std::string("synth");

    static bool header_printed = false;
    if (!header_printed) {
        std::printf("%12s %5s %8s %9s %7s %11s %10s "
                    "%12s %12s %12s %12s %10s %10s %5s %4s %7s %12s\n",
                    "src", "P", "M", "sigma", "miss",
                    "mean|f2|", "kappaMax",
                    "maxRel_nat", "medRel_nat", "maxRel_emu", "medRel_emu",
                    "t_nat_ms", "t_emu_ms", "mBits", "det", "emuEqNat", "verdict");
        header_printed = true;
    }
    std::printf("%12s %5d %8d %9.1e %7.2f %11.3e %10.2e "
                "%12.3e %12.3e %12.3e %12.3e %10.4f %10.4f %5d %4s %7s %12s\n",
                srcLabel.c_str(), P, M, sigma, missingFrac,
                esNative.meanAbsF2, kappaMax,
                esNative.maxRel, esNative.medRel,
                esEmu.maxRel,    esEmu.medRel,
                t_native_ms, t_emu_ms,
                mantissaBits, deterministic ? "yes" : "NO",
                emuEqNative ? "YES" : "no", v);

    if (loadMode) {
        std::fprintf(stderr, "[load] dir=%s P=%d M=%d (sigma/miss are 'real', "
                     "printed as NaN)\n", srcDir.c_str(), P, M);
    }

    // ---- Loud warnings (defect (2): fail loudly, never silently) ------------
    if (emuEqNative) {
        std::fprintf(stderr,
            "  [warn] EMULATED f2 is BIT-IDENTICAL to NATIVE f2 -> emulation was "
            "almost certainly IGNORED / silently fell back to native FP64 "
            "(Ozaki output is not bit-identical to native; architecture.md "
            "line 614/738). The emu column does NOT reflect emulation; verdict "
            "forced to FAIL_IGNORED.%s\n",
            STEPPE_HAVE_EMU_TUNING ? "" :
            " (Built without -DSTEPPE_HAVE_EMU_TUNING: the EAGER strategy that "
            "forces emulation of tiny GEMMs is not active -- the library may have "
            "declined to emulate. Confirm the tuning symbol names and rebuild "
            "with -DSTEPPE_HAVE_EMU_TUNING=1.)");
    }
#if STEPPE_HAVE_EMU_TUNING
    if (mantissaBits < 0) {
        std::fprintf(stderr,
            "  [warn] mantissaBitCount=%d -> the verification hook reports "
            "emulation did NOT engage for this cell; emu column reflects native "
            "FP64, not Ozaki.\n", mantissaBits);
    }
#endif
    if (!deterministic) {
        std::fprintf(stderr,
            "  [warn] emulated FP64 was NOT run-to-run deterministic across "
            "repeats with a fixed workspace -- investigate before trusting any "
            "bit-stability claim (architecture.md section 12).\n");
    }

    // ---- Cleanup ------------------------------------------------------------
    CUBLAS_CHECK(cublasDestroy(hNative));
    CUBLAS_CHECK(cublasDestroy(hEmu));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dG));
    CUDA_CHECK(cudaFree(dVpair));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dF2));
    CUDA_CHECK(cudaFree(dWorkspace));
#if STEPPE_HAVE_EMU_TUNING
    CUDA_CHECK(cudaFree(dMantissaBits));
#endif

    // Non-zero exit on a genuine FAIL so a sweep harness / CI can detect it.
    if (std::strcmp(v, "FAIL") == 0 || std::strcmp(v, "FAIL_IGNORED") == 0)
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
