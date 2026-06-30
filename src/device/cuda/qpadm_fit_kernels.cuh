// src/device/cuda/qpadm_fit_kernels.cuh
//
// Narrow launch-wrapper declarations for the qpAdm fit (M(fit-4)) device kernels
// (the FROZEN CONTRACT §2). Host orchestration (cuda_backend.cu) calls these
// `void launch_*` functions; the kernel bodies + `<<<>>>` live only in the .cu —
// the architecture.md §7 "host code never includes kernel bodies or <<<>>>" rule,
// mirroring f2_batched_kernel.cuh's style. All math here is NATIVE FP64: the f4
// 4-slab combine is the cancellation-sensitive f-stat difference, and the
// jackknife/xtau reductions reproduce the CpuBackend long-double oracle's operation
// order in FP64 (at nb=708, m=10 this matches the golden to the gate tier; §12).
//
// This header names CUDA types (cudaStream_t) and so is PRIVATE to steppe_device
// (architecture.md §4) — the device-internal seam between the backend and the
// kernels, NOT the CUDA-free public ComputeBackend seam.
#ifndef STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH
#define STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH

#include <cuda_runtime.h>

namespace steppe::device {

/// S3 f4-gather (the FROZEN CONTRACT §2a; CpuBackend assemble_f4, src/device/cpu/cpu_backend.cpp). Build
/// the per-block f4 matrix X[k + m*b] for k = j + nr*i (ROW-MAJOR vectorization),
/// batched over ALL nb blocks in one launch (grid over (k, b)). Reads the RESIDENT
/// f2 tensor `d_f2` (column-major i + P*j + P*P*b, VRAM) directly — NO D2H. The
/// 4-slab combine
///   X[j+nr*i, b] = 0.5*( f2(Li,R0,b) + f2(L0,Rj,b) - f2(L0,R0,b) - f2(Li,Rj,b) )
/// with Li = d_left[i+1], Rj = d_right[j+1], L0 = d_left[0], R0 = d_right[0]. Native
/// FP64 (the cancellation-sensitive f-stat difference). `d_left`/`d_right` are the
/// small H2D'd index buffers (length nl+1 / nr+1). Output dX[m*nb], m = nl*nr.
/// F1 / OQ-12: `nb` is the SURVIVOR block count and `d_surv` (length nb, ASCENDING)
/// maps a compacted survivor index to its ORIGINAL resident block id (so the gather
/// reads the resident f2 at the original block, writing the dense compacted dX). A
/// MISSING block (Vpair==0 for any pair; AT2 read_f2(remove_na=TRUE)) is excluded from
/// d_surv. `d_surv==nullptr` ⇒ identity (no drop — the maxmiss=0 no-missing-block path,
/// bit-identical to the pre-F1 gather). The keep-mask is built by launch_f2_block_keep.
void launch_assemble_f4_gather(const double* d_f2, int P,
                               const int* d_left, const int* d_right,
                               int nl, int nr, int nb, const int* d_surv,
                               double* dX, cudaStream_t stream);

/// STANDALONE f4 quartet-gather (the run_f4 seam; fit-engine §6). Build the per-block f4
/// matrix dX[k + N*b] whose m axis is the N QUARTETS — each quartet k has its OWN
/// (p1,p2,p3,p4), unlike launch_assemble_f4_gather whose single (L0,R0) spans an nl×nr
/// grid. `d_quartets` is the H2D'd flattened index quad array (length 4*N, quad k at
/// [4*k .. 4*k+3] = {p1,p2,p3,p4}). Per (k, b):
///   dX[k + N*b] = 0.5*( f2(p2,p3,b) + f2(p1,p4,b) - f2(p1,p3,b) - f2(p2,p4,b) )
/// the SAME four-slab AT2 identity specialized to nl=1,nr=1 — ZERO new math, just a
/// per-column quad instead of the shared (L0,R0). Reads the RESIDENT f2 (column-major
/// i + P*j + P*P*b) — NO D2H. Native FP64 (the cancellation carve-out). F1/OQ-12 SURVIVOR
/// COMPACTION: `nb` is the SURVIVOR block count and `d_surv` (length nb, ASCENDING) maps a
/// compacted survivor index to its ORIGINAL resident block id; d_surv==nullptr ⇒ identity.
void launch_assemble_f4_quartets_gather(const double* d_f2, int P,
                                        const int* d_quartets, int N, int nb,
                                        const int* d_surv,
                                        double* dX, cudaStream_t stream);

/// STANDALONE f3 triple-gather (the run_f3 seam; fit-engine §6) — the THREE-slab clone of
/// launch_assemble_f4_quartets_gather. Build the per-block f3 matrix dX[k + N*b] whose m
/// axis is the N TRIPLES — each triple k has its OWN (C,A,B). `d_triples` is the H2D'd
/// flattened index triple array (length 3*N, triple k at [3*k .. 3*k+2] = {p1=C,p2=A,p3=B}).
/// Per (k, b):
///   dX[k + N*b] = 0.5*( f2(C,A,b) + f2(C,B,b) - f2(A,B,b) )
/// the THREE-slab AT2 f3 identity — the ONE new math seam f3 adds over f4. Reads the
/// RESIDENT f2 (column-major i + P*j + P*P*b) — NO D2H. Native FP64 (the cancellation
/// carve-out). F1/OQ-12 SURVIVOR COMPACTION: `nb` is the SURVIVOR block count and `d_surv`
/// (length nb, ASCENDING) maps a compacted survivor index to its ORIGINAL resident block
/// id; d_surv==nullptr ⇒ identity.
void launch_assemble_f3_triples_gather(const double* d_f2, int P,
                                       const int* d_triples, int N, int nb,
                                       const int* d_surv,
                                       double* dX, cudaStream_t stream);

/// F1 / OQ-12 keep-mask: one thread per resident block writes d_keep[b]=0 iff block b
/// is PARTIALLY covered (≥1 pair Vpair==0 AND ≥1 pair Vpair>0 — AT2 read_f2's
/// `!is.finite` drop), else 1. A fully-zero slab is the "no Vpair info" sentinel (the
/// legacy/parity zero-fill), NOT a missing block ⇒ kept. `d_vpair` is the RESIDENT
/// [P×P×nb] Vpair tensor; `d_keep` is length nb. The host reads d_keep down and builds
/// the ASCENDING survivor id list. Mirrors the CpuBackend oracle survivor_blocks and
/// shares the single-source predicate core::pair_block_is_missing.
void launch_f2_block_keep(const double* d_vpair, int P, int nb, int* d_keep,
                          cudaStream_t stream);

/// S3 est_to_loo + x_total + tot_line (the FROZEN CONTRACT §2a; CpuBackend
/// compute_loo_and_total, src/device/cpu/cpu_backend.cpp). One thread per k (m = nl*nr
/// small) reduces over the nb blocks, reproducing the CpuBackend's operation order
/// in FP64 (the long-double accumulators become FP64; at nb=708 this matches the
/// golden to rtol 1e-6, the gate tier). For each k:
///   tot_ij    = (Σ_b X[k,b]*bl_b) / n
///   loo[k,b]  = (tot_ij - X[k,b]*rel_b) / (1 - rel_b),  rel_b = bl_b/n
///   tot_line  = (Σ_b loo[k,b]*(1-bl_b/n)) / (Σ_b (1-bl_b/n))
///   est[k]    = (Σ_b (tot_line - loo[k,b])) + (Σ_b loo[k,b]*bl_b)/n
/// (the CpuBackend's `mean(...)*nb` simplifies to the bare sum; reproduced exactly).
/// `d_block_sizes` is the H2D'd per-block SNP count (length nb), `n` = Σ bl.
/// Outputs dLoo[m*nb], dTotal[m], dTotLine[m].
void launch_f4_loo_total(const double* dX, const int* d_block_sizes,
                         int m, int nb, double n,
                         double* dLoo, double* dTotal, double* dTotLine,
                         cudaStream_t stream);

/// S4 xtau pseudo-values (the FROZEN CONTRACT §2b; CpuBackend xtau, src/device/cpu/cpu_backend.cpp).
/// One thread per (k, b): h = n/bl_b, sh = sqrt(h-1),
///   xtau[k,b] = (est[k]*h - loo[k,b]*(h-1) - tot_line[k]) / sh
/// laid out COLUMN-MAJOR (k + m*b) so cublasDsyrk(OP_N, lda=m, k=nb) forms
/// Q = xtau·xtauᵀ. Native FP64. dEst = dTotal (length m), dTotLine length m,
/// dLoo[m*nb]. Output dXtau[m*nb].
void launch_f4_xtau(const double* dLoo, const double* dEst, const double* dTotLine,
                    const int* d_block_sizes, int m, int nb, double n,
                    double* dXtau, cudaStream_t stream);

/// S4 DIAGONAL-only jackknife variance (the per-item f-stat SE production shape; the OOM
/// fix for the sweep). One thread per item k: var[k] = (1/nb)·Σ_b dXtau[k + m*b]² — EXACTLY
/// the diagonal Q[k+m*k] = (xtau·xtauᵀ/nb)[k,k] that f4/f3 read, WITHOUT forming the dense
/// m×m Q, the cublasDsyrk, the fudge diag, or the potrf/potri (O(m·nb) work, O(m) memory —
/// no N²/OOM). dXtau is the SAME column-major (k + m*b) buffer launch_f4_xtau produces (native
/// carve-out), so var[k] re-passes the existing FP64 goldens BY CONSTRUCTION. Native FP64.
/// Mined from wip/fstats-massive-overbuild.
void launch_f4_diag_var(const double* dXtau, int m, int nb, double* dVar,
                        cudaStream_t stream);

// ---- GPU-ONLY SWEEP kernels (the fix for the CPU-bound host-enumeration disaster) ----

/// On-device combinatorial UNRANK of a CHUNK of QUARTETS: one thread per local item t in
/// [0,C); global colex rank = c0 + t; writes dQuartets[4*t..4*t+3] = (c0<c1<c2<c3), the EXACT
/// flat 4*C layout assemble_f4_quartets_gather reads as a DEVICE pointer. This REPLACES the
/// per-chunk H2D of a host-enumerated quartet list — NO host enumeration. Native int math.
void launch_sweep_unrank_quartets(long long c0, int C, int range, const int* d_subset,
                                  int* dQuartets, cudaStream_t stream);

/// On-device UNRANK of a CHUNK of TRIPLES (k=3) — the f3 sibling of the quartet unrank.
void launch_sweep_unrank_triples(long long c0, int C, int range, const int* d_subset,
                                 int* dTriples, cudaStream_t stream);

/// On-device |z| FILTER for the sweep: one thread per item k computes est=dXtotal[k],
/// se=sqrt(dVar[k]), z=est/se (written to dEst/dSe/dZ) and the survivor flag
/// d_flags[k] = (mode==0 ? (|z|>=min_z) : 1) as uint8 0/1. mode 0 = MinZ filter on device;
/// mode 1 = keep-all (TopK/All — selection ranked host-side on the compacted set). NaN z
/// (degenerate var) flags 0 under MinZ (NaN compares false). Native FP64.
void launch_sweep_zfilter(const double* dXtotal, const double* dVar, int C, int mode,
                          double min_z, double* dEst, double* dSe, double* dZ,
                          unsigned char* d_flags, cudaStream_t stream);

/// Deinterleave the flat k-per-item device key array d_items[k*t+c] into k SEPARATE contiguous
/// int columns d_c0..d_c3[t] so each can be CUB-stream-compacted with the SAME survivor flags
/// (CUB Flagged takes one input iterator per call). k<=4; for k=3 the 4th column is set 0.
void launch_sweep_deinterleave_keys(const int* d_items, int C, int k,
                                    int* d_c0, int* d_c1, int* d_c2, int* d_c3,
                                    cudaStream_t stream);

// ---- BOUNDED DEVICE TOP-K reservoir (the fix for the unbounded-host-vector sweep OOM) ----

/// |z| FILTER with a DEVICE-RESIDENT rising threshold: like launch_sweep_zfilter but the cut is
/// read from d_tau[0] (the live top-K threshold, raised on each compact) instead of a host
/// constant, and the |z| sort key is written into dAbsZ. d_flags[t] = (|z| > d_tau[0]). NaN z
/// (degenerate var) flags 0 (never enters the top-K). Native FP64.
void launch_sweep_zfilter_tau(const double* dXtotal, const double* dVar, int C,
                              const double* d_tau, double* dEst, double* dSe, double* dZ,
                              double* dAbsZ, unsigned char* d_flags, cudaStream_t stream);

/// Fill d_idx[0..n) = 0..n-1 (the permutation VALUE array for CUB SortPairsDescending).
void launch_sweep_topk_iota(int* d_idx, int n, cudaStream_t stream);

/// GATHER reservoir rows by a permutation: out[r] = in[d_perm[r]] for r in [0,m), across all
/// columns (est/se/z/absz + 4 key cols) with the SAME perm so the row tuple stays intact. Used
/// to reorder the reservoir into |z|-descending order (then truncate to K). Out-of-place.
void launch_sweep_topk_gather(const int* d_perm, int m,
                              const double* inEst, const double* inSe, const double* inZ,
                              const double* inAbsZ, const int* inC0, const int* inC1,
                              const int* inC2, const int* inC3,
                              double* outEst, double* outSe, double* outZ, double* outAbsZ,
                              int* outC0, int* outC1, int* outC2, int* outC3,
                              cudaStream_t stream);

/// RAISE the rising-tau threshold to the new K-th-largest |z|: d_tau[0] = max(d_tau[0],
/// d_sorted_absz[K-1]) when mode==1 (TopK) and K>0. Monotone (never lowers); a no-op in MinZ
/// mode (mode 0, tau pinned at the min_z floor). A single device thread.
void launch_sweep_topk_raise_tau(const double* d_sorted_absz, int K, int mode,
                                 double* d_tau, cudaStream_t stream);

/// Mirror the LOWER triangle of an n×n COLUMN-MAJOR matrix into the UPPER (so a
/// SYRK/potri result that fills one triangle becomes the full symmetric matrix the
/// CpuBackend writes both triangles of). In-place. One thread per (i,j), i>j writes
/// (j,i) = (i,j). Native FP64.
void launch_symmetrize_lower_to_full(double* dM, int n, cudaStream_t stream);

/// Add `fudge * tr` to the diagonal of an n×n COLUMN-MAJOR matrix (the FROZEN
/// CONTRACT §2b fudge step; CpuBackend cpu_backend.cpp:480-481). `dM[k + n*k] +=
/// fudge*tr`. `tr` is supplied by the host (computed via a trace reduction). One
/// thread per diagonal entry. Native FP64.
void launch_add_fudge_diag(double* dM, int n, double fudge, double tr,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------------
// Small-dense on-device LA for the rank-test / ALS / weight / chisq (S5/S6) — the
// FROZEN CONTRACT §2c/§2d. At the golden sizes everything is tiny (m=10, nl=2,
// nr=5, r=1, t≤5). These run as SINGLE-THREAD device kernels that TRANSLITERATE the
// CpuBackend's exact native-FP64 scalar operations (small_linalg.hpp jacobi_svd /
// lu_factor / solve; cpu_backend.cpp opt_A/opt_B/als_weights/chisq_of) — same ops,
// same order, on the GPU — so the GPU fit is bit-exact vs the FP64 oracle AND runs
// device-resident (the production path; the binding requirement). They consume only
// device buffers (dXmat from x_total, dQinv) and write device outputs (dW/dA/dB,
// dchisq). The constrained weight solve + chisq are folded into one kernel.
// ---------------------------------------------------------------------------------

/// S5 SVD seed (the FROZEN CONTRACT §2c; CpuBackend seed_AB, src/device/cpu/cpu_backend.cpp).
/// One-sided Jacobi SVD of the nl×nr COLUMN-MAJOR `dXmat` (transliterating
/// core::jacobi_svd), then B = t(V[:,0:r]) (r×nr), A = xmat·t(B) (nl×r). dA[nl*r],
/// dB[r*nr] written column-major. Single-thread kernel (nl,nr small). Native FP64.
void launch_qpadm_seed_ab(const double* dXmat, int nl, int nr, int r,
                          double* dA, double* dB, cudaStream_t stream);

/// L1: rank_Q ON-DEVICE — the numerical rank of the m×m COLUMN-MAJOR covariance Q
/// (model_well_determined.rank_Q, observability-only, the last bounded per-model
/// host-compute item). Runs the SAME one-sided-Jacobi sweep dev_jacobi_svd_V
/// transliterates (core::jacobi_svd) over VRAM scratch, counting the singular values
/// above tol = smax * m * eps (eps = std::numeric_limits<double>::epsilon(), passed so
/// the constant is bit-identical to the host). BIT-IDENTICAL to the CpuBackend oracle
/// rank_Q (the V-accumulation is dropped — sigma is independent of V — but W evolves
/// identically, so the count is exact). `dScratch` holds W[m*m] | sigma[m] (m*m+m
/// doubles); `dIntScratch` holds order[m] (m ints); `dRank` is the [1] int output.
/// Single thread, native FP64 (the §4 SVD carve-out, same as the host path it replaces).
void launch_qpadm_rank_via_jacobi(const double* dQ, int m, double eps,
                                  double* dScratch, int* dIntScratch, int* dRank,
                                  cudaStream_t stream);

/// S6 ALS opt_A then opt_B for `als_iters` iterations (the FROZEN CONTRACT §2d;
/// CpuBackend als_weights loop / opt_A / opt_B, src/device/cpu/cpu_backend.cpp).
/// Transliterates the Kronecker coeffs/rhs build + the LU solve
/// (core::solve) + the byrow reshape, in native FP64, single-thread. Seeds A,B in
/// place from the caller's dA/dB (filled by launch_qpadm_seed_ab) and overwrites
/// them with the refined factors. `dQinv` is the m×m (m=nl*nr) column-major inverse.
/// On a singular ALS system the corresponding factor is left zero (CpuBackend
/// opt_A/opt_B, src/device/cpu/cpu_backend.cpp). Native FP64.
void launch_qpadm_als(const double* dXmat, const double* dQinv,
                      int nl, int nr, int r, double fudge, int als_iters,
                      double* dA, double* dB, cudaStream_t stream);

/// S6 constrained weight solve + chisq (the FROZEN CONTRACT §2d/§2c; CpuBackend
/// als_weights / chisq_of, src/device/cpu/cpu_backend.cpp). From the refined dA
/// (nl×r) build RHS=crossprod(cbind(A,1)) (nl×nl), LHS=ones, LU-solve, normalize
/// Σw=1 → dW[nl]; and chisq = vec(E)'·Qinv·vec(E), E = xmat - A·B → dchisq[0].
/// `d_status` (length 1, int) is set to 0=Ok, 6=RankDeficient (the weight solve was
/// singular), matching CpuBackend als_weights, src/device/cpu/cpu_backend.cpp. Single-thread, native FP64.
/// For r==0 the trivial path (w=ones, chisq=chisq_of with empty A,B) is taken.
void launch_qpadm_weights_chisq(const double* dXmat, const double* dQinv,
                                const double* dA, const double* dB,
                                int nl, int nr, int r,
                                double* dW, double* dchisq, int* d_status,
                                cudaStream_t stream);

/// Build the nl×nr COLUMN-MAJOR xmat from the ROW-MAJOR x_total / loo slice
/// (k = j + nr*i ⇒ xmat(i,j) at i + nl*j; CpuBackend xmat_from_total
/// xmat_from_total, src/device/cpu/cpu_backend.cpp). For the full-data fit `dTotalSrc` = dTotal (one slice);
/// for batched S7 a per-block slice is passed. Single-thread kernel. Native FP64.
void launch_qpadm_xmat_from_rowmajor(const double* dTotalSrc, int nl, int nr,
                                     double* dXmat, cudaStream_t stream);

// ---------------------------------------------------------------------------------
// S7 BATCHED leave-one-block-out re-fits (the FROZEN CONTRACT §2e). The whole
// per-block fit (xmat from x_loo[:,:,b] → seed → 20 ALS iters → weight solve →
// normalize) is run for ALL nb blocks in ONE launch (one thread/block-of-threads
// per replicate), transliterating the same CpuBackend ops as the single fit. This
// replaces the 708 host gls_weights calls with a single batched device kernel
// (NOT a host loop; the binding requirement). Output dWmat[nb*nl] ROW-MAJOR
// (b*nl + i, the AT2 replicate matrix). Native FP64.
void launch_qpadm_loo_batched(const double* dLoo, const double* dQinv,
                              int nl, int nr, int r, double fudge, int als_iters,
                              int nb, double* dWmat, cudaStream_t stream);

// ---------------------------------------------------------------------------------
// LARGE-path kernels (the FROZEN CONTRACT §1/§2): models exceeding the bit-parity
// small envelope (nl>5 || nr>10 || r>4, e.g. NRBIG nr=39). The per-model working set
// moves OUT of per-thread LOCAL memory (which CUDA reserves for the device's max
// resident-thread count ⇒ OOM at launch for big nr) into caller-provided VRAM
// scratch (`dScratch`/`dIntScratch`). The SVD seed is NOT a kernel here — it is the
// cuSOLVER `large_svd_V` on the host (cuda_backend.cu) followed by
// `launch_qpadm_seed_from_V` (the cheap GEMM-shaped part of the seed). Same math /
// same op order as the small templated kernels; native FP64. Single-model-per-launch
// (grid=1) for THIS milestone; the `dScratch` offset layout is the S8 batching seam.
// ---------------------------------------------------------------------------------

/// Transpose the nl×nr COLUMN-MAJOR `dXmat` into the nr×nl COLUMN-MAJOR `dXt`
/// (dXt[j + nr*i] = dXmat[i + nl*j]). One thread per element. Used to orient the
/// matrix handed to cuSOLVER gesvd as rows>=cols (the §1.3 determinism rule): for the
/// common large case nr>=nl, Xt is nr×nl (rows>=cols), and U(Xt) == V(Xmat). Native FP64.
void launch_transpose_small(const double* dXmat, int nl, int nr,
                            double* dXt, cudaStream_t stream);

/// LARGE-path seed from the cuSOLVER right singular vectors V[:,0:r] (nr×r col-major
/// in `dVout`, descending). Only the cheap GEMM-shaped part of `dev_seed_ab` AFTER the
/// SVD: B[p,j] = V[j,p] (r×nr), A = xmat·t(B) (nl×r). No Vfull/W local arrays. Single
/// thread (one model). Native FP64. Writes dA[nl*r], dB[r*nr] column-major.
void launch_qpadm_seed_from_V(const double* dXmat, const double* dVout,
                              int nl, int nr, int r,
                              double* dA, double* dB, cudaStream_t stream);

/// LARGE-path ALS opt_A/opt_B loop — same math as launch_qpadm_als but the per-model
/// working set (xvec[m], Wm[m*t], coeffs[t*t], rhs[t], lu[t*t], y[t], piv[t], A2/B2[t])
/// lives in caller-provided VRAM scratch (dScratch / dIntScratch), so arbitrary
/// nl/nr/m/r fit. Single-thread kernel (one model). Native FP64. Seeds in dA/dB
/// (filled by launch_qpadm_seed_from_V); overwrites them with the refined factors. On
/// a singular ALS system the corresponding factor is left zero (CpuBackend parity).
void launch_qpadm_als_large(const double* dXmat, const double* dQinv,
                            int nl, int nr, int r, double fudge, int als_iters,
                            double* dA, double* dB,
                            double* dScratch, int* dIntScratch, cudaStream_t stream);

/// LARGE-path weight solve + chisq — same math as launch_qpadm_weights_chisq with
/// VRAM scratch (RHS[nl*nl], LHS[nl], wv[nl], lu[nl*nl], y[nl], piv[nl], e[m]). Sets
/// d_status 0=Ok / 6=RankDeficient. For r==0 the trivial path (w=ones, chisq with empty
/// A,B). Single-thread, native FP64.
void launch_qpadm_weights_chisq_large(const double* dXmat, const double* dQinv,
                                      const double* dA, const double* dB,
                                      int nl, int nr, int r,
                                      double* dW, double* dchisq, int* d_status,
                                      double* dScratch, int* dIntScratch,
                                      cudaStream_t stream);

/// LARGE-path PARALLEL LOO re-fits — the throughput-scaled large-model jackknife SE.
/// One thread per (model, block) runs the SAME als_large + weights_chisq_large math as
/// the serial large path, seeded from a per-(model,block) cuSOLVER SVD slice (dAseed/
/// dBseed, precomputed host-side so the seed is BIT-IDENTICAL), using a per-thread slice
/// of a runtime-sized VRAM arena (stride dbl_refit doubles / int_refit ints per refit;
/// layout xmat[m]|A[nl*r]|B[r*nr]|union[large_dbl_scratch]). Writes the UNSCALED
/// normalized weights to dWmat[(model*nb + b)*nl + i] (status!=0 ⇒ zeros). Replaces the
/// host nb-serial refit loop with nb (and the future B·nb) concurrent refits ⇒ the SE is
/// bit-identical (only the parallelism changes), the host long-double variance reduction
/// in se_from_loo is unchanged. Native FP64.
void launch_qpadm_loo_large_batched(const double* dLoo, const double* dQinv,
                                    const double* dAseed, const double* dBseed,
                                    int nl, int nr, int r, double fudge, int als_iters,
                                    int nb, int n_models, long dbl_refit, long int_refit,
                                    double* dScratch, int* dIntScratch, double* dWmat,
                                    cudaStream_t stream);

// ---------------------------------------------------------------------------------
// M(fit-6) S8 MODEL-BATCHED kernels (the ROTATION primitive — the FROZEN CONTRACT
// §2.2). The single-model launchers above are LIFTED to a MODEL-batch axis: each
// kernel adds a `model` grid dimension and reads/writes a per-model SLICE of a
// strided arena. ONE batched dispatch fits a whole BUCKET of B same-shape
// (nl,nr,r) models — NOT a per-model host loop. Same math, same op order, native
// FP64 (the gate carve-out); the model axis is the only addition. These serve the
// SMALL-path bit-parity bucket (nl<=5, nr<=10, r<=4 — the rotation common case);
// the >32 / large tail still runs the per-model device path (one dispatch/model is
// correct for the tail, design §5). The covariance Q (strided-batched GEMM) and the
// Qinv (cuSOLVER potrfBatched/potrsBatched) live in cuda_backend.cu — these kernels
// are the element-wise / small-reduction / single-thread-per-model steps.
// ---------------------------------------------------------------------------------

/// S3 f4-gather, MODEL-BATCHED. Grid over (k + m*b, model). Reads the RESIDENT f2
/// tensor `d_f2` (col-major i + P*j + P*P*b) with PER-MODEL index arenas
/// `d_left_arena` ([B][nl+1] row-major) / `d_right_arena` ([B][nr+1]) and writes the
/// strided arena `dX` (per-model slice dX + model*(m*nb), layout k + m*bs). Native FP64.
/// F1 / OQ-12: `nb` is the SURVIVOR block count; `d_surv` (length nb, ASCENDING, SHARED
/// across models) maps the compacted survivor index to the original resident block id
/// (a missing block — Vpair==0 for any pair — is dropped). `d_surv==nullptr` ⇒ identity.
void launch_assemble_f4_gather_models_batched(const double* d_f2, int P,
                                              const int* d_left_arena,
                                              const int* d_right_arena,
                                              int nl, int nr, int nb, int n_models,
                                              const int* d_surv,
                                              double* dX, cudaStream_t stream);

/// S3 est_to_loo + x_total + tot_line, MODEL-BATCHED. One thread per (k, model); reduces
/// over nb blocks of THIS model's dX slice (dX + model*(m*nb)) reproducing the FP64 op
/// order. Writes per-model slices dLoo[m*nb], dTotal[m], dTotLine[m]. Native FP64.
void launch_f4_loo_total_models_batched(const double* dX, const int* d_block_sizes,
                                        int m, int nb, double n, int n_models,
                                        double* dLoo, double* dTotal, double* dTotLine,
                                        cudaStream_t stream);

/// S4 xtau pseudo-values, MODEL-BATCHED. One thread per (k + m*b, model). Reads this
/// model's dLoo/dEst(=dTotal)/dTotLine slices and writes dXtau slice (col-major k+m*b
/// per model, the layout the strided-batched SYRK/GEMM consumes). Native FP64.
void launch_f4_xtau_models_batched(const double* dLoo, const double* dEst,
                                   const double* dTotLine, const int* d_block_sizes,
                                   int m, int nb, double n, int n_models,
                                   double* dXtau, cudaStream_t stream);

/// Per-model fudge-diag, MODEL-BATCHED. For each model: tr = trace(Q_model) (computed
/// in-thread), then Qf_model.diag += fudge*tr. dQ is the input strided arena (m*m per
/// model, col-major), dQf the output copy. One block per model (or grid-stride). Native FP64.
void launch_add_fudge_diag_models_batched(const double* dQ, double* dQf, int m,
                                          double fudge, int n_models, cudaStream_t stream);

/// Build a BATCHED identity RHS arena [m*m*B] (per-model m×m identity, col-major) for
/// the cuSOLVER potrsBatched inverse. One thread per (element, model). Native FP64.
void launch_fill_identity_batched(double* dI, int m, int n_models, cudaStream_t stream);

/// The MODEL-BATCHED fit: ONE thread per model runs the WHOLE small-path qpAdm fit
/// (the FROZEN CONTRACT §2c/§2d/§2e + M(fit-2) rank-sweep + popdrop) for its model,
/// reading per-model slices of `dTotal` (x_total, m per model), `dQinv` (m*m per model,
/// the batched Cholesky inverse), `dLoo` (m*nb per model), and `d_block_sizes`. It
/// transliterates run_impl exactly (seed -> ALS -> constrained weight solve -> chisq;
/// the rank sweep r=0..rmax; the per-block LOO re-fits -> SE diag; the popdrop
/// leave-one-LEFT-out reduced fits over the per-model X+Qinv). Per model it writes:
///   d_weight[B*nl]       (full-rank weights, Σ=1; 0 on RankDeficient)
///   d_se[B*nl]           (jackknife SE diag, native-FP64 sample variance / nb-1)
///   d_chisq[B]           (fitted-rank chisq)
///   d_status[B]          (0 Ok / 6 RankDeficient)
///   d_rank_chisq[B*(rmax+1)]   (chisq(r) for r=0..rmax)
///   d_pop_chisq[B*(nl+1)]      (popdrop chisq: full row, then each single-source drop)
///   d_pop_wfull[B*nl]          (FULL-model popdrop-row weights at rank nl-1, the feasibility source)
/// The host assembles the QpAdmResult fields (p via pchisq, rankdrop nested table,
/// popdrop pattern strings + feasibility) from these — exactly as run_impl/ranktest do.
/// `rmax` = min(nl,nr)-1; `r_fit` (= the reported rank) is nl-1 by default (opts.rank<0).
/// MANY threads ⇒ the SMALL per-thread local bound (kQpMax*); the host bucketer
/// guarantees nl<=5, nr<=10, r<=4 before launching. Native FP64.
void launch_qpadm_fit_models_batched(const double* dTotal, const double* dQinv,
                                     const double* dLoo, const int* d_block_sizes,
                                     int nl, int nr, int r_fit, int rmax,
                                     double fudge, int als_iters, int nb, int n_models,
                                     double* d_weight, double* d_se, double* d_chisq,
                                     int* d_status, double* d_rank_chisq,
                                     double* d_pop_chisq, double* d_pop_wfull,
                                     cudaStream_t stream);

/// S7 LOO per-block re-fits, MODEL-BATCHED across (model, block) — the throughput-
/// scaled SE source. One thread per (model, b) fits that block's leave-one-out weights
/// and writes the SCALED weight vector (×s, s=(nb-1)/sqrt(nb)) to dWmat[model*nb*nl +
/// b*nl + i]. B·nb parallel threads (vs the per-model serial nb-loop). Native FP64.
void launch_qpadm_loo_models_batched(const double* dLoo, const double* dQinv,
                                     int nl, int nr, int r_fit, double fudge,
                                     int als_iters, int nb, int n_models, double s,
                                     double* dWmat, cudaStream_t stream);

/// S7 SE diag from the LOO wmat, MODEL-BATCHED across (model, weight col). One thread
/// per (model, i): SE[i] = sqrt(Σ_b (wmat[b,i]-mean_i)²/(nb-1)) in a FIXED reduction
/// order (deterministic, no atomics ⇒ G=1==G=2 bit-identical). Writes d_se[B*nl]. Native FP64.
void launch_qpadm_se_from_wmat_batched(const double* dWmat, int nl, int nb,
                                       int n_models, double* d_se, cudaStream_t stream);

/// SE-policy survivor GATHER (the two-pass §M(fit-3) seam). Compact the per-model
/// dLoo (m*nb) + dQinv (m*m) slices of the n_surv survivor positions `d_surv` (chunk
/// positions, ASCENDING) into dense output arenas dLooDst/dQinvDst — ONE launch vs a
/// per-survivor cudaMemcpyAsync D2D loop. Pure data movement (parity-NEUTRAL: a
/// survivor's compacted slice is BIT-IDENTICAL to its full-arena slice), so the SE
/// kernels run UNCHANGED over n_models=n_surv. Native FP64 (it only copies doubles).
void launch_qpadm_gather_loo_qinv(const double* dLooSrc, const double* dQinvSrc,
                                  const int* d_surv, int m, int nb, int n_surv,
                                  double* dLooDst, double* dQinvDst, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPADM_FIT_KERNELS_CUH
