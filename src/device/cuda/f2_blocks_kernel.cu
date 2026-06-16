// src/device/cuda/f2_blocks_kernel.cu
//
// M4 — the BATCHED per-block f2 path on the GPU (architecture.md §5 S2 "Batched
// over all n_block blocks via cublasDgemmStridedBatched ... accumulating into the
// resident f2_blocks and Vpair", §7, §11.1, §12; ROADMAP §3 M4). The
// SPIKE-CHOSEN design (experiments/f2_block_batched_spike, MEASURED on real AADR
// P=768/M=584131): SIZE-GROUPED strided-batched.
//
// WHY GROUPED (the spike verdict, ROADMAP M4):
//   * loop of per-block GEMMs is launch-bound (~2271 tiny launches): 591 ms @40b
//     vs grouped 317 ms — grouped is 1.9× faster.
//   * naive global-s_max strided-batched needs P²·s_max·n_block ≈ 53.8 GB > 32 GB
//     at P=768 — NOT VRAM-viable.
//   * grouped: bucket blocks by ceil-pow2 of size; one strided-batched call per
//     bucket, padded only to the bucket width; ONE bucket resident at a time. 1.43×
//     pad waste (vs 2.76× global), fits VRAM, AND is the fastest: 7.2×/8.9× over
//     native FP64 at 40/32-bit — the per-block regime preserves the M0 big-GEMM win.
//
// This TU owns the three batched kernels around the grouped GEMMs:
//   1. launch_gather_group           — gather a bucket's SNP columns out of the
//                                      block-contiguous feeder into a padded slab
//                                      layout (pad cols V=0 ⇒ contribute nothing).
//   2. run_f2_gemms_group            — the three GEMMs as cublasGemmStridedBatchedEx
//                                      over the bucket's slabs (precision-governed).
//   3. launch_assemble_blocks_group  — fused numerator+divide (NATIVE FP64, the
//                                      catastrophic-cancellation step) scattered
//                                      into the resident [P×P×n_block] tensors.
// The fused elementwise feeder over ALL SNPs is the EXISTING block-agnostic
// launch_f2_feeder (f2_block_kernel.cu) — it is per-element, so M4 reuses it
// verbatim (no second feeder; architecture.md §2 DRY). The precision engagement +
// compute-type mapping are the SHARED engage_f2_precision / f2_compute_type (also
// f2_block_kernel.cu) — the single source of the precision policy.
//
// CUDA TU: PRIVATE to steppe_device (architecture.md §4). Includes the SHARED
// host/device f2 primitive so the CPU oracle and this path cannot diverge on the
// formula (architecture.md §13).
#include "device/cuda/f2_blocks_kernel.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include "core/internal/f2_estimator.hpp"  // assemble_f2_numerator, finalize_f2, grid_for, kCdivBlock
#include "device/cuda/check.cuh"           // STEPPE_CUDA_CHECK, CUBLAS_CHECK, STEPPE_CUDA_CHECK_KERNEL
#include "device/cuda/f2_block_kernel.cuh" // engage_f2_precision, f2_compute_type (SHARED policy)
#include "steppe/config.hpp"               // kCdivBlock

namespace steppe::device {

using core::assemble_f2_numerator;
using core::finalize_f2;

namespace {

// =============================================================================
// Kernel: gather one size-group into a padded, batched slab layout.
//
// Each thread owns one (population i, padded-column c, group-slab k) entry. Block
// k of the group is the global block d_block_ids_in_group[k], whose SNPs are the
// contiguous feeder columns [off, off + sz) where off = d_block_offsets[id], sz =
// d_block_sizes[id]. For c < sz we copy the real feeder column; for c ≥ sz (the
// pad) we write 0 in Q, V, and S so the padded GEMM is identical to the unpadded
// one (V=0 ⇒ zero contribution; architecture.md §5 S2). Column-major slabs:
//   dQg/dVg [P × s_pad × n]  : element (i, c, k) at i + P·c + P·s_pad·k
//   dSg     [2P × s_pad × n] : element (i, c, k) at i + 2P·c + 2P·s_pad·k
// =============================================================================
__global__ void gather_group_kernel(const double* __restrict__ Q_all,
                                    const double* __restrict__ V_all,
                                    const double* __restrict__ S_all,
                                    const int* __restrict__ block_ids_in_group,
                                    const long* __restrict__ block_offsets,
                                    const int* __restrict__ block_sizes,
                                    int P, int s_pad, int n_in_group,
                                    double* __restrict__ Qg,
                                    double* __restrict__ Vg,
                                    double* __restrict__ Sg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // population row
    const int c = blockIdx.y * blockDim.y + threadIdx.y;   // padded column
    const int k = blockIdx.z;                              // group slab (block)
    if (i >= P || c >= s_pad || k >= n_in_group) return;

    const long Pl = static_cast<long>(P);
    const long twoP = 2 * Pl;
    const long Psp = Pl * s_pad;
    const long twoPsp = twoP * s_pad;

    const int id = block_ids_in_group[k];
    const int sz = block_sizes[id];

    // Destination (i, c) within slab k.
    const long dQidx = static_cast<long>(i) + Pl * c + Psp * k;
    const long dSqsq = static_cast<long>(i) + twoP * c + twoPsp * k;
    const long dShc = (Pl + static_cast<long>(i)) + twoP * c + twoPsp * k;

    if (c < sz) {
        const long src = block_offsets[id] + c;              // contiguous feeder column
        const long sQ = static_cast<long>(i) + Pl * src;     // (i, src) in [P × M]
        const long sSqsq = static_cast<long>(i) + twoP * src;        // Qsq row
        const long sShc = (Pl + static_cast<long>(i)) + twoP * src;  // Hc row
        Qg[dQidx] = Q_all[sQ];
        Vg[dQidx] = V_all[sQ];
        Sg[dSqsq] = S_all[sSqsq];
        Sg[dShc] = S_all[sShc];
    } else {
        Qg[dQidx] = 0.0;
        Vg[dQidx] = 0.0;
        Sg[dSqsq] = 0.0;
        Sg[dShc] = 0.0;
    }
}

// =============================================================================
// Kernel: fused numerator+divide for a group, scattered into the resident tensors.
//
// Each thread owns one (i, j, group-slab k) output entry. Reads the group-local
// GEMM outputs (Gg/Vpairg [P×P×n], Rg [2P×P×n]) and assembles f2(i,j) for global
// block id = block_ids_in_group[k] via the SHARED assemble_f2_numerator /
// finalize_f2 — the SAME functions the CPU oracle calls, so the cancellation
// formula cannot diverge (architecture.md §13). NATIVE FP64 (the Precision knob
// governs only the GEMMs; architecture.md §12). Writes into the [P×P×n_block]
// tensors at i + P·j + P·P·id; Vpairg is carried through unchanged.
//
//   Rg slab is [2P × P] column-major (lda = 2P): top P rows Σp², bottom P Σhc:
//     sumsq_i = Rg(i,   j)   sumsq_j = Rg(j,   i)
//     hsum_i  = Rg(P+i, j)   hsum_j  = Rg(P+j, i)
// =============================================================================
__global__ void assemble_blocks_group_kernel(const double* __restrict__ Gg,
                                            const double* __restrict__ Vpairg,
                                            const double* __restrict__ Rg,
                                            const int* __restrict__ block_ids_in_group,
                                            int P, int n_in_group,
                                            double* __restrict__ f2_all,
                                            double* __restrict__ vpair_all) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // row
    const int j = blockIdx.y * blockDim.y + threadIdx.y;  // col
    const int k = blockIdx.z;                             // group slab
    if (i >= P || j >= P || k >= n_in_group) return;

    const size_t Pp = static_cast<size_t>(P);
    const size_t twoP = 2 * Pp;
    const size_t si = static_cast<size_t>(i);
    const size_t sj = static_cast<size_t>(j);

    const size_t gSlab = Pp * Pp * static_cast<size_t>(k);          // [P×P] slab base
    const size_t rSlab = twoP * Pp * static_cast<size_t>(k);        // [2P×P] slab base

    const double Gij = Gg[gSlab + si + sj * Pp];
    const double vp = Vpairg[gSlab + si + sj * Pp];
    const double sumsq_i = Rg[rSlab + si + sj * twoP];          // Rg(i,   j)
    const double sumsq_j = Rg[rSlab + sj + si * twoP];          // Rg(j,   i)
    const double hsum_i = Rg[rSlab + (Pp + si) + sj * twoP];    // Rg(P+i, j)
    const double hsum_j = Rg[rSlab + (Pp + sj) + si * twoP];    // Rg(P+j, i)

    const double num =
        assemble_f2_numerator(sumsq_i, sumsq_j, Gij, hsum_i, hsum_j);

    const int id = block_ids_in_group[k];
    const size_t dstSlab = Pp * Pp * static_cast<size_t>(id);   // resident [P×P×n_block] slab
    f2_all[dstSlab + si + sj * Pp] = finalize_f2(num, vp);
    vpair_all[dstSlab + si + sj * Pp] = vp;
}

}  // namespace

// =============================================================================
// Launch wrappers (architecture.md §7 — host code never sees <<<>>>).
// =============================================================================

void launch_gather_group(const double* dQ_all, const double* dV_all, const double* dS_all,
                         const int* d_block_ids_in_group, const long* d_block_offsets,
                         const int* d_block_sizes,
                         int P, int s_pad, int n_in_group,
                         double* dQg, double* dVg, double* dSg,
                         cudaStream_t stream) {
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(s_pad)),
                    static_cast<unsigned>(n_in_group));
    gather_group_kernel<<<grid, block, 0, stream>>>(
        dQ_all, dV_all, dS_all, d_block_ids_in_group, d_block_offsets, d_block_sizes,
        P, s_pad, n_in_group, dQg, dVg, dSg);
    STEPPE_CUDA_CHECK_KERNEL();
}

void run_f2_gemms_group(cublasHandle_t handle, const Precision& precision,
                        int P, int s_pad, int n_in_group,
                        const double* dQg, const double* dVg, const double* dSg,
                        double* dGg, double* dVpairg, double* dRg) {
    // No cublasSetStream here: the handle's stream + emulated-FP64 workspace are
    // bound ONCE at backend construction via CublasHandle::set_stream
    // (architecture.md §12; cleanup X-1/B1). The M4 grouped path is called once
    // PER CHUNK, so a per-call cublasSetStream would reset the workspace to the
    // default pool (cuBLAS §2.4.7) before EVERY chunk's strided-batched GEMMs —
    // the worst incarnation of the B1 determinism void. The handle's math mode /
    // FIXED-slice control are engaged ONCE by the caller (engage_f2_precision
    // before the group loop); set only the per-call compute type here, matching
    // that engaged policy (architecture.md §12, §2 DRY).
    const cublasComputeType_t ct = f2_compute_type(precision);
    const double one = 1.0;
    const double zero = 0.0;
    const int twoP = 2 * P;
    const long Psp = static_cast<long>(P) * s_pad;
    const long twoPsp = static_cast<long>(twoP) * s_pad;

    // G[P×P×n] = Qg · Qgᵀ per slab. A=Qg (m=P,k=s_pad,lda=P,stride=P·s_pad),
    //   B=Qg (n=P,ldb=P,stride=P·s_pad), C=Gg (ldc=P,stride=P·P).
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, s_pad,
        &one, dQg, CUDA_R_64F, P, Psp, dQg, CUDA_R_64F, P, Psp,
        &zero, dGg, CUDA_R_64F, P, static_cast<long>(P) * P,
        n_in_group, ct, CUBLAS_GEMM_DEFAULT));

    // Vpair[P×P×n] = Vg · Vgᵀ per slab (RETAINED — the S4 jackknife weight).
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, s_pad,
        &one, dVg, CUDA_R_64F, P, Psp, dVg, CUDA_R_64F, P, Psp,
        &zero, dVpairg, CUDA_R_64F, P, static_cast<long>(P) * P,
        n_in_group, ct, CUBLAS_GEMM_DEFAULT));

    // R[2P×P×n] = Sg · Vgᵀ per slab. A=Sg (m=2P,k=s_pad,lda=2P,stride=2P·s_pad),
    //   B=Vg (n=P,ldb=P,stride=P·s_pad), C=Rg (ldc=2P,stride=2P·P).
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, twoP, P, s_pad,
        &one, dSg, CUDA_R_64F, twoP, twoPsp, dVg, CUDA_R_64F, P, Psp,
        &zero, dRg, CUDA_R_64F, twoP, static_cast<long>(twoP) * P,
        n_in_group, ct, CUBLAS_GEMM_DEFAULT));
}

void launch_assemble_blocks_group(const double* dGg, const double* dVpairg, const double* dRg,
                                  const int* d_block_ids_in_group,
                                  int P, int n_in_group,
                                  double* dF2_all, double* dVpair_all,
                                  cudaStream_t stream) {
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(n_in_group));
    assemble_blocks_group_kernel<<<grid, block, 0, stream>>>(
        dGg, dVpairg, dRg, d_block_ids_in_group, P, n_in_group, dF2_all, dVpair_all);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
