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

#include "core/internal/f2_estimator.hpp"  // assemble_f2_numerator, finalize_f2
#include "core/internal/host_device.hpp"   // STEPPE_ASSERT (the §2 fail-fast facility)
#include "core/internal/launch_config.hpp" // grid_for, grid_z_extent, kMaxGridZ (launch-math home)
#include "device/cuda/check.cuh"           // STEPPE_CUDA_CHECK, CUBLAS_CHECK, STEPPE_CUDA_CHECK_KERNEL
#include "device/cuda/f2_block_kernel.cuh" // engage_f2_precision, f2_compute_type (SHARED policy)
#include "steppe/config.hpp"               // kCdivBlock

namespace steppe::device {

using core::assemble_f2_numerator;
using core::finalize_f2;
using core::kF2StackedBlocks;  // the [2P × …] stacked-S row-block count (f2_estimator.hpp)

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
// [21.4] __launch_bounds__(kCdivBlock * kCdivBlock) == 16*16 = 256 pins the register
// cap to the SOLE launch block (the fixed dim3(kCdivBlock, kCdivBlock), launch_gather_
// group); the fixed block never exceeds the bound ⇒ no launch failure (Prog Guide §5.4).
__global__ void __launch_bounds__(steppe::kCdivBlock * steppe::kCdivBlock)
gather_group_kernel(const double* __restrict__ Q_all,
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
    const long twoP = kF2StackedBlocks * Pl;
    const long Psp = Pl * s_pad;
    const long twoPsp = twoP * s_pad;

    const int id = block_ids_in_group[k];
    const int sz = block_sizes[id];

    // Destination (i, c) within slab k.
    const long dQidx = static_cast<long>(i) + Pl * c + Psp * k;
    const long dst_sumsq_row = static_cast<long>(i) + twoP * c + twoPsp * k;
    const long dst_hc_row = (Pl + static_cast<long>(i)) + twoP * c + twoPsp * k;

    if (c < sz) {
        const long src = block_offsets[id] + c;              // contiguous feeder column
        const long sQ = static_cast<long>(i) + Pl * src;     // (i, src) in [P × M]
        const long src_sumsq_row = static_cast<long>(i) + twoP * src;        // Qsq row
        const long src_hc_row = (Pl + static_cast<long>(i)) + twoP * src;    // Hc row
        Qg[dQidx] = Q_all[sQ];
        Vg[dQidx] = V_all[sQ];
        Sg[dst_sumsq_row] = S_all[src_sumsq_row];
        Sg[dst_hc_row] = S_all[src_hc_row];
    } else {
        Qg[dQidx] = 0.0;
        Vg[dQidx] = 0.0;
        Sg[dst_sumsq_row] = 0.0;
        Sg[dst_hc_row] = 0.0;
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
//
// COALESCING (cleanup 20.1/MED — assemble-transpose, ACCEPTED COST): consecutive
// warp lanes vary si (= row i, threadIdx.x), so the column-major reads `sumsq_i` /
// `hsum_i` (column factor sj) are unit-stride / coalesced, while the TRANSPOSED
// `sumsq_j` = Rg(j,i) / `hsum_j` = Rg(P+j,i) (column factor si) stride by twoP (=2P,
// up to ~5000 doubles at P=2500) ⇒ uncoalesced. This transpose is MATHEMATICALLY
// REQUIRED — symmetric f2 needs BOTH the i-row and the j-row column sums of the
// [2P×P] Rg — so it is not a layout choice, and the per-element math/precision is
// unchanged. It is accepted as a bandwidth cost rather than staged through shared
// memory because: (1) it touches only the SMALL per-slab [2P×P] Rg (P ≤ ~2500), not
// the SNP-scale [P×M] arenas the feeder/decode fixes target; (2) the M4 grouped path
// is dominated by the three batched GEMMs (the spike cost center), not this
// catastrophic-cancellation NATIVE-FP64 assemble; (3) the symmetric (i-row + j-row)
// access would need TWO off-diagonal shared tiles plus block-diagonal bounds — real
// complexity and a new bank-conflict surface for a bounded win off the critical path.
// If the assemble ever profiles hot, stage the slab's needed Rg rows in a padded
// shared tile (finding's other option). The feeder/decode 20.1 fixes are applied;
// this one is the finding's documented-accepted-cost branch.
// =============================================================================
// [21.4] __launch_bounds__(kCdivBlock * kCdivBlock) == 16*16 = 256: SOLE launch block is
// the fixed dim3(kCdivBlock, kCdivBlock) (launch_assemble_blocks_group); register cap to
// the occupancy target, never under-launched ⇒ no launch failure (Prog Guide §5.4).
__global__ void __launch_bounds__(steppe::kCdivBlock * steppe::kCdivBlock)
assemble_blocks_group_kernel(const double* __restrict__ Gg,
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

    const size_t Pz = static_cast<size_t>(P);
    const size_t twoP = kF2StackedBlocks * Pz;
    const size_t si = static_cast<size_t>(i);
    const size_t sj = static_cast<size_t>(j);

    const size_t g_slab = Pz * Pz * static_cast<size_t>(k);         // [P×P] slab base
    const size_t r_slab = twoP * Pz * static_cast<size_t>(k);       // [2P×P] slab base
    const size_t pp_off = si + sj * Pz;                            // (i,j) in the [P×P] slab (20.3/LOW hoist)

    const double Gij = Gg[g_slab + pp_off];
    const double vp = Vpairg[g_slab + pp_off];
    const double sumsq_i = Rg[r_slab + si + sj * twoP];         // Rg(i,   j)
    const double sumsq_j = Rg[r_slab + sj + si * twoP];         // Rg(j,   i)
    const double hsum_i = Rg[r_slab + (Pz + si) + sj * twoP];   // Rg(P+i, j)
    const double hsum_j = Rg[r_slab + (Pz + sj) + si * twoP];   // Rg(P+j, i)

    const double num =
        assemble_f2_numerator(sumsq_i, sumsq_j, Gij, hsum_i, hsum_j);

    const int id = block_ids_in_group[k];
    const size_t dst_slab = Pz * Pz * static_cast<size_t>(id);  // resident [P×P×n_block] slab
    f2_all[dst_slab + pp_off] = finalize_f2(num, vp);
    vpair_all[dst_slab + pp_off] = vp;
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
    // s_pad >= 1 precondition (cleanup F6/B24): the bucket width is ceil-pow2 of a
    // block's SNP count and is always >= kBlockGroupPadBase, so s_pad == 0 cannot
    // reach here from the sole caller — but s_pad rides gridDim.y via grid_for(s_pad),
    // and grid_for(0) == 0 is a ZERO y-extent the driver rejects with
    // cudaErrorInvalidConfiguration (grid_for's own assert only rejects the OVER-limit
    // case, not a zero extent). Pin the documented invariant fail-fast (architecture.md
    // §2). n_in_group >= 1 is pinned below by grid_z_extent (a zero gridDim.z is the
    // same invalid launch).
    STEPPE_ASSERT(s_pad >= 1,
                  "launch_gather_group: s_pad must be >= 1 (a zero gridDim.y is an "
                  "invalid launch; cleanup F6/B24)");
    // Grid math via the single launch-config home (architecture.md §4, §7, §8).
    // x = P-tile, y = s_pad-tile (both square-block axes ⇒ grid_for, which now
    // debug-asserts the y/z 65 535 cap); z = the batch count, which is set DIRECTLY
    // and so bypasses grid_for — it routes through the dedicated grid_z_extent guard
    // (cleanup X-7/B6: a grid_for clamp alone would miss this direct z-extent, and it
    // also fail-fasts on n_in_group <= 0 — a zero gridDim.z, F6/B24). The backend tiles
    // the batch into chunks of ≤ kMaxGridZ blocks (vram_budget.hpp), so n_in_group ≤
    // kMaxGridZ holds at the call site; the assert pins both bounds.
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(s_pad)),
                    core::grid_z_extent(n_in_group));
    gather_group_kernel<<<grid, block, 0, stream>>>(
        dQ_all, dV_all, dS_all, d_block_ids_in_group, d_block_offsets, d_block_sizes,
        P, s_pad, n_in_group, dQg, dVg, dSg);
    STEPPE_CUDA_CHECK_KERNEL();
}

void run_f2_gemms_group(cublasHandle_t handle, const Precision& precision,
                        int P, int s_pad, int n_in_group,
                        const double* dQg, const double* dVg, const double* dSg,
                        double* dGg, double* dVpairg, double* dRg) {
    // Degenerate-batch guards (cleanup F6/B24). Unlike the two kernel-launch
    // wrappers, this routine issues no <<<>>> and so never passes n_in_group through
    // grid_z_extent — its z-axis equivalent is cublasGemmStridedBatchedEx's
    // batchCount (= n_in_group) and its contraction extent k (= s_pad). The backend
    // guards P/M/n_block and the bucket width is always >= kBlockGroupPadBase, so
    // n_in_group >= 1 and s_pad >= 1 hold from the sole caller — but a batchCount == 0
    // is a degenerate batched GEMM and a k == 0 (s_pad == 0) GEMM is a beta-only
    // scale with no contraction; both silently produce a wrong/empty result rather
    // than the intended one. Pin the documented preconditions fail-fast, mirroring
    // the gather/assemble wrappers' grid_z_extent assert (architecture.md §2; M4.5
    // sharding can hand a device an empty SNP shard ⇒ an empty bucket).
    STEPPE_ASSERT(n_in_group >= 1,
                  "run_f2_gemms_group: n_in_group (batchCount) must be >= 1 "
                  "(cleanup F6/B24)");
    STEPPE_ASSERT(s_pad >= 1,
                  "run_f2_gemms_group: s_pad (GEMM contraction extent k) must be >= 1 "
                  "(cleanup F6/B24)");
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
    const int twoP = kF2StackedBlocks * P;
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
    // Grid math via the single launch-config home (architecture.md §4, §7, §8).
    // x = y = P-tile (square-block axes ⇒ grid_for, with the y/z-cap assert); z =
    // the batch count, set DIRECTLY and so bypassing grid_for — routed through the
    // dedicated grid_z_extent guard (cleanup X-7/B6), which also fail-fasts on
    // n_in_group <= 0 (a zero gridDim.z is an invalid launch — F6/B24). This wrapper
    // has no s_pad parameter (its slabs are [P×P], not the s_pad-wide gather slabs).
    // The backend tiles the batch so n_in_group ≤ kMaxGridZ holds; the assert pins it.
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(P)),
                    core::grid_z_extent(n_in_group));
    assemble_blocks_group_kernel<<<grid, block, 0, stream>>>(
        dGg, dVpairg, dRg, d_block_ids_in_group, P, n_in_group, dF2_all, dVpair_all);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
