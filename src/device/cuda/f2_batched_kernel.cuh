// src/device/cuda/f2_batched_kernel.cuh
//
// Narrow launch-wrapper declarations for the M4 BATCHED per-block f2 path
// (architecture.md §5 S2 "Batched over all n_block blocks", §7 "host code never
// includes kernel bodies or <<<>>>"; ROADMAP §3 M4). Host orchestration
// (cuda_backend.cu) calls these `void launch_*` / `run_*` functions; the kernel
// bodies + `<<<>>>` + the cublas*StridedBatched calls live only in the .cu.
//
// THE SPIKE-CHOSEN DESIGN (experiments/f2_block_batched_spike, ROADMAP M4 spike,
// MEASURED on real AADR P=768/M=584131): size-grouped strided-batched. Blocks are
// bucketed by ceil-power-of-2 of their SNP count; each bucket is run as ONE
// cublasGemmStridedBatchedEx per GEMM (G, Vpair, R), padded only to that bucket's
// width (pad columns carry V=0 ⇒ contribute nothing). This beats both the loop of
// per-block GEMMs (launch-bound: 591 ms vs grouped 317 ms @ 40-bit) and the naive
// global-s_max strided design (which needs 53.8 GB > 32 GB VRAM at P=768 and is
// NOT viable). Grouped is the FASTEST and the only VRAM-frugal batched design
// (per-group resident; 1.43× pad waste vs the global design's 2.76×). At 40-bit
// it runs 7.2× / at 32-bit 8.9× over native FP64 — the per-block regime preserves
// the M0 big-GEMM win.
//
// This header names CUDA types (cublasHandle_t) and so is PRIVATE to
// steppe_device (architecture.md §4) — the device-internal seam between the
// backend and the kernels, not the CUDA-free public ComputeBackend seam.
#ifndef STEPPE_DEVICE_CUDA_F2_BATCHED_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_F2_BATCHED_KERNEL_CUH

#include <cublas_v2.h>

#include "steppe/config.hpp"  // steppe::Precision

namespace steppe::device {

/// Gather one size-group's SNP columns out of the block-contiguous feeder outputs
/// (dQ/dV/dS over ALL M SNPs, column-major [P×M] / [2P×M]) into a PADDED, batched
/// slab layout for one cublasGemmStridedBatchedEx call (architecture.md §5 S2).
///
/// The group holds `n_in_group` blocks, each padded to `s_pad` columns; block k
/// of the group is the global block `block_ids_in_group[k]`, whose SNPs occupy
/// columns `[block_offsets[block_ids_in_group[k]], +block_sizes[...])` in the
/// contiguous feeder. Pad columns (c ≥ that block's size) are written as 0 in Q,
/// V, and S so they contribute nothing to the masked GEMMs (V=0 ⇒ zero rows/cols).
/// Outputs are column-major slabs: dQg/dVg are [P × s_pad × n_in_group] (slab
/// stride P·s_pad), dSg is [2P × s_pad × n_in_group] (slab stride 2P·s_pad).
/// All native FP64 (a memory-bound gather; reduced precision buys nothing).
///
/// PRECONDITIONS (debug-asserted; cleanup F6/B24): `s_pad >= 1` (it rides
/// gridDim.y, so s_pad == 0 would be a zero y-extent the driver rejects) and
/// `1 <= n_in_group <= kMaxGridZ` (the gridDim.z batch count, pinned by
/// grid_z_extent — a zero gridDim.z is an invalid launch). The backend always
/// satisfies both (bucket width >= kBlockGroupPadBase; the batch is tiled to
/// kMaxGridZ), so this is a fail-fast on a corrupted call, e.g. an empty SNP shard
/// under M4.5 sharding. Throws CublasError/CudaError on a CUDA failure (§7, §10).
void launch_gather_group(const double* dQ_all, const double* dV_all, const double* dS_all,
                         const int* d_block_ids_in_group, const long* d_block_offsets,
                         const int* d_block_sizes,
                         int P, int s_pad, int n_in_group,
                         double* dQg, double* dVg, double* dSg,
                         cudaStream_t stream);

/// Run the three f2 GEMMs for ONE size-group as cublasGemmStridedBatchedEx over
/// its `n_in_group` padded slabs (architecture.md §5 S2; the spike-chosen grouped
/// design). Each padded slab is `s_pad` wide:
///   dGg     [P  × P  × n] = Qg · Qgᵀ     per slab (slab stride P·P)
///   dVpairg [P  × P  × n] = Vg · Vgᵀ     per slab (RETAINED — S4 weight)
///   dRg     [2P × P  × n] = Sg · Vgᵀ     per slab (top P rows Σp², bottom P Σhc)
/// `precision` governs ONLY these GEMMs (architecture.md §12): EmulatedFp64 ⇒
/// FIXED-slice Ozaki at `precision.mantissa_bits`; Fp64 ⇒ native CUBLAS_COMPUTE_64F.
/// The handle must already be created (RAII) with its stream + emulated-FP64
/// determinism workspace bound ONCE via CublasHandle::set_stream/set_workspace
/// (architecture.md §12; cleanup X-1/B1). Engaging the precision on the handle is
/// done ONCE by the caller (run_f2_gemms in f2_block_kernel.cu shares the policy);
/// this routine sets only the per-call compute type. It takes NO stream and never
/// calls `cublasSetStream` — doing so per chunk would reset the workspace to the
/// default pool (cuBLAS §2.4.7) before every chunk's GEMMs and defeat §12.
///
/// PRECONDITIONS (debug-asserted; cleanup F6/B24): `s_pad >= 1` (the GEMM
/// contraction extent k; k == 0 is a beta-only scale, not the intended product)
/// and `n_in_group >= 1` (the cublasGemmStridedBatchedEx batchCount; 0 is a
/// degenerate empty batch). This wrapper issues no <<<>>> launch and so cannot
/// reuse the gather/assemble wrappers' grid_z_extent guard — it asserts these
/// directly (architecture.md §2 fail-fast). Throws CublasError on a cuBLAS failure
/// (§7, §10).
void run_f2_gemms_group(cublasHandle_t handle, const Precision& precision,
                        int P, int s_pad, int n_in_group,
                        const double* dQg, const double* dVg, const double* dSg,
                        double* dGg, double* dVpairg, double* dRg);

/// Scatter one size-group's assembled f2 + Vpair into the resident
/// [P × P × n_block] tensors (architecture.md §5 S2, §11.1). Each slab k holds the
/// group-local GEMM outputs Gg/Vpairg/Rg for global block `block_ids_in_group[k]`;
/// this fuses the numerator+divide (NATIVE FP64, the catastrophic-cancellation
/// step — architecture.md §12, via the SHARED assemble_f2_numerator/finalize_f2
/// primitives) and writes the result into dF2_all / dVpair_all at the block's
/// [P×P] slab `i + P·j + P·P·block_id`. dVpairg is carried through unchanged.
///
/// PRECONDITION (debug-asserted; cleanup F6/B24): `1 <= n_in_group <= kMaxGridZ`
/// (the gridDim.z batch count, pinned by grid_z_extent — a zero gridDim.z is an
/// invalid launch). This wrapper has no s_pad parameter (its slabs are [P×P]).
/// Throws CublasError/CudaError on a CUDA failure (§7, §10).
void launch_assemble_blocks_group(const double* dGg, const double* dVpairg, const double* dRg,
                                  const int* d_block_ids_in_group,
                                  int P, int n_in_group,
                                  double* dF2_all, double* dVpair_all,
                                  cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_F2_BATCHED_KERNEL_CUH
