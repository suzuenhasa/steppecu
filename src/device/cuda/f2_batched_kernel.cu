// src/device/cuda/f2_batched_kernel.cu
//
// M4 size-grouped strided-batched f2: blocks are bucketed by ceil-pow2 SNP count and
// each bucket runs as one strided-batched GEMM group, one bucket resident at a time.
// Owns the three batched kernels/wrappers around the grouped GEMMs; reuses the shared
// f2 feeder and precision policy from f2_block_kernel.cu.
//
// Reference: docs/reference/src_device_cuda_f2_batched_kernel.cu.md
#include "device/cuda/f2_batched_kernel.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include "core/internal/f2_estimator.hpp"
#include "core/internal/host_device.hpp"
#include "core/internal/launch_config.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/f2_block_kernel.cuh"
#include "steppe/config.hpp"

namespace steppe::device {

using core::assemble_f2_numerator;
using core::finalize_f2;
using core::kF2StackedBlocks;

namespace {

// Stage 1: gather into padded slabs — reference §4
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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z;
    if (i >= P || c >= s_pad || k >= n_in_group) return;

    const long Pl = static_cast<long>(P);
    const long twoP = kF2StackedBlocks * Pl;
    const long Psp = Pl * s_pad;
    const long twoPsp = twoP * s_pad;

    const int id = block_ids_in_group[k];
    const int sz = block_sizes[id];

    const long dQidx = static_cast<long>(i) + Pl * c + Psp * k;
    const long dst_sumsq_row = static_cast<long>(i) + twoP * c + twoPsp * k;
    const long dst_hc_row = (Pl + static_cast<long>(i)) + twoP * c + twoPsp * k;

    if (c < sz) {
        const long src = block_offsets[id] + c;
        const long sQ = static_cast<long>(i) + Pl * src;
        const long src_sumsq_row = static_cast<long>(i) + twoP * src;
        const long src_hc_row = (Pl + static_cast<long>(i)) + twoP * src;
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

// Stage 3: fused assemble + scatter — reference §6
__global__ void __launch_bounds__(steppe::kCdivBlock * steppe::kCdivBlock)
assemble_blocks_group_kernel(const double* __restrict__ Gg,
                                            const double* __restrict__ Vpairg,
                                            const double* __restrict__ Rg,
                                            const int* __restrict__ block_ids_in_group,
                                            int P, int n_in_group,
                                            double* __restrict__ f2_all,
                                            double* __restrict__ vpair_all) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z;
    if (i >= P || j >= P || k >= n_in_group) return;

    const size_t Pz = static_cast<size_t>(P);
    const size_t twoP = kF2StackedBlocks * Pz;
    const size_t si = static_cast<size_t>(i);
    const size_t sj = static_cast<size_t>(j);

    const size_t g_slab = Pz * Pz * static_cast<size_t>(k);
    const size_t r_slab = twoP * Pz * static_cast<size_t>(k);
    const size_t pp_off = si + sj * Pz;

    const double Gij = Gg[g_slab + pp_off];
    const double vp = Vpairg[g_slab + pp_off];
    const double sumsq_i = Rg[r_slab + si + sj * twoP];
    const double sumsq_j = Rg[r_slab + sj + si * twoP];
    const double hsum_i = Rg[r_slab + (Pz + si) + sj * twoP];
    const double hsum_j = Rg[r_slab + (Pz + sj) + si * twoP];

    const double num =
        assemble_f2_numerator(sumsq_i, sumsq_j, Gij, hsum_i, hsum_j);

    const int id = block_ids_in_group[k];
    const size_t dst_slab = Pz * Pz * static_cast<size_t>(id);
    f2_all[dst_slab + pp_off] = finalize_f2(num, vp);
    vpair_all[dst_slab + pp_off] = vp;
}

}  // namespace

// Gather launch wrapper — reference §4, §8
void launch_gather_group(const double* dQ_all, const double* dV_all, const double* dS_all,
                         const int* d_block_ids_in_group, const long* d_block_offsets,
                         const int* d_block_sizes,
                         int P, int s_pad, int n_in_group,
                         double* dQg, double* dVg, double* dSg,
                         cudaStream_t stream) {
    STEPPE_ASSERT(s_pad >= 1,
                  "launch_gather_group: s_pad must be >= 1 (a zero gridDim.y is an "
                  "invalid launch; cleanup F6/B24)");
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(s_pad)),
                    core::grid_z_extent(n_in_group));
    gather_group_kernel<<<grid, block, 0, stream>>>(
        dQ_all, dV_all, dS_all, d_block_ids_in_group, d_block_offsets, d_block_sizes,
        P, s_pad, n_in_group, dQg, dVg, dSg);
    STEPPE_CUDA_CHECK_KERNEL();
}

// Stage 2: the three batched GEMMs — reference §5
void run_f2_gemms_group(cublasHandle_t handle, const Precision& precision,
                        int P, int s_pad, int n_in_group,
                        const double* dQg, const double* dVg, const double* dSg,
                        double* dGg, double* dVpairg, double* dRg) {
    STEPPE_ASSERT(n_in_group >= 1,
                  "run_f2_gemms_group: n_in_group (batchCount) must be >= 1 "
                  "(cleanup F6/B24)");
    STEPPE_ASSERT(s_pad >= 1,
                  "run_f2_gemms_group: s_pad (GEMM contraction extent k) must be >= 1 "
                  "(cleanup F6/B24)");
    const cublasComputeType_t ct = f2_compute_type(precision);
    const double one = 1.0;
    const double zero = 0.0;
    const int twoP_dim = kF2StackedBlocks * P;
    const long Psp = static_cast<long>(P) * s_pad;
    const long twoPsp = static_cast<long>(twoP_dim) * s_pad;

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, s_pad,
        &one, dQg, CUDA_R_64F, P, Psp, dQg, CUDA_R_64F, P, Psp,
        &zero, dGg, CUDA_R_64F, P, static_cast<long>(P) * P,
        n_in_group, ct, CUBLAS_GEMM_DEFAULT));

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, s_pad,
        &one, dVg, CUDA_R_64F, P, Psp, dVg, CUDA_R_64F, P, Psp,
        &zero, dVpairg, CUDA_R_64F, P, static_cast<long>(P) * P,
        n_in_group, ct, CUBLAS_GEMM_DEFAULT));

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, twoP_dim, P, s_pad,
        &one, dSg, CUDA_R_64F, twoP_dim, twoPsp, dVg, CUDA_R_64F, P, Psp,
        &zero, dRg, CUDA_R_64F, twoP_dim, static_cast<long>(twoP_dim) * P,
        n_in_group, ct, CUBLAS_GEMM_DEFAULT));
}

// Assemble launch wrapper — reference §6, §8
void launch_assemble_blocks_group(const double* dGg, const double* dVpairg, const double* dRg,
                                  const int* d_block_ids_in_group,
                                  int P, int n_in_group,
                                  double* dF2_all, double* dVpair_all,
                                  cudaStream_t stream) {
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(P)),
                    core::grid_z_extent(n_in_group));
    assemble_blocks_group_kernel<<<grid, block, 0, stream>>>(
        dGg, dVpairg, dRg, d_block_ids_in_group, P, n_in_group, dF2_all, dVpair_all);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
