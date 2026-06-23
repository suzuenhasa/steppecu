// src/device/cuda/dstat_kernel.cuh
//
// Narrow launch-wrapper declaration for the GPU genotype-path NORMALIZED-D per-SNP
// reduction — qpDstat Part B (include/steppe/dstat.hpp; the S2 divergence). Host
// orchestration (cuda_backend.cu) calls this `void launch_dstat_block_reduce(...)`;
// the kernel body and `<<<>>>` live only in dstat_kernel.cu (architecture.md §7
// "host code never includes kernel bodies or <<<>>>").
//
// This header names a CUDA type (cudaStream_t) and so is PRIVATE to steppe_device
// (architecture.md §4) — the device-internal seam between the backend and the D
// kernel TU, not the CUDA-free public seam (include/steppe/dstat.hpp).
#ifndef STEPPE_DEVICE_CUDA_DSTAT_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_DSTAT_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

/// Per-block, per-quadruple normalized-D reduction on the GPU (the S2 divergence; D's
/// Part B back-end). One thread owns one (quadruple k, block b) output cell: it walks
/// the block's contiguous SNP columns [block_begin[b], block_begin[b]+block_size[b]) and,
/// for each SNP s where V==1 in ALL of p1..p4 (the allsnps=TRUE per-(block,quadruple)
/// finiteness mask), accumulates with a,b,c,d = Q[p*P + ... ] (column-major [P × M]):
///   numsum += (a-b)*(c-d),  densum += (a+b-2ab)*(c+d-2cd),  cnt += 1.
/// All accumulation in double (the §12 num/den cancellation; native FP64). Outputs are
/// ROW-MAJOR [N × n_block]: cell (k, b) at k*n_block + b.
///
///   d_Q,d_V        [P × M] column-major decode_af outputs (element (i,s) at i + P·s)
///   d_quad         [4 × N] flattened P-axis index quads (quad k at 4*k .. 4*k+3)
///   d_block_begin  [n_block] the first SNP column index of each block
///   d_block_size   [n_block] the SNP count of each block (Σ == M; contiguous in file order)
///   d_numsum/d_densum/d_cnt  [N * n_block] row-major outputs (cell (k,b) at k*n_block+b)
///
/// `P`/`M` are the Q/V dims; `N` the quadruple count; `n_block` the block count.
void launch_dstat_block_reduce(const double* d_Q, const double* d_V, int P, long M,
                               const int* d_quad, int N,
                               const int* d_block_begin, const int* d_block_size,
                               int n_block,
                               double* d_numsum, double* d_densum, double* d_cnt,
                               cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_DSTAT_KERNEL_CUH
