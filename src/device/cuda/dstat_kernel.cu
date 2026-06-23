// src/device/cuda/dstat_kernel.cu
//
// The GPU genotype-path NORMALIZED-D per-SNP reduction — qpDstat Part B (the S2
// divergence; include/steppe/dstat.hpp). NEVER the f2 GEMM, NEVER the f2 cache: a
// segmented reduction over each jackknife block's SNP columns, accumulating the
// AT2 qpdstat_geno num/den per (quadruple, block) over the allsnps=TRUE
// per-(block,quadruple) finiteness mask.
//
// STATISTIC (AT2 qpdstat_geno, allsnps=TRUE, f4mode=FALSE; pinned to the AT2 R golden):
// for quadruple (p1,p2,p3,p4) at SNP s with a=Q[p1,s], b=Q[p2,s], c=Q[p3,s], d=Q[p4,s],
//   num = (a-b)*(c-d),  den = (a+b-2ab)*(c+d-2cd)
// summed per block ONLY over SNPs valid in all 4 pops (V==1 for p1..p4).
//
// DESIGN-FOR-SCALE: batched over BOTH the N-quadruple axis and the n_block axis (one
// thread per (k,b) output cell), reading the resident Q/V [P × M] in VRAM. The output
// numsum/densum/cnt are [N × n_block] (tiny). All accumulation in double (the §12 num/den
// cancellation; native FP64). The blocks are contiguous in file order (assign_blocks is
// non-decreasing), so a cell's SNP columns are the slice [begin, begin+size).
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). The kernel body +
// <<<>>> live ONLY here; the backend reaches it through the narrow launch wrapper
// (dstat_kernel.cuh), never includes this body (architecture.md §7).
#include <cuda_runtime.h>

#include "device/cuda/check.cuh"          // STEPPE_CUDA_CHECK
#include "device/cuda/dstat_kernel.cuh"   // launch_dstat_block_reduce (the narrow seam)

namespace steppe::device {

namespace {

/// One thread per (quadruple k, block b) output cell. Walks the block's contiguous SNP
/// columns and accumulates the AT2 num/den over the 4-pop finiteness mask. Q/V are
/// column-major [P × M] (element (pop i, SNP s) at i + P*s). Outputs ROW-MAJOR [N × nb]:
/// cell (k,b) at k*nb + b.
__global__ void dstat_block_reduce_kernel(const double* __restrict__ Q,
                                          const double* __restrict__ V, int P, long M,
                                          const int* __restrict__ quad, int N,
                                          const int* __restrict__ block_begin,
                                          const int* __restrict__ block_size, int n_block,
                                          double* __restrict__ numsum,
                                          double* __restrict__ densum,
                                          double* __restrict__ cnt) {
    (void)M;  // the block layout (begin/size) bounds the SNP walk; M is the documented dim.
    const long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(N) * static_cast<long>(n_block);
    if (cell >= total) return;

    const int k = static_cast<int>(cell / n_block);  // quadruple
    const int b = static_cast<int>(cell % n_block);  // block

    const int p1 = quad[4 * k + 0];
    const int p2 = quad[4 * k + 1];
    const int p3 = quad[4 * k + 2];
    const int p4 = quad[4 * k + 3];

    const int s0 = block_begin[b];
    const int sz = block_size[b];

    double nsum = 0.0;
    double dsum = 0.0;
    double c = 0.0;
    for (int s = s0; s < s0 + sz; ++s) {
        const long col = static_cast<long>(P) * static_cast<long>(s);
        // allsnps=TRUE per-(block,quadruple) finiteness: a SNP is used iff V==1 for all 4.
        if (V[col + p1] == 0.0 || V[col + p2] == 0.0 ||
            V[col + p3] == 0.0 || V[col + p4] == 0.0) {
            continue;
        }
        const double a = Q[col + p1];
        const double bb = Q[col + p2];
        const double cc = Q[col + p3];
        const double dd = Q[col + p4];
        nsum += (a - bb) * (cc - dd);
        dsum += (a + bb - 2.0 * a * bb) * (cc + dd - 2.0 * cc * dd);
        c += 1.0;
    }

    const long out = static_cast<long>(k) * static_cast<long>(n_block) + b;
    numsum[out] = nsum;
    densum[out] = dsum;
    cnt[out] = c;
}

}  // namespace

void launch_dstat_block_reduce(const double* d_Q, const double* d_V, int P, long M,
                               const int* d_quad, int N,
                               const int* d_block_begin, const int* d_block_size,
                               int n_block,
                               double* d_numsum, double* d_densum, double* d_cnt,
                               cudaStream_t stream) {
    (void)M;  // forwarded to the kernel (documented dim); the block layout bounds the walk.
    const long total = static_cast<long>(N) * static_cast<long>(n_block);
    if (total <= 0) return;
    constexpr int kThreads = 256;
    const long blocks = (total + kThreads - 1) / kThreads;
    dstat_block_reduce_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        d_Q, d_V, P, M, d_quad, N, d_block_begin, d_block_size, n_block,
        d_numsum, d_densum, d_cnt);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace steppe::device
