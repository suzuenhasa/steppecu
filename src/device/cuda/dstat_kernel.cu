// src/device/cuda/dstat_kernel.cu
//
// The GPU genotype-path NORMALIZED-D per-SNP reduction — qpDstat Part B (the S2
// divergence; include/steppe/dstat.hpp) AND the qpfstats genotype-f4 numerator reduce
// (the ~305k-combo wall). NEVER the f2 GEMM, NEVER the f2 cache: a segmented reduction
// over each jackknife block's SNP columns, accumulating the AT2 qpdstat_geno num/den per
// (combo, block) over the allsnps=TRUE per-(block,combo) finiteness mask.
//
// STATISTIC (AT2 qpdstat_geno, allsnps=TRUE, f4mode=FALSE; pinned to the AT2 R golden):
// for combo (p1,p2,p3,p4) at SNP s with a=Q[p1,s], b=Q[p2,s], c=Q[p3,s], d=Q[p4,s],
//   num = (a-b)*(c-d),  den = (a+b-2ab)*(c+d-2cd)
// summed per block ONLY over SNPs valid in all 4 pops (V==1 for p1..p4).
//
// THE PERFORMANCE FIX — SNP-TILED PAIRWISE-DIFFERENCE REUSE (the shared kernel):
// the legacy mapping was one-thread-per-(combo,block), each thread re-reading its 4 pops'
// Q/V down the block at stride P -> ZERO reuse across the ~305k combos -> ~11.6 TB of
// uncoalesced global loads, memory-bound at ~8% of peak. Every numerator (Q[a]-Q[b])·
// (Q[c]-Q[d]) uses only C(P,2) distinct per-SNP differences. The tiled kernel: one
// grid-block per jackknife block (gridDim.x = n_block) so the whole [begin,begin+size)
// range is owned by one grid-block and accumulation stays ascending-s within the block
// (NO cross-tile reduction); gridDim.y = ceil(N/T) combo-tiles, T = blockDim.x threads.
// Per SNP the block's T threads COOPERATIVELY load Q[P·s+i]/V[P·s+i] for i in [0,P) into
// shared memory ONCE (one coalesced contiguous-column read, reused by all T combos), then
// fill the C(P,2) shared diff[i<j]=Q[i]-Q[j] and het[i<j]=Q[i]+Q[j]-2·Q[i]·Q[j] arrays;
// each thread then reconstructs its combo's num/den from the shared pairwise tables and a
// per-pop V-mask. One coalesced pass of Q+V (~191 MB each) per combo-tile -> compute-bound.
//
// GOLDEN-EXACT (the §12 cancellation carve-out; NATIVE FP64, never emulated):
//   - PRODUCT identity: f1 = (p1<p2 ? +diff[pair(p1,p2)] : -diff[pair(p2,p1)]). In IEEE-754
//     negation is exact and subtraction is symmetric so -(Q[p2]-Q[p1]) == Q[p1]-Q[p2]
//     bit-for-bit -> f1·f2 is bit-identical to the legacy (a-b)·(c-d). den uses the
//     SYMMETRIC het[ij] -> direct lookup, bit-identical to (a+b-2ab)·(c+d-2cd).
//   - MASK identity: mask = V[p1]·V[p2]·V[p3]·V[p4] != 0 == the legacy 4-pop joint V==1 test.
//   - ORDER: each (combo,block) sum is owned by ONE grid-block (gridDim.x=b) and one
//     register, summed in ASCENDING s within [begin,begin+size) -> per-(combo,b) sums are
//     BIT-IDENTICAL to the legacy kernel, not merely within rtol.
//
// HUGE-npop FALLBACK: when 2·C(P,2)+2·P doubles no longer fit the 99 KB opt-in shared budget
// (P > ~112 with den), launch_dstat_block_reduce falls back to the RETAINED legacy
// per-(combo,block) kernel (same TU, same wrapper — a runtime branch, NOT a forked path).
// steppe's resident tier OOMs past ~250 pops anyway, so the fallback band is ~112..250.
//
// DESIGN-FOR-SCALE: batched over BOTH the N-combo axis and the n_block axis; reads the
// resident Q/V [P × M] in VRAM. The output numsum/densum/cnt are [N × n_block] (tiny).
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). The kernel body +
// <<<>>> live ONLY here; the backend reaches it through the narrow launch wrapper
// (dstat_kernel.cuh), never includes this body (architecture.md §7).
#include <cuda_runtime.h>

#include <cstddef>                         // std::size_t

#include "core/internal/launch_config.hpp"  // grid_for, kMaxGridY (the launch-config home)
#include "device/cuda/check.cuh"          // STEPPE_CUDA_CHECK
#include "device/cuda/dstat_kernel.cuh"   // launch_dstat_block_reduce (the narrow seam)

namespace steppe::device {

namespace {

/// Row-major upper-triangle pair index for an off-diagonal pop pair (i,j), i<j, over the
/// C(P,2) pairs — EXACTLY the host pair_index (src/core/stats/qpfstats.cpp) and AT2 indmat order.
/// Symmetric in (i,j); callers pass the SMALLER index first.
__device__ __forceinline__ int pair_index_lo_hi(int lo, int hi, int P) {
    return lo * P - (lo * (lo + 1)) / 2 + (hi - lo - 1);
}

/// One thread per (combo k, block b) output cell — the RETAINED legacy reduction (the
/// cold huge-npop branch, P above the shared-memory budget). Walks the block's contiguous
/// SNP columns and accumulates the AT2 num/den over the 4-pop finiteness mask. Q/V are
/// column-major [P × M] (element (pop i, SNP s) at i + P*s). Outputs ROW-MAJOR [N × nb]:
/// cell (k,b) at k*nb + b. This is BYTE-IDENTICAL to the historical kernel — automatically
/// golden-safe for the >threshold band.
__global__ void dstat_block_reduce_legacy_kernel(
    const double* __restrict__ Q, const double* __restrict__ V, int P, long M,
    const int* __restrict__ quad, int N, const int* __restrict__ block_begin,
    const int* __restrict__ block_size, int n_block, double* __restrict__ numsum,
    double* __restrict__ densum, double* __restrict__ cnt) {
    (void)M;  // the block layout (begin/size) bounds the SNP walk; M is the documented dim.
    const long cell = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x;
    const long total = static_cast<long>(N) * static_cast<long>(n_block);
    if (cell >= total) return;

    const int k = static_cast<int>(cell / n_block);  // combo
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
        // allsnps=TRUE per-(block,combo) finiteness: a SNP is used iff V==1 for all 4.
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

/// SNP-TILED PAIRWISE-DIFFERENCE-REUSE reduction (the hot path; the ~305k-combo win). One
/// grid-block per jackknife block (gridDim.x = n_block) so accumulation stays ascending-s
/// within [begin,begin+size) and needs NO cross-tile reduction; gridDim.y = combo-tiles.
/// Each thread owns ONE combo (k = blockIdx.y*blockDim.x + threadIdx.x) and one block
/// (b = blockIdx.x), in three persistent FP64 register accumulators. Per SNP the block
/// cooperatively loads Q/V for all P pops into shared (ONE coalesced read, reused by all
/// threads), forms the C(P,2) shared diff/het tables once, then each thread reconstructs
/// its combo's per-SNP num/den from the shared tables + a per-pop V-mask. The diff/het are
/// formed ONCE per SNP and reused by all blockDim.x combos -> compute-bound.
///
/// Dynamic shared layout (one extern double[] block, npairs = C(P,2)):
///   [0          .. P)            Qsh[i]   = Q[P*s + i]
///   [P          .. 2P)           Vsh[i]   = V[P*s + i]
///   [2P         .. 2P+npairs)    diff[ij] = Qsh[i]-Qsh[j]  (i<j)
///   [2P+npairs  .. 2P+2*npairs)  het[ij]  = Qsh[i]+Qsh[j]-2*Qsh[i]*Qsh[j]  (i<j, symmetric)
__global__ void dstat_block_reduce_tiled_kernel(
    const double* __restrict__ Q, const double* __restrict__ V, int P, long M,
    const int* __restrict__ quad, int N, const int* __restrict__ block_begin,
    const int* __restrict__ block_size, int n_block, double* __restrict__ numsum,
    double* __restrict__ densum, double* __restrict__ cnt) {
    (void)M;  // the block layout (begin/size) bounds the SNP walk; M is the documented dim.

    const int b = static_cast<int>(blockIdx.x);                       // jackknife block
    const long k = blockIdx.y * static_cast<long>(blockDim.x) + threadIdx.x;  // combo

    const int npairs = P * (P - 1) / 2;

    extern __shared__ double s_mem[];
    double* Qsh = s_mem;             // [P]
    double* Vsh = s_mem + P;         // [P]
    double* diff = s_mem + 2 * P;    // [npairs]
    double* het = diff + npairs;     // [npairs]

    // This thread's combo (may be OOB in the last combo-tile — it still cooperates in the
    // shared-memory loads/fills below; only its private accumulate + the final write are
    // guarded by `active`). All P pops are loaded regardless of which combos are queried,
    // so qpDstat's arbitrary combos look up the same shared tables.
    const bool active = (k < static_cast<long>(N));
    int p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    if (active) {
        p1 = quad[4 * k + 0];
        p2 = quad[4 * k + 1];
        p3 = quad[4 * k + 2];
        p4 = quad[4 * k + 3];
    }

    const int s0 = block_begin[b];
    const int sz = block_size[b];
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    double nsum = 0.0;
    double dsum = 0.0;
    double c = 0.0;

    for (int s = s0; s < s0 + sz; ++s) {
        const long col = static_cast<long>(P) * static_cast<long>(s);

        // (a) Cooperative coalesced load of the current SNP column (P pops) into shared.
        for (int i = tid; i < P; i += nthreads) {
            Qsh[i] = Q[col + i];
            Vsh[i] = V[col + i];
        }
        __syncthreads();

        // (b) Cooperative fill of the C(P,2) diff + het tables (each pair once). diff and
        // het use the SAME subtraction/expression the legacy kernel does, so the
        // per-combo reconstruction below is bit-identical.
        for (int idx = tid; idx < npairs; idx += nthreads) {
            // Decode the flat upper-triangle index idx -> (i,j), i<j (row-major upper-tri).
            int i = 0;
            int rem = idx;
            int row = P - 1;  // entries in row 0
            while (rem >= row) { rem -= row; ++i; --row; }
            const int j = i + 1 + rem;
            const double qi = Qsh[i];
            const double qj = Qsh[j];
            diff[idx] = qi - qj;                          // Q[i]-Q[j], i<j
            het[idx] = qi + qj - 2.0 * qi * qj;           // symmetric in (i,j)
        }
        __syncthreads();

        // (c) Per-combo reconstruction from the shared tables + per-pop V mask.
        if (active) {
            if (Vsh[p1] != 0.0 && Vsh[p2] != 0.0 && Vsh[p3] != 0.0 && Vsh[p4] != 0.0) {
                // f1 = Q[p1]-Q[p2] reconstructed via the canonical-order diff (sign exact).
                double f1;
                if (p1 < p2)       f1 = diff[pair_index_lo_hi(p1, p2, P)];
                else if (p1 > p2)  f1 = -diff[pair_index_lo_hi(p2, p1, P)];
                else               f1 = 0.0;  // p1==p2 (arbitrary qpDstat combo): a-a == 0
                double f2;
                if (p3 < p4)       f2 = diff[pair_index_lo_hi(p3, p4, P)];
                else if (p3 > p4)  f2 = -diff[pair_index_lo_hi(p4, p3, P)];
                else               f2 = 0.0;  // p3==p4
                nsum += f1 * f2;
                // den: het is symmetric; look up with the smaller index first (p_i!=p_j
                // for any real combo; on p_i==p_j the het factor is q+q-2qq, supplied
                // directly below to stay bit-faithful to the legacy expression).
                double h1;
                if (p1 != p2) h1 = het[pair_index_lo_hi(min(p1, p2), max(p1, p2), P)];
                else { const double q = Qsh[p1]; h1 = q + q - 2.0 * q * q; }
                double h2;
                if (p3 != p4) h2 = het[pair_index_lo_hi(min(p3, p4), max(p3, p4), P)];
                else { const double q = Qsh[p3]; h2 = q + q - 2.0 * q * q; }
                dsum += h1 * h2;
                c += 1.0;
            }
        }
        __syncthreads();  // protect the shared tables before the next SNP overwrites them.
    }

    if (active) {
        const long out = k * static_cast<long>(n_block) + b;
        numsum[out] = nsum;
        densum[out] = dsum;
        cnt[out] = c;
    }
}

}  // namespace

void launch_dstat_block_reduce(const double* d_Q, const double* d_V, int P, long M,
                               const int* d_quad, int N,
                               const int* d_block_begin, const int* d_block_size,
                               int n_block,
                               double* d_numsum, double* d_densum, double* d_cnt,
                               cudaStream_t stream) {
    const long total = static_cast<long>(N) * static_cast<long>(n_block);
    if (total <= 0) return;
    constexpr int kThreads = 256;

    // Shared-memory budget for the tiled path: Qsh[P]+Vsh[P]+diff[C(P,2)]+het[C(P,2)] doubles.
    const long npairs = static_cast<long>(P) * (P - 1) / 2;
    const std::size_t smem = static_cast<std::size_t>(2L * P + 2L * npairs) * sizeof(double);

    // The opt-in shared cap on sm_120 / CUDA 13 (RTX 5090) is 99 KB; the 48 KB default
    // needs no opt-in. We engage the opt-in only when smem > 48 KB (the 78<P<=112 band).
    constexpr std::size_t kDefaultSmem = 48u * 1024u;   // 49152 (cudaDevAttr default)
    constexpr std::size_t kOptinSmem = 99u * 1024u;     // 101376 (sharedMemPerBlockOptin)

    if (P >= 2 && smem <= kOptinSmem) {
        // ---- The HOT tiled diff/het-reuse path (qpfstats ~305k combos; qpDstat too). ----
        if (smem > kDefaultSmem) {
            // Opt-in to >48 KB dynamic shared (CUDA 13 / sm_120: returns no error up to
            // sharedMemPerBlockOptin=99 KB). Idempotent per (kernel,attr); cheap to set.
            STEPPE_CUDA_CHECK(cudaFuncSetAttribute(
                dstat_block_reduce_tiled_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem)));
        }
        // The combo/scale axis N rides the CAPPED gridDim.y (max kMaxGridY = 65 535);
        // route it through the single launch-config home so an over-limit extent
        // fails-fast in debug (the X-7/B6 grid_for assert) instead of an opaque
        // cudaErrorInvalidConfiguration at launch — exactly as launch_f2_feeder routes
        // P. grid_for(N, kThreads) is the same cdiv value as the prior open-coded form,
        // so the launch geometry is unchanged (architecture.md §7; STANDARD §3.3).
        const unsigned tilesY = static_cast<unsigned>(core::grid_for(N, kThreads));
        const dim3 grid(static_cast<unsigned>(n_block), tilesY, 1);
        dstat_block_reduce_tiled_kernel<<<grid, kThreads, smem, stream>>>(
            d_Q, d_V, P, M, d_quad, N, d_block_begin, d_block_size, n_block,
            d_numsum, d_densum, d_cnt);
        STEPPE_CUDA_CHECK(cudaGetLastError());
        return;
    }

    // ---- The COLD large-npop fallback: the RETAINED legacy per-(combo,block) kernel
    // (P > ~112: C(P,2) no longer fits the 99 KB opt-in shared budget). Byte-identical
    // to the historical reduction -> automatically golden-safe. ----
    const long blocks = (total + kThreads - 1) / kThreads;
    dstat_block_reduce_legacy_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        d_Q, d_V, P, M, d_quad, N, d_block_begin, d_block_size, n_block,
        d_numsum, d_densum, d_cnt);
    STEPPE_CUDA_CHECK(cudaGetLastError());
}

}  // namespace steppe::device
