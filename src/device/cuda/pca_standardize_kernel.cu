// src/device/cuda/pca_standardize_kernel.cu
//
// GPU kernels for standalone genotype PCA (`steppe pca`), plus the thin host launch
// wrappers. The Patterson center/scale fold (pca_snp_scale) and the standardize map
// (pca_standardize_one) are shared with the CPU oracle via core/internal/pca_standardize.hpp,
// so the GPU and reference paths cannot drift on the centering, the 1/sqrt(p(1-p)) scale, the
// missing->0 convention, or the monomorphic exclusion.
//
// Access pattern mirrors the proven fst/decode kernels: byte = s / kCodesPerByte, pos =
// s % kCodesPerByte, and each individual g's code at SNP s is packed[g*bpr + byte]. Unlike
// FST (which folds per-population segments) PCA folds over ALL N individuals per SNP; the
// standardized matrix Z is written COLUMN-MAJOR (sample the fast axis) so it is exactly the
// N x tileM operand cublasDsyrk consumes to accumulate the sample x sample covariance.
#include "device/cuda/pca_standardize_kernel.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/internal/decode_af.hpp"       // genotype_code, kCodesPerByte, genotype_valid
#include "core/internal/launch_config.hpp"   // kDecodeBlockX/Y, grid_for_x
#include "core/internal/pca_standardize.hpp"  // pca_snp_scale, pca_standardize_one, PcaSnpScale
#include "device/cuda/check.cuh"

namespace steppe::device {

using core::genotype_code;
using core::kDecodeBlockX;
using core::kDecodeBlockY;
using core::PcaSnpScale;

namespace {

constexpr int kPcaBlock = kDecodeBlockX * kDecodeBlockY;  // 256 threads / block

// One thread per SNP in the tile: fold all N diploid codes -> Patterson center/inv_scale.
__global__ void pca_snp_scale_kernel(const std::uint8_t* __restrict__ packed,
                                     std::size_t bytes_per_record, std::size_t N, long s_lo,
                                     long tileM, double* __restrict__ center,
                                     double* __restrict__ inv_scale,
                                     unsigned long long* __restrict__ used_count) {
    const long sl = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sl >= tileM) return;
    const long s = s_lo + sl;

    const std::size_t byte_in_record =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);

    long ac = 0;
    long nn = 0;
    for (std::size_t g = 0; g < N; ++g) {
        const std::uint8_t code = genotype_code(packed[g * bytes_per_record + byte_in_record],
                                                pos_in_byte);
        if (core::genotype_valid(code)) {
            ac += static_cast<long>(code);
            ++nn;
        }
    }

    const PcaSnpScale sc = core::pca_snp_scale(ac, nn);
    center[sl] = sc.center;
    inv_scale[sl] = sc.used ? (1.0 / sqrt(sc.pq)) : 0.0;
    if (sc.used) atomicAdd(used_count, 1ULL);
}

// One thread per (individual i, local SNP sl): write the column-major Z operand.
__global__ void pca_standardize_kernel(const std::uint8_t* __restrict__ packed,
                                       std::size_t bytes_per_record, std::size_t N, long s_lo,
                                       long tileM, const double* __restrict__ center,
                                       const double* __restrict__ inv_scale,
                                       double* __restrict__ Z) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(N) * tileM;
    if (t >= total) return;
    const long i = t % static_cast<long>(N);   // sample (fast axis => column-major leading dim)
    const long sl = t / static_cast<long>(N);  // local SNP (column)
    const long s = s_lo + sl;

    const std::size_t byte_in_record =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);
    const std::uint8_t code = genotype_code(
        packed[static_cast<std::size_t>(i) * bytes_per_record + byte_in_record], pos_in_byte);

    Z[t] = core::pca_standardize_one(code, center[sl], inv_scale[sl]);
}

// One thread per (sample i, PC k): coord[i*K + k] = evec[i + (ncol-1-k)*N] * sqrt(eval[ncol-1-k]).
// evec is a column-major N x ncol block (leading dim N); ascending eigenvalues put the largest
// pair in the last column, so PC k reads column ncol-1-k.
__global__ void pca_coords_kernel(const double* __restrict__ evec,
                                  const double* __restrict__ eval, int N, int ncol, int K,
                                  double* __restrict__ coords) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(N) * static_cast<long>(K);
    if (t >= total) return;
    const int i = static_cast<int>(t / static_cast<long>(K));
    const int k = static_cast<int>(t % static_cast<long>(K));
    const int col = ncol - 1 - k;  // ascending eigenvalues -> largest at the tail
    const double lambda = eval[col];
    const double scale = lambda > 0.0 ? sqrt(lambda) : 0.0;
    coords[static_cast<std::size_t>(i) * static_cast<std::size_t>(K) + static_cast<std::size_t>(k)] =
        evec[static_cast<std::size_t>(i) + static_cast<std::size_t>(col) * static_cast<std::size_t>(N)] *
        scale;
}

// splitmix64 finalizer (counter-based, stateless) — one 64-bit mix step.
__device__ __forceinline__ unsigned long long pca_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// One thread per element: two counter-based uniforms in (0,1) -> Box-Muller -> one N(0,1).
__global__ void pca_fill_omega_kernel(double* __restrict__ omega, std::size_t n_elem,
                                      unsigned long long seed) {
    const std::size_t t =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_elem) return;
    // Two independent 64-bit streams keyed on (seed, element index) — no shared state.
    const unsigned long long h0 = pca_splitmix64(seed ^ (t * 2ULL + 0ULL));
    const unsigned long long h1 = pca_splitmix64(seed ^ (t * 2ULL + 1ULL));
    // Map to (0,1): keep the top 53 bits, nudge off 0 so log() is finite.
    constexpr double kInv53 = 1.0 / 9007199254740992.0;  // 2^-53
    double u1 = (static_cast<double>(h0 >> 11) + 0.5) * kInv53;
    const double u2 = (static_cast<double>(h1 >> 11) + 0.5) * kInv53;
    if (u1 < 1e-300) u1 = 1e-300;
    const double r = sqrt(-2.0 * log(u1));
    const double theta = 6.283185307179586476925286766559 * u2;  // 2*pi
    omega[t] = r * cos(theta);
}

// Block reduction of the column-major diagonal C[i + i*N] into *out via one atomicAdd/block.
__global__ void pca_trace_kernel(const double* __restrict__ C, int N,
                                 double* __restrict__ out) {
    __shared__ double sdata[kPcaBlock];
    double local = 0.0;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < N;
         i += static_cast<long>(gridDim.x) * blockDim.x) {
        local += C[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) *
                                                    static_cast<std::size_t>(N)];
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

}  // namespace

void launch_pca_snp_scale(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                          std::size_t N, long s_lo, long tileM, double* d_center,
                          double* d_inv_scale, unsigned long long* d_used_count,
                          cudaStream_t stream) {
    if (tileM <= 0) return;
    const int grid = core::grid_for_x(
        tileM, kPcaBlock, "pca scale gridDim.x (SNP tile axis) exceeds kMaxGridX");
    pca_snp_scale_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_packed, bytes_per_record, N, s_lo, tileM, d_center, d_inv_scale, d_used_count);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_standardize(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                            std::size_t N, long s_lo, long tileM, const double* d_center,
                            const double* d_inv_scale, double* d_Z, cudaStream_t stream) {
    if (tileM <= 0 || N == 0) return;
    const long total = static_cast<long>(N) * tileM;
    const int grid = core::grid_for_x(
        total, kPcaBlock, "pca standardize gridDim.x (N*tileM axis) exceeds kMaxGridX");
    pca_standardize_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_packed, bytes_per_record, N, s_lo, tileM, d_center, d_inv_scale, d_Z);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_coords(const double* d_evec, const double* d_eval, int N, int ncol, int K,
                       double* d_coords, cudaStream_t stream) {
    if (N <= 0 || K <= 0 || ncol <= 0) return;
    const long total = static_cast<long>(N) * static_cast<long>(K);
    const int grid = core::grid_for_x(
        total, kPcaBlock, "pca coords gridDim.x (N*K axis) exceeds kMaxGridX");
    pca_coords_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_evec, d_eval, N, ncol, K, d_coords);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_fill_omega(double* d_omega, std::size_t n_elem, unsigned long long seed,
                           cudaStream_t stream) {
    if (n_elem == 0) return;
    const int grid = core::grid_for_x(
        static_cast<long>(n_elem), kPcaBlock,
        "pca omega gridDim.x (N*L axis) exceeds kMaxGridX");
    pca_fill_omega_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_omega, n_elem, seed);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_trace(const double* d_C, int N, double* d_out, cudaStream_t stream) {
    if (N <= 0) return;
    // A modest fixed grid; the block-stride loop covers any N and each block does one atomicAdd.
    int grid = static_cast<int>((static_cast<long>(N) + kPcaBlock - 1) / kPcaBlock);
    if (grid < 1) grid = 1;
    if (grid > 1024) grid = 1024;
    pca_trace_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(d_C, N, d_out);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
