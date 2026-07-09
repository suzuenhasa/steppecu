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

// -------- lsqproject kernels --------

// Fold only the reference rows (rows[r]) of one SNP tile -> Patterson center/inv_scale.
__global__ void pca_snp_scale_gather_kernel(const std::uint8_t* __restrict__ packed,
                                            std::size_t bytes_per_record,
                                            const int* __restrict__ rows, long n_rows, long s_lo,
                                            long tileM, double* __restrict__ center,
                                            double* __restrict__ inv_scale,
                                            unsigned long long* __restrict__ used_count) {
    const long sl = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (sl >= tileM) return;
    const long s = s_lo + sl;
    const std::size_t byte_in_record =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);

    long ac = 0, nn = 0;
    for (long r = 0; r < n_rows; ++r) {
        const std::size_t g = static_cast<std::size_t>(rows[r]);
        const std::uint8_t code =
            genotype_code(packed[g * bytes_per_record + byte_in_record], pos_in_byte);
        if (core::genotype_valid(code)) { ac += static_cast<long>(code); ++nn; }
    }
    const PcaSnpScale sc = core::pca_snp_scale(ac, nn);
    center[sl] = sc.center;
    inv_scale[sl] = sc.used ? (1.0 / sqrt(sc.pq)) : 0.0;
    if (sc.used) atomicAdd(used_count, 1ULL);
}

// Standardize a gathered row block into the column-major Z operand (n_rows x tileM).
__global__ void pca_standardize_gather_kernel(const std::uint8_t* __restrict__ packed,
                                              std::size_t bytes_per_record,
                                              const int* __restrict__ rows, long n_rows, long s_lo,
                                              long tileM, const double* __restrict__ center,
                                              const double* __restrict__ inv_scale,
                                              double* __restrict__ Z) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = n_rows * tileM;
    if (t >= total) return;
    const long r = t % n_rows;    // gathered sample (fast axis => column-major leading dim)
    const long sl = t / n_rows;   // local SNP (column)
    const long s = s_lo + sl;
    const std::size_t g = static_cast<std::size_t>(rows[r]);
    const std::size_t byte_in_record =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);
    const std::uint8_t code =
        genotype_code(packed[g * bytes_per_record + byte_in_record], pos_in_byte);
    Z[t] = core::pca_standardize_one(code, center[sl], inv_scale[sl]);
}

// Standardize the target block + emit the observed (ref-polymorphic, non-missing) mask.
__global__ void pca_standardize_target_kernel(const std::uint8_t* __restrict__ packed,
                                              std::size_t bytes_per_record,
                                              const int* __restrict__ rows, long n_rows, long s_lo,
                                              long tileM, const double* __restrict__ center,
                                              const double* __restrict__ inv_scale,
                                              double* __restrict__ Ztgt,
                                              std::uint8_t* __restrict__ mask) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = n_rows * tileM;
    if (t >= total) return;
    const long r = t % n_rows;
    const long sl = t / n_rows;
    const long s = s_lo + sl;
    const std::size_t g = static_cast<std::size_t>(rows[r]);
    const std::size_t byte_in_record =
        static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
    const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);
    const std::uint8_t code =
        genotype_code(packed[g * bytes_per_record + byte_in_record], pos_in_byte);
    const bool obs = core::genotype_valid(code) && (inv_scale[sl] != 0.0);
    Ztgt[t] = core::pca_standardize_one(code, center[sl], inv_scale[sl]);
    mask[t] = obs ? std::uint8_t{1} : std::uint8_t{0};
}

// Pack U (N_ref x K) from the ascending Ritz block (largest at the tail column L-1-k).
__global__ void pca_pack_basis_U_kernel(const double* __restrict__ evecL, int N_ref, int L, int K,
                                        double* __restrict__ U) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(N_ref) * static_cast<long>(K);
    if (t >= total) return;
    const int i = static_cast<int>(t % static_cast<long>(N_ref));
    const int k = static_cast<int>(t / static_cast<long>(N_ref));
    const int col = L - 1 - k;
    U[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * static_cast<std::size_t>(N_ref)] =
        evecL[static_cast<std::size_t>(i) +
              static_cast<std::size_t>(col) * static_cast<std::size_t>(N_ref)];
}

// inv_S[k] = 1/sqrt(max(eval[L-1-k], tiny)) — the loadings normalizer 1/S_k.
__global__ void pca_pack_basis_invS_kernel(const double* __restrict__ evalL, int L, int K,
                                           double* __restrict__ inv_S) {
    const int k = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= K) return;
    const double lambda = evalL[L - 1 - k];
    inv_S[k] = lambda > 0.0 ? (1.0 / sqrt(lambda)) : 0.0;
}

// W[s + k*tileM] *= inv_S[k].
__global__ void pca_scale_loadings_kernel(double* __restrict__ W, long tileM, int K,
                                          const double* __restrict__ inv_S) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = tileM * static_cast<long>(K);
    if (t >= total) return;
    const long s = t % tileM;
    const int k = static_cast<int>(t / tileM);
    W[static_cast<std::size_t>(s) + static_cast<std::size_t>(k) * static_cast<std::size_t>(tileM)] *=
        inv_S[k];
}

// A[j*K*K + k + kp*K] += Σ_s mask[j + s*N_tgt] W[s+k*tileM] W[s+kp*tileM].
__global__ void pca_accumulate_A_kernel(const double* __restrict__ W,
                                        const std::uint8_t* __restrict__ mask, long tileM,
                                        int N_tgt, int K, double* __restrict__ A) {
    const long t = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(N_tgt) * static_cast<long>(K) * static_cast<long>(K);
    if (t >= total) return;
    const int kp = static_cast<int>(t % static_cast<long>(K));
    const long rem = t / static_cast<long>(K);
    const int k = static_cast<int>(rem % static_cast<long>(K));
    const int j = static_cast<int>(rem / static_cast<long>(K));
    double acc = 0.0;
    for (long s = 0; s < tileM; ++s) {
        if (mask[static_cast<std::size_t>(j) + static_cast<std::size_t>(s) *
                                                   static_cast<std::size_t>(N_tgt)] == 0)
            continue;
        const double wk = W[static_cast<std::size_t>(s) +
                            static_cast<std::size_t>(k) * static_cast<std::size_t>(tileM)];
        const double wkp = W[static_cast<std::size_t>(s) +
                             static_cast<std::size_t>(kp) * static_cast<std::size_t>(tileM)];
        acc += wk * wkp;
    }
    A[static_cast<std::size_t>(j) * static_cast<std::size_t>(K) * static_cast<std::size_t>(K) +
      static_cast<std::size_t>(k) + static_cast<std::size_t>(kp) * static_cast<std::size_t>(K)] +=
        acc;
}

// m_obs[j] += Σ_s mask[j + s*N_tgt].
__global__ void pca_accumulate_mobs_kernel(const std::uint8_t* __restrict__ mask, long tileM,
                                           int N_tgt, unsigned long long* __restrict__ m_obs) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= N_tgt) return;
    unsigned long long c = 0;
    for (long s = 0; s < tileM; ++s)
        c += mask[static_cast<std::size_t>(j) +
                  static_cast<std::size_t>(s) * static_cast<std::size_t>(N_tgt)];
    m_obs[j] += c;
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

void launch_pca_snp_scale_gather(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                 const int* d_rows, long n_rows, long s_lo, long tileM,
                                 double* d_center, double* d_inv_scale,
                                 unsigned long long* d_used_count, cudaStream_t stream) {
    if (tileM <= 0 || n_rows <= 0) return;
    const int grid = core::grid_for_x(
        tileM, kPcaBlock, "pca scale(gather) gridDim.x (SNP tile axis) exceeds kMaxGridX");
    pca_snp_scale_gather_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_packed, bytes_per_record, d_rows, n_rows, s_lo, tileM, d_center, d_inv_scale,
        d_used_count);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_standardize_gather(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                   const int* d_rows, long n_rows, long s_lo, long tileM,
                                   const double* d_center, const double* d_inv_scale, double* d_Z,
                                   cudaStream_t stream) {
    if (tileM <= 0 || n_rows <= 0) return;
    const long total = n_rows * tileM;
    const int grid = core::grid_for_x(
        total, kPcaBlock, "pca standardize(gather) gridDim.x (n_rows*tileM axis) exceeds kMaxGridX");
    pca_standardize_gather_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_packed, bytes_per_record, d_rows, n_rows, s_lo, tileM, d_center, d_inv_scale, d_Z);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_standardize_target(const std::uint8_t* d_packed, std::size_t bytes_per_record,
                                   const int* d_rows, long n_rows, long s_lo, long tileM,
                                   const double* d_center, const double* d_inv_scale,
                                   double* d_Ztgt, std::uint8_t* d_mask, cudaStream_t stream) {
    if (tileM <= 0 || n_rows <= 0) return;
    const long total = n_rows * tileM;
    const int grid = core::grid_for_x(
        total, kPcaBlock, "pca standardize(target) gridDim.x (n_rows*tileM axis) exceeds kMaxGridX");
    pca_standardize_target_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_packed, bytes_per_record, d_rows, n_rows, s_lo, tileM, d_center, d_inv_scale, d_Ztgt,
        d_mask);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_pack_basis(const double* d_evecL, const double* d_evalL, int N_ref, int L, int K,
                           double* d_U, double* d_inv_S, cudaStream_t stream) {
    if (N_ref <= 0 || K <= 0 || L <= 0) return;
    const long total = static_cast<long>(N_ref) * static_cast<long>(K);
    const int grid = core::grid_for_x(
        total, kPcaBlock, "pca pack_basis gridDim.x (N_ref*K axis) exceeds kMaxGridX");
    pca_pack_basis_U_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_evecL, N_ref, L, K, d_U);
    STEPPE_CUDA_CHECK_KERNEL();
    const int gridK = core::grid_for_x(
        static_cast<long>(K), kPcaBlock, "pca pack_basis invS gridDim.x (K axis) exceeds kMaxGridX");
    pca_pack_basis_invS_kernel<<<static_cast<unsigned>(gridK), kPcaBlock, 0, stream>>>(
        d_evalL, L, K, d_inv_S);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_scale_loadings(double* d_W, long tileM, int K, const double* d_inv_S,
                               cudaStream_t stream) {
    if (tileM <= 0 || K <= 0) return;
    const long total = tileM * static_cast<long>(K);
    const int grid = core::grid_for_x(
        total, kPcaBlock, "pca scale_loadings gridDim.x (tileM*K axis) exceeds kMaxGridX");
    pca_scale_loadings_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_W, tileM, K, d_inv_S);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_accumulate_A(const double* d_W, const std::uint8_t* d_mask, long tileM, int N_tgt,
                             int K, double* d_A, cudaStream_t stream) {
    if (tileM <= 0 || N_tgt <= 0 || K <= 0) return;
    const long total = static_cast<long>(N_tgt) * static_cast<long>(K) * static_cast<long>(K);
    const int grid = core::grid_for_x(
        total, kPcaBlock, "pca accumulate_A gridDim.x (N_tgt*K*K axis) exceeds kMaxGridX");
    pca_accumulate_A_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_W, d_mask, tileM, N_tgt, K, d_A);
    STEPPE_CUDA_CHECK_KERNEL();
}

void launch_pca_accumulate_mobs(const std::uint8_t* d_mask, long tileM, int N_tgt,
                                unsigned long long* d_m_obs, cudaStream_t stream) {
    if (tileM <= 0 || N_tgt <= 0) return;
    const int grid = core::grid_for_x(
        static_cast<long>(N_tgt), kPcaBlock,
        "pca accumulate_mobs gridDim.x (N_tgt axis) exceeds kMaxGridX");
    pca_accumulate_mobs_kernel<<<static_cast<unsigned>(grid), kPcaBlock, 0, stream>>>(
        d_mask, tileM, N_tgt, d_m_obs);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
