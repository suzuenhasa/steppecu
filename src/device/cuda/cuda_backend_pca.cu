// src/device/cuda/cuda_backend_pca.cu
//
// CudaBackend override for standalone genotype PCA (`steppe pca`). It uploads the packed
// genotype tile device-resident, SNP-tiles the Patterson-standardize kernel + cuBLAS SYRK
// to accumulate the sample x sample covariance C = Z*Z^T entirely on the GPU (emulated-FP64
// default, the matmul-heavy path — copied from the fit engine's SYRK block), symmetrizes it,
// eigendecomposes it with cuSOLVER Dsyevd (native FP64 — the cancellation/eigen carve-out),
// projects the top-K eigenvectors to sample coordinates, and D2H's ONLY the small N*K coords
// + the eigen spectrum + the used-SNP counter. The covariance/eigen compute is all on the
// GPU (no host genotype loop). A CUDA TU private to steppe_device, mirroring
// cuda_backend_fst.cu / cuda_backend_sfs.cu.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/f2_block_kernel.cuh"      // engage_f2_precision, emulation_honorable
#include "device/cuda/pca_standardize_kernel.cuh"
#include "device/cuda/qpadm_fit_kernels.cuh"    // launch_symmetrize_lower_to_full

namespace steppe::device {

namespace {
// Fraction of free VRAM the standardized Z operand tile may occupy (leaves room for the
// packed tile, the N x N covariance, and the eigensolver workspace).
constexpr std::size_t kPcaZBudgetDivisor = 4;
constexpr std::size_t kPcaFreeVramFallbackBytes = std::size_t{1} << 30;  // 1 GiB
}  // namespace

PcaEig CudaBackend::pca_covariance_eig(const DecodeTileView& tile, int k,
                                       const Precision& precision) {
    guard_device();
    PcaEig out;
    out.precision_tag =
        (precision.kind == Precision::Kind::EmulatedFp64 &&
         capabilities().emulated_fp64_honorable)
            ? Precision::Kind::EmulatedFp64
            : Precision::Kind::Fp64;

    const long N = static_cast<long>(tile.n_individuals);
    const long M = static_cast<long>(tile.n_snp);
    if (N <= 0 || M <= 0 || k <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }
    const int Nn = static_cast<int>(N);
    const int K = static_cast<int>(std::min<long>(k, N));

    // Upload the whole packed tile device-resident (individual-major).
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
    if (packed_bytes > 0) {
        h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());
    }

    // Sample x sample covariance accumulator (column-major N x N), zeroed for beta=0/1 SYRK.
    const std::size_t NN = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    DeviceBuffer<double> dC(NN);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dC.data(), 0, NN * sizeof(double), stream_.get()));

    DeviceBuffer<unsigned long long> dUsed(1);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dUsed.data(), 0, sizeof(unsigned long long), stream_.get()));

    // Choose a SNP tile that bounds the Z operand footprint to ~free/kPcaZBudgetDivisor,
    // keeping the standardize+SYRK memory O(N*tileM + N^2) (the extract_f2 tiling idiom).
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kPcaFreeVramFallbackBytes;
    std::size_t budget = free_b / kPcaZBudgetDivisor;
    const std::size_t z_row_bytes = static_cast<std::size_t>(N) * sizeof(double);
    if (budget < z_row_bytes) budget = z_row_bytes;
    long tileM = static_cast<long>(budget / z_row_bytes);
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;

    DeviceBuffer<double> dCenter(static_cast<std::size_t>(tileM));
    DeviceBuffer<double> dInv(static_cast<std::size_t>(tileM));
    DeviceBuffer<double> dZ(static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM));

    {
        // Covariance SYRK: emulated-FP64 default (matmul-heavy), the fit-engine block.
        const MathModeScope syrk_mode_scope(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            launch_pca_snp_scale(dPacked.data(), tile.bytes_per_record,
                                 tile.n_individuals, s_lo, tm, dCenter.data(),
                                 dInv.data(), dUsed.data(), stream_.get());
            launch_pca_standardize(dPacked.data(), tile.bytes_per_record,
                                   tile.n_individuals, s_lo, tm, dCenter.data(),
                                   dInv.data(), dZ.data(), stream_.get());
            const double alpha = 1.0;
            const double beta = (s_lo == 0) ? 0.0 : 1.0;
            CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                     Nn, static_cast<int>(tm), &alpha, dZ.data(), Nn,
                                     &beta, dC.data(), Nn));
        }
    }
    launch_symmetrize_lower_to_full(dC.data(), Nn, stream_.get());

    // Eigendecomposition (native FP64): C = V diag(W) V^T, ascending W, V overwrites dC.
    solver_.set_stream(stream_.get());
    const CusolverMathModeScope eig_scope(solver_.get(), /*honorable=*/false);
    DeviceBuffer<double> dW(static_cast<std::size_t>(N));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver_.get(), CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, Nn, dC.data(), Nn,
                                               dW.data(), &lwork));
    DeviceBuffer<double> dWork(static_cast<std::size_t>(lwork > 0 ? lwork : 1));
    DeviceBuffer<int> dInfo(1);
    CUSOLVER_CHECK(cusolverDnDsyevd(solver_.get(), CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER, Nn, dC.data(), Nn, dW.data(),
                                    dWork.data(), lwork, dInfo.data()));
    int info = 0;
    d2h_async(&info, dInfo, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) {
        out.status = Status::NonSpdCovariance;
        return out;
    }

    // Project the top-K eigenvectors to sample PC coordinates (coord = evec*sqrt(eval)).
    DeviceBuffer<double> dCoords(static_cast<std::size_t>(N) * static_cast<std::size_t>(K));
    launch_pca_coords(dC.data(), dW.data(), Nn, K, dCoords.data(), stream_.get());

    // D2H only the small results: coords, the full eigenvalue vector, the used counter.
    out.coords.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(K), 0.0);
    std::vector<double> w_all(static_cast<std::size_t>(N), 0.0);
    unsigned long long used_h = 0;
    d2h_async(out.coords.data(), dCoords,
              static_cast<std::size_t>(N) * static_cast<std::size_t>(K), stream_.get());
    d2h_async(w_all.data(), dW, static_cast<std::size_t>(N), stream_.get());
    d2h_async(&used_h, dUsed, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    long double total = 0.0L;
    for (double v : w_all) total += static_cast<long double>(v);
    out.eigenvalues.assign(static_cast<std::size_t>(K), 0.0);
    out.var_explained.assign(static_cast<std::size_t>(K), 0.0);
    for (int kk = 0; kk < K; ++kk) {
        const double lambda = w_all[static_cast<std::size_t>(N - 1 - kk)];
        out.eigenvalues[static_cast<std::size_t>(kk)] = lambda;
        out.var_explained[static_cast<std::size_t>(kk)] =
            (total != 0.0L) ? static_cast<double>(static_cast<long double>(lambda) / total) : 0.0;
    }

    out.N = Nn;
    out.K = K;
    out.n_snp_used = static_cast<long>(used_h);
    out.n_snp_monomorphic = M - out.n_snp_used;
    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
