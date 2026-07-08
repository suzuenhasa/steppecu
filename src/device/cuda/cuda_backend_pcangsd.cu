// src/device/cuda/cuda_backend_pcangsd.cu
//
// CudaBackend override for PCAngsd genotype-likelihood PCA (`steppe pcangsd`). It
// uploads the host GL tile to a resident LikelihoodTensor (residency-checksummed
// host<->device sum), runs the individual-allele-frequency EM device-resident
// (emMAF over the tensor -> updateNormal init -> updatePCAngsd loop -> covPCAngsd),
// and reuses the cuBLAS SYRK gram + cuSOLVER Dsyevd eigen + launch_pca_coords tail
// from the `steppe pca` path. The elementwise EM (E-build, dCov, emMAF) is native
// FP64 (the GL cancellation carve-out); the E^T E / V V^T / reconstruction matmuls
// run emulated-FP64 (the matmul-heavy default, via engage_f2_precision); every
// cuSOLVER eigen runs native FP64. Only the small N*N cov + N*e coords + spectrum +
// (optional) freq/pi cross PCIe. A CUDA TU private to steppe_device, mirroring
// cuda_backend_pca.cu.
//
// Math reference: core::pcangsd_reference (pcangsd_em.hpp) — the GPU reproduces it.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/f2_block_kernel.cuh"      // engage_f2_precision
#include "device/cuda/likelihood_tensor_impl.cuh"
#include "device/cuda/pca_standardize_kernel.cuh"  // launch_pca_coords
#include "device/cuda/pcangsd_kernels.cuh"
#include "device/cuda/qpadm_fit_kernels.cuh"     // launch_symmetrize_lower_to_full

namespace steppe::device {

PcangsdFit CudaBackend::pcangsd_fit(const double* host_l, const std::uint8_t* host_present,
                                    long n_site, int n_sample, int e, int max_iter, double tol,
                                    double maf, int maf_iter, double maf_tol, bool want_pi,
                                    const Precision& precision) {
    guard_device();
    PcangsdFit out;
    out.M_total = n_site;
    out.precision_tag = (precision.kind == Precision::Kind::EmulatedFp64 &&
                         capabilities().emulated_fp64_honorable)
                            ? Precision::Kind::EmulatedFp64
                            : Precision::Kind::Fp64;
    const int N = n_sample;
    if (n_site <= 0 || N <= 0 || e <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }

    // --- resident GL tensor + host<->device residency proof ------------------
    LikelihoodTensor t = upload_likelihood_tensor(host_l, host_present, n_site, N);
    {
        const long n_elem = n_site * static_cast<long>(N) * 3;
        double host_sum = 0.0;
        for (long i = 0; i < n_elem; ++i) host_sum += host_l[i];
        const double dev_sum = likelihood_tensor_checksum(t);
        std::fprintf(stderr,
                     "steppe pcangsd: GL tensor resident on device %d — checksum host=%.6f "
                     "device=%.6f (|diff|=%.3e)\n",
                     device_id_, host_sum, dev_sum, std::fabs(host_sum - dev_sum));
    }
    const double* d_l = t.impl->l.data();

    // --- emMAF over ALL sites (device), then host-side MAF filter -------------
    DeviceBuffer<double> d_f_all(static_cast<std::size_t>(n_site));
    launch_pcangsd_emmaf(d_l, n_site, N, maf_iter, maf_tol, d_f_all.data(), stream_.get());
    std::vector<double> f_all(static_cast<std::size_t>(n_site), 0.0);
    d2h_async(f_all.data(), d_f_all, static_cast<std::size_t>(n_site), stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    std::vector<int> kept;
    std::vector<double> fk;
    kept.reserve(static_cast<std::size_t>(n_site));
    fk.reserve(static_cast<std::size_t>(n_site));
    for (long j = 0; j < n_site; ++j) {
        const double fj = f_all[static_cast<std::size_t>(j)];
        if (std::min(fj, 1.0 - fj) >= maf) {
            kept.push_back(static_cast<int>(j));
            fk.push_back(fj);
        }
    }
    const long Mw = static_cast<long>(kept.size());
    out.M_used = Mw;
    if (Mw <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }
    const int K = static_cast<int>(std::min<long>(e, N));
    out.freq = fk;

    DeviceBuffer<int> d_kept(static_cast<std::size_t>(Mw));
    DeviceBuffer<double> d_fk(static_cast<std::size_t>(Mw));
    h2d_async(d_kept, kept.data(), static_cast<std::size_t>(Mw), stream_.get());
    h2d_async(d_fk, fk.data(), static_cast<std::size_t>(Mw), stream_.get());

    const std::size_t NMw = static_cast<std::size_t>(N) * static_cast<std::size_t>(Mw);
    const std::size_t NN = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    DeviceBuffer<double> dE(NMw), dP(NMw), dPprev(NMw);
    DeviceBuffer<double> dG(NN), dW(NN), dwG(static_cast<std::size_t>(N));
    DeviceBuffer<int> dInfo(1);

    solver_.set_stream(stream_.get());
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(solver_.get(), CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_LOWER, N, dG.data(), N, dwG.data(),
                                               &lwork));
    DeviceBuffer<double> dWork(static_cast<std::size_t>(lwork > 0 ? lwork : 1));

    // gram_reconstruct: dG = dE dE^T (emulated SYRK, lower) -> eigvecs (native
    // Dsyevd, dG overwritten) -> dW = Vk Vk^T over the top-K columns (SYRK + full
    // symmetrize) -> dP = dW dE (emulated GEMM). Returns false on a bad eigensolve.
    auto gram_reconstruct = [&]() -> bool {
        {
            const MathModeScope syrk_mode(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            const double a = 1.0, b = 0.0;
            CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N,
                                     static_cast<int>(Mw), &a, dE.data(), N, &b, dG.data(), N));
        }
        {
            const CusolverMathModeScope eig_scope(solver_.get(), /*honorable=*/false);
            CUSOLVER_CHECK(cusolverDnDsyevd(solver_.get(), CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_LOWER, N, dG.data(), N, dwG.data(),
                                            dWork.data(), lwork, dInfo.data()));
        }
        int info = 0;
        d2h_async(&info, dInfo, 1, stream_.get());
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        if (info != 0) return false;
        {
            // Top-K eigenvectors = the last K columns of dG (ascending eigenvalues).
            const double* Vk = dG.data() + static_cast<std::size_t>(N - K) * static_cast<std::size_t>(N);
            const MathModeScope syrk_mode(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            const double a = 1.0, b = 0.0;
            CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K, &a, Vk,
                                     N, &b, dW.data(), N));
        }
        launch_symmetrize_lower_to_full(dW.data(), N, stream_.get());
        {
            const MathModeScope gemm_mode(blas_.get(), CUBLAS_PEDANTIC_MATH);
            engage_f2_precision(blas_.get(), precision);
            const double a = 1.0, b = 0.0;
            CUBLAS_CHECK(cublasDgemm(blas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, N,
                                     static_cast<int>(Mw), N, &a, dW.data(), N, dE.data(), N, &b,
                                     dP.data(), N));
        }
        return true;
    };

    // --- init: updateNormal (weights + centering by f) then estimatePi -------
    launch_pcangsd_build_E(d_l, d_kept.data(), d_fk.data(), /*P=*/nullptr, N, Mw, N,
                           /*standardize=*/false, dE.data(), stream_.get());
    if (!gram_reconstruct()) {
        out.status = Status::NonSpdCovariance;
        return out;
    }

    // --- main loop: updatePCAngsd + estimatePi to rmse2d(pi) < tol -----------
    // This is a PLAIN, unaccelerated fixed-point EM (the reference pcangsd wraps the
    // same update in a SQUAREM/quasi-Newton accelerator, so it reaches a given tol in
    // ~1/8 the iterations). Both converge to the SAME fixed point — the covariance /
    // PCs / IAF agree — but at the default tol=1e-5 this loop commonly runs to max_iter.
    // The precision knob does NOT change the trajectory: emulated-FP64 and native FP64
    // produce an identical rmse path (the slow tail is linear EM convergence, not a
    // reconstruction noise floor). SQUAREM acceleration is a documented follow-up.
    DeviceBuffer<double> d_acc(1);
    double rmse = 0.0;
    int iters = 0;
    for (int it = 0; it < max_iter; ++it) {
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dPprev.data(), dP.data(), NMw * sizeof(double),
                                          cudaMemcpyDeviceToDevice, stream_.get()));
        launch_pcangsd_build_E(d_l, d_kept.data(), d_fk.data(), dP.data(), N, Mw, N,
                               /*standardize=*/false, dE.data(), stream_.get());
        if (!gram_reconstruct()) {
            out.status = Status::NonSpdCovariance;
            return out;
        }
        STEPPE_CUDA_CHECK(cudaMemsetAsync(d_acc.data(), 0, sizeof(double), stream_.get()));
        launch_pcangsd_sqdiff(dP.data(), dPprev.data(), static_cast<long>(NMw), d_acc.data(),
                              stream_.get());
        double acc = 0.0;
        d2h_async(&acc, d_acc, 1, stream_.get());
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        rmse = 0.5 * std::sqrt(acc / static_cast<double>(NMw));  // pi-scale rmse2d
        iters = it + 1;
        if (rmse < tol) break;
    }
    out.iters_run = iters;
    out.final_rmse = rmse;

    // --- final covariance: covPCAngsd (standardized E) + dCov diagonal -------
    launch_pcangsd_build_E(d_l, d_kept.data(), d_fk.data(), dP.data(), N, Mw, N,
                           /*standardize=*/true, dE.data(), stream_.get());
    DeviceBuffer<double> dC(NN);
    {
        const MathModeScope syrk_mode(blas_.get(), CUBLAS_PEDANTIC_MATH);
        engage_f2_precision(blas_.get(), precision);
        const double a = 1.0, b = 0.0;
        CUBLAS_CHECK(cublasDsyrk(blas_.get(), CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N,
                                 static_cast<int>(Mw), &a, dE.data(), N, &b, dC.data(), N));
    }
    launch_symmetrize_lower_to_full(dC.data(), N, stream_.get());
    DeviceBuffer<double> d_dcov(static_cast<std::size_t>(N));
    launch_pcangsd_dcov(d_l, d_kept.data(), d_fk.data(), dP.data(), N, Mw, N, d_dcov.data(),
                        stream_.get());
    launch_pcangsd_finalize_cov(dC.data(), d_dcov.data(), N, 1.0 / static_cast<double>(Mw),
                                stream_.get());

    // --- eigendecompose C for coords (C copied aside; Dsyevd overwrites) -----
    DeviceBuffer<double> dCeig(NN);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dCeig.data(), dC.data(), NN * sizeof(double),
                                      cudaMemcpyDeviceToDevice, stream_.get()));
    DeviceBuffer<double> dwC(static_cast<std::size_t>(N));
    {
        const CusolverMathModeScope eig_scope(solver_.get(), /*honorable=*/false);
        CUSOLVER_CHECK(cusolverDnDsyevd(solver_.get(), CUSOLVER_EIG_MODE_VECTOR,
                                        CUBLAS_FILL_MODE_LOWER, N, dCeig.data(), N, dwC.data(),
                                        dWork.data(), lwork, dInfo.data()));
    }
    int info = 0;
    d2h_async(&info, dInfo, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    if (info != 0) {
        out.status = Status::NonSpdCovariance;
        return out;
    }
    DeviceBuffer<double> dCoords(static_cast<std::size_t>(N) * static_cast<std::size_t>(K));
    // pcangsd keeps the full N-column eigenvector block (ncol = N).
    launch_pca_coords(dCeig.data(), dwC.data(), N, N, K, dCoords.data(), stream_.get());

    // --- optional individual allele-2 frequencies ----------------------------
    DeviceBuffer<double> d_pi(want_pi ? NMw : std::size_t{1});
    if (want_pi) launch_pcangsd_pi(dP.data(), d_fk.data(), Mw, N, d_pi.data(), stream_.get());

    // --- D2H the small results ----------------------------------------------
    out.cov.assign(NN, 0.0);
    out.coords.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(K), 0.0);
    std::vector<double> w_all(static_cast<std::size_t>(N), 0.0);
    d2h_async(out.cov.data(), dC, NN, stream_.get());
    d2h_async(out.coords.data(), dCoords,
              static_cast<std::size_t>(N) * static_cast<std::size_t>(K), stream_.get());
    d2h_async(w_all.data(), dwC, static_cast<std::size_t>(N), stream_.get());
    if (want_pi) {
        out.pi.assign(NMw, 0.0);
        d2h_async(out.pi.data(), d_pi, NMw, stream_.get());
    }
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

    out.N = N;
    out.e = K;
    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
