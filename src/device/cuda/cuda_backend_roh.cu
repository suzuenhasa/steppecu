// src/device/cuda/cuda_backend_roh.cu
//
// Out-of-line CudaBackend::roh_fb — the host orchestrator for the GPU hapROH
// (K+1)-state copying forward-backward (the `steppe roh` FB core). Uploads the target
// observations + the shared reference-haplotype panel + per-SNP allele-freq/transition,
// allocates the always-on checkpoint/recompute scratch (identical sizing discipline to
// cuda_backend_li_stephens.cu), runs the block-per-target scan, and returns only the
// n_target*M ROH posterior. A CUDA TU private to steppe_device.
//
// Reference: docs/planning/haproh-face-spec.md §3
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/roh_fb_kernel.cuh"

namespace steppe::device {

RohPosterior CudaBackend::roh_fb(const std::uint8_t* ob, const std::uint8_t* refhaps,
                                 const double* p, const double* T, int K, long M, int n_target,
                                 double e_rate, double in_val, const Precision& precision) {
    (void)precision;  // native FP64 by construction (the FB is a sub-one product scan)
    guard_device();
    STEPPE_NVTX_RANGE("roh_fb");
    RohPosterior out;
    out.n_target = n_target;
    out.M = M;
    if (K <= 0 || M <= 0 || n_target <= 0) {
        out.status = Status::Ok;
        return out;
    }

    // Checkpoint stride C = ceil(sqrt(M)); nck = ceil(M/C) blocks (matches the LS path).
    int C = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(M))));
    if (C < 1) C = 1;
    const int nck = static_cast<int>((M + C - 1) / C);

    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Ms = static_cast<std::size_t>(M);
    const std::size_t Nt = static_cast<std::size_t>(n_target);
    const std::size_t KM = Ks * Ms;
    const std::size_t NtM = Nt * Ms;
    const std::size_t NtK = Nt * Ks;
    const std::size_t nckK = static_cast<std::size_t>(nck) * Ks;

    DeviceBuffer<std::uint8_t> dOb(NtM);
    DeviceBuffer<std::uint8_t> dRefhaps(KM);
    DeviceBuffer<double> dP(Ms);
    DeviceBuffer<double> dT(Ms * 9);
    DeviceBuffer<double> dProh(NtM);
    DeviceBuffer<double> dCheckRoh(Nt * nckK);
    DeviceBuffer<double> dCheck0(Nt * static_cast<std::size_t>(nck));
    DeviceBuffer<double> dAlphaA(NtK), dAlphaB(NtK);
    DeviceBuffer<double> dAlphaBlk(Nt * static_cast<std::size_t>(C) * Ks);
    DeviceBuffer<double> dA0Blk(Nt * static_cast<std::size_t>(C));
    DeviceBuffer<double> dBetaA(NtK), dBetaB(NtK);

    h2d_async(dOb, ob, NtM, stream_.get());
    h2d_async(dRefhaps, refhaps, KM, stream_.get());
    h2d_async(dP, p, Ms, stream_.get());
    h2d_async(dT, T, Ms * 9, stream_.get());

    launch_roh_fb(dOb.data(), dRefhaps.data(), dP.data(), dT.data(), K, M, n_target, C, nck, e_rate,
                  in_val, dProh.data(), dCheckRoh.data(), dCheck0.data(), dAlphaA.data(),
                  dAlphaB.data(), dAlphaBlk.data(), dA0Blk.data(), dBetaA.data(), dBetaB.data(),
                  stream_.get());

    out.p_roh.assign(NtM, 0.0);
    d2h_async(out.p_roh.data(), dProh, NtM, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
