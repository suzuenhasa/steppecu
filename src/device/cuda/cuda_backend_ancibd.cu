// src/device/cuda/cuda_backend_ancibd.cu
//
// Out-of-line CudaBackend::ancibd_fb — the host orchestrator for the GPU ancIBD
// forward-backward (the `steppe ibd` FB core). Uploads the device-resident GP
// tensor + phased-GT bits + per-SNP AF/transition + the pair list, derives the
// per-haplotype ancestral-prob table on-device (LoadH5Multi2.get_haplo_prob), runs
// the block-per-pair 5-state scaled scan, and returns only the n_pair*M IBD
// posterior. A CUDA TU private to steppe_device.
//
// Reference: docs/planning/ancibd-face-spec.md §3
#include <cstdint>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/ancibd_fb_kernel.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"

namespace steppe::device {

AncibdPosterior CudaBackend::ancibd_fb(const double* gp3, const std::uint8_t* phased2,
                                       const double* p, const double* T, const int* pair_idx,
                                       int n_sample, long M, int n_pair, double in_val,
                                       double p_min, double min_error, const Precision& precision) {
    (void)precision;  // native FP64 by construction (the FB is a sub-one product scan)
    guard_device();
    STEPPE_NVTX_RANGE("ancibd_fb");
    AncibdPosterior out;
    out.n_pair = n_pair;
    out.M = M;
    if (n_sample <= 0 || M <= 0 || n_pair <= 0) {
        out.status = Status::Ok;
        return out;
    }

    const std::size_t Ms = static_cast<std::size_t>(M);
    const std::size_t nsamp = static_cast<std::size_t>(n_sample);
    const std::size_t npair = static_cast<std::size_t>(n_pair);
    const std::size_t gp3_n = Ms * nsamp * 3;
    const std::size_t ph_n = Ms * nsamp * 2;
    const std::size_t hts_n = 2 * nsamp * Ms;
    const std::size_t pairM = npair * Ms;

    DeviceBuffer<double> dGp(gp3_n);
    DeviceBuffer<std::uint8_t> dPhased(ph_n);
    DeviceBuffer<double> dP(Ms);
    DeviceBuffer<double> dT(Ms * 9);
    DeviceBuffer<int> dPairIdx(npair * 2);
    DeviceBuffer<double> dHts(hts_n);
    DeviceBuffer<double> dFwd0(pairM);
    DeviceBuffer<double> dC(pairM);
    DeviceBuffer<double> dPibd(pairM);

    h2d_async(dGp, gp3, gp3_n, stream_.get());
    h2d_async(dPhased, phased2, ph_n, stream_.get());
    h2d_async(dP, p, Ms, stream_.get());
    h2d_async(dT, T, Ms * 9, stream_.get());
    h2d_async(dPairIdx, pair_idx, npair * 2, stream_.get());

    // (1) derive the haplotype ancestral-prob table on-device from GP + phased bits.
    launch_ancibd_derive_hts(dGp.data(), dPhased.data(), dHts.data(), n_sample, M, min_error,
                             stream_.get());
    // (2) the 5-state scaled forward-backward, one block per pair.
    launch_ancibd_fb(dHts.data(), dP.data(), dT.data(), dPairIdx.data(), n_sample, M, n_pair,
                     in_val, p_min, dFwd0.data(), dC.data(), dPibd.data(), stream_.get());

    out.p_ibd.assign(pairM, 0.0);
    d2h_async(out.p_ibd.data(), dPibd, pairM, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
