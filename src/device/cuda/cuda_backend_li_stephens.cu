// src/device/cuda/cuda_backend_li_stephens.cu
//
// Out-of-line CudaBackend::ls_forward_backward — the host orchestrator for the
// GPU Li-Stephens copying forward-backward (the `steppe paint` FB core, Phase 1).
// Uploads the FB inputs, launches the block-per-recipient FB kernel with its
// always-on checkpoint/recompute scratch, and returns the per-column copying
// posterior gamma (donor-major, K*M). A CUDA TU private to steppe_device.
//
// Reference: docs/planning/li-stephens-engine-scope.md §2a.
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/li_stephens_fb_kernel.cuh"

namespace steppe::device {

LsPosterior CudaBackend::ls_forward_backward(const std::uint8_t* recipient,
                                             const std::uint8_t* donors, const double* pi,
                                             const double* rho, const double* mu, int K, long M,
                                             const Precision& precision) {
    (void)precision;  // the recurrence runs in native FP64 by construction (§2c)
    guard_device();
    STEPPE_NVTX_RANGE("ls_forward_backward");
    LsPosterior out;
    out.K = K;
    out.M = M;
    if (K <= 0 || M <= 0) {
        out.status = Status::Ok;
        return out;
    }

    // Phase-1 override drives ONE recipient per call (matches the CpuBackend
    // signature); the kernel itself is batch-ready over the block axis.
    constexpr int n_recip = 1;

    // Checkpoint stride C = ceil(sqrt(M)); nck = ceil(M/C) blocks. Always on, so
    // even the M=256 golden exercises the recompute path (never a resident-table
    // shortcut).
    int C = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(M))));
    if (C < 1) C = 1;
    const int nck = static_cast<int>((M + C - 1) / C);

    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Ms = static_cast<std::size_t>(M);
    const std::size_t KM = Ks * Ms;

    DeviceBuffer<std::uint8_t> dRecip(Ms);        // n_recip == 1
    DeviceBuffer<std::uint8_t> dDonors(KM);
    DeviceBuffer<double> dPi(Ks);
    DeviceBuffer<double> dRho(Ms);
    DeviceBuffer<double> dMu(Ms);
    DeviceBuffer<double> dGamma(KM);
    DeviceBuffer<double> dCheck(static_cast<std::size_t>(nck) * Ks);
    DeviceBuffer<double> dAlphaA(Ks), dAlphaB(Ks);
    DeviceBuffer<double> dAlphaBlk(static_cast<std::size_t>(C) * Ks);
    DeviceBuffer<double> dBetaA(Ks), dBetaB(Ks);

    h2d_async(dRecip, recipient, Ms, stream_.get());
    h2d_async(dDonors, donors, KM, stream_.get());
    h2d_async(dPi, pi, Ks, stream_.get());
    h2d_async(dRho, rho, Ms, stream_.get());
    h2d_async(dMu, mu, Ms, stream_.get());

    launch_ls_forward_backward(dRecip.data(), dDonors.data(), dPi.data(), dRho.data(), dMu.data(),
                               K, M, n_recip, C, nck, dGamma.data(), dCheck.data(), dAlphaA.data(),
                               dAlphaB.data(), dAlphaBlk.data(), dBetaA.data(), dBetaB.data(),
                               stream_.get());

    out.gamma.assign(KM, 0.0);
    d2h_async(out.gamma.data(), dGamma, KM, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    out.status = Status::Ok;
    return out;
}

LsCoancestry CudaBackend::ls_paint_coancestry(const std::uint8_t* recipients,
                                              const std::uint8_t* donors, const double* pi,
                                              const double* rho, const double* mu, const double* w,
                                              int K, long M, int N, const Precision& precision) {
    (void)precision;  // native FP64 by construction (§2c) — the sink is a reduction, not a GEMM
    guard_device();
    STEPPE_NVTX_RANGE("ls_paint_coancestry");
    LsCoancestry out;
    out.K = K;
    out.N = N;
    if (K <= 0 || M <= 0 || N <= 0) {
        out.status = Status::Ok;
        return out;
    }

    // Checkpoint stride C = ceil(sqrt(M)); nck = ceil(M/C) blocks (matches the gate path).
    int C = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(M))));
    if (C < 1) C = 1;
    const int nck = static_cast<int>((M + C - 1) / C);

    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Ms = static_cast<std::size_t>(M);
    const std::size_t Ns = static_cast<std::size_t>(N);
    const std::size_t KM = Ks * Ms;
    const std::size_t NK = Ns * Ks;
    const std::size_t nckK = static_cast<std::size_t>(nck) * Ks;

    // Batched over N recipients: one CUDA block per recipient, donors shared. Only the
    // small N*K accumulators return to host; the K*M posterior is never allocated.
    DeviceBuffer<std::uint8_t> dRecip(Ns * Ms);
    DeviceBuffer<std::uint8_t> dDonors(KM);
    DeviceBuffer<double> dPi(NK);
    DeviceBuffer<double> dRho(Ms);
    DeviceBuffer<double> dMu(Ms);
    DeviceBuffer<double> dW(Ms);
    DeviceBuffer<double> dAccCnt(NK);
    DeviceBuffer<double> dAccLen(NK);
    DeviceBuffer<double> dCheck(Ns * nckK);
    DeviceBuffer<double> dCheckPrev(Ns * nckK);
    DeviceBuffer<double> dAlphaA(NK), dAlphaB(NK);
    DeviceBuffer<double> dAlphaBlk(Ns * static_cast<std::size_t>(C) * Ks);
    DeviceBuffer<double> dBetaA(NK), dBetaB(NK);

    h2d_async(dRecip, recipients, Ns * Ms, stream_.get());
    h2d_async(dDonors, donors, KM, stream_.get());
    h2d_async(dPi, pi, NK, stream_.get());
    h2d_async(dRho, rho, Ms, stream_.get());
    h2d_async(dMu, mu, Ms, stream_.get());
    h2d_async(dW, w, Ms, stream_.get());

    // Paint mode: gamma output null, accumulators + w + companion checkpoints live.
    launch_ls_forward_backward(dRecip.data(), dDonors.data(), dPi.data(), dRho.data(), dMu.data(),
                               K, M, N, C, nck, /*d_gamma=*/nullptr, dCheck.data(), dAlphaA.data(),
                               dAlphaB.data(), dAlphaBlk.data(), dBetaA.data(), dBetaB.data(),
                               stream_.get(), dW.data(), dAccCnt.data(), dAccLen.data(),
                               dCheckPrev.data());

    out.chunkcounts.assign(NK, 0.0);
    out.chunklengths.assign(NK, 0.0);
    d2h_async(out.chunkcounts.data(), dAccCnt, NK, stream_.get());
    d2h_async(out.chunklengths.data(), dAccLen, NK, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    out.status = Status::Ok;
    return out;
}

LsLocalAncestry CudaBackend::ls_localanc(const std::uint8_t* recipients,
                                         const std::uint8_t* donors, const double* pi,
                                         const double* rho, const double* mu,
                                         const int* donor_group, int K, long M, int N, int P,
                                         const Precision& precision) {
    (void)precision;  // native FP64 by construction (§2c) — the sink is a reduction, not a GEMM
    guard_device();
    STEPPE_NVTX_RANGE("ls_localanc");
    LsLocalAncestry out;
    out.P = P;
    out.M = M;
    out.N = N;
    if (K <= 0 || M <= 0 || N <= 0 || P <= 0) {
        out.status = Status::Ok;
        return out;
    }

    // Checkpoint stride C = ceil(sqrt(M)); nck = ceil(M/C) blocks (matches the gate path).
    int C = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(M))));
    if (C < 1) C = 1;
    const int nck = static_cast<int>((M + C - 1) / C);

    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Ms = static_cast<std::size_t>(M);
    const std::size_t Ns = static_cast<std::size_t>(N);
    const std::size_t Ps = static_cast<std::size_t>(P);
    const std::size_t KM = Ks * Ms;
    const std::size_t NK = Ns * Ks;
    const std::size_t NMP = Ns * Ms * Ps;
    const std::size_t nckK = static_cast<std::size_t>(nck) * Ks;

    // Batched over N recipients: one CUDA block per recipient, donors shared. Only the
    // N*M*P per-SNP posterior returns to host; the K*M gamma is never allocated. Same
    // checkpoint/recompute scratch as coancestry, MINUS the companion-checkpoint buffer
    // (localanc has no switch term).
    DeviceBuffer<std::uint8_t> dRecip(Ns * Ms);
    DeviceBuffer<std::uint8_t> dDonors(KM);
    DeviceBuffer<double> dPi(NK);
    DeviceBuffer<double> dRho(Ms);
    DeviceBuffer<double> dMu(Ms);
    DeviceBuffer<int> dGroup(Ks);
    DeviceBuffer<double> dPost(NMP);
    DeviceBuffer<double> dCheck(Ns * nckK);
    DeviceBuffer<double> dAlphaA(NK), dAlphaB(NK);
    DeviceBuffer<double> dAlphaBlk(Ns * static_cast<std::size_t>(C) * Ks);
    DeviceBuffer<double> dBetaA(NK), dBetaB(NK);

    h2d_async(dRecip, recipients, Ns * Ms, stream_.get());
    h2d_async(dDonors, donors, KM, stream_.get());
    h2d_async(dPi, pi, NK, stream_.get());
    h2d_async(dRho, rho, Ms, stream_.get());
    h2d_async(dMu, mu, Ms, stream_.get());
    h2d_async(dGroup, donor_group, Ks, stream_.get());

    // Localanc mode: gamma + acc outputs null (and no companion checkpoints); the group
    // labels + per-SNP posterior buffer + P are the live output (the trailing three params).
    launch_ls_forward_backward(dRecip.data(), dDonors.data(), dPi.data(), dRho.data(), dMu.data(),
                               K, M, N, C, nck, /*d_gamma=*/nullptr, dCheck.data(), dAlphaA.data(),
                               dAlphaB.data(), dAlphaBlk.data(), dBetaA.data(), dBetaB.data(),
                               stream_.get(), /*d_w=*/nullptr, /*d_acc_cnt=*/nullptr,
                               /*d_acc_len=*/nullptr, /*d_checkpts_prev=*/nullptr, dGroup.data(),
                               dPost.data(), P);

    out.post.assign(NMP, 0.0);
    d2h_async(out.post.data(), dPost, NMP, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
