// src/device/cuda/cuda_backend_li_stephens.cu
//
// Out-of-line CudaBackend::ls_forward_backward / ls_paint_coancestry / ls_localanc —
// the host orchestrators for the GPU Li-Stephens copying forward-backward (the
// `steppe paint` FB core). ls_forward_backward drives ONE recipient (Phase 1, gate
// path); the two batched sinks (coancestry Phase 2, localanc Phase 3) sweep N
// recipients through the shared map-only WAVE core (device/cuda/wave_batch.cuh) so
// peak VRAM is bounded by the wave width W, not the full recipient count N. A CUDA TU
// private to steppe_device.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_li_stephens.cu.md
// Reference: docs/planning/li-stephens-engine-scope.md §2a.
// Reference: docs/planning/batch-over-items-driver-design.md §(c) — li_stephens adoption.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/li_stephens_fb_kernel.cuh"
#include "device/cuda/wave_batch.cuh"
#include "device/tier_select.hpp"  // free_host_ram_bytes (the host budget for the wave sizer)
#include "device/wave_budget.hpp"

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
    const std::size_t Cs = static_cast<std::size_t>(C);
    const std::size_t KM = Ks * Ms;
    const std::size_t NK = Ns * Ks;
    const std::size_t nckK = static_cast<std::size_t>(nck) * Ks;

    // WAVE BATCHING over the N recipients (shared map-only core, device/cuda/wave_batch.cuh).
    // Each recipient is INDEPENDENT — the FB kernel is one block per recipient (blockIdx.x =
    // rid) reading only its own rid-strided slice, with donors/rho/mu/w read WITHOUT a rid
    // offset. So restructuring the single full-N launch into N/W waves of Wc item-blocks
    // reorders nothing: block rid in a Wc-wide wave computes byte-identically to block rid in a
    // full-N launch. This is parity-neutral (no cross-recipient reduction) and fixes the latent
    // OOM: every buffer is sized to the WAVE width W, not N.
    //
    // Shared operand (uploaded ONCE, run-resident, const for the sweep): donors + rho + mu + w.
    // Per-recipient (waved): recipient/pi inputs, chunkcount/chunklength outputs, and the
    // always-on checkpoint/recompute FB scratch — all sized to W.
    DeviceBuffer<std::uint8_t> dDonors(KM);
    DeviceBuffer<double> dRho(Ms);
    DeviceBuffer<double> dMu(Ms);
    DeviceBuffer<double> dW(Ms);
    h2d_async(dDonors, donors, KM, stream_.get());
    h2d_async(dRho, rho, Ms, stream_.get());
    h2d_async(dMu, mu, Ms, stream_.get());
    h2d_async(dW, w, Ms, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));  // resident before the VRAM measure

    // Per-item footprint, declared (not hand-summed). Staging pairs (recipient/pi in,
    // acc_cnt/acc_len out) count on BOTH the device and pinned-host axes; the device-only FB
    // scratch counts on the device axis alone. wave_width caps W by min(VRAM, pinned-host).
    PerItemBytes pib;
    pib.pair_add<std::uint8_t>(Ms)  // recipient (in)
        .pair_add<double>(Ks)       // pi (in)
        .pair_add<double>(Ks)       // acc_cnt (out)
        .pair_add<double>(Ks);      // acc_len (out)
    pib.dev_add<double>(nckK)       // check
        .dev_add<double>(nckK)      // check_prev (companion, paint only)
        .dev_add<double>(Ks * 4)    // alphaA + alphaB + betaA + betaB
        .dev_add<double>(Cs * Ks);  // alpha_blk

    std::size_t freeB = 0, totalB = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));  // shared operand already resident
    const std::size_t host_free = free_host_ram_bytes();
    const long Wv = wave_width(freeB, pib.dev, host_free, pib.host, N);
    const std::size_t Wz = static_cast<std::size_t>(Wv);

    // ONE set of W-wide staging pairs (pinned<->device) + device-only FB scratch, sized once,
    // reused every wave. recipient/pi are staged in; acc_cnt/acc_len are staged out.
    WaveStage<std::uint8_t> recip(Wv, Ms, stream_.get());
    WaveStage<double> piS(Wv, Ks, stream_.get());
    WaveStage<double> accCnt(Wv, Ks, stream_.get());  // chunkcounts (d2h)
    WaveStage<double> accLen(Wv, Ks, stream_.get());  // chunklengths (d2h)
    DeviceBuffer<double> dCheck(Wz * nckK), dCheckPrev(Wz * nckK);
    DeviceBuffer<double> dAlphaA(Wz * Ks), dAlphaB(Wz * Ks);
    DeviceBuffer<double> dAlphaBlk(Wz * Cs * Ks);
    DeviceBuffer<double> dBetaA(Wz * Ks), dBetaB(Wz * Ks);

    out.chunkcounts.assign(NK, 0.0);
    out.chunklengths.assign(NK, 0.0);

    WaveMapOps ops;
    // build: copy this wave's recipient/pi slice (contiguous recipient-major) into the pinned
    // staging, then H2D the live [0,Wc) prefix. Each recipient is a disjoint slice.
    ops.build = [&](long w0, long Wc) {
        const std::size_t wcM = static_cast<std::size_t>(Wc) * Ms;
        const std::size_t wcK = static_cast<std::size_t>(Wc) * Ks;
        std::memcpy(recip.host(), recipients + static_cast<std::size_t>(w0) * Ms, wcM);
        std::memcpy(piS.host(), pi + static_cast<std::size_t>(w0) * Ks, wcK * sizeof(double));
        recip.h2d(Wc, stream_.get());
        piS.h2d(Wc, stream_.get());
    };
    // launch: one grid of Wc recipient-blocks vs the resident donors/rho/mu/w; paint sink
    // (gamma null, acc_cnt/acc_len live, companion checkpoints live). Then D2H the outputs.
    ops.launch = [&](long /*w0*/, long Wc) {
        launch_ls_forward_backward(recip.device(), dDonors.data(), piS.device(), dRho.data(),
                                   dMu.data(), K, M, static_cast<int>(Wc), C, nck,
                                   /*d_gamma=*/nullptr, dCheck.data(), dAlphaA.data(),
                                   dAlphaB.data(), dAlphaBlk.data(), dBetaA.data(), dBetaB.data(),
                                   stream_.get(), dW.data(), accCnt.device(), accLen.device(),
                                   dCheckPrev.data());
        accCnt.d2h(Wc, stream_.get());
        accLen.d2h(Wc, stream_.get());
    };
    // consume: STRICT ascending w0 (wave_map contract). Each recipient's [0,K) accumulator
    // slice lands at its own recipient-major offset — byte-stable for any W.
    ops.consume = [&](long w0, long Wc) {
        const std::size_t wcK = static_cast<std::size_t>(Wc) * Ks;
        std::memcpy(out.chunkcounts.data() + static_cast<std::size_t>(w0) * Ks, accCnt.host(),
                    wcK * sizeof(double));
        std::memcpy(out.chunklengths.data() + static_cast<std::size_t>(w0) * Ks, accLen.host(),
                    wcK * sizeof(double));
    };

    wave_map(N, Wv, ops, stream_.get());

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
    const std::size_t Cs = static_cast<std::size_t>(C);
    const std::size_t KM = Ks * Ms;
    const std::size_t MP = Ms * Ps;
    const std::size_t NMP = Ns * MP;
    const std::size_t nckK = static_cast<std::size_t>(nck) * Ks;

    // WAVE BATCHING over the N recipients (shared map-only core, device/cuda/wave_batch.cuh) —
    // same parity-neutral restructure as coancestry (one independent block per recipient). This
    // is the payoff case: the localanc per-SNP posterior is N*M*P (dPost, :NMP), the dominant
    // OOM term, so sizing it to the WAVE width W instead of N is what caps peak VRAM. Same
    // checkpoint/recompute scratch as coancestry MINUS the companion-checkpoint buffer (localanc
    // has no switch term).
    //
    // Shared operand (uploaded ONCE, run-resident, const for the sweep): donors + rho + mu +
    // donor_group. Per-recipient (waved to W): recipient/pi inputs, the M*P posterior output,
    // and the FB scratch.
    DeviceBuffer<std::uint8_t> dDonors(KM);
    DeviceBuffer<double> dRho(Ms);
    DeviceBuffer<double> dMu(Ms);
    DeviceBuffer<int> dGroup(Ks);
    h2d_async(dDonors, donors, KM, stream_.get());
    h2d_async(dRho, rho, Ms, stream_.get());
    h2d_async(dMu, mu, Ms, stream_.get());
    h2d_async(dGroup, donor_group, Ks, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));  // resident before the VRAM measure

    // Per-item footprint, declared. Staging pairs (recipient/pi in, the large M*P post out) count
    // on BOTH axes; the device-only FB scratch on the device axis alone. The M*P output term is
    // what forces W down (and what was N*M*P resident before) — wave_width counts it here.
    PerItemBytes pib;
    pib.pair_add<std::uint8_t>(Ms)  // recipient (in)
        .pair_add<double>(Ks)       // pi (in)
        .pair_add<double>(MP);      // post (out) — the large per-item term
    pib.dev_add<double>(nckK)       // check
        .dev_add<double>(Ks * 4)    // alphaA + alphaB + betaA + betaB
        .dev_add<double>(Cs * Ks);  // alpha_blk

    std::size_t freeB = 0, totalB = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));  // shared operand already resident
    const std::size_t host_free = free_host_ram_bytes();
    const long Wv = wave_width(freeB, pib.dev, host_free, pib.host, N);
    const std::size_t Wz = static_cast<std::size_t>(Wv);

    // ONE set of W-wide staging pairs + device-only FB scratch, sized once, reused every wave.
    WaveStage<std::uint8_t> recip(Wv, Ms, stream_.get());
    WaveStage<double> piS(Wv, Ks, stream_.get());
    WaveStage<double> postS(Wv, MP, stream_.get());  // per-SNP posterior (d2h)
    DeviceBuffer<double> dCheck(Wz * nckK);
    DeviceBuffer<double> dAlphaA(Wz * Ks), dAlphaB(Wz * Ks);
    DeviceBuffer<double> dAlphaBlk(Wz * Cs * Ks);
    DeviceBuffer<double> dBetaA(Wz * Ks), dBetaB(Wz * Ks);

    out.post.assign(NMP, 0.0);

    WaveMapOps ops;
    ops.build = [&](long w0, long Wc) {
        const std::size_t wcM = static_cast<std::size_t>(Wc) * Ms;
        const std::size_t wcK = static_cast<std::size_t>(Wc) * Ks;
        std::memcpy(recip.host(), recipients + static_cast<std::size_t>(w0) * Ms, wcM);
        std::memcpy(piS.host(), pi + static_cast<std::size_t>(w0) * Ks, wcK * sizeof(double));
        recip.h2d(Wc, stream_.get());
        piS.h2d(Wc, stream_.get());
    };
    // launch: localanc sink — gamma + acc null (no companion checkpoints); group labels +
    // per-SNP posterior + P are the live output. Then D2H the M*P prefix.
    ops.launch = [&](long /*w0*/, long Wc) {
        launch_ls_forward_backward(recip.device(), dDonors.data(), piS.device(), dRho.data(),
                                   dMu.data(), K, M, static_cast<int>(Wc), C, nck,
                                   /*d_gamma=*/nullptr, dCheck.data(), dAlphaA.data(),
                                   dAlphaB.data(), dAlphaBlk.data(), dBetaA.data(), dBetaB.data(),
                                   stream_.get(), /*d_w=*/nullptr, /*d_acc_cnt=*/nullptr,
                                   /*d_acc_len=*/nullptr, /*d_checkpts_prev=*/nullptr,
                                   dGroup.data(), postS.device(), P);
        postS.d2h(Wc, stream_.get());
    };
    // consume: STRICT ascending w0. Each recipient's [0,M*P) posterior block lands at its own
    // recipient-major offset — byte-stable for any W.
    ops.consume = [&](long w0, long Wc) {
        const std::size_t wcMP = static_cast<std::size_t>(Wc) * MP;
        std::memcpy(out.post.data() + static_cast<std::size_t>(w0) * MP, postS.host(),
                    wcMP * sizeof(double));
    };

    wave_map(N, Wv, ops, stream_.get());

    out.status = Status::Ok;
    return out;
}

}  // namespace steppe::device
