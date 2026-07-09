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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "device/cuda/roh_fb_kernel.cuh"
#include "device/cuda/stream.hpp"

namespace steppe::device {

namespace {

// The ONCE-uploaded, run-resident hapROH panel owned by a RohPanelHandle (via
// shared_ptr<void>). Holds the donor-major panel bytes (Kpanel*Mp) + the selected-donor
// row map (K) in device memory so the batch pipeline gathers each target's compacted
// refhaps on-device instead of re-gathering + re-uploading K*M host bytes per target.
struct RohResidentPanel {
    DeviceBuffer<std::uint8_t> panel;  // Kpanel*Mp donor-major
    DeviceBuffer<int> donor_map;       // K, resident-panel row per selected donor
    long Mp = 0;
    int K = 0;
};

}  // namespace

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

RohPanelHandle CudaBackend::roh_upload_panel(const std::uint8_t* host_panel, int Kpanel, long Mp,
                                             const int* donor_map, int K) {
    guard_device();
    STEPPE_NVTX_RANGE("roh_upload_panel");
    RohPanelHandle h;
    h.host_panel = host_panel;
    h.donor_map = donor_map;
    h.Kpanel = Kpanel;
    h.K = K;
    h.Mp = Mp;
    if (Kpanel <= 0 || Mp <= 0 || K <= 0) return h;  // nothing resident; batch is a no-op

    auto res = std::make_shared<RohResidentPanel>();
    res->Mp = Mp;
    res->K = K;
    res->panel = DeviceBuffer<std::uint8_t>(static_cast<std::size_t>(Kpanel) *
                                            static_cast<std::size_t>(Mp));
    res->donor_map = DeviceBuffer<int>(static_cast<std::size_t>(K));
    // ONE H2D of the whole panel + the donor map (never re-touched per target).
    h2d_async(res->panel, host_panel,
              static_cast<std::size_t>(Kpanel) * static_cast<std::size_t>(Mp), stream_.get());
    h2d_async(res->donor_map, donor_map, static_cast<std::size_t>(K), stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    h.device = std::move(res);
    return h;
}

void CudaBackend::roh_fb_batch(const RohPanelHandle& panel, long N, long Mmax, double e_rate,
                               double in_val, const Precision& precision,
                               const RohItemBuilder& build, const RohPosteriorConsumer& consume) {
    (void)precision;  // native FP64 by construction (the FB is a sub-one product scan)
    guard_device();
    STEPPE_NVTX_RANGE("roh_fb_batch");
    const int K = panel.K;
    if (K <= 0 || Mmax <= 0 || N <= 0) return;
    auto res = std::static_pointer_cast<RohResidentPanel>(panel.device);
    if (!res) {  // no device residency — fall back to the serial oracle path
        ComputeBackend::roh_fb_batch(panel, N, Mmax, e_rate, in_val, precision, build, consume);
        return;
    }

    // WAVE BATCHING: replace the 3-slot look-ahead pipeline (which kept only ~2-3 blocks
    // resident on a 170-SM GPU) with a single grid of W item-blocks per wave. One block per
    // item (blockIdx.x = item within the wave), each block indexing the ONCE-resident panel
    // DIRECTLY via its site_map (PanelRefhaps) — no per-item compacted refhaps gather, so the
    // per-item VRAM footprint drops from ~K*Mmax bytes (~145 MB) to ~16 MB and hundreds of
    // items run concurrently. Bit-identical to the old per-item path: same FB math, same
    // per-item site axis / neighbor distances (per-item ob/site_map/p/T/M/C/nck from build),
    // build() ascending + consume() ascending -> byte-stable seg + summary.
    //
    // Scratch is sized to the WAVE width W (not N). Checkpoint stride bound: C(M)=ceil(sqrt(M))
    // <= ceil(sqrt(Mmax))=Cmax and nck(M)=ceil(M/C(M)) <= Cmax for all M<=Mmax, so both the
    // per-item C-tile and the nck checkpoint axis are bounded by Cmax (nckstride == Cstride).
    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Mm = static_cast<std::size_t>(Mmax);
    int Cmax = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(Mmax))));
    if (Cmax < 1) Cmax = 1;
    const std::size_t Cs = static_cast<std::size_t>(Cmax);

    // Wave width W from the FREE VRAM (the panel is already resident). Per-item device bytes:
    //   ob(Mm) + site_map(4*Mm) + p(8*Mm) + T(72*Mm) + proh(8*Mm)
    //   + check_roh(8*Cs*Ks) + check0(8*Cs) + alpha_blk(8*Cs*Ks) + a0_blk(8*Cs)
    //   + alphaA/B + betaA/B (4 * 8*Ks). Reserve 1 GB headroom, then take 90%.
    const std::size_t per_item = Mm * 1 + Mm * 4 + Mm * 8 + Mm * 9 * 8 + Mm * 8 + Cs * Ks * 8 +
                                 Cs * 8 + Cs * Ks * 8 + Cs * 8 + Ks * 8 * 4 + 64;
    std::size_t freeB = 0, totalB = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    const std::size_t reserve = static_cast<std::size_t>(1) << 30;  // 1 GB headroom
    std::size_t budget = (freeB > reserve) ? (freeB - reserve) : (freeB / 2);
    budget = static_cast<std::size_t>(static_cast<double>(budget) * 0.9);
    long W = (per_item > 0) ? static_cast<long>(budget / per_item) : N;
    if (W < 1) W = 1;
    if (W > N) W = N;
    const std::size_t Wz = static_cast<std::size_t>(W);

    // ONE set of wave-strided pinned + device buffers, sized to W (not N), reused every wave.
    PinnedBuffer<std::uint8_t> h_ob(Wz * Mm);
    PinnedBuffer<int> h_map(Wz * Mm);
    PinnedBuffer<double> h_p(Wz * Mm), h_T(Wz * Mm * 9), h_post(Wz * Mm);
    PinnedBuffer<int> h_M(Wz), h_C(Wz), h_nck(Wz);
    DeviceBuffer<std::uint8_t> dOb(Wz * Mm);
    DeviceBuffer<int> dMap(Wz * Mm);
    DeviceBuffer<double> dP(Wz * Mm), dT(Wz * Mm * 9), dProh(Wz * Mm);
    DeviceBuffer<int> dM(Wz), dC(Wz), dNck(Wz);
    DeviceBuffer<double> dCheckRoh(Wz * Cs * Ks), dCheck0(Wz * Cs);
    DeviceBuffer<double> dAlphaA(Wz * Ks), dAlphaB(Wz * Ks);
    DeviceBuffer<double> dAlphaBlk(Wz * Cs * Ks), dA0Blk(Wz * Cs);
    DeviceBuffer<double> dBetaA(Wz * Ks), dBetaB(Wz * Ks);

    for (long w0 = 0; w0 < N; w0 += W) {
        const long Wc = std::min<long>(W, N - w0);  // items in this wave
        // Phase P: build each item into its wave-strided pinned slot; record M/C/nck.
        for (long j = 0; j < Wc; ++j) {
            std::uint8_t* ob_j = h_ob.data() + static_cast<std::size_t>(j) * Mm;
            int* map_j = h_map.data() + static_cast<std::size_t>(j) * Mm;
            double* p_j = h_p.data() + static_cast<std::size_t>(j) * Mm;
            double* T_j = h_T.data() + static_cast<std::size_t>(j) * Mm * 9;
            const long M = build(w0 + j, ob_j, map_j, p_j, T_j);
            h_M.data()[j] = static_cast<int>(M);
            if (M > 0) {
                int C = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(M))));
                if (C < 1) C = 1;
                const int nck = static_cast<int>((M + C - 1) / C);
                h_C.data()[j] = C;
                h_nck.data()[j] = nck;
            } else {
                h_C.data()[j] = 1;
                h_nck.data()[j] = 0;
            }
        }
        // H2D the used [0,Wc) region (gaps M_j<Mmax within an item are never read by a block).
        const std::size_t Wcz = static_cast<std::size_t>(Wc);
        h2d_async(dOb, h_ob.data(), Wcz * Mm, stream_.get());
        h2d_async(dMap, h_map.data(), Wcz * Mm, stream_.get());
        h2d_async(dP, h_p.data(), Wcz * Mm, stream_.get());
        h2d_async(dT, h_T.data(), Wcz * Mm * 9, stream_.get());
        h2d_async(dM, h_M.data(), Wcz, stream_.get());
        h2d_async(dC, h_C.data(), Wcz, stream_.get());
        h2d_async(dNck, h_nck.data(), Wcz, stream_.get());

        // ONE grid launch: Wc item-blocks, each indexing the resident panel directly.
        launch_roh_fb_wave(dOb.data(), res->panel.data(), res->donor_map.data(), dMap.data(),
                           dP.data(), dT.data(), dM.data(), dC.data(), dNck.data(), K, res->Mp,
                           /*Mstride*/static_cast<long>(Mmax), /*Cstride*/static_cast<long>(Cmax),
                           static_cast<int>(Wc), e_rate, in_val, dProh.data(), dCheckRoh.data(),
                           dCheck0.data(), dAlphaA.data(), dAlphaB.data(), dAlphaBlk.data(),
                           dA0Blk.data(), dBetaA.data(), dBetaB.data(), stream_.get());

        d2h_async(h_post.data(), dProh, Wcz * Mm, stream_.get());
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

        // Phase C: consume in STRICT item order (each item reads only its own [0,M)).
        for (long j = 0; j < Wc; ++j) {
            const long M = h_M.data()[j];
            if (M > 0) consume(w0 + j, h_post.data() + static_cast<std::size_t>(j) * Mm, M);
        }
    }
}

}  // namespace steppe::device
