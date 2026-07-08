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
#include <memory>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/block_sink.cuh"  // kStreamStagingSlots (the shared 3-slot pipeline depth)
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

    // Scratch is sized ONCE to the batch-wide max item length Mmax (each item only touches
    // its own [0,M)). Checkpoint stride bound: C(M) = ceil(sqrt(M)) <= ceil(sqrt(Mmax)) =
    // Cmax, and nck(M) = ceil(M/C(M)) <= Cmax for all M <= Mmax — so both the C and the nck
    // scratch dimensions are safely bounded by Cmax (mirrors cuda_backend_roh.cu sizing).
    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Mm = static_cast<std::size_t>(Mmax);
    int Cmax = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(Mmax))));
    if (Cmax < 1) Cmax = 1;
    const std::size_t Cs = static_cast<std::size_t>(Cmax);

    // One look-ahead pipeline slot: its own stream + event, pinned host inputs/posterior
    // (built AHEAD in phase P, drained in phase C), and persistent device scratch reused
    // every item so no per-item cudaMalloc/cudaFree stalls the stream.
    struct Slot {
        Stream stream{};
        Event ev{};
        PinnedBuffer<std::uint8_t> h_ob;
        PinnedBuffer<int> h_map;
        PinnedBuffer<double> h_p, h_T, h_post;
        DeviceBuffer<std::uint8_t> dOb, dRefhaps;
        DeviceBuffer<int> dMap;
        DeviceBuffer<double> dP, dT, dProh;
        DeviceBuffer<double> dCheckRoh, dCheck0, dAlphaA, dAlphaB, dAlphaBlk, dA0Blk, dBetaA, dBetaB;
        long item = -1;
        long M = 0;
    };
    const int S = kStreamStagingSlots;  // the 3-slot depth (block_sink StagingRing constant)
    std::vector<Slot> slots(static_cast<std::size_t>(S));
    for (Slot& s : slots) {
        s.h_ob = PinnedBuffer<std::uint8_t>(Mm);
        s.h_map = PinnedBuffer<int>(Mm);
        s.h_p = PinnedBuffer<double>(Mm);
        s.h_T = PinnedBuffer<double>(Mm * 9);
        s.h_post = PinnedBuffer<double>(Mm);
        s.dOb = DeviceBuffer<std::uint8_t>(Mm);
        s.dRefhaps = DeviceBuffer<std::uint8_t>(Ks * Mm);
        s.dMap = DeviceBuffer<int>(Mm);
        s.dP = DeviceBuffer<double>(Mm);
        s.dT = DeviceBuffer<double>(Mm * 9);
        s.dProh = DeviceBuffer<double>(Mm);
        s.dCheckRoh = DeviceBuffer<double>(Cs * Ks);
        s.dCheck0 = DeviceBuffer<double>(Cs);
        s.dAlphaA = DeviceBuffer<double>(Ks);
        s.dAlphaB = DeviceBuffer<double>(Ks);
        s.dAlphaBlk = DeviceBuffer<double>(Cs * Ks);
        s.dA0Blk = DeviceBuffer<double>(Cs);
        s.dBetaA = DeviceBuffer<double>(Ks);
        s.dBetaB = DeviceBuffer<double>(Ks);
    }

    // Ordered look-ahead: issue up to S items ahead, drain in STRICT item order (a slot is
    // reused only after item (issue-S) has been drained — guaranteed by (issue-drain)<S — so
    // its buffers/event are free). Draining in item order makes the consumer's segment output
    // byte-stable regardless of which stream finishes first.
    long issue = 0, drain = 0;
    while (drain < N) {
        while (issue < N && (issue - drain) < S) {
            Slot& s = slots[static_cast<std::size_t>(issue % S)];
            const long M = build(issue, s.h_ob.data(), s.h_map.data(), s.h_p.data(), s.h_T.data());
            s.item = issue;
            s.M = M;
            if (M > 0) {
                int C = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(M))));
                if (C < 1) C = 1;
                const int nck = static_cast<int>((M + C - 1) / C);
                const std::size_t Ms = static_cast<std::size_t>(M);
                h2d_async(s.dOb, s.h_ob.data(), Ms, s.stream.get());
                h2d_async(s.dMap, s.h_map.data(), Ms, s.stream.get());
                h2d_async(s.dP, s.h_p.data(), Ms, s.stream.get());
                h2d_async(s.dT, s.h_T.data(), Ms * 9, s.stream.get());
                launch_roh_gather(res->panel.data(), res->donor_map.data(), s.dMap.data(), K, M,
                                  res->Mp, s.dRefhaps.data(), s.stream.get());
                launch_roh_fb(s.dOb.data(), s.dRefhaps.data(), s.dP.data(), s.dT.data(), K, M,
                              /*n_target=*/1, C, nck, e_rate, in_val, s.dProh.data(),
                              s.dCheckRoh.data(), s.dCheck0.data(), s.dAlphaA.data(),
                              s.dAlphaB.data(), s.dAlphaBlk.data(), s.dA0Blk.data(),
                              s.dBetaA.data(), s.dBetaB.data(), s.stream.get());
                d2h_async(s.h_post.data(), s.dProh, Ms, s.stream.get());
            }
            s.ev.record(s.stream);
            ++issue;
        }
        Slot& d = slots[static_cast<std::size_t>(drain % S)];
        d.ev.synchronize();  // normally already satisfied — we run S items ahead of the drain
        if (d.M > 0) consume(d.item, d.h_post.data(), d.M);
        ++drain;
    }
}

}  // namespace steppe::device
