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
#include <cstdlib>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

#include "core/internal/nvtx.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/pinned_buffer.cuh"
#include "device/cuda/roh_fb_kernel.cuh"
#include "device/cuda/stream.hpp"
#include "device/cuda/wave_batch.cuh"
#include "device/tier_select.hpp"  // free_host_ram_bytes (the host budget for the wave sizer)
#include "device/wave_budget.hpp"

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

// F2: default upper bound on the Phase-P host-build worker threads (mirrors cmd_roh.cpp's
// kRohDefaultThreadCap). Multi-device dispatch runs D device host-threads, each spawning up to
// this many build workers, so a modest cap keeps D*cap under the box core count.
constexpr std::size_t kRohBuildThreadCap = 64;

// Read the Phase-P build thread cap from STEPPE_ROH_THREADS. cuda_backend_roh.cu is a SEPARATE
// translation unit from cmd_roh.cpp (whose roh_thread_cap() is file-local), so it needs its OWN
// getenv — otherwise the parity harness's STEPPE_ROH_THREADS=1 (forced-serial) invariance proof
// would never reach this region. A positive integer pins the cap (=1 forces the fully-serial
// Phase-P, matching the reference binary); unset/invalid -> kRohBuildThreadCap.
[[nodiscard]] std::size_t roh_build_thread_cap() {
    const char* e = std::getenv("STEPPE_ROH_THREADS");
    if (e != nullptr && *e != '\0') {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && v >= 1) return static_cast<std::size_t>(v);
    }
    return kRohBuildThreadCap;
}

// Final clamped block/thread count for a wave of Wc items:
// T = max(1, min(hardware_concurrency(), STEPPE_ROH_THREADS cap, n)). Capping by n means a small
// wave never over-spawns; hardware_concurrency()==0 falls back to serial. Deterministic in n.
[[nodiscard]] std::size_t roh_build_T(std::size_t n) {
    if (n == 0) return 0;
    const unsigned hw = std::thread::hardware_concurrency();
    const std::size_t base = (hw == 0) ? std::size_t{1} : static_cast<std::size_t>(hw);
    std::size_t T = std::min<std::size_t>({base, roh_build_thread_cap(), n});
    return std::max<std::size_t>(std::size_t{1}, T);
}

// Partition [0, n) into T contiguous ascending blocks (T = roh_build_T(n)) and run
// fn(begin, end, block_index) on one std::thread per non-empty block, then join. T==1 (or a tiny
// wave) runs fn inline with no thread spawn. Mirrors cmd_roh.cpp::parallel_blocks EXACTLY: each
// worker body is wrapped so a throw (e.g. bad_alloc from build()'s local gpos/Tv/cov vectors) is
// captured as an exception_ptr and rethrown AFTER join — an uncaught throw crossing a std::thread
// boundary would call std::terminate. Because every item writes only its OWN disjoint output
// slice, concurrent blocks are order-independent and byte-identical to the serial fill.
template <typename Fn>
void parallel_blocks_roh(std::size_t n, Fn&& fn) {
    const std::size_t T = roh_build_T(n);
    if (T <= 1) {
        if (n > 0) fn(std::size_t{0}, n, std::size_t{0});
        return;
    }
    const std::size_t block = (n + T - 1) / T;  // ceil, so T blocks cover [0, n)
    std::vector<std::thread> threads;
    threads.reserve(T);
    std::vector<std::exception_ptr> errs(T);
    for (std::size_t b = 0; b < T; ++b) {
        const std::size_t begin = b * block;
        if (begin >= n) break;  // ceil rounding can leave trailing blocks empty
        const std::size_t end = std::min(n, begin + block);
        threads.emplace_back([&fn, &errs, begin, end, b]() {
            try {
                fn(begin, end, b);
            } catch (...) {
                errs[b] = std::current_exception();
            }
        });
    }
    for (std::thread& th : threads) th.join();
    for (const std::exception_ptr& e : errs)
        if (e) std::rethrow_exception(e);
}

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

    // WAVE BATCHING (via the shared map-only wave core, device/cuda/wave_batch.cuh): replace
    // the 3-slot look-ahead pipeline (which kept only ~2-3 blocks resident on a 170-SM GPU)
    // with a single grid of W item-blocks per wave. One block per item (blockIdx.x = item
    // within the wave), each block indexing the ONCE-resident panel DIRECTLY via its site_map
    // (PanelRefhaps) — no per-item compacted refhaps gather, so the per-item VRAM footprint
    // drops from ~K*Mmax bytes (~145 MB) to ~16 MB and hundreds of items run concurrently.
    // Bit-identical to the old per-item path: same FB math, same per-item site axis / neighbor
    // distances (per-item ob/site_map/p/T/M/C/nck from build), build() ascending +
    // consume() strictly ascending -> byte-stable seg + summary. The wave width W is an
    // execution detail only: every item is independent and its posterior is a disjoint output
    // slice, so the emitted table is byte-identical for ANY W (this is what lets the shared
    // sizer's dual budget re-clamp W without a parity change).
    //
    // Scratch is sized to the WAVE width W (not N). Checkpoint stride bound: C(M)=ceil(sqrt(M))
    // <= ceil(sqrt(Mmax))=Cmax and nck(M)=ceil(M/C(M)) <= Cmax for all M<=Mmax, so both the
    // per-item C-tile and the nck checkpoint axis are bounded by Cmax (nckstride == Cstride).
    const std::size_t Ks = static_cast<std::size_t>(K);
    const std::size_t Mm = static_cast<std::size_t>(Mmax);
    int Cmax = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(Mmax))));
    if (Cmax < 1) Cmax = 1;
    const std::size_t Cs = static_cast<std::size_t>(Cmax);

    // Per-item footprint, declared (not hand-summed). Staging pairs (built on host, uploaded,
    // or downloaded) count on BOTH the device and the pinned-host axes; the device-only FB
    // scratch (check/alpha/beta) counts on the device axis alone. wave_width caps W by
    // min(VRAM-derived, pinned-host-derived) — the pinned-host arm is the latent budget bug
    // this shared sizer fixes: since the host per-item footprint (no scratch) is smaller than
    // the device one, the host arm binds only when free host RAM is far below free VRAM.
    PerItemBytes pib;
    pib.pair_add<std::uint8_t>(Mm)   // ob
        .pair_add<int>(Mm)           // site_map
        .pair_add<double>(Mm)        // p
        .pair_add<double>(Mm * 9)    // T
        .pair_add<double>(Mm)        // proh (posterior output)
        .pair_add<int>(1)            // M
        .pair_add<int>(1)            // C
        .pair_add<int>(1);           // nck
    pib.dev_add<double>(Cs * Ks)     // check_roh
        .dev_add<double>(Cs)         // check0
        .dev_add<double>(Cs * Ks)    // alpha_blk
        .dev_add<double>(Cs)         // a0_blk
        .dev_add<double>(Ks * 4);    // alphaA + alphaB + betaA + betaB

    std::size_t freeB = 0, totalB = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));  // panel already resident (decode_budget seam)
    const std::size_t host_free = free_host_ram_bytes();
    const long W = wave_width(freeB, pib.dev, host_free, pib.host, N);
    const std::size_t Wz = static_cast<std::size_t>(W);

    // ONE set of W-wide staging pairs (pinned<->device), sized once, reused every wave. The
    // dOb/dMap/... pairs of the old hand-rolled loop become WaveStage<T>; the FB scratch stays
    // device-only DeviceBuffer<T>. WaveStage owns both halves; wave_map drives them.
    WaveStage<std::uint8_t> ob(W, Mm, stream_.get());
    WaveStage<int> map(W, Mm, stream_.get());
    WaveStage<double> p(W, Mm, stream_.get());
    WaveStage<double> Tt(W, Mm * 9, stream_.get());
    WaveStage<double> post(W, Mm, stream_.get());  // posterior output (d2h only)
    WaveStage<int> mM(W, 1, stream_.get());
    WaveStage<int> mC(W, 1, stream_.get());
    WaveStage<int> mNck(W, 1, stream_.get());
    DeviceBuffer<double> dCheckRoh(Wz * Cs * Ks), dCheck0(Wz * Cs);
    DeviceBuffer<double> dAlphaA(Wz * Ks), dAlphaB(Wz * Ks);
    DeviceBuffer<double> dAlphaBlk(Wz * Cs * Ks), dA0Blk(Wz * Cs);
    DeviceBuffer<double> dBetaA(Wz * Ks), dBetaB(Wz * Ks);

    WaveMapOps ops;
    // Phase P (build): fill each item into its wave-strided pinned slot; record M/C/nck; then
    // H2D the live [0,Wc) prefix of every input stage. PARALLELIZED across the wave items via
    // parallel_blocks_roh, which stays INSIDE the build callback (consumer policy). Each item j
    // writes ONLY its own disjoint pinned slice (ob/map/p/T at stride j*Mm) + its own
    // mM/mC/mNck[j] scalar, and build() reads only const shared prep, so concurrent per-item
    // builds are order-independent and byte-identical to the serial fill. The parallel join
    // happens before the H2D, so the uploaded bytes are exactly the serial-fill bytes.
    // STEPPE_ROH_THREADS=1 forces this fully serial (parity harness invariance proof).
    ops.build = [&](long w0, long Wc) {
        parallel_blocks_roh(static_cast<std::size_t>(Wc),
                            [&](std::size_t j0, std::size_t j1, std::size_t) {
            for (std::size_t j = j0; j < j1; ++j) {
                std::uint8_t* ob_j = ob.host() + j * Mm;
                int* map_j = map.host() + j * Mm;
                double* p_j = p.host() + j * Mm;
                double* T_j = Tt.host() + j * Mm * 9;
                const long M = build(w0 + static_cast<long>(j), ob_j, map_j, p_j, T_j);
                mM.host()[j] = static_cast<int>(M);
                if (M > 0) {
                    int C = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(M))));
                    if (C < 1) C = 1;
                    const int nck = static_cast<int>((M + C - 1) / C);
                    mC.host()[j] = C;
                    mNck.host()[j] = nck;
                } else {
                    mC.host()[j] = 1;
                    mNck.host()[j] = 0;
                }
            }
        });
        // Upload the used [0,Wc) prefix (gaps M_j<Mmax within an item are never read by a block).
        ob.h2d(Wc, stream_.get());
        map.h2d(Wc, stream_.get());
        p.h2d(Wc, stream_.get());
        Tt.h2d(Wc, stream_.get());
        mM.h2d(Wc, stream_.get());
        mC.h2d(Wc, stream_.get());
        mNck.h2d(Wc, stream_.get());
    };
    // ONE grid launch: Wc item-blocks, each indexing the resident panel directly; then D2H the
    // posterior prefix.
    ops.launch = [&](long /*w0*/, long Wc) {
        launch_roh_fb_wave(ob.device(), res->panel.data(), res->donor_map.data(), map.device(),
                           p.device(), Tt.device(), mM.device(), mC.device(), mNck.device(), K,
                           res->Mp, /*Mstride*/ static_cast<long>(Mmax),
                           /*Cstride*/ static_cast<long>(Cmax), static_cast<int>(Wc), e_rate,
                           in_val, post.device(), dCheckRoh.data(), dCheck0.data(), dAlphaA.data(),
                           dAlphaB.data(), dAlphaBlk.data(), dA0Blk.data(), dBetaA.data(),
                           dBetaB.data(), stream_.get());
        post.d2h(Wc, stream_.get());
    };
    // Phase C (consume): STRICT item order (wave_map guarantees ascending w0), each item reads
    // only its own [0,M) posterior slice.
    ops.consume = [&](long w0, long Wc) {
        for (long j = 0; j < Wc; ++j) {
            const long M = mM.host()[j];
            if (M > 0) consume(w0 + j, post.host() + static_cast<std::size_t>(j) * Mm, M);
        }
    };

    wave_map(N, W, ops, stream_.get());
}

}  // namespace steppe::device
