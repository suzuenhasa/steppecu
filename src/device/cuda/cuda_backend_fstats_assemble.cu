// src/device/cuda/cuda_backend_fstats_assemble.cu
//
// CudaBackend — f-stat sweep + f4/f3 assemble subsystem TU (cuda_backend.cu split T6;
// docs/kimiactions/05-cuda-backend-split.md §2.3 TU-G). Out-of-line homes of the
// device-resident f-statistic assemble + all-quartet/all-triple sweep family:
// `CudaBackend::device_survivor_blocks` (the SINGLE definition of the Vpair missing-block
// survivor predicate — header-declared, so it is also CALLED cross-TU by the dstat TU's
// `f4ratio_blocks_jackknife` and the qpadm-fit TU's `fit_models_batched`),
// `run_fstat_sweep_device` (the on-device k=3/k=4 all-combination f4/f3 sweep: enumerate →
// f4 loo/total reduce → z-filter → CUB radix top-K reservoir), the six device-resident
// assemble doors `assemble_f4` / `assemble_f4_quartets` / `assemble_f3_triples`
// (DeviceF2Blocks) plus their CpuBackend-oracle host-tensor throw-twins, and the
// `f4_sweep` / `f3_sweep` thin k-dispatch wrappers. Bodies MOVED VERBATIM from
// cuda_backend.cu; nothing about codegen / math / precision / file-order changed by the
// split.
//
// Shared mutable state stays as a class member (cuda_backend.cuh): `tot_line_` — WRITTEN
// here (the assemble methods cache the per-block total line), READ by the qpadm-fit TU's
// `jackknife_cov` / `jackknife_diag`. The f-stat sweep is NATIVE-FP64 by the §12
// catastrophic-cancellation carve-out (the `precision` arg is intentionally ignored),
// consistent with the f4-numerator policy.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). Joins the SAME
// steppe_device target, so it inherits identical codegen/macros/RDC.
#include <cub/device/device_radix_sort.cuh>  // cub::DeviceRadixSort::SortPairsDescending — the sweep top-K reservoir sort
#include <cub/device/device_select.cuh>      // cub::DeviceSelect::Flagged — the GPU-only z-filter survivor compaction

#include <algorithm>  // std::max — the sweep reservoir / top-K sizing clamps
#include <cmath>      // (per the §2.3 TU-G include set; the f-stat sweep z/tau math)
#include <cstdlib>    // std::getenv / std::strtol — the STEPPE_FSTAT_CHUNK sweep chunk lever
#include <stdexcept>  // std::runtime_error — the host-tensor assemble throw-twins
#include <vector>     // std::vector — survivor / est / se / z host staging

#include "core/domain/block_partition_rule.hpp"        // core::block_ranges / core::BlockRange (per the §2.3 TU-G include set)
#include "core/internal/nvtx.hpp"                       // STEPPE_NVTX_RANGE (per the §2.3 TU-G include set; coarse phase-boundary marker)
#include "device/cuda/cuda_backend.cuh"                 // the CudaBackend class declaration (split T0)
#include "device/cuda/check.cuh"                        // STEPPE_CUDA_CHECK
#include "device/cuda/device_f2_blocks_impl.cuh"        // DeviceF2Blocks::Impl (the DeviceBuffer<double> f2/vpair owners behind f2.*_device())
#include "device/cuda/f2_block_kernel.cuh"              // (per the §2.3 TU-G include set; the f2 precision-probe seam)
#include "device/cuda/qpadm_fit_kernels.cuh"            // launch_f2_block_keep + f4 gather/loo-total/xtau/diag-var + sweep unrank/zfilter/topk launch wrappers
#include "steppe/config.hpp"                            // Precision, kFstatDefaultSweepTopK / kSweepFilterTopK / kSweepFilterMinZ

namespace steppe::device {

std::vector<int> CudaBackend::device_survivor_blocks(
    const steppe::device::DeviceF2Blocks& f2, int nb, int P) {
    std::vector<int> surv;
    if (nb <= 0) return surv;
    surv.reserve(static_cast<std::size_t>(nb));
    // No resident Vpair ⇒ keep every block (no missing-block info ⇒ no drop).
    if (f2.vpair_device() == nullptr) {
        for (int b = 0; b < nb; ++b) surv.push_back(b);
        return surv;
    }
    DeviceBuffer<int> dKeep(static_cast<std::size_t>(nb));
    launch_f2_block_keep(f2.vpair_device(), P, nb, dKeep.data(), stream_.get());
    std::vector<int> keep(static_cast<std::size_t>(nb), 1);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(keep.data(), dKeep.data(),
                                      static_cast<std::size_t>(nb) * sizeof(int),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    for (int b = 0; b < nb; ++b)
        if (keep[static_cast<std::size_t>(b)] != 0) surv.push_back(b);
    return surv;
}

SweepSurvivors CudaBackend::run_fstat_sweep_device(
    const steppe::device::DeviceF2Blocks& f2, const SweepConfig& cfg,
    const Precision& precision, int k) {
    (void)precision;  // native FP64 by the f-stat cancellation carve-out (consistent policy)
    guard_device();

    // TU-local sizing constants for the sweep reservoir (§3.3 occupancy/sizing tuning — kept
    // next to the sweep, not in the public config header).
    constexpr std::size_t kFstatIntClampMax = 0x40000000;  // 2^30 CUB num_items / int-kernel ceiling
    constexpr std::size_t kFstatReservoirBytesPerSlot = 160;  // 112 B/slot rounded up for CUB sort temp

    SweepSurvivors out;
    const int P = f2.P;
    const int nb_full = f2.n_block;
    const int range = cfg.pop_subset.empty() ? P : static_cast<int>(cfg.pop_subset.size());

    // Total enumerated C(range, k) (saturating; the host driver already capped, but echo it).
    unsigned long long enumerated = 1ULL;
    if (k < 0 || range < k) {
        enumerated = 0ULL;
    } else {
        for (int i = 1; i <= k; ++i) {
            const unsigned long long num = static_cast<unsigned long long>(range - k + i);
            if (num != 0 && enumerated > (~0ULL) / num) { enumerated = ~0ULL; break; }
            enumerated = enumerated * num / static_cast<unsigned long long>(i);
        }
    }
    out.enumerated = static_cast<std::size_t>(
        enumerated > static_cast<unsigned long long>(SIZE_MAX) ? SIZE_MAX : enumerated);

    if (range < k || k < 1 || nb_full <= 0 || f2.f2_device() == nullptr || enumerated == 0ULL) {
        out.status = Status::Ok;  // nothing to sweep — a clean empty Ok result.
        return out;
    }

    // F1/OQ-12 SURVIVOR-block drop (ONCE for the whole sweep — block-missingness is a
    // property of the f2, not the item). The gather/loo/xtau/diag_var all use nb_s.
    const std::vector<int> surv = device_survivor_blocks(f2, nb_full, P);
    const int nb_s = static_cast<int>(surv.size());
    if (nb_s <= 0) { out.status = Status::Ok; return out; }
    const bool dropped = (nb_s != nb_full);

    // Survivor block sizes + n = Σ block_sizes (the jackknife weight total).
    std::vector<int> surv_block_sizes(static_cast<std::size_t>(nb_s), 0);
    long long n_ll = 0;
    for (int bs = 0; bs < nb_s; ++bs) {
        const int orig = surv[static_cast<std::size_t>(bs)];
        const int sz = f2.block_sizes[static_cast<std::size_t>(orig)];
        surv_block_sizes[static_cast<std::size_t>(bs)] = sz;
        n_ll += sz;
    }
    const double n = static_cast<double>(n_ll);

    // Persistent device buffers (uploaded ONCE): survivor block sizes + survivor map + subset.
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), surv_block_sizes.data(),
                                      static_cast<std::size_t>(nb_s) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    if (dropped)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));
    const bool use_subset = !cfg.pop_subset.empty();
    DeviceBuffer<int> dSubset(static_cast<std::size_t>(use_subset ? range : 1));
    if (use_subset)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSubset.data(), cfg.pop_subset.data(),
                                          static_cast<std::size_t>(range) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

    // The bounded top-K reservoir target K (clamped to INT_MAX + the enumerated total). Sized
    // FIRST so its FIXED device footprint (CAP=2K reservoir + 2K gather scratch + sort
    // outputs ≈ 5·CAP·~52 B) is subtracted from free VRAM before the chunk is sized.
    // SINGLE-SOURCE: K_sz / CAP_sz are computed exactly ONCE here and reused below for both
    // the reservoir sizing (reservoir_bytes) and the live reservoir state (K / CAP). A
    // clamp-policy change therefore lands in one place — no sizing/state drift.
    std::size_t K_sz = (cfg.top_k > 0) ? cfg.top_k : kFstatDefaultSweepTopK;
    if (K_sz > static_cast<std::size_t>(enumerated)) K_sz = static_cast<std::size_t>(enumerated);
    if (K_sz > kFstatIntClampMax) K_sz = kFstatIntClampMax;
    if (K_sz < 1) K_sz = 1;
    const std::size_t CAP_sz = K_sz * 2;  // slack = K (proven: chunk always fits pre-compact).
    // Reservoir + gather-scratch + sort-out per CAP slot: 4 doubles + 4 ints (reservoir) +
    // 4 doubles + 4 ints (gather scratch) + 1 double (sort keys) + 2 ints (perm in/out) ≈
    // 9·8 + 10·4 = 112 B/slot. Round up generously for the CUB sort temp.
    const std::size_t reservoir_bytes = CAP_sz * kFstatReservoirBytesPerSlot;

    // CHUNK size from free VRAM. Per item the device holds: the k-int key (k·4B) + x_blocks
    // (nb_s·8) + x_loo (nb_s·8) + xtau (nb_s·8) + est/loo-total/tot_line (3·8) + diag var
    // (8) + est/se/z/absz (4·8) + 4 key columns (4·4) + flag (1) + the compacted outputs
    // (5·8 + 4·4). Bound by 0.4·(free − reservoir) to leave headroom for the CUB temp storage
    // + the resident f2 + the fixed reservoir footprint.
    std::size_t free_b = 0, total_b = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const std::size_t free_after_res = (free_b > reservoir_bytes) ? (free_b - reservoir_bytes)
                                                                  : (free_b / 4);
    const std::size_t per_item_bytes =
        static_cast<std::size_t>(k) * sizeof(int) +                       // dItems
        static_cast<std::size_t>(nb_s) * sizeof(double) * 3 +             // dX + dLoo + dXtau
        sizeof(double) * 4 +                                              // dTotal,dTotLine,dVar,(pad)
        sizeof(double) * 4 +                                             // dEst,dSe,dZ,dAbsZ
        static_cast<std::size_t>(4) * sizeof(int) +                       // 4 key cols
        sizeof(unsigned char) +                                          // flag
        sizeof(double) * 5 + static_cast<std::size_t>(4) * sizeof(int);   // compacted outs (+absz)
    std::size_t budget = static_cast<std::size_t>(static_cast<double>(free_after_res) * 0.4);
    if (budget < per_item_bytes) budget = per_item_bytes;  // at least one item
    std::size_t chunk_sz = budget / (per_item_bytes == 0 ? 1 : per_item_bytes);
    // Optional override (the chunk lever) via STEPPE_FSTAT_CHUNK.
    if (const char* env = std::getenv("STEPPE_FSTAT_CHUNK")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) chunk_sz = static_cast<std::size_t>(v);
    }
    if (chunk_sz < 1) chunk_sz = 1;
    // Clamp a chunk to INT_MAX (CUB num_items + our int kernels) and to the total.
    if (chunk_sz > kFstatIntClampMax) chunk_sz = kFstatIntClampMax;
    if (static_cast<unsigned long long>(chunk_sz) > enumerated)
        chunk_sz = static_cast<std::size_t>(enumerated);
    const int C_max = static_cast<int>(chunk_sz);

    // Allocate the per-chunk device working set ONCE (sized to C_max); reuse across chunks.
    const std::size_t Cm = static_cast<std::size_t>(C_max);
    const std::size_t kk = static_cast<std::size_t>(k);
    DeviceBuffer<int> dItems(Cm * kk);
    DeviceBuffer<double> dX(Cm * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dLoo(Cm * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dXtau(Cm * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dTotal(Cm);
    DeviceBuffer<double> dTotLine(Cm);
    DeviceBuffer<double> dVar(Cm);
    DeviceBuffer<double> dEst(Cm), dSe(Cm), dZ(Cm), dAbsZ(Cm);
    DeviceBuffer<int> dC0(Cm), dC1(Cm), dC2(Cm), dC3(Cm);
    DeviceBuffer<unsigned char> dFlags(Cm);
    // Per-chunk compacted survivors (above the CURRENT tau; bounded by C_max).
    DeviceBuffer<double> dEstSel(Cm), dSeSel(Cm), dZSel(Cm), dAbsZSel(Cm);
    DeviceBuffer<int> dC0Sel(Cm), dC1Sel(Cm), dC2Sel(Cm), dC3Sel(Cm);
    DeviceBuffer<int> dNumSel(1);

    // ---- BOUNDED DEVICE TOP-K reservoir (the fix for the unbounded-host-vector OOM) ----
    // The host NEVER accumulates survivors. Instead a FIXED CAP=O(K) device reservoir keeps
    // the running top-K (largest |z|) with a MONOTONICALLY RISING threshold tau: the zfilter
    // pre-drops items at-or-below the current K-th |z|, so the per-chunk survivor count only
    // SHRINKS as the sweep proceeds. Host RAM is O(K) (~40 MB at K=1e6) regardless of the
    // billions computed. kSweepFilterTopK = TopK (tau rises); kSweepFilterMinZ = MinZ (tau
    // pinned at min_z, but the reservoir still caps to K as a hard safety ceiling so a MinZ
    // sweep cannot OOM).
    const int zmode = cfg.filter_mode;  // kSweepFilterMinZ = fixed-tau MinZ; kSweepFilterTopK = rising-tau TopK.
    // K = the reservoir target; CAP = 2K slack so a chunk's survivors always fit before a
    // compact (each compact returns the fill to <=K, leaving >=K free headroom). K is also
    // clamped to INT_MAX (CUB num_items + our int kernels) and to the enumerated total.
    // K_sz / CAP_sz were clamped+derived ONCE above (the single-source for reservoir sizing);
    // here we only narrow them to the int kernel/CUB types.
    const int K = static_cast<int>(K_sz);
    const int CAP = (CAP_sz > static_cast<std::size_t>(INT_MAX)) ? INT_MAX
                                                                 : static_cast<int>(CAP_sz);

    // Persistent reservoir state allocated ONCE (fixed device footprint, ~80-100 B/slot).
    DeviceBuffer<double> dResEst(CAP_sz), dResSe(CAP_sz), dResZ(CAP_sz), dResAbsZ(CAP_sz);
    DeviceBuffer<int> dResC0(CAP_sz), dResC1(CAP_sz), dResC2(CAP_sz), dResC3(CAP_sz);
    // Double-buffered sort outputs (CUB SortPairsDescending, non-overwrite) + gather scratch.
    DeviceBuffer<double> dSortAbsZ(CAP_sz);    // sorted keys (descending).
    DeviceBuffer<int> dPermIn(CAP_sz), dPermOut(CAP_sz);  // iota -> sorted permutation.
    DeviceBuffer<double> dGEst(CAP_sz), dGSe(CAP_sz), dGZ(CAP_sz), dGAbsZ(CAP_sz);
    DeviceBuffer<int> dGC0(CAP_sz), dGC1(CAP_sz), dGC2(CAP_sz), dGC3(CAP_sz);
    DeviceBuffer<double> dTau(1);
    // tau floor = min_z (for TopK this is just the floor; for MinZ it stays fixed here).
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTau.data(), &cfg.min_z, sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    int res_n = 0;  // host mirror of the reservoir fill (advanced by num_sel, reset on compact).

    // CUB temp storage — size ONCE at the max num_items each call ever sees (C_max for the
    // per-chunk Flagged compaction; CAP for the reservoir SortPairsDescending) and reuse.
    // Two-call idiom: d_temp_storage=nullptr first to query temp_storage_bytes.
    std::size_t sel_bytes = 0;
    STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, sel_bytes, dEst.data(), dFlags.data(),
        dEstSel.data(), dNumSel.data(),
        static_cast<std::int64_t>(C_max), stream_.get()));
    std::size_t sort_bytes = 0;
    STEPPE_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_bytes, dResAbsZ.data(), dSortAbsZ.data(), dPermIn.data(),
        dPermOut.data(), static_cast<std::int64_t>(CAP), 0, sizeof(double) * 8, stream_.get()));
    const std::size_t cub_bytes = std::max(sel_bytes, sort_bytes);
    DeviceBuffer<unsigned char> dCubTemp(cub_bytes == 0 ? 1 : cub_bytes);

    // COMPACT-AND-RAISE: sort the reservoir by |z| descending, truncate to K, raise tau to the
    // new K-th |z|. Fires when the reservoir would overflow CAP (and once at the end). After
    // it, res_n <= K. Because |z| is monotone in significance and tau only RISES, no kept item
    // is ever wrongly evicted relative to a GLOBAL top-K (an item dropped at the zfilter when
    // |z|<=tau could never make a top-K whose K-th value is already tau).
    auto compact_and_raise = [&](int keep) {
        if (res_n <= 0) return;
        const int m = res_n;
        launch_sweep_topk_iota(dPermIn.data(), m, stream_.get());
        std::size_t sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            dCubTemp.data(), sb, dResAbsZ.data(), dSortAbsZ.data(), dPermIn.data(),
            dPermOut.data(), static_cast<std::int64_t>(m), 0, sizeof(double) * 8,
            stream_.get()));
        const int newn = (m < keep) ? m : keep;  // truncate to <=K.
        // Gather the top `newn` rows (by the sorted permutation) into scratch, then swap back.
        launch_sweep_topk_gather(
            dPermOut.data(), newn, dResEst.data(), dResSe.data(), dResZ.data(),
            dResAbsZ.data(), dResC0.data(), dResC1.data(), dResC2.data(), dResC3.data(),
            dGEst.data(), dGSe.data(), dGZ.data(), dGAbsZ.data(),
            dGC0.data(), dGC1.data(), dGC2.data(), dGC3.data(), stream_.get());
        const std::size_t db = static_cast<std::size_t>(newn) * sizeof(double);
        const std::size_t ib = static_cast<std::size_t>(newn) * sizeof(int);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResEst.data(), dGEst.data(), db, cudaMemcpyDeviceToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResSe.data(),  dGSe.data(),  db, cudaMemcpyDeviceToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResZ.data(),   dGZ.data(),   db, cudaMemcpyDeviceToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResAbsZ.data(),dGAbsZ.data(),db, cudaMemcpyDeviceToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC0.data(),  dGC0.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC1.data(),  dGC1.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC2.data(),  dGC2.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC3.data(),  dGC3.data(),  ib, cudaMemcpyDeviceToDevice, stream_.get()));
        // Raise tau to the new K-th |z| (only when the reservoir was actually FULL to K, and
        // only in TopK mode — MinZ keeps tau at the floor). dSortAbsZ is |z| descending, so
        // position K-1 is the K-th-largest. A no-op when newn < K (we have not yet seen K items).
        if (zmode == kSweepFilterTopK && newn >= K)
            launch_sweep_topk_raise_tau(dSortAbsZ.data(), K, zmode, dTau.data(), stream_.get());
        res_n = newn;
    };

    // The HOST chunk loop: advance c0, run the chunk pipeline, APPEND survivors into the
    // device reservoir (D2D — NO host transfer of survivors). NO per-item host work (no
    // enumeration, no filter — those are device kernels); the only host touch is the num_sel
    // int readback (one int per chunk).
    for (unsigned long long c0 = 0; c0 < enumerated; c0 += static_cast<unsigned long long>(C_max)) {
        const unsigned long long remaining = enumerated - c0;
        const int C = (remaining < static_cast<unsigned long long>(C_max))
                          ? static_cast<int>(remaining) : C_max;
        if (C <= 0) break;

        // (1) UNRANK -> dItems (the device index list; NO host enumeration).
        if (k == 4)
            launch_sweep_unrank_quartets(static_cast<long long>(c0), C, range,
                                         use_subset ? dSubset.data() : nullptr,
                                         dItems.data(), stream_.get());
        else
            launch_sweep_unrank_triples(static_cast<long long>(c0), C, range,
                                        use_subset ? dSubset.data() : nullptr,
                                        dItems.data(), stream_.get());

        // (2) GATHER (the SAME device kernel the explicit path uses) + loo/total + xtau +
        // diag_var. nl=C, nr=1, m=C.
        if (k == 4)
            launch_assemble_f4_quartets_gather(f2.f2_device(), P, dItems.data(), C, nb_s,
                                               dropped ? dSurv.data() : nullptr,
                                               dX.data(), stream_.get());
        else
            launch_assemble_f3_triples_gather(f2.f2_device(), P, dItems.data(), C, nb_s,
                                              dropped ? dSurv.data() : nullptr,
                                              dX.data(), stream_.get());
        launch_f4_loo_total(dX.data(), dBlockSizes.data(), C, nb_s, n,
                            dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());
        launch_f4_xtau(dLoo.data(), dTotal.data(), dTotLine.data(), dBlockSizes.data(),
                       C, nb_s, n, dXtau.data(), stream_.get());
        launch_f4_diag_var(dXtau.data(), C, nb_s, dVar.data(), stream_.get());

        // (3) |z| FILTER against the LIVE rising tau (read from dTau, not a host constant)
        // -> dEst/dSe/dZ/dAbsZ + the uint8 survivor flag (|z| > tau). Device-side, no host.
        launch_sweep_zfilter_tau(dTotal.data(), dVar.data(), C, dTau.data(),
                                 dEst.data(), dSe.data(), dZ.data(), dAbsZ.data(),
                                 dFlags.data(), stream_.get());

        // Deinterleave the key columns for compaction.
        launch_sweep_deinterleave_keys(dItems.data(), C, k,
                                       dC0.data(), dC1.data(), dC2.data(), dC3.data(),
                                       stream_.get());

        // (4) COMPACT the chunk's above-tau survivors ON THE DEVICE (CUB Flagged; num_selected
        // written on device). The SAME 8 columns (est/se/z/absz + 4 keys) with the SAME flags.
        std::size_t sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dEst.data(), dFlags.data(),
                                   dEstSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));
        sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dSe.data(), dFlags.data(),
                                   dSeSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));
        sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dZ.data(), dFlags.data(),
                                   dZSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));
        sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dAbsZ.data(), dFlags.data(),
                                   dAbsZSel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));
        sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC0.data(), dFlags.data(),
                                   dC0Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));
        sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC1.data(), dFlags.data(),
                                   dC1Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));
        sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC2.data(), dFlags.data(),
                                   dC2Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));
        sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dC3.data(), dFlags.data(),
                                   dC3Sel.data(), dNumSel.data(), static_cast<std::int64_t>(C), stream_.get()));

        int num_sel = 0;
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(&num_sel, dNumSel.data(), sizeof(int),
                                          cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        if (num_sel <= 0) continue;

        // (5) APPEND the chunk's survivors into the device reservoir (D2D). If they would
        // overflow CAP, compact-and-raise FIRST (frees >=K headroom), then append. A single
        // chunk may yield more than K survivors (the first chunks, before tau rises); append
        // in pieces of <=(CAP-res_n) and compact between, so the reservoir never overruns.
        int off = 0;
        while (off < num_sel) {
            if (res_n >= CAP) compact_and_raise(K);  // reservoir full -> truncate to K.
            int take = num_sel - off;
            if (take > CAP - res_n) take = CAP - res_n;  // fill to CAP, then compact.
            const std::size_t db = static_cast<std::size_t>(take) * sizeof(double);
            const std::size_t ib = static_cast<std::size_t>(take) * sizeof(int);
            const std::size_t doff = static_cast<std::size_t>(off);
            const std::size_t rdoff = static_cast<std::size_t>(res_n);
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResEst.data() + rdoff, dEstSel.data() + doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResSe.data()  + rdoff, dSeSel.data()  + doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResZ.data()   + rdoff, dZSel.data()   + doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResAbsZ.data()+ rdoff, dAbsZSel.data()+ doff, db, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC0.data()  + rdoff, dC0Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC1.data()  + rdoff, dC1Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC2.data()  + rdoff, dC2Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dResC3.data()  + rdoff, dC3Sel.data()  + doff, ib, cudaMemcpyDeviceToDevice, stream_.get()));
            res_n += take;
            off += take;
        }
    }

    // FINAL compact: sort the reservoir by |z| descending + truncate to min(res_n,K). The
    // device now holds EXACTLY the top-K rows (sorted) — D2H ONLY these <=K rows.
    compact_and_raise(K);
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    const std::size_t ns = static_cast<std::size_t>(res_n < 0 ? 0 : res_n);

    std::vector<double> h_est(ns), h_se(ns), h_z(ns);
    std::vector<int> h_c0(ns), h_c1(ns), h_c2(ns), h_c3(ns);
    if (ns > 0) {
        const std::size_t db = ns * sizeof(double);
        const std::size_t ib = ns * sizeof(int);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_est.data(), dResEst.data(), db, cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_se.data(),  dResSe.data(),  db, cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_z.data(),   dResZ.data(),   db, cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c0.data(),  dResC0.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c1.data(),  dResC1.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c2.data(),  dResC2.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(h_c3.data(),  dResC3.data(),  ib, cudaMemcpyDeviceToHost, stream_.get()));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }

    // Pack the bounded top-K survivor set onto the CUDA-free POD (host RAM is O(K)).
    out.keys.resize(ns);
    out.est = std::move(h_est);
    out.se = std::move(h_se);
    out.z = std::move(h_z);
    for (std::size_t r = 0; r < ns; ++r)
        out.keys[r] = {h_c0[r], h_c1[r], h_c2[r], (k >= 4 ? h_c3[r] : 0)};
    out.status = Status::Ok;
    return out;
}

F4Blocks CudaBackend::assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                   std::span<const int> left_idx,
                                   std::span<const int> right_idx,
                                   const Precision& precision) {
    // CANCELLATION CARVE-OUT (unified precision policy; fit-engine.md §1.4, §OQ-5).
    // The f4 4-slab combine is a catastrophic-cancellation f-stat difference, held
    // native FP64 ALWAYS regardless of the requested `precision` — exactly the
    // f2-numerator carve-out: Ozaki emulation faithfully forms a product but cannot
    // recover bits annihilated by a prior subtraction. So even with the fit default
    // now EmulatedFp64{40}, this stage stays native.
    (void)precision;  // native FP64 (the cancellation-sensitive f-stat diff)
    guard_device();

    const int nl = static_cast<int>(left_idx.size()) - 1;
    const int nr = static_cast<int>(right_idx.size()) - 1;
    const int nb = f2.n_block;
    const int P = f2.P;

    F4Blocks out;
    out.nl = nl;
    out.nr = nr;
    const std::size_t m = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
    out.x_total.assign(m, 0.0);
    tot_line_.assign(m, 0.0);
    if (nl <= 0 || nr <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
        out.n_block = 0;
        return out;
    }

    // F1 / OQ-12 — MISSING-block drop (AT2 read_f2(remove_na=TRUE)), the GPU mirror
    // of the CpuBackend oracle. The keep-mask is computed ON-DEVICE from the resident
    // Vpair (launch_f2_block_keep, sharing core::pair_block_is_missing with the
    // oracle), read down as a tiny [nb] int vector, and the host builds the ASCENDING
    // SURVIVOR id list + the survivor block_sizes. The gather then COMPACTS dX onto
    // the survivor axis (d_surv maps compacted→original block). With NO Vpair (the
    // legacy upload path that leaves it empty) or no missing block, every block
    // survives ⇒ d_surv is identity and the path is bit-identical to pre-F1.
    const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
    const int nb_s = static_cast<int>(surv.size());
    out.n_block = nb_s;
    out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        out.block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    if (nb_s <= 0) return out;  // all blocks missing — degenerate (caller-gated)

    // Whether the survivor set differs from the full set (a real drop occurred);
    // identity survivor ⇒ pass d_surv = nullptr (the bit-identical no-drop kernel arm).
    const bool dropped = (nb_s != nb);

    // H2D the small model index vectors (length nl+1 / nr+1) + the SURVIVOR block
    // sizes + (only when a drop occurred) the survivor map.
    DeviceBuffer<int> dLeft(left_idx.size());
    DeviceBuffer<int> dRight(right_idx.size());
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dLeft.data(), left_idx.data(),
                                      left_idx.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dRight.data(), right_idx.data(),
                                      right_idx.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), out.block_sizes.data(),
                                      static_cast<std::size_t>(nb_s) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (dropped)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

    // Device-resident X / loo / total / tot_line (sized to the SURVIVOR block count).
    DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dTotal(m);
    DeviceBuffer<double> dTotLine(m);

    // S3 gather (reads RESIDENT f2; native FP64 4-slab combine), compacted onto the
    // survivor axis via d_surv (nullptr when no drop ⇒ identity, bit-identical).
    launch_assemble_f4_gather(f2.f2_device(), P, dLeft.data(), dRight.data(),
                              nl, nr, nb_s, dropped ? dSurv.data() : nullptr,
                              dX.data(), stream_.get());

    // n = Σ SURVIVOR block_sizes (host int → double; the jackknife normalizer).
    long long n_ll = 0;
    for (int v : out.block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    // est_to_loo + x_total + tot_line (on-device reduction over SURVIVORS; FP64 order).
    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

    // D2H the small fit intermediates across the CUDA-free seam (SURVIVOR-sized).
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_blocks.data(), dX.data(),
                                      m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_loo.data(), dLoo.data(),
                                      m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_total.data(), dTotal.data(),
                                      m * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(tot_line_.data(), dTotLine.data(),
                                      m * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

F4Blocks CudaBackend::assemble_f4(const F2BlockTensor& f2,
                                   std::span<const int> left_idx,
                                   std::span<const int> right_idx,
                                   const Precision& precision) {
    (void)f2; (void)left_idx; (void)right_idx; (void)precision;
    throw std::runtime_error(
        "CudaBackend::assemble_f4(host): the GPU path reads DEVICE-RESIDENT f2 "
        "(assemble_f4(DeviceF2Blocks)); the host-tensor overload is the CpuBackend "
        "oracle door");
}

F4Blocks CudaBackend::assemble_f4_quartets(
    const steppe::device::DeviceF2Blocks& f2,
    std::span<const int> quartets,
    const Precision& precision) {
    (void)precision;  // native FP64 (the cancellation-sensitive f-stat diff)
    guard_device();

    const int N = static_cast<int>(quartets.size()) / 4;  // quartet count (m axis)
    const int nb = f2.n_block;
    const int P = f2.P;

    F4Blocks out;
    out.nl = N;  // m = nl*nr = N (the batched f4 m-axis convention; matches the oracle)
    out.nr = 1;
    const std::size_t m = static_cast<std::size_t>(N);
    out.x_total.assign(m, 0.0);
    tot_line_.assign(m, 0.0);
    if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
        out.n_block = 0;
        return out;
    }

    // F1 / OQ-12 SURVIVOR drop (the SAME on-device keep-mask the fit path uses).
    const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
    const int nb_s = static_cast<int>(surv.size());
    out.n_block = nb_s;
    out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        out.block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    if (nb_s <= 0) return out;  // all blocks missing — degenerate (caller-gated)

    const bool dropped = (nb_s != nb);

    // H2D the flattened quartet quad array (4*N) + the SURVIVOR block sizes + (only
    // on a real drop) the survivor map.
    DeviceBuffer<int> dQuartets(quartets.size());
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuartets.data(), quartets.data(),
                                      quartets.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), out.block_sizes.data(),
                                      static_cast<std::size_t>(nb_s) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (dropped)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

    DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dTotal(m);
    DeviceBuffer<double> dTotLine(m);

    // S3 per-quartet gather (reads RESIDENT f2; native FP64 4-slab combine).
    launch_assemble_f4_quartets_gather(f2.f2_device(), P, dQuartets.data(), N, nb_s,
                                       dropped ? dSurv.data() : nullptr,
                                       dX.data(), stream_.get());

    long long n_ll = 0;
    for (int v : out.block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    // est_to_loo + x_total + tot_line (REUSE the fit kernel; it reads only m = N).
    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

    // D2H the small fit intermediates across the CUDA-free seam (SURVIVOR-sized).
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_blocks.data(), dX.data(),
                                      m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_loo.data(), dLoo.data(),
                                      m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_total.data(), dTotal.data(),
                                      m * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(tot_line_.data(), dTotLine.data(),
                                      m * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

F4Blocks CudaBackend::assemble_f4_quartets(const F2BlockTensor& f2,
                                            std::span<const int> quartets,
                                            const Precision& precision) {
    (void)f2; (void)quartets; (void)precision;
    throw std::runtime_error(
        "CudaBackend::assemble_f4_quartets(host): the GPU path reads DEVICE-RESIDENT "
        "f2 (assemble_f4_quartets(DeviceF2Blocks)); the host-tensor overload is the "
        "CpuBackend oracle door");
}

F4Blocks CudaBackend::assemble_f3_triples(
    const steppe::device::DeviceF2Blocks& f2,
    std::span<const int> triples,
    const Precision& precision) {
    (void)precision;  // native FP64 (the cancellation-sensitive f-stat diff)
    guard_device();

    const int N = static_cast<int>(triples.size()) / 3;  // triple count (m axis)
    const int nb = f2.n_block;
    const int P = f2.P;

    F4Blocks out;
    out.nl = N;  // m = nl*nr = N (the batched f3 m-axis convention; matches the oracle)
    out.nr = 1;
    const std::size_t m = static_cast<std::size_t>(N);
    out.x_total.assign(m, 0.0);
    tot_line_.assign(m, 0.0);
    if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
        out.n_block = 0;
        return out;
    }

    // F1 / OQ-12 SURVIVOR drop (the SAME on-device keep-mask the fit/f4 path uses).
    const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
    const int nb_s = static_cast<int>(surv.size());
    out.n_block = nb_s;
    out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        out.block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    if (nb_s <= 0) return out;  // all blocks missing — degenerate (caller-gated)

    const bool dropped = (nb_s != nb);

    // H2D the flattened triple array (3*N) + the SURVIVOR block sizes + (only on a real
    // drop) the survivor map.
    DeviceBuffer<int> dTriples(triples.size());
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTriples.data(), triples.data(),
                                      triples.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), out.block_sizes.data(),
                                      static_cast<std::size_t>(nb_s) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (dropped)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

    DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dTotal(m);
    DeviceBuffer<double> dTotLine(m);

    // S3 per-triple gather (reads RESIDENT f2; native FP64 3-slab combine).
    launch_assemble_f3_triples_gather(f2.f2_device(), P, dTriples.data(), N, nb_s,
                                      dropped ? dSurv.data() : nullptr,
                                      dX.data(), stream_.get());

    long long n_ll = 0;
    for (int v : out.block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    // est_to_loo + x_total + tot_line (REUSE the fit/f4 kernel; it reads only m = N).
    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

    // D2H the small fit intermediates across the CUDA-free seam (SURVIVOR-sized).
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_blocks.data(), dX.data(),
                                      m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_loo.data(), dLoo.data(),
                                      m * static_cast<std::size_t>(nb_s) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.x_total.data(), dTotal.data(),
                                      m * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(tot_line_.data(), dTotLine.data(),
                                      m * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

F4Blocks CudaBackend::assemble_f3_triples(const F2BlockTensor& f2,
                                           std::span<const int> triples,
                                           const Precision& precision) {
    (void)f2; (void)triples; (void)precision;
    throw std::runtime_error(
        "CudaBackend::assemble_f3_triples(host): the GPU path reads DEVICE-RESIDENT "
        "f2 (assemble_f3_triples(DeviceF2Blocks)); the host-tensor overload is the "
        "CpuBackend oracle door");
}

SweepSurvivors CudaBackend::f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                      const SweepConfig& cfg,
                                      const Precision& precision) {
    return run_fstat_sweep_device(f2, cfg, precision, /*k=*/4);
}

SweepSurvivors CudaBackend::f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                      const SweepConfig& cfg,
                                      const Precision& precision) {
    return run_fstat_sweep_device(f2, cfg, precision, /*k=*/3);
}

}  // namespace steppe::device
