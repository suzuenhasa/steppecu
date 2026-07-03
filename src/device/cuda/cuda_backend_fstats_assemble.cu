// src/device/cuda/cuda_backend_fstats_assemble.cu
//
// CudaBackend TU for the device-resident f4/f3 assemble doors and the on-device
// all-combination f4/f3 sweep. Split out of cuda_backend.cu (bodies moved
// verbatim), so it joins the same steppe_device target and inherits its codegen.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_fstats_assemble.cu.md
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/internal/nvtx.hpp"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/device_f2_blocks_impl.cuh"
#include "device/cuda/f2_block_kernel.cuh"
#include "device/cuda/qpadm_fit_kernels.cuh"
#include "steppe/config.hpp"

namespace steppe::device {

// device_survivor_blocks — missing-block survivor filter — reference §3
std::vector<int> CudaBackend::device_survivor_blocks(
    const steppe::device::DeviceF2Blocks& f2, int nb, int P) {
    std::vector<int> surv;
    if (nb <= 0) return surv;
    surv.reserve(static_cast<std::size_t>(nb));
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

// run_fstat_sweep_device — all-combination sweep: pipeline, top-K reservoir, VRAM sizing — reference §6–§8
SweepSurvivors CudaBackend::run_fstat_sweep_device(
    const steppe::device::DeviceF2Blocks& f2, const SweepConfig& cfg,
    const Precision& precision, int k) {
    (void)precision;
    guard_device();

    constexpr std::size_t kFstatIntClampMax = 0x40000000;
    constexpr std::size_t kFstatReservoirBytesPerSlot = 160;

    SweepSurvivors out;
    const int P = f2.P;
    const int nb_full = f2.n_block;
    const int range = cfg.pop_subset.empty() ? P : static_cast<int>(cfg.pop_subset.size());

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
        out.status = Status::Ok;
        return out;
    }

    const std::vector<int> surv = device_survivor_blocks(f2, nb_full, P);
    const int nb_s = static_cast<int>(surv.size());
    if (nb_s <= 0) { out.status = Status::Ok; return out; }
    const bool dropped = (nb_s != nb_full);

    std::vector<int> surv_block_sizes(static_cast<std::size_t>(nb_s), 0);
    long long n_ll = 0;
    for (int bs = 0; bs < nb_s; ++bs) {
        const int orig = surv[static_cast<std::size_t>(bs)];
        const int sz = f2.block_sizes[static_cast<std::size_t>(orig)];
        surv_block_sizes[static_cast<std::size_t>(bs)] = sz;
        n_ll += sz;
    }
    const double n = static_cast<double>(n_ll);

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

    std::size_t K_sz = (cfg.top_k > 0) ? cfg.top_k : kFstatDefaultSweepTopK;
    if (K_sz > static_cast<std::size_t>(enumerated)) K_sz = static_cast<std::size_t>(enumerated);
    if (K_sz > kFstatIntClampMax) K_sz = kFstatIntClampMax;
    if (K_sz < 1) K_sz = 1;
    const std::size_t CAP_sz = K_sz * 2;
    const std::size_t reservoir_bytes = CAP_sz * kFstatReservoirBytesPerSlot;

    std::size_t free_b = 0, total_b = 0;
    STEPPE_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const std::size_t free_after_res = (free_b > reservoir_bytes) ? (free_b - reservoir_bytes)
                                                                  : (free_b / 4);
    const std::size_t per_item_bytes =
        static_cast<std::size_t>(k) * sizeof(int) +
        static_cast<std::size_t>(nb_s) * sizeof(double) * 3 +
        sizeof(double) * 4 +
        sizeof(double) * 4 +
        static_cast<std::size_t>(4) * sizeof(int) +
        sizeof(unsigned char) +
        sizeof(double) * 5 + static_cast<std::size_t>(4) * sizeof(int);
    std::size_t budget = static_cast<std::size_t>(static_cast<double>(free_after_res) * 0.4);
    if (budget < per_item_bytes) budget = per_item_bytes;
    std::size_t chunk_sz = budget / (per_item_bytes == 0 ? 1 : per_item_bytes);
    if (const char* env = std::getenv("STEPPE_FSTAT_CHUNK")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) chunk_sz = static_cast<std::size_t>(v);
    }
    if (chunk_sz < 1) chunk_sz = 1;
    if (chunk_sz > kFstatIntClampMax) chunk_sz = kFstatIntClampMax;
    if (static_cast<unsigned long long>(chunk_sz) > enumerated)
        chunk_sz = static_cast<std::size_t>(enumerated);
    const int C_max = static_cast<int>(chunk_sz);

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
    DeviceBuffer<double> dEstSel(Cm), dSeSel(Cm), dZSel(Cm), dAbsZSel(Cm);
    DeviceBuffer<int> dC0Sel(Cm), dC1Sel(Cm), dC2Sel(Cm), dC3Sel(Cm);
    DeviceBuffer<int> dNumSel(1);

    const int zmode = cfg.filter_mode;
    const int K = static_cast<int>(K_sz);
    const int CAP = (CAP_sz > static_cast<std::size_t>(INT_MAX)) ? INT_MAX
                                                                 : static_cast<int>(CAP_sz);

    DeviceBuffer<double> dResEst(CAP_sz), dResSe(CAP_sz), dResZ(CAP_sz), dResAbsZ(CAP_sz);
    DeviceBuffer<int> dResC0(CAP_sz), dResC1(CAP_sz), dResC2(CAP_sz), dResC3(CAP_sz);
    DeviceBuffer<double> dSortAbsZ(CAP_sz);
    DeviceBuffer<int> dPermIn(CAP_sz), dPermOut(CAP_sz);
    DeviceBuffer<double> dGEst(CAP_sz), dGSe(CAP_sz), dGZ(CAP_sz), dGAbsZ(CAP_sz);
    DeviceBuffer<int> dGC0(CAP_sz), dGC1(CAP_sz), dGC2(CAP_sz), dGC3(CAP_sz);
    DeviceBuffer<double> dTau(1);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dTau.data(), &cfg.min_z, sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    int res_n = 0;

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

    auto compact_and_raise = [&](int keep) {
        if (res_n <= 0) return;
        const int m = res_n;
        launch_sweep_topk_iota(dPermIn.data(), m, stream_.get());
        std::size_t sb = cub_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            dCubTemp.data(), sb, dResAbsZ.data(), dSortAbsZ.data(), dPermIn.data(),
            dPermOut.data(), static_cast<std::int64_t>(m), 0, sizeof(double) * 8,
            stream_.get()));
        const int newn = (m < keep) ? m : keep;
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
        if (zmode == kSweepFilterTopK && newn >= K)
            launch_sweep_topk_raise_tau(dSortAbsZ.data(), K, zmode, dTau.data(), stream_.get());
        res_n = newn;
    };

    for (unsigned long long c0 = 0; c0 < enumerated; c0 += static_cast<unsigned long long>(C_max)) {
        const unsigned long long remaining = enumerated - c0;
        const int C = (remaining < static_cast<unsigned long long>(C_max))
                          ? static_cast<int>(remaining) : C_max;
        if (C <= 0) break;

        if (k == 4)
            launch_sweep_unrank_quartets(static_cast<long long>(c0), C, range,
                                         use_subset ? dSubset.data() : nullptr,
                                         dItems.data(), stream_.get());
        else
            launch_sweep_unrank_triples(static_cast<long long>(c0), C, range,
                                        use_subset ? dSubset.data() : nullptr,
                                        dItems.data(), stream_.get());

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

        launch_sweep_zfilter_tau(dTotal.data(), dVar.data(), C, dTau.data(),
                                 dEst.data(), dSe.data(), dZ.data(), dAbsZ.data(),
                                 dFlags.data(), stream_.get());

        launch_sweep_deinterleave_keys(dItems.data(), C, k,
                                       dC0.data(), dC1.data(), dC2.data(), dC3.data(),
                                       stream_.get());

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

        int off = 0;
        while (off < num_sel) {
            if (res_n >= CAP) compact_and_raise(K);
            int take = num_sel - off;
            if (take > CAP - res_n) take = CAP - res_n;
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

    out.keys.resize(ns);
    out.est = std::move(h_est);
    out.se = std::move(h_se);
    out.z = std::move(h_z);
    for (std::size_t r = 0; r < ns; ++r)
        out.keys[r] = {h_c0[r], h_c1[r], h_c2[r], (k >= 4 ? h_c3[r] : 0)};
    out.status = Status::Ok;
    return out;
}

// assemble_f4 — device-resident assemble door — reference §4
F4Blocks CudaBackend::assemble_f4(const steppe::device::DeviceF2Blocks& f2,
                                   std::span<const int> left_idx,
                                   std::span<const int> right_idx,
                                   const Precision& precision) {
    (void)precision;
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

    const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
    const int nb_s = static_cast<int>(surv.size());
    out.n_block = nb_s;
    out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        out.block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    if (nb_s <= 0) return out;

    const bool dropped = (nb_s != nb);

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

    DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dTotal(m);
    DeviceBuffer<double> dTotLine(m);

    launch_assemble_f4_gather(f2.f2_device(), P, dLeft.data(), dRight.data(),
                              nl, nr, nb_s, dropped ? dSurv.data() : nullptr,
                              dX.data(), stream_.get());

    long long n_ll = 0;
    for (int v : out.block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

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

// assemble_f4 host-tensor twin — throws (CpuBackend oracle door) — reference §5
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

// assemble_f4_quartets — device-resident assemble door — reference §4
F4Blocks CudaBackend::assemble_f4_quartets(
    const steppe::device::DeviceF2Blocks& f2,
    std::span<const int> quartets,
    const Precision& precision) {
    (void)precision;
    guard_device();

    const int N = static_cast<int>(quartets.size()) / 4;
    const int nb = f2.n_block;
    const int P = f2.P;

    F4Blocks out;
    out.nl = N;
    out.nr = 1;
    const std::size_t m = static_cast<std::size_t>(N);
    out.x_total.assign(m, 0.0);
    tot_line_.assign(m, 0.0);
    if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
        out.n_block = 0;
        return out;
    }

    const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
    const int nb_s = static_cast<int>(surv.size());
    out.n_block = nb_s;
    out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        out.block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    if (nb_s <= 0) return out;

    const bool dropped = (nb_s != nb);

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

    launch_assemble_f4_quartets_gather(f2.f2_device(), P, dQuartets.data(), N, nb_s,
                                       dropped ? dSurv.data() : nullptr,
                                       dX.data(), stream_.get());

    long long n_ll = 0;
    for (int v : out.block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

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

// assemble_f4_quartets host-tensor twin — throws — reference §5
F4Blocks CudaBackend::assemble_f4_quartets(const F2BlockTensor& f2,
                                            std::span<const int> quartets,
                                            const Precision& precision) {
    (void)f2; (void)quartets; (void)precision;
    throw std::runtime_error(
        "CudaBackend::assemble_f4_quartets(host): the GPU path reads DEVICE-RESIDENT "
        "f2 (assemble_f4_quartets(DeviceF2Blocks)); the host-tensor overload is the "
        "CpuBackend oracle door");
}

// assemble_f3_triples — device-resident assemble door — reference §4
F4Blocks CudaBackend::assemble_f3_triples(
    const steppe::device::DeviceF2Blocks& f2,
    std::span<const int> triples,
    const Precision& precision) {
    (void)precision;
    guard_device();

    const int N = static_cast<int>(triples.size()) / 3;
    const int nb = f2.n_block;
    const int P = f2.P;

    F4Blocks out;
    out.nl = N;
    out.nr = 1;
    const std::size_t m = static_cast<std::size_t>(N);
    out.x_total.assign(m, 0.0);
    tot_line_.assign(m, 0.0);
    if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
        out.n_block = 0;
        return out;
    }

    const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
    const int nb_s = static_cast<int>(surv.size());
    out.n_block = nb_s;
    out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
    for (int bs = 0; bs < nb_s; ++bs)
        out.block_sizes[static_cast<std::size_t>(bs)] =
            f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
    out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    out.x_loo.assign(m * static_cast<std::size_t>(nb_s), 0.0);
    if (nb_s <= 0) return out;

    const bool dropped = (nb_s != nb);

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

    launch_assemble_f3_triples_gather(f2.f2_device(), P, dTriples.data(), N, nb_s,
                                      dropped ? dSurv.data() : nullptr,
                                      dX.data(), stream_.get());

    long long n_ll = 0;
    for (int v : out.block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);

    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

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

// assemble_f3_triples host-tensor twin — throws — reference §5
F4Blocks CudaBackend::assemble_f3_triples(const F2BlockTensor& f2,
                                           std::span<const int> triples,
                                           const Precision& precision) {
    (void)f2; (void)triples; (void)precision;
    throw std::runtime_error(
        "CudaBackend::assemble_f3_triples(host): the GPU path reads DEVICE-RESIDENT "
        "f2 (assemble_f3_triples(DeviceF2Blocks)); the host-tensor overload is the "
        "CpuBackend oracle door");
}

// f4_sweep / f3_sweep — thin k-dispatch wrappers — reference §9
SweepSurvivors CudaBackend::f4_sweep(const steppe::device::DeviceF2Blocks& f2,
                                      const SweepConfig& cfg,
                                      const Precision& precision) {
    return run_fstat_sweep_device(f2, cfg, precision, 4);
}

SweepSurvivors CudaBackend::f3_sweep(const steppe::device::DeviceF2Blocks& f2,
                                      const SweepConfig& cfg,
                                      const Precision& precision) {
    return run_fstat_sweep_device(f2, cfg, precision, 3);
}

}
