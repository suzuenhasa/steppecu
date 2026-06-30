// src/device/cuda/cuda_backend_dstat.cu
//
// CudaBackend — qpDstat block-reduce + shared ratio-block-jackknife subsystem TU
// (cuda_backend.cu split T4; docs/kimiactions/05-cuda-backend-split.md §2.3 TU-D).
// Out-of-line homes of `CudaBackend::dstat_block_reduce_device` (the resident per-block
// num/den/cnt D-statistic reduction core), its `dstat_block_reduce` host-pointer +
// DeviceDecodeResult overloads, and the device-resident ratio-block-jackknife family:
// `f4ratio_blocks_jackknife(DeviceF2Blocks)` (S3 per-quartet 4-slab gather + est_to_loo/
// total → the SHARED launch_ratio_block_jackknife over resident dX/dLoo, no [m·nb_s] D2H),
// `dstat_blocks_jackknife(DeviceDecodeResult)` (resident dNum/dDen/dCnt → the SAME shared
// jackknife, tot_mode=1), and the two host-overload oracle-door throw-twins. Bodies MOVED
// VERBATIM from cuda_backend.cu; nothing about codegen / math / precision / file-order
// changed by the split.
//
// CROSS-TU: `f4ratio_blocks_jackknife(DeviceF2Blocks)` calls `device_survivor_blocks` — a
// header-declared member fn still DEFINED in the cuda_backend.cu remainder at T4 (moves to
// the fstats-assemble TU at T6); it links via the cuda_backend.cuh declaration.
//
// This is a CUDA TU: PRIVATE to steppe_device (architecture.md §4). Joins the SAME
// steppe_device target, so it inherits identical codegen/macros/RDC.
#include <algorithm>  // std::max — the NaN-fill row clamp in the empty-N / empty-survivor guards
#include <cmath>      // std::nan — the empty/degenerate-jackknife NaN fill
#include <stdexcept>  // std::runtime_error — the host-overload oracle-door throw-twins
#include <vector>     // std::vector — block begin/size + survivor + jk.est/se/z/p host buffers

#include "core/domain/block_partition_rule.hpp"           // core::block_ranges / BlockRange (per-block SNP layout; the assign_blocks inverse)
#include "core/internal/nvtx.hpp"                          // STEPPE_NVTX_RANGE (coarse "jackknife" phase-boundary marker)
#include "device/cuda/cuda_backend.cuh"                    // the CudaBackend class declaration (split T0)
#include "device/cuda/check.cuh"                           // STEPPE_CUDA_CHECK
#include "device/cuda/device_decode_result_impl.cuh"      // DeviceDecodeResult::Impl (resident Q/V owners behind dec.q_device/v_device)
#include "device/cuda/device_f2_blocks_impl.cuh"          // DeviceF2Blocks::Impl (resident f2 owner behind f2.f2_device)
#include "device/cuda/dstat_kernel.cuh"                    // launch_dstat_block_reduce (the per-block num/den/cnt D-stat reduction)
#include "device/cuda/qpadm_fit_kernels.cuh"              // launch_assemble_f4_quartets_gather + launch_f4_loo_total (f4ratio S3 gather + est_to_loo/total)
#include "device/cuda/ratio_block_jackknife_kernel.cuh"   // launch_ratio_block_jackknife + struct DRatioJackArray (the SHARED resident jackknife)

namespace steppe::device {

void CudaBackend::dstat_block_reduce_device(const double* dQ, const double* dV, int P, long M,
                               const int* block_id, int n_block,
                               std::span<const int> quadruples,
                               double* numsum, double* densum, double* cnt) {
    const int N = static_cast<int>(quadruples.size() / 4);
    if (P <= 0 || M <= 0 || N <= 0 || n_block <= 0) return;

    // Per-block contiguous SNP layout from block_id (the SINGLE-SOURCE inverse of
    // assign_blocks; same primitive the f2 path uses). begin/size as int (M fits int
    // by the same B22 guard the f2 path enforces upstream).
    const std::vector<core::BlockRange> ranges =
        core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                           M, n_block);
    std::vector<int> begin(static_cast<std::size_t>(n_block));
    std::vector<int> size(static_cast<std::size_t>(n_block));
    for (int b = 0; b < n_block; ++b) {
        begin[static_cast<std::size_t>(b)] = static_cast<int>(ranges[static_cast<std::size_t>(b)].begin);
        size[static_cast<std::size_t>(b)]  = static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }

    const std::size_t nq = static_cast<std::size_t>(N) * 4u;
    const std::size_t nb_out = static_cast<std::size_t>(N) * static_cast<std::size_t>(n_block);

    DeviceBuffer<int> dQuad(nq);
    DeviceBuffer<int> dBegin(static_cast<std::size_t>(n_block));
    DeviceBuffer<int> dSize(static_cast<std::size_t>(n_block));
    DeviceBuffer<double> dNum(nb_out), dDen(nb_out), dCnt(nb_out);

    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuad.data(), quadruples.data(), nq * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBegin.data(), begin.data(),
                                      static_cast<std::size_t>(n_block) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSize.data(), size.data(),
                                      static_cast<std::size_t>(n_block) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));

    launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                              dBegin.data(), dSize.data(), n_block,
                              dNum.data(), dDen.data(), dCnt.data(), stream_.get());

    STEPPE_CUDA_CHECK(cudaMemcpyAsync(numsum, dNum.data(), nb_out * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(densum, dDen.data(), nb_out * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(cnt, dCnt.data(), nb_out * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
}

void CudaBackend::dstat_block_reduce(const double* Q, const double* V, int P, long M,
                        const int* block_id, int n_block,
                        std::span<const int> quadruples,
                        double* numsum, double* densum, double* cnt) {
    guard_device();
    const int N = static_cast<int>(quadruples.size() / 4);
    if (P <= 0 || M <= 0 || N <= 0 || n_block <= 0) return;
    // Host-pointer path (the CpuBackend-parity entry / non-resident callers): H2D
    // Q/V into resident buffers, then the shared core. The DeviceDecodeResult
    // overload below SKIPS this H2D (the M4 cure).
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    DeviceBuffer<double> dQ(pm), dV(pm);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQ.data(), Q, pm * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dV.data(), V, pm * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    dstat_block_reduce_device(dQ.data(), dV.data(), P, M, block_id, n_block,
                              quadruples, numsum, densum, cnt);
}

void CudaBackend::dstat_block_reduce(const steppe::device::DeviceDecodeResult& dec,
                        const int* block_id, int n_block,
                        std::span<const int> quadruples,
                        double* numsum, double* densum, double* cnt) {
    guard_device();
    if (dec.empty() || dec.q_device() == nullptr) return;
    dstat_block_reduce_device(dec.q_device(), dec.v_device(), dec.P, dec.M_kept,
                              block_id, n_block, quadruples, numsum, densum, cnt);
}

RatioBlockJackknife CudaBackend::f4ratio_blocks_jackknife(
    const steppe::device::DeviceF2Blocks& f2, std::span<const int> flat, int N,
    double setmiss_thresh, const Precision& precision) {
    (void)precision;  // native FP64 (the cancellation-sensitive ratio diff)
    guard_device();
    STEPPE_NVTX_RANGE("jackknife");  // coarse phase boundary: f4ratio block-jackknife
    RatioBlockJackknife jk;
    jk.N = N;
    const int nb = f2.n_block;
    const int P = f2.P;
    const std::size_t m = static_cast<std::size_t>(2) * static_cast<std::size_t>(N);  // 2N rows
    if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
        jk.est.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.se.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.z.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.status = Status::Ok;
        return jk;
    }

    // F1 / OQ-12 SURVIVOR drop (the SAME on-device keep-mask the assemble path uses).
    const std::vector<int> surv = device_survivor_blocks(f2, nb, P);
    const int nb_s = static_cast<int>(surv.size());
    if (nb_s <= 0) {
        jk.est.assign(static_cast<std::size_t>(N), std::nan(""));
        jk.se.assign(static_cast<std::size_t>(N), std::nan(""));
        jk.z.assign(static_cast<std::size_t>(N), std::nan(""));
        jk.status = Status::Ok;
        return jk;
    }
    std::vector<int> block_sizes(static_cast<std::size_t>(nb_s));
    std::vector<double> block_sizes_d(static_cast<std::size_t>(nb_s));
    for (int bs = 0; bs < nb_s; ++bs) {
        const int v = f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
        block_sizes[static_cast<std::size_t>(bs)] = v;
        block_sizes_d[static_cast<std::size_t>(bs)] = static_cast<double>(v);
    }
    const bool dropped = (nb_s != nb);

    // The interleaved num/den quartet flat array is the SAME `flat` (length 8N == 4*2N).
    const int M2 = 2 * N;  // the m-axis quartet count (num then den halves).
    DeviceBuffer<int> dQuartets(flat.size());
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dBlockSizesD(static_cast<std::size_t>(nb_s));
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuartets.data(), flat.data(),
                                      flat.size() * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizes.data(), block_sizes.data(),
                                      static_cast<std::size_t>(nb_s) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBlockSizesD.data(), block_sizes_d.data(),
                                      static_cast<std::size_t>(nb_s) * sizeof(double),
                                      cudaMemcpyHostToDevice, stream_.get()));
    if (dropped)
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSurv.data(), surv.data(),
                                          static_cast<std::size_t>(nb_s) * sizeof(int),
                                          cudaMemcpyHostToDevice, stream_.get()));

    DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dTotal(m);
    DeviceBuffer<double> dTotLine(m);

    // S3 per-quartet gather (RESIDENT f2; native FP64 4-slab combine) + est_to_loo/total.
    launch_assemble_f4_quartets_gather(f2.f2_device(), P, dQuartets.data(), M2, nb_s,
                                       dropped ? dSurv.data() : nullptr,
                                       dX.data(), stream_.get());
    long long n_ll = 0;
    for (int v : block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);
    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

    // The SHARED ratio-block-jackknife over the RESIDENT dX/dLoo — NO [m·nb_s] D2H.
    // num/den ARE the est_to_loo replicates dLoo (rows k / N+k); xblk_num/xblk_den ARE the
    // per-block f4 dX (rows k / N+k); weight = block_sizes broadcast. dX/dLoo are
    // COLUMN-MAJOR in the m axis: element [row + m*b] ⇒ item_stride=1, block_stride=m.
    const long ms = static_cast<long>(m);
    DRatioJackArray dnum{dLoo.data(), 0, 1, ms};
    DRatioJackArray dden{dLoo.data(), static_cast<long>(N), 1, ms};
    DRatioJackArray dweight{dBlockSizesD.data(), 0, 0, 1};  // broadcast across items.
    DRatioJackArray dxbn{dX.data(), 0, 1, ms};
    DRatioJackArray dxbd{dX.data(), static_cast<long>(N), 1, ms};

    DeviceBuffer<double> dEst(static_cast<std::size_t>(N));
    DeviceBuffer<double> dSe(static_cast<std::size_t>(N));
    DeviceBuffer<double> dZ(static_cast<std::size_t>(N));
    launch_ratio_block_jackknife(dnum, dden, dweight, dxbn, dxbd, N, nb_s,
                                 /*tot_mode=*/0, setmiss_thresh, /*compute_p=*/false,
                                 dEst.data(), dSe.data(), dZ.data(), /*d_p=*/nullptr,
                                 stream_.get());

    jk.est.assign(static_cast<std::size_t>(N), 0.0);
    jk.se.assign(static_cast<std::size_t>(N), 0.0);
    jk.z.assign(static_cast<std::size_t>(N), 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.est.data(), dEst.data(),
                                      static_cast<std::size_t>(N) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.se.data(), dSe.data(),
                                      static_cast<std::size_t>(N) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.z.data(), dZ.data(),
                                      static_cast<std::size_t>(N) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    jk.status = Status::Ok;
    return jk;
}

RatioBlockJackknife CudaBackend::dstat_blocks_jackknife(
    const steppe::device::DeviceDecodeResult& dec, const int* block_id, int n_block,
    std::span<const int> quadruples) {
    guard_device();
    const int N = static_cast<int>(quadruples.size() / 4);
    RatioBlockJackknife jk;
    jk.N = N;
    if (dec.empty() || dec.q_device() == nullptr || N <= 0 || n_block <= 0) {
        jk.est.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.se.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.z.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.p.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.status = Status::Ok;
        return jk;
    }
    const double* dQ = dec.q_device();
    const double* dV = dec.v_device();
    const int P = dec.P;
    const long M = dec.M_kept;

    // Per-block contiguous SNP layout from block_id (the SAME inverse of assign_blocks the
    // dstat_block_reduce_device path uses).
    const std::vector<core::BlockRange> ranges =
        core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                           M, n_block);
    std::vector<int> begin(static_cast<std::size_t>(n_block));
    std::vector<int> size(static_cast<std::size_t>(n_block));
    for (int b = 0; b < n_block; ++b) {
        begin[static_cast<std::size_t>(b)] = static_cast<int>(ranges[static_cast<std::size_t>(b)].begin);
        size[static_cast<std::size_t>(b)]  = static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }
    const std::size_t nq = static_cast<std::size_t>(N) * 4u;
    const std::size_t nb_out = static_cast<std::size_t>(N) * static_cast<std::size_t>(n_block);

    DeviceBuffer<int> dQuad(nq);
    DeviceBuffer<int> dBegin(static_cast<std::size_t>(n_block));
    DeviceBuffer<int> dSize(static_cast<std::size_t>(n_block));
    DeviceBuffer<double> dNum(nb_out), dDen(nb_out), dCnt(nb_out);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dQuad.data(), quadruples.data(), nq * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dBegin.data(), begin.data(),
                                      static_cast<std::size_t>(n_block) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(dSize.data(), size.data(),
                                      static_cast<std::size_t>(n_block) * sizeof(int),
                                      cudaMemcpyHostToDevice, stream_.get()));

    launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                              dBegin.data(), dSize.data(), n_block,
                              dNum.data(), dDen.data(), dCnt.data(), stream_.get());

    // The SHARED ratio-block-jackknife over the RESIDENT dNum/dDen/dCnt — NO [N·nb] D2H.
    // numsum/densum/cnt are ROW-MAJOR [N × n_block], element [k*nb+b] ⇒ item_stride=n_block,
    // block_stride=1. xblk pair is null (dstat tot_mode=1 never reads it).
    const long nbl = static_cast<long>(n_block);
    DRatioJackArray dnum{dNum.data(), 0, nbl, 1};
    DRatioJackArray dden{dDen.data(), 0, nbl, 1};
    DRatioJackArray dweight{dCnt.data(), 0, nbl, 1};
    DRatioJackArray dnull{nullptr, 0, 0, 0};

    DeviceBuffer<double> dEst(static_cast<std::size_t>(N));
    DeviceBuffer<double> dSe(static_cast<std::size_t>(N));
    DeviceBuffer<double> dZ(static_cast<std::size_t>(N));
    DeviceBuffer<double> dP(static_cast<std::size_t>(N));
    launch_ratio_block_jackknife(dnum, dden, dweight, dnull, dnull, N, n_block,
                                 /*tot_mode=*/1, /*setmiss_thresh=*/0.0, /*compute_p=*/true,
                                 dEst.data(), dSe.data(), dZ.data(), dP.data(),
                                 stream_.get());

    jk.est.assign(static_cast<std::size_t>(N), 0.0);
    jk.se.assign(static_cast<std::size_t>(N), 0.0);
    jk.z.assign(static_cast<std::size_t>(N), 0.0);
    jk.p.assign(static_cast<std::size_t>(N), 0.0);
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.est.data(), dEst.data(),
                                      static_cast<std::size_t>(N) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.se.data(), dSe.data(),
                                      static_cast<std::size_t>(N) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.z.data(), dZ.data(),
                                      static_cast<std::size_t>(N) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemcpyAsync(jk.p.data(), dP.data(),
                                      static_cast<std::size_t>(N) * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream_.get()));
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    jk.status = Status::Ok;
    return jk;
}

RatioBlockJackknife CudaBackend::f4ratio_blocks_jackknife(
    const F2BlockTensor& f2, std::span<const int> flat, int N, double setmiss_thresh,
    const Precision& precision) {
    (void)f2; (void)flat; (void)N; (void)setmiss_thresh; (void)precision;
    throw std::runtime_error(
        "CudaBackend::f4ratio_blocks_jackknife(host): the GPU path reads DEVICE-RESIDENT f2 "
        "(the DeviceF2Blocks overload); the host-tensor overload is the CpuBackend oracle door");
}

RatioBlockJackknife CudaBackend::dstat_blocks_jackknife(
    const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
    std::span<const int> quadruples) {
    (void)Q; (void)V; (void)P; (void)M; (void)block_id; (void)n_block; (void)quadruples;
    throw std::runtime_error(
        "CudaBackend::dstat_blocks_jackknife(host): the GPU path reads the DEVICE-RESIDENT "
        "decode (the DeviceDecodeResult overload); the host-pointer overload is the CpuBackend "
        "oracle door");
}

}  // namespace steppe::device
