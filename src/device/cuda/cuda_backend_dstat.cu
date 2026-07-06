// src/device/cuda/cuda_backend_dstat.cu
//
// CudaBackend translation unit for the qpDstat D-statistic and f4-ratio: the
// per-block num/den/cnt reduction plus the device-resident block jackknives that run
// over resident data with no per-block copy back to the host. A private CUDA TU of
// steppe_device, split out of cuda_backend.cu with the bodies moved verbatim.
//
// Reference: docs/reference/src_device_cuda_cuda_backend_dstat.cu.md
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/internal/nvtx.hpp"
#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/device_decode_result_impl.cuh"
#include "device/cuda/device_f2_blocks_impl.cuh"
#include "device/cuda/dstat_kernel.cuh"
#include "device/cuda/qpadm_fit_kernels.cuh"
#include "device/cuda/ratio_block_jackknife_kernel.cuh"

namespace steppe::device {

// D-statistic block reduction — reference §3
void CudaBackend::dstat_block_reduce_device(const double* dQ, const double* dV, int P, long M,
                               const int* block_id, int n_block,
                               std::span<const int> quadruples,
                               double* numsum, double* densum, double* cnt) {
    const int N = static_cast<int>(quadruples.size() / 4);
    if (P <= 0 || M <= 0 || N <= 0 || n_block <= 0) return;

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

    h2d_async(dQuad, quadruples.data(), nq, stream_.get());
    h2d_async(dBegin, begin.data(), static_cast<std::size_t>(n_block), stream_.get());
    h2d_async(dSize, size.data(), static_cast<std::size_t>(n_block), stream_.get());

    launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                              dBegin.data(), dSize.data(), n_block,
                              dNum.data(), dDen.data(), dCnt.data(), stream_.get());

    d2h_async(numsum, dNum, nb_out, stream_.get());
    d2h_async(densum, dDen, nb_out, stream_.get());
    d2h_async(cnt, dCnt, nb_out, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
}

void CudaBackend::dstat_block_reduce(const double* Q, const double* V, int P, long M,
                        const int* block_id, int n_block,
                        std::span<const int> quadruples,
                        double* numsum, double* densum, double* cnt) {
    guard_device();
    const int N = static_cast<int>(quadruples.size() / 4);
    if (P <= 0 || M <= 0 || N <= 0 || n_block <= 0) return;
    const std::size_t pm = static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
    DeviceBuffer<double> dQ(pm), dV(pm);
    h2d_async(dQ, Q, pm, stream_.get());
    h2d_async(dV, V, pm, stream_.get());
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

// f4-ratio block jackknife (resident f2) — reference §5
RatioBlockJackknife CudaBackend::f4ratio_blocks_jackknife(
    const steppe::device::DeviceF2Blocks& f2, std::span<const int> flat, int N,
    double setmiss_thresh, const Precision& precision) {
    (void)precision;
    guard_device();
    STEPPE_NVTX_RANGE("jackknife");
    RatioBlockJackknife jk;
    jk.N = N;
    const int nb = f2.n_block;
    const int P = f2.P;
    const std::size_t m = static_cast<std::size_t>(2) * static_cast<std::size_t>(N);
    if (N <= 0 || nb <= 0 || f2.f2_device() == nullptr) {
        jk.est.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.se.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.z.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
        jk.status = Status::Ok;
        return jk;
    }

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

    const int M2 = 2 * N;
    DeviceBuffer<int> dQuartets(flat.size());
    DeviceBuffer<int> dBlockSizes(static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dBlockSizesD(static_cast<std::size_t>(nb_s));
    DeviceBuffer<int> dSurv(static_cast<std::size_t>(dropped ? nb_s : 1));
    h2d_async(dQuartets, flat.data(), flat.size(), stream_.get());
    h2d_async(dBlockSizes, block_sizes.data(), static_cast<std::size_t>(nb_s), stream_.get());
    h2d_async(dBlockSizesD, block_sizes_d.data(), static_cast<std::size_t>(nb_s), stream_.get());
    if (dropped)
        h2d_async(dSurv, surv.data(), static_cast<std::size_t>(nb_s), stream_.get());

    DeviceBuffer<double> dX(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dLoo(m * static_cast<std::size_t>(nb_s));
    DeviceBuffer<double> dTotal(m);
    DeviceBuffer<double> dTotLine(m);

    launch_assemble_f4_quartets_gather(f2.f2_device(), P, dQuartets.data(), M2, nb_s,
                                       dropped ? dSurv.data() : nullptr,
                                       dX.data(), stream_.get());
    long long n_ll = 0;
    for (int v : block_sizes) n_ll += v;
    const double n = static_cast<double>(n_ll);
    launch_f4_loo_total(dX.data(), dBlockSizes.data(),
                        static_cast<int>(m), nb_s, n,
                        dLoo.data(), dTotal.data(), dTotLine.data(), stream_.get());

    const long ms = static_cast<long>(m);
    DRatioJackArray dnum{dLoo.data(), 0, 1, ms};
    DRatioJackArray dden{dLoo.data(), static_cast<long>(N), 1, ms};
    DRatioJackArray dweight{dBlockSizesD.data(), 0, 0, 1};
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
    d2h_async(jk.est.data(), dEst, static_cast<std::size_t>(N), stream_.get());
    d2h_async(jk.se.data(), dSe, static_cast<std::size_t>(N), stream_.get());
    d2h_async(jk.z.data(), dZ, static_cast<std::size_t>(N), stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    jk.status = Status::Ok;
    return jk;
}

// qpDstat block jackknife (resident decode) — reference §6
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
    h2d_async(dQuad, quadruples.data(), nq, stream_.get());
    h2d_async(dBegin, begin.data(), static_cast<std::size_t>(n_block), stream_.get());
    h2d_async(dSize, size.data(), static_cast<std::size_t>(n_block), stream_.get());

    launch_dstat_block_reduce(dQ, dV, P, M, dQuad.data(), N,
                              dBegin.data(), dSize.data(), n_block,
                              dNum.data(), dDen.data(), dCnt.data(), stream_.get());

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
    d2h_async(jk.est.data(), dEst, static_cast<std::size_t>(N), stream_.get());
    d2h_async(jk.se.data(), dSe, static_cast<std::size_t>(N), stream_.get());
    d2h_async(jk.z.data(), dZ, static_cast<std::size_t>(N), stream_.get());
    d2h_async(jk.p.data(), dP, static_cast<std::size_t>(N), stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    jk.status = Status::Ok;
    return jk;
}

// Host-overload oracle-door throws — reference §8
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
