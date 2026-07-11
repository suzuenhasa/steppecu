// src/device/cuda/cuda_backend_fst.cu
//
// CudaBackend override for the per-site Weir & Cockerham 1984 FST (`steppe fst`). It
// uploads the packed genotype tile device-resident, resolves the two population segments
// from the tile's pop_offsets, launches the per-site WC kernel over the tile, then runs a
// device-resident (native-FP64) Σnum/Σden/n_valid reduction masked to the summary-eligible
// valid sites (autosomes) via cub::DeviceReduce. Only the per-SNP arrays + the three
// summary scalars cross PCIe; the variance-component compute is all on the GPU. A CUDA TU
// private to steppe_device.
#include <cub/device/device_reduce.cuh>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/fst_wc_kernel.cuh"

namespace steppe::device {

FstPerSite CudaBackend::fst_wc_per_site(const DecodeTileView& tile, int popA, int popB,
                                        std::span<const std::uint8_t> summary_include) {
    guard_device();
    FstPerSite out;
    out.precision_tag = Precision::Kind::Fp64;

    const long M = static_cast<long>(tile.n_snp);
    const int P = tile.n_pop;
    if (M <= 0 || P <= 0 || popA < 0 || popB < 0 || popA >= P || popB >= P) {
        return out;
    }

    const std::size_t Mz = static_cast<std::size_t>(M);
    out.num.assign(Mz, 0.0);
    out.den.assign(Mz, 0.0);
    out.fst.assign(Mz, 0.0);
    out.valid.assign(Mz, std::uint8_t{0});

    const std::size_t segA_begin = tile.pop_offsets[static_cast<std::size_t>(popA)];
    const std::size_t segA_end = tile.pop_offsets[static_cast<std::size_t>(popA) + 1];
    const std::size_t segB_begin = tile.pop_offsets[static_cast<std::size_t>(popB)];
    const std::size_t segB_end = tile.pop_offsets[static_cast<std::size_t>(popB) + 1];

    // Upload the whole packed tile device-resident (individual-major, population-contiguous).
    DeviceBuffer<std::uint8_t> dPacked;  // staging; empty when the tile is device-resident
    const std::uint8_t* packed_dev = packed_device_ptr(tile, dPacked);

    DeviceBuffer<double> dNum(Mz), dDen(Mz), dFst(Mz);
    DeviceBuffer<std::uint8_t> dValid(Mz);
    launch_fst_wc(packed_dev, tile.bytes_per_record, segA_begin, segA_end, segB_begin,
                  segB_end, M, dNum.data(), dDen.data(), dFst.data(), dValid.data(),
                  stream_.get());

    // Genome-wide summary: Σnum/Σden/n_valid over (valid && summary-included) sites,
    // reduced ON-DEVICE in native FP64.
    const bool have_inc = summary_include.size() == Mz;
    DeviceBuffer<std::uint8_t> dInclude(have_inc ? Mz : 1u);
    if (have_inc) {
        h2d_async(dInclude, summary_include.data(), Mz, stream_.get());
    }
    DeviceBuffer<double> dCnum(Mz), dCden(Mz);
    DeviceBuffer<long> dCnt(Mz);
    launch_fst_summary_contrib(dNum.data(), dDen.data(), dValid.data(),
                               have_inc ? dInclude.data() : nullptr, M, dCnum.data(),
                               dCden.data(), dCnt.data(), stream_.get());

    DeviceBuffer<double> dSumNum(1), dSumDen(1);
    DeviceBuffer<long> dSumCnt(1);
    {
        std::size_t tmp_d = 0, tmp_c = 0;
        STEPPE_CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, tmp_d, dCnum.data(), dSumNum.data(),
                                                 static_cast<int>(M), stream_.get()));
        STEPPE_CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, tmp_c, dCnt.data(), dSumCnt.data(),
                                                 static_cast<int>(M), stream_.get()));
        const std::size_t tmp_bytes = tmp_d > tmp_c ? tmp_d : tmp_c;
        DeviceBuffer<unsigned char> dTmp(tmp_bytes == 0 ? 1u : tmp_bytes);
        std::size_t sb = tmp_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceReduce::Sum(dTmp.data(), sb, dCnum.data(), dSumNum.data(),
                                                 static_cast<int>(M), stream_.get()));
        sb = tmp_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceReduce::Sum(dTmp.data(), sb, dCden.data(), dSumDen.data(),
                                                 static_cast<int>(M), stream_.get()));
        sb = tmp_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceReduce::Sum(dTmp.data(), sb, dCnt.data(), dSumCnt.data(),
                                                 static_cast<int>(M), stream_.get()));
    }

    long n_valid = 0;
    d2h_async(out.num.data(), dNum, Mz, stream_.get());
    d2h_async(out.den.data(), dDen, Mz, stream_.get());
    d2h_async(out.fst.data(), dFst, Mz, stream_.get());
    d2h_async(out.valid.data(), dValid, Mz, stream_.get());
    d2h_async(&out.sum_num, dSumNum, 1, stream_.get());
    d2h_async(&out.sum_den, dSumDen, 1, stream_.get());
    d2h_async(&n_valid, dSumCnt, 1, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    out.n_valid = n_valid;
    return out;
}

}  // namespace steppe::device
