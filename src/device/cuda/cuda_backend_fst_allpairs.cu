// src/device/cuda/cuda_backend_fst_allpairs.cu
// Reference: docs/reference/src_device_cuda_cuda_backend_fst_allpairs.cu.md
//
// CudaBackend override for the all-pairs Weir & Cockerham 1984 FST matrix
// (`steppe fst --all-pairs`). It uploads the packed genotype tile device-resident, then
// streams the SNP axis by tile (the extract_f2 / PCA s_lo idiom): per tile it decodes the
// per-(pop, SNP) sufficient statistic {n, ac, het} ONCE (launch_fst_suffstat_decode) and
// folds every C(P,2) pair's wc_finalize into the persistent per-pair Σnum/Σden/n_valid
// (launch_fst_allpairs_accumulate). Only the three small C(P,2) pair vectors cross PCIe;
// the P x M x 3 sufficient-stat tensor is never materialized (only a P x tileM window is
// resident) and the P^2 combine stays on the GPU. Native FP64 (the reduction carve-out).
// A CUDA TU private to steppe_device, mirroring cuda_backend_fst.cu / cuda_backend_pca.cu.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/fst_allpairs_kernel.cuh"
#include "steppe/config.hpp"  // kFstatMaxComb

namespace steppe::device {

namespace {
// Fraction of free VRAM the resident sufficient-stat tensor (3 * P * tileM doubles) may
// occupy, leaving room for the packed tile + the per-pair accumulators.
constexpr std::size_t kFstSuffstatBudgetDivisor = 4;
constexpr std::size_t kFstFreeVramFallbackBytes = std::size_t{1} << 30;  // 1 GiB
// Per-chunk pair-count clamp so the flat pair index stays comfortably int-addressable and
// the launch grid stays under kMaxGridX (mirrors the sweep's kFstatIntClampMax).
constexpr long long kFstPairChunkClamp = 0x40000000LL;  // 2^30
}  // namespace

FstMatrix CudaBackend::fst_wc_all_pairs(const DecodeTileView& tile,
                                        std::span<const std::uint8_t> summary_include, bool sure) {
    guard_device();
    FstMatrix out;
    out.precision_tag = Precision::Kind::Fp64;

    const long M = static_cast<long>(tile.n_snp);
    const int P = tile.n_pop;
    out.P = P;
    if (M <= 0 || P <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }

    const long long npairs =
        static_cast<long long>(P) * static_cast<long long>(P - 1) / 2;
    out.enumerated = static_cast<std::size_t>(npairs);

    // Maxcomb cap: refuse a runaway pair count unless --sure (mirrors the f-stat sweep). The
    // cap is on the PAIR COUNT only, not on the C(P,2)*M accumulation volume.
    if (static_cast<unsigned long long>(npairs) > kFstatMaxComb && !sure) {
        out.capped = true;
        out.status = Status::InvalidConfig;
        return out;
    }

    out.pair_num.assign(static_cast<std::size_t>(npairs), 0.0);
    out.pair_den.assign(static_cast<std::size_t>(npairs), 0.0);
    out.pair_cnt.assign(static_cast<std::size_t>(npairs), 0L);
    out.status = Status::Ok;
    if (npairs == 0) return out;  // P < 2: a well-formed empty (diagonal-only) matrix.

    // Upload the packed tile (individual-major, population-contiguous) + the pop offsets.
    const std::size_t packed_bytes = tile.n_individuals * tile.bytes_per_record;
    DeviceBuffer<std::uint8_t> dPacked(packed_bytes == 0 ? 1u : packed_bytes);
    if (packed_bytes > 0) {
        h2d_async(dPacked, tile.packed, packed_bytes, stream_.get());
    }
    DeviceBuffer<std::size_t> dPopOff(static_cast<std::size_t>(P) + 1u);
    h2d_async(dPopOff, tile.pop_offsets, static_cast<std::size_t>(P) + 1u, stream_.get());

    // Optional autosome summary mask, indexed by the GLOBAL SNP position in the kernel.
    const std::size_t Mz = static_cast<std::size_t>(M);
    const bool have_inc = summary_include.size() == Mz;
    DeviceBuffer<std::uint8_t> dInclude(have_inc ? Mz : 1u);
    if (have_inc) {
        h2d_async(dInclude, summary_include.data(), Mz, stream_.get());
    }

    // Persistent per-pair accumulators (reduced across all SNP-tiles by the kernel's +=).
    const std::size_t np = static_cast<std::size_t>(npairs);
    DeviceBuffer<double> dPairNum(np), dPairDen(np);
    DeviceBuffer<long> dPairCnt(np);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dPairNum.data(), 0, np * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dPairDen.data(), 0, np * sizeof(double), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dPairCnt.data(), 0, np * sizeof(long), stream_.get()));

    // Choose a SNP tile that bounds the resident {n, ac, het} tensor (3 * P * tileM doubles)
    // to ~free/divisor, keeping the decode window O(P * tileM) (the extract_f2 tiling idiom).
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kFstFreeVramFallbackBytes;
    const std::size_t row_bytes = static_cast<std::size_t>(P) * 3u * sizeof(double);
    std::size_t budget = free_b / kFstSuffstatBudgetDivisor;
    if (budget < row_bytes) budget = row_bytes;
    long tileM = static_cast<long>(budget / row_bytes);
    if (const char* env = std::getenv("STEPPE_FST_TILE")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) tileM = v;
    }
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;

    const std::size_t pane = static_cast<std::size_t>(P) * static_cast<std::size_t>(tileM);
    DeviceBuffer<double> dN(pane), dAc(pane), dHet(pane);

    const std::uint8_t* d_inc = have_inc ? dInclude.data() : nullptr;
    const long long chunk = std::min<long long>(npairs, kFstPairChunkClamp);

    for (long s_lo = 0; s_lo < M; s_lo += tileM) {
        const long tm = std::min<long>(tileM, M - s_lo);
        launch_fst_suffstat_decode(dPacked.data(), tile.bytes_per_record, dPopOff.data(), P,
                                   s_lo, tm, dN.data(), dAc.data(), dHet.data(), stream_.get());
        for (long long pair0 = 0; pair0 < npairs; pair0 += chunk) {
            const long long C = std::min<long long>(chunk, npairs - pair0);
            launch_fst_allpairs_accumulate(dN.data(), dAc.data(), dHet.data(), P, tm, s_lo, d_inc,
                                           pair0, C, dPairNum.data(), dPairDen.data(),
                                           dPairCnt.data(), stream_.get());
        }
    }

    d2h_async(out.pair_num.data(), dPairNum, np, stream_.get());
    d2h_async(out.pair_den.data(), dPairDen, np, stream_.get());
    d2h_async(out.pair_cnt.data(), dPairCnt, np, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

}  // namespace steppe::device
