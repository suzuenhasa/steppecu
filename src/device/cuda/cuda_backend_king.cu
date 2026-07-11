// src/device/cuda/cuda_backend_king.cu
//
// CudaBackend override for the KING-robust kinship pair sweep (`steppe kinship`). It uploads
// the packed diploid genotype tile device-resident, then streams the SNP axis by tile (the
// fst all-pairs / extract_f2 s_lo idiom): per tile it decodes the per-individual dosage code
// ONCE into a compact N x tileM byte tensor (launch_king_dosage_decode) and folds every pair's
// five KING counts into the persistent per-pair accumulators (launch_king_allpairs_accumulate).
// Only the five small per-pair count vectors cross PCIe; the N x M code tensor is never
// materialized (only an N x tileM window is resident) and the pair combine stays on the GPU.
// Integer counting (native FP64 tag only). A CUDA TU private to steppe_device, mirroring
// cuda_backend_fst_allpairs.cu.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/king_allpairs_kernel.cuh"
#include "steppe/config.hpp"  // kFstatMaxComb

namespace steppe::device {

namespace {
// Fraction of free VRAM the resident N x tileM code tensor may occupy, leaving room for the
// packed tile + the per-pair accumulators.
constexpr std::size_t kKingCodeBudgetDivisor = 4;
constexpr std::size_t kKingFreeVramFallbackBytes = std::size_t{1} << 30;  // 1 GiB
// Per-chunk pair-count clamp so grid.x = C stays under kMaxGridX (mirrors the fst clamp).
constexpr long long kKingPairChunkClamp = 0x40000000LL;  // 2^30
}  // namespace

KingMatrix CudaBackend::king_robust_all_pairs(const DecodeTileView& tile,
                                              std::span<const std::uint8_t> summary_include,
                                              std::span<const int> pairs_i,
                                              std::span<const int> pairs_j, bool sure) {
    guard_device();
    KingMatrix out;
    out.precision_tag = Precision::Kind::Fp64;

    const long M = static_cast<long>(tile.n_snp);
    const int N = tile.n_pop;  // singleton partition: one pop per individual
    out.N = N;
    if (M <= 0 || N <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }

    // Pair set: an explicit list (biobank-scale --pairs) or the full C(N,2) upper triangle.
    const bool explicit_pairs = !pairs_i.empty();
    long long npairs = 0;
    if (explicit_pairs) {
        if (pairs_i.size() != pairs_j.size()) {
            out.status = Status::InvalidConfig;
            return out;
        }
        npairs = static_cast<long long>(pairs_i.size());
    } else {
        npairs = static_cast<long long>(N) * static_cast<long long>(N - 1) / 2;
        // Maxcomb cap: refuse a runaway pair count unless --sure (all-pairs only; the explicit
        // list is user-bounded). Mirrors fst_wc_all_pairs.
        if (static_cast<unsigned long long>(npairs) > kFstatMaxComb && !sure) {
            out.capped = true;
            out.enumerated = static_cast<std::size_t>(npairs);
            out.status = Status::InvalidConfig;
            return out;
        }
    }
    out.enumerated = static_cast<std::size_t>(npairs);

    const std::size_t np = static_cast<std::size_t>(npairs);
    out.nsnp.assign(np, 0L);
    out.hethet.assign(np, 0L);
    out.ibs0.assign(np, 0L);
    out.het_i.assign(np, 0L);
    out.het_j.assign(np, 0L);
    out.status = Status::Ok;
    if (npairs == 0) return out;  // N < 2 or an empty --pairs list: a well-formed empty result.

    // Upload the packed tile (individual-major; singleton pops -> record g == sample g).
    DeviceBuffer<std::uint8_t> dPacked;  // staging; empty when the tile is device-resident
    const std::uint8_t* packed_dev = packed_device_ptr(tile, dPacked);

    // Optional autosome mask, indexed by the GLOBAL SNP position in the kernel.
    const std::size_t Mz = static_cast<std::size_t>(M);
    const bool have_inc = summary_include.size() == Mz;
    DeviceBuffer<std::uint8_t> dInclude(have_inc ? Mz : 1u);
    if (have_inc) {
        h2d_async(dInclude, summary_include.data(), Mz, stream_.get());
    }

    // Optional explicit pair index arrays.
    DeviceBuffer<int> dPairsI(explicit_pairs ? np : 1u);
    DeviceBuffer<int> dPairsJ(explicit_pairs ? np : 1u);
    if (explicit_pairs) {
        h2d_async(dPairsI, pairs_i.data(), np, stream_.get());
        h2d_async(dPairsJ, pairs_j.data(), np, stream_.get());
    }

    // Persistent per-pair count accumulators (reduced across all SNP-tiles by the kernel's +=).
    DeviceBuffer<long> dNsnp(np), dHetHet(np), dIbs0(np), dHetI(np), dHetJ(np);
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dNsnp.data(), 0, np * sizeof(long), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetHet.data(), 0, np * sizeof(long), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dIbs0.data(), 0, np * sizeof(long), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetI.data(), 0, np * sizeof(long), stream_.get()));
    STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetJ.data(), 0, np * sizeof(long), stream_.get()));

    // Choose a SNP tile that bounds the resident N x tileM code tensor to ~free/divisor.
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kKingFreeVramFallbackBytes;
    const std::size_t row_bytes = static_cast<std::size_t>(N);  // one byte per sample per SNP
    std::size_t budget = free_b / kKingCodeBudgetDivisor;
    if (budget < row_bytes) budget = row_bytes;
    long tileM = static_cast<long>(budget / row_bytes);
    if (const char* env = std::getenv("STEPPE_KING_TILE")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) tileM = v;
    }
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;

    const std::size_t pane = static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM);
    DeviceBuffer<std::uint8_t> dCode(pane == 0 ? 1u : pane);

    const std::uint8_t* d_inc = have_inc ? dInclude.data() : nullptr;
    const int* d_pi = explicit_pairs ? dPairsI.data() : nullptr;
    const int* d_pj = explicit_pairs ? dPairsJ.data() : nullptr;
    const long long chunk = std::min<long long>(npairs, kKingPairChunkClamp);

    for (long s_lo = 0; s_lo < M; s_lo += tileM) {
        const long tm = std::min<long>(tileM, M - s_lo);
        launch_king_dosage_decode(packed_dev, tile.bytes_per_record, N, s_lo, tm, dCode.data(),
                                  stream_.get());
        for (long long pair0 = 0; pair0 < npairs; pair0 += chunk) {
            const long long C = std::min<long long>(chunk, npairs - pair0);
            launch_king_allpairs_accumulate(dCode.data(), N, tm, s_lo, d_inc, d_pi, d_pj, pair0, C,
                                            /*out_offset=*/0, dNsnp.data(), dHetHet.data(),
                                            dIbs0.data(), dHetI.data(), dHetJ.data(), stream_.get());
        }
    }

    d2h_async(out.nsnp.data(), dNsnp, np, stream_.get());
    d2h_async(out.hethet.data(), dHetHet, np, stream_.get());
    d2h_async(out.ibs0.data(), dIbs0, np, stream_.get());
    d2h_async(out.het_i.data(), dHetI, np, stream_.get());
    d2h_async(out.het_j.data(), dHetJ, np, stream_.get());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return out;
}

}  // namespace steppe::device
