// src/device/cuda/cuda_backend_king_streamed.cu
//
// CudaBackend::king_robust_filtered — the biobank-scale STREAMED/compacted KING-robust all-pairs
// sweep behind `steppe kinship --min-kinship` and `--king-cutoff`. It reuses the dense KING path's
// decode + one-block-per-pair fold (king_allpairs_kernel) VERBATIM; the ONE design change is the
// loop nest. The dense path (cuda_backend_king.cu) is SNP-tile-OUTER / pair-INNER and reduces into
// PERSISTENT 5*C(N,2) accumulators — O(C(N,2)) device memory, which forces the ~14k cap. This path
// is pair-BLOCK-OUTER / SNP-tile-INNER: only one pair-block's 5*B accumulators persist, so per
// block it (1) folds the block's B pairs over the WHOLE SNP axis, (2) computes phi via the SHARED
// king_phi and flags survivors (phi vs the emit threshold), and (3) cub::DeviceSelect::Flagged
// compacts the survivor SoA {i, j, nsnp, hethet, ibs0, phi}; only the survivors cross PCIe. Peak
// device memory is 5*B + O(survivors) — NEVER 5*C(N,2) — so the cap does not bind. Because the
// fold + king_phi are the exact shared primitives the dense path uses, the emitted phi is
// bit-identical to king_robust_all_pairs (the values gate).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include <cub/device/device_select.cuh>

#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/king_allpairs_kernel.cuh"
#include "device/cuda/king_streamed_kernel.cuh"

namespace steppe::device {

namespace {
// Fraction of free VRAM the resident N x tileM code pane may occupy (mirrors the dense path).
constexpr std::size_t kKingCodeBudgetDivisor = 4;
constexpr std::size_t kKingFreeVramFallbackBytes = std::size_t{1} << 30;  // 1 GiB
// Per-accumulate-launch pair-count clamp so grid.x = C stays < kMaxGridX (mirrors the dense path).
constexpr long long kKingPairChunkClamp = 0x40000000LL;  // 2^30
// Bytes per pair the streamed block holds: 5 long accumulators (40) + the compaction inputs
// {i,j (int), phi (double), flag (byte)} (17) + the six selected survivor columns {i,j (int),
// nsnp,hethet,ibs0 (long), phi (double)} (40). The pair-block B is sized so B * this <= a fraction
// of free VRAM, leaving room for the code pane, the packed tile, and the cub temp buffer.
constexpr std::size_t kKingStreamPerPairBytes = 40 + 17 + 40;
// Free-VRAM fraction the pair-block arrays may claim (the remainder covers the code pane + tile).
constexpr std::size_t kKingStreamBlockVramDivisor = 8;
}  // namespace

KingStreamResult CudaBackend::king_robust_filtered(const DecodeTileView& tile,
                                                   std::span<const std::uint8_t> summary_include,
                                                   double emit_threshold, bool strict_greater) {
    guard_device();
    KingStreamResult out;
    out.precision_tag = Precision::Kind::Fp64;

    const long M = static_cast<long>(tile.n_snp);
    const int N = tile.n_pop;  // singleton partition: one pop per individual
    out.N = N;
    if (M <= 0 || N <= 0) {
        out.status = Status::InvalidConfig;
        return out;
    }
    const long long npairs = static_cast<long long>(N) * static_cast<long long>(N - 1) / 2;
    out.enumerated = static_cast<std::size_t>(npairs);
    out.status = Status::Ok;
    if (npairs == 0) return out;  // N < 2: a well-formed empty result (no cap: memory is O(surv)).

    // Upload the packed diploid tile (individual-major; singleton pops -> record g == sample g).
    DeviceBuffer<std::uint8_t> dPacked;  // staging; empty when the tile is device-resident
    const std::uint8_t* packed_dev = packed_device_ptr(tile, dPacked);

    // Optional autosome mask, indexed by the GLOBAL SNP position in the fold kernel.
    const std::size_t Mz = static_cast<std::size_t>(M);
    const bool have_inc = summary_include.size() == Mz;
    DeviceBuffer<std::uint8_t> dInclude(have_inc ? Mz : 1u);
    if (have_inc) h2d_async(dInclude, summary_include.data(), Mz, stream_.get());
    const std::uint8_t* d_inc = have_inc ? dInclude.data() : nullptr;

    // SNP tile: bound the resident N x tileM code pane to ~free/divisor (mirrors the dense path).
    std::size_t free_b = capabilities().free_vram_bytes;
    if (free_b == 0) free_b = kKingFreeVramFallbackBytes;
    const std::size_t row_bytes = static_cast<std::size_t>(N);
    std::size_t code_budget = free_b / kKingCodeBudgetDivisor;
    if (code_budget < row_bytes) code_budget = row_bytes;
    long tileM = static_cast<long>(code_budget / row_bytes);
    if (const char* env = std::getenv("STEPPE_KING_TILE")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) tileM = v;
    }
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;
    const std::size_t pane = static_cast<std::size_t>(N) * static_cast<std::size_t>(tileM);
    DeviceBuffer<std::uint8_t> dCode(pane == 0 ? 1u : pane);

    // Pair-block B: bound the per-block arrays to ~free / divisor. The streamed invariant is that
    // ONLY B (not C(N,2)) pairs are resident at once, so B can stay far below npairs at biobank N.
    long long B = static_cast<long long>((free_b / kKingStreamBlockVramDivisor) /
                                         kKingStreamPerPairBytes);
    if (const char* env = std::getenv("STEPPE_KING_BLOCK")) {
        const long long v = std::strtoll(env, nullptr, 10);
        if (v > 0) B = v;
    }
    if (B < 1) B = 1;
    if (B > npairs) B = npairs;
    const std::size_t Bz = static_cast<std::size_t>(B);

    // Persistent per-BLOCK accumulators (block-local index: the fold writes r - block_pair0).
    DeviceBuffer<long> dNsnp(Bz), dHetHet(Bz), dIbs0(Bz), dHetI(Bz), dHetJ(Bz);
    // Compaction inputs (block-local) + selected survivor columns.
    DeviceBuffer<int> dI(Bz), dJ(Bz);
    DeviceBuffer<double> dPhi(Bz);
    DeviceBuffer<std::uint8_t> dFlag(Bz);
    DeviceBuffer<int> dISel(Bz), dJSel(Bz);
    DeviceBuffer<long> dNsnpSel(Bz), dHetHetSel(Bz), dIbs0Sel(Bz);
    DeviceBuffer<double> dPhiSel(Bz);
    DeviceBuffer<int> dNumSel(1);

    // cub::DeviceSelect::Flagged temp — size once at the max item count (B), reuse per column/block
    // (the two-call sizing idiom already used in cuda_backend_fstats_assemble.cu).
    std::size_t sel_bytes = 0;
    STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, sel_bytes, dNsnp.data(), dFlag.data(),
                                                 dNsnpSel.data(), dNumSel.data(),
                                                 static_cast<std::int64_t>(B), stream_.get()));
    DeviceBuffer<unsigned char> dCubTemp(sel_bytes == 0 ? 1 : sel_bytes);

    // Reusable host staging for the survivors of one block.
    std::vector<int> hI, hJ;
    std::vector<long> hNsnp, hHetHet, hIbs0;
    std::vector<double> hPhi;

    for (long long block_pair0 = 0; block_pair0 < npairs; block_pair0 += B) {
        const long long blockC = std::min<long long>(B, npairs - block_pair0);
        const std::size_t blockCz = static_cast<std::size_t>(blockC);

        // Fresh accumulators for this block's pairs.
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dNsnp.data(), 0, blockCz * sizeof(long), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetHet.data(), 0, blockCz * sizeof(long), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dIbs0.data(), 0, blockCz * sizeof(long), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetI.data(), 0, blockCz * sizeof(long), stream_.get()));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetJ.data(), 0, blockCz * sizeof(long), stream_.get()));

        // Fold this block's B pairs over the WHOLE SNP axis (SNP-tile INNER). Decode is re-run per
        // block (cheap 2-bit unpack vs the fold); only 5*B accumulators persist across the tiles.
        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            launch_king_dosage_decode(packed_dev, tile.bytes_per_record, N, s_lo, tm,
                                      dCode.data(), stream_.get());
            for (long long sub = 0; sub < blockC; sub += kKingPairChunkClamp) {
                const long long C = std::min<long long>(kKingPairChunkClamp, blockC - sub);
                launch_king_allpairs_accumulate(
                    dCode.data(), N, tm, s_lo, d_inc, /*pairs_i=*/nullptr, /*pairs_j=*/nullptr,
                    /*pair0=*/block_pair0 + sub, C, /*out_offset=*/block_pair0, dNsnp.data(),
                    dHetHet.data(), dIbs0.data(), dHetI.data(), dHetJ.data(), stream_.get());
            }
        }

        // Per-block phi + emit flag, then compact the survivor SoA. king_phi here == the dense host
        // finalize's king_phi over identical integer counts -> the survivors are bit-identical.
        launch_king_streamed_flag(dNsnp.data(), dHetHet.data(), dIbs0.data(), dHetI.data(),
                                  dHetJ.data(), N, block_pair0, blockC, emit_threshold,
                                  strict_greater, dI.data(), dJ.data(), dPhi.data(), dFlag.data(),
                                  stream_.get());

        const std::int64_t items = static_cast<std::int64_t>(blockC);
        std::size_t sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dI.data(), dFlag.data(),
                                                     dISel.data(), dNumSel.data(), items,
                                                     stream_.get()));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dJ.data(), dFlag.data(),
                                                     dJSel.data(), dNumSel.data(), items,
                                                     stream_.get()));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dNsnp.data(),
                                                     dFlag.data(), dNsnpSel.data(), dNumSel.data(),
                                                     items, stream_.get()));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dHetHet.data(),
                                                     dFlag.data(), dHetHetSel.data(),
                                                     dNumSel.data(), items, stream_.get()));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dIbs0.data(),
                                                     dFlag.data(), dIbs0Sel.data(), dNumSel.data(),
                                                     items, stream_.get()));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dPhi.data(), dFlag.data(),
                                                     dPhiSel.data(), dNumSel.data(), items,
                                                     stream_.get()));

        int num_sel = 0;
        d2h_async(&num_sel, dNumSel, 1, stream_.get());
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
        if (num_sel <= 0) continue;

        const std::size_t ns = static_cast<std::size_t>(num_sel);
        hI.resize(ns); hJ.resize(ns); hNsnp.resize(ns); hHetHet.resize(ns);
        hIbs0.resize(ns); hPhi.resize(ns);
        d2h_async(hI.data(), dISel, ns, stream_.get());
        d2h_async(hJ.data(), dJSel, ns, stream_.get());
        d2h_async(hNsnp.data(), dNsnpSel, ns, stream_.get());
        d2h_async(hHetHet.data(), dHetHetSel, ns, stream_.get());
        d2h_async(hIbs0.data(), dIbs0Sel, ns, stream_.get());
        d2h_async(hPhi.data(), dPhiSel, ns, stream_.get());
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));

        // Append this block's survivors (cub::DeviceSelect::Flagged is STABLE, and blocks run in
        // ascending rank order -> survivors accumulate in global-rank order, matching the dense
        // path's row order after its identical phi filter).
        out.i.insert(out.i.end(), hI.begin(), hI.end());
        out.j.insert(out.j.end(), hJ.begin(), hJ.end());
        out.nsnp.insert(out.nsnp.end(), hNsnp.begin(), hNsnp.end());
        out.hethet.insert(out.hethet.end(), hHetHet.begin(), hHetHet.end());
        out.ibs0.insert(out.ibs0.end(), hIbs0.begin(), hIbs0.end());
        out.phi.insert(out.phi.end(), hPhi.begin(), hPhi.end());
    }

    out.emitted = out.i.size();
    return out;
}

}  // namespace steppe::device
