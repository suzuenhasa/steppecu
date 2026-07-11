// src/device/cuda/cuda_backend_king_streamed.cu
//
// CudaBackend::king_robust_filtered — the biobank-scale STREAMED/compacted KING-robust all-pairs
// sweep behind `steppe kinship --min-kinship` and `--king-cutoff`. Like the dense path it reuses the
// bitplane pack + the SHARED king_fold_word / king_phi VERBATIM; the design difference is the loop
// nest. The dense path (cuda_backend_king.cu) reduces into PERSISTENT 5*C(N,2) accumulators — O(C(N,2))
// device memory, which forces the ~14k cap. This path keeps only block-local accumulators + the
// survivors resident, so the cap does not bind.
//
// TWO folds share the block/compact/emit machinery:
//   * TILED (default when the full-axis pane fits): build the three bitplanes (HOMREF/HET/HOMALT)
//     ONCE resident over the whole SNP axis (N x ceil(M/32) words), then iterate RECTANGULAR sample-
//     tile BLOCKS (RT x CT tiles). Each block folds via launch_king_block_tiled_accumulate into a
//     dense block-local (RT*32)x(CT*32) accumulator (each sample bitplane word read O(N/TB) times, the
//     dense path's memory pattern — NOT the warp path's O(N) global re-read), then king_phi + emit
//     flag + the SAME six cub::DeviceSelect::Flagged compaction. The survivors are collected across
//     blocks (block-schedule order) and STABLE-sorted by ascending global rank = j*(j-1)/2 + i at the
//     end so the emitted table is byte-identical to the rank-ascending warp/dense order.
//   * WARP (fallback: the pane does not fit, or STEPPE_KING_STREAM_WARP=1 to force it as the gate
//     oracle): pair-BLOCK-OUTER / SNP-tile-INNER, folding each rank-contiguous block warp-per-pair
//     (king_perpair_accumulate). Rank-contiguous blocks + stable cub already emit in rank order.
//
// Because both folds use king_fold_word (counts) + king_phi (phi) + the same emit predicate, the
// survivor SET is bit-identical between them and to the dense path; only the emission order differs
// pre-sort, which the tiled path's final rank sort reconciles.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <span>
#include <vector>

#include <cub/device/device_select.cuh>

#include "device/cuda/cuda_backend.cuh"
#include "device/cuda/check.cuh"
#include "device/cuda/king_allpairs_kernel.cuh"
#include "device/cuda/king_streamed_kernel.cuh"

namespace steppe::device {

namespace {

// ---- WARP-path (fallback) sizing (unchanged from the original streamed driver) -----------------
// Fraction of free VRAM the resident N x tileM code pane may occupy (mirrors the dense path).
constexpr std::size_t kKingCodeBudgetDivisor = 4;
constexpr std::size_t kKingFreeVramFallbackBytes = std::size_t{1} << 30;  // 1 GiB
// Per-accumulate-launch pair-count clamp so grid.x = C stays < kMaxGridX (mirrors the dense path).
constexpr long long kKingPairChunkClamp = 0x40000000LL;  // 2^30
// Bytes per pair the warp block holds: 5 long accumulators (40) + the compaction inputs
// {i,j (int), phi (double), flag (byte)} (17) + the six selected survivor columns (40).
constexpr std::size_t kKingStreamPerPairBytes = 40 + 17 + 40;
// Free-VRAM fraction the pair-block arrays may claim (the remainder covers the code pane + tile).
constexpr std::size_t kKingStreamBlockVramDivisor = 8;

// ---- TILED-path sizing -------------------------------------------------------------------------
constexpr int kKingBitTile = 32;  // TB: samples per tile (== kKingTile in the kernel TU)
// Default SQUARE block dimension in TILES per side (BT x BT tiles == BT*32 samples/block; 32 ->
// 1024 samples/block). It MUST be square: the block schedule iterates upper-triangular block-pairs
// BR<=BC over the SAME tile-block partition on both axes, so a row-block and a col-block must cover
// identical tile ranges (a non-square split would map some i<j pair to BR>BC and skip it). BT=32 is
// big enough that each block launch has ~BT*BT full-occupancy CTAs, small enough that the (BT*32)^2
// block accumulators are ~100 MB. STEPPE_KING_BT overrides for tuning without a rebuild.
constexpr int kKingBlockTilesDefault = 32;
// Bytes the block accumulators + compaction inputs + selected columns hold PER CELL: 5 long
// accumulators (40) + {i,j (int), phi (double), flag (byte)} (17) + six selected columns (40).
constexpr std::size_t kKingTiledPerCellBytes = 40 + 17 + 40;
// VRAM headroom (beyond the pane + the block cells) kept free for the cub temp + safety.
constexpr std::size_t kKingTiledReserveBytes = std::size_t{512} << 20;  // 512 MiB

[[nodiscard]] int env_pos_int(const char* name, int fallback) {
    if (const char* env = std::getenv(name)) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) return static_cast<int>(v);
    }
    return fallback;
}

// ------------------------------------------------------------------------------------------------
// WARP fold (the fallback / gate oracle) — the original pair-block-outer / SNP-tile-inner driver,
// extracted verbatim. Fills out.{i,j,nsnp,hethet,ibs0,phi} in ascending global-rank order.
// ------------------------------------------------------------------------------------------------
void king_stream_warp(cudaStream_t stream, const std::uint8_t* packed_dev,
                      std::size_t bytes_per_record, const std::uint8_t* d_inc, int N, long M,
                      long long npairs, std::size_t free_b, double emit_threshold,
                      bool strict_greater, KingStreamResult& out) {
    // SNP tile: bound the resident three-bitplane pane to ~free/divisor (mirrors the dense path).
    const std::size_t code_budget = free_b / kKingCodeBudgetDivisor;
    const std::size_t bytes_per_sample_word = 3u * sizeof(std::uint32_t);
    std::size_t words_budget = code_budget / (bytes_per_sample_word * static_cast<std::size_t>(N));
    if (words_budget < 1) words_budget = 1;
    long tileM = static_cast<long>(words_budget) * 32;
    if (const char* env = std::getenv("STEPPE_KING_TILE")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) tileM = v;
    }
    if (tileM < 1) tileM = 1;
    if (tileM > M) tileM = M;
    const long Wmax = (tileM + 31) / 32;
    const std::size_t plane_elems = static_cast<std::size_t>(N) * static_cast<std::size_t>(Wmax);
    DeviceBuffer<std::uint32_t> dHomRef(plane_elems == 0 ? 1u : plane_elems);
    DeviceBuffer<std::uint32_t> dHet(plane_elems == 0 ? 1u : plane_elems);
    DeviceBuffer<std::uint32_t> dHomAlt(plane_elems == 0 ? 1u : plane_elems);

    long long B = static_cast<long long>((free_b / kKingStreamBlockVramDivisor) /
                                         kKingStreamPerPairBytes);
    if (const char* env = std::getenv("STEPPE_KING_BLOCK")) {
        const long long v = std::strtoll(env, nullptr, 10);
        if (v > 0) B = v;
    }
    if (B < 1) B = 1;
    if (B > npairs) B = npairs;
    const std::size_t Bz = static_cast<std::size_t>(B);

    DeviceBuffer<long> dNsnp(Bz), dHetHet(Bz), dIbs0(Bz), dHetI(Bz), dHetJ(Bz);
    DeviceBuffer<int> dI(Bz), dJ(Bz);
    DeviceBuffer<double> dPhi(Bz);
    DeviceBuffer<std::uint8_t> dFlag(Bz);
    DeviceBuffer<int> dISel(Bz), dJSel(Bz);
    DeviceBuffer<long> dNsnpSel(Bz), dHetHetSel(Bz), dIbs0Sel(Bz);
    DeviceBuffer<double> dPhiSel(Bz);
    DeviceBuffer<int> dNumSel(1);

    std::size_t sel_bytes = 0;
    STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, sel_bytes, dNsnp.data(), dFlag.data(),
                                                 dNsnpSel.data(), dNumSel.data(),
                                                 static_cast<std::int64_t>(B), stream));
    DeviceBuffer<unsigned char> dCubTemp(sel_bytes == 0 ? 1 : sel_bytes);

    std::vector<int> hI, hJ;
    std::vector<long> hNsnp, hHetHet, hIbs0;
    std::vector<double> hPhi;

    for (long long block_pair0 = 0; block_pair0 < npairs; block_pair0 += B) {
        const long long blockC = std::min<long long>(B, npairs - block_pair0);
        const std::size_t blockCz = static_cast<std::size_t>(blockC);

        STEPPE_CUDA_CHECK(cudaMemsetAsync(dNsnp.data(), 0, blockCz * sizeof(long), stream));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetHet.data(), 0, blockCz * sizeof(long), stream));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dIbs0.data(), 0, blockCz * sizeof(long), stream));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetI.data(), 0, blockCz * sizeof(long), stream));
        STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetJ.data(), 0, blockCz * sizeof(long), stream));

        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            const long W = (tm + 31) / 32;
            launch_king_bitplane_build(packed_dev, bytes_per_record, N, s_lo, tm, W, d_inc,
                                       dHomRef.data(), dHet.data(), dHomAlt.data(), stream);
            for (long long sub = 0; sub < blockC; sub += kKingPairChunkClamp) {
                const long long C = std::min<long long>(kKingPairChunkClamp, blockC - sub);
                launch_king_perpair_accumulate(
                    dHomRef.data(), dHet.data(), dHomAlt.data(), N, W, /*pairs_i=*/nullptr,
                    /*pairs_j=*/nullptr, /*pair0=*/block_pair0 + sub, C, /*out_offset=*/block_pair0,
                    dNsnp.data(), dHetHet.data(), dIbs0.data(), dHetI.data(), dHetJ.data(), stream);
            }
        }

        launch_king_streamed_flag(dNsnp.data(), dHetHet.data(), dIbs0.data(), dHetI.data(),
                                  dHetJ.data(), N, block_pair0, blockC, emit_threshold,
                                  strict_greater, dI.data(), dJ.data(), dPhi.data(), dFlag.data(),
                                  stream);

        const std::int64_t items = static_cast<std::int64_t>(blockC);
        std::size_t sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dI.data(), dFlag.data(),
                                                     dISel.data(), dNumSel.data(), items, stream));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dJ.data(), dFlag.data(),
                                                     dJSel.data(), dNumSel.data(), items, stream));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dNsnp.data(),
                                                     dFlag.data(), dNsnpSel.data(), dNumSel.data(),
                                                     items, stream));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dHetHet.data(),
                                                     dFlag.data(), dHetHetSel.data(),
                                                     dNumSel.data(), items, stream));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dIbs0.data(),
                                                     dFlag.data(), dIbs0Sel.data(), dNumSel.data(),
                                                     items, stream));
        sb = sel_bytes;
        STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dPhi.data(), dFlag.data(),
                                                     dPhiSel.data(), dNumSel.data(), items, stream));

        int num_sel = 0;
        d2h_async(&num_sel, dNumSel, 1, stream);
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (num_sel <= 0) continue;

        const std::size_t ns = static_cast<std::size_t>(num_sel);
        hI.resize(ns); hJ.resize(ns); hNsnp.resize(ns); hHetHet.resize(ns);
        hIbs0.resize(ns); hPhi.resize(ns);
        d2h_async(hI.data(), dISel, ns, stream);
        d2h_async(hJ.data(), dJSel, ns, stream);
        d2h_async(hNsnp.data(), dNsnpSel, ns, stream);
        d2h_async(hHetHet.data(), dHetHetSel, ns, stream);
        d2h_async(hIbs0.data(), dIbs0Sel, ns, stream);
        d2h_async(hPhi.data(), dPhiSel, ns, stream);
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));

        out.i.insert(out.i.end(), hI.begin(), hI.end());
        out.j.insert(out.j.end(), hJ.begin(), hJ.end());
        out.nsnp.insert(out.nsnp.end(), hNsnp.begin(), hNsnp.end());
        out.hethet.insert(out.hethet.end(), hHetHet.begin(), hHetHet.end());
        out.ibs0.insert(out.ibs0.end(), hIbs0.begin(), hIbs0.end());
        out.phi.insert(out.phi.end(), hPhi.begin(), hPhi.end());
    }
}

// ------------------------------------------------------------------------------------------------
// TILED fold (the default) — full-axis resident bitplane pane, rectangular sample-tile blocks folded
// by launch_king_block_tiled_accumulate into block-local accumulators, then king_phi + cub compaction
// (identical to the warp path), then a final rank-ascending sort of the collected survivors so the
// emitted order is byte-identical to the warp/dense rank order.
// ------------------------------------------------------------------------------------------------
void king_stream_tiled(cudaStream_t stream, const std::uint8_t* packed_dev,
                       std::size_t bytes_per_record, const std::uint8_t* d_inc, int N, long M,
                       long W_full, double emit_threshold, bool strict_greater,
                       KingStreamResult& out) {
    // 1) Build the three bitplanes ONCE over the whole SNP axis (resident pane, N x W_full words).
    const std::size_t pane_elems =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(W_full);
    DeviceBuffer<std::uint32_t> dHomRef(pane_elems == 0 ? 1u : pane_elems);
    DeviceBuffer<std::uint32_t> dHet(pane_elems == 0 ? 1u : pane_elems);
    DeviceBuffer<std::uint32_t> dHomAlt(pane_elems == 0 ? 1u : pane_elems);
    launch_king_bitplane_build(packed_dev, bytes_per_record, N, /*s_lo=*/0, /*tm=*/M, W_full, d_inc,
                               dHomRef.data(), dHet.data(), dHomAlt.data(), stream);

    // 2) Block geometry: a SQUARE BT x BT tile block (BT*32 samples per side).
    const int n_tiles = static_cast<int>((static_cast<long>(N) + kKingBitTile - 1) / kKingBitTile);
    int BT = env_pos_int("STEPPE_KING_BT", kKingBlockTilesDefault);
    if (BT > n_tiles) BT = n_tiles;
    if (BT < 1) BT = 1;
    const int n_blocks = (n_tiles + BT - 1) / BT;
    const int colStride = BT * kKingBitTile;                              // block accumulator col stride
    const long nCells = static_cast<long>(BT * kKingBitTile) * colStride;  // (BT*32)^2
    const std::size_t nCellsz = static_cast<std::size_t>(nCells);

    // 3) Block-local accumulators + compaction buffers (sized once at the max cell count).
    DeviceBuffer<long> dNsnp(nCellsz), dHetHet(nCellsz), dIbs0(nCellsz), dHetI(nCellsz),
        dHetJ(nCellsz);
    DeviceBuffer<int> dI(nCellsz), dJ(nCellsz);
    DeviceBuffer<double> dPhi(nCellsz);
    DeviceBuffer<std::uint8_t> dFlag(nCellsz);
    DeviceBuffer<int> dISel(nCellsz), dJSel(nCellsz);
    DeviceBuffer<long> dNsnpSel(nCellsz), dHetHetSel(nCellsz), dIbs0Sel(nCellsz);
    DeviceBuffer<double> dPhiSel(nCellsz);
    DeviceBuffer<int> dNumSel(1);

    std::size_t sel_bytes = 0;
    STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, sel_bytes, dNsnp.data(), dFlag.data(),
                                                 dNsnpSel.data(), dNumSel.data(),
                                                 static_cast<std::int64_t>(nCells), stream));
    DeviceBuffer<unsigned char> dCubTemp(sel_bytes == 0 ? 1 : sel_bytes);

    std::vector<int> hI, hJ;
    std::vector<long> hNsnp, hHetHet, hIbs0;
    std::vector<double> hPhi;

    // 4) Iterate upper-triangular block-pairs (BR <= BC). Each is ONE fold launch over the full W.
    for (int BR = 0; BR < n_blocks; ++BR) {
        const int rowTileLo = BR * BT;
        const int nRowTiles = std::min(BT, n_tiles - rowTileLo);
        const int rowSampleLo = rowTileLo * kKingBitTile;
        for (int BC = BR; BC < n_blocks; ++BC) {
            const int colTileLo = BC * BT;
            const int nColTiles = std::min(BT, n_tiles - colTileLo);
            const int colSampleLo = colTileLo * kKingBitTile;
            const bool diagonal = (BR == BC);

            STEPPE_CUDA_CHECK(cudaMemsetAsync(dNsnp.data(), 0, nCellsz * sizeof(long), stream));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetHet.data(), 0, nCellsz * sizeof(long), stream));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dIbs0.data(), 0, nCellsz * sizeof(long), stream));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetI.data(), 0, nCellsz * sizeof(long), stream));
            STEPPE_CUDA_CHECK(cudaMemsetAsync(dHetJ.data(), 0, nCellsz * sizeof(long), stream));

            launch_king_block_tiled_accumulate(
                dHomRef.data(), dHet.data(), dHomAlt.data(), N, W_full, rowTileLo, colTileLo,
                nRowTiles, nColTiles, diagonal, colStride, dNsnp.data(), dHetHet.data(),
                dIbs0.data(), dHetI.data(), dHetJ.data(), stream);

            launch_king_block_flag(dNsnp.data(), dHetHet.data(), dIbs0.data(), dHetI.data(),
                                   dHetJ.data(), N, rowSampleLo, colSampleLo, nCells, colStride,
                                   emit_threshold, strict_greater, dI.data(), dJ.data(),
                                   dPhi.data(), dFlag.data(), stream);

            const std::int64_t items = static_cast<std::int64_t>(nCells);
            std::size_t sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dI.data(),
                                                         dFlag.data(), dISel.data(),
                                                         dNumSel.data(), items, stream));
            sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dJ.data(),
                                                         dFlag.data(), dJSel.data(),
                                                         dNumSel.data(), items, stream));
            sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dNsnp.data(),
                                                         dFlag.data(), dNsnpSel.data(),
                                                         dNumSel.data(), items, stream));
            sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dHetHet.data(),
                                                         dFlag.data(), dHetHetSel.data(),
                                                         dNumSel.data(), items, stream));
            sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dIbs0.data(),
                                                         dFlag.data(), dIbs0Sel.data(),
                                                         dNumSel.data(), items, stream));
            sb = sel_bytes;
            STEPPE_CUDA_CHECK(cub::DeviceSelect::Flagged(dCubTemp.data(), sb, dPhi.data(),
                                                         dFlag.data(), dPhiSel.data(),
                                                         dNumSel.data(), items, stream));

            int num_sel = 0;
            d2h_async(&num_sel, dNumSel, 1, stream);
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
            if (num_sel <= 0) continue;

            const std::size_t ns = static_cast<std::size_t>(num_sel);
            hI.resize(ns); hJ.resize(ns); hNsnp.resize(ns); hHetHet.resize(ns);
            hIbs0.resize(ns); hPhi.resize(ns);
            d2h_async(hI.data(), dISel, ns, stream);
            d2h_async(hJ.data(), dJSel, ns, stream);
            d2h_async(hNsnp.data(), dNsnpSel, ns, stream);
            d2h_async(hHetHet.data(), dHetHetSel, ns, stream);
            d2h_async(hIbs0.data(), dIbs0Sel, ns, stream);
            d2h_async(hPhi.data(), dPhiSel, ns, stream);
            STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));

            out.i.insert(out.i.end(), hI.begin(), hI.end());
            out.j.insert(out.j.end(), hJ.begin(), hJ.end());
            out.nsnp.insert(out.nsnp.end(), hNsnp.begin(), hNsnp.end());
            out.hethet.insert(out.hethet.end(), hHetHet.begin(), hHetHet.end());
            out.ibs0.insert(out.ibs0.end(), hIbs0.begin(), hIbs0.end());
            out.phi.insert(out.phi.end(), hPhi.begin(), hPhi.end());
        }
    }

    // 5) Reorder survivors into ascending global rank = j*(j-1)/2 + i (a UNIQUE total key over i<j),
    // so the tiled path's block-schedule emission reproduces the warp/dense rank-ascending bytes.
    const std::size_t ne = out.i.size();
    if (ne > 1) {
        std::vector<std::size_t> order(ne);
        std::iota(order.begin(), order.end(), std::size_t{0});
        const auto rank_of = [&out](std::size_t k) {
            const long long j = out.j[k];
            const long long i = out.i[k];
            return j * (j - 1) / 2 + i;
        };
        std::sort(order.begin(), order.end(),
                  [&](std::size_t x, std::size_t y) { return rank_of(x) < rank_of(y); });
        std::vector<int> ni(ne), nj(ne);
        std::vector<long> nn(ne), nh(ne), nb(ne);
        std::vector<double> np(ne);
        for (std::size_t k = 0; k < ne; ++k) {
            const std::size_t s = order[k];
            ni[k] = out.i[s]; nj[k] = out.j[s];
            nn[k] = out.nsnp[s]; nh[k] = out.hethet[s]; nb[k] = out.ibs0[s];
            np[k] = out.phi[s];
        }
        out.i = std::move(ni); out.j = std::move(nj);
        out.nsnp = std::move(nn); out.hethet = std::move(nh); out.ibs0 = std::move(nb);
        out.phi = std::move(np);
    }
}

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

    // Optional autosome mask, indexed by the GLOBAL SNP position in the fold/build kernels.
    const std::size_t Mz = static_cast<std::size_t>(M);
    const bool have_inc = summary_include.size() == Mz;
    DeviceBuffer<std::uint8_t> dInclude(have_inc ? Mz : 1u);
    if (have_inc) h2d_async(dInclude, summary_include.data(), Mz, stream_.get());
    const std::uint8_t* d_inc = have_inc ? dInclude.data() : nullptr;

    // Pick the fold. The TILED fold needs the full-axis three-bitplane pane resident
    // (12*N*ceil(M/32) bytes) plus the block accumulators; if that does not fit the free VRAM (with
    // headroom), or STEPPE_KING_STREAM_WARP forces it, fall back to the SNP-tiled warp fold.
    std::size_t free_b = capabilities().free_vram_bytes;  // reflects packed already resident/staged
    if (free_b == 0) free_b = kKingFreeVramFallbackBytes;

    const long W_full = (M + 31) / 32;
    const std::size_t pane_bytes = std::size_t{3} * sizeof(std::uint32_t) *
                                   static_cast<std::size_t>(N) * static_cast<std::size_t>(W_full);
    const int n_tiles = static_cast<int>((static_cast<long>(N) + kKingBitTile - 1) / kKingBitTile);
    int BT = env_pos_int("STEPPE_KING_BT", kKingBlockTilesDefault);
    if (BT > n_tiles) BT = n_tiles;
    if (BT < 1) BT = 1;
    const std::size_t block_side = static_cast<std::size_t>(BT) * kKingBitTile;
    const std::size_t block_cells = block_side * block_side;
    const std::size_t tiled_bytes = pane_bytes + block_cells * kKingTiledPerCellBytes +
                                    kKingTiledReserveBytes;

    const bool force_warp = std::getenv("STEPPE_KING_STREAM_WARP") != nullptr;
    const bool pane_fits = tiled_bytes <= free_b;

    if (force_warp || !pane_fits) {
        king_stream_warp(stream_.get(), packed_dev, tile.bytes_per_record, d_inc, N, M, npairs,
                         free_b, emit_threshold, strict_greater, out);
    } else {
        king_stream_tiled(stream_.get(), packed_dev, tile.bytes_per_record, d_inc, N, M, W_full,
                          emit_threshold, strict_greater, out);
    }

    out.emitted = out.i.size();
    return out;
}

}  // namespace steppe::device
