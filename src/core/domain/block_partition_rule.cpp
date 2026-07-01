// src/core/domain/block_partition_rule.cpp
//
// The whole-ordering SNP→block assignment (architecture.md §5, §8, §12; ROADMAP
// §3 M3). Host-pure, CUDA-free; the single source of the jackknife block id that
// `io` and the device kernels both consume (the §8 DRY invariant). The per-SNP
// floor primitive (`block_of`) and the cM→Morgan conversion site
// (`block_size_cm_to_morgans`) are header-inline in block_partition_rule.hpp; the
// pass below is split out here because it carries state and a loop.
#include "core/domain/block_partition_rule.hpp"

#include <cstddef>
#include <cstdio>  // std::fprintf — the AT2 "no genetic map" stderr warning

namespace steppe::core {

namespace {

// The AT2 SNP-anchored cumulative walk (ADMIXTOOLS DReichLab qpsubs.c:1698-1759;
// admixtools R 2.0.10 get_block_lengths, R/resampling.R:278-300; the bit-tight
// parity TARGET — docs/research/block-partition-at2.md): a single deterministic
// pass in file order (architecture.md §12: the block id is a pure,
// launch-order-independent function of (chrom, pos)). Carry the position of the
// current block's FIRST SNP (the anchor); a new block opens on a chromosome
// change OR when the cumulative distance FROM THE ANCHOR reaches `window`; on a
// cut the anchor RE-SETS to the SNP that opens the block, so the sub-window
// remainder rolls FORWARD. This is NOT a fixed-grid floor-bin: blocks are
// anchored at actual SNP positions, never at grid multiples k*window, and a block
// spans >= window (can be wider) except the trailing/short-chrom remnant.
//
// GENERIC over the position axis + window so the ONE walk serves BOTH regimes
// with byte-identical structure: (genpos_morgans, block_size_morgans) on a real
// genetic map, and (physpos, bp_window) for the AT2 bp fallback. Caller
// guarantees `pos.size() >= M`, M > 0, and `window > 0` (validated in
// assign_blocks); this is the loop only.
[[nodiscard]] BlockPartition block_walk(std::span<const int> chrom,
                                        std::span<const double> pos, long M,
                                        double window) {
    BlockPartition out;
    out.block_id.resize(static_cast<std::size_t>(M));

    double fpos = -1.0e20;  // position of the current block's first SNP; the AT2
                            // sentinel forces the first real SNP to open block 0.
    int prev_chrom = -1;    // sentinel: any real chrom (steppe codes are 1..24)
                            // differs from -1, so the first SNP opens block 0.
    long global = -1;       // dense 0-based block counter; first SNP bumps it to 0.

    for (long s = 0; s < M; ++s) {
        // The SNP column index, widened ONCE per iteration ([7.3] dedup): the two
        // parallel-array subscripts below both index column s. idx() is the shared
        // long→size_t cast helper (block_partition_rule.hpp), used here and in
        // block_ranges so the boilerplate lives in one place.
        const std::size_t us = idx(s);
        const int c = chrom[us];
        const double p = pos[us];

        // CUT on a chromosome change OR when the cumulative distance from the
        // anchor reaches `window`. The >= is INCLUSIVE — it ports the AT2 C
        // `dis >= blocklen` and R `dis >= blgsize` verbatim (a SNP exactly
        // `window` from the anchor cuts). On a cut, re-anchor to THIS SNP's
        // position (NOT to fpos + window and NOT to a grid line) so the remainder
        // rolls forward — that re-anchoring is the whole difference from the old
        // floor-bin rule. The -1e20/-1 sentinels make the first SNP cut.
        if (c != prev_chrom || (p - fpos) >= window) {
            ++global;
            fpos = p;
            prev_chrom = c;
        }

        out.block_id[us] = static_cast<int>(global);
    }

    out.n_block = static_cast<int>(global) + 1;  // dense 0..n_block-1.
    return out;
}

// True iff EVERY entry of `pos[0, M)` is exactly 0.0 — the "no map" test. AT2
// switches to the bp fallback when the whole genetic-position column is zero
// (get_block_lengths); the .snp/.bim readers write exact 0.0 for a "0.000000"
// column, so an exact == 0.0 compare is correct (no epsilon). Short-circuits on
// the first non-zero, so a real map (first SNP non-zero) costs O(1).
[[nodiscard]] bool all_zero(std::span<const double> pos, long M) {
    for (long s = 0; s < M; ++s) {
        if (pos[idx(s)] != 0.0) return false;
    }
    return true;
}

}  // namespace

BlockPartition assign_blocks(std::span<const int> chrom,
                             std::span<const double> genpos_morgans,
                             double block_size_morgans,
                             std::span<const double> physpos,
                             double bp_window) {
    BlockPartition out;

    // FAIL-FAST guard on the block width (architecture.md §2; cleanup X-3/B13).
    // The AT2 walk compares `(pos - fpos) >= block_size_morgans`, so a
    // non-positive or NaN width is a hard contract violation that produces a
    // SILENTLY WRONG partition rather than the AT2-matching one:
    //   * 0.0 / negative → the threshold trips on EVERY SNP within a chromosome
    //     (a 0 or negative gap always satisfies `>=`), so every SNP becomes its
    //     own block — a silent over-partition that still LOOKS dense;
    //   * NaN → `(pos - fpos) >= NaN` is ALWAYS false, so NO interior cut ever
    //     fires and a whole chromosome collapses to one block — silently merged.
    // Either defeats the single-source rule that must match AT2. The !(x > 0.0)
    // form rejects 0.0, every negative, AND NaN in one test (NaN > 0.0 is false,
    // so its complement is true), closing all three silent-wrong paths here — the
    // only enforceable site today (the ConfigBuilder::build() that would normally
    // validate this does not exist yet). The illegal width yields an empty
    // partition (n_block == 0), the same defined, harmless result as empty input.
    if (!(block_size_morgans > 0.0)) {
        return out;  // 0 / negative / NaN width → empty block_id, n_block == 0.
    }

    // Parallel arrays, one entry per SNP: a length mismatch is a programming
    // error upstream. Be defensive against it (use the shorter extent) rather
    // than reading out of bounds; an honest caller always passes equal lengths.
    const long M = static_cast<long>(
        chrom.size() < genpos_morgans.size() ? chrom.size() : genpos_morgans.size());
    if (M <= 0) {
        return out;  // empty input → empty block_id, n_block == 0.
    }

    // THE AT2 bp BLOCK-FALLBACK (block_partition_rule.hpp; the all-zero-map case).
    // On a dataset with NO genetic linkage map the genetic-position column is all
    // zero, so `(genpos - anchor)` is always 0, the only cuts are chromosome
    // changes, and the walk collapses to 1 block/chrom — breaking the jackknife SE
    // (a single-chrom subset → 1 block → NA SE / non-SPD covariance). AT2 detects
    // this and partitions by a 2 Mb PHYSICAL-position window instead. Reproduce it
    // EXACTLY: the fallback fires ONLY when the genetic map is all zero AND a
    // NON-DEGENERATE physical axis is supplied (physpos length >= M, `bp_window`
    // positive, and physpos not itself all zero — an all-zero bp axis cannot
    // anchor blocks either, so there is nothing to gain and we keep the genetic
    // walk). A dataset WITH a real map (first genpos non-zero) short-circuits
    // all_zero(genpos) at s == 0 and takes the genetic walk UNCHANGED — the fix is
    // a strict no-op there (the AADR golden-parity pass gate).
    const bool have_physaxis = physpos.size() >= static_cast<std::size_t>(M) &&
                               bp_window > 0.0;
    if (have_physaxis && all_zero(genpos_morgans, M) && !all_zero(physpos, M)) {
        // Warn like AT2 (admixtools 2.0.10 get_block_lengths): the verbatim
        // message, once per assign_blocks call. %g of 2e6 prints "2e+06".
        std::fprintf(stderr,
                     "steppe: No genetic linkage map found! Defining blocks by "
                     "base pair distance of %g\n",
                     bp_window);
        return block_walk(chrom, physpos, M, bp_window);
    }

    // The normal genetic-map walk (unchanged; genpos_morgans + block_size_morgans).
    return block_walk(chrom, genpos_morgans, M, block_size_morgans);
}

}  // namespace steppe::core
