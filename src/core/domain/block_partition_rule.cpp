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

namespace steppe::core {

BlockPartition assign_blocks(std::span<const int> chrom,
                             std::span<const double> genpos_morgans,
                             double block_size_morgans) {
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

    out.block_id.resize(static_cast<std::size_t>(M));

    // Single deterministic pass in file order (architecture.md §12: the block id
    // is a pure, launch-order-independent function of (chrom, genpos)).
    //
    // THE AT2 setblocks() convention (ADMIXTOOLS DReichLab qpsubs.c:1698-1759;
    // admixtools R 2.0.10 get_block_lengths, R/resampling.R:278-300; the bit-tight
    // parity TARGET — docs/research/block-partition-at2.md): a SNP-ANCHORED
    // cumulative walk. A new block opens on a chromosome change OR when the
    // cumulative genetic distance FROM THE BLOCK'S FIRST SNP reaches
    // block_size_morgans; on a cut the anchor RE-SETS to the SNP that opens the
    // block, so the sub-blgsize remainder rolls FORWARD into the next block. This
    // is NOT a fixed-grid floor-bin: blocks are anchored at actual SNP positions,
    // never at grid multiples k*block_size_morgans, and a block spans
    // >= block_size_morgans (can be wider) except the trailing/short-chrom remnant.
    double fpos = -1.0e20;  // genpos of the current block's first SNP; the AT2
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
        const double pos = genpos_morgans[us];

        // CUT on a chromosome change OR when the cumulative distance from the
        // anchor reaches block_size_morgans. The >= is INCLUSIVE — it ports the
        // AT2 C `dis >= blocklen` and R `dis >= blgsize` verbatim (a SNP exactly
        // block_size_morgans from the anchor cuts). On a cut, re-anchor to THIS
        // SNP's position (NOT to fpos + width and NOT to a grid line) so the
        // remainder rolls forward — that re-anchoring is the whole difference from
        // the old floor-bin rule. The -1e20/-1 sentinels make the first SNP cut.
        if (c != prev_chrom || (pos - fpos) >= block_size_morgans) {
            ++global;
            fpos = pos;
            prev_chrom = c;
        }

        out.block_id[us] = static_cast<int>(global);
    }

    out.n_block = static_cast<int>(global) + 1;  // dense 0..n_block-1.
    return out;
}

}  // namespace steppe::core
