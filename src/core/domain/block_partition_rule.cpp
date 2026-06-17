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
    // block_of divides by block_size_morgans, so a non-positive width is a hard
    // contract violation: 0.0 makes the quotient ±Inf (or NaN for 0/0) and a
    // negative width silently INVERTS the bin order (floor of a negative
    // quotient) — both then reach static_cast<int>(...) in block_of, which for
    // ±Inf / NaN is UNDEFINED BEHAVIOR ([conv.fpint]: "if the truncated value
    // cannot be represented in the destination type, the behavior is
    // undefined"), and for a negative width is a silent WRONG partition that
    // still LOOKS dense — defeating the single-source rule that must match AT2.
    // The !(x > 0.0) form rejects 0.0, every negative, AND NaN in one test
    // (NaN > 0.0 is false, so its complement is true), closing the float→int UB
    // and the silent-inversion path here, the only enforceable site today (the
    // ConfigBuilder::build() that would normally validate this does not exist
    // yet). The illegal width yields an empty partition (n_block == 0), the same
    // defined, harmless result as empty input.
    if (!(block_size_morgans > 0.0)) {
        return out;  // 0 / negative / NaN width → empty block_id, n_block == 0.
    }

    // Parallel arrays, one entry per SNP: a length mismatch is a programming
    // error upstream. Be defensive against it (use the shorter extent) rather
    // than reading out of bounds; an honest caller always passes equal lengths.
    const long m = static_cast<long>(
        chrom.size() < genpos_morgans.size() ? chrom.size() : genpos_morgans.size());
    if (m <= 0) {
        return out;  // empty input → empty block_id, n_block == 0.
    }

    out.block_id.resize(static_cast<std::size_t>(m));

    // Single deterministic pass in file order (architecture.md §12: the block id
    // is a pure, launch-order-independent function of (chrom, genpos)).
    int prev_chrom = 0;
    int prev_local_bin = 0;
    long global = -1;  // global block counter; first SNP bumps it to 0.

    for (long s = 0; s < m; ++s) {
        const int c = chrom[static_cast<std::size_t>(s)];
        const int local = block_of(genpos_morgans[static_cast<std::size_t>(s)], block_size_morgans);

        // Open a NEW global block on the first SNP, on a chromosome boundary, or
        // on a local-bin change; otherwise reuse the current global block.
        // Interior empty local bins are absorbed (the dense-id policy); a
        // chromosome boundary always forces a fresh block; a negative-position
        // bin (floor → -1) is just another local bin and never aliases bin 0.
        const bool open_new = (s == 0) || (c != prev_chrom) || (local != prev_local_bin);
        if (open_new) {
            ++global;
        }

        out.block_id[static_cast<std::size_t>(s)] = static_cast<int>(global);
        prev_chrom = c;
        prev_local_bin = local;
    }

    out.n_block = static_cast<int>(global) + 1;  // dense 0..n_block-1.
    return out;
}

}  // namespace steppe::core
