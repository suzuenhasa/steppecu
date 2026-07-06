// src/core/domain/block_partition_rule.cpp
//
// Host-pure SNP→jackknife-block assignment: the single source of the block id
// that both io and the device kernels consume, split out from the header only
// because the walk carries state across a loop.
//
// Reference: docs/reference/src_core_domain_block_partition_rule.cpp.md
#include "core/domain/block_partition_rule.hpp"

#include <cstddef>
#include <cstdio>

namespace steppe::core {

namespace {

// The block walk — reference §2
[[nodiscard]] BlockPartition block_walk(std::span<const int> chrom,
                                        std::span<const double> pos, long M,
                                        double window) {
    BlockPartition out;
    out.block_id.resize(idx(M));

    double fpos = -1.0e20;
    int prev_chrom = -1;
    long global = -1;

    for (long s = 0; s < M; ++s) {
        const std::size_t us = idx(s);
        const int c = chrom[us];
        const double p = pos[us];

        if (c != prev_chrom || (p - fpos) >= window) {
            ++global;
            fpos = p;
            prev_chrom = c;
        }

        out.block_id[us] = static_cast<int>(global);
    }

    out.n_block = static_cast<int>(global) + 1;
    return out;
}

// Detecting a missing genetic map — reference §3
[[nodiscard]] bool all_zero(std::span<const double> pos, long M) {
    for (long s = 0; s < M; ++s) {
        if (pos[idx(s)] != 0.0) return false;
    }
    return true;
}

}  // namespace

// assign_blocks — guards and dispatch — reference §4
BlockPartition assign_blocks(std::span<const int> chrom,
                             std::span<const double> genpos_morgans,
                             double block_size_morgans,
                             std::span<const double> physpos,
                             double bp_window) {
    BlockPartition out;

    if (!(block_size_morgans > 0.0)) {
        return out;
    }

    const long M = static_cast<long>(
        chrom.size() < genpos_morgans.size() ? chrom.size() : genpos_morgans.size());
    if (M <= 0) {
        return out;
    }

    const bool have_physaxis = physpos.size() >= idx(M) &&
                               bp_window > 0.0;
    if (have_physaxis && all_zero(genpos_morgans, M) && !all_zero(physpos, M)) {
        std::fprintf(stderr,
                     "steppe: No genetic linkage map found! Defining blocks by "
                     "base pair distance of %g\n",
                     bp_window);
        return block_walk(chrom, physpos, M, bp_window);
    }

    return block_walk(chrom, genpos_morgans, M, block_size_morgans);
}

}  // namespace steppe::core
