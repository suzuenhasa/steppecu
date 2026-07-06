// src/core/domain/block_partition_rule.hpp
//
// The single source of truth for which jackknife block a SNP belongs to. Host-pure
// and CUDA-free, so both the io front-end and the device kernels include it and get
// bit-identical block ids; the SNP→block rule lives here and is never re-derived.
//
// Reference: docs/reference/src_core_domain_block_partition_rule.hpp.md
#ifndef STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
#define STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "steppe/config.hpp"

namespace steppe::core {

// block_of — the per-SNP block primitive — reference §3
[[nodiscard]] inline int block_of(double genpos_morgans, double block_size_morgans) noexcept {
    return static_cast<int>(std::floor(genpos_morgans / block_size_morgans));
}

// block_size_cm_to_morgans — the single cM↔Morgan conversion site — reference §2
[[nodiscard]] constexpr double block_size_cm_to_morgans(double cm) noexcept {
    return cm / kCentimorgansPerMorgan;
}

// BlockPartition — the result of assign_blocks — reference §6
struct BlockPartition {
    std::vector<int> block_id;

    int n_block = 0;
};

// assign_blocks — the whole-genome block assignment (and bp fallback) — reference §4, §5
[[nodiscard]] BlockPartition assign_blocks(std::span<const int> chrom,
                                           std::span<const double> genpos_morgans,
                                           double block_size_morgans,
                                           std::span<const double> physpos = {},
                                           double bp_window = kBpFallbackWindow);

// BlockRange — one block's half-open SNP column range — reference §7
struct BlockRange {
    long begin = 0;
    long end = 0;

    [[nodiscard]] long size() const noexcept { return end - begin; }
};

// block_ranges — the inverse of assign_blocks — reference §7
[[nodiscard]] inline std::vector<BlockRange> block_ranges(std::span<const int> block_id,
                                                          long M, int n_block) {
    if (M <= 0 || n_block <= 0) {
        return {};
    }

    const auto fail = [](const std::string& msg) {
        throw std::runtime_error("core::block_ranges: " + msg);
    };

    if (block_id.size() < idx(M)) {
        fail("block_id has " + std::to_string(block_id.size()) +
             " entries but M = " + std::to_string(M) +
             " columns are required (partition shorter than the SNP count)");
    }

    std::vector<BlockRange> ranges(idx(n_block));

    long s = 0;
    int prev_b = -1;
    while (s < M) {
        const int b = block_id[idx(s)];
        if (b < 0 || b >= n_block) {
            fail("block_id[" + std::to_string(s) + "] = " +
                 std::to_string(b) + " is out of range [0, " + std::to_string(n_block) + ")");
        }
        if (b < prev_b) {
            fail("block_id is not non-decreasing at column " +
                 std::to_string(s) + " (" + std::to_string(b) + " < " +
                 std::to_string(prev_b) + "); the partition is not contiguous");
        }

        long e = s;
        while (e < M && block_id[idx(e)] == b) ++e;
        ranges[idx(b)] = BlockRange{s, e};
        prev_b = b;
        s = e;
    }

    return ranges;
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_DOMAIN_BLOCK_PARTITION_RULE_HPP
