// src/device/shard_plan.cpp
//
// plan_block_shards — the BLOCK-ALIGNED, SNP-count-balanced shard planner
// (architecture.md §11.4 SPMG tile sharding; design §2). Host-pure, CUDA-FREE: it
// reaches no GPU and includes no CUDA header, so it compiles as a plain .cpp into
// steppe_device (alongside resources.cpp) and is equally exercisable host-only by
// the parity test. The block-aligned greedy assignment is the single home of the
// block→device mapping (DRY, §8); the orchestrator and the parity test both call
// it so they shard identically.
#include "device/shard_plan.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // steppe::core::BlockRange
#include "core/internal/launch_config.hpp"        // steppe::core::cdiv (the single home of ceiling division)

namespace steppe::device {

std::vector<DeviceShard> plan_block_shards(
    std::span<const steppe::core::BlockRange> ranges,
    std::size_t G) {
    // ---- Fail-fast preconditions (architecture.md §2) -----------------------
    // A shard plan needs at least one device; build_resources guarantees G >= 1,
    // but the planner is the single home of the assignment so it owns the guard.
    if (G == 0) {
        throw std::runtime_error(
            "steppe::device::plan_block_shards: G == 0 — the SPMG shard plan "
            "requires at least one device (architecture.md §9)");
    }
    // `ranges` is the SOLE description of the partition (no parallel block_sizes
    // array to drift, cleanup B6 / X1): each block's SNP count is `ranges[b].size()`
    // (`end - begin`, a non-negative `long` for any validated block_ranges output).

    const std::size_t n_block = ranges.size();
    std::vector<DeviceShard> plan(G);  // value-initialized: every entry {0,0,0,0} (empty)

    // ---- Degenerate: nothing to shard ---------------------------------------
    // No blocks ⇒ every device gets the empty shard {0,0,0,0}; the orchestrator
    // hands each device n_block_local == 0 and the backend early-returns empty.
    if (n_block == 0) {
        return plan;
    }

    // ---- Total SNP count and the DERIVED per-device balance target ----------
    // Balance by SNP count (blocks vary in size; design §2). Each block's count is
    // `ranges[b].size()` (already `long`, ≥ 0). The target is the even split
    // total_snps / G, derived from the inputs — NOT a magic number (§8). long
    // accumulators: total SNPs is the whole-genome M (~6e5), well within long; the
    // sum cannot overflow for any realistic partition.
    long total_snps = 0;
    for (std::size_t b = 0; b < n_block; ++b) {
        total_snps += ranges[b].size();
    }
    // Ceiling division so G devices cover total_snps with the LAST device taking
    // the remainder rather than overflowing into a (G+1)-th: target * G >= total.
    // With target == ceil(total/G) the greedy close-on-cross below never needs
    // more than G ranges (proof: each closed range holds >= ~target SNPs minus one
    // block's worth, so the cumulative reaches total within G closes). core::cdiv
    // is the single home of ceiling division (§8; launch_config.hpp); the long
    // overload computes the identical (total + G - 1) / G with no overflow here.
    const long G_signed = static_cast<long>(G);
    const long target_per_device =
        core::cdiv(total_snps, G_signed);  // >= 1 when total_snps >= 1

    // ---- Greedy block-aligned assignment ------------------------------------
    // Walk blocks in order, accumulating SNP count into the current device's range.
    // CLOSE the current device (advance to the next) once its cumulative SNP count
    // crosses target_per_device AND there are still devices left to fill — never
    // splitting a block (the block-aligned invariant). The LAST device absorbs all
    // remaining blocks (we stop closing once g == G-1), so no block is ever
    // orphaned even if rounding would otherwise leave a tail.
    std::size_t g = 0;          // current device index
    std::size_t b0 = 0;         // first block of the current device's range
    long device_snps = 0;       // SNPs accumulated into the current device

    // One home for the {b0,b1,s0,s1} shard shape (the int-narrowing of the block ids
    // + the begin/end column lookup off `ranges`): the mid-loop close and the final
    // close differ only by the upper block bound, so they build the shard identically
    // here (cleanup group-7 7.1 — no two copies to drift, e.g. on the `hi-1` end index).
    const auto make_shard = [&](std::size_t lo, std::size_t hi) {
        return DeviceShard{
            static_cast<int>(lo),
            static_cast<int>(hi),
            ranges[lo].begin,
            ranges[hi - 1].end};
    };

    for (std::size_t b = 0; b < n_block; ++b) {
        device_snps += ranges[b].size();

        // Close the current device's range when it has reached its SNP target and
        // we are NOT yet on the last device (the last device takes the rest). The
        // range closes AFTER block b (inclusive), so the next device starts at b+1.
        const bool more_devices_left = (g + 1 < G);
        const bool reached_target = (device_snps >= target_per_device);
        if (more_devices_left && reached_target && (b + 1 < n_block)) {
            const std::size_t b1 = b + 1;
            plan[g] = make_shard(b0, b1);
            ++g;
            b0 = b1;
            device_snps = 0;
        }
    }

    // Close the FINAL non-empty range (device g) over the remaining blocks
    // [b0, n_block). This always runs because the loop never closes the last
    // device (the b+1 < n_block guard leaves the trailing run for here).
    plan[g] = make_shard(b0, n_block);

    // Devices g+1 .. G-1 (if any) keep their value-initialized EMPTY shard
    // {0,0,0,0} — the n_block < G case where trailing devices own nothing. An
    // empty shard's empty() == true; the orchestrator hands those devices
    // n_block_local == 0 and the backend early-returns an empty partial (design §2).
    return plan;
}

}  // namespace steppe::device
