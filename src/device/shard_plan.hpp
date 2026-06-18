// src/device/shard_plan.hpp
//
// The BLOCK-ALIGNED SNP shard plan — the host-pure, CUDA-FREE planning step the
// single-node multi-GPU (SPMG) precompute drives against (architecture.md §11.4
// "Precompute: tile sharding + a host-side fixed-order combine", §12 parity;
// design §2). It assigns CONTIGUOUS whole-block ranges to the G devices, balanced
// by SNP count, so EACH BLOCK IS COMPUTED ENTIRELY ON ONE DEVICE — the single
// property that makes a device's per-block partial BIT-IDENTICAL to the single-GPU
// run for that block (same SNP columns ⇒ same feeder bits ⇒ same s_pad bucket ⇒
// same independent strided-batched slab; design §0). This is the floor the §12
// PARITY LAW (bit-identical across G and to single-GPU) rests on.
//
// CUDA-FREE BY CONTRACT, like device/backend.hpp / device/resources.hpp: it names
// only the CUDA-free `core::BlockRange` + std types, so it compiles into core/the
// tests AND the device layer without the CUDA toolkit. It is "device-layer" only
// by placement (it is internal SPMG orchestration plumbing, not public API, and is
// SHARED by the device-layer multi-GPU orchestrator and the parity test) — it
// reaches no GPU and includes no <cuda_runtime.h> (design §1, §7, §8).
#ifndef STEPPE_DEVICE_SHARD_PLAN_HPP
#define STEPPE_DEVICE_SHARD_PLAN_HPP

#include <cstddef>
#include <span>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // steppe::core::BlockRange (the single-source block column ranges)

namespace steppe::device {

/// The half-open BLOCK range `[b0, b1)` device g owns, and the corresponding
/// half-open SNP COLUMN range `[s0, s1)`. The column range is CONTIGUOUS because
/// `block_id` is non-decreasing (the assign_blocks postcondition / block_ranges
/// contract), so a contiguous block range maps to a contiguous column span — the
/// zero-copy sub-view property the multi-GPU orchestrator relies on (design §2:
/// `MatView{Q.data + P*s0, P, s1 - s0}`). An EMPTY shard has `b0 == b1` (and
/// `s0 == s1`): a clean no-op a device early-returns on (cuda_backend.cu's
/// degenerate guard yields an empty F2BlockTensor for n_block_local == 0).
struct DeviceShard {
    int  b0 = 0;   ///< first owned block id (inclusive).
    int  b1 = 0;   ///< one past the last owned block id (exclusive).
    long s0 = 0;   ///< first owned SNP column (== ranges[b0].begin), inclusive.
    long s1 = 0;   ///< one past the last owned SNP column (== ranges[b1-1].end), exclusive.

    /// True when this device owns no blocks (b0 == b1 ⇒ s0 == s1): the device
    /// early-returns an empty partial and the combine places nothing for it.
    [[nodiscard]] bool empty() const noexcept { return b0 >= b1; }
};

/// Partition the `n_block` jackknife blocks into G CONTIGUOUS block ranges,
/// balanced by SNP count (architecture.md §11.4 SPMG tile sharding; design §2).
///
/// DETERMINISTIC PURE FUNCTION of `(ranges, G)`: the SAME plan for a given
/// (partition, G) regardless of device speed / residency, so the shard assignment
/// — and therefore the combine layout — is reproducible. This is what lets the
/// §12 parity hold: a block lands on the same device every run, so its bits are
/// stable, and the combine sums the per-device partials in the fixed g=0..G-1
/// order (combine_f2_partials_host / _p2p) onto a zero-initialized full tensor.
///
/// BALANCE BY SNP COUNT (not block count): blocks vary in size, so balancing by
/// block count would imbalance the GEMM work. Greedy single pass: target
/// ~`total_snps / G` SNPs per device; CLOSE a device's range once its cumulative
/// SNP count crosses the target (NEVER splitting a block — the block-aligned
/// invariant that makes a block computed entirely on one device). The per-block
/// SNP count is `ranges[b].size()` (`end - begin`, a `long` ≥ 0 for any validated
/// `block_ranges` output) — it is the SINGLE source of both the balance math AND
/// each shard's `[s0, s1)`, so there is no parallel array to drift or to narrow
/// (cleanup B6 / X1 / shard_plan D-1). The target is DERIVED from the inputs (no
/// magic number; design §8). Every block lands in exactly one device's range; the
/// ranges are contiguous and tile `[0, n_block)`.
///
/// EDGE CASES (fail-fast / clean degenerate, architecture.md §2):
///   * G == 0 ⇒ throws (a shard plan needs at least one device — the caller
///     resolves G from Resources::device_count(), which build_resources guarantees
///     >= 1; design §1).
///   * G == 1 ⇒ the single range `[0, n_block)` covering all columns — the EXACT
///     single-GPU shard, so the multi-GPU path with one device is a structural
///     no-op (design §5; the orchestrator special-cases G==1 even before planning,
///     but the plan is correct either way).
///   * n_block == 0 ⇒ G empty shards (`{0,0,0,0}`); nothing to compute or combine.
///   * n_block < G ⇒ the first `n_block` devices each get exactly one block (or a
///     contiguous few), and TRAILING devices get EMPTY shards (`b0 == b1`) — a
///     device handed no blocks early-returns an empty partial (design §2/§5).
///
/// @param ranges       the per-block half-open column ranges from
///                     `core::block_ranges` (length `n_block == ranges.size()`).
///                     The SOLE input describing the partition: each shard's
///                     `[s0, s1)` is `[ranges[b0].begin, ranges[b1-1].end)`, and
///                     the per-block SNP count for the balance is `ranges[b].size()`
///                     (`end - begin`). There is no separate `block_sizes` array —
///                     it was redundant with `ranges[b].size()` and is derived here
///                     (cleanup B6 / X1).
/// @param G            number of devices (`Resources::device_count()`); must be
///                     >= 1.
/// @return  exactly G `DeviceShard` entries; `plan[g]` is the range device g owns,
///          in g=0..G-1 order. Contiguous and covering `[0, n_block)` (the union
///          of the non-empty ranges).
/// @throws std::runtime_error if G == 0.
[[nodiscard]] std::vector<DeviceShard> plan_block_shards(
    std::span<const steppe::core::BlockRange> ranges,
    std::size_t G);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_SHARD_PLAN_HPP
