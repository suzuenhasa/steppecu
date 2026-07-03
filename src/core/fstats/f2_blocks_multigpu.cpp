// src/core/fstats/f2_blocks_multigpu.cpp
//
// Host-pure, CUDA-free entry points that compute the per-block f2 tensor across
// every GPU in one box and stitch the shards back together. Whole blocks are
// assigned to one device each and merged in fixed device order, so every path is
// bit-identical to a single-GPU run.
//
// Reference: docs/reference/src_core_fstats_f2_blocks_multigpu.cpp.md
#include "core/fstats/f2_blocks_multigpu.hpp"

#include <cstddef>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/fstats/f2_blocks_multigpu_core.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "core/internal/views.hpp"
#include "core/internal/host_device.hpp"
#include "core/internal/log.hpp"
#include "device/p2p_combine.hpp"
#include "device/device_partial.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "device/shard_plan.hpp"
#include "device/tier_select.hpp"
#include "device/f2_blocks_out.hpp"
#include "device/stream_f2_blocks.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"

namespace steppe::core {

namespace {

// Operator override env-var keys — reference §8
inline constexpr char kEnvForceTier[] = "STEPPE_FORCE_TIER";
inline constexpr char kEnvF2CachePath[] = "STEPPE_F2_CACHE_PATH";

// Defensive non-negative clamp — reference §9
[[nodiscard]] constexpr int clamp_nonneg(int n) noexcept { return n < 0 ? 0 : n; }

[[nodiscard]] constexpr std::size_t nonneg_count(int n) noexcept {
    return static_cast<std::size_t>(n < 0 ? 0 : n);
}
[[nodiscard]] constexpr std::size_t nonneg_count(long n) noexcept {
    return static_cast<std::size_t>(n < 0 ? 0 : n);
}

// The combine gate — reference §4
[[nodiscard]] bool requested_p2p_combine(const steppe::device::Resources& resources,
                                         std::size_t G) noexcept {
    return resources.config.prefer_p2p_combine && resources.config.enable_peer_access &&
           G >= 2;
}

[[nodiscard]] bool select_p2p_combine(const steppe::device::Resources& resources,
                                      std::size_t G) noexcept {
    return requested_p2p_combine(resources, G) && resources.gpus[0].caps.can_access_peer;
}

// Shared entry-contract guards — reference §9
void validate_multigpu_inputs([[maybe_unused]] const MatView& Q,
                              [[maybe_unused]] const MatView& V,
                              [[maybe_unused]] const MatView& N,
                              [[maybe_unused]] const BlockPartition& partition,
                              [[maybe_unused]] long M,
                              [[maybe_unused]] const char* fn) {
    STEPPE_ASSERT(Q.P == V.P && V.P == N.P,
                  "validate_multigpu_inputs: Q/V/N disagree on P");
    STEPPE_ASSERT(Q.M == V.M && V.M == N.M,
                  "validate_multigpu_inputs: Q/V/N disagree on M");
    STEPPE_ASSERT(Q.P >= 0 && Q.M >= 0,
                  "validate_multigpu_inputs: negative P or M (uninitialized MatView)");
    STEPPE_ASSERT(partition.block_id.size() == nonneg_count(M),
                  "validate_multigpu_inputs: block_id length != M");
}

[[nodiscard]] std::size_t require_at_least_one_device(
    const steppe::device::Resources& resources, const char* fn) {
    const std::size_t G = resources.device_count();
    if (G < 1) {
        throw std::runtime_error(
            std::string("steppe::core::") + fn + ": Resources has 0 devices — "
            "the SPMG precompute requires at least one (architecture.md §9)");
    }
    return G;
}

// Shared streamed-tier finisher — reference §7
template <typename TierHandle>
void finish_streamed_tier(steppe::device::Resources& resources,
                          const MatView& Q, const MatView& V, const MatView& N,
                          const BlockPartition& partition, int n_block,
                          const Precision& precision,
                          steppe::device::StreamTarget& target,
                          steppe::device::F2BlocksOut& out,
                          const TierHandle& tier_handle,
                          const steppe::device::RedecodeSource* redecode) {
    resources.gpus[0].backend->compute_f2_blocks_streamed(
        Q, V, N, partition.block_id.data(), n_block, precision, target, redecode);
    out.P = tier_handle.P;
    if (!tier_handle.block_sizes.empty()) out.block_sizes = tier_handle.block_sizes;
}

}  // namespace

// Host-returning entry point — reference §3
F2BlockTensor compute_f2_blocks_multigpu(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision) {
    const int  P = Q.P;
    const long M = Q.M;
    const int  n_block = partition.n_block;

    validate_multigpu_inputs(Q, V, N, partition, M, "compute_f2_blocks_multigpu");
    const std::size_t G = require_at_least_one_device(resources, "compute_f2_blocks_multigpu");

    if (G == 1) {
        return resources.gpus[0].backend->compute_f2_blocks(
            Q, V, N, partition.block_id.data(), n_block, precision);
    }

    const bool use_p2p = select_p2p_combine(resources, G);

    const std::vector<steppe::device::DeviceShard> shards =
        core::plan_multigpu_shards(partition, M, n_block, G);
    const std::span<const steppe::device::DeviceShard> shards_span(shards.data(), shards.size());

    if (use_p2p) {
        return compute_f2_blocks_multigpu_device(resources, Q, V, N, partition, precision)
            .to_host();
    }

    F2BlockTensor out;
    out.P = P;
    out.n_block = n_block;
    const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total = slab * nonneg_count(n_block);
    out.f2.resize(total);
    out.vpair.resize(total);
    out.block_sizes.assign(nonneg_count(n_block), 0);

    core::compute_multigpu_partials_into(
        resources, Q, V, N, partition, shards_span,
        out.f2.data(), out.vpair.data(), out.block_sizes.data(), precision);

    if (requested_p2p_combine(resources, G) && !use_p2p) {
        STEPPE_LOG_WARN(
            "P2P combine unavailable (no peer access) -> host-staged fixed-order combine");
    }
    resources.last_combine_path = steppe::device::CombinePath::HostStaged;
    return out;
}

// Device-resident entry point — reference §3
steppe::device::DeviceF2Blocks compute_f2_blocks_multigpu_device(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision) {
    const int  P = Q.P;
    const long M = Q.M;
    const int  n_block = partition.n_block;

    validate_multigpu_inputs(Q, V, N, partition, M, "compute_f2_blocks_multigpu_device");
    const std::size_t G = require_at_least_one_device(resources, "compute_f2_blocks_multigpu_device");

    if (G == 1) {
        return resources.gpus[0].backend->compute_f2_blocks_device(
            Q, V, N, partition.block_id.data(), n_block, precision);
    }

    const bool use_p2p = select_p2p_combine(resources, G);

    const std::vector<steppe::device::DeviceShard> shards =
        core::plan_multigpu_shards(partition, M, n_block, G);
    const std::span<const steppe::device::DeviceShard> shards_span(shards.data(), shards.size());

    if (use_p2p) {
        std::vector<steppe::device::DevicePartial> partials =
            core::compute_multigpu_partials_resident(resources, Q, V, N, partition,
                                                     shards_span, precision);
        resources.last_combine_path = steppe::device::CombinePath::P2pDeviceResident;
        return steppe::device::combine_f2_partials_resident_device(
            std::span<steppe::device::DevicePartial>(partials.data(), partials.size()),
            shards_span, P, n_block, resources.gpus[0].device_id);
    }

    steppe::F2BlockTensor host =
        compute_f2_blocks_multigpu(resources, Q, V, N, partition, precision);
    return steppe::device::upload_f2_blocks_to_device(host, resources.gpus[0].device_id);
}

// Adaptive tiered entry point — reference §7
steppe::device::F2BlocksOut compute_f2_blocks_multigpu_tiered(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision,
    const steppe::device::RedecodeSource* redecode) {
    const int  P = Q.P;
    const long M = Q.M;
    const int  n_block = partition.n_block;

    // In re-decode mode Q/V/N.data is null (only P and M_kept are carried in the
    // MatView shape); skip validate's Q/V/N data-shape cross-checks. P + M come from
    // the MatView shape either way, and Resident is excluded by the caller's tier clamp.
    if (redecode == nullptr)
        validate_multigpu_inputs(Q, V, N, partition, M, "compute_f2_blocks_multigpu_tiered");
    (void)require_at_least_one_device(resources, "compute_f2_blocks_multigpu_tiered");

    const std::size_t free_vram_bytes = resources.gpus[0].caps.free_vram_bytes;
    const std::size_t free_host_bytes = steppe::device::free_host_ram_bytes();
    const steppe::device::OutputTier tier = steppe::device::resolve_output_tier(
        resources.config.force_tier, std::getenv(kEnvForceTier),
        P, M, n_block, free_vram_bytes, free_host_bytes);

    steppe::device::F2BlocksOut out;
    out.n_block = clamp_nonneg(n_block);

    {
        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(partition.block_id.data(),
                                                    nonneg_count(M)),
                               M, n_block);
        out.block_sizes.assign(nonneg_count(n_block), 0);
        for (int b = 0; b < n_block; ++b)
            out.block_sizes[static_cast<std::size_t>(b)] =
                static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }

    switch (tier) {
        case steppe::device::OutputTier::Resident: {
            out.tier = steppe::device::OutputTier::Resident;
            out.resident = resources.gpus[0].backend->compute_f2_blocks_device(
                Q, V, N, partition.block_id.data(), n_block, precision);
            out.P = out.resident.P;
            out.n_block = clamp_nonneg(out.resident.n_block);
            if (!out.resident.block_sizes.empty()) out.block_sizes = out.resident.block_sizes;
            break;
        }
        case steppe::device::OutputTier::HostRam: {
            out.tier = steppe::device::OutputTier::HostRam;
            steppe::device::StreamTarget target;
            target.tier = steppe::device::OutputTier::HostRam;
            target.host_dst = &out.host;
            finish_streamed_tier(resources, Q, V, N, partition, n_block, precision,
                                 target, out, out.host, redecode);
            if (out.host.n_block >= 0) out.n_block = out.host.n_block;
            break;
        }
        case steppe::device::OutputTier::Disk: {
            out.tier = steppe::device::OutputTier::Disk;
            std::string path = resources.config.disk_cache_path;
            if (path.empty()) {
                const char* env = std::getenv(kEnvF2CachePath);
                path = (env && env[0]) ? std::string(env)
                                       : std::string(steppe::kDefaultDiskCachePath);
            }
            steppe::device::StreamTarget target;
            target.tier = steppe::device::OutputTier::Disk;
            target.disk_path = path;
            target.disk_dst = &out.disk;
            finish_streamed_tier(resources, Q, V, N, partition, n_block, precision,
                                 target, out, out.disk, redecode);
            out.n_block = clamp_nonneg(out.disk.n_block);
            break;
        }
    }
    return out;
}

}  // namespace steppe::core
