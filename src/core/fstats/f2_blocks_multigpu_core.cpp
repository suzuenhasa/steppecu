// src/core/fstats/f2_blocks_multigpu_core.cpp
//
// The host-only, CUDA-free core of the multi-GPU f2-block precompute: the
// block-aligned shard plan and the three concurrent per-device fan-out entries
// that produce each device's partial (host-staged, device-resident, or written
// straight into a caller buffer), all over one shared fan-out helper. Touches
// only CUDA-free seams, so a GPU-free host test can link and drive it.
//
// Reference: docs/reference/src_core_fstats_f2_blocks_multigpu_core.cpp.md
#include "core/fstats/f2_blocks_multigpu_core.hpp"

#include <cstddef>
#include <exception>
#include <functional>
#include <span>
#include <thread>
#include <vector>

#include "core/internal/index_cast.hpp"
#include "core/internal/views.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "device/resources.hpp"
#include "device/shard_plan.hpp"
#include "device/device_partial.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"

namespace steppe::core {

namespace {

// Per-worker seam callable — reference §5
using ShardSeam = std::function<void(std::size_t /*g*/,
                                     const MatView& /*Qg*/, const MatView& /*Vg*/,
                                     const MatView& /*Ng*/, const int* /*block_id_local*/,
                                     int /*n_block_local*/,
                                     const steppe::device::DeviceShard& /*sh*/)>;

// Shared per-device concurrent fan-out — reference §4
void fan_out_shards(const MatView& Q, const MatView& V, const MatView& N,
                    const BlockPartition& partition,
                    std::span<const steppe::device::DeviceShard> shards,
                    const ShardSeam& seam) {
    const int P = Q.P;
    const std::size_t G = shards.size();
    std::vector<std::exception_ptr> worker_errors(G);
    {
        std::vector<std::jthread> workers;
        workers.reserve(G);
        for (std::size_t g = 0; g < G; ++g) {
            workers.emplace_back([&, g]() {
                try {
                    const steppe::device::DeviceShard& sh = shards[g];
                    const long s0 = sh.s0;
                    const long s1 = sh.s1;
                    const long M_local = s1 - s0;
                    const int  n_block_local = sh.b1 - sh.b0;

                    const std::size_t col_off =
                        idx(P) * nonneg_count(s0);
                    const MatView Qg{Q.data + col_off, P, M_local};
                    const MatView Vg{V.data + col_off, P, M_local};
                    const MatView Ng{N.data + col_off, P, M_local};

                    std::vector<int> block_id_local(
                        nonneg_count(M_local), 0);
                    for (long k = 0; k < M_local; ++k) {
                        block_id_local[idx(k)] =
                            partition.block_id[idx(s0 + k)] - sh.b0;
                    }

                    seam(g, Qg, Vg, Ng, block_id_local.data(), n_block_local, sh);
                } catch (...) {
                    worker_errors[g] = std::current_exception();
                }
            });
        }
    }

    for (std::size_t g = 0; g < G; ++g) {
        if (worker_errors[g]) {
            std::rethrow_exception(worker_errors[g]);
        }
    }
}

}  // namespace

// Block-aligned shard plan — reference §3
std::vector<steppe::device::DeviceShard> plan_multigpu_shards(
    const BlockPartition& partition, long M, int n_block, std::size_t G) {
    const std::vector<BlockRange> ranges = core::block_ranges(
        std::span<const int>(partition.block_id.data(),
                             nonneg_count(M)),
        M, n_block);

    return steppe::device::plan_block_shards(
        std::span<const BlockRange>(ranges.data(), ranges.size()), G);
}

// Host-staged partials fan-out — reference §5
std::vector<F2BlockTensor> compute_multigpu_partials(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision) {
    const std::size_t G = shards.size();

    std::vector<F2BlockTensor> partials(G);
    fan_out_shards(Q, V, N, partition, shards,
                   [&](std::size_t g, const MatView& Qg, const MatView& Vg,
                       const MatView& Ng, const int* block_id_local, int n_block_local,
                       const steppe::device::DeviceShard&) {
                       partials[g] = resources.gpus[g].backend->compute_f2_blocks(
                           Qg, Vg, Ng, block_id_local, n_block_local, precision);
                   });
    return partials;
}

// Device-resident partials fan-out — reference §5
std::vector<steppe::device::DevicePartial> compute_multigpu_partials_resident(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision) {
    const std::size_t G = shards.size();

    std::vector<steppe::device::DevicePartial> partials(G);
    fan_out_shards(Q, V, N, partition, shards,
                   [&](std::size_t g, const MatView& Qg, const MatView& Vg,
                       const MatView& Ng, const int* block_id_local, int n_block_local,
                       const steppe::device::DeviceShard& sh) {
                       partials[g] = resources.gpus[g].backend->compute_f2_blocks_resident(
                           Qg, Vg, Ng, block_id_local, n_block_local, sh.b0, precision);
                   });
    return partials;
}

// Direct-into-buffer partials fan-out — reference §5
void compute_multigpu_partials_into(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    double* dst_f2, double* dst_vpair, int* block_sizes_dst,
    const Precision& precision) {
    fan_out_shards(Q, V, N, partition, shards,
                   [&](std::size_t g, const MatView& Qg, const MatView& Vg,
                       const MatView& Ng, const int* block_id_local, int n_block_local,
                       const steppe::device::DeviceShard& sh) {
                       resources.gpus[g].backend->compute_f2_blocks_into(
                           Qg, Vg, Ng, block_id_local, n_block_local, sh.b0,
                           dst_f2, dst_vpair, block_sizes_dst, precision);
                   });
}

}  // namespace steppe::core
