// src/core/fstats/f2_from_blocks.cpp
//
// Host-side composition root for the f2 computation: validates the input
// contract and the block partition, then dispatches to an injected, CUDA-free
// ComputeBackend (CPU oracle or GPU). This file computes nothing itself and
// includes no CUDA header.
//
// Reference: docs/reference/src_core_fstats_f2_from_blocks.cpp.md
#include "core/fstats/f2_from_blocks.hpp"

#include "device/backend.hpp"
#include "core/internal/views.hpp"
#include "core/internal/host_device.hpp"
#include "core/internal/index_cast.hpp"
#include "core/internal/qvn_assert.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "steppe/fstats.hpp"
#include "steppe/config.hpp"

#include <cstddef>

namespace steppe::core {

namespace {

// Dense/non-decreasing block-id scan — reference §5
[[maybe_unused]] [[nodiscard]]
bool block_ids_dense_nondecreasing(const BlockPartition& partition, long M) {
    int prev = -1;
    for (long s = 0; s < M; ++s) {
        const int id = partition.block_id[idx(s)];
        if (id < 0 || id >= partition.n_block || id < prev) return false;
        prev = id;
    }
    return true;
}

// Block-partition contract — reference §5
void validate_partition([[maybe_unused]] const BlockPartition& partition,
                        [[maybe_unused]] long M) {
    STEPPE_ASSERT(partition.block_id.size() == nonneg_count(M),
                  "compute_f2_blocks: block_id length != M (partition does not "
                  "describe exactly the SNP columns)");
    STEPPE_ASSERT(M <= 0 || partition.n_block > 0,
                  "compute_f2_blocks: n_block <= 0 with M > 0 columns");
    STEPPE_ASSERT(M <= 0 || static_cast<long>(partition.n_block) <= M,
                  "compute_f2_blocks: n_block > M (more blocks than SNPs)");
    STEPPE_ASSERT(M <= 0 || block_ids_dense_nondecreasing(partition, M),
                  "compute_f2_blocks: block_id has an out-of-range or "
                  "non-decreasing entry (malformed partition)");
}

}  // namespace

// f2 entry points — reference §2
F2Result compute_f2_block(ComputeBackend& backend, const MatView& Q, const MatView& V,
                          const MatView& N, const Precision& precision) {
    assert_qvn_consistent(Q, V, N);
    return backend.compute_f2(Q, V, N, precision);
}

F2BlockTensor compute_f2_blocks(ComputeBackend& backend, const MatView& Q, const MatView& V,
                                const MatView& N, const BlockPartition& partition,
                                const Precision& precision) {
    assert_qvn_consistent(Q, V, N);
    validate_partition(partition, Q.M);
    return backend.compute_f2_blocks(Q, V, N, partition.block_id.data(),
                                     partition.n_block, precision);
}

}  // namespace steppe::core
