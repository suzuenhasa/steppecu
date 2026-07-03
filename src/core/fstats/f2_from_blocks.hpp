// src/core/fstats/f2_from_blocks.hpp
//
// Host-side orchestration that assembles the bias-corrected f2 statistic from the
// Q/V/N genotype contract, dispatching the matrix math through an injected
// ComputeBackend. Host-pure and CUDA-free, so it compiles into `core` without
// pulling in the device toolkit.
//
// Reference: docs/reference/src_core_fstats_f2_from_blocks.hpp.md
#ifndef STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP
#define STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP

#include "device/backend.hpp"
#include "core/internal/views.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"

namespace steppe::core {

// compute_f2_block: one SNP block — reference §3
[[nodiscard]] F2Result compute_f2_block(ComputeBackend& backend, const MatView& Q,
                                        const MatView& V, const MatView& N,
                                        const Precision& precision);

// compute_f2_blocks: the full per-block tensor — reference §4
[[nodiscard]] F2BlockTensor compute_f2_blocks(ComputeBackend& backend, const MatView& Q,
                                              const MatView& V, const MatView& N,
                                              const BlockPartition& partition,
                                              const Precision& precision);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP
