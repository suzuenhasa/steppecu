// src/core/fstats/f2_from_blocks.hpp
//
// Declaration of the `core`-owned f2 assembly orchestration (architecture.md §5
// S2; ROADMAP §2, M0). See f2_from_blocks.cpp for the rationale. Host-pure and
// CUDA-free: it names only the CUDA-free ComputeBackend seam and the shared
// Q/V/N views, so it compiles into `core` without dragging in the device toolkit
// (architecture.md §4 layering rule).
#ifndef STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP
#define STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP

#include "device/backend.hpp"                  // steppe::ComputeBackend, steppe::F2Result
#include "core/internal/views.hpp"             // steppe::core::MatView (Q/V/N contract)
#include "core/domain/block_partition_rule.hpp" // steppe::core::BlockPartition
#include "steppe/config.hpp"                   // steppe::Precision
#include "steppe/fstats.hpp"                   // steppe::F2BlockTensor

namespace steppe::core {

/// Compute the bias-corrected f2 matrix + pairwise-valid counts for ONE SNP block
/// from the Q/V/N contract, dispatching through the injected `backend`
/// (architecture.md §5 S2, §8). The orchestration owns the policy; the backend
/// owns the implementation (CpuBackend oracle vs CudaBackend 3-GEMM). `precision`
/// governs only the GPU matmul-heavy GEMMs (architecture.md §12); the CPU oracle
/// ignores it. Returns f2 [P × P] + Vpair [P × P] (column-major), the latter
/// retained as the S4 jackknife weight (architecture.md §5 S2 caveat (a)).
///
/// PRECONDITION (debug fail-fast, cleanup B11): Q, V, N must share the same P and
/// M and have non-negative extents (the documented backend.hpp contract). In a
/// debug build a violation aborts here (STEPPE_ASSERT, file/line) rather than
/// reading past a short view inside the backend; under NDEBUG the check is
/// compiled out and the caller owns the contract.
[[nodiscard]] F2Result compute_f2_block(ComputeBackend& backend, const MatView& Q,
                                        const MatView& V, const MatView& N,
                                        const Precision& precision);

/// Compute the M4 PER-BLOCK f2 tensor `f2_blocks [P × P × n_block]` + retained
/// Vpair from the FULL per-SNP Q/V/N and a SNP→block partition, dispatching
/// through the injected `backend` (architecture.md §5 S2, §11.1; ROADMAP M4). The
/// `partition` comes from the shared `assign_blocks` (block_partition_rule.hpp) —
/// `core` owns the block policy; the backend owns the batched GPU implementation
/// (the spike-chosen size-grouped strided-batched design) vs the CPU per-block
/// long-double oracle. `precision` governs only the matmul-heavy GEMMs. This stays
/// CUDA-free: it names only the ComputeBackend seam + the host-pure block rule.
///
/// PRECONDITION (debug fail-fast, cleanup B11): in addition to the Q/V/N contract
/// (same P/M, non-negative), the `partition` must describe exactly the M columns —
/// `partition.block_id.size() == Q.M` (catches a short/null block_id, which the
/// backend's `block_ranges` cannot see across the length-erasing `(const int*,
/// int)` seam), `0 < n_block <= M`, and `block_id` dense/non-decreasing in
/// `[0, n_block)`. A violation aborts here in debug rather than driving a silent
/// out-of-bounds host-vector write / device read in the backend.
[[nodiscard]] F2BlockTensor compute_f2_blocks(ComputeBackend& backend, const MatView& Q,
                                              const MatView& V, const MatView& N,
                                              const BlockPartition& partition,
                                              const Precision& precision);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP
