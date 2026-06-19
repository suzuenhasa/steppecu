// src/core/fstats/f2_blocks_multigpu_core.hpp
//
// The HOST-PURE, CUDA-FREE, P2P-FREE core of the SPMG precompute orchestrator
// (architecture.md §11.4 SPMG, §12 parity; design §5; cleanup D1/T1 + B9). The
// public entry point compute_f2_blocks_multigpu (f2_blocks_multigpu.{hpp,cpp}) is a
// thin composition over the two steps declared here PLUS the §4 capability-gated
// combine fork (host-staged baseline vs the opt-in device-resident cudaMemcpyPeer
// fast-path). The combine fork — and ONLY it — references the CUDA-RDC device symbol
// combine_f2_partials_p2p (device/p2p_combine.hpp), so it stays in the entry-point TU.
//
// WHY THIS SPLIT EXISTS (cleanup D1/T1, B9). These two steps — the block-aligned
// shard PLAN and the per-device concurrent FAN-OUT that produces each device's
// compact partial — are the host-pure heart of the multi-GPU algorithm: the
// sub-view / dense-local-block-id transform, the std::jthread fan-out, the
// exception-ptr rethrow, the empty-/n_block<G-shard handling. They depend ONLY on
// the CUDA-free ComputeBackend seam (Resources::gpus[g].backend->compute_f2_blocks),
// the CUDA-free plan_block_shards, and core::block_ranges — NEVER on the device-RDC
// P2P symbol. Factoring them here lets a GPU-FREE host .cpp test
// (tests/unit/test_f2_blocks_multigpu.cpp) drive them against a FAKE ComputeBackend
// with NO GPU, NO CUDA toolkit, and NO device-link — the fast inner-loop gate the
// slow GPU parity test (tests/reference/test_f2_multigpu_parity.cu) cannot be (it
// must device-link the real CUDA backend + the P2P combine). The locked §12
// bit-identity is UNTOUCHED: this is a pure refactor — the entry point calls these
// in the identical sequence over the identical inputs, so the partials and the shard
// plan are byte-for-byte what the inline body produced; the combine reads partials[g]
// in the fixed g=0..G-1 order AFTER the join barrier exactly as before.
//
// CUDA-FREE, host-pure, in steppe::core: it names only the CUDA-free Resources /
// MatView / BlockPartition / Precision / F2BlockTensor / DeviceShard, so it compiles
// into steppe_core without the device toolkit (design §5, §8). A CUDA leak here would
// fail the host-only test compile (the §4 layering proof).
#ifndef STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_CORE_HPP
#define STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_CORE_HPP

#include <cstddef>
#include <span>
#include <vector>

#include "core/internal/views.hpp"               // steppe::core::MatView (Q/V/N contract)
#include "core/domain/block_partition_rule.hpp"  // steppe::core::BlockPartition
#include "steppe/config.hpp"                      // steppe::Precision
#include "steppe/fstats.hpp"                      // steppe::F2BlockTensor
#include "device/resources.hpp"                   // steppe::device::Resources (CUDA-free)
#include "device/shard_plan.hpp"                  // steppe::device::DeviceShard (CUDA-free plan)
#include "device/device_partial.hpp"             // steppe::device::DevicePartial (CUDA-free opaque resident handle)

namespace steppe::core {

/// Build the BLOCK-ALIGNED shard plan for the G >= 2 multi-GPU precompute: invert the
/// SNP→block partition via the single-source core::block_ranges (which validates the
/// partition contract once) and hand the per-block column ranges to the CUDA-free
/// plan_block_shards (the single home of the block→device mapping, §8). Returns
/// exactly G DeviceShards in g=0..G-1 order, contiguous and tiling [0, n_block) (with
/// trailing empty shards when n_block < G).
///
/// Pure function of (partition, M, n_block, G) — no device, no Resources, no CUDA. The
/// SOLE input to the planner is `ranges` (each block's SNP count is ranges[b].size(),
/// a `long`; cleanup B6/X1 dropped the redundant block_sizes vector).
///
/// @param partition  the SHARED SNP→block partition (assign_blocks); `block_id`
///                   non-decreasing ⇒ each block's columns are contiguous.
/// @param M          the SNP count (== Q.M); the planner span is block_id[0..M).
/// @param n_block    the block count (== partition.n_block).
/// @param G          number of devices (Resources::device_count()); must be >= 1.
/// @return  exactly G DeviceShards in g=0..G-1 order.
/// @throws std::runtime_error via plan_block_shards if G == 0; via core::block_ranges
///         on a malformed partition.
[[nodiscard]] std::vector<steppe::device::DeviceShard> plan_multigpu_shards(
    const BlockPartition& partition, long M, int n_block, std::size_t G);

/// Compute the G per-device COMPACT partials over zero-copy column sub-views, fanned
/// out CONCURRENTLY (one std::jthread per device; architecture.md §7, §11.4 — the SPMG
/// speedup). Each device g computes its [P × P × (b1-b0)] partial from the column
/// SUB-VIEW Q/V/N[s0,s1) and a dense zero-based LOCAL block_id (block_id[s0+k] - b0),
/// via the UNMODIFIED ComputeBackend::compute_f2_blocks (NOT reimplemented; design §0).
/// An EMPTY shard (b0 == b1) hands the backend n_block_local == 0 / M_local == 0 and
/// the backend early-returns an empty F2BlockTensor (the combine then places nothing).
///
/// CONCURRENCY + PARITY (§12). Each worker writes ONLY its own pre-sized partials[g]
/// slot (distinct vector elements, no resize/aliasing), builds its own block_id_local,
/// and drives its own backend gpus[g] bound to its own device (cudaSetDevice is a
/// per-host-thread property), so there is NO shared mutable state. The per-device GEMM
/// bits are fixed by the block-aligned shard and are INDEPENDENT of which wall-clock
/// slot ran them, so the fan-out is PARITY-NEUTRAL — bit-identical to a serial drive.
/// The returned partials[g] are in the fixed g=0..G-1 order; the caller combines them
/// AFTER this returns (the join barrier is the happens-before the fixed-order combine
/// depends on).
///
/// EXCEPTION SAFETY. A std::jthread whose entry function lets an exception escape calls
/// std::terminate ([thread.thread.constr]); each worker therefore catches everything
/// into its own std::exception_ptr and this function rethrows the FIRST (lowest g)
/// after the join barrier, so a backend / device fault propagates as a normal throw,
/// deterministically (independent of which worker raced to fail first).
///
/// @param resources  the G-device bundle; gpus[g] drives device g (non-const: each
///                   compute_f2_blocks mutates the backend's device scratch).
/// @param Q,V,N      the FULL per-SNP Q/V/N contract, column-major [P × M]; each
///                   device receives a zero-copy column sub-view (Q.data + P*s0).
/// @param partition  the SHARED partition (for the dense local block_id transform).
/// @param shards     the block-aligned plan from plan_multigpu_shards (length G).
/// @param precision  forwarded UNCHANGED to every compute_f2_blocks (§12).
/// @return  G compact partials in g=0..G-1 order (partials[g] for shards[g]).
/// @throws  rethrows the first worker failure (runtime_error on a malformed
///          sub-partition, CudaError on a device fault).
[[nodiscard]] std::vector<F2BlockTensor> compute_multigpu_partials(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision);

/// The DEVICE-RESIDENT sibling of compute_multigpu_partials (M4.5 cure, doc §4 Item 1):
/// the EXACT same concurrent jthread fan-out — same zero-copy column sub-views, same
/// dense zero-based LOCAL block_id, same exception-ptr rethrow, same empty-shard
/// handling — but each worker calls the RESIDENT seam method
/// `compute_f2_blocks_resident(..., sh.b0, precision)`, which LEAVES the device g's
/// f2/Vpair partial RESIDENT on device g (NO D2H, NO free) and returns an opaque,
/// move-only `DevicePartial` owning it. The handles are MOVED into the pre-sized
/// `partials[g]` slot (a pointer swap, no CUDA call, safe from any thread) and SURVIVE
/// the jthread join — the returned vector outlives the workers, and the resident
/// buffers free only AFTER the device-resident combine consumed them (§7). The result
/// feeds combine_f2_partials_resident (device/p2p_combine.hpp).
///
/// CUDA-FREE: `DevicePartial` is CUDA-free (opaque pimpl) and `compute_f2_blocks_resident`
/// is a virtual on the CUDA-free ComputeBackend seam, so this function compiles into
/// steppe_core without the device toolkit, exactly like its host sibling.
///
/// @param resources  the G-device bundle; gpus[g] drives device g (non-const).
/// @param Q,V,N      the FULL per-SNP Q/V/N contract, column-major [P × M]; each device
///                   receives a zero-copy column sub-view (Q.data + P*s0).
/// @param partition  the SHARED partition (for the dense local block_id transform).
/// @param shards     the block-aligned plan from plan_multigpu_shards (length G);
///                   `shards[g].b0` is the global placement offset carried on the handle.
/// @param precision  forwarded UNCHANGED to every compute_f2_blocks_resident (§12).
/// @return  G resident DevicePartial handles in g=0..G-1 order (partials[g] for shards[g]).
/// @throws  rethrows the first worker failure (runtime_error on a malformed sub-partition,
///          CudaError on a device fault).
[[nodiscard]] std::vector<steppe::device::DevicePartial> compute_multigpu_partials_resident(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision);

/// Host-staged-DIRECT fan-out (M4.5 d2h-speed cure): the EXACT same concurrent
/// jthread fan-out as compute_multigpu_partials — same zero-copy sub-views, same
/// dense local block_id, same exception-ptr rethrow, same empty-shard handling — but
/// each worker calls compute_f2_blocks_into(..., sh.b0, dst_f2, dst_vpair,
/// block_sizes_dst, precision), which D2Hs its compact f2/vpair DIRECTLY (pinned)
/// into its DISJOINT slice [slab*b0, slab*(b0+nb)) of the SHARED result the caller
/// pre-allocated. No per-device F2BlockTensor, no combine copy. The disjoint
/// block-aligned shards guarantee the G workers write non-overlapping slices, so the
/// concurrent writes into one buffer are race-free; PARITY-NEUTRAL (same bytes, same
/// offsets, fixed g order independent of wall-clock).
///
/// @param dst_f2/dst_vpair  the SHARED result f2/vpair base pointers, pre-sized to
///                          P*P*n_block_full by the caller (the orchestrator).
/// @param block_sizes_dst   the SHARED block_sizes base, pre-sized to n_block_full.
/// @throws rethrows the first worker failure (lowest g), deterministically.
void compute_multigpu_partials_into(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    double* dst_f2, double* dst_vpair, int* block_sizes_dst,
    const Precision& precision);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_CORE_HPP
