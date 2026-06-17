// src/core/fstats/f2_blocks_multigpu.hpp
//
// The single-node multi-GPU (SPMG) precompute ENTRY POINT — the seam the
// API / f2_from_blocks layer calls to compute the per-block f2 tensor across the
// G devices of a Resources bundle (architecture.md §5 S2, §11.4 SPMG, §9 Resources
// injection; design §5). It shards whole BLOCKS across the G devices, computes each
// device's partial via the UNMODIFIED per-device `CudaBackend::compute_f2_blocks`
// (it does NOT reimplement the per-block GEMM; design §0-§2), and COMBINES the
// partials in the FIXED g=0..G-1 device order — BIT-IDENTICAL to the single-GPU
// reference and identical across G (architecture.md §12 PARITY LAW).
//
// CUDA-FREE, host-pure, in `steppe::core`: it names only the CUDA-free
// `Resources` / `MatView` / `BlockPartition` / `Precision` / `F2BlockTensor` and
// the CUDA-free host-staged combine, so it compiles into steppe_core without the
// device toolkit (design §5, §8). The combine path is the host-staged baseline
// (the portable parity baseline, architecture.md §11.4); the opt-in device-resident
// cudaMemcpyPeer fast-path (device/p2p_combine.hpp) is a separate unit and is
// bit-identical to the baseline by construction (§12).
#ifndef STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_HPP
#define STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_HPP

#include "core/internal/views.hpp"               // steppe::core::MatView (Q/V/N contract)
#include "core/domain/block_partition_rule.hpp"  // steppe::core::BlockPartition
#include "steppe/config.hpp"                      // steppe::Precision
#include "steppe/fstats.hpp"                      // steppe::F2BlockTensor
#include "device/resources.hpp"                   // steppe::device::Resources (CUDA-free)

namespace steppe::core {

/// Compute the per-block f2 tensor `f2_blocks [P × P × n_block]` + the retained
/// per-block `Vpair` across the G devices in `resources`, BIT-IDENTICAL to the
/// single-GPU `CudaBackend::compute_f2_blocks` over the same full inputs
/// (architecture.md §12). The bit-identity holds because the sharding is
/// BLOCK-ALIGNED — each block is computed entirely on one device from exactly its
/// own contiguous SNP columns, so its slab bits equal the single-GPU slab — and the
/// combine sums the per-device partials in the fixed g=0..G-1 order onto a
/// zero-initialized full tensor (design §0).
///
/// G == 1 (`resources.device_count() == 1`) is the EXACT current single-GPU path:
/// no shard, no combine — it calls `resources.gpus[0].backend->compute_f2_blocks`
/// over the FULL Q/V/N + partition and returns it UNCHANGED (zero behavior change,
/// bit-for-bit the existing result; design §5). G >= 2: plan_block_shards →
/// per-device sub-view compute → host-staged fixed-order combine
/// (combine_f2_partials_host, the portable parity baseline; architecture.md §11.4).
///
/// The per-device CUDA work is driven SEQUENTIALLY (one host thread) for this unit:
/// the combine is off the bandwidth critical path (kB-MB; §11.4) and per-device
/// parallel host threads are a later performance workflow, NOT a parity concern —
/// parity is independent of execution concurrency (each device's bits are fixed by
/// its block-aligned shard).
///
/// @param resources  the G-device bundle (build_resources). `gpus[g]` in the FIXED
///                   g=0..G-1 combine order (DeviceConfig::devices order). Non-const
///                   because driving each `backend->compute_f2_blocks` mutates the
///                   backend's device-side scratch (the backends are move-only,
///                   owned here).
/// @param Q,V,N      the FULL per-SNP Q/V/N contract, column-major [P × M]
///                   (views.hpp). Each device receives a zero-copy column sub-view.
/// @param partition  the SHARED SNP→block partition (assign_blocks); `block_id`
///                   non-decreasing ⇒ each block's columns are contiguous, the
///                   property block-aligned sharding requires (design §2).
/// @param precision  governs the per-device f2 GEMMs only (architecture.md §12),
///                   forwarded UNCHANGED to each compute_f2_blocks so every device
///                   uses the same fixed-slice Ozaki / native FP64 path.
/// @return  the combined full F2BlockTensor.
[[nodiscard]] F2BlockTensor compute_f2_blocks_multigpu(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_BLOCKS_MULTIGPU_HPP
