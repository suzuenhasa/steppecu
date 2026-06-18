// src/device/p2p_combine.hpp
//
// The OPT-IN device-resident P2P f2 combine — DECLARATION (CUDA-FREE) of the
// single-node multi-GPU (SPMG) fast-path combine (architecture.md §11.4
// "Capability-tiered combine ... GPU 0 pulls each peer's partial via cudaMemcpyPeer
// (a byte-exact DMA copy) and sums them in the same fixed g = 0..G-1 order
// on-device — BIT-IDENTICAL to the host-staged combine and the single-GPU
// reference", §12 PARITY LAW; design §4, §5). GPU 0 (the combine root, gpus[0])
// pulls each peer device's COMPACT partial via cudaMemcpyPeer (MEASURED 55.6 GB/s,
// canAccessPeer==1 both directions on rtxbox) and sums on-device in the SAME FIXED
// g=0..G-1 device order the host-staged baseline uses.
//
// CUDA-FREE BY CONTRACT, exactly like device/backend_factory.hpp: it names only the
// public CUDA-free `F2BlockTensor` (the host result it returns) and the CUDA-free
// `DeviceShard` plan + std types — NO <cuda_runtime.h>. This split (CUDA-free decl
// here, CUDA definition in cuda/p2p_combine.cu) is what lets the CUDA-free core
// entry point `compute_f2_blocks_multigpu` (steppe::core, src/core/fstats) reach the
// device-resident combine WITHOUT pulling the CUDA toolkit into steppe_core — the
// same pattern make_cuda_backend uses (a CUDA-free factory decl in backend_factory
// .hpp, the `new CudaBackend` body in cuda_backend.cu; design §5 "Rename note").
// The .cu definition #includes this CUDA-free decl, mirroring cuda_backend.cu
// including backend_factory.hpp.
//
// WHY THIS IS BIT-IDENTICAL TO THE HOST-STAGED BASELINE (architecture.md §11.4,
// §12): the transport (cudaMemcpyPeer) only MOVES BYTES — the exact f2/vpair doubles
// each device's compute_f2_blocks produced — and the on-device add SUMS THE SAME
// FIXED g=0..G-1 ORDER onto a zero-initialized full tensor, the identical arithmetic
// combine_f2_partials_host performs on the host. "the transport only moves bytes;
// software fixes the order" — so the two combine tiers are parity-NEUTRAL siblings,
// and both equal the single-GPU reference. NEVER an NCCL AllReduce (its order varies
// with G; §12).
#ifndef STEPPE_DEVICE_P2P_COMBINE_HPP
#define STEPPE_DEVICE_P2P_COMBINE_HPP

#include <span>

#include "steppe/fstats.hpp"      // steppe::F2BlockTensor (public, CUDA-free — the host result)
#include "device/shard_plan.hpp"  // steppe::device::DeviceShard (CUDA-free plan)

namespace steppe::device {

/// Sum the G per-device COMPACT partials on the combine root (GPU `root_device_id`,
/// gpus[0]) in the FIXED g=0..G-1 device order via cudaMemcpyPeer + an on-device
/// FP64 placement-add, returning the full `[P × P × n_block_full]` F2BlockTensor
/// copied back to host. BIT-IDENTICAL to `core::combine_f2_partials_host` (design
/// §3) and to the single-GPU reference — both are the fixed-order sum of each
/// device's compact partial, PLACED at its block offset, onto a ZERO-initialized
/// full tensor (architecture.md §11.4, §12); this one performs that sum on-device,
/// after pulling each peer's partial across the PCIe/NVLink fabric with the
/// byte-exact `cudaMemcpyPeer` DMA.
///
/// THE TRANSPORT IS A REAL cudaMemcpyPeer (architecture.md §11.4): each partial g
/// whose owning device (`device_ids[g]`) is NOT the root is pre-staged onto that
/// owning (peer) device and then pulled peer->root with cudaMemcpyPeer (the credited
/// 55.6 GB/s byte-exact DMA on rtxbox); the root's own partial (`device_ids[g] ==
/// root_device_id`) uploads straight to the root (no self-peer-copy). `device_ids[g]`
/// IS the physical CUDA ordinal that COMPUTED partial g — i.e.
/// `resources.gpus[g].device_id`, in the FIXED g=0..G-1 == DeviceConfig::devices
/// order, with `device_ids[0] == root_device_id` (the root is gpus[0]). The CUDA-free
/// seam does not surface per-device ordinals on its own, so the caller threads them
/// in here; the transport is parity-NEUTRAL (it only moves bytes), so the result is
/// identical regardless of which physical ordinal stages each partial.
///
/// THE PEER-ACCESS GATE IS THE CALLER'S (architecture.md §11.4 §4): the caller
/// (`compute_f2_blocks_multigpu`) has ALREADY verified
/// `config.prefer_p2p_combine && gpus[0].caps.can_access_peer && G >= 2` before
/// calling — this routine does NOT re-probe `cudaDeviceCanAccessPeer` (it is the
/// chosen path). It DOES enable peer access root→each owning peer here via
/// `cudaDeviceEnablePeerAccess` routed through the NON-throwing STEPPE_CUDA_WARN
/// (check.cuh): `cudaErrorPeerAccessAlreadyEnabled` is an EXPECTED, tagged, non-fatal
/// status (the device may already be peer-enabled from a prior call). A GENUINE
/// peer-enable failure on a device the caller PROMISED is peer-reachable is a fault —
/// the DMA below would fail anyway — so the routine fails-fast through the throwing
/// STEPPE_CUDA_CHECK on the cudaMemcpyPeer itself (architecture.md §2 fail-fast); the
/// caller's degrade-to-host-staged decision is made BEFORE the call, on the
/// capability probe, never inside this routine.
///
/// PRECONDITIONS (fail-fast, architecture.md §2; same shape as
/// combine_f2_partials_host): `partials.size() == shards.size() == device_ids.size()`
/// (== G); every non-empty partial shares `P`; the union of the shard block ranges
/// tiles `[0, n_block_full)` contiguously; each partial g spans exactly its shard's
/// blocks (`partials[g].n_block == shards[g].b1 - shards[g].b0`). A violation throws.
///
/// @param partials        G compact F2BlockTensors in g=0..G-1 order; `partials[g]`
///                        is device g's `[P × P × (shards[g].b1 - shards[g].b0)]`
///                        result (HOST storage, from each device's compute_f2_blocks).
///                        An EMPTY shard's partial has n_block == 0 (placed nothing).
/// @param shards          the block-aligned plan (plan_block_shards); `shards[g].b0`
///                        is partial g's placement offset.
/// @param device_ids      the physical CUDA ordinal that COMPUTED each partial, in
///                        the FIXED g=0..G-1 order (== resources.gpus[g].device_id);
///                        `device_ids[0] == root_device_id`. The peer source of each
///                        cudaMemcpyPeer (the root's own partial skips the peer hop).
/// @param P               population count (the leading dim of every slab).
/// @param n_block_full    total block count of the combined tensor.
/// @param root_device_id  the combine root (gpus[0].device_id, GPU 0) — the device
///                        the accumulator lives on and that pulls each peer partial.
/// @return  the full `[P × P × n_block_full]` F2BlockTensor (host), BIT-IDENTICAL to
///          combine_f2_partials_host over the same partials/shards.
/// @throws std::runtime_error on a precondition violation; CudaError on a genuine
///         CUDA fault (allocation, the cudaMemcpyPeer DMA, the H2D upload, or the
///         result copy-back) — peer-enable "already enabled" is NOT a fault (it is
///         WARN-tagged and tolerated; see above).
[[nodiscard]] F2BlockTensor combine_f2_partials_p2p(
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    std::span<const int> device_ids,
    int P, int n_block_full, int root_device_id);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_P2P_COMBINE_HPP
