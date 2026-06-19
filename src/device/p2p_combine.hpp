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

#include "steppe/fstats.hpp"        // steppe::F2BlockTensor (public, CUDA-free — the host result)
#include "device/shard_plan.hpp"    // steppe::device::DeviceShard (CUDA-free plan)
#include "device/device_partial.hpp"  // steppe::device::DevicePartial (CUDA-free opaque resident handle)
#include "device/device_f2_blocks.hpp"  // steppe::device::DeviceF2Blocks (CUDA-free opaque FULL device-resident result handle)

namespace steppe::device {

/// Combine the G per-device RESIDENT partials on the combine root (GPU
/// `root_device_id`, gpus[0]) in the FIXED g=0..G-1 device order — consuming
/// DevicePartial handles whose f2/Vpair DeviceBuffers were LEFT RESIDENT by
/// `compute_f2_blocks_resident` (NO host bounce; the M4.5 cure, doc §4 Item 1) — and
/// returning the full `[P × P × n_block_full]` F2BlockTensor copied back to host with
/// ONE final D2H. BIT-IDENTICAL to `core::combine_f2_partials_host` (design §3) and to
/// the single-GPU reference: because the block-aligned shards are DISJOINT and TILE
/// `[0, n_block_full)` exactly, every result slab is written EXACTLY ONCE by its
/// owning device's resident slab-stack — so the combine is a verbatim PLACEMENT (a
/// raw D2D/peer byte copy into the disjoint slice), NOT a sum onto a zeroed
/// accumulator. The raw copy reproduces a −0.0 element byte-for-byte, exactly like the
/// host baseline's `std::copy_n`; there is NO cudaMemset and NO place-add `+=` (which
/// had a latent −0.0 flip masked only by memset(0); cleanup B7).
///
/// THE TRANSPORT IS A REAL cudaMemcpyPeer (architecture.md §11.4): each partial g
/// whose owning device (`partials[g].device_id`) is NOT the root is pulled straight
/// from its RESIDENT peer buffer (`impl->f2/impl->vpair.data()`) into its disjoint
/// result slice via `cudaMemcpyPeerAsync` (the credited 55.6 GB/s byte-exact DMA on
/// rtxbox); the root's own partial (`device_id == root_device_id`) is a plain
/// `cudaMemcpyAsync(...,DeviceToDevice)` into its slice — NO H2D re-upload, NO staging
/// buffer. Each handle carries its own `device_id` (the peer source) and `b0` (the
/// disjoint placement offset), so the CUDA-free caller no longer threads a parallel
/// `device_ids` span.
///
/// THE PEER-ACCESS GATE IS THE CALLER'S (architecture.md §11.4 §4): the caller
/// (`compute_f2_blocks_multigpu`) has ALREADY verified the four-term §4 gate — defined
/// ONCE at the `use_p2p` computation in f2_blocks_multigpu.cpp ("THE §4 COMBINE GATE",
/// §8 single-source) — before calling, so this routine does NOT re-probe
/// `cudaDeviceCanAccessPeer` (it is the chosen path). That gate's
/// `config.enable_peer_access` term is the user's PERMISSION (MAY-WE) for exactly the
/// `cudaDeviceEnablePeerAccess` this routine calls per peer, so reaching here implies
/// that permission was granted (cleanup C-1). `cudaErrorPeerAccessAlreadyEnabled` is an
/// EXPECTED, tagged, non-fatal status (WARN-tolerant; the device may already be
/// peer-enabled from a prior call). A GENUINE peer-enable failure surfaces on the
/// cudaMemcpyPeerAsync below via the throwing STEPPE_CUDA_CHECK (architecture.md §2).
///
/// LIFETIME: this routine READS the resident buffers in place and does NOT take
/// ownership or free them — the caller's `partials` vector outlives this call and the
/// handles free AFTER it returns (§7); the final `cudaStreamSynchronize` drains every
/// DMA before returning, so no read outlives a freed source.
///
/// PRECONDITIONS (fail-fast, architecture.md §2; same shape contract as
/// combine_f2_partials_host): `partials.size() == shards.size()` (== G); every
/// non-empty partial shares `P`; each partial g spans exactly its shard's blocks
/// (`partials[g].n_block_local == shards[g].b1 - shards[g].b0`) at offset
/// `partials[g].b0 == shards[g].b0`; the union of the shard block ranges tiles
/// `[0, n_block_full)` contiguously. A violation throws.
///
/// @param partials        G resident DevicePartial handles in g=0..G-1 order (NON-const
///                        span: the resident buffers are read in place; the handles are
///                        not modified but the span is non-const to mirror the
///                        consume-in-place intent). `partials[g]` is device g's
///                        `[P × P × (shards[g].b1 - shards[g].b0)]` partial, RESIDENT on
///                        `partials[g].device_id`. An EMPTY shard's handle has
///                        n_block_local == 0 / impl == nullptr (placed nothing).
/// @param shards          the block-aligned plan (plan_block_shards); kept for the
///                        authoritative tiling cross-check in validate_resident_partials
///                        (`shards[g].b0`/`.b1` vs each handle's b0/n_block_local).
/// @param P               population count (the leading dim of every slab).
/// @param n_block_full    total block count of the combined tensor.
/// @param root_device_id  the combine root (gpus[0].device_id, GPU 0) — the device the
///                        full result lives on and that pulls each peer partial.
/// @return  the full `[P × P × n_block_full]` F2BlockTensor (host), BIT-IDENTICAL to
///          combine_f2_partials_host over the same partials/shards.
/// @throws std::runtime_error on a precondition violation; CudaError on a genuine
///         CUDA fault (allocation, the cudaMemcpyPeerAsync DMA, the D2D copy, or the
///         final D2H) — peer-enable "already enabled" is NOT a fault (WARN-tolerant).
[[nodiscard]] F2BlockTensor combine_f2_partials_resident(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id);

/// DEVICE-RESIDENT assembly (M4.5 device-resident output): identical to
/// combine_f2_partials_resident — same fixed g=0..G-1 disjoint placement via D2D /
/// cudaMemcpyPeerAsync of each resident partial into one root-resident full tensor —
/// but it STOPS before the final D2H and returns the assembled tensor as a VRAM
/// DeviceF2Blocks (root_device_id-resident). NO host F2BlockTensor, NO final D2H. The
/// caller may .to_host() it on request. Bit-identical assembly to
/// combine_f2_partials_resident (same bytes, same placement; §12). block_sizes are
/// placed host-side in the fixed g order onto the returned handle.
[[nodiscard]] DeviceF2Blocks combine_f2_partials_resident_device(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_P2P_COMBINE_HPP
