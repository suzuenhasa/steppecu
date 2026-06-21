// src/device/cuda/p2p_combine.cu
//
// combine_f2_partials_resident — the OPT-IN device-resident P2P f2 combine,
// DEFINITION (architecture.md §11.4 capability-tiered combine, §12 PARITY LAW;
// design §4). This is the CUDA translation unit: it owns the cross-device DMA
// (cudaMemcpyPeerAsync) and the on-device disjoint PLACEMENT, so it is PRIVATE to
// steppe_device (architecture.md §4 layering — CUDA headers never compile into
// core/api/CLI). It #includes the CUDA-FREE declaration `device/p2p_combine.hpp`
// (the same pattern cuda_backend.cu uses to define the CUDA-free-declared
// make_cuda_backend; design §5), so the CUDA-free core entry point reaches this
// routine without seeing a CUDA header.
//
// THE M4.5 CURE (doc §4 Item 1): the per-device compute now LEAVES its f2/Vpair
// partial RESIDENT on the device that computed it (compute_f2_blocks_resident ->
// DevicePartial). This routine consumes those resident handles directly: it allocates
// ONE full-shape device result on the root, D2D-copies the root's own resident partial
// into its disjoint slice, cudaMemcpyPeerAsync's each PEER's resident partial straight
// into its disjoint slice, then does the ONE final D2H. NO host bounce, NO H2D
// re-upload, NO staging buffer, NO zeroed accumulator, NO place-add. The prior path's
// second full 7.14 GB D2H (the +1017 ms regression, doc §1) is gone.
//
// BIT-IDENTITY ARGUMENT (architecture.md §11.4, §12; design §4). The result is
// BIT-IDENTICAL to core::combine_f2_partials_host and to the single-GPU reference
// because:
//   (1) The bytes are the SAME. Each device's resident partial holds the exact
//       f2/vpair doubles its GEMM produced (block-aligned shard ⇒ each block's slab
//       bits equal the single-GPU run's, design §0). The D2D copy and the
//       cudaMemcpyPeerAsync DMA are byte-exact — they MOVE those bytes, they do not
//       recompute them ("the transport only moves bytes").
//   (2) The PLACEMENT is the SAME. Block-aligned shards are DISJOINT and tile
//       [0, n_block_full) exactly (validate's covered == n_block_full), so every
//       result slab is written EXACTLY ONCE by its owner at the SAME disjoint offset
//       slab·b0 the host baseline uses (f2_combine.cpp:103-104). A raw byte copy into
//       that slice is the BYTE-FAITHFUL placement — it reproduces a −0.0 element
//       exactly, like std::copy_n. The OLD memset(0)+place-add (`+=`) had a latent
//       −0.0 flip masked only by the memset; DELETING it (copy, never add) keeps the
//       result faithful AND removes the second D2H (cleanup B7). The fixed g=0..G-1
//       order is preserved for block_sizes placement (host int).
// So the two combine tiers are parity-NEUTRAL siblings (architecture.md §11.4) and
// the §12 PARITY LAW (bit-identical across G and to single-GPU) holds on this tier.
// NEVER an NCCL AllReduce (its reduction order varies with G; §12).
#include "device/p2p_combine.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <vector>  // std::vector<int>& (the shared place_partials_into block_sizes out-param)

#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK (fault), STEPPE_CUDA_WARN (recoverable peer-enable)
#include "device/cuda/device_buffer.cuh"    // steppe::device::DeviceBuffer<double> (the allowlisted RAII owner)
#include "device/cuda/device_partial_impl.cuh"  // DevicePartial::Impl (resident DeviceBuffer<double> f2/vpair owners)
#include "device/cuda/device_f2_blocks_impl.cuh"  // DeviceF2Blocks::Impl (the device-resident FULL result buffers)
#include "device/cuda/pinned_buffer.cuh"    // steppe::device::RegisteredHostRegion (pin the final D2H — M4.5 d2h-speed)
#include "device/cuda/stream.hpp"           // steppe::device::Stream (owning non-blocking root stream)
#include "steppe/fstats.hpp"                // steppe::F2BlockTensor
#include "device/shard_plan.hpp"            // steppe::device::DeviceShard
#include "core/fstats/f2_partials_validate.hpp"  // shared validate_resident_partials (Option A)

namespace steppe::device {

namespace {

/// FIXED-ORDER g = 0 .. G-1 placement of each device's resident partial into the
/// DISJOINT slice [slab·b0, slab·(b0+nb)) of ONE root-resident full result, the SINGLE
/// home of the placement loop the two combine entries used to copy-paste verbatim
/// (they differed ONLY by their dst base pointers — local result_* vs the
/// DeviceF2Blocks::Impl raw pointers; [7.1] dedup). Also fills out_block_sizes[b0+lb]
/// host-side in the same fixed g order (int, no device math). The caller has already
/// bound the root device + created root_stream and drains it ONCE after this returns.
///
/// PARITY (architecture.md §11.4/§12): a peer-access fix to the peer branch now lands
/// in BOTH entries at once (the wart the finding flags). The placement is byte-exact —
/// the root's own partial via D2D, each peer's via the byte-faithful cudaMemcpyPeerAsync
/// DMA (signature DOC-VERIFIED against the CUDA 13.x Runtime API:
/// cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)) — reproducing
/// −0.0 exactly, NO place-add, NO accumulator zeroing. cudaErrorPeerAccessAlreadyEnabled
/// is the EXPECTED status on the 2nd+ combine (the link is already up) — a tagged,
/// non-fatal degrade through the NON-throwing STEPPE_CUDA_WARN; the sticky last-error it
/// may set is then cleared (cudaGetLastError reads AND resets) so a later error check
/// does not surface this stale, already-handled status. A genuine peer-enable failure on
/// a device the caller PROMISED is peer-reachable surfaces via the throwing
/// STEPPE_CUDA_CHECK on the copy. NO per-peer sync here: the resident peer buffers are
/// NOT freed mid-loop (they live on the DevicePartial until the caller frees them AFTER
/// the combine, §7), so there is nothing to fence against; ONE drain below covers all.
void place_partials_into(double* dst_f2_base, double* dst_vpair_base,
                         std::span<DevicePartial> partials, std::size_t slab,
                         int root_device_id, cudaStream_t root_stream,
                         std::vector<int>& out_block_sizes) {
    for (std::size_t g = 0; g < partials.size(); ++g) {
        DevicePartial& part = partials[g];

        // block_sizes: place this device's per-block SNP counts at offset b0 (host int,
        // identical to the host baseline). Done even for an empty shard before the
        // early-continue.
        for (int lb = 0; lb < part.n_block_local; ++lb) {
            out_block_sizes[static_cast<std::size_t>(part.b0 + lb)] =
                part.block_sizes[static_cast<std::size_t>(lb)];
        }
        if (part.empty()) continue;  // empty shard: no resident buffers, nothing to copy

        const std::size_t part_elems = slab * static_cast<std::size_t>(part.n_block_local);
        const std::size_t part_bytes = part_elems * sizeof(double);
        const std::size_t dst_off = slab * static_cast<std::size_t>(part.b0);  // disjoint slice base

        double* dst_f2 = dst_f2_base + dst_off;
        double* dst_vpair = dst_vpair_base + dst_off;
        const double* src_f2 = part.impl->f2.data();      // resident on part.device_id
        const double* src_vpair = part.impl->vpair.data();

        if (part.device_id == root_device_id) {
            // ROOT's own resident partial: D2D copy into its disjoint slice (no peer hop).
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst_f2, src_f2, part_bytes,
                                              cudaMemcpyDeviceToDevice, root_stream));
            STEPPE_CUDA_CHECK(cudaMemcpyAsync(dst_vpair, src_vpair, part_bytes,
                                              cudaMemcpyDeviceToDevice, root_stream));
        } else {
            // PEER's resident partial: enable peer access root<-device_id. The enable is
            // DIRECTIONAL and set FROM the device that ISSUES the access (the root, which
            // reads the peer's buffer), so it is enabled while the ROOT is current,
            // naming `part.device_id` as the peer. AlreadyEnabled is expected on the 2nd+
            // combine (tagged, non-fatal); then clear the sticky last-error.
            (void)STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(part.device_id, 0));
            (void)cudaGetLastError();
            // Pull peer->root via the byte-exact cudaMemcpyPeerAsync DMA straight into the
            // disjoint slice (NO H2D, NO stage). With peer access enabled it is a direct
            // fabric DMA. A genuine peer-enable failure on a device the caller PROMISED is
            // peer-reachable surfaces here via the throwing STEPPE_CUDA_CHECK.
            STEPPE_CUDA_CHECK(cudaMemcpyPeerAsync(dst_f2, root_device_id,
                                                  src_f2, part.device_id, part_bytes, root_stream));
            STEPPE_CUDA_CHECK(cudaMemcpyPeerAsync(dst_vpair, root_device_id,
                                                  src_vpair, part.device_id, part_bytes, root_stream));
        }
        // NO per-peer cudaDeviceSynchronize here (Item 2): the resident peer buffers are
        // NOT freed mid-loop (they live on the DevicePartial until the caller frees them
        // AFTER this returns, §7). ONE sync (by the caller) drains all the enqueued copies.
    }
}

}  // namespace

F2BlockTensor combine_f2_partials_resident(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id) {
    // Fail-fast precondition guard. validate_resident_partials is the device-resident
    // sibling of validate_f2_partials (the host tier's guard): the SAME contract
    // (count, P, span, b0, tiling) checked over DevicePartial handles, so the two
    // tiers reject IDENTICALLY (cleanup B5; architecture.md §2, §8). Their
    // parity-neutrality (§11.4, §12) requires it.
    steppe::core::validate_resident_partials(
        "steppe::device::combine_f2_partials_resident", partials, shards, P, n_block_full);

    // ---- Bind to the root for the whole routine (the result + every copy land here).
    // RAII-restore the caller's device on exit: the caller drives each per-device
    // compute (which cudaSetDevice-binds its own device); leaving root_device_id
    // current would silently retarget a later call. The guard is exception-safe (it
    // fires on the early-throw / fault paths too). cudaFree on the resident handles
    // (freed by the caller AFTER this returns) is pointer-device-aware, so the
    // restored device does not matter for their teardown (§7).
    int prev_device = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&prev_device));
    struct DeviceGuard {
        int dev;
        explicit DeviceGuard(int d) noexcept : dev(d) {}
        ~DeviceGuard() { (void)STEPPE_CUDA_WARN(cudaSetDevice(dev)); }
        // Side-effecting dtor (cudaSetDevice restore) ⇒ non-copyable AND non-movable
        // (standard §2.12): a copy/move would re-fire the restore (a redundant rebind).
        // Scope-restore guard, never relocated, so all four are deleted — not move-only.
        // The explicit int ctor is REQUIRED: a user-declared (deleted) copy/move ctor
        // makes the class a non-aggregate under C++20 (P1008R1 — "no user-declared ...
        // constructors"), so the `restore{prev_device}` init can no longer rely on
        // aggregate init and must reach a real ctor. Behavior-neutral. [16.3]
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
        DeviceGuard(DeviceGuard&&) = delete;
        DeviceGuard& operator=(DeviceGuard&&) = delete;
    } restore{prev_device};  // restore on scope exit (teardown WARN, never throw)
    STEPPE_CUDA_CHECK(cudaSetDevice(root_device_id));

    // The combine's OWN non-blocking stream on the root (the backends' per-device
    // streams are private to the backends). One non-default stream lets the peer
    // copies pipeline (Item 2) and keeps the combine off the legacy default stream.
    Stream root_stream_owner;  // created on the current device (root), cudaStreamNonBlocking
    const cudaStream_t root_stream = root_stream_owner.get();

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    // validate_resident_partials already rejected a negative n_block_full.
    const std::size_t total = slab * static_cast<std::size_t>(n_block_full);

    // ---- ONE full-shape device RESULT on the root (NO memset) ----------------
    // Every block of [0, n_block_full) is written EXACTLY ONCE by its owning device's
    // disjoint slab-stack (validate's covered == n_block_full guarantees the tiling),
    // so there is NO unwritten slab to zero and NO overlap to accumulate. The raw
    // D2D/peer copy into the disjoint slice IS the byte-faithful placement (it
    // reproduces −0.0 exactly, like the host std::copy_n; a memset+`+=` would have
    // flipped −0.0 to +0.0 — cleanup B7). NO place-add kernel, NO accumulator zeroing.
    DeviceBuffer<double> result_f2(total);
    DeviceBuffer<double> result_vpair(total);

    // ---- The combined result (host) ------------------------------------------
    // f2/vpair are resized, NOT zero-assigned: the single final D2H overwrites every
    // element (the disjoint tiling covers the whole tensor). block_sizes is placed
    // HOST-side in the fixed g order (int, no device math) — identical to the host
    // baseline.
    F2BlockTensor out;
    out.P = P;
    out.n_block = n_block_full;
    out.f2.resize(total);
    out.vpair.resize(total);
    out.block_sizes.assign(static_cast<std::size_t>(n_block_full), 0);

    // ---- FIXED-ORDER g = 0 .. G-1 placement into DISJOINT result slices ------
    // Single-homed in place_partials_into (the [7.1] dedup); the dst base is the local
    // result_* buffers for this host-returning entry. The final D2H follows below.
    place_partials_into(result_f2.data(), result_vpair.data(), partials, slab,
                        root_device_id, root_stream, out.block_sizes);

    // ---- ONE sync, then the SINGLE final D2H of the full result (the only D2H) ----
    // PIN the host destinations for the D2H window (M4.5 d2h-speed; RegisteredHostRegion,
    // graceful pageable degrade). PARITY-NEUTRAL: pinned vs pageable moves identical
    // bytes into the SAME disjoint placement (the D2H overwrites every element). The
    // trailing cudaStreamSynchronize keeps the pin alive across the DMA (the RAII
    // unregister fires at the if-scope exit, AFTER the sync).
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(root_stream));
    if (total > 0) {
        const std::size_t bytes = total * sizeof(double);
        RegisteredHostRegion pin_f2(out.f2.data(), bytes);   // pinned D2H (graceful degrade)
        RegisteredHostRegion pin_vpair(out.vpair.data(), bytes);
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.f2.data(), result_f2.data(),
                                          bytes, cudaMemcpyDeviceToHost, root_stream));
        STEPPE_CUDA_CHECK(cudaMemcpyAsync(out.vpair.data(), result_vpair.data(),
                                          bytes, cudaMemcpyDeviceToHost, root_stream));
        STEPPE_CUDA_CHECK(cudaStreamSynchronize(root_stream));
    }
    return out;  // restore{} fires; the DevicePartial handles are freed by the caller AFTER this (§7)
}

// DEVICE-RESIDENT assembly (M4.5 device-resident output). Shares the SAME placement
// loop as combine_f2_partials_resident above via place_partials_into ([7.1] dedup) —
// same validate, same root bind, same fixed g=0..G-1 disjoint placement (D2D for the
// root's own partial, cudaMemcpyPeerAsync for each peer's), same single drain — but it
// builds the full result directly into a DeviceF2Blocks::Impl on the root and OMITS the
// final D2H. The ONLY difference at the placement call is the dst base pointers
// (out.impl->f2/vpair raw vs the host sibling's local result_*). The assembled tensor
// is bit-identical to the host-returning combine's result (same bytes, same placement,
// raw copy preserving −0.0; §12); the ONLY difference is it stays RESIDENT in VRAM.
DeviceF2Blocks combine_f2_partials_resident_device(
    std::span<DevicePartial> partials,
    std::span<const steppe::device::DeviceShard> shards,
    int P, int n_block_full, int root_device_id) {
    // Same fail-fast precondition guard as the host-returning sibling (REUSED).
    steppe::core::validate_resident_partials(
        "steppe::device::combine_f2_partials_resident_device", partials, shards, P, n_block_full);

    // Bind to the root for the whole routine (the result + every copy land here);
    // RAII-restore the caller's device on exit (exception-safe), exactly as the
    // host-returning sibling.
    int prev_device = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&prev_device));
    struct DeviceGuard {
        int dev;
        explicit DeviceGuard(int d) noexcept : dev(d) {}
        ~DeviceGuard() { (void)STEPPE_CUDA_WARN(cudaSetDevice(dev)); }
        // Side-effecting dtor (cudaSetDevice restore) ⇒ non-copyable AND non-movable
        // (standard §2.12): a copy/move would re-fire the restore (a redundant rebind).
        // Scope-restore guard, never relocated, so all four are deleted — not move-only.
        // The explicit int ctor is REQUIRED: a user-declared (deleted) copy/move ctor
        // makes the class a non-aggregate under C++20 (P1008R1 — "no user-declared ...
        // constructors"), so the `restore{prev_device}` init can no longer rely on
        // aggregate init and must reach a real ctor. Behavior-neutral. [16.3]
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
        DeviceGuard(DeviceGuard&&) = delete;
        DeviceGuard& operator=(DeviceGuard&&) = delete;
    } restore{prev_device};
    STEPPE_CUDA_CHECK(cudaSetDevice(root_device_id));

    // The combine's OWN non-blocking stream on the root (off the legacy default stream).
    Stream root_stream_owner;
    const cudaStream_t root_stream = root_stream_owner.get();

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total = slab * static_cast<std::size_t>(n_block_full);

    // ---- The DEVICE-RESIDENT full result (NO host F2BlockTensor, NO final D2H) ----
    // ONE full-shape device RESULT on the root, allocated as the DeviceF2Blocks::Impl
    // buffers (REPLACING the host-returning sibling's local result_f2/result_vpair): the
    // peer/D2D copies write DIRECTLY into out.impl->f2/out.impl->vpair. NO memset (the
    // disjoint tiling covers every slab exactly once). block_sizes placed host-side in
    // the fixed g order (int; identical to the host baseline).
    DeviceF2Blocks out;
    out.P = P;
    out.n_block = n_block_full;
    out.device_id = root_device_id;
    out.block_sizes.assign(static_cast<std::size_t>(n_block_full), 0);
    out.impl = std::make_unique<DeviceF2Blocks::Impl>();
    out.impl->f2 = DeviceBuffer<double>(total);
    out.impl->vpair = DeviceBuffer<double>(total);
    double* result_f2 = out.impl->f2.data();
    double* result_vpair = out.impl->vpair.data();

    // ---- FIXED-ORDER g = 0 .. G-1 placement into DISJOINT result slices ------
    // Single-homed in place_partials_into (the [7.1] dedup); the dst base is the
    // DeviceF2Blocks::Impl raw pointers for this device-resident entry — the ONLY
    // difference from the host-returning sibling (which passes its local result_*).
    place_partials_into(result_f2, result_vpair, partials, slab,
                        root_device_id, root_stream, out.block_sizes);

    // ---- ONE sync, then RETURN the device-resident result (NO final D2H) ----------
    // The result STAYS resident in out.impl on the root. The single drain guarantees
    // every D2D/peer DMA has completed before the resident buffers are read downstream
    // (or .to_host()'d) and before any source partial is freed.
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(root_stream));
    return out;  // restore{} fires; the DevicePartial handles are freed by the caller AFTER this (§7)
}

}  // namespace steppe::device
