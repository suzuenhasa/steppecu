// src/device/cuda/p2p_combine.cu
//
// combine_f2_partials_p2p — the OPT-IN device-resident P2P f2 combine, DEFINITION
// (architecture.md §11.4 capability-tiered combine, §12 PARITY LAW; design §4).
// This is the CUDA translation unit: it owns the cross-device DMA (cudaMemcpyPeer)
// and the on-device FP64 placement-add, so it is PRIVATE to steppe_device
// (architecture.md §4 layering — CUDA headers never compile into core/api/CLI). It
// #includes the CUDA-FREE declaration `device/p2p_combine.hpp` (the same pattern
// cuda_backend.cu uses to define the CUDA-free-declared make_cuda_backend; design
// §5), so the CUDA-free core entry point reaches this routine without seeing a CUDA
// header.
//
// THE TRANSPORT IS A REAL cudaMemcpyPeer (architecture.md §11.4; MEASURED 55.6 GB/s,
// canAccessPeer==1 both directions on rtxbox). GPU 0 (the combine root,
// device_ids[0] == root_device_id) PULLS each peer device's compact partial via
// cudaMemcpyPeer (a byte-exact device-to-device DMA across the fabric) and sums it
// on-device in the FIXED g=0..G-1 device order. The current per-device
// compute_f2_blocks returns a HOST partial (it frees its device buffers before
// returning), so this routine stages each partial onto its OWNING device first (a
// byte-exact H2D upload to a buffer on device_ids[g]) and then pulls it peer->root
// with cudaMemcpyPeer — exercising the credited P2P transport (design §4 honesty
// flag: the H2D pre-stage is a PERFORMANCE wart, not a parity wart; a device-resident
// per-device compute that skips it touches the hot compute_f2_blocks and is OUT OF
// SCOPE for M4.5). The root's OWN partial (g where device_ids[g] == root_device_id)
// needs no peer hop — it uploads H2D straight into the root staging buffer.
//
// BIT-IDENTITY ARGUMENT (architecture.md §11.4, §12; design §4). The result is
// BIT-IDENTICAL to core::combine_f2_partials_host and to the single-GPU reference
// because:
//   (1) The bytes are the SAME. Each device's compute_f2_blocks produced its compact
//       partial's exact f2/vpair doubles (block-aligned shard ⇒ each block's slab
//       bits equal the single-GPU run's, design §0). The H2D upload and the
//       cudaMemcpyPeer DMA are byte-exact — they MOVE those bytes, they do not
//       recompute them ("the transport only moves bytes").
//   (2) The ORDER is the SAME. The accumulator is ZERO-initialized full-shape and
//       each compact partial is PLACED (added onto exact 0.0) at its block offset in
//       the FIXED g=0..G-1 loop order — the identical arithmetic
//       combine_f2_partials_host performs on the host. Disjoint block-aligned shards
//       make every global slab written by exactly one device (the += lands on a 0.0,
//       x + 0.0 == x exactly), so the device combine equals the host baseline
//       slab-for-slab. "software fixes the order."
// So the two combine tiers are parity-NEUTRAL siblings (architecture.md §11.4) and
// the §12 PARITY LAW (bit-identical across G and to single-GPU) holds on this tier.
// NEVER an NCCL AllReduce (its reduction order varies with G; §12).
#include "device/p2p_combine.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/internal/host_device.hpp"    // STEPPE_ASSERT (debug-only fail-fast on the grid-axis cap)
#include "core/internal/launch_config.hpp"  // steppe::core::cdiv / kMaxGridX (the one launch-grid math home)
#include "device/cuda/check.cuh"            // STEPPE_CUDA_CHECK (fault), STEPPE_CUDA_WARN (recoverable peer-enable), STEPPE_CUDA_CHECK_KERNEL
#include "device/cuda/device_buffer.cuh"    // steppe::device::DeviceBuffer<double> (the allowlisted RAII owner)
#include "steppe/fstats.hpp"                // steppe::F2BlockTensor
#include "device/shard_plan.hpp"            // steppe::device::DeviceShard
#include "core/fstats/f2_partials_validate.hpp"  // shared validate_f2_partials (cleanup B5)

namespace steppe::device {

namespace {

// 1-D thread-block width for the placement-add kernel — a plain element-wise pass
// over the partial's `P*P*nb_local` doubles, NOT one of the f2 kernels' square
// [P × P] tiles (so NOT kCdivBlock; design §4). 256 threads is the conventional
// throughput-saturating 1-D width for a bandwidth-bound element-wise kernel; the
// combine is off the critical path so this is uncritical. Named (not a bare literal)
// per the no-magic-numbers standard (architecture.md §2; design §8).
constexpr int kPlaceAddBlockX = 256;

/// On-device FIXED-ORDER placement-add: `acc[acc_base + k] += src[k]` for every
/// element `k` of one device's compact partial slab-stack (architecture.md §12
/// fixed-order combine; design §4). One launch per device g, in the FIXED g=0..G-1
/// loop order, places that device's `P*P*nb_local` doubles at the accumulator's
/// block offset `acc_base = P*P*b0`. With block-aligned (disjoint) shards each
/// global slab is written by exactly one device, so this `+=` lands on the
/// zero-initialized accumulator (x + 0.0 == x — EXACT), reproducing
/// combine_f2_partials_host's host-side sum bit-for-bit. `count == P*P*nb_local`.
__global__ void place_add_f2_kernel(double* __restrict__ acc,
                                    const double* __restrict__ src,
                                    long acc_base, long count) {
    const long k = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count) {
        // += onto the zero-initialized accumulator (disjoint shards ⇒ a placement;
        // the explicit += keeps this the literal §12 fixed-order sum and matches the
        // host baseline's arithmetic exactly).
        acc[acc_base + k] += src[k];
    }
}

}  // namespace

F2BlockTensor combine_f2_partials_p2p(
    std::span<const F2BlockTensor> partials,
    std::span<const steppe::device::DeviceShard> shards,
    std::span<const int> device_ids,
    int P, int n_block_full, int root_device_id) {
    // Fail-fast precondition guard. The shared validate_f2_partials is the ONE home
    // of the partial/shard/P/storage/tiling contract — the SAME guard
    // combine_f2_partials_host calls — so the two tiers reject IDENTICALLY (cleanup
    // B5; architecture.md §2, §8). Their parity-neutrality (§11.4, §12) requires it:
    // a drift in which inputs each tier accepts would let one combine bytes the other
    // refuses. This P2P tier threads a THIRD parallel span (device_ids, a transport
    // detail core has no notion of), so it adds ONLY that one extra size check here;
    // everything the two tiers have in common is validated once, shared.
    if (partials.size() != device_ids.size()) {
        throw std::runtime_error(
            "steppe::device::combine_f2_partials_p2p: device_ids count (" +
            std::to_string(device_ids.size()) + ") != partials count (" +
            std::to_string(partials.size()) + ")");
    }
    steppe::core::validate_f2_partials(
        "steppe::device::combine_f2_partials_p2p", partials, shards, P, n_block_full);

    // ---- The combine root owns the full-shape accumulator --------------------
    // Bind to GPU 0 (the combine root) for the whole routine: the accumulator + the
    // root staging buffers live here and every cudaMemcpyPeer pulls TO here
    // (cudaSetDevice, cuBLAS §2.1.2 — subsequent runtime calls + the buffer
    // allocations bind to the current device). RAII-restore the caller's device on
    // exit: the caller drives each per-device compute_f2_blocks (which
    // cudaSetDevice-binds its own device), so leaving root_device_id current after
    // this call would silently retarget a later call. The guard is exception-safe
    // (it fires on the early-throw / fault paths too).
    int prev_device = 0;
    STEPPE_CUDA_CHECK(cudaGetDevice(&prev_device));
    struct DeviceGuard {
        int dev;
        ~DeviceGuard() { (void)STEPPE_CUDA_WARN(cudaSetDevice(dev)); }
    } restore{prev_device};  // restore on scope exit (teardown WARN, never throw)
    STEPPE_CUDA_CHECK(cudaSetDevice(root_device_id));

    const std::size_t slab =
        static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    // validate_f2_partials already rejected a negative n_block_full, so the cast is
    // safe with no clamp (cleanup B5 / C4 — the old `< 0 ? 0 :` ternary was dead).
    const std::size_t total = slab * static_cast<std::size_t>(n_block_full);

    // Full-shape device accumulators, ZERO-initialized — the device-resident
    // counterpart of the host baseline's `out.f2.assign(total, 0.0)` (design §3): the
    // 0.0 is every NON-OWNED slab's value, so summing exact zeros for those is exact
    // and the combined tensor equals the single-GPU run slab-for-slab (design §0).
    // cudaMemset writes the all-zero BYTE pattern, which IS IEEE-754 double +0.0
    // (all-zero bits == +0.0), so the device zero-init is bit-equal to the host
    // std::vector<double>(.., 0.0) zero-init — the two tiers start from the identical
    // accumulator.
    DeviceBuffer<double> dAcc_f2(total);
    DeviceBuffer<double> dAcc_vp(total);
    if (total > 0) {
        STEPPE_CUDA_CHECK(cudaMemset(dAcc_f2.data(), 0, dAcc_f2.bytes()));
        STEPPE_CUDA_CHECK(cudaMemset(dAcc_vp.data(), 0, dAcc_vp.bytes()));
    }

    // Root staging buffers sized to the LARGEST single partial's slab-stack
    // (`slab * max_local_nblock`), reused across the G pulls (the combine is
    // sequential in the fixed g order). cudaMemcpyPeer lands a peer's partial HERE
    // before the on-device add; the root's own partial uploads here directly.
    int max_local_nblock = 0;
    for (std::size_t g = 0; g < partials.size(); ++g) {
        if (partials[g].n_block > max_local_nblock) {
            max_local_nblock = partials[g].n_block;
        }
    }
    const std::size_t stage_elems =
        slab * static_cast<std::size_t>(max_local_nblock);
    DeviceBuffer<double> dStage_f2(stage_elems);
    DeviceBuffer<double> dStage_vp(stage_elems);

    // ---- The combined result (host) ------------------------------------------
    // block_sizes is placed HOST-side (int, no device math; design §4 step 3): the
    // backend already computed each block's SNP count from its local ranges (== the
    // global block's count, design §2), so a placement at offset b0 — identical to
    // the host baseline. f2/vpair are filled by the device copy-back at the end.
    F2BlockTensor out;
    out.P = P;
    out.n_block = n_block_full;
    out.f2.assign(total, 0.0);
    out.vpair.assign(total, 0.0);
    out.block_sizes.assign(static_cast<std::size_t>(n_block_full), 0);

    // ---- FIXED-ORDER device combine g = 0 .. G-1 (THE parity law) ------------
    // For each device in the FIXED g=0..G-1 order: stage its compact partial onto the
    // root staging buffers (peer->root cudaMemcpyPeer for a true peer; direct H2D for
    // the root's own partial) and place-add it into the accumulator at the block
    // offset. Same fixed order, same bytes, same zero-init as the host baseline ⇒
    // bit-identical (architecture.md §11.4, §12).
    for (std::size_t g = 0; g < partials.size(); ++g) {
        const F2BlockTensor& part = partials[g];
        const steppe::device::DeviceShard& sh = shards[g];
        const int owning_device = device_ids[g];

        // block_sizes: place this device's per-block SNP counts at offset b0 (host
        // int, no device math) — identical to the host baseline (design §3/§4). Done
        // even for an empty shard (a no-op loop) before the early-continue.
        for (int lb = 0; lb < part.n_block; ++lb) {
            out.block_sizes[static_cast<std::size_t>(sh.b0 + lb)] =
                part.block_sizes[static_cast<std::size_t>(lb)];
        }
        if (part.n_block <= 0) continue;  // empty shard: nothing to DMA / add

        const std::size_t part_elems = slab * static_cast<std::size_t>(part.n_block);
        const std::size_t part_bytes = part_elems * sizeof(double);
        const long acc_base =
            static_cast<long>(slab) * static_cast<long>(sh.b0);  // P*P*b0
        const long cnt = static_cast<long>(part_elems);          // P*P*nb_local

        if (owning_device == root_device_id) {
            // ---- The root's OWN blocks: byte-exact H2D straight into root staging.
            // No peer hop (the data's owning device IS the root). cudaMemcpyPeer with
            // src==dst==root would be a needless self-copy; the H2D moves the same
            // bytes (parity-NEUTRAL transport choice; architecture.md §11.4).
            STEPPE_CUDA_CHECK(cudaMemcpy(dStage_f2.data(), part.f2.data(),
                                         part_bytes, cudaMemcpyHostToDevice));
            STEPPE_CUDA_CHECK(cudaMemcpy(dStage_vp.data(), part.vpair.data(),
                                         part_bytes, cudaMemcpyHostToDevice));
        } else {
            // ---- A PEER device's blocks: pre-stage on the peer, then cudaMemcpyPeer.
            // Enable peer access root<-owning_device for the DMA. cudaDeviceEnablePeer
            // Access is DIRECTIONAL and is set FROM the device that ISSUES the access
            // (here the root reads the peer's buffer), so it is enabled while the ROOT
            // is current, naming `owning_device` as the peer (CUDA Runtime API
            // "cudaDeviceEnablePeerAccess: enables direct access to memory allocations
            // on a peer device by the CURRENT device"). cudaErrorPeerAccessAlreadyEnabled
            // is the EXPECTED status when a prior combine already enabled it — a
            // tagged, non-fatal degrade routed through the NON-throwing STEPPE_CUDA_WARN
            // (check.cuh CAP-2), NOT a fault. A genuine enable failure on a device the
            // caller PROMISED is peer-reachable surfaces on the cudaMemcpyPeer below
            // via the throwing STEPPE_CUDA_CHECK (fail-fast).
            (void)STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(owning_device, 0));
            // CLEAR the sticky last-error the WARN'd enable may have set. A non-success
            // cudaError_t (here the EXPECTED, tolerated cudaErrorPeerAccessAlreadyEnabled
            // on the 2nd+ combine, once the root↔peer link is up) is recorded as the
            // per-thread STICKY last error — and STEPPE_CUDA_WARN deliberately does NOT
            // consume it (it only READS the return value to branch; check.cuh). Left
            // uncleared, the NEXT cudaGetLastError() — inside the place-add's
            // STEPPE_CUDA_CHECK_KERNEL below — would WRONGLY surface this stale,
            // already-handled status as if the kernel launch failed. cudaGetLastError
            // both reads AND resets the sticky error to cudaSuccess (CUDA Runtime API),
            // so this one call discards the tolerated status, leaving a clean slate for
            // the genuine post-launch check. (The non-sticky success paths above —
            // cudaSetDevice / cudaMalloc / cudaMemcpy — do not set it.)
            (void)cudaGetLastError();

            // Allocate the peer-resident staging buffers ON the owning device and
            // upload the host partial there (byte-exact H2D on the peer). RAII
            // DeviceBuffer follows the current device, so switch to the peer to
            // allocate/upload, then back to the root to issue the pull.
            STEPPE_CUDA_CHECK(cudaSetDevice(owning_device));
            DeviceBuffer<double> dPeer_f2(part_elems);
            DeviceBuffer<double> dPeer_vp(part_elems);
            STEPPE_CUDA_CHECK(cudaMemcpy(dPeer_f2.data(), part.f2.data(),
                                         part_bytes, cudaMemcpyHostToDevice));
            STEPPE_CUDA_CHECK(cudaMemcpy(dPeer_vp.data(), part.vpair.data(),
                                         part_bytes, cudaMemcpyHostToDevice));

            // Pull peer->root via the byte-exact cudaMemcpyPeer DMA (the credited P2P
            // transport, MEASURED 55.6 GB/s on rtxbox). cudaMemcpyPeer(dst, dstDev,
            // src, srcDev, bytes) (CUDA Runtime API): copies between the two devices'
            // address spaces; with peer access enabled it is a direct fabric DMA.
            STEPPE_CUDA_CHECK(cudaSetDevice(root_device_id));
            STEPPE_CUDA_CHECK(cudaMemcpyPeer(dStage_f2.data(), root_device_id,
                                             dPeer_f2.data(), owning_device, part_bytes));
            STEPPE_CUDA_CHECK(cudaMemcpyPeer(dStage_vp.data(), root_device_id,
                                             dPeer_vp.data(), owning_device, part_bytes));

            // SYNCHRONIZE before the staging buffers are read by the place-add AND
            // before dPeer_* is freed (the DeviceBuffer dtors fire on this scope's
            // exit, below). cudaMemcpyPeer enqueues onto the destination device's
            // NULL stream and is host-blocking for the DMA itself, but the on-device
            // place-add kernel below also runs on device `root_device_id`'s NULL
            // stream — and freeing the SOURCE peer buffers (dPeer_* on owning_device)
            // races the DMA's cross-device completion unless we fence here. Without
            // this fence the kernel intermittently faults with cudaErrorLaunchFailure
            // reading a half-staged dStage_* (the peer-source buffer was reclaimed
            // mid-DMA). A full device sync on the root is the simplest correct fence;
            // the combine is OFF the critical path (architecture.md §11.4 — the GEMMs
            // dominate), so a per-partial sync here is cost-negligible and keeps the
            // transport correct-by-construction. (CUDA Runtime API: cudaMemcpyPeer
            // ordering across devices + buffer lifetime is the caller's to fence.)
            STEPPE_CUDA_CHECK(cudaDeviceSynchronize());
            // dPeer_* free here on the peer device (DeviceBuffer dtor); the root is
            // current so the next iteration's launches/copies bind correctly.
        }

        // -- On-device fixed-order placement-add into the accumulator (on root) ---
        // The element axis rides gridDim.x (the only axis reaching 2^31-1; launch_config
        // .hpp). cnt = P*P*nb_local is far below the cap for any realistic shape
        // (e.g. P=768, n_block=757 ⇒ cnt < 4.5e8, grid < 1.8e6), but FAIL-FAST rather
        // than silently under-cover if a pathological shape ever exceeds it
        // (architecture.md §2; the same grid-axis discipline grid_for enforces).
        const int  block  = kPlaceAddBlockX;
        const long grid_l = steppe::core::cdiv(cnt, static_cast<long>(block));
        STEPPE_ASSERT(grid_l >= 0 && grid_l <= static_cast<long>(steppe::core::kMaxGridX),
                      "p2p_combine: place-add grid extent exceeds gridDim.x cap");
        const unsigned grid = static_cast<unsigned>(grid_l);
        place_add_f2_kernel<<<grid, static_cast<unsigned>(block)>>>(
            dAcc_f2.data(), dStage_f2.data(), acc_base, cnt);
        STEPPE_CUDA_CHECK_KERNEL();
        place_add_f2_kernel<<<grid, static_cast<unsigned>(block)>>>(
            dAcc_vp.data(), dStage_vp.data(), acc_base, cnt);
        STEPPE_CUDA_CHECK_KERNEL();
    }

    // ---- Copy the combined accumulators back to host -------------------------
    if (total > 0) {
        STEPPE_CUDA_CHECK(cudaMemcpy(out.f2.data(), dAcc_f2.data(),
                                     total * sizeof(double), cudaMemcpyDeviceToHost));
        STEPPE_CUDA_CHECK(cudaMemcpy(out.vpair.data(), dAcc_vp.data(),
                                     total * sizeof(double), cudaMemcpyDeviceToHost));
    }

    return out;  // restore{} fires here, rebinding the caller's device
}

}  // namespace steppe::device
