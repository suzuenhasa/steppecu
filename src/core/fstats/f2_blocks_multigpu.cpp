// src/core/fstats/f2_blocks_multigpu.cpp
//
// compute_f2_blocks_multigpu — the SPMG precompute entry point (architecture.md
// §5 S2, §11.4 SPMG, §9 Resources; design §5). Host-pure, CUDA-FREE, in
// steppe::core: it drives each per-device backend ONLY through the CUDA-free
// ComputeBackend seam (Resources::gpus[g].backend->compute_f2_blocks) and combines
// the partials with the CUDA-free host-staged fixed-order combine. It issues no
// GEMM, allocates no device memory, and includes no CUDA header (architecture.md
// §2, §4) — the whole multi-GPU algorithm sits ABOVE the per-device-instance seam,
// exactly as backend.hpp:193-202 mandates, REUSING the unmodified per-device
// compute_f2_blocks verbatim (design §0: it is NOT reimplemented here).
//
// THE PARITY ARGUMENT (design §0, restated): block-aligned column sharding assigns
// each whole block to ONE device, so that device computes the block from exactly
// its own contiguous SNP columns — identical feeder bits, identical s_pad bucket,
// identical independent strided-batched slab ⇒ identical slab bits to the single-GPU
// run. The host-staged combine places each device's compact partial at its block
// offset and sums in fixed g=0..G-1 order onto a zero-initialized full tensor, so
// the combined tensor equals the single-GPU tensor slab-for-slab, BIT-IDENTICALLY
// (architecture.md §12 PARITY LAW).
//
// COMBINE PATH — host-staged baseline + the opt-in device-resident cudaMemcpyPeer
// fast-path (device/p2p_combine.{hpp,cu}). The host-staged path is the portable
// parity baseline (architecture.md §11.4): it is the sole path on a no-peer budget
// box and is bit-identical to the P2P fast-path. The §4 gate that picks between them
// is defined ONCE — see "THE §4 COMBINE GATE" comment at the `use_p2p` computation
// below (the single authoritative home of the four-term gate predicate, §8). Either
// path is bit-identical to single-GPU (the gate is parity-NEUTRAL, §12).
#include "core/fstats/f2_blocks_multigpu.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/fstats/f2_blocks_multigpu_core.hpp"  // plan_multigpu_shards, compute_multigpu_partials[_resident] (host-pure, P2P-free)
#include "core/fstats/f2_combine.hpp"            // steppe::core::combine_f2_partials_host (the baseline)
#include "core/internal/views.hpp"               // steppe::core::MatView
#include "core/internal/host_device.hpp"         // STEPPE_ASSERT (debug-only fail-fast)
#include "core/internal/log.hpp"                 // STEPPE_LOG_WARN (the one warn sink; the tagged degrade)
#include "device/p2p_combine.hpp"                 // steppe::device::combine_f2_partials_resident (CUDA-free decl of the P2P fast-path)
#include "device/device_partial.hpp"             // steppe::device::DevicePartial (CUDA-free opaque resident handle)
#include "device/resources.hpp"                  // steppe::device::Resources, CombinePath
#include "device/shard_plan.hpp"                 // steppe::device::DeviceShard
#include "steppe/config.hpp"                      // steppe::Precision
#include "steppe/fstats.hpp"                      // steppe::F2BlockTensor

namespace steppe::core {

F2BlockTensor compute_f2_blocks_multigpu(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision) {
    const int  P = Q.P;
    const long M = Q.M;
    const int  n_block = partition.n_block;

    // ---- Shared contract (debug fail-fast; same guards as f2_from_blocks) ----
    // Q/V/N agree on P/M and are non-negative; the partition describes exactly the
    // M columns (block_id length == M, dense/non-decreasing in [0, n_block)). These
    // mirror f2_from_blocks.cpp's validate_qvn / validate_partition; the per-device
    // compute_f2_blocks the orchestrator drives also validates each sub-view's
    // partition via core::block_ranges (the single-source inverse), so a malformed
    // global partition is caught here AND defensively at the seam.
    STEPPE_ASSERT(Q.P == V.P && V.P == N.P,
                  "compute_f2_blocks_multigpu: Q/V/N disagree on P");
    STEPPE_ASSERT(Q.M == V.M && V.M == N.M,
                  "compute_f2_blocks_multigpu: Q/V/N disagree on M");
    STEPPE_ASSERT(Q.P >= 0 && Q.M >= 0,
                  "compute_f2_blocks_multigpu: negative P or M (uninitialized MatView)");
    STEPPE_ASSERT(partition.block_id.size() == static_cast<std::size_t>(M < 0 ? 0 : M),
                  "compute_f2_blocks_multigpu: block_id length != M");

    // ---- Fail-fast: at least one device (architecture.md §2) -----------------
    const std::size_t G = resources.device_count();
    if (G < 1) {
        throw std::runtime_error(
            "steppe::core::compute_f2_blocks_multigpu: Resources has 0 devices — "
            "the SPMG precompute requires at least one (architecture.md §9)");
    }

    // ---- G == 1: the EXACT single-GPU path (zero behavior change) ------------
    // No shard, no combine: drive the one backend over the FULL Q/V/N + partition
    // and return it unchanged — bit-for-bit the existing single-GPU result. This
    // makes single-GPU invariance STRUCTURAL: G==1 adds no code to the value path
    // (design §5). gpus[0] is the lone device.
    if (G == 1) {
        return resources.gpus[0].backend->compute_f2_blocks(
            Q, V, N, partition.block_id.data(), n_block, precision);
    }

    // ---- THE §4 COMBINE GATE — decided BEFORE the fan-out (it depends ONLY on
    //      config + caps + G, all known up front; doc CRITICAL ORCHESTRATION POINT) --
    //
    // ============================ THE §4 COMBINE GATE =========================
    // SINGLE AUTHORITATIVE HOME of the device-resident-P2P gate predicate
    // (architecture.md §8 single-source; cleanup X6/X8/B4). Every other site that
    // names this gate (resources.hpp CombinePath / last_combine_path doc,
    // p2p_combine.hpp "the gate is the caller's" doc, config.hpp enable_peer_access /
    // prefer_p2p_combine docs, src/core/CMakeLists.txt module comment) CROSS-REFERENCES
    // this comment rather than restating the predicate — so the predicate has ONE
    // place to drift from. The §4 gate selects the OPT-IN device-resident
    // cudaMemcpyPeer combine when ALL FOUR terms hold:
    //
    //     use_p2p == prefer_p2p_combine     // WHICH-PATH intent  (config.hpp)
    //             && enable_peer_access     // MAY-WE permission   (config.hpp)
    //             && gpus[0].caps.can_access_peer  // DISCOVERED probe (build_resources)
    //             && G >= 2                 // STRUCTURAL (dead-true here; see below)
    //
    // (all four true on rtxbox: PRO 6000 stock-driver P2P, canAccessPeer==1 both
    // ways), else it DEGRADES to the host-staged fixed-order combine baseline. Both
    // tiers place the SAME bytes at the SAME disjoint offsets in the SAME fixed
    // g=0..G-1 order, and are therefore BIT-IDENTICAL to each other and to the
    // single-GPU reference (the gate is parity-NEUTRAL — the transport only moves
    // bytes; software fixes the order; architecture.md §11.4, §12). NEVER an NCCL
    // AllReduce. The chosen path is recorded OUT-OF-BAND on Resources (NEVER on the
    // numeric F2BlockTensor; cleanup §(2).2), and a genuine degrade (P2P requested +
    // permitted but peer access unavailable) emits the architecture-mandated tagged
    // WARN via the non-throwing path.
    //
    // WHY THE GATE MOVED BEFORE THE FAN-OUT (M4.5 cure, doc §4 Item 1): the P2P arm now
    // needs a DIFFERENT fan-out — the DEVICE-RESIDENT compute that leaves each partial
    // on its device (compute_multigpu_partials_resident -> DevicePartial) feeding the
    // device-resident combine — while the host-staged baseline keeps the EXACT existing
    // host-partial fan-out (compute_multigpu_partials -> combine_f2_partials_host). The
    // gate depends only on config + caps + G (all known up front), so it is decided
    // here, BEFORE either fan-out, and forks into the matching resident-vs-host pair.
    //
    // THE FOUR TERMS:
    //   * prefer_p2p_combine — the WHICH-PATH knob: once peer access IS permitted and
    //     available, prefer the device-resident combine over the host-staged baseline.
    //   * enable_peer_access — the MAY-WE knob: "whether the backend is permitted to
    //     call cudaDeviceEnablePeerAccess at all" (config.hpp). The device-resident
    //     combine combine_f2_partials_resident DOES call cudaDeviceEnablePeerAccess
    //     (cuda/p2p_combine.cu), so taking that path with enable_peer_access==false
    //     would directly VIOLATE the veto the user set. The gate ANDs it in so a user
    //     who set enable_peer_access=false (forbid the enable) is honored and the
    //     enable is reached only WITH permission (cleanup CT2 / config C-1/K-2;
    //     config.hpp OVERRIDE-KNOB banner). The two override-intent knobs are DISTINCT
    //     and BOTH must permit P2P.
    //   * can_access_peer — the DISCOVERED capability probe (set ONCE at
    //     build_resources, never re-probed here).
    //   * G >= 2 — STRUCTURAL, and DEAD-TRUE at this point: the G==1 single-GPU
    //     fast-path returned at the top of this function (the `if (G == 1)` early
    //     return), so control only reaches this gate with G >= 2. The term is spelled
    //     in the gate so the CODE MATCHES the predicate documented across the seam
    //     (cleanup X6: it was documented as 4-term in five files while shipping as
    //     3-term — a latent hazard if the gate were ever lifted into a reusable
    //     select_combine_path(resources) that could be reached at G==1). It changes NO
    //     reached path (parity-NEUTRAL: G is always >= 2 here, so the AND is identity).
    //
    // The gate is parity-NEUTRAL either way (both transports are bit-identical, §12),
    // so the four-term AND only changes WHICH transport runs, never a reported number.
    // =========================================================================
    const bool use_p2p =
        resources.config.prefer_p2p_combine && resources.config.enable_peer_access &&
        resources.gpus[0].caps.can_access_peer && G >= 2;

    // BLOCK-ALIGNED shard plan — shared by both arms (the parity floor, design §0/§2):
    // block_ranges (the single-source partition inverse) → plan_block_shards (the
    // single home of the block→device mapping, §8). Computed once before the fork.
    const std::vector<steppe::device::DeviceShard> shards =
        core::plan_multigpu_shards(partition, M, n_block, G);
    const std::span<const steppe::device::DeviceShard> shards_span(shards.data(), shards.size());

    if (use_p2p) {
        // ---- RESIDENT fan-out -> device-resident combine (the M4.5 cure) ------
        // Each device computes its partial and LEAVES it RESIDENT (no D2H, no free);
        // the combine pulls each peer's resident partial straight into one full device
        // result via cudaMemcpyPeerAsync (root's own via D2D), then ONE final D2H. The
        // handles free HERE, AFTER the combine consumed them (§7).
        std::vector<steppe::device::DevicePartial> partials =
            core::compute_multigpu_partials_resident(resources, Q, V, N, partition,
                                                     shards_span, precision);
        resources.last_combine_path = steppe::device::CombinePath::P2pDeviceResident;
        return steppe::device::combine_f2_partials_resident(
            std::span<steppe::device::DevicePartial>(partials.data(), partials.size()),
            shards_span, P, n_block, resources.gpus[0].device_id);
    }

    // ---- Host-staged baseline — KEEP EXACTLY AS-IS (byte-for-byte) ------------
    // Taken when P2P is not preferred, peer access is FORBIDDEN by the user
    // (enable_peer_access=false), or peer access is UNAVAILABLE on the device. The
    // host-partial fan-out (compute_multigpu_partials, the UNMODIFIED per-device
    // compute_f2_blocks) → combine_f2_partials_host, exactly as before — the parity
    // test exercises this arm and it must stay bit-identical to single-GPU.
    const std::vector<F2BlockTensor> partials = core::compute_multigpu_partials(
        resources, Q, V, N, partition, shards_span, precision);
    const std::span<const F2BlockTensor> partials_span(partials.data(), partials.size());

    // Emit the EXACT architecture-mandated tagged degrade ONLY on a genuine degrade —
    // the user both PREFERRED P2P and PERMITTED peer access, but the device cannot
    // peer-access (architecture.md §11.4; config.hpp prefer_p2p_combine doc). A user
    // who set prefer_p2p_combine=false OR enable_peer_access=false took the baseline by
    // a DELIBERATE choice — no WARN. Both outcomes are bit-identical to single-GPU.
    if (resources.config.prefer_p2p_combine && resources.config.enable_peer_access &&
        !resources.gpus[0].caps.can_access_peer && G >= 2) {
        STEPPE_LOG_WARN(
            "P2P combine unavailable (no peer access) -> host-staged fixed-order combine");
    }
    resources.last_combine_path = steppe::device::CombinePath::HostStaged;
    return steppe::core::combine_f2_partials_host(partials_span, shards_span, P, n_block);
}

}  // namespace steppe::core
