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
#include <cstdlib>   // std::getenv (STEPPE_FORCE_TIER / STEPPE_F2_CACHE_PATH)
#include <span>
#include <stdexcept>
#include <vector>

#include "core/fstats/f2_blocks_multigpu_core.hpp"  // plan_multigpu_shards, compute_multigpu_partials[_resident|_into] (host-pure, P2P-free)
#include "core/domain/block_partition_rule.hpp"   // steppe::core::block_ranges, BlockRange (the block_sizes single-source inverse)
#include "core/internal/views.hpp"               // steppe::core::MatView
#include "core/internal/host_device.hpp"         // STEPPE_ASSERT (debug-only fail-fast)
#include "core/internal/log.hpp"                 // STEPPE_LOG_WARN (the one warn sink; the tagged degrade)
#include "device/p2p_combine.hpp"                 // steppe::device::combine_f2_partials_resident[_device] (CUDA-free decl of the P2P fast-path)
#include "device/device_partial.hpp"             // steppe::device::DevicePartial (CUDA-free opaque resident handle)
#include "device/device_f2_blocks.hpp"           // steppe::device::DeviceF2Blocks (CUDA-free device-resident result handle)
#include "device/resources.hpp"                  // steppe::device::Resources, CombinePath
#include "device/shard_plan.hpp"                 // steppe::device::DeviceShard
#include "device/tier_select.hpp"                // steppe::device::resolve_output_tier, free_host_ram_bytes, OutputTier
#include "device/f2_blocks_out.hpp"              // steppe::device::F2BlocksOut (the adaptive tiered result)
#include "device/stream_f2_blocks.hpp"           // steppe::device::StreamTarget (the CUDA-free streamed-tier request)
#include "steppe/config.hpp"                      // steppe::Precision
#include "steppe/fstats.hpp"                      // steppe::F2BlockTensor

namespace steppe::core {

namespace {

// ============================ THE §4 COMBINE GATE =========================
// SINGLE AUTHORITATIVE HOME of the device-resident-P2P gate predicate, now a
// CUDA-free file-local helper so the four-term AND exists ONCE in code, not
// copy-pasted into each entry (architecture.md §8 single-source; cleanup X6/X8/B4
// + group-5 5.3). Every caller (the host entry, the device entry, and the genuine-
// degrade WARN) derives from THIS function, so a term add/reorder is a single edit
// that cannot drift the host/device entries apart. The four terms (see the per-term
// rationale at the gate's only call site below):
//
//     prefer_p2p_combine            // WHICH-PATH intent  (config.hpp)
//  && enable_peer_access            // MAY-WE permission   (config.hpp)
//  && gpus[0].caps.can_access_peer  // DISCOVERED probe (build_resources)
//  && G >= 2                        // STRUCTURAL (dead-true past the G==1 fast-path)
//
// "Requested" (prefer && enable && G>=2) is split out so the genuine-degrade
// detection (P2P requested+permitted but peer access unavailable) is exactly
// `requested && !use_p2p` — it cannot drift from the gate it describes. Both
// transports are bit-identical (parity-NEUTRAL, §12): this only picks WHICH runs.

/// True iff the user REQUESTED + PERMITTED the device-resident P2P combine for a
/// real multi-GPU run (the gate MINUS the discovered can_access_peer probe). Split
/// out so select_p2p_combine and the degrade WARN share the same intent terms.
[[nodiscard]] bool requested_p2p_combine(const steppe::device::Resources& resources,
                                         std::size_t G) noexcept {
    return resources.config.prefer_p2p_combine && resources.config.enable_peer_access &&
           G >= 2;
}

/// THE §4 COMBINE GATE predicate (the four-term AND). True ⇒ take the opt-in
/// device-resident cudaMemcpyPeer combine; false ⇒ the host-staged fixed-order
/// baseline. Reads gpus[0].caps.can_access_peer, so the caller must hold G >= 1
/// (guaranteed past each entry's fail-fast).
[[nodiscard]] bool select_p2p_combine(const steppe::device::Resources& resources,
                                      std::size_t G) noexcept {
    return requested_p2p_combine(resources, G) && resources.gpus[0].caps.can_access_peer;
}

}  // namespace

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
    // The four-term gate predicate is single-homed in `select_p2p_combine` (the
    // file-local helper above) so it exists ONCE in code — architecture.md §8
    // single-source; cleanup X6/X8/B4 + group-5 5.3. Every other site that names this
    // gate (resources.hpp CombinePath / last_combine_path doc, p2p_combine.hpp "the
    // gate is the caller's" doc, config.hpp enable_peer_access / prefer_p2p_combine
    // docs, src/core/CMakeLists.txt module comment) CROSS-REFERENCES it rather than
    // restating the predicate. The §4 gate selects the OPT-IN device-resident
    // cudaMemcpyPeer combine when ALL FOUR terms hold:
    //
    //     prefer_p2p_combine               // WHICH-PATH intent  (config.hpp)
    //  && enable_peer_access               // MAY-WE permission   (config.hpp)
    //  && gpus[0].caps.can_access_peer     // DISCOVERED probe (build_resources)
    //  && G >= 2                           // STRUCTURAL (dead-true here; see below)
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
    // WARN via the non-throwing path (derived from `requested_p2p_combine && !use_p2p`,
    // so it cannot drift from the gate).
    //
    // WHY THE GATE MOVED BEFORE THE FAN-OUT (M4.5 cure, doc §4 Item 1): the P2P arm now
    // needs a DIFFERENT fan-out — the DEVICE-RESIDENT compute that leaves each partial
    // on its device (compute_multigpu_partials_resident -> DevicePartial) feeding the
    // device-resident combine — while the host-staged baseline keeps the EXACT existing
    // host-partial fan-out (compute_multigpu_partials -> combine_f2_partials_host). The
    // gate depends only on config + caps + G (all known up front), so it is decided
    // here, BEFORE either fan-out, and forks into the matching resident-vs-host pair.
    //
    // THE FOUR TERMS (spelled in select_p2p_combine / requested_p2p_combine):
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
    const bool use_p2p = select_p2p_combine(resources, G);

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
        // Share the new no-final-D2H device-resident assembly, then materialize ONCE
        // (.to_host()) for this host-returning entry. Bit-identical to the prior
        // host-returning combine_f2_partials_resident (same bytes, same placement; §12).
        return steppe::device::combine_f2_partials_resident_device(
            std::span<steppe::device::DevicePartial>(partials.data(), partials.size()),
            shards_span, P, n_block, resources.gpus[0].device_id).to_host();
    }

    // ---- Host-staged DIRECT path — pinned, sharded D2H into ONE shared result ----
    // Taken when P2P is not preferred, peer access is FORBIDDEN by the user
    // (enable_peer_access=false), or peer access is UNAVAILABLE on the device. The
    // M4.5 d2h-speed cure (architecture-audit Flaw 3): pre-allocate the full-shape
    // result and fan out so each device D2Hs its compact partial DIRECTLY (pinned) into
    // its disjoint block-slice [slab*b0, slab*(b0+nb)) of this ONE result — NO per-device
    // partials buffer, NO combine_f2_partials_host alloc + copy_n. The bytes/offsets are
    // bit-identical to the deleted host-staged combine (§12): out.P/out.n_block/
    // out.block_sizes init exactly as combine_f2_partials_host, total = slab*n_block,
    // each device writes the SAME disjoint slab range. The parity test exercises this
    // arm and it stays bit-identical to single-GPU.
    F2BlockTensor out;
    out.P = P;
    out.n_block = n_block;
    const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const std::size_t total = slab * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
    out.f2.resize(total);                          // pinned + written by the workers
    out.vpair.resize(total);
    out.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);

    core::compute_multigpu_partials_into(
        resources, Q, V, N, partition, shards_span,
        out.f2.data(), out.vpair.data(), out.block_sizes.data(), precision);

    // Emit the EXACT architecture-mandated tagged degrade ONLY on a genuine degrade —
    // the user both PREFERRED P2P and PERMITTED peer access, but the device cannot
    // peer-access (architecture.md §11.4; config.hpp prefer_p2p_combine doc). A user
    // who set prefer_p2p_combine=false OR enable_peer_access=false took the baseline by
    // a DELIBERATE choice — no WARN. Both outcomes are bit-identical to single-GPU.
    // Derived from `requested && !use_p2p` (requested = prefer && enable && G>=2) so it
    // CANNOT drift from the gate: requested but !use_p2p ⇒ the only missing term is the
    // discovered can_access_peer probe (group-5 5.3).
    if (requested_p2p_combine(resources, G) && !use_p2p) {
        STEPPE_LOG_WARN(
            "P2P combine unavailable (no peer access) -> host-staged fixed-order combine");
    }
    resources.last_combine_path = steppe::device::CombinePath::HostStaged;
    return out;
}

// =============================================================================
// compute_f2_blocks_multigpu_device — the M4.5 DEVICE-RESIDENT PRIMARY entry (the
// cure). Same asserts + G fail-fast + §4 gate + shard plan as the host entry above;
// the only difference is WHERE the result lives (VRAM handle) and WHEN it materializes
// to host (only on opt-in .to_host()). PARITY-NEUTRAL (computed bits unchanged; §12).
// =============================================================================
steppe::device::DeviceF2Blocks compute_f2_blocks_multigpu_device(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision) {
    const int  P = Q.P;
    const long M = Q.M;
    const int  n_block = partition.n_block;

    // ---- Shared contract (debug fail-fast; identical to the host entry) -------
    STEPPE_ASSERT(Q.P == V.P && V.P == N.P,
                  "compute_f2_blocks_multigpu_device: Q/V/N disagree on P");
    STEPPE_ASSERT(Q.M == V.M && V.M == N.M,
                  "compute_f2_blocks_multigpu_device: Q/V/N disagree on M");
    STEPPE_ASSERT(Q.P >= 0 && Q.M >= 0,
                  "compute_f2_blocks_multigpu_device: negative P or M (uninitialized MatView)");
    STEPPE_ASSERT(partition.block_id.size() == static_cast<std::size_t>(M < 0 ? 0 : M),
                  "compute_f2_blocks_multigpu_device: block_id length != M");

    // ---- Fail-fast: at least one device (architecture.md §2) -----------------
    const std::size_t G = resources.device_count();
    if (G < 1) {
        throw std::runtime_error(
            "steppe::core::compute_f2_blocks_multigpu_device: Resources has 0 devices — "
            "the SPMG precompute requires at least one (architecture.md §9)");
    }

    // ---- G == 1: the HEADLINE WIN — the result is already on the one GPU after the
    //      GEMM; KEEP it resident and return the handle. NO D2H AT ALL. ------------
    if (G == 1) {
        return resources.gpus[0].backend->compute_f2_blocks_device(
            Q, V, N, partition.block_id.data(), n_block, precision);
    }

    // ---- THE §4 COMBINE GATE — the SAME single-homed predicate the host entry uses
    //      (select_p2p_combine, architecture.md §8 single-source). One function, no
    //      copy-paste: the host/device entries cannot disagree on transport selection.
    const bool use_p2p = select_p2p_combine(resources, G);

    // BLOCK-ALIGNED shard plan — shared by both arms (the parity floor; design §0/§2).
    const std::vector<steppe::device::DeviceShard> shards =
        core::plan_multigpu_shards(partition, M, n_block, G);
    const std::span<const steppe::device::DeviceShard> shards_span(shards.data(), shards.size());

    if (use_p2p) {
        // P2P box: per-device partials stay RESIDENT, assembled DEVICE-RESIDENT into
        // ONE root-resident DeviceF2Blocks — NO host bounce, NO final D2H.
        std::vector<steppe::device::DevicePartial> partials =
            core::compute_multigpu_partials_resident(resources, Q, V, N, partition,
                                                     shards_span, precision);
        resources.last_combine_path = steppe::device::CombinePath::P2pDeviceResident;
        return steppe::device::combine_f2_partials_resident_device(
            std::span<steppe::device::DevicePartial>(partials.data(), partials.size()),
            shards_span, P, n_block, resources.gpus[0].device_id);
    }

    // ---- NO-PEER box (the consumer 5090): documented limitation. Full single-tensor
    //      assembly across 2 GPUs without P2P needs a host bounce. The per-device
    //      compute stays RESIDENT (no premature D2H per device — the host-staged DIRECT
    //      path D2Hs each device's compact partial straight into its disjoint slice of
    //      ONE host result); we then UPLOAD that to the root device so the PRIMARY
    //      return is still a DeviceF2Blocks (the precompute->fit handoff is
    //      device-resident even on the no-peer tier; the host bounce is the cross-card
    //      assembly transport, NOT a forced output copy). [architecture.md §11.4: on
    //      the no-P2P tier a single device-resident tensor across G cards is not
    //      achievable without P2P or a host bounce.] last_combine_path is set to
    //      HostStaged inside the host wrapper.
    steppe::F2BlockTensor host =
        compute_f2_blocks_multigpu(resources, Q, V, N, partition, precision);
    return steppe::device::upload_f2_blocks_to_device(host, resources.gpus[0].device_id);
}

// =============================================================================
// compute_f2_blocks_multigpu_tiered — the M5 ADAPTIVE TIERED entry (single-GPU first).
// The result lives in the FASTEST tier it FITS in, selected by resolve_output_tier from
// the RUNTIME free-VRAM probe (resources.gpus[0].caps.free_vram_bytes) + the RUNTIME
// free-host-RAM probe (free_host_ram_bytes via sysinfo) — never hardcoded — or the
// force-tier override. PARITY-NEUTRAL (§12): the tier moves no bits.
//
// TIER 0 (Resident) IS THE EXISTING PATH UNCHANGED: on OutputTier::Resident the
// orchestrator calls compute_f2_blocks_device EXACTLY as the device entry does (no sink,
// no staging, no triple-buffer) — P=512 keeps its 3.9x and never touches streaming.
// HostRam + Disk drive the streamed seam (compute_f2_blocks_streamed) ONLY when the
// result does not fit VRAM (opt-in-by-need).
// =============================================================================
steppe::device::F2BlocksOut compute_f2_blocks_multigpu_tiered(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    const Precision& precision) {
    const int  P = Q.P;
    const long M = Q.M;
    const int  n_block = partition.n_block;

    // ---- Shared contract (debug fail-fast; identical to the other entries) -----
    STEPPE_ASSERT(Q.P == V.P && V.P == N.P,
                  "compute_f2_blocks_multigpu_tiered: Q/V/N disagree on P");
    STEPPE_ASSERT(Q.M == V.M && V.M == N.M,
                  "compute_f2_blocks_multigpu_tiered: Q/V/N disagree on M");
    STEPPE_ASSERT(Q.P >= 0 && Q.M >= 0,
                  "compute_f2_blocks_multigpu_tiered: negative P or M (uninitialized MatView)");
    STEPPE_ASSERT(partition.block_id.size() == static_cast<std::size_t>(M < 0 ? 0 : M),
                  "compute_f2_blocks_multigpu_tiered: block_id length != M");

    const std::size_t G = resources.device_count();
    if (G < 1) {
        throw std::runtime_error(
            "steppe::core::compute_f2_blocks_multigpu_tiered: Resources has 0 devices — "
            "the SPMG precompute requires at least one (architecture.md §9)");
    }

    // ---- TIER SELECT (runtime probes; never hardcoded) -----------------------
    // Free VRAM is the per-device probe captured ONCE at build_resources
    // (backend.hpp:166-167); free host RAM is sysinfo(2) read NOW. The override
    // precedence (config.force_tier wins, then STEPPE_FORCE_TIER, then automatic) is
    // resolved by the frozen helper. The tier is gated on gpus[0].caps.free_vram_bytes
    // (the ROOT device only) because this tiered path is single-GPU — it always drives
    // gpus[0] regardless of G (multi-GPU tiered sharding is the follow-on).
    const std::size_t free_vram = resources.gpus[0].caps.free_vram_bytes;
    const std::size_t free_host = steppe::device::free_host_ram_bytes();
    const steppe::device::OutputTier tier = steppe::device::resolve_output_tier(
        resources.config.force_tier, std::getenv("STEPPE_FORCE_TIER"),
        P, M, n_block, free_vram, free_host);

    steppe::device::F2BlocksOut out;
    out.P = P;
    out.n_block = (n_block < 0 ? 0 : n_block);

    // Block_sizes are needed on the result for every tier (the S4 jackknife metadata).
    // The streamed sinks fill them from the per-device compute's block_ranges; for
    // Resident the DeviceF2Blocks carries them. To keep the result self-describing in
    // ALL tiers we derive them once from the partition's block_id (the single-source
    // inverse), matching what the per-device compute records.
    {
        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(partition.block_id.data(),
                                                    static_cast<std::size_t>(M < 0 ? 0 : M)),
                               M, n_block);
        out.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
        for (int b = 0; b < n_block; ++b)
            out.block_sizes[static_cast<std::size_t>(b)] =
                static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }

    switch (tier) {
        case steppe::device::OutputTier::Resident: {
            // TIER 0 UNCHANGED — the 3.9x path, NO sink, NO streaming. EXACTLY the
            // device entry's G==1 call (f2_blocks_multigpu.cpp verbatim).
            out.tier = steppe::device::OutputTier::Resident;
            out.resident = resources.gpus[0].backend->compute_f2_blocks_device(
                Q, V, N, partition.block_id.data(), n_block, precision);
            // The handle carries its own block_sizes/P/n_block; mirror onto out for the
            // tier-agnostic surface.
            out.P = out.resident.P;
            out.n_block = out.resident.n_block < 0 ? 0 : out.resident.n_block;
            if (!out.resident.block_sizes.empty()) out.block_sizes = out.resident.block_sizes;
            break;
        }
        case steppe::device::OutputTier::HostRam: {
            out.tier = steppe::device::OutputTier::HostRam;
            steppe::device::StreamTarget target;
            target.tier = steppe::device::OutputTier::HostRam;
            target.host_dst = &out.host;
            resources.gpus[0].backend->compute_f2_blocks_streamed(
                Q, V, N, partition.block_id.data(), n_block, precision, target);
            if (out.host.n_block >= 0) out.n_block = out.host.n_block;
            out.P = out.host.P;
            if (!out.host.block_sizes.empty()) out.block_sizes = out.host.block_sizes;
            break;
        }
        case steppe::device::OutputTier::Disk: {
            out.tier = steppe::device::OutputTier::Disk;
            // Disk path precedence: config.disk_cache_path, else STEPPE_F2_CACHE_PATH
            // env, else the frozen default "./steppe_f2_blocks.cache" in the cwd.
            std::string path = resources.config.disk_cache_path;
            if (path.empty()) {
                const char* env = std::getenv("STEPPE_F2_CACHE_PATH");
                path = (env && env[0]) ? std::string(env) : std::string("./steppe_f2_blocks.cache");
            }
            steppe::device::StreamTarget target;
            target.tier = steppe::device::OutputTier::Disk;
            target.disk_path = path;
            target.disk_dst = &out.disk;
            resources.gpus[0].backend->compute_f2_blocks_streamed(
                Q, V, N, partition.block_id.data(), n_block, precision, target);
            out.P = out.disk.P;
            out.n_block = out.disk.n_block < 0 ? 0 : out.disk.n_block;
            if (!out.disk.block_sizes.empty()) out.block_sizes = out.disk.block_sizes;
            break;
        }
    }
    return out;
}

}  // namespace steppe::core
