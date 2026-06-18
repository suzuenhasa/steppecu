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
// COMBINE PATH — host-staged baseline (this unit). The portable parity baseline
// (architecture.md §11.4) is the only combine path here; it is the sole path on the
// budget box and is bit-identical to the opt-in device-resident cudaMemcpyPeer
// fast-path (device/p2p_combine.{hpp,cu}, a separate workflow). When the P2P unit
// lands, the §4 gate (resources.config.prefer_p2p_combine &&
// resources.config.enable_peer_access && resources.gpus[0].caps.can_access_peer &&
// G >= 2) selects the device-resident combine here; until then every G >= 2 run
// takes the host-staged baseline, which is bit-identical to single-GPU regardless
// (the gate is parity-NEUTRAL, §12). The two-knob AND (the MAY-WE enable_peer_access
// + the WHICH-PATH prefer_p2p_combine) honors both documented config levers so the
// P2P path's cudaDeviceEnablePeerAccess is reached only with the user's permission
// (cleanup CT2 / config C-1).
#include "core/fstats/f2_blocks_multigpu.hpp"

#include <cstddef>
#include <exception>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/fstats/f2_combine.hpp"            // steppe::core::combine_f2_partials_host (the baseline)
#include "core/internal/views.hpp"               // steppe::core::MatView
#include "core/internal/host_device.hpp"         // STEPPE_ASSERT (debug-only fail-fast)
#include "core/internal/log.hpp"                 // STEPPE_LOG_WARN (the one warn sink; the tagged degrade)
#include "core/domain/block_partition_rule.hpp"  // steppe::core::block_ranges, BlockRange
#include "device/p2p_combine.hpp"                 // steppe::device::combine_f2_partials_p2p (CUDA-free decl of the P2P fast-path)
#include "device/resources.hpp"                  // steppe::device::Resources, CombinePath
#include "device/shard_plan.hpp"                 // steppe::device::plan_block_shards, DeviceShard
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

    // ---- G >= 2: BLOCK-ALIGNED shard plan ------------------------------------
    // Per-block column ranges via the SINGLE-SOURCE inverse of assign_blocks
    // (core::block_ranges; validates the partition contract ONCE). block_id
    // non-decreasing ⇒ each block's columns are contiguous, so a contiguous block
    // range maps to a contiguous SNP-column span (design §2). Then plan_block_shards
    // assigns contiguous whole-block ranges to the G devices, balanced by SNP count
    // — the single home of the block→device mapping (DRY, §8).
    const std::vector<BlockRange> ranges = core::block_ranges(
        std::span<const int>(partition.block_id.data(), static_cast<std::size_t>(M < 0 ? 0 : M)),
        M, n_block);

    std::vector<int> block_sizes(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
    for (int b = 0; b < n_block; ++b) {
        block_sizes[static_cast<std::size_t>(b)] =
            static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
    }

    const std::vector<steppe::device::DeviceShard> shards =
        steppe::device::plan_block_shards(
            std::span<const int>(block_sizes.data(), block_sizes.size()),
            std::span<const BlockRange>(ranges.data(), ranges.size()),
            G);

    // ---- Per-device partials over zero-copy sub-views (CONCURRENT fan-out) ----
    // Each device g computes its compact [P × P × (b1-b0)] partial from a column
    // SUB-VIEW of Q/V/N and a LOCAL, dense, zero-based block_id (design §2):
    //   * sub-view: columns [s0, s1) are the contiguous span data + P*s0 of length
    //     P*(s1-s0) — a pointer offset + adjusted M, ZERO COPY. The same s0/s1 for
    //     Q, V, N (they share P/M).
    //   * local block_id: the global block_id[s0 .. s1) shifted down by b0, so the
    //     backend's [P × P × n_block_local] tensor is indexed 0..n_block_local-1
    //     (block_id_local[k] = block_id[s0+k] - b0). n_block_local = b1 - b0.
    // An EMPTY shard (b0 == b1) hands the backend n_block_local == 0 / M_local == 0,
    // and the backend early-returns an empty F2BlockTensor (its degenerate guard) —
    // the combine then places nothing for that device. The per-block GEMM is the
    // UNMODIFIED compute_f2_blocks — NOT reimplemented here (design §0).
    //
    // CONCURRENCY (architecture.md §7, §11.4 — the SPMG speedup): the G devices run
    // their GEMMs CONCURRENTLY, one host thread per device. CudaBackend::compute_f2_blocks
    // is self-contained and blocking — it guard_device()s its OWN device, owns its OWN
    // stream/handle/buffers, and ends on its own cudaStreamSynchronize — so the host
    // loop that issued them one-at-a-time (each blocking on its trailing sync before
    // the next was even issued) serialized work the hardware can overlap: each device
    // has its own default stream and commands on distinct devices' default streams run
    // concurrently (CUDA Programming Guide §3.4, "Programming Systems with Multiple
    // GPUs"). Wall-clock was Σ_g time(g); fanned out it is max_g time(g).
    //
    // No shared mutable state: each worker writes ONLY its own pre-sized partials[g]
    // slot (distinct elements of a vector sized to G before any thread starts — no
    // resize, no aliasing), builds its own block_id_local, and drives its own backend
    // gpus[g] bound to its own device. cudaSetDevice sets the calling HOST THREAD's
    // current device (CUDA Runtime API: the current device is a per-host-thread
    // property), so a worker's guard_device() cannot clobber another worker's current
    // device. The combine reads partials[g] in fixed g=0..G-1 order AFTER the join
    // barrier; the per-device GEMM bits are fixed by the block-aligned shard and are
    // INDEPENDENT of which wall-clock slot ran them (§12) — so the fan-out is
    // PARITY-NEUTRAL: bit-identical to the former serial drive.
    //
    // EXCEPTION SAFETY: a std::thread/std::jthread whose entry function lets an
    // exception escape calls std::terminate ([thread.thread.constr]). Each worker
    // therefore CATCHES everything into its own std::exception_ptr slot; after the
    // join barrier the orchestrator rethrows the FIRST captured exception (lowest g),
    // so a backend / device fault still propagates to the caller as a normal throw
    // (R3: runtime_error on a malformed sub-partition, CudaError on a device fault).
    std::vector<F2BlockTensor>     partials(G);
    std::vector<std::exception_ptr> worker_errors(G);  // value-init to nullptr
    {
        // jthread joins in its destructor (RAII §2), so every worker is joined before
        // worker_errors / partials are read below — the join is the happens-before
        // barrier the fixed-order combine depends on. Reserve so the launch loop does
        // not reallocate the vector (which would move not-yet-joined threads).
        std::vector<std::jthread> workers;
        workers.reserve(G);
        for (std::size_t g = 0; g < G; ++g) {
            workers.emplace_back([&, g]() {
                try {
                    const steppe::device::DeviceShard& sh = shards[g];
                    const long s0 = sh.s0;
                    const long s1 = sh.s1;
                    const long M_local = s1 - s0;
                    const int  n_block_local = sh.b1 - sh.b0;

                    // Zero-copy column sub-views (data + P*s0). For an empty shard
                    // M_local == 0 and the offset is harmless (the backend early-
                    // returns before any deref).
                    const std::size_t col_off =
                        static_cast<std::size_t>(P) * static_cast<std::size_t>(s0 < 0 ? 0 : s0);
                    const MatView Qg{Q.data + col_off, P, M_local};
                    const MatView Vg{V.data + col_off, P, M_local};
                    const MatView Ng{N.data + col_off, P, M_local};

                    // Dense, zero-based LOCAL block_id of length M_local. Built in the
                    // worker (off the issue-critical path; owns its own slice).
                    std::vector<int> block_id_local(
                        static_cast<std::size_t>(M_local < 0 ? 0 : M_local), 0);
                    for (long k = 0; k < M_local; ++k) {
                        block_id_local[static_cast<std::size_t>(k)] =
                            partition.block_id[static_cast<std::size_t>(s0 + k)] - sh.b0;
                    }

                    partials[g] = resources.gpus[g].backend->compute_f2_blocks(
                        Qg, Vg, Ng, block_id_local.data(), n_block_local, precision);
                } catch (...) {
                    // NEVER let it escape the thread (std::terminate); surface it on join.
                    worker_errors[g] = std::current_exception();
                }
            });
        }
    }  // <-- join barrier: all workers joined here (jthread dtors)

    // Rethrow the FIRST worker failure (lowest g) so a device/backend fault propagates
    // as a normal exception to the caller, deterministically (independent of which
    // worker raced to fail first).
    for (std::size_t g = 0; g < G; ++g) {
        if (worker_errors[g]) {
            std::rethrow_exception(worker_errors[g]);
        }
    }

    // ---- Fixed-order combine, capability-gated (THE §4 fork) -----------------
    // The §4 gate selects the OPT-IN device-resident cudaMemcpyPeer combine when
    //   prefer_p2p_combine && enable_peer_access && gpus[0].caps.can_access_peer && G >= 2
    // (true on rtxbox: PRO 6000 stock-driver P2P, canAccessPeer==1 both ways), else
    // it DEGRADES to the host-staged fixed-order combine baseline. Both tiers sum the
    // SAME fixed g=0..G-1 order onto a zero-initialized full tensor and are therefore
    // BIT-IDENTICAL to each other and to the single-GPU reference (the gate is
    // parity-NEUTRAL — the transport only moves bytes; software fixes the order;
    // architecture.md §11.4, §12). NEVER an NCCL AllReduce. The chosen path is
    // recorded OUT-OF-BAND on Resources (NEVER on the numeric F2BlockTensor; cleanup
    // §(2).2), and a genuine degrade (P2P requested but peer access unavailable) emits
    // the architecture-mandated tagged WARN via the non-throwing path.
    //
    // THE TWO-KNOB GATE (cleanup CT2 / config C-1/K-2; config.hpp OVERRIDE-KNOB
    // banner). The two override-intent knobs are DISTINCT and BOTH must permit P2P:
    //   * enable_peer_access — the MAY-WE knob: "whether the backend is permitted to
    //     call cudaDeviceEnablePeerAccess at all" (config.hpp). The device-resident
    //     combine combine_f2_partials_p2p DOES call cudaDeviceEnablePeerAccess
    //     (cuda/p2p_combine.cu), so taking that path with enable_peer_access==false
    //     would directly VIOLATE the veto the user set. The gate must AND it in so a
    //     user who set enable_peer_access=false (forbid the enable) is honored and the
    //     enable is reached only WITH permission.
    //   * prefer_p2p_combine — the WHICH-PATH knob: once peer access IS permitted and
    //     available, prefer the device-resident combine over the host-staged baseline.
    // can_access_peer is the DISCOVERED probe (set once at build_resources). The gate
    // is parity-NEUTRAL either way (both transports are bit-identical, §12), so AND-ing
    // in enable_peer_access only changes WHICH transport runs, never a reported number.
    const std::span<const F2BlockTensor> partials_span(partials.data(), partials.size());
    const std::span<const steppe::device::DeviceShard> shards_span(shards.data(), shards.size());

    const bool use_p2p =
        resources.config.prefer_p2p_combine && resources.config.enable_peer_access &&
        resources.gpus[0].caps.can_access_peer;
    if (use_p2p) {
        // device_ids[g] == the physical ordinal that computed partial g (gpus[g] in
        // the fixed g=0..G-1 == DeviceConfig::devices order; gpus[0] is the root).
        // The CUDA-free P2P seam needs them to source each cudaMemcpyPeer (the root's
        // own partial skips the peer hop). Built here, host-side; gpus[0].device_id
        // is the combine root.
        std::vector<int> device_ids(G, 0);
        for (std::size_t g = 0; g < G; ++g) {
            device_ids[g] = resources.gpus[g].device_id;
        }
        resources.last_combine_path = steppe::device::CombinePath::P2pDeviceResident;
        return steppe::device::combine_f2_partials_p2p(
            partials_span, shards_span,
            std::span<const int>(device_ids.data(), device_ids.size()),
            P, n_block, resources.gpus[0].device_id);
    }

    // ---- Host-staged baseline (the portable parity baseline) -----------------
    // Taken when P2P is not preferred, peer access is FORBIDDEN by the user
    // (enable_peer_access=false), or peer access is UNAVAILABLE on the device. Emit
    // the EXACT architecture-mandated tagged degrade ONLY on a genuine degrade — the
    // user both PREFERRED P2P and PERMITTED peer access, but the device cannot
    // peer-access (architecture.md §11.4; config.hpp prefer_p2p_combine doc). A user
    // who set prefer_p2p_combine=false OR enable_peer_access=false took the baseline by
    // a DELIBERATE choice (a baseline preference or a forbidden enable, not a thwarted
    // capability) — no WARN. Both outcomes are bit-identical to single-GPU.
    if (resources.config.prefer_p2p_combine && resources.config.enable_peer_access &&
        !resources.gpus[0].caps.can_access_peer) {
        STEPPE_LOG_WARN(
            "P2P combine unavailable (no peer access) -> host-staged fixed-order combine");
    }
    resources.last_combine_path = steppe::device::CombinePath::HostStaged;
    return steppe::core::combine_f2_partials_host(partials_span, shards_span, P, n_block);
}

}  // namespace steppe::core
