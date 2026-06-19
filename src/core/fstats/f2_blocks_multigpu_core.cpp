// src/core/fstats/f2_blocks_multigpu_core.cpp
//
// The HOST-PURE, CUDA-FREE, P2P-FREE core of the SPMG precompute orchestrator
// (f2_blocks_multigpu_core.hpp): the block-aligned shard PLAN and the per-device
// concurrent FAN-OUT that produces each device's compact partial. References ONLY
// the CUDA-free ComputeBackend seam, the CUDA-free plan_block_shards, and
// core::block_ranges — NEVER the device-RDC P2P symbol — so a GPU-free host test can
// link it without device-linking the CUDA backend (cleanup D1/T1, B9).
//
// This is a PURE refactor (the body was lifted verbatim from f2_blocks_multigpu.cpp):
// the §12 bit-identity is untouched — same plan, same fan-out, same partials in the
// fixed g=0..G-1 order; the public entry point combines them exactly as before.
#include "core/fstats/f2_blocks_multigpu_core.hpp"

#include <cstddef>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#include "core/internal/views.hpp"               // steppe::core::MatView
#include "core/domain/block_partition_rule.hpp"  // steppe::core::block_ranges, BlockRange
#include "device/resources.hpp"                  // steppe::device::Resources
#include "device/shard_plan.hpp"                 // steppe::device::plan_block_shards, DeviceShard
#include "device/device_partial.hpp"             // steppe::device::DevicePartial (CUDA-free opaque resident handle)
#include "steppe/config.hpp"                     // steppe::Precision
#include "steppe/fstats.hpp"                     // steppe::F2BlockTensor

namespace steppe::core {

std::vector<steppe::device::DeviceShard> plan_multigpu_shards(
    const BlockPartition& partition, long M, int n_block, std::size_t G) {
    // ---- G >= 2: BLOCK-ALIGNED shard plan ------------------------------------
    // Per-block column ranges via the SINGLE-SOURCE inverse of assign_blocks
    // (core::block_ranges; validates the partition contract ONCE). block_id
    // non-decreasing ⇒ each block's columns are contiguous, so a contiguous block
    // range maps to a contiguous SNP-column span (design §2). Then plan_block_shards
    // assigns contiguous whole-block ranges to the G devices, balanced by SNP count
    // — the single home of the block→device mapping (DRY, §8).
    const std::vector<BlockRange> ranges = core::block_ranges(
        std::span<const int>(partition.block_id.data(),
                             static_cast<std::size_t>(M < 0 ? 0 : M)),
        M, n_block);

    // `ranges` is the SOLE input to the planner: it carries each block's SNP count
    // (`ranges[b].size()`, a `long`) for the balance AND each shard's [s0, s1). No
    // redundant block_sizes vector (cleanup B6 / X1): building one only narrowed the
    // `long` size() to `int` and forced a parallel-array contract on the planner.
    return steppe::device::plan_block_shards(
        std::span<const BlockRange>(ranges.data(), ranges.size()), G);
}

std::vector<F2BlockTensor> compute_multigpu_partials(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision) {
    const int P = Q.P;
    const std::size_t G = shards.size();

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
    // the next was even issued) serialized work the hardware can overlap. Each device's
    // backend owns its OWN non-blocking statistic stream (cuda_backend.cu, P2/F1: a
    // `cudaStreamNonBlocking` stream that does NOT implicitly serialize against the
    // process-wide legacy default stream — CUDA Runtime API `cudaStreamCreateWithFlags`),
    // and commands on distinct devices' streams run concurrently (CUDA Programming Guide
    // §3.4, "Programming Systems with Multiple GPUs"). The fan-out's H2D inputs are
    // additionally pinned through the per-backend registry (P4/L2) so the two devices'
    // uploads run as concurrent DMAs rather than contending pageable copies. Wall-clock
    // was Σ_g time(g); fanned out it trends toward max_g time(g).
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
    std::vector<F2BlockTensor>      partials(G);
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

    return partials;
}

std::vector<steppe::device::DevicePartial> compute_multigpu_partials_resident(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision) {
    const int P = Q.P;
    const std::size_t G = shards.size();

    // The EXACT same concurrent fan-out as compute_multigpu_partials — same zero-copy
    // sub-views, same dense local block_id, same exception-ptr rethrow, same empty-shard
    // handling (design §0/§2; §7 lifetime) — differing ONLY in that each worker calls the
    // RESIDENT seam method (which leaves device g's f2/Vpair RESIDENT and returns a
    // move-only DevicePartial) and move-assigns the handle into its pre-sized slot. The
    // DeviceBuffer move inside the handle is a pointer swap (no CUDA call), safe from any
    // thread; the handle (and its resident buffers) SURVIVES the join — partials outlives
    // the workers — and frees only AFTER the combine consumed it (§7). PARITY-NEUTRAL:
    // each device's GEMM bits are fixed by the block-aligned shard, not the wall-clock
    // slot; the combine reads partials[g] in fixed g=0..G-1 order AFTER the join barrier.
    std::vector<steppe::device::DevicePartial> partials(G);
    std::vector<std::exception_ptr> worker_errors(G);  // value-init to nullptr
    {
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
                    // M_local == 0 and the offset is harmless (the backend early-returns
                    // an empty DevicePartial before any deref).
                    const std::size_t col_off =
                        static_cast<std::size_t>(P) * static_cast<std::size_t>(s0 < 0 ? 0 : s0);
                    const MatView Qg{Q.data + col_off, P, M_local};
                    const MatView Vg{V.data + col_off, P, M_local};
                    const MatView Ng{N.data + col_off, P, M_local};

                    // Dense, zero-based LOCAL block_id of length M_local.
                    std::vector<int> block_id_local(
                        static_cast<std::size_t>(M_local < 0 ? 0 : M_local), 0);
                    for (long k = 0; k < M_local; ++k) {
                        block_id_local[static_cast<std::size_t>(k)] =
                            partition.block_id[static_cast<std::size_t>(s0 + k)] - sh.b0;
                    }

                    // The RESIDENT seam method — leaves the partial resident on device g
                    // and carries sh.b0 (the global placement offset) on the handle. The
                    // returned handle MOVES into the pre-sized slot (pointer swap, no CUDA).
                    partials[g] = resources.gpus[g].backend->compute_f2_blocks_resident(
                        Qg, Vg, Ng, block_id_local.data(), n_block_local, sh.b0, precision);
                } catch (...) {
                    // NEVER let it escape the thread (std::terminate); surface it on join.
                    // partials[g] stays a default-constructed empty DevicePartial (frees
                    // nothing); any successfully-allocated peers free via RAII on unwind.
                    worker_errors[g] = std::current_exception();
                }
            });
        }
    }  // <-- join barrier: all workers joined here (jthread dtors)

    // Rethrow the FIRST worker failure (lowest g), deterministically.
    for (std::size_t g = 0; g < G; ++g) {
        if (worker_errors[g]) {
            std::rethrow_exception(worker_errors[g]);
        }
    }

    return partials;
}

}  // namespace steppe::core
