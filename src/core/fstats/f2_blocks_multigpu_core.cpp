// src/core/fstats/f2_blocks_multigpu_core.cpp
//
// The HOST-PURE, CUDA-FREE, P2P-FREE core of the SPMG precompute orchestrator
// (f2_blocks_multigpu_core.hpp): the block-aligned shard PLAN (plan_multigpu_shards)
// and THREE per-device concurrent FAN-OUT entries that produce each device's compact
// partial, all over one shared fan_out_shards helper — compute_multigpu_partials
// (host-staged), compute_multigpu_partials_resident (device-resident), and
// compute_multigpu_partials_into (direct into a caller buffer). References ONLY
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
#include <functional>  // std::function (the per-worker seam callable type)
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

namespace {

// ============================ THE SHARED FAN-OUT =============================
// The SINGLE home of the per-device CONCURRENT fan-out the three
// compute_multigpu_partials[_resident|_into] entries used to copy-paste verbatim
// ([7.1] dedup): jthread-per-device launch + the worker body (zero-copy column
// sub-views + the dense zero-based LOCAL block_id) + the per-worker exception_ptr
// try/catch + the join-barrier deterministic first-error rethrow. The three entries
// differed ONLY in the trailing seam call + result-slot write, which is now the
// per-worker `seam` callable they each supply.
//
// PARITY-NEUTRAL (architecture.md §12): each device's GEMM bits are fixed by the
// block-aligned shard, not the wall-clock slot; the combine reads the per-g slot in
// fixed g=0..G-1 order AFTER the join barrier. The op order is byte-identical to the
// former three inline copies — only the seam call moved behind the callable.
//
// EXCEPTION SAFETY: a std::jthread whose entry function lets an exception escape calls
// std::terminate ([thread.thread.constr]). Each worker CATCHES everything into its own
// exception_ptr slot; after the join barrier we rethrow the FIRST captured exception
// (lowest g), so a backend / device fault propagates to the caller as a normal throw
// (R3) — deterministically, independent of which worker raced to fail first.
//
// The `seam` receives (g, Qg, Vg, Ng, block_id_local, n_block_local, sh): the
// per-device zero-copy column sub-views, the LOCAL dense zero-based block_id pointer,
// the local block count, and the shard (for sh.b0, the global placement offset).
// fan_out_shards owns block_id_local for the duration of the seam call (it stays alive
// until the seam returns inside the worker lambda).
using ShardSeam = std::function<void(std::size_t /*g*/,
                                     const MatView& /*Qg*/, const MatView& /*Vg*/,
                                     const MatView& /*Ng*/, const int* /*block_id_local*/,
                                     int /*n_block_local*/,
                                     const steppe::device::DeviceShard& /*sh*/)>;

void fan_out_shards(const MatView& Q, const MatView& V, const MatView& N,
                    const BlockPartition& partition,
                    std::span<const steppe::device::DeviceShard> shards,
                    const ShardSeam& seam) {
    const int P = Q.P;
    const std::size_t G = shards.size();
    std::vector<std::exception_ptr> worker_errors(G);  // value-init to nullptr
    {
        // jthread joins in its destructor (RAII §2), so every worker is joined before
        // worker_errors is read below — the join is the happens-before barrier the
        // fixed-order combine depends on. Reserve so the launch loop does not
        // reallocate the vector (which would move not-yet-joined threads).
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
                    // returns before any deref). The same s0/s1 for Q, V, N (they
                    // share P/M). The `s0 < 0 ? 0` clamp is belt-and-suspenders against
                    // a malformed negative range so the unsigned col_off cannot wrap
                    // (well-formed block-aligned shards tile [0, n_block) non-negative).
                    const std::size_t col_off =
                        static_cast<std::size_t>(P) * static_cast<std::size_t>(s0 < 0 ? 0 : s0);
                    const MatView Qg{Q.data + col_off, P, M_local};
                    const MatView Vg{V.data + col_off, P, M_local};
                    const MatView Ng{N.data + col_off, P, M_local};

                    // Dense, zero-based LOCAL block_id of length M_local. Built in the
                    // worker (off the issue-critical path; owns its own slice):
                    // block_id_local[k] = global block_id[s0+k] - sh.b0. The
                    // `M_local < 0 ? 0` clamp likewise guards a malformed negative range
                    // so the unsigned vector length cannot wrap (empty shard ⇒ M_local==0).
                    std::vector<int> block_id_local(
                        static_cast<std::size_t>(M_local < 0 ? 0 : M_local), 0);
                    for (long k = 0; k < M_local; ++k) {
                        block_id_local[static_cast<std::size_t>(k)] =
                            partition.block_id[static_cast<std::size_t>(s0 + k)] - sh.b0;
                    }

                    seam(g, Qg, Vg, Ng, block_id_local.data(), n_block_local, sh);
                } catch (...) {
                    // NEVER let it escape the thread (std::terminate); surface it on join.
                    worker_errors[g] = std::current_exception();
                }
            });
        }
    }  // <-- join barrier: all workers joined here (jthread dtors)

    // Rethrow the FIRST worker failure (lowest g) so a device/backend fault propagates
    // as a normal exception to the caller, deterministically.
    for (std::size_t g = 0; g < G; ++g) {
        if (worker_errors[g]) {
            std::rethrow_exception(worker_errors[g]);
        }
    }
}

}  // namespace

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
    const std::size_t G = shards.size();

    // ---- Per-device partials over zero-copy sub-views (CONCURRENT fan-out) ----
    // The fan-out (jthread-per-device, zero-copy sub-views, dense LOCAL block_id,
    // deterministic first-error rethrow) is the SHARED fan_out_shards above ([7.1]
    // dedup); this entry supplies only the seam: drive the UNMODIFIED HOST-RETURNING
    // compute_f2_blocks and move its compact [P × P × (b1-b0)] F2BlockTensor into the
    // pre-sized per-g slot.
    //
    // An EMPTY shard (b0 == b1) hands the backend n_block_local == 0 / M_local == 0,
    // and the backend early-returns an empty F2BlockTensor (its degenerate guard) —
    // the combine then places nothing for that device. The per-block GEMM is the
    // UNMODIFIED compute_f2_blocks — NOT reimplemented here (design §0).
    //
    // CONCURRENCY (architecture.md §7, §11.4 — the SPMG speedup): the G devices run
    // their GEMMs CONCURRENTLY, one host thread per device. CudaBackend::compute_f2_blocks
    // is self-contained and blocking — it guard_device()s its OWN device, owns its OWN
    // stream/handle/buffers, and ends on its own cudaStreamSynchronize. Each device's
    // backend owns its OWN non-blocking statistic stream (cuda_backend.cu, P2/F1), and
    // commands on distinct devices' streams run concurrently (CUDA Programming Guide
    // §3.4). The fan-out's H2D inputs are additionally pinned through the per-backend
    // registry (P4/L2) so the two devices' uploads run as concurrent DMAs.
    //
    // No shared mutable state: each worker writes ONLY its own pre-sized partials[g]
    // slot (distinct elements of a vector sized to G before any thread starts — no
    // resize, no aliasing) and drives its own backend gpus[g] bound to its own device.
    // The combine reads partials[g] in fixed g=0..G-1 order AFTER the join barrier; the
    // per-device GEMM bits are fixed by the block-aligned shard and are INDEPENDENT of
    // which wall-clock slot ran them (§12) — PARITY-NEUTRAL.
    std::vector<F2BlockTensor> partials(G);
    fan_out_shards(Q, V, N, partition, shards,
                   [&](std::size_t g, const MatView& Qg, const MatView& Vg,
                       const MatView& Ng, const int* block_id_local, int n_block_local,
                       const steppe::device::DeviceShard&) {
                       partials[g] = resources.gpus[g].backend->compute_f2_blocks(
                           Qg, Vg, Ng, block_id_local, n_block_local, precision);
                   });
    return partials;
}

std::vector<steppe::device::DevicePartial> compute_multigpu_partials_resident(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    const Precision& precision) {
    const std::size_t G = shards.size();

    // The EXACT same concurrent fan-out as compute_multigpu_partials (the shared
    // fan_out_shards above; [7.1] dedup) — same zero-copy sub-views, same dense local
    // block_id, same exception-ptr rethrow, same empty-shard handling (design §0/§2; §7
    // lifetime) — differing ONLY in the seam: each worker calls the RESIDENT seam method
    // (which leaves device g's f2/Vpair RESIDENT and returns a move-only DevicePartial
    // carrying sh.b0, the global placement offset) and move-assigns the handle into its
    // pre-sized slot. The DeviceBuffer move inside the handle is a pointer swap (no CUDA
    // call), safe from any thread; the handle (and its resident buffers) SURVIVES the
    // join — partials outlives the workers — and frees only AFTER the combine consumed it
    // (§7). On a worker throw the slot stays a default-constructed empty DevicePartial
    // (frees nothing); any successfully-allocated peers free via RAII on unwind.
    // PARITY-NEUTRAL: each device's GEMM bits are fixed by the block-aligned shard, not
    // the wall-clock slot; the combine reads partials[g] in fixed g=0..G-1 order AFTER
    // the join barrier.
    std::vector<steppe::device::DevicePartial> partials(G);
    fan_out_shards(Q, V, N, partition, shards,
                   [&](std::size_t g, const MatView& Qg, const MatView& Vg,
                       const MatView& Ng, const int* block_id_local, int n_block_local,
                       const steppe::device::DeviceShard& sh) {
                       partials[g] = resources.gpus[g].backend->compute_f2_blocks_resident(
                           Qg, Vg, Ng, block_id_local, n_block_local, sh.b0, precision);
                   });
    return partials;
}

void compute_multigpu_partials_into(
    steppe::device::Resources& resources,
    const MatView& Q, const MatView& V, const MatView& N,
    const BlockPartition& partition,
    std::span<const steppe::device::DeviceShard> shards,
    double* dst_f2, double* dst_vpair, int* block_sizes_dst,
    const Precision& precision) {
    // The EXACT same concurrent fan-out as compute_multigpu_partials (the shared
    // fan_out_shards above; [7.1] dedup) — same zero-copy sub-views, same dense local
    // block_id, same exception-ptr rethrow, same empty-shard handling (design §0/§2; §6
    // lifetime) — differing ONLY in the seam: each worker calls the host-staged-DIRECT
    // seam method compute_f2_blocks_into(...), which D2Hs its compact f2/vpair (pinned)
    // DIRECTLY into its DISJOINT slice [slab*b0, slab*(b0+nb)) of the SHARED result the
    // caller pre-allocated — NO per-device F2BlockTensor, NO combine copy. NO shared
    // mutable state across workers: worker g writes ONLY its disjoint slab range
    // [slab*b0, slab*(b0+nb)) of dst_f2/dst_vpair and [b0, b0+nb) of block_sizes_dst;
    // the block-aligned shards tile [0, n_block) disjointly (shard_plan.hpp;
    // plan_block_shards), so for g != g' the ranges never overlap and the concurrent
    // D2Hs into one host buffer are race-free. PARITY-NEUTRAL: each device's GEMM bits
    // are fixed by the block-aligned shard, not the wall-clock slot; the join barrier is
    // the happens-before before the orchestrator reads/returns the result. A worker that
    // threw left its disjoint slice partially written, but fan_out_shards rethrows the
    // first error before this function returns, so no partial result escapes.
    fan_out_shards(Q, V, N, partition, shards,
                   [&](std::size_t g, const MatView& Qg, const MatView& Vg,
                       const MatView& Ng, const int* block_id_local, int n_block_local,
                       const steppe::device::DeviceShard& sh) {
                       resources.gpus[g].backend->compute_f2_blocks_into(
                           Qg, Vg, Ng, block_id_local, n_block_local, sh.b0,
                           dst_f2, dst_vpair, block_sizes_dst, precision);
                   });
}

}  // namespace steppe::core
