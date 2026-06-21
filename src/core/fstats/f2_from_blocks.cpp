// src/core/fstats/f2_from_blocks.cpp
//
// Host orchestration of the f2 assembly (architecture.md §5 S2 "assembled by
// core/fstats/f2_from_blocks.cpp"; ROADMAP §2, M0). This is the `core`-owned
// entry point that drives the f2 computation THROUGH the `ComputeBackend` seam —
// it never issues a GEMM, allocates device memory, or includes a CUDA header
// itself (architecture.md §2, §4: `core` is pure host C++20 and reaches the GPU
// only via the CUDA-free ComputeBackend interface). The same orchestration runs
// unchanged against `CpuBackend` (the reference oracle) and `CudaBackend` (the
// GPU 3-GEMM reformulation), which is exactly the dependency-injection property
// that makes the pipeline GPU-free-testable (architecture.md §8).
//
// SCOPE: both f2 entry points now live here. `compute_f2_block` (M0) drives a
// single SNP block: the Q/V/N contract (views.hpp) goes in; the bias-corrected
// f2 matrix + the retained pairwise-valid count (Vpair, the S4 jackknife weight)
// come out via F2Result. `compute_f2_blocks` (M4) drives the FULL per-SNP Q/V/N
// + a SNP→block partition (the shared assign_blocks rule) into the per-block
// [P × P × n_block] f2_blocks tensor, batched over the block axis. This is the
// host-side composition root: it OWNS the block/precision policy and the
// fail-fast contract, while the backend owns the implementation.
//
// FAIL-FAST (architecture.md §2; cleanup B11): the orchestration is the ONE host
// point that sees all three views Q/V/N together with the full BlockPartition
// (including `block_id.size()`, which the `(const int*, int)` seam erases before
// the backend sees it). So the documented Q/V/N + partition preconditions are
// enforced HERE, once, via the debug-only STEPPE_ASSERT facility
// (core/internal/host_device.hpp) — compiled out under NDEBUG so the release hot
// path pays nothing, but a malformed contract aborts with file/line under a
// debugger / compute-sanitizer instead of degrading to a silent out-of-bounds
// read/write deep inside a backend (or a null/short `block_id.data()` deref the
// backend's `block_ranges` size-check cannot catch, since it trusts the pointer).
//
// LAYERING: includes only the CUDA-free seam (device/backend.hpp), the shared
// host-pure views (core/internal/views.hpp), the host-pure block rule + public
// F2BlockTensor (named directly here — IWYU), the debug-assert home, and the
// public config — no CUDA.
#include "core/fstats/f2_from_blocks.hpp"

#include "device/backend.hpp"                    // steppe::ComputeBackend, steppe::F2Result
#include "core/internal/views.hpp"               // steppe::core::MatView (Q/V/N contract)
#include "core/internal/host_device.hpp"         // STEPPE_ASSERT (debug-only fail-fast)
#include "core/domain/block_partition_rule.hpp"  // steppe::core::BlockPartition (used directly)
#include "steppe/fstats.hpp"                      // steppe::F2BlockTensor (returned directly)
#include "steppe/config.hpp"                      // steppe::Precision

#include <cstddef>  // std::size_t

namespace steppe::core {

namespace {

// Enforce the SHARED Q/V/N precondition the backend documents but assumes
// (backend.hpp compute_f2 preconditions: "Q, V, N share the same P and M"): the
// three views must agree on P and M, and both extents must be non-negative (a
// negative P/M is an uninitialized/garbage view that would cast to a colossal
// size_t in the backend allocation). Debug-only — under NDEBUG STEPPE_ASSERT is
// a no-op, so the hot path is unchanged and the parameters are then unreferenced
// (hence [[maybe_unused]], so warnings-as-errors holds on a Release build); in
// debug it aborts with file/line rather than reading past a short view's storage
// (views.hpp::element does no bounds check). One home for both the M0 and M4
// entry points so they cannot diverge (§8 DRY). Cleanup B11 / F-2 / F-3.
void validate_qvn([[maybe_unused]] const MatView& Q, [[maybe_unused]] const MatView& V,
                  [[maybe_unused]] const MatView& N) {
    STEPPE_ASSERT(Q.P == V.P && V.P == N.P,
                  "compute_f2: Q/V/N disagree on P (population count)");
    STEPPE_ASSERT(Q.M == V.M && V.M == N.M,
                  "compute_f2: Q/V/N disagree on M (SNP count)");
    STEPPE_ASSERT(Q.P >= 0 && Q.M >= 0,
                  "compute_f2: negative P or M (uninitialized MatView)");
}

// Debug-only O(M) scan: every block_id is in [0, n_block) and the sequence is
// non-decreasing — the dense, contiguous-run invariant assign_blocks guarantees
// and both backends' block_ranges relies on. Lives behind #ifndef NDEBUG so the
// release build neither compiles nor calls it (it is referenced only from the
// debug STEPPE_ASSERT below). block_ranges throws on these too, but only AFTER
// the data has crossed the CUDA-free seam; catching it here keeps the fault
// attributable to the orchestration's own input (§2 fail-fast).
#ifndef NDEBUG
[[nodiscard]] bool block_ids_dense_nondecreasing(const BlockPartition& partition, long M) {
    int prev = -1;  // last id seen; ids must be non-decreasing.
    for (long s = 0; s < M; ++s) {
        const int id = partition.block_id[static_cast<std::size_t>(s)];
        if (id < 0 || id >= partition.n_block || id < prev) return false;
        prev = id;
    }
    return true;
}
#endif

// Enforce the BlockPartition contract before it crosses the CUDA-free seam as a
// bare `(const int*, int)` pair that erases the length and the dense/ordering
// invariants (cleanup B11 / F-1). `M` is the SNP count the backend trusts (Q.M);
// the partition must describe exactly those columns. Without this, an
// inconsistent partition (n_block disagreeing with the contents of block_id, a
// short or null block_id with n_block>0) drives an out-of-bounds host-vector
// write / device read in BOTH backends — the backend's `block_ranges` validates
// `0<=id<n_block` + non-decreasing, but it CANNOT see `block_id.size()` (it
// builds a span over the raw pointer it is handed), so the length/null contract
// is the orchestration's to enforce, here, where the owning vector is still in
// scope. Debug-only (incl. the O(M) non-decreasing scan); [[maybe_unused]] keeps
// the release build (where the body collapses to no-ops) warning-clean. §2.
void validate_partition([[maybe_unused]] const BlockPartition& partition,
                        [[maybe_unused]] long M) {
    // block_id is parallel to the M SNP columns: exactly M entries. This also
    // pins the null-data() sub-case — an empty vector with n_block>0 fails the
    // size check before any deref (an empty vector's data() may legally be null).
    STEPPE_ASSERT(partition.block_id.size() == static_cast<std::size_t>(M < 0 ? 0 : M),
                  "compute_f2_blocks: block_id length != M (partition does not "
                  "describe exactly the SNP columns)");
    // n_block must be a sane distinct-block count: > 0 when there are SNPs, never
    // more blocks than SNPs (each block holds >= 1 SNP), and the ids dense /
    // non-decreasing over [0, n_block).
    STEPPE_ASSERT(M <= 0 || partition.n_block > 0,
                  "compute_f2_blocks: n_block <= 0 with M > 0 columns");
    STEPPE_ASSERT(M <= 0 || static_cast<long>(partition.n_block) <= M,
                  "compute_f2_blocks: n_block > M (more blocks than SNPs)");
    STEPPE_ASSERT(M <= 0 || block_ids_dense_nondecreasing(partition, M),
                  "compute_f2_blocks: block_id has an out-of-range or "
                  "non-decreasing entry (malformed partition)");
}

}  // namespace

F2Result compute_f2_block(ComputeBackend& backend, const MatView& Q, const MatView& V,
                          const MatView& N, const Precision& precision) {
    // Pure dispatch through the injected backend, guarded by the shared Q/V/N
    // precondition (B11). The orchestration owns the policy (which block, which
    // precision) and the fail-fast contract; the backend owns HOW the f2 is
    // computed (scalar oracle vs 3-GEMM GPU). `core` issues no device call
    // directly (architecture.md §2, §5).
    validate_qvn(Q, V, N);
    return backend.compute_f2(Q, V, N, precision);
}

F2BlockTensor compute_f2_blocks(ComputeBackend& backend, const MatView& Q, const MatView& V,
                                const MatView& N, const BlockPartition& partition,
                                const Precision& precision) {
    // Dispatch the M4 per-block tensor through the injected backend, guarded by
    // the shared Q/V/N precondition AND the partition contract (B11). `core` owns
    // the block policy (the BlockPartition from the shared assign_blocks rule),
    // the precision policy, and the fail-fast guard; the backend owns the
    // implementation (the size-grouped strided-batched GPU path vs the CPU
    // per-block oracle). The block_id[] vector and n_block are the partition's two
    // fields; passing the raw pointer keeps the ComputeBackend seam CUDA-free and
    // free of any std-container ABI dependency — which is exactly why the length /
    // null / ordering contract is validated HERE (the seam erases it).
    validate_qvn(Q, V, N);
    validate_partition(partition, Q.M);
    return backend.compute_f2_blocks(Q, V, N, partition.block_id.data(),
                                     partition.n_block, precision);
}

}  // namespace steppe::core
