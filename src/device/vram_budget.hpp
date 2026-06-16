// src/device/vram_budget.hpp
//
// THE single home for the M4 per-block f2 VRAM budget math (architecture.md
// §11.1/§11.2; ROADMAP §4; cleanup X-5/B5 + X-13/B26).
//
// This header is the host-pure, CUDA-FREE extraction of the chunk-budget integer
// arithmetic that used to live inline in `cuda_backend.cu::compute_f2_blocks`. It
// is pulled out so the budget logic — which is plain `std::size_t` arithmetic with
// no device call — is unit-testable on the host with NO GPU (architecture.md §13;
// the cuda_backend review F14/F15 testability finding), and so the §11.2
// `build()`-time budget check and the in-stream chunk sizing share ONE
// implementation and can never drift (architecture.md §8 single-source).
//
// It lives in `src/device/` (NOT in a CUDA TU and NOT in `core`): it is a
// device-path policy (the resident-tensor footprint + the cuBLAS workspace are
// device facts), but it is CUDA-FREE so a plain C++ test and the M4 backend both
// include it. The `src/` root is on steppe_device's PUBLIC include dir, so a test
// reaches it as "device/vram_budget.hpp". No CUDA header here.
//
// WHAT IT ACCOUNTS FOR (the B26 correctness core). The M4 device path keeps TWO
// equal-sized resident FP64 tensors co-resident for the whole bucket loop — `f2`
// AND the retained `vpair` (the S4 jackknife weight), each `P²·n_block·8` bytes
// (fstats.hpp; architecture.md §11.2). The pre-cleanup budget counted only ONE,
// under-budgeting the resident set by ~2×. This helper subtracts BOTH tensors
// (`resident_tensor_bytes`) AND the cuBLAS determinism workspace
// (`kCublasWorkspaceBytes`) from free VRAM BEFORE applying the utilization
// fraction, so the chunk slabs are sized against the VRAM that genuinely remains
// for one transient strided-batched chunk — never the gross free figure.
#ifndef STEPPE_DEVICE_VRAM_BUDGET_HPP
#define STEPPE_DEVICE_VRAM_BUDGET_HPP

#include <algorithm>
#include <cstddef>

#include "steppe/config.hpp"  // kMaxVramUtilizationFraction, kCublasWorkspaceBytes

namespace steppe::device {

// The fraction is a utilization budget: a non-positive fraction would budget zero
// (a guaranteed clamp-to-1 + likely OOM) and a fraction > 1 would over-commit free
// VRAM (the exact §11.1 "never commit all free VRAM" rule it exists to enforce), so
// pin the legal (0, 1] domain at compile time here, where the budget math consumes
// it (config review 10.1; architecture.md §11.1/§11.2).
static_assert(kMaxVramUtilizationFraction > 0.0 && kMaxVramUtilizationFraction <= 1.0,
              "kMaxVramUtilizationFraction must lie in (0, 1] — it is the fraction "
              "of free VRAM the resident working set may occupy (architecture.md §11.1).");

/// Bytes of the M4 resident f2+Vpair tensors held co-resident for the whole run.
///
/// BOTH `f2` and `vpair` are `[P × P × n_block]` FP64 (`P²·n_block·8` bytes each;
/// fstats.hpp, architecture.md §11.2), so the resident footprint of the pair is
/// `2·P²·n_block·8` — the term the §11.2 budget must reserve and the pre-cleanup
/// path under-counted by 2× (cleanup X-13/B26). All arithmetic is done in
/// `std::size_t` (cast each `int` BEFORE multiplying) so the product cannot wrap a
/// 32-bit count; negative inputs are clamped to 0 (the caller guards `P,n_block>0`,
/// but this stays well-defined regardless).
///
/// @param P        number of populations (the [P × P] slab edge).
/// @param n_block  number of jackknife blocks (the outer/batch axis length).
/// @return         `2 · P² · n_block · sizeof(double)` bytes (0 if P or n_block ≤ 0).
[[nodiscard]] inline constexpr std::size_t resident_tensor_bytes(int P, int n_block) noexcept {
    if (P <= 0 || n_block <= 0) return 0;
    const std::size_t p = static_cast<std::size_t>(P);
    const std::size_t nb = static_cast<std::size_t>(n_block);
    // f2 AND vpair: two equal [P×P×n_block] FP64 tensors (the B26 2× term).
    return 2u * p * p * nb * sizeof(double);
}

/// Per-block-in-chunk byte footprint of one strided-batched M4 chunk at width
/// `s_pad`. A chunk holds, per block: the gathered inputs Qg + Vg (`P·s_pad` each)
/// and Sg (`2·P·s_pad`) = `4·P·s_pad` doubles, plus the GEMM outputs Gg + Vpairg
/// (`P²` each) and Rg (`2·P²`) = `4·P²` doubles. ×8 bytes. (The `4`s are the
/// structural input/output stack counts, not tunable; documented in
/// cuda_backend.cu's per-block-bytes comment.) Done in `std::size_t`.
///
/// @param P      number of populations.
/// @param s_pad  the bucket's padded SNP-block width (≥ 1 by construction).
/// @return       `(4·P·s_pad + 4·P²)·sizeof(double)` bytes (≥ sizeof(double)).
[[nodiscard]] inline constexpr std::size_t per_block_chunk_bytes(int P, int s_pad) noexcept {
    const std::size_t p = static_cast<std::size_t>(P < 0 ? 0 : P);
    const std::size_t sp = static_cast<std::size_t>(s_pad < 0 ? 0 : s_pad);
    const std::size_t slab = p * p;
    return (4u * p * sp + 4u * slab) * sizeof(double);
}

/// VRAM left for ONE M4 chunk's slabs after the resident set + workspace, at the
/// configured utilization fraction (architecture.md §11.2 `budget · free`).
///
/// Subtracts BOTH resident tensors (`resident_tensor_bytes`, the B26 2× term) AND
/// the cuBLAS determinism workspace (`kCublasWorkspaceBytes`) from `free_vram`
/// BEFORE scaling by `kMaxVramUtilizationFraction` — so the budget is sound
/// regardless of whether those resident allocations are committed before or after
/// the `cudaMemGetInfo` query, closing the X-5 "doesn't subtract the workspace"
/// nuance. Saturates to 0 if free VRAM cannot even cover the resident set +
/// workspace (the caller then clamps `max_blocks` up to 1 and may OOM cleanly —
/// fail-fast, not silent corruption).
///
/// @param free_vram  free device VRAM in bytes (e.g. cudaMemGetInfo's `free`).
/// @param P          number of populations.
/// @param n_block    number of jackknife blocks.
/// @return           bytes available for one chunk's transient slabs.
[[nodiscard]] inline std::size_t chunk_budget_bytes(std::size_t free_vram, int P,
                                                    int n_block) noexcept {
    const std::size_t reserved = resident_tensor_bytes(P, n_block) + kCublasWorkspaceBytes;
    const std::size_t net = (free_vram > reserved) ? (free_vram - reserved) : 0u;
    return static_cast<std::size_t>(kMaxVramUtilizationFraction * static_cast<double>(net));
}

/// Largest number of blocks of one bucket that fit in a single strided-batched
/// chunk, given the budget and the per-block-in-chunk footprint.
///
/// THE B5/B26 budget gate, host-pure and unit-tested. The quotient
/// `chunk_budget / per_block` is computed and clamped in `std::size_t` BEFORE the
/// `int` narrowing (closing the X-5/F9 wrap where a `size_t` quotient above
/// `INT_MAX` casts negative, then clamps catastrophically to 1): the result is
/// `min(quotient, nb_total)` (never more blocks than the bucket has), floored at 1
/// (a single block must always be attempted), so the returned `int` is always in
/// `[1, nb_total]` and never overflows. Never lets a chunk's slabs exceed
/// `kMaxVramUtilizationFraction` of the post-resident-set/post-workspace VRAM.
///
/// @param free_vram  free device VRAM in bytes.
/// @param P          number of populations.
/// @param n_block    number of jackknife blocks (for the resident-tensor reserve).
/// @param s_pad      the bucket's padded block width.
/// @param nb_total   number of blocks in this bucket (the upper clamp).
/// @return           blocks per chunk, clamped to `[1, nb_total]`.
[[nodiscard]] inline int max_blocks_per_chunk(std::size_t free_vram, int P, int n_block,
                                              int s_pad, int nb_total) noexcept {
    if (nb_total <= 0) return 0;  // empty bucket — nothing to chunk.
    const std::size_t budget = chunk_budget_bytes(free_vram, P, n_block);
    const std::size_t per_block = per_block_chunk_bytes(P, s_pad);
    // per_block ≥ 4·sizeof(double) > 0 whenever P > 0 (the caller's guard); guard
    // the divide anyway so the helper is total on any input.
    const std::size_t fit = (per_block > 0u) ? (budget / per_block) : 0u;
    // Clamp the quotient against nb_total IN size_t, THEN narrow — the X-5/F9 fix.
    const std::size_t capped =
        std::min(fit, static_cast<std::size_t>(nb_total));
    // A single block must always be attempted even if the budget says zero (it
    // then OOMs cleanly mid-chunk rather than silently producing nothing).
    const std::size_t clamped = std::max<std::size_t>(capped, 1u);
    return static_cast<int>(clamped);
}

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_VRAM_BUDGET_HPP
