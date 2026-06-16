// src/device/backend.hpp
//
// ComputeBackend — the dependency-injection seam between `core` orchestration
// and the device layer (architecture.md §4, §8; ROADMAP §2, M0).
//
// THIS HEADER IS CUDA-FREE BY CONTRACT. It is the only door `core` uses to reach
// the GPU: `core` is pure host C++20 and never includes a CUDA header or calls
// cuBLAS/cuSOLVER directly (architecture.md §2, §4). CUDA is PRIVATE to
// steppe_device, so this interface compiles into `core` and the CLI without
// dragging in the device toolkit. Two implementations satisfy it — `CudaBackend`
// (the 3-GEMM reformulation on the GPU) and `CpuBackend` (the scalar reference
// oracle) — and the compute layer is written once against the interface, never
// branching on GPU-vs-CPU (architecture.md §8). The CPU backend is both the DRY
// reference and the correctness anchor the GPU is continuously diffed against
// (architecture.md §13; ROADMAP §5).
//
// M0 scope: the minimal-but-real method is `compute_f2` — compute the f2 matrix
// from the Q/V/N contract at a given Precision, returning f2 [P × P] and the
// pairwise-valid-SNP count Vpair [P × P]. Vpair is RETAINED, not discarded: it
// is the weighted-block-jackknife weight at S4 (architecture.md §5 S2 caveat
// (a)). Later milestones add decode / gemm / jackknife / svd methods here.
#ifndef STEPPE_DEVICE_BACKEND_HPP
#define STEPPE_DEVICE_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "steppe/config.hpp"        // steppe::Precision
#include "steppe/fstats.hpp"        // steppe::F2BlockTensor (the M4 deliverable)
#include "core/internal/views.hpp"  // steppe::core::MatView (Q/V/N contract)

namespace steppe {

/// Output of an f2 computation: the symmetric f2 matrix and the pairwise-valid
/// SNP counts, both column-major [P × P] (element (i,j) at `i + P·j`). Plain
/// host vectors so the type crosses the CUDA-free seam (architecture.md §4);
/// the CUDA backend copies its device results into these before returning.
struct F2Result {
    /// f2 matrix, column-major [P × P]. Bias-corrected, AT2-unbiased estimator
    /// f2(i,j) = mean over jointly-valid SNPs of (p_i − p_j)² − hc_i − hc_j.
    ///
    /// DIAGONAL CONVENTION (pinned; cleanup X-2/B4): f2(i,i) carries the FULL
    /// (i,i) computation, NOT a forced 0. With p_i == p_i the per-SNP summand is
    /// (p_i − p_i)² − hc_i − hc_i = −2·hc_i, so f2(i,i) = −2·mean within-pop het
    /// correction (generally nonzero, and a within-pop het quantity, NOT a
    /// between-pop f2). BOTH backends agree on the diagonal by construction: the
    /// GPU assemble_f2_kernel writes every (i,j) with no i==j guard, and the CPU
    /// oracle loops j = i. The diagonal is never consumed downstream (f3/f4 read
    /// off-diagonal f2 only) but is kept consistent across backends so the §13
    /// oracle≡GPU diff at the F2Result seam compares the FULL matrix, diagonal
    /// included, and an M7 F2Result round-trip cannot reintroduce a backend split.
    std::vector<double> f2;

    /// Pairwise-valid SNP count, column-major [P × P]: Vpair(i,j) = number of
    /// SNPs valid in BOTH i and j. RETAINED as the S4 jackknife weight
    /// (architecture.md §5 S2 caveat (a)); the per-pair divide and S4 weighting
    /// must compose to AT2's f2_blocks definition, not double-normalize. The
    /// DIAGONAL Vpair(i,i) is i's own valid-SNP count (the i==j case of "valid in
    /// both"); like f2(i,i) it is filled, not 0, and agrees across backends
    /// (cleanup X-2/B4).
    std::vector<double> vpair;

    /// Number of populations P (the leading dimension of both matrices).
    int P = 0;
};

/// A CUDA-FREE, NON-OWNING view of one packed genotype tile + its population
/// partition — the input to `decode_af` (architecture.md §5 S0/S1; ROADMAP M1).
/// This is the plain-data seam between the `io` leaf (which produces an
/// io::GenotypeTile of the same shape) and the device/CPU decode backend: `io`
/// does NOT depend on `device`, and `backend.hpp` does NOT depend on the `io`
/// leaf — `app`/the test bridge them by filling this view from a GenotypeTile.
/// All pointers are borrowed (the caller owns the storage for the call's
/// duration); no CUDA type appears.
///
/// TGENO individual-major packing: `packed` holds `n_individuals` records of
/// `bytes_per_record` bytes each, gathered POPULATION-CONTIGUOUS; SNP `s` of
/// gathered individual `g` is the 2-bit code at byte `g*bytes_per_record + s/4`,
/// position `s%4` (MSB-first). Population `p` owns gathered individuals
/// `[pop_offsets[p], pop_offsets[p+1])`. `n_pop == pop_offsets length − 1`.
struct DecodeTileView {
    const std::uint8_t* packed = nullptr;   ///< packed bytes, pop-contiguous records
    std::size_t bytes_per_record = 0;       ///< stride between individual records
    std::size_t n_snp = 0;                  ///< SNPs in the tile (the column axis M)
    std::size_t n_individuals = 0;          ///< gathered individuals across all pops
    const std::size_t* pop_offsets = nullptr;  ///< P+1 segment boundaries (sample axis)
    int n_pop = 0;                          ///< number of populations P (row axis)

    /// Per-sample PLOIDY factor for N: 2 diploid (the AADR case — every sample is
    /// diploid, so every N is even), 1 pseudo-haploid (ancient DNA). A METADATA
    /// parameter, NEVER auto-detected from genotypes (ROADMAP Q/V/N contract).
    /// Default 2 (diploid), documented.
    int ploidy = 2;
};

/// Output of a genotype-decode: the Q/V/N contract as plain column-major [P × M]
/// host arrays (element (pop i, snp s) at `i + P·s` — the views.hpp layout that
/// drops straight into MatView/compute_f2). Plain host vectors so the type
/// crosses the CUDA-free seam (architecture.md §4); the CUDA backend copies its
/// device results into these before returning.
struct DecodeResult {
    /// Reference-allele frequency in [0,1], 0 where invalid. Column-major [P × M].
    std::vector<double> q;
    /// Validity mask (1.0 valid / 0.0 missing). Column-major [P × M].
    std::vector<double> v;
    /// Non-missing haploid count (ploidy × non-missing individuals). [P × M].
    std::vector<double> n;
    /// Number of populations P (the leading dimension of Q/V/N).
    int P = 0;
    /// Number of SNPs M (the column count of Q/V/N).
    long M = 0;
};

/// Abstract compute backend. One interface, two implementations (CUDA, CPU
/// reference). All device operations route through here; `core` never issues a
/// GEMM/SVD/Cholesky itself (architecture.md §2, §8). Move-only ownership of
/// concrete backends is by `std::unique_ptr<ComputeBackend>` in `Resources`
/// (architecture.md §9).
class ComputeBackend {
public:
    ComputeBackend() = default;
    ComputeBackend(const ComputeBackend&) = delete;
    ComputeBackend& operator=(const ComputeBackend&) = delete;
    ComputeBackend(ComputeBackend&&) = delete;
    ComputeBackend& operator=(ComputeBackend&&) = delete;
    virtual ~ComputeBackend() = default;

    /// Compute the bias-corrected f2 matrix and pairwise-valid counts from the
    /// Q/V/N contract (column-major [P × M] views, views.hpp) at the requested
    /// precision.
    ///
    /// @param Q  reference-allele frequencies in [0,1], zero-filled where
    ///           invalid (the zero is what makes the masked GEMM correct).
    /// @param V  validity mask (1.0 valid / 0.0 missing).
    /// @param N  non-missing haploid count (2 × diploids, or 1 × pseudo-haploids
    ///           for ancient DNA). Enters only the het correction.
    /// @param precision  governs the matmul-heavy f2 GEMMs ONLY (default
    ///           EmulatedFp64{40} ⇒ ≈ native, MEASURED 7–17× faster on real
    ///           AADR; ROADMAP §0). The small numerator/divide stays native
    ///           FP64 regardless (architecture.md §12). The CPU backend computes
    ///           the native-FP64 reference and ignores the matmul mode.
    ///
    /// Preconditions: Q, V, N share the same P and M and refer to the same SNP
    /// block. Returns f2 [P × P] + Vpair [P × P] (column-major). Both backends
    /// must agree at the f2/Vpair seam within the tight tolerance tier
    /// (architecture.md §12, §13) — the formula is shared via
    /// core/internal/f2_estimator.hpp so they cannot diverge.
    [[nodiscard]] virtual F2Result compute_f2(const core::MatView& Q,
                                              const core::MatView& V,
                                              const core::MatView& N,
                                              const Precision& precision) = 0;

    /// Compute the PER-BLOCK f2 tensor `f2_blocks [P × P × n_block]` + the retained
    /// per-block `Vpair` from the FULL per-SNP Q/V/N contract and a SNP→block
    /// assignment — the M4 deliverable (architecture.md §5 S2, §11.1; ROADMAP M4).
    ///
    /// The inputs Q/V/N are the SAME column-major [P × M] contract as `compute_f2`,
    /// but spanning ALL M SNPs (not one block). `block_id` is an M-vector from the
    /// shared `assign_blocks` (core/domain/block_partition_rule.hpp): `block_id[s]`
    /// is the dense jackknife block of SNP s, in `0 .. n_block-1`, non-decreasing in
    /// file order (so a block's SNPs are CONTIGUOUS — the property the batched
    /// reorder relies on). The GPU backend runs the spike-chosen SIZE-GROUPED
    /// strided-batched design over the blocks; the CPU backend is the per-block
    /// long-double oracle. `precision` governs only the matmul-heavy GEMMs
    /// (architecture.md §12); the numerator/divide stays native FP64.
    ///
    /// @param Q         reference-allele frequencies in [0,1], 0 where invalid,
    ///                  column-major [P × M] over ALL SNPs.
    /// @param V         validity mask (1.0 valid / 0.0 missing), [P × M].
    /// @param N         non-missing haploid count, [P × M] (enters het correction).
    /// @param block_id  per-SNP dense block id (length M), from assign_blocks;
    ///                  non-decreasing ⇒ each block's columns are contiguous.
    /// @param n_block   number of distinct blocks (== max(block_id)+1).
    /// @param precision governs the f2 GEMMs only (default EmulatedFp64{40}).
    /// @return  F2BlockTensor with f2 + Vpair [P × P × n_block] (block-major outer,
    ///          column-major within a block: `i + P·j + P·P·b`) + block_sizes.
    [[nodiscard]] virtual F2BlockTensor compute_f2_blocks(const core::MatView& Q,
                                                          const core::MatView& V,
                                                          const core::MatView& N,
                                                          const int* block_id,
                                                          int n_block,
                                                          const Precision& precision) = 0;

    /// Decode a packed genotype tile into the Q/V/N contract (architecture.md §5
    /// S0 Format decode + S1 Allele-freq reduction; ROADMAP M1). Unpacks the
    /// 2-bit codes (raw-value mapping: 0/1/2 ref-allele copies, 3 missing),
    /// reduces over the individuals within each population segment into integer
    /// AC (ref-allele count) / AN (non-missing individual count), then a single
    /// FP64 divide yields Q = AC/(ploidy·AN), N = ploidy·AN, V = (AN>0) — the EXACT
    /// oracle math, shared with the CPU reference via core/internal/decode_af.hpp
    /// so the two cannot diverge (architecture.md §13).
    ///
    /// @param tile  the packed tile + population partition + ploidy (CUDA-free
    ///              view; the `io` leaf fills it from an io::GenotypeTile).
    /// @return Q/V/N as column-major [P × M] host arrays (i + P·s), dropping
    ///         straight into MatView/compute_f2. Both backends must agree exactly
    ///         (N, V integer-valued ⇒ zero diff; Q via integer-accumulate AC/AN +
    ///         single FP64 divide ⇒ exact is the goal/gate).
    [[nodiscard]] virtual DecodeResult decode_af(const DecodeTileView& tile) = 0;
};

}  // namespace steppe

#endif  // STEPPE_DEVICE_BACKEND_HPP
