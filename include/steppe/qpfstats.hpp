// include/steppe/qpfstats.hpp
//
// PUBLIC, CUDA-FREE genotype-path JOINT f-stat SMOOTHER entry point — qpfstats.
// UNLIKE run_f4 / run_qpdstat(--f2-dir) (which read the f2 cache and report f4/D),
// run_qpfstats reads the GENOTYPE TRIPLE (.geno/.snp/.ind) directly through the SAME
// extract-f2 / qpDstat-B decode front-end (io reader + decode_af [per-SNP Q/V/N] +
// assign_blocks [from genpos] + SNP-tile streaming), DRIVES the qpDstat-B genotype-f4
// numerator engine (ComputeBackend::dstat_block_reduce, the dstat_kernel.cu allsnps=TRUE
// per-(item,block) finiteness reduction) over the FULL f2/f3/f4 population-comb set AT2
// qpfstats regresses, then runs the SMOOTHING REGRESSION onto the n(n-1)/2 outgroup-f
// basis and emits a *smoothed* per-block f2 TENSOR (the F2BlockTensor / f2-dir format)
// that downstream qpAdm / f4 / qpGraph consume — NOT an est/se/z/p table.
//
// THE ALGORITHM (AT2 admixtools::qpfstats, R-ridge path; pinned to the AT2 R golden):
//   1. popcomb = f2 ∪ f3 ∪ f4 over the n sorted pops (for n=9: 36 + 252 + 378 = 666 combos):
//        f2: (p1,p2,p1,p2) for p1<p2                  (C(n,2) = 36)
//        f3: (p1,p2,p1,p4) the 3-rotation expansion   (C(n,3)*3 = 252)
//        f4: (p1,p2,p3,p4) the 3-rotation expansion   (C(n,4)*3 = 378)
//   2. numer[popcomb,block] = MEAN_snp (a-b)(c-d) over the allsnps=TRUE mask (V==1 for all
//      4 pops of the comb) = the qpDstat-B numsum/cnt. f4mode=TRUE → numerator-only (the
//      D denominator densum is IGNORED). This is the genotype-f4 engine, batched over the
//      666-comb axis AND the n_block axis ON THE GPU (dstat_block_reduce; device-resident).
//   3. x[popcomb,pair] = the construct_fstat_matrix design (±1/2 f4-identity coefficients;
//      *2 for a pure-f2 row, /2 all). Pair index = row-major upper-triangle (i<j) over the
//      n(n-1)/2 = C(n,2) off-diagonal pairs (the f-stat vector-space basis).
//   4. ymat = t(numer) [npopcomb × nblock]; y = matrix_jackknife_est(numer,cnt) [npopcomb]
//      (the global per-comb jackknife estimate). A_shared = x'x + ridge*I (ridge=1e-5);
//      per block b: b[:,b] = solve(A_shared, x'·ymat[:,b]); a block with NaN comb-rows
//      downdates A_b = A_shared - x[nan]'x[nan] then solves; an ALL-NaN block → b=0.
//      bglob = solve(A_shared (or A_y downdated), x'·y).
//   5. scatter b → f2blocks[n,n,nblock] (off-diagonal, symmetric); recenter
//      f2blocks2 = f2blocks - f2(f2blocks)$est + bglob (the AT2 recentering). The diagonal
//      stays 0. THIS is the smoothed F2BlockTensor.
//
// PARITY PINS (inherited from qpDstat-B, proven on box5090): (1) ALLELE FREQ — AT2 plain
// ref/an/2, FORCED diploid (NOT the extract-f2 Auto per-sample detection). (2) BLOCKS —
// assign_blocks == AT2 get_block_lengths (byte-identical). (3) SNP MASK — allsnps=TRUE
// per-(comb,block) finiteness; NO maxmiss/MAF/drop-mono; autosomes_only ON.
//
// CUDA-FREE BY CONTRACT (architecture.md §4): standard-C++ only; the per-SNP numerator
// reduction + the smoothing solve run on the GPU through the ComputeBackend seam (the
// kernels are .cu PRIVATE to steppe_device). device::Resources is forward-declared
// CUDA-free; the .cpp includes the real header. The app/binding reach the GPU ONLY through
// this seam (like run_dstat). The output is plain host FP64 storage (F2BlockTensor).
#ifndef STEPPE_QPFSTATS_HPP
#define STEPPE_QPFSTATS_HPP

#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"   // steppe::Precision
#include "steppe/error.hpp"    // steppe::Status
#include "steppe/fstats.hpp"   // steppe::F2BlockTensor (the smoothed-f2 output)

namespace steppe {

namespace device {
struct Resources;  // CUDA-free fwd-decl (real decl: src/device/resources.hpp)
}  // namespace device

/// One qpfstats result: the SMOOTHED per-block f2 tensor [P × P × n_block] (the AT2
/// qpfstats() output, R-ridge path), plus the P-axis labels (sorted ASC, the AT2 dimnames
/// order). The tensor is symmetric per block with a ZERO diagonal (the AT2 convention:
/// f2blocks is built from the off-diagonal pair basis only). Drops straight into
/// write_f2_dir so qpAdm / f4 consume the smoothed f2 like any extract-f2 cache.
struct QpfstatsResult {
    /// The smoothed per-block f2 tensor (F2BlockTensor layout: f2[i + P·j + P·P·b]). The
    /// `vpair` field is filled with the per-block kept-SNP counts replicated across pairs
    /// (so write_f2_dir's missing-block detection sees nonzero where a block contributed),
    /// and block_sizes carries the per-block SNP counts. P = n_pops, n_block = #blocks.
    F2BlockTensor f2;

    /// The P population labels in P-axis index order (sorted ASC = the AT2 dimnames order).
    std::vector<std::string> pop_labels;

    /// PER-CALL outcome. Status::Ok for a populated result; a structural domain outcome
    /// (e.g. the SPD factor is non-invertible — should not happen with ridge>0) is a value,
    /// never an exception for a domain outcome (architecture.md §10). An io fault throws.
    Status status = Status::Ok;

    /// Which arithmetic produced the smoothing solve (native FP64 by default; the matmul
    /// sub-steps run EmulatedFp64{40} by the LANDED fit precision policy, the Cholesky/solve
    /// native — but the engaged tag is recorded here for provenance/meta.json).
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

/// qpfstats over the genotype triple (.geno/.snp/.ind) — the genotype-path JOINT f2
/// smoother. `geno`/`snp`/`ind` are the EIGENSTRAT/TGENO triple paths. `pops` is the set of
/// population names to smooth over (AT2 `pops`); run_qpfstats SORTS them ASC (the AT2
/// dimnames order) and reads ONLY these pops (read_ind(Explicit{pops}), NOT the whole
/// prefix). `blgsize_morgans` is the jackknife block size in MORGANS (AT2 blgsize default
/// 0.05). `precision` governs ONLY the matmul sub-steps of the solve (x'x SYRK, x'·ymat
/// GEMM); the Cholesky factor + triangular solve default native FP64 (the §12 carve-out).
/// Routes the per-SNP numerator reduction + the smoothing solve through
/// resources.gpus[0].backend (GPU-first, device-resident). Ploidy is FORCED diploid (the
/// AT2 plain ref/an/2 pin); allsnps=TRUE; autosomes_only ON. include_f2/f3/f4 are all TRUE
/// (the AT2 default; the FULL basis). An io fault PROPAGATES as an exception.
[[nodiscard]] QpfstatsResult run_qpfstats(const std::string& geno,
                                          const std::string& snp,
                                          const std::string& ind,
                                          std::span<const std::string> pops,
                                          double blgsize_morgans,
                                          const Precision& precision,
                                          device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPFSTATS_HPP
