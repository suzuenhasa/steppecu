// include/steppe/dstat.hpp
//
// PUBLIC, CUDA-FREE genotype-path NORMALIZED-D entry point — qpDstat Part B. UNLIKE
// run_f4 / run_qpdstat(--f2-dir) (which read the f2 cache and report f4), run_dstat reads
// the GENOTYPE TRIPLE (.geno/.snp/.ind) directly through the SAME extract-f2 decode
// front-end (io reader + decode_af [per-SNP Q/V/N] + assign_blocks [from genpos] +
// SNP-tile streaming), then DIVERGES at S2 into the per-SNP D kernel — it NEVER touches
// the f2 cache (docs/research/dates-genotype-stat-seam.md decision (i)).
//
// THE STATISTIC (AT2 qpdstat_geno, allsnps=TRUE, f4mode=FALSE; pinned EMPIRICALLY to the
// AT2 R golden to rtol ~1e-15 on the block components): for quadruple (p1,p2,p3,p4) with
// per-SNP reference-allele frequencies a=Q[p1], b=Q[p2], c=Q[p3], d=Q[p4],
//   num = (a-b)*(c-d)
//   den = (a+b-2ab)*(c+d-2cd)
// per BLOCK b, accumulated ONLY over SNPs valid in all 4 pops (V==1 for p1..p4):
//   est_num[k,b] = mean_snp(num) = Σnum/cnt,  est_den[k,b] = mean_snp(den) = Σden/cnt
// then D = est_num/est_den jackknifed across blocks (the f4-ratio jackknife-of-the-ratio
// shape, but per-(block,quadruple) cnt weights and NO setmiss). D = est, se = sqrt(var),
// z = D/se, p = 2*pnorm_upper(|z|). The est is the NORMALIZED D (range ~±0.06), DISTINCT
// from the f2-path f4 (~10x smaller); the z's closely track the f4 z's.
//
// PARITY PINS (proven on box5090): (1) ALLELE FREQ — AT2 uses PLAIN ref/an/2, NO
// pseudo-haploid adjustment; run_dstat FORCES diploid ploidy (cfg::PloidyMode::Diploid),
// NOT the extract-f2 Auto per-sample detection (which flips the sign of near-zero D).
// (2) BLOCKS — assign_blocks == AT2 get_block_lengths (byte-identical). (3) SNP MASK —
// allsnps=TRUE per-(block,quadruple) finiteness: NO maxmiss, NO MAF, NO drop-mono;
// autosomes_only ON (AT2 auto_only default).
//
// CUDA-FREE BY CONTRACT (architecture.md §4): standard-C++ only; the per-SNP D reduction
// runs on the GPU through the ComputeBackend seam (the kernel is a .cu PRIVATE to
// steppe_device). device::Resources is forward-declared CUDA-free; the .cpp includes the
// real header. The app/binding reach the GPU ONLY through this seam (like run_f4ratio).
#ifndef STEPPE_DSTAT_HPP
#define STEPPE_DSTAT_HPP

#include <array>
#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"  // steppe::Precision
#include "steppe/error.hpp"   // steppe::Status

namespace steppe {

namespace device {
struct Resources;  // CUDA-free fwd-decl (real decl: src/device/resources.hpp)
}  // namespace device

/// One genotype-path normalized-D result table, one parallel-array slot per input quadruple
/// (in INPUT order). pX[k] is the P-axis index of pop X of quadruple k (echoed for the
/// emitter/binding to label the rows). Mirrors F4Result (columns p1..p4,est,se,z,p) so the
/// emit_f4_result emitter + the binding marshalling are REUSED verbatim — the D est/se/z/p
/// ARE the AT2 D sign/Z/p convention. A degenerate quadruple (no survivor blocks) is a
/// per-row NaN sentinel, never an exception (architecture.md §10; record-and-continue).
struct DstatResult {
    std::vector<int>    p1, p2, p3, p4;  ///< the P-axis indices of each quadruple (len N).
    std::vector<double> est;             ///< the NORMALIZED D = est_num/est_den (jackknife $est).
    std::vector<double> se;              ///< sqrt of the AT2 num/den block-jackknife variance.
    std::vector<double> z;               ///< est / se.
    std::vector<double> p;               ///< 2*pnorm_upper(|z|) (two-sided normal; AT2 D convention).

    /// PER-CALL outcome (Ok for a populated result; the per-row NaN sentinel rides on a
    /// degenerate quadruple). NEVER an exception for a domain outcome (architecture.md §10).
    Status status = Status::Ok;

    /// Which arithmetic produced this (the D reduction is native FP64 / long-double host
    /// accumulation by the §12 cancellation carve-out, so this is always Fp64).
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

/// Genotype-path normalized D over the genotype triple (.geno/.snp/.ind) — qpDstat Part B.
/// `geno`/`snp`/`ind` are the EIGENSTRAT/TGENO triple paths. `pop_union` is the set of pop
/// names referenced by the quadruples (the AT2 indvec — read_ind(Explicit{pop_union}) decodes
/// ONLY these populations, NOT the whole 27594-ind prefix; the P axis is that partition,
/// SORTED ASC by label). `quadruples` is a span of (p1,p2,p3,p4) indices into THAT sorted
/// partition — the caller resolves the quadruple names against the same Explicit{pop_union}
/// order (the app builds a PopResolver over read_ind(Explicit{pop_union})). `blgsize_morgans`
/// is the jackknife block size in MORGANS (AT2 blgsize default 0.05). The result carries one
/// row per quadruple in input order. Routes the per-SNP D reduction through
/// resources.gpus[0].backend (GPU-first, device-resident). Ploidy is FORCED diploid (the AT2
/// plain ref/an/2 pin); allsnps=TRUE (no maxmiss/MAF/drop-mono); autosomes_only ON.
[[nodiscard]] DstatResult run_dstat(const std::string& geno,
                                    const std::string& snp,
                                    const std::string& ind,
                                    std::span<const std::string> pop_union,
                                    std::span<const std::array<int, 4>> quadruples,
                                    double blgsize_morgans,
                                    device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_DSTAT_HPP
