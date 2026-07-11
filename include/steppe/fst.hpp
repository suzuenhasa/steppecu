// include/steppe/fst.hpp
//
// Public, CUDA-free entry point for the standalone per-SNP Weir & Cockerham 1984 FST
// (`steppe fst`). Reads a .geno/.snp/.ind triple through the shared genotype decode
// front-end, then runs a per-site WC variance-component kernel over the device-resident
// genotype tile for one population pair — an engine-independent standalone stat (no
// f2 cache, no Li-Stephens). Gated against plink2 `--fst method=wc`, NOT ADMIXTOOLS2, so
// the emulated-FP64-matmul parity policy does not bind: FST uses native FP64.
//
// v1 scope: pairwise (2-pop) per-SNP WC FST. Multi-pop / all-pairs is a documented
// follow-up.
#ifndef STEPPE_FST_HPP
#define STEPPE_FST_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

// FstResult — the per-SNP FST table plus the genome-wide summary.
struct FstResult {
    // Per-SNP (kept order), length M:
    std::vector<double> num;                 // WC numerator a (== plink2 FST_NUMER)
    std::vector<double> den;                 // a + b + c       (== plink2 FST_DENOM)
    std::vector<double> fst;                 // a/(a+b+c)       (== plink2 WC_FST; NaN if invalid)
    std::vector<std::uint8_t> valid;         // 1 = usable site, 0 = invalid (NaN fst)
    std::vector<int> chrom;
    std::vector<double> physpos;
    std::vector<std::string> snp_id;
    std::vector<char> a1;                    // .snp ref allele
    std::vector<char> a2;                    // .snp alt allele

    std::string popA;
    std::string popB;

    // Genome-wide summary over the autosomal valid sites:
    double fst_ratio = 0.0;                  // sum_num / sum_den (== plink2 .fst.summary WC_FST)
    double fst_mean = 0.0;                   // unweighted mean of per-SNP fst (secondary)
    long n_valid = 0;

    // SNP ids retained by the QC filter (kept order); empty when no filter was active.
    std::vector<std::string> kept_snp_ids;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_fst — the pairwise per-SNP WC FST driver. popA/popB are .ind population labels. `filter`
// subsets the SNP axis before the WC accumulation; an inactive (default) filter is a no-op.
[[nodiscard]] FstResult run_fst(const std::string& geno,
                                const std::string& snp,
                                const std::string& ind,
                                const std::string& popA,
                                const std::string& popB,
                                device::Resources& resources,
                                const FilterConfig& filter = FilterConfig{});

// FstMatrixResult — the all-pairs (P x P) genome-wide WC FST matrix. Every cell (i,j) is
// the ratio-of-averages fst_ratio = Σnum/Σden the single-pair path reports for that pair
// over the SAME autosomal valid sites (per-SNP num/den are bit-exact vs the single-pair
// kernel — the shared wc_finalize on integer-exact {n,ac,het}; the genome-wide Σ agrees to
// FP64 round-off, not last-ULP-identical, because the reductions differ in order — see
// docs/planning/fst-all-pairs-scope.md §7). Symmetric, zero diagonal; a pair whose every
// shared site is invalid (den == 0) gets the NaN sentinel, mirroring run_fst.
struct FstMatrixResult {
    std::vector<std::string> pops;    // row/col order (length P), tile pop order
    std::vector<double> fst;          // P*P row-major, symmetric, diag 0.0 (NaN if den==0)
    std::vector<double> num;          // P*P row-major, Σ WC numerator a  (diag 0)
    std::vector<double> den;          // P*P row-major, Σ WC denominator   (diag 0)
    std::vector<long>   n_valid;      // P*P row-major, per-pair valid-site count (diag 0)

    std::size_t enumerated = 0;       // C(P,2) pairs enumerated
    bool capped = false;              // refused: C(P,2) > the maxcomb cap and no --sure

    // SNP ids retained by the QC filter (kept order); empty when no filter was active.
    std::vector<std::string> kept_snp_ids;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// FstWindowedResult — the per-window Weir & Cockerham 1984 FST selection scan over one pop
// pair (`steppe fst --windowed SIZE:STEP`). Windows are bp intervals of `size` sliding by
// `step`, reset per chromosome, matching scikit-allel's windowed_weir_cockerham_fst window
// layout exactly (1-based inclusive [start, end], end clipped to the chromosome's last SNP).
// Each window's Fst is the RATIO OF AVERAGES Σnum/Σden over the sites it covers (num = WC a,
// den = a+b+c; per-site invalid → 0 contribution, mirroring allel's nansum), NaN when the
// window is empty (n_snp == 0) or every covered site is monomorphic (Σden == 0). n_snp counts
// ALL positions in the window, not just the valid ones.
struct FstWindowedResult {
    std::vector<int>    chrom;   // per window
    std::vector<long>   start;   // window start bp (1-based inclusive)
    std::vector<long>   end;     // window end bp   (1-based inclusive, clipped to chrom last)
    std::vector<long>   n_snp;   // positions in [start, end] (ALL, not just valid)
    std::vector<double> num;     // Σ WC numerator a over the window
    std::vector<double> den;     // Σ WC denominator a+b+c over the window
    std::vector<double> fst;     // num/den (NaN if n_snp==0 or den==0)

    std::string popA;
    std::string popB;

    std::vector<std::string> kept_snp_ids;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// FstPbsResult — the per-window population-branch statistic (Yi et al. 2010) for three pops
// A, B, C (`steppe fst --pbs A,B,C --windowed SIZE:STEP`). Per window it carries the three
// pairwise window WC Fst (F_AB, F_AC, F_BC — each the same ratio-of-averages the windowed path
// reports) and PBS_A/B/C = the branch lengths T = -ln(1 - Fst) (Fst clamped to [0, 1) so T is
// finite) combined as PBS_A = (T_AB + T_AC - T_BC)/2 (and symmetric). A window whose pairwise
// Fst is NaN (empty / monomorphic) yields NaN for that branch statistic.
struct FstPbsResult {
    std::vector<int>    chrom;
    std::vector<long>   start;
    std::vector<long>   end;
    std::vector<long>   n_snp;
    std::vector<double> fst_ab, fst_ac, fst_bc;   // per-window pairwise WC Fst
    std::vector<double> pbs_a, pbs_b, pbs_c;      // per-window PBS (raw Yi 2010)

    std::string popA;
    std::string popB;
    std::string popC;

    std::vector<std::string> kept_snp_ids;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_fst_windowed — the per-window WC FST scan driver for one pop pair. Reuses the genotype
// front-end + apply_snp_filter, builds the per-chromosome bp windows (allel-exact), and folds
// the per-site WC num/den into windows on the GPU (fst_wc_windowed). win_size / win_step are bp
// (win_step defaults to win_size at the call site for non-overlapping windows).
[[nodiscard]] FstWindowedResult run_fst_windowed(const std::string& geno,
                                                 const std::string& snp,
                                                 const std::string& ind,
                                                 const std::string& popA,
                                                 const std::string& popB,
                                                 long win_size,
                                                 long win_step,
                                                 device::Resources& resources,
                                                 const FilterConfig& filter = FilterConfig{});

// run_fst_pbs — the per-window PBS scan driver for three pops A, B, C. Decodes the sufficient
// statistic once, folds all three pairwise per-site WC num/den into the SAME bp windows on the
// GPU, then applies the host PBS transform. win_size / win_step are bp.
[[nodiscard]] FstPbsResult run_fst_pbs(const std::string& geno,
                                       const std::string& snp,
                                       const std::string& ind,
                                       const std::string& popA,
                                       const std::string& popB,
                                       const std::string& popC,
                                       long win_size,
                                       long win_step,
                                       device::Resources& resources,
                                       const FilterConfig& filter = FilterConfig{});

// run_fst_all_pairs — the GPU all-pairs WC FST matrix driver. Decodes the panel ONCE into a
// per-(pop, SNP) sufficient-statistic tensor {n, ac, het}, streamed by SNP-tile, then
// combines all C(P,2) pairs on-device (sweep_unrank k=2 -> wc_finalize -> per-pair Σ). When
// `pops` is non-empty it selects exactly those populations (order = the emitted matrix
// order); otherwise ALL populations with at least `min_n` individuals are used. `sure` lifts
// the C(P,2) maxcomb cap for very large P (mirrors the f-stat sweep --sure).
[[nodiscard]] FstMatrixResult run_fst_all_pairs(const std::string& geno,
                                                const std::string& snp,
                                                const std::string& ind,
                                                const std::vector<std::string>& pops,
                                                int min_n,
                                                bool sure,
                                                device::Resources& resources,
                                                const FilterConfig& filter = FilterConfig{});

}  // namespace steppe

#endif  // STEPPE_FST_HPP
