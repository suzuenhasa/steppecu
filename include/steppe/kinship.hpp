// include/steppe/kinship.hpp
//
// Public, CUDA-free entry point for `steppe kinship` — KING-robust between-family kinship
// (Manichaikul et al. 2010) over a diploid genotype triple. The diploid counterpart of
// `steppe readv2` (which is pseudo-haploid only): it decodes the per-individual diploid
// dosage ONCE per SNP-tile and sweeps every pair on the GPU, accumulating the four integer
// KING counts (hetHet, IBS0, het_i, het_j) that give phi = (hetHet - 2*IBS0)/(het_i+het_j).
// The unit is an INDIVIDUAL (singleton partition, exactly like readv2), NOT a population.
//
// Gated against plink2 `--make-king-table` KINSHIP, NOT ADMIXTOOLS2 — integer counting +
// one host division, so the emulated-FP64-matmul policy does not bind (native).
#ifndef STEPPE_KINSHIP_HPP
#define STEPPE_KINSHIP_HPP

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

// KinshipResult — the per-pair KING table (plink2 .kin0-shaped, one row per emitted pair).
// All vectors are row-aligned (length == number of emitted pairs). id1 <= id2 lexically.
struct KinshipResult {
    std::vector<std::string> id1;      // Genetic ID (canonicalized id1 <= id2)
    std::vector<std::string> id2;
    std::vector<long>        nsnp;     // shared autosomal non-missing sites (considered)
    std::vector<long>        hethet;   // #{ci==1 & cj==1}
    std::vector<long>        ibs0;     // #{opposite homozygotes}
    std::vector<double>      phi;      // KING kinship (NaN if het_i+het_j==0)
    std::vector<std::string> degree;   // dup | 1st | 2nd | 3rd | unrelated | undefined

    int N = 0;                         // individuals considered
    std::size_t enumerated = 0;        // pairs enumerated (C(N,2) for all-pairs, or #pairs)
    std::size_t emitted = 0;           // rows returned (after the --min-kinship filter)
    long        autosomal_snps = 0;    // autosomal SNPs in the mask (informational)
    bool capped = false;               // refused: C(N,2) > the maxcomb cap and no --sure

    // SNP ids retained by the QC filter (kept order); empty when no filter was active.
    std::vector<std::string> kept_snp_ids;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_kinship_all_pairs — the GPU all-pairs KING driver. Resolves each selected Genetic ID
// to its own singleton index (the readv2 per-individual partition), decodes the diploid tile
// ONCE per SNP-tile, and folds every C(N,2) pair's KING counts on-device. `samples` restricts
// to a set of Genetic IDs (nullopt = all present). `min_kinship` emits only pairs with
// phi >= it (default -inf = all). `sure` lifts the C(N,2) maxcomb cap for large N.
[[nodiscard]] KinshipResult run_kinship_all_pairs(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, double min_kinship, bool sure,
    device::Resources& resources, const FilterConfig& filter = FilterConfig{});

// run_kinship_pairs — the targeted (biobank-scale) KING driver over an EXPLICIT pair list
// (each pair a {id1, id2} of Genetic IDs). Same decode-once path; the pair sweep walks the
// given list instead of the C(N,2) enumeration. `samples` optionally restricts the resolved
// individual set the pair IDs must belong to. `min_kinship` filters the emitted rows.
[[nodiscard]] KinshipResult run_kinship_pairs(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples,
    const std::vector<std::pair<std::string, std::string>>& pairs, double min_kinship,
    device::Resources& resources, const FilterConfig& filter = FilterConfig{});

}  // namespace steppe

#endif  // STEPPE_KINSHIP_HPP
