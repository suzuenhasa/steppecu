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

// KinshipCutoffResult — the result of `steppe kinship --king-cutoff <phi>` (the plink2
// --king-cutoff analogue): the greedy relatedness prune's RETAINED / REMOVED Genetic ID sets,
// plus the above-cutoff EDGE list (the sparse relatedness graph) so a plink2-compatible .kin0 can
// be written for cross-checking. Retained + removed partition all N considered individuals. The
// prune reproduces plink2's KinshipPruneDestructive exactly (remove the partner of the lowest-
// index degree-one vertex if one exists, else the lowest-index maximum-degree vertex), so the
// retained set matches `plink2 --king-cutoff` bit-for-bit given the same edge set and sample order.
struct KinshipCutoffResult {
    std::vector<std::string> retained;   // Genetic IDs kept (.king.cutoff.in), singleton-index order
    std::vector<std::string> removed;    // Genetic IDs pruned (.king.cutoff.out), removal order

    // The above-cutoff relatedness graph (phi > cutoff*(1+2^-44), matching plink2's nudged edge
    // rule), one entry per edge — for writing the sparse .kin0 relatedness table.
    std::vector<std::string> edge_id1;   // Genetic ID i (singleton index i < j)
    std::vector<std::string> edge_id2;
    std::vector<long>        edge_nsnp;
    std::vector<long>        edge_hethet;
    std::vector<long>        edge_ibs0;
    std::vector<double>      edge_phi;

    int N = 0;                           // individuals considered
    std::size_t enumerated = 0;          // C(N,2) pairs swept
    std::size_t n_edges = 0;             // above-cutoff pairs (relatedness graph edges)
    long        autosomal_snps = 0;      // autosomal SNPs in the mask (informational)
    double      cutoff = 0.0;            // the phi cutoff used

    std::vector<std::string> kept_snp_ids;  // QC-retained SNP ids (empty if no filter)

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_kinship_streamed — the biobank-scale STREAMED all-pairs KING driver (the `--min-kinship`
// related-only path). Same decode/singleton partition as run_kinship_all_pairs, but the pair
// sweep runs pair-block-by-block on the GPU and keeps ONLY pairs with phi >= `min_kinship` via
// on-device compaction, so the 5*C(N,2) accumulator (and its ~14k cap) is never allocated. The
// emitted rows are bit-identical to run_kinship_all_pairs filtered to phi >= min_kinship.
[[nodiscard]] KinshipResult run_kinship_streamed(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, double min_kinship,
    device::Resources& resources, const FilterConfig& filter = FilterConfig{});

// run_kinship_cutoff — `steppe kinship --king-cutoff <phi>`: computes all-pairs phi (streamed,
// cap-free), builds the above-cutoff relatedness graph, and runs plink2's exact greedy prune to
// the retained/removed sample sets. `samples` restricts the considered individuals (nullopt =
// all); the sample index / prune tie-break order is the .ind record order (== a matching plink2
// roster's order).
[[nodiscard]] KinshipCutoffResult run_kinship_cutoff(
    const std::string& geno, const std::string& snp, const std::string& ind,
    const std::optional<std::vector<std::string>>& samples, double cutoff,
    device::Resources& resources, const FilterConfig& filter = FilterConfig{});

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
