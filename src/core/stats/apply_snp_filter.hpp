// src/core/stats/apply_snp_filter.hpp
//
// The one shared per-SNP QC-filter seam for the genotype-path tools (pca / admixture / fst /
// kinship). Each tool reads its triple through the front-end into the canonical individual-
// major tile + SnpTable, then calls apply_snp_filter ONCE — before it builds its DecodeTileView
// — to subset the SNP axis in place. The keep decision is the same SNP-global mask
// build_snp_keep_mask produces for extract-f2 (pooled folded MAF + per-individual missing +
// monomorphic + strand/multiallelic/transversion + autosome + include/exclude/prune), so a
// filtered run is byte-for-byte a run on an externally pre-subset triple: the filter changes
// ONLY which SNP columns survive, never the downstream reduction order or math. That is what
// keeps fst/kinship values bit-exact, admixture-Q bit-exact under deterministic EM, and the
// pca covariance (hence eigenvectors up to per-component sign) bit-exact.
//
// It also fires the same-ascertainment guard: when a --keep-snps / prune.in / include list is
// supplied and it is largely absent from the target .snp (a cross-panel intersection — the
// consumer-DNA f4-bias failure), it throws std::invalid_argument unless the config sets
// allow_mixed_ascertainment. A no-op (inactive filter, or a mask that keeps every SNP) leaves
// the tile and SnpTable untouched.
//
// Design note (host repack): the tile is host-resident (GenotypeTile owns std::vector<uint8_t>)
// and every tool H2D-copies it inside its own backend kernel; the genotype-scale COMPUTE for
// the mask (the per-pop allele-frequency decode) runs on the device via ComputeBackend::decode_af.
// The kept-column repack is a pure 2-bit gather with no arithmetic, so it is done host-side
// using the shared genotype_code packing helper (bit-exact by construction) rather than paying
// an H2D+D2H round-trip only to hand the result straight back to the host tile the tool re-uploads.
#ifndef STEPPE_CORE_STATS_APPLY_SNP_FILTER_HPP
#define STEPPE_CORE_STATS_APPLY_SNP_FILTER_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "io/genotype_tile.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"

namespace steppe {

class ComputeBackend;

namespace core {

// The outcome of one apply_snp_filter call.
struct SnpFilterOutcome {
    bool applied = false;                 // true iff any SNP was dropped (the tile was repacked)
    long n_in = 0;                        // SNP count before filtering
    long n_kept = 0;                      // SNP count after filtering
    std::vector<std::string> kept_ids;    // the retained SNP ids, in kept order (== snptab.id post)
};

// Subset `tile` + `snptab` in place to the SNPs the QC filter keeps (see the file header).
// A no-op when the filter is inactive or keeps every SNP. Throws std::invalid_argument on a
// same-ascertainment refusal or when the filter would keep zero SNPs.
[[nodiscard]] SnpFilterOutcome apply_snp_filter(io::GenotypeTile& tile, io::SnpTable& snptab,
                                                const FilterConfig& cfg, ComputeBackend& backend);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_APPLY_SNP_FILTER_HPP
