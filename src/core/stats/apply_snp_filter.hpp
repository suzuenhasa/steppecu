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
// Design note (device-resident compaction): the migrated tools (fst / kinship / pca / admixture)
// build the canonical tile DEVICE-RESIDENT (the GPU-native load), so when a filter is active the
// kept-column 2-bit gather runs ON DEVICE — compact_tile_columns_device compacts the resident tile
// straight into a new resident tile (byte-for-byte identical to the old host repack_tile_columns:
// same genotype_code extract, same MSB-first shift), and `tile` keeps only the refreshed descriptor
// with an empty packed vector. No host materialize, no single-core gather, no re-upload. The keep
// COMPUTE (the per-pop allele-frequency decode) already ran on the device via
// decode_af_pooled_summary. STEPPE_HOST_FILTER=1 forces the legacy host path (materialize +
// repack_tile_columns + re-upload) as the byte-exact invariance oracle; and when no device tile is
// available (CpuBackend / STEPPE_GPU_LOAD=0 / M0 == 0) the host repack path runs unchanged.
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
struct DeviceGenotypeTile;  // device/backend.hpp — the device-resident tile (fwd for the seam)

namespace core {

// The outcome of one apply_snp_filter call.
struct SnpFilterOutcome {
    bool applied = false;                 // true iff any SNP was dropped (the tile was repacked)
    long n_in = 0;                        // SNP count before filtering
    long n_kept = 0;                      // SNP count after filtering
    std::vector<std::string> kept_ids;    // the retained SNP ids, in kept order (== snptab.id post)
};

// host_filter_forced — STEPPE_HOST_FILTER selector. When set (non-empty, not "0"/"off"/"false"/
// "no") it forces the legacy host materialize + repack_tile_columns + re-upload filter path — the
// byte-exact invariance oracle for the device-column-compaction gate. Default OFF (the tools pass
// allow_device=true so a filtered load is device-resident and the compaction runs on the GPU).
[[nodiscard]] bool host_filter_forced() noexcept;

// Subset `tile` + `snptab` in place to the SNPs the QC filter keeps (see the file header). When
// `dev_tile` is valid (the GPU-native filtered load) the 2-bit column compaction runs ON DEVICE
// via ComputeBackend::compact_tile_columns_device (dev_tile is replaced with the compacted resident
// tile and `tile` keeps only the descriptor with an empty packed vector); otherwise the host
// repack_tile_columns path runs unchanged. A no-op when the filter is inactive or keeps every SNP.
// Throws std::invalid_argument on a same-ascertainment refusal or when the filter keeps zero SNPs.
[[nodiscard]] SnpFilterOutcome apply_snp_filter(io::GenotypeTile& tile,
                                                DeviceGenotypeTile& dev_tile, io::SnpTable& snptab,
                                                const FilterConfig& cfg, ComputeBackend& backend);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_APPLY_SNP_FILTER_HPP
