// src/core/stats/genotype_front_end.hpp
//
// The single shared genotype decode front-end: the one place the four genotype-path
// tools (f2 extract, genotype-path D-stat, qpfstats, DATES) turn a genotype triple on
// disk into the canonical individual-major tile plus its parsed SNP table and partition.
// It reads the inputs to a fixed boundary {tile, snptab, part, fmt, M0} and stops there;
// each caller decodes past that on its own.
//
// Reference: docs/reference/src_core_stats_genotype_front_end.hpp.md
#ifndef STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP
#define STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP

#include <cstddef>
#include <span>
#include <string>

#include "device/backend.hpp"  // DeviceGenotypeTile
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe {

// Layering: CUDA-free compute seam, forward-declared — reference §6
class ComputeBackend;

namespace core {

// GenotypeFrontEnd result struct — reference §3.
// GPU-native load: when the selector builds the tile DEVICE-RESIDENT, `dev_tile` is valid and
// `tile` carries only the descriptor (n_snp/n_individuals/pop_offsets/pop_labels/bytes_per_record)
// with an EMPTY packed vector — the 7 GB matrix never materializes on host. Callers pick the view
// off dev_tile when valid, else off tile (the legacy host path). `dev_tile` is invalid on the host
// path (CpuBackend, STEPPE_GPU_LOAD=0, STEPPE_HOST_FILTER=1 with an active filter, or M0 == 0). An
// active SNP filter no longer forces the host path — the filter compacts the resident tile
// on-device (apply_snp_filter -> compact_tile_columns_device).
struct GenotypeFrontEnd {
    io::GenotypeTile tile;
    io::SnpTable snptab;
    io::IndPartition part;
    io::GenoFormat fmt = io::GenoFormat::Unknown;
    std::size_t M0 = 0;
    DeviceGenotypeTile dev_tile;
};

// read_genotype_front_end: primary entry point — reference §4.
// allow_device: opt in to the GPU-native device-resident load. A migrated caller (fst / kinship /
// pca / admixture) passes true unconditionally — a filtered run compacts the resident tile
// on-device via apply_snp_filter, so an active filter no longer needs the host packed bytes;
// STEPPE_HOST_FILTER=1 makes it pass false to force the legacy host repack oracle. Even then the
// load is device-resident only when a CUDA backend is present and STEPPE_GPU_LOAD != 0; otherwise
// the host path runs. Default false keeps the un-migrated callers (dstat / qpfstats / dates / sfs,
// which filter through the f2 regime-B decode path) on the host tile.
[[nodiscard]] GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                                       const std::string& snp,
                                                       const std::string& ind,
                                                       const io::PopSelection& sel,
                                                       ComputeBackend& backend,
                                                       bool allow_device = false);

// pop_labels convenience overload — reference §5
[[nodiscard]] GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                                       const std::string& snp,
                                                       const std::string& ind,
                                                       std::span<const std::string> pop_labels,
                                                       ComputeBackend& backend,
                                                       bool allow_device = false);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP
