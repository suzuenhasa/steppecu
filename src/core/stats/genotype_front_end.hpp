// src/core/stats/genotype_front_end.hpp
//
// genotype_front_end — the shared genotype DECODE FRONT-END (C1). The single place the
// four genotype-path tools (extract_f2 / qpDstat-B / qpfstats / DATES) turn the genotype
// triple (.geno/.snp/.ind or .bed/.bim/.fam) into the canonical individual-major
// io::GenotypeTile + the parsed SnpTable/IndPartition, STOPPING at the tile/snptab handoff.
// It WRAPS read_canonical_tile (the M-FR-2 format dispatch): open the io::GenoReader, read
// the IndPartition (read_ind_partition) and the SnpTable (read_snp_table, SIZE_MAX), compute
// M0 = min(header().n_snp, snptab.count), then read the canonical tile [0, M0).
//
// The shared boundary is exactly {tile, snptab, part, fmt, M0}; each caller DIVERGES
// afterward — dstat / qpfstats / DATES force diploid + autosome-keep, extract does
// per-sample ploidy + the regime-B filter — so this helper deliberately stops at the handoff
// and does NOT absorb the decode. It is the single edit point for any read-time front-end
// change (a future ploidy / filter / M0 tweak no longer has to be applied in four places in
// lockstep, the parity-divergence risk this dedup closes).
//
// LAYERING (architecture.md §4): like read_canonical_tile this is the orchestration wiring
// point where the CUDA-FREE io leaf meets the CUDA-FREE ComputeBackend seam — NOT a CUDA TU.
// Compiled into steppe_core (which links steppe::io + steppe::device) and reused by both
// steppe_core's genotype tools AND steppe_extract's run_extract_f2 (which links steppe::core).
// No CUDA header appears here; the transpose is reached through read_canonical_tile's
// CUDA-free backend virtual.
#ifndef STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP
#define STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP

#include <cstddef>
#include <span>
#include <string>

#include "io/geno_reader.hpp"      // GenoReader, GenotypeTile
#include "io/genotype_source.hpp"  // read_snp_table / read_ind_partition (format-aware), GenoFormat
#include "io/ind_reader.hpp"       // IndPartition, PopSelection
#include "io/snp_reader.hpp"       // SnpTable

namespace steppe {

class ComputeBackend;  // device/backend.hpp (CUDA-free seam); fwd-declared to keep this lean.

namespace core {

/// The shared genotype front-end result: the canonical individual-major tile plus the parsed
/// SNP table / individual partition and the on-disk format / spanned SNP count the four
/// callers consume. The boundary is exactly {tile, snptab, part, fmt, M0} — each caller
/// diverges into its own per-tool decode after this.
struct GenotypeFrontEnd {
    io::GenotypeTile tile;                          ///< canonical individual-major tile (read_canonical_tile)
    io::SnpTable snptab;                            ///< parsed .snp/.bim metadata (read_snp_table, SIZE_MAX)
    io::IndPartition part;                          ///< selected pops in Q/V/N row order (read_ind_partition)
    io::GenoFormat fmt = io::GenoFormat::Unknown;   ///< the on-disk format pinned by the GenoReader ctor
    std::size_t M0 = 0;                             ///< min(header().n_snp, snptab.count) — the spanned SNP count
};

/// Open `geno`, read the `ind` partition for `sel` and the `snp` table, and read the
/// canonical individual-major tile (SNPs [0, M0)). `backend` is consulted ONLY on the
/// non-TGENO transpose path (forwarded to read_canonical_tile); the helper does NOT own the
/// backend (the caller binds it and reuses it for the decode). Throws std::runtime_error on
/// any reader/transpose failure (the same exception contract read_canonical_tile and the io
/// readers carry).
[[nodiscard]] GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                                       const std::string& snp,
                                                       const std::string& ind,
                                                       const io::PopSelection& sel,
                                                       ComputeBackend& backend);

/// Convenience overload: build the Explicit PopSelection from `pop_labels` (the present
/// subset is kept, sorted ASC by label) then delegate. Collapses the 3-line Explicit-sel
/// build the genotype-D / qpfstats / DATES callers each repeat.
[[nodiscard]] GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                                       const std::string& snp,
                                                       const std::string& ind,
                                                       std::span<const std::string> pop_labels,
                                                       ComputeBackend& backend);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP
