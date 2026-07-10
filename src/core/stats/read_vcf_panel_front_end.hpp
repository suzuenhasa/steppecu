// src/core/stats/read_vcf_panel_front_end.hpp
//
// The dedicated phased-VCF front-end: read SNPs + individuals INLINE from a phased
// multi-sample .vcf.gz and return the SAME core::GenotypeFrontEnd {tile, snptab,
// part, fmt, M0} the paint / IBD path already consumes — with no .snp/.ind sidecar
// files and no bcftools prep.
//
// It is a forward-only streaming producer, NOT a random-access io::GenoReader
// window and NOT a GenoFormat::Vcf switch arm in read_canonical_tile (the design's
// rejected Option A). It calls the io leaf io::read_vcf_panel to build the SNP-major
// haplotype tile + inline SnpTable, then runs the ONE shared on-device transpose
// core::transpose_snp_major to hand paint the canonical individual-major tile.
//
// Reference: docs/planning/multisample-phased-vcf-reader-design.md (critic §"Corrected
// build order" item 2: replace the 6th switch arm with this dedicated front-end).
#ifndef STEPPE_CORE_STATS_READ_VCF_PANEL_FRONT_END_HPP
#define STEPPE_CORE_STATS_READ_VCF_PANEL_FRONT_END_HPP

#include <string>

#include "core/stats/genotype_front_end.hpp"
#include "io/vcf_panel_reader.hpp"

namespace steppe {

class ComputeBackend;

namespace core {

// Read a phased VCF into the canonical GenotypeFrontEnd. `opts` carries the optional
// genetic-map join (map_path -> genpos Morgans), the optional bounded POS-range
// filter (region), and the unphased-drop guard (unphased_max) that the io reader
// enforces (fails loud above the threshold). Throws std::runtime_error on I/O
// failure, a malformed header, or phase loss above the guard.
//
// The returned struct is populated so paint reads it exactly like a file triple:
//   .tile   canonical individual-major haplotype panel (one haploid column / column)
//   .snptab inline-parsed SNP table (genpos_morgans from the map join)
//   .part   a single "PANEL" partition over all haplotype columns (struct-complete;
//           paint reads pop labels off the tile, not part)
//   .fmt    GenoFormat::Unknown (inline VCF is not an on-disk GenoReader format)
//   .M0     the emitted-site count
[[nodiscard]] GenotypeFrontEnd read_vcf_panel_front_end(const std::string& vcf,
                                                        const io::VcfPanelOptions& opts,
                                                        ComputeBackend& backend);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_READ_VCF_PANEL_FRONT_END_HPP
