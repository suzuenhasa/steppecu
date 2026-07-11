// src/io/vcf_panel_reader.hpp
//
// Multi-sample phased-VCF -> canonical haplotype-panel reader (Phase-1 core).
//
// Streams a phased multi-sample .vcf.gz (1000G phase-3 shape) forward once and
// turns every kept biallelic SNP into TWO haploid columns per sample (allele
// before '|' = hap1, after '|' = hap2), packing them SNP-major into an
// io::SnpMajorTile whose 2-bit codes are {0,2,3} (allele0->0, allele1->2,
// missing/unphased-het->3 — NEVER code 1). The tile is exactly what
// core::transpose_snp_major consumes to build the canonical individual-major
// io::GenotypeTile the paint/IBD compute path already accepts.
//
// This is a DEDICATED forward-only streaming producer: it reads SNPs and
// individuals INLINE from the VCF (no .snp/.ind sidecars) and never masquerades
// as a random-access io::GenoReader window. Unlike the GL path it does NOT hold
// the whole file in a position-keyed map — each record is decoded and packed as
// it streams, so RAM stays O(kept sites x haps) for the streamed region.
//
// Pure host C++20 io-leaf (links zlib only, through GzipLineReader). Failures
// surface as std::runtime_error.
#ifndef STEPPE_IO_VCF_PANEL_READER_HPP
#define STEPPE_IO_VCF_PANEL_READER_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "io/snp_major_tile.hpp"
#include "io/snp_reader.hpp"

namespace steppe::io {

// An optional in-stream POS-range filter (the bounded gate region). No tabix —
// the reader stays forward-streaming; it simply skips records outside the range
// and stops once it passes the range end on the matching chromosome (VCF is
// position-sorted within a chromosome).
struct VcfPanelRegion {
    bool active = false;
    std::string chrom;        // matched against the 'chr'-stripped CHROM column
    long long start = 0;      // inclusive
    long long end = 0;        // inclusive
};

// Reader options.
struct VcfPanelOptions {
    std::string map_path;     // OPTIONAL plink/HapMap genetic map; empty -> genpos 0
    VcfPanelRegion region;    // OPTIONAL bounded POS-range filter
    // Fail loud when the fraction of diploid GT calls that were UNPHASED HET
    // (dropped to a missing haplotype pair) exceeds this. Paint's own n_diploid
    // gate does NOT catch phase loss (unphased -> code 3 slips it), so the reader
    // must guard it. Default 1.0 = report-only; set low to make it fatal.
    // IGNORED in hardcall mode (an unphased het is a legitimate dosage-1 call).
    double unphased_max = 1.0;
    // HARDCALL (unphased diploid) mode: decode each biallelic-SNP GT into ONE
    // dosage column per sample (codes {0,1,2,3}: 0/0->0, het->1, 1/1->2, ./.->3;
    // phase-agnostic) instead of two haploid columns. n_individuals == n_sample
    // (not 2*n_sample). Everything else — BGZF scan, region/dup/biallelic filters,
    // the SNP-major pack, the CPU/GPU selector — is shared with the phased path.
    bool hardcall = false;
};

// Streaming pass counters (reported on stderr by the CLI).
struct VcfPanelCounts {
    long long records_seen = 0;             // data records scanned
    long long emitted_sites = 0;            // biallelic-SNP sites kept (== tile.n_snp)
    long long skipped_multiallelic = 0;     // ALT carried ',' (>1 ALT)
    long long skipped_non_snp = 0;          // indel / symbolic / '*' allele
    long long skipped_dup_pos = 0;          // norm -d all: duplicate POS collapsed
    long long skipped_out_of_region = 0;    // outside the --region filter
    long long diploid_calls = 0;            // emitted_sites * n_sample
    long long unphased_het_dropped = 0;     // unphased-het calls set to a missing hap pair
                                            // (phased mode only; always 0 in hardcall mode)
    long long half_missing_haps = 0;        // one hap present, one '.' (e.g. 0|.)
                                            // (phased mode only; always 0 in hardcall mode)
    long long missing_haps = 0;             // phased: haplotype cells emitted as code 3;
                                            // hardcall: per-sample dosage calls emitted as code 3
};

// The producer result: the SNP-major panel tile, its inline-parsed SNP table
// (genpos_morgans filled from the map join), the resolved sample ids, and the
// pass counts.
struct VcfPanelResult {
    SnpMajorTile tile;
    SnpTable snptab;
    std::vector<std::string> sample_ids;    // #CHROM column order, length n_sample
    std::size_t n_sample = 0;
    VcfPanelCounts counts;
};

// Stream the VCF into a haplotype panel. Throws std::runtime_error on I/O
// failure, a malformed header, or (when opts.unphased_max is exceeded) on phase
// loss.
[[nodiscard]] VcfPanelResult read_vcf_panel(const std::string& vcf_path,
                                            const VcfPanelOptions& opts);

// Host-only gate helper: dump the SNP-major tile back to the sites x columns
// matrix the bcftools oracle produces — one row per SNP, "CHROM<TAB>POS" then one
// code per output column, tab-separated. Width-generic over tile.n_individuals:
// for the phased panel that is the sites x haps {0,2,3} matrix (sample0.hap1,
// sample0.hap2, sample1.hap1, ...); for the hardcall panel it is the sites x
// samples {0,1,2,3} diploid-dosage matrix (one code per sample). The bit-exact
// panel-gate artifact for both paths.
void dump_hap_codes(const SnpMajorTile& tile, const SnpTable& snptab, std::ostream& os);

}  // namespace steppe::io

#endif  // STEPPE_IO_VCF_PANEL_READER_HPP
