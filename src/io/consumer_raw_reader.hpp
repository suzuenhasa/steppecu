// src/io/consumer_raw_reader.hpp
//
// ConsumerRawReader — steppe's direct-to-consumer (DTC) raw-genotype reader (the
// seventh reader arm). Parses a 23andMe, AncestryDNA, or MyHeritage raw export
// (layout auto-detected from the header) and genotypes it against the SAME
// GRCh37 TargetSites the native VcfReader uses, emitting a byte-identical
// VcfIngestResult (per-site VcfSiteCall rows + the canonical individual-major
// SnpMajorTile for the one sample), so every downstream consumer — the report
// writer, the panel-merge writer, and the shared device transpose — reuses it
// verbatim.
//
// Unlike the VCF path there is NO gVCF ref-confidence block, NO DP/GQ floor, and
// NO liftover: consumer files carry one hardcall per assayed SNP on GRCh37 plus/
// forward strand. Each site's two observed nucleotide letters are reconciled to
// copies-of-A1 through the shared reconcile() (io/allele_reconcile.hpp) — the same
// strand/allele resolver and the same A1-counted polarity as the VCF hardcall
// path, so a merged panel is never flipped vs an existing panel. A no-call
// (--, 00, DD/II) or any non-ACGT letter -> MISSING; palindromic (A/T, C/G) target
// sites are dropped exactly as the VCF path. Records join to the panel by rsID,
// falling back to chrom+pos37.
//
// Pure host C++20 io-leaf (zlib only, via GzipLineReader); failures surface as
// std::runtime_error.
#ifndef STEPPE_IO_CONSUMER_RAW_READER_HPP
#define STEPPE_IO_CONSUMER_RAW_READER_HPP

#include <string>

#include "io/target_sites.hpp"
#include "io/vcf_reader.hpp"  // VcfIngestResult / VcfSiteCall / VcfCall / VcfCounts

namespace steppe::io {

// The three consumer raw-export layouts, auto-detected from the header line.
enum class RawLayout {
    Unknown,
    TwentyThreeAndMe,  // TSV: rsid chromosome position genotype        (genotype = 2 letters, or --)
    AncestryDNA,       // TSV: rsid chromosome position allele1 allele2 (0 == no-call)
    MyHeritage,        // CSV (quoted): RSID,CHROMOSOME,POSITION,RESULT  (RESULT = 2 letters)
};

class ConsumerRawReader {
public:
    // sample_id == "" -> derive a label from the file stem.
    ConsumerRawReader(std::string raw_path, const TargetSites& targets, std::string sample_id);

    // Genotype the file against the target sites, returning the SAME shape as
    // VcfReader::genotype() (canonical tile + per-site report rows + counts).
    [[nodiscard]] VcfIngestResult genotype();

    // The auto-detected layout after genotype() (Unknown before, or on failure).
    [[nodiscard]] RawLayout layout() const noexcept { return layout_; }

private:
    std::string raw_path_;
    const TargetSites& targets_;
    std::string sample_id_;
    RawLayout layout_ = RawLayout::Unknown;
};

}  // namespace steppe::io

#endif  // STEPPE_IO_CONSUMER_RAW_READER_HPP
