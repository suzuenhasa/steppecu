// src/io/faidx_reader.hpp
//
// FaidxReader — a native indexed-FASTA (.fai) random-access reader. Fetches the
// GRCh38 reference base at a 1-based coordinate, replacing the orchestrated
// `samtools faidx` (Stage-0 oracle.fetch_ref_bases) with in-process C++. Used by
// the Stage-2 native panel harmonizer (target_build) to fill the per-locus ref38
// column for block-interior sites the VCF carries no base for.
//
// .fai format (5 tab/ws columns per contig):
//     NAME  LENGTH  OFFSET  LINEBASES  LINEWIDTH
//   OFFSET   = byte offset of the contig's first base
//   LINEBASES = bases per line
//   LINEWIDTH = bytes per line incl. the newline(s) (so '\r\n' => LINEWIDTH-LINEBASES==2)
//
// Byte math for a 1-based pos1:
//     byte = OFFSET + (pos1-1)/LINEBASES * LINEWIDTH + (pos1-1)%LINEBASES
//
// Bases are UPPER-cased on return (soft-mask lowercase -> upper); a literal 'N'
// in the assembly is returned as 'N' (flows to reconcile -> ref_change, matching
// the oracle). Contig-name tolerance: the queried name is tried as-is, then with
// a 'chr' prefix toggled, so an unprefixed "1" resolves a "chr1" FASTA and vice
// versa (spec §1: nikki + panel + Ensembl FASTA are all unprefixed).
//
// Pure host C++20 io-leaf (no zlib, no CUDA). NOT thread-safe: base_at() is const
// but seeks a mutable ifstream, so a single FaidxReader is single-thread only.
#ifndef STEPPE_IO_FAIDX_READER_HPP
#define STEPPE_IO_FAIDX_READER_HPP

#include <fstream>
#include <string>
#include <unordered_map>

namespace steppe::io {

class FaidxReader {
public:
    // Opens `fasta_path` and its sibling index `fasta_path + ".fai"`.
    // Throws std::runtime_error if either cannot be opened / parsed.
    explicit FaidxReader(const std::string& fasta_path);

    // Returns the UPPER-cased reference base at 1-based `pos1` on `contig`.
    // Throws std::runtime_error on an unknown contig or an out-of-range pos1
    // (pos1 < 1 or pos1 > contig length) — a past-end lift is corrupt input and
    // is surfaced, never silently masked to 'N'.
    [[nodiscard]] char base_at(const std::string& contig, long long pos1) const;

    [[nodiscard]] bool has_contig(const std::string& contig) const;

private:
    struct FaiEntry {
        long long length = 0;
        long long offset = 0;
        long long linebases = 0;
        long long linewidth = 0;
    };

    [[nodiscard]] const FaiEntry* resolve(const std::string& contig) const;

    std::string fasta_path_;
    std::unordered_map<std::string, FaiEntry> index_;
    mutable std::ifstream fasta_;  // random-access; const-but-mutating => single-thread
};

}  // namespace steppe::io

#endif  // STEPPE_IO_FAIDX_READER_HPP
