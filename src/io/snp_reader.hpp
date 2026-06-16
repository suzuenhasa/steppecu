// src/io/snp_reader.hpp
//
// .snp parse → per-SNP genetic position (Morgans), chromosome, ref/alt allele
// (architecture.md §4 `io` LEAF, §5 S0; ROADMAP M1).
//
// The EIGENSTRAT .snp file has one whitespace-separated record per SNP:
//   <snp-id>  <chromosome>  <genetic-pos-Morgans>  <physical-pos>  <ref>  <alt>
// (e.g. `rs3094315  1  0.020130  752566  G  A`). Column 3 is the genetic position
// in MORGANS as-read — it feeds M3's shared block_partition_rule (`block_of`),
// which speaks Morgans directly (core/domain/block_partition_rule.hpp); this
// reader surfaces it unchanged and does NOT compute block ids itself (re-deriving
// the block rule in `io` is the named smell — architecture.md §2, §8).
//
// Column 5 is the REFERENCE allele: the .geno 2-bit code counts copies of THIS
// allele, so the reference allele fixes Q's polarity (Q = ref-allele frequency),
// consistent across populations with no per-pop re-polarization (views.hpp Q/V/N
// contract; oracle build_tgeno_matrix.py).
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, depends on
// nothing in core/device.
#ifndef STEPPE_IO_SNP_READER_HPP
#define STEPPE_IO_SNP_READER_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace steppe::io {

/// Per-SNP metadata read from the .snp, parallel arrays in file order (SNP `s` is
/// element `s` of each vector). `chrom` is the chromosome code (only equality
/// between adjacent SNPs matters to the block rule); `genpos_morgans` is column 3
/// as-read (Morgans), fed to the shared block_partition_rule. `ref`/`alt` are the
/// single-character allele codes (col 5 = reference fixing Q's polarity, col 6 =
/// alternate). All vectors have the same length `count` (== number of SNPs read).
struct SnpTable {
    std::vector<int> chrom;                 ///< chromosome code per SNP (file order)
    std::vector<double> genpos_morgans;     ///< genetic position, Morgans, as-read
    std::vector<char> ref;                  ///< reference allele (Q polarity)
    std::vector<char> alt;                  ///< alternate allele
    std::size_t count = 0;                  ///< number of SNPs read (== M)
};

/// Parse the .snp at `path`, reading the first `max_snps` records in file order
/// (pass SIZE_MAX for every SNP). `max_snps` is the SNP cap — for M1 it is the
/// derived_acc `--snp-cap 100000` (the first 100k SNPs in file order), the SAME
/// prefix the oracle decodes.
///
/// Chromosome codes are parsed as integers when numeric; the common
/// non-numeric/sex labels (X→23, Y→24, MT→90 — EIGENSTRAT conventions) are mapped
/// so adjacent-equality (all the block rule needs) is well-defined. Any other
/// non-numeric code maps to a stable negative sentinel per distinct label.
///
/// Throws std::runtime_error on a missing/unreadable file (the `io` leaf surfaces
/// I/O failures as exceptions to its caller; architecture.md §4).
[[nodiscard]] SnpTable read_snp(const std::string& path, std::size_t max_snps);

}  // namespace steppe::io

#endif  // STEPPE_IO_SNP_READER_HPP
