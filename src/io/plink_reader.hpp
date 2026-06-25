// src/io/plink_reader.hpp
//
// PLINK .bim / .fam parsers (M-FR PLINK; docs/design/format-readers.md §3.2, §3.3
// P2). The PLINK genotype triple is .bed / .bim / .fam — NEITHER the .bim nor the
// .fam matches the EIGENSTRAT .snp/.ind COLUMN LAYOUT, so read_snp / read_ind do NOT
// apply (format-readers.md §3.2: ".bim is 6 cols (chrom, id, genpos-Morgans, physpos,
// A1, A2); .fam is 6 cols (FID, IID, pat, mat, sex, pheno) — neither matches the
// EIGENSTRAT .snp/.ind layout"). This file is the PLINK twin of snp_reader / ind_reader:
//
//   * read_bim(path, max_snps) -> io::SnpTable, the SAME struct read_snp produces
//     (id / chrom / genpos_morgans / ref / alt), so the downstream filter / block
//     partition / decode consume it byte-for-byte identically. The ONLY difference is
//     the column ORDER (chrom is col 1, id is col 2; ref/alt are A1/A2 in cols 5/6).
//     CANONICAL POLARITY: the .bed 2-bit code counts A1 copies, so the canonical
//     reference allele (the allele steppe's Q frequency counts) is DEFINED as A1 and
//     the alternate as A2 — ref := A1, alt := A2. A standalone PLINK dataset is
//     self-describing via the .bim; there is no external .snp ref to reconcile against,
//     so A1 IS the reference by construction (the .bed LUT in geno_reader then yields
//     the canonical ref-copy code directly — format-readers.md §3.2, the highest-risk
//     plug-in's polarity is pinned by the Tier-1 bit-exact gate vs the PA tile).
//
//   * read_fam(path, sel, n_records_present) -> io::IndPartition, the SAME struct
//     read_ind produces (selected pops in Q/V/N row order). The .fam population (egroup)
//     label is column 6 (the PHENOTYPE), per AT2's own PLINK reader (admixtools
//     mcio.c:1180-1205: "1"->Control, "2"->Case, "9"/"-9"/"0"->ignore, any other string IS
//     the population). convertf PACKEDPED with `outputgroup: YES` writes the EIGENSTRAT
//     population label into col6; the FID (col 1) is just a numeric counter (mcio.c:3917).
//     read_fam reproduces read_ind's selection semantics EXACTLY (Explicit / AutoTopK /
//     MinN; the final set sorted ASC by label) so the PLINK partition matches the
//     PA/EIGENSTRAT partition of the SAME subset — and the partition AT2 itself builds.
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, depends on
// nothing in core/device, no CUDA. I/O failures surface as std::runtime_error.
#ifndef STEPPE_IO_PLINK_READER_HPP
#define STEPPE_IO_PLINK_READER_HPP

#include <cstddef>
#include <string>

#include "io/ind_reader.hpp"  // IndPartition, PopSelection (read_fam reuses read_ind's selection types)
#include "io/snp_reader.hpp"  // SnpTable (read_bim produces the SAME struct as read_snp)

namespace steppe::io {

/// Parse the PLINK .bim at `path`, reading the first `max_snps` records in file order
/// (pass SIZE_MAX for every SNP). The .bim is one whitespace-separated record per SNP:
///   <chromosome>  <snp-id>  <genetic-pos-Morgans>  <physical-pos>  <A1>  <A2>
/// i.e. the EIGENSTRAT .snp columns with chrom and id SWAPPED and the alleles being
/// A1/A2 (not ref/alt). This returns the SAME io::SnpTable read_snp produces, with:
///   * id              = col 2 (the SNP id — the M2 membership key);
///   * chrom           = col 1 (parsed via the SAME chrom_code convention as read_snp:
///                       numeric pass-through; X/Y/MT -> the EIGENSOFT codes; other ->
///                       a stable negative sentinel);
///   * genpos_morgans  = col 3 as-read (Morgans), fed to the shared block rule;
///   * ref             = col 5 (A1) — the CANONICAL reference allele (Q polarity);
///   * alt             = col 6 (A2) — the alternate allele.
/// CANONICAL POLARITY (the load-bearing PLINK decision): the .bed 2-bit code counts
/// A1 copies (format-readers.md §3.2), so ref := A1 makes the .bed LUT yield the
/// canonical ref-copy code with NO per-SNP flip — A1 IS the reference for a standalone
/// PLINK file. The Tier-1 bit-exact gate vs the PA tile pins this: convertf writes the
/// PA/EIGENSTRAT ref as PLINK A1, so PLINK ref(:=A1) == PA ref for every SNP.
///
/// Throws std::runtime_error on: a missing/unreadable file; a malformed record (fewer
/// than 6 whitespace-separated fields, or a non-finite/garbage genetic position); or an
/// interior blank line (the .bim row index IS the SNP index — a silently-skipped record
/// desyncs the SNP axis from the .bed). The diagnostic carries the 1-based line number.
/// A single trailing blank line at EOF is tolerated. (Mirrors read_snp's contract.)
[[nodiscard]] SnpTable read_bim(const std::string& path, std::size_t max_snps);

/// Parse the PLINK .fam at `path` and apply the population selection, returning the
/// SAME io::IndPartition read_ind produces (selected pops in Q/V/N row order, sorted
/// ASC by label). The .fam is one whitespace-separated record per individual:
///   <FID>  <IID>  <pat>  <mat>  <sex>  <pheno>
/// The POPULATION (egroup) LABEL is column 6 (the PHENOTYPE), per AT2's own PLINK reader
/// (admixtools mcio.c:1180-1205): a col6 of "1" -> Control, "2" -> Case, "9"/"-9"/"0" ->
/// the individual is IGNORED (excluded from every group, never selected), and ANY OTHER
/// string IS the population label. convertf PACKEDPED with `outputgroup: YES` writes the
/// EIGENSTRAT population label into THIS column (mcio.c:3940/3997); the FID (col 1) is a
/// numeric counter convertf makes up (mcio.c:3917), NOT the population, so it is NOT used
/// for grouping. read_fam groups individual-record indices by the col6 egroup and
/// reproduces read_ind's selection EXACTLY (Explicit / AutoTopK / MinN; first-appearance
/// tie-break for AutoTopK; the final set sorted ASC by label), so the PLINK partition is
/// identical to the PA/EIGENSTRAT partition of the SAME convertf subset — and identical to
/// the partition AT2 itself would build from this .fam.
///
/// The .bed individual axis is 1:1 with the .fam lines (the universal PLINK invariant);
/// an "ignored" (col6 in {9,-9,0}) individual still consumes its .bed row index (it is just
/// never grouped), so the remaining individuals stay aligned with their .bed records.
///
/// `n_records_present` caps the individual axis to the records actually present in the
/// .bed (rows at index >= n_records_present are ignored; pass the .bed individual count,
/// or SIZE_MAX to use every .fam row) — the SAME partial-file cap read_ind takes.
///
/// Throws std::runtime_error on a missing/unreadable file or an empty selection
/// (mirrors read_ind's contract — the io leaf surfaces I/O failures as exceptions).
[[nodiscard]] IndPartition read_fam(const std::string& path,
                                    const PopSelection& sel,
                                    std::size_t n_records_present);

}  // namespace steppe::io

#endif  // STEPPE_IO_PLINK_READER_HPP
