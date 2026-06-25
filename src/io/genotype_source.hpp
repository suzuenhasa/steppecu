// src/io/genotype_source.hpp
//
// Format-aware genotype-triple RESOLUTION + the SnpTable / IndPartition front-door
// (M-FR PLINK; docs/design/format-readers.md §3.2). steppe accepts a single `--prefix P`
// and expands it to the on-disk genotype triple. For the EIGENSTRAT family
// (TGENO / GENO / EIGENSTRAT) that is `P.geno` / `P.snp` / `P.ind`; for PLINK it is
// `P.bed` / `P.bim` / `P.fam` — DIFFERENT extensions AND different per-file parsers
// (read_bim vs read_snp, read_fam vs read_ind). This header centralizes BOTH the
// extension choice and the parser choice so the five genotype-path callers
// (extract_f2 / qpDstat-B / qpfstats / DATES / the CLI) do NOT each re-spell the
// `.geno/.snp/.ind` vs `.bed/.bim/.fam` fork.
//
//   * resolve_genotype_triple(prefix) — probe the filesystem: if `P.bed` exists (and
//     `P.geno` does not), the triple is PLINK (`.bed/.bim/.fam`); otherwise it is the
//     EIGENSTRAT family (`.geno/.snp/.ind`, the default — TGENO/GENO/EIGENSTRAT all
//     share these). The actual on-disk FORMAT (Tgeno/Geno/Eigenstrat/Plink) is then
//     pinned by the GenoReader ctor reading `triple.geno` (magic / content / .bed magic);
//     this resolver only chooses the EXTENSIONS, never decodes.
//
//   * read_snp_table(geno_format, snp_or_bim_path, max) — dispatch to read_snp (the
//     EIGENSTRAT-family .snp) or read_bim (PLINK), returning the SAME io::SnpTable.
//
//   * read_ind_partition(geno_format, ind_or_fam_path, sel, n_present) — dispatch to
//     read_ind (.ind) or read_fam (PLINK .fam), returning the SAME io::IndPartition.
//
// LAYERING: `io`-leaf header (architecture.md §4) — pure host C++20, no CUDA, no
// core/device dependency. The GenoFormat parameter is the io-leaf enum
// (eigenstrat_format.hpp); the format is determined by the GenoReader ctor and passed
// here, so this front-door never re-probes the magic.
#ifndef STEPPE_IO_GENOTYPE_SOURCE_HPP
#define STEPPE_IO_GENOTYPE_SOURCE_HPP

#include <string>

#include "io/eigenstrat_format.hpp"  // GenoFormat
#include "io/ind_reader.hpp"         // IndPartition, PopSelection
#include "io/snp_reader.hpp"         // SnpTable

namespace steppe::io {

/// The resolved on-disk genotype triple for a `--prefix` (the three sibling paths the
/// genotype-path tools open). `geno` feeds io::GenoReader (which pins the real format);
/// `snp` and `ind` feed read_snp_table / read_ind_partition (which dispatch the parser
/// on the GenoReader's detected format). For the EIGENSTRAT family these are
/// `P.geno/.snp/.ind`; for PLINK they are `P.bed/.bim/.fam`.
struct GenotypeTriple {
    std::string geno;  ///< the .geno (TGENO/GENO/EIGENSTRAT) or .bed (PLINK) path
    std::string snp;   ///< the .snp (EIGENSTRAT family) or .bim (PLINK) path
    std::string ind;   ///< the .ind (EIGENSTRAT family) or .fam (PLINK) path
    bool is_plink = false;  ///< true iff the .bed/.bim/.fam extensions were chosen
};

/// Expand `prefix` to its genotype triple, choosing the EXTENSIONS by a filesystem
/// probe: if `prefix.bed` exists AND `prefix.geno` does NOT, the triple is PLINK
/// (`.bed/.bim/.fam`); otherwise it is the EIGENSTRAT family (`.geno/.snp/.ind` — the
/// default, covering TGENO/GENO/EIGENSTRAT, which all share those extensions). This is
/// EXTENSION resolution only — the authoritative format is pinned later by the
/// GenoReader ctor opening `triple.geno`. The `.geno`-present-wins rule keeps every
/// existing EIGENSTRAT-family prefix resolving EXACTLY as before (no behavior change for
/// the established formats); PLINK is reached only when there is no `.geno` but there is
/// a `.bed`.
[[nodiscard]] GenotypeTriple resolve_genotype_triple(const std::string& prefix);

/// Read the per-SNP metadata table, dispatching the parser on the on-disk format:
/// PLINK -> read_bim(path, max_snps); everything else -> read_snp(path, max_snps). Both
/// return the SAME io::SnpTable (id/chrom/genpos_morgans/ref/alt), so the caller's
/// downstream filter / block partition / decode is byte-for-byte unchanged.
[[nodiscard]] SnpTable read_snp_table(GenoFormat format, const std::string& path,
                                      std::size_t max_snps);

/// Read the .ind/.fam partition, dispatching the parser on the on-disk format: PLINK ->
/// read_fam(path, sel, n_records_present); everything else -> read_ind(...). Both return
/// the SAME io::IndPartition (selected pops in Q/V/N row order, sorted ASC by label).
[[nodiscard]] IndPartition read_ind_partition(GenoFormat format, const std::string& path,
                                              const PopSelection& sel,
                                              std::size_t n_records_present);

}  // namespace steppe::io

#endif  // STEPPE_IO_GENOTYPE_SOURCE_HPP
