// src/io/genotype_source.cpp
//
// Implementation of the format-aware genotype-triple resolver + the SnpTable /
// IndPartition front-door (genotype_source.hpp; M-FR PLINK). EXTENSION resolution is a
// filesystem probe (.geno present wins, else .bed -> PLINK); the per-file parser is
// dispatched on the GenoReader's detected format.
//
// LAYERING: `io`-leaf TU (architecture.md §4) — pure host C++20, no CUDA, no
// core/device dependency.
#include "io/genotype_source.hpp"

#include <filesystem>

#include "io/plink_reader.hpp"  // read_bim, read_fam (the PLINK parsers)

namespace steppe::io {

GenotypeTriple resolve_genotype_triple(const std::string& prefix) {
    GenotypeTriple t;
    std::error_code ec;
    const bool has_geno = std::filesystem::exists(prefix + ".geno", ec);
    const bool has_bed = std::filesystem::exists(prefix + ".bed", ec);
    // .geno present ALWAYS wins (keeps every TGENO/GENO/EIGENSTRAT prefix resolving
    // exactly as before); PLINK is chosen ONLY when there is a .bed and no .geno.
    if (!has_geno && has_bed) {
        t.geno = prefix + ".bed";
        t.snp = prefix + ".bim";
        t.ind = prefix + ".fam";
        t.is_plink = true;
    } else {
        t.geno = prefix + ".geno";
        t.snp = prefix + ".snp";
        t.ind = prefix + ".ind";
        t.is_plink = false;
    }
    return t;
}

SnpTable read_snp_table(GenoFormat format, const std::string& path,
                        std::size_t max_snps) {
    if (format == GenoFormat::Plink) return read_bim(path, max_snps);
    return read_snp(path, max_snps);
}

IndPartition read_ind_partition(GenoFormat format, const std::string& path,
                                const PopSelection& sel,
                                std::size_t n_records_present) {
    if (format == GenoFormat::Plink) return read_fam(path, sel, n_records_present);
    return read_ind(path, sel, n_records_present);
}

}  // namespace steppe::io
