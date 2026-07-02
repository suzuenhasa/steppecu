// src/io/genotype_source.cpp
//
// Format-aware genotype-triple resolver plus the SnpTable / IndPartition
// front-door: .geno present wins, else .bed selects the PLINK path, and the
// per-file parser is dispatched on the reader's detected format.
#include "io/genotype_source.hpp"

#include <filesystem>

#include "io/plink_reader.hpp"

namespace steppe::io {

GenotypeTriple resolve_genotype_triple(const std::string& prefix) {
    GenotypeTriple t;
    std::error_code ec;
    const bool has_geno = std::filesystem::exists(prefix + ".geno", ec);
    const bool has_bed = std::filesystem::exists(prefix + ".bed", ec);
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
