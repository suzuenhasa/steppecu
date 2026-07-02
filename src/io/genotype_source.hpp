// src/io/genotype_source.hpp
//
// The front door that expands a `--prefix` into its on-disk genotype triple and
// reads the SNP/individual metadata sidecars. Centralizes the EIGENSTRAT-family
// (.geno/.snp/.ind) versus PLINK (.bed/.bim/.fam) fork — both the extension
// choice and the parser choice — in one host-only io-leaf place.
//
// Reference: docs/reference/src_io_genotype_source.hpp.md
#ifndef STEPPE_IO_GENOTYPE_SOURCE_HPP
#define STEPPE_IO_GENOTYPE_SOURCE_HPP

#include <string>

#include "io/eigenstrat_format.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe::io {

// GenotypeTriple — reference §2
struct GenotypeTriple {
    std::string geno;
    std::string snp;
    std::string ind;
    bool is_plink = false;
};

// resolve_genotype_triple — reference §3
[[nodiscard]] GenotypeTriple resolve_genotype_triple(const std::string& prefix);

// read_snp_table — reference §4
[[nodiscard]] SnpTable read_snp_table(GenoFormat format, const std::string& path,
                                      std::size_t max_snps);

// read_ind_partition — reference §5
[[nodiscard]] IndPartition read_ind_partition(GenoFormat format, const std::string& path,
                                              const PopSelection& sel,
                                              std::size_t n_records_present);

}  // namespace steppe::io

#endif  // STEPPE_IO_GENOTYPE_SOURCE_HPP
