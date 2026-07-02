// src/io/plink_reader.hpp
//
// Host-side parsers for the two PLINK metadata sidecar files: read_bim (the SNP
// table) and read_fam (the individual/population table). Both return the SAME
// structs their EIGENSTRAT twins read_snp / read_ind produce, so everything
// downstream consumes a PLINK dataset identically.
//
// Reference: docs/reference/src_io_plink_reader.hpp.md
#ifndef STEPPE_IO_PLINK_READER_HPP
#define STEPPE_IO_PLINK_READER_HPP

#include <cstddef>
#include <string>

#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe::io {

// read_bim — the .bim SNP-table parser — reference §2
[[nodiscard]] SnpTable read_bim(const std::string& path, std::size_t max_snps);

// read_fam — the .fam individual/population parser — reference §4
[[nodiscard]] IndPartition read_fam(const std::string& path,
                                    const PopSelection& sel,
                                    std::size_t n_records_present);

}  // namespace steppe::io

#endif  // STEPPE_IO_PLINK_READER_HPP
