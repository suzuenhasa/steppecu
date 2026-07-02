// src/io/snp_reader.hpp
//
// Reader for the EIGENSTRAT .snp file: one whitespace-separated record per SNP
// parsed into per-SNP metadata (id, chromosome, genetic/physical position,
// ref/alt alleles). Pure host C++20 io-leaf — it surfaces the raw file values
// and derives nothing (no block ids, no re-polarization).
//
// Reference: docs/reference/src_io_snp_reader.hpp.md
#ifndef STEPPE_IO_SNP_READER_HPP
#define STEPPE_IO_SNP_READER_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace steppe::io {

// SnpTable — the per-SNP metadata — reference §3
struct SnpTable {
    std::vector<std::string> id;
    std::vector<int> chrom;
    std::vector<double> genpos_morgans;
    std::vector<double> physpos;
    std::vector<char> ref;
    std::vector<char> alt;
    std::size_t count = 0;
};

// read_snp — parsing the file — reference §4
[[nodiscard]] SnpTable read_snp(const std::string& path, std::size_t max_snps);

}  // namespace steppe::io

#endif  // STEPPE_IO_SNP_READER_HPP
