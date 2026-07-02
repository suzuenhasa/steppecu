// src/io/geno_reader.hpp
//
// GenoReader opens a packed .geno genotype file, parses its header, and serves
// tiled raw packed bytes — it reads and gathers bytes only; decoding the 2-bit
// codes happens later on the device. Pure host C++; failures surface as
// std::runtime_error.
//
// Reference: docs/reference/src_io_geno_reader.hpp.md
#ifndef STEPPE_IO_GENO_READER_HPP
#define STEPPE_IO_GENO_READER_HPP

#include <cstddef>
#include <string>

#include "io/eigenstrat_format.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_major_tile.hpp"

namespace steppe::io {

// GenoReader: individual-major vs SNP-major reading paths — reference §2
class GenoReader {
public:
    // Open + parse + size-validate; header()/records_present() accessors — reference §3
    explicit GenoReader(const std::string& geno_path);

    [[nodiscard]] const GenoHeader& header() const noexcept { return header_; }

    [[nodiscard]] std::size_t records_present() const noexcept { return records_present_; }

    // read_tile: TGENO individual-major reader — reference §5
    [[nodiscard]] GenotypeTile read_tile(const IndPartition& part,
                                         std::size_t snp_begin,
                                         std::size_t snp_end);

    // read_snp_major_tile: GENO / PACKEDANCESTRYMAP reader — reference §6
    [[nodiscard]] SnpMajorTile read_snp_major_tile(const IndPartition& part,
                                                   std::size_t snp_begin,
                                                   std::size_t snp_end);

    // read_eigenstrat_snp_major_tile: EIGENSTRAT reader — reference §7
    [[nodiscard]] SnpMajorTile read_eigenstrat_snp_major_tile(const IndPartition& part,
                                                             std::size_t snp_begin,
                                                             std::size_t snp_end);

    // read_plink_snp_major_tile: PLINK .bed reader — reference §8
    [[nodiscard]] SnpMajorTile read_plink_snp_major_tile(const IndPartition& part,
                                                        std::size_t snp_begin,
                                                        std::size_t snp_end);

    // read_ancestrymap_snp_major_tile: ANCESTRYMAP reader — reference §9
    [[nodiscard]] SnpMajorTile read_ancestrymap_snp_major_tile(const IndPartition& part,
                                                              std::size_t snp_begin,
                                                              std::size_t snp_end);

    // Ownership: move-only — reference §12
    GenoReader(const GenoReader&) = delete;
    GenoReader& operator=(const GenoReader&) = delete;
    GenoReader(GenoReader&&) noexcept = default;
    GenoReader& operator=(GenoReader&&) noexcept = default;
    ~GenoReader() = default;

private:
    // Private SNP-major helpers: build_selection + checked_alloc_snp_major — reference §11
    void build_selection(const IndPartition& part,
                         std::size_t src_bpr,
                         std::size_t tile_snps,
                         const char* reader_name,
                         SnpMajorTile& tile) const;

    void checked_alloc_snp_major(std::size_t tile_snps,
                                 std::size_t src_bpr,
                                 const char* reader_name,
                                 bool zero_init,
                                 SnpMajorTile& tile) const;

    std::string path_;
    GenoHeader header_;
    std::size_t records_present_ = 0;
};

}  // namespace steppe::io

#endif  // STEPPE_IO_GENO_READER_HPP
