// src/io/snp_major_tile.hpp
//
// SnpMajorTile — the host-only io-leaf struct carrying a chunk of a SNP-major
// genotype source (raw bytes + the individual selection) up to the app layer,
// which runs the on-device transpose that turns it into a canonical
// individual-major GenotypeTile.
//
// Reference: docs/reference/src_io_snp_major_tile.hpp.md
#ifndef STEPPE_IO_SNP_MAJOR_TILE_HPP
#define STEPPE_IO_SNP_MAJOR_TILE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace steppe::io {

// SnpMajorTile (raw SNP-major tile + selection) — reference §4
struct SnpMajorTile {
    std::vector<std::uint8_t> snp_major;

    std::size_t src_bytes_per_record = 0;

    std::size_t n_snp = 0;

    std::vector<std::size_t> sel_rows;

    std::size_t n_individuals = 0;

    std::vector<std::size_t> pop_offsets;

    std::vector<std::string> pop_labels;

    [[nodiscard]] std::size_t n_pop() const noexcept { return pop_labels.size(); }
};

}  // namespace steppe::io

#endif  // STEPPE_IO_SNP_MAJOR_TILE_HPP
