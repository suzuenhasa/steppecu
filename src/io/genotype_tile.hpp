// src/io/genotype_tile.hpp
//
// GenotypeTile — the plain-data hand-off struct the `io` reader produces and the
// decode backend consumes: packed genotype bytes plus the layout descriptor and
// population partition a decoder needs (io reads and tiles; the backend decodes).
// TGENO individual-major packing — each gathered individual's SNP-prefix bytes,
// laid out population-contiguous. Host-only std::vector/POD, no CUDA.
//
// Reference: docs/reference/src_io_genotype_tile.hpp.md
#ifndef STEPPE_IO_GENOTYPE_TILE_HPP
#define STEPPE_IO_GENOTYPE_TILE_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace steppe::io {

// Decoded genotype tile — reference §3
struct GenotypeTile {
    std::vector<std::uint8_t> packed;

    std::size_t bytes_per_record = 0;

    std::size_t n_snp = 0;

    std::size_t n_individuals = 0;

    std::vector<std::size_t> pop_offsets;

    std::vector<std::string> pop_labels;

    // Per-sample ploidy — reference §5
    std::vector<int> sample_ploidy;

    // Population count P — reference §6
    [[nodiscard]] std::size_t n_pop() const noexcept { return pop_labels.size(); }
};

}  // namespace steppe::io

#endif  // STEPPE_IO_GENOTYPE_TILE_HPP
