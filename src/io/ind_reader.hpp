// src/io/ind_reader.hpp
//
// Reader for the EIGENSTRAT .ind file: parses per-individual population labels
// into a pop→genotype-record-index partition. The .ind row order IS the genotype
// individual axis, so each row index is that individual's genotype record index.
//
// Reference: docs/reference/src_io_ind_reader.hpp.md
#ifndef STEPPE_IO_IND_READER_HPP
#define STEPPE_IO_IND_READER_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace steppe::io {

// Population selection modes — reference §3
struct PopSelection {
    enum class Mode {
        Explicit,
        AutoTopK,
        MinN
    };

    Mode mode = Mode::AutoTopK;

    std::size_t k = 0;

    std::size_t min_n = 1;

    std::vector<std::string> labels;
};

// One selected population — reference §4
struct PopGroup {
    std::string label;
    std::vector<std::size_t> rows;
};

// The parsed result — reference §5
struct IndPartition {
    std::vector<PopGroup> groups;
    std::size_t n_individuals_total = 0;
};

// Parse the file — reference §6
[[nodiscard]] IndPartition read_ind(const std::string& path,
                                    const PopSelection& sel,
                                    std::size_t n_records_present);

}  // namespace steppe::io

#endif  // STEPPE_IO_IND_READER_HPP
