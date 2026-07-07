// src/io/individual_partition.hpp
//
// The per-individual selection helper READv2 needs: unlike read_ind (which keys the
// partition on the GroupID/pop column and collapses individuals into pops), this reads
// the .ind/.fam and emits one SINGLETON PopGroup per retained sample, labelled by its
// Genetic ID (the sample identity), in genotype-record order. That singleton index
// space is exactly what the per-pair all-pairs sweep enumerates. Pure host C++20 io-leaf.
#ifndef STEPPE_IO_INDIVIDUAL_PARTITION_HPP
#define STEPPE_IO_INDIVIDUAL_PARTITION_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "io/eigenstrat_format.hpp"
#include "io/ind_reader.hpp"

namespace steppe::io {

// Read a per-individual singleton partition.
//   format             : the on-disk genotype format (selects .ind vs .fam id column)
//   path               : the .ind (EIGENSTRAT family) or .fam (PLINK) sidecar
//   samples            : optional set of Genetic IDs to restrict to; nullopt = all present
//   n_records_present  : genotype records actually present (rows past this are ignored)
// Each retained row becomes a singleton PopGroup{ label = Genetic ID, rows = {row} } in
// row order. Throws std::runtime_error on: an unreadable file, a requested --samples ID
// not found, or a duplicate Genetic ID among the retained samples (fail-fast — a
// duplicate would make the name->index resolver ambiguous).
[[nodiscard]] IndPartition read_individual_partition(
    GenoFormat format, const std::string& path,
    const std::optional<std::vector<std::string>>& samples, std::size_t n_records_present);

}  // namespace steppe::io

#endif  // STEPPE_IO_INDIVIDUAL_PARTITION_HPP
