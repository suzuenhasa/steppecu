// src/core/stats/genotype_front_end.hpp
//
// The single shared genotype decode front-end: the one place the four genotype-path
// tools (f2 extract, genotype-path D-stat, qpfstats, DATES) turn a genotype triple on
// disk into the canonical individual-major tile plus its parsed SNP table and partition.
// It reads the inputs to a fixed boundary {tile, snptab, part, fmt, M0} and stops there;
// each caller decodes past that on its own.
//
// Reference: docs/reference/src_core_stats_genotype_front_end.hpp.md
#ifndef STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP
#define STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP

#include <cstddef>
#include <span>
#include <string>

#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe {

// Layering: CUDA-free compute seam, forward-declared — reference §6
class ComputeBackend;

namespace core {

// GenotypeFrontEnd result struct — reference §3
struct GenotypeFrontEnd {
    io::GenotypeTile tile;
    io::SnpTable snptab;
    io::IndPartition part;
    io::GenoFormat fmt = io::GenoFormat::Unknown;
    std::size_t M0 = 0;
};

// read_genotype_front_end: primary entry point — reference §4
[[nodiscard]] GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                                       const std::string& snp,
                                                       const std::string& ind,
                                                       const io::PopSelection& sel,
                                                       ComputeBackend& backend);

// pop_labels convenience overload — reference §5
[[nodiscard]] GenotypeFrontEnd read_genotype_front_end(const std::string& geno,
                                                       const std::string& snp,
                                                       const std::string& ind,
                                                       std::span<const std::string> pop_labels,
                                                       ComputeBackend& backend);

}  // namespace core
}  // namespace steppe

#endif  // STEPPE_CORE_STATS_GENOTYPE_FRONT_END_HPP
