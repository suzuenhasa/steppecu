// src/io/filter/mind_prepass.hpp
//
// The streaming pre-pass behind the per-sample --mind missing-data filter: it
// counts each sample's non-missing calls across all SNPs and emits the kept-sample
// index set. --mind is an aggregate filter (it needs every SNP for a sample), so it
// gets its own light streaming pass. Host-only C++20, no CUDA — an io-leaf header.
//
// Reference: docs/reference/src_io_filter_mind_prepass.hpp.md
#ifndef STEPPE_IO_FILTER_MIND_PREPASS_HPP
#define STEPPE_IO_FILTER_MIND_PREPASS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "steppe/config.hpp"

namespace steppe::io::filter {

// MindPrepassInput: the packed-genotype view — reference §3
struct MindPrepassInput {
    const std::uint8_t* packed = nullptr;
    std::size_t bytes_per_record = 0;
    std::size_t n_snp = 0;
    std::size_t n_individuals = 0;
};

// MindSummary: the per-sample result — reference §4
struct MindSummary {
    std::vector<std::size_t> nonmissing;
    std::vector<double> missing_frac;
    std::vector<std::size_t> kept;
};

// run_mind_prepass: behavior and edge cases — reference §5
[[nodiscard]] MindSummary run_mind_prepass(const MindPrepassInput& in,
                                           const FilterConfig& cfg);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_MIND_PREPASS_HPP
