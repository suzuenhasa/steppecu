// src/io/ploidy_detect.hpp
//
// Per-sample ploidy auto-detection (AT2 adjust_pseudohaploid): classifies each
// gathered sample in a genotype tile as diploid (2) or pseudo-haploid (1). A leaf
// io header — pure host C++20, no CUDA.
//
// Reference: docs/reference/src_io_ploidy_detect.hpp.md
#ifndef STEPPE_IO_PLOIDY_DETECT_HPP
#define STEPPE_IO_PLOIDY_DETECT_HPP

#include <vector>

#include "io/genotype_tile.hpp"

namespace steppe::io {

// detect_sample_ploidy — reference §5
[[nodiscard]] std::vector<int> detect_sample_ploidy(const GenotypeTile& tile);

}  // namespace steppe::io

#endif  // STEPPE_IO_PLOIDY_DETECT_HPP
