// src/io/filter/filter_plan.hpp
//
// The FilterPlan struct: the resolved set of SNPs and samples a filtered read of one
// dataset keeps, plus the live in-tile thresholds. A plan, never an on-disk rewrite —
// the decisions are applied on the fly as each tile is read, in a single pass.
//
// Reference: docs/reference/src_io_filter_filter_plan.hpp.md
#ifndef STEPPE_IO_FILTER_FILTER_PLAN_HPP
#define STEPPE_IO_FILTER_FILTER_PLAN_HPP

#include <cstddef>
#include <vector>

#include "steppe/config.hpp"

namespace steppe::io::filter {

// The resolved filter plan — reference §5
struct FilterPlan {
    std::vector<bool> snp_keep;
    std::size_t n_snp_kept = 0;
    std::vector<std::size_t> kept_samples;
    FilterConfig in_tile;
};

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_FILTER_PLAN_HPP
