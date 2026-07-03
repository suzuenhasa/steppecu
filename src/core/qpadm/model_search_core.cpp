// src/core/qpadm/model_search_core.cpp — the S8 model->device shard planner.
//
// Count-balanced contiguous tiling covers [0, n_models) in fixed g=0..G-1 order
// with no gaps or overlaps — the deterministic order the host re-sort relies on.

#include "core/qpadm/model_search_core.hpp"

#include <stdexcept>

namespace steppe::core::qpadm {

std::vector<ModelShard> plan_model_shards(std::size_t n_models, std::size_t G) {
    if (G == 0) {
        throw std::runtime_error("plan_model_shards: G must be >= 1");
    }
    const std::size_t base = n_models / G;
    const std::size_t rem = n_models % G;

    std::vector<ModelShard> shards;
    shards.reserve(G);
    std::size_t cursor = 0;
    for (std::size_t g = 0; g < G; ++g) {
        const std::size_t count = base + (g < rem ? 1 : 0);
        const std::size_t lo = cursor;
        const std::size_t hi = cursor + count;
        shards.push_back(ModelShard{static_cast<int>(g), lo, hi});
        cursor = hi;
    }
    return shards;
}

}  // namespace steppe::core::qpadm
