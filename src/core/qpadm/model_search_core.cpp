// src/core/qpadm/model_search_core.cpp — the S8 model->device shard planner.

#include "core/qpadm/model_search_core.hpp"

#include <stdexcept>

namespace steppe::core::qpadm {

std::vector<ModelShard> plan_model_shards(std::size_t n_models, std::size_t G) {
    if (G == 0) {
        throw std::runtime_error("plan_model_shards: G must be >= 1");
    }
    // Count-balanced contiguous tiling: the first (n % G) devices get ceil(n/G),
    // the rest get floor(n/G). This covers [0, n_models) with no gaps/overlaps in
    // g=0..G-1 order — the fixed, deterministic order the host re-sort relies on.
    const std::size_t base = n_models / G;   // floor
    const std::size_t rem = n_models % G;    // the first `rem` devices get +1

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
    // Post-condition: the last cursor lands exactly on n_models (full cover).
    return shards;
}

}  // namespace steppe::core::qpadm
