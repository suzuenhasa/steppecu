// src/core/qpadm/model_search_core.hpp
//
// M(fit-6) S8 ROTATION shard planner — HOST-PURE, CUDA-FREE, GPU-FREE-TESTABLE.
// The ONLY thing here is the deterministic model->device shard plan: pure index
// math over (n_models, G). No CUDA, no Resources device calls, so it is unit-
// testable against a fake backend with no GPU (the proven test_f2_blocks_multigpu
// pattern). The re-sort itself is IMPLICIT and needs no code: each worker writes
// only results[models[i].model_index] into a vector pre-sized to n_models before
// any thread starts, so the pre-sized-slot write IS the deterministic re-sort
// (the same discipline as f2_blocks_multigpu_core's pre-sized partials[g]).
#ifndef STEPPE_CORE_QPADM_MODEL_SEARCH_CORE_HPP
#define STEPPE_CORE_QPADM_MODEL_SEARCH_CORE_HPP

#include <cstddef>
#include <vector>

namespace steppe::core::qpadm {

/// Contiguous-range model->device shard plan over G devices: device g owns the
/// models in [lo, hi). CONTIGUOUS (not round-robin) so each device's batched
/// dispatch sees a dense model range it can pack into one batched arena (design
/// §1.2). Balanced by count: the first (n % G) devices get ceil(n/G), the rest
/// floor(n/G). G must be >= 1. An empty shard (lo == hi) is a valid no-op slot for
/// a device that drew nothing (n < G). The plan covers [0, n_models) with no gaps
/// and no overlaps, in g=0..G-1 order — the same fixed order the f2 combine uses.
struct ModelShard {
    int         g;   ///< the device index 0..G-1.
    std::size_t lo;  ///< first model (inclusive) into models[].
    std::size_t hi;  ///< one-past-last model (exclusive) into models[].
};

[[nodiscard]] std::vector<ModelShard> plan_model_shards(std::size_t n_models,
                                                        std::size_t G);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_MODEL_SEARCH_CORE_HPP
