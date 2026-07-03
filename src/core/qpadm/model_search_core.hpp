// src/core/qpadm/model_search_core.hpp
//
// The model->device shard planner for the multi-GPU model search: given a batch
// of models and a device count, it hands each device a contiguous slice. Host-
// only, pure integer index math with no CUDA, so it is unit-testable without a
// GPU.
//
// Reference: docs/reference/src_core_qpadm_model_search_core.hpp.md
#ifndef STEPPE_CORE_QPADM_MODEL_SEARCH_CORE_HPP
#define STEPPE_CORE_QPADM_MODEL_SEARCH_CORE_HPP

#include <cstddef>
#include <vector>

namespace steppe::core::qpadm {

// Contiguous model->device shard — reference §2
struct ModelShard {
    int         g;
    std::size_t lo;
    std::size_t hi;
};

// Balanced-by-count shard plan over G devices — reference §3
[[nodiscard]] std::vector<ModelShard> plan_model_shards(std::size_t n_models,
                                                        std::size_t G);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_MODEL_SEARCH_CORE_HPP
