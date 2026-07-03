// src/core/qpadm/model_search.hpp
//
// Public face of the qpAdm model-space search: declares the shared per-model
// fit body. Host-pure and CUDA-free — it names only the CUDA-free ComputeBackend
// seam, so it links without a GPU toolchain in scope.
//
// Reference: docs/reference/src_core_qpadm_model_search.hpp.md
#ifndef STEPPE_CORE_QPADM_MODEL_SEARCH_HPP
#define STEPPE_CORE_QPADM_MODEL_SEARCH_HPP

#include <span>
#include <vector>

#include "device/backend.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::core::qpadm {

// fit_one_model_device, the shared per-model fit body — reference §3
[[nodiscard]] QpAdmResult fit_one_model_device(ComputeBackend& be,
                                               const device::DeviceF2Blocks& f2,
                                               const QpAdmModel& model,
                                               const QpAdmOptions& opts);

}  // namespace steppe::core::qpadm

#endif  // STEPPE_CORE_QPADM_MODEL_SEARCH_HPP
