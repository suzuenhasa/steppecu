// src/core/internal/primary_backend.hpp
//
// primary_backend() — the single primary-GPU backend accessor. Collapses the
// per-call `*resources.gpus.at(0).backend` seam the qpAdm/qpGraph entry points
// each open-coded onto one helper. Names only CUDA-free seam types.
//
// Reference: docs/reference/src_core_internal_primary_backend.hpp.md
#ifndef STEPPE_CORE_INTERNAL_PRIMARY_BACKEND_HPP
#define STEPPE_CORE_INTERNAL_PRIMARY_BACKEND_HPP

#include <cstddef>

#include "device/backend.hpp"
#include "device/resources.hpp"

namespace steppe::device {

// Primary-GPU index — the fixed first device the single-GPU entry points fit on.
inline constexpr std::size_t kPrimaryGpu = 0;

[[nodiscard]] inline ComputeBackend& primary_backend(Resources& resources) {
    return *resources.gpus.at(kPrimaryGpu).backend;
}

}  // namespace steppe::device

#endif  // STEPPE_CORE_INTERNAL_PRIMARY_BACKEND_HPP
