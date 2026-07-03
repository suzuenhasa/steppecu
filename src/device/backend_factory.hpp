// src/device/backend_factory.hpp
//
// The single place the two ComputeBackend factories — CPU reference and GPU — are
// declared. CUDA-free by contract: names only the abstract ComputeBackend interface,
// so core, the CLI, and the tests can construct either backend without the CUDA toolkit.
//
// Reference: docs/reference/src_device_backend_factory.hpp.md
#ifndef STEPPE_DEVICE_BACKEND_FACTORY_HPP
#define STEPPE_DEVICE_BACKEND_FACTORY_HPP

#include <memory>

#include "device/backend.hpp"

namespace steppe::device {

// CPU reference backend factory — reference §4
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend();

// GPU backend factory, bound to one device — reference §5
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(int device_id = 0);

// Visible CUDA device count, no backend needed — reference §6
[[nodiscard]] int visible_device_count();

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_BACKEND_FACTORY_HPP
