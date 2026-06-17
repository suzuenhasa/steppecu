// src/device/backend_factory.hpp
//
// Backend FACTORIES — the single-source declarations of how a ComputeBackend is
// constructed (cleanup X-9/B8; architecture.md §8 DRY single-home, §9 Resources).
//
// THIS HEADER IS CUDA-FREE BY CONTRACT, like device/backend.hpp itself: it names
// only the CUDA-free `ComputeBackend` interface and `std::unique_ptr`, so `core`,
// the CLI, and the tests can construct either backend without pulling in the CUDA
// toolkit (CUDA is PRIVATE to steppe_device; architecture.md §4). A backend's
// CONSTRUCTION is part of its public contract and belongs next to the interface,
// not hand-declared at each call site — before this header every consumer (the
// reference tests + the M4.5 `Resources` wiring) forward-declared the factory
// prototype itself, in two different namespaces, so a signature change silently
// broke linkage/ODR with no single source (cleanup X-9/B8; device-backend §9.1).
//
// NAMESPACE: both factories live in `namespace steppe::device`. `make_cuda_backend`
// always did (cuda_backend.cu); `make_cpu_backend` and the `CpuBackend` class were
// MOVED here from `steppe::core` (they always compiled into steppe_device, so the
// old `steppe::core` placement was a namespace/layer mismatch — TODO §A; the two
// implementations of ONE interface now share ONE namespace). Both still return the
// abstract `ComputeBackend` so callers never name the concrete type.
#ifndef STEPPE_DEVICE_BACKEND_FACTORY_HPP
#define STEPPE_DEVICE_BACKEND_FACTORY_HPP

#include <memory>

#include "device/backend.hpp"  // steppe::ComputeBackend (the CUDA-free DI seam)

namespace steppe::device {

/// Construct the CPU REFERENCE backend (the long-double correctness oracle the GPU
/// is continuously diffed against; architecture.md §13). Returned as the abstract
/// `ComputeBackend` so callers depend only on the CUDA-free interface (the DI seam,
/// architecture.md §8; injected into `Resources`, §9). Defined in cpu_backend.cpp.
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend();

/// Construct the GPU backend (the 3-GEMM reformulation; architecture.md §9 —
/// backend chosen at build()). Returned as the abstract interface so `core` /
/// `Resources` never name the concrete type or touch a CUDA header (architecture
/// .md §4, §8). Defined in cuda_backend.cu.
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend();

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_BACKEND_FACTORY_HPP
