// src/core/qpadm/f3_triples.hpp
//
// run_f3 assemble driver: two thin inline wrappers that forward to
// ComputeBackend::assemble_f3_triples — one over device-resident f2 blocks, one
// over a host f2 tensor. `triples` is the flattened 3*N P-axis index array
// (triple k = {p1=C, p2=A, p3=B}); the three-slab combine lives in the backend.
#ifndef STEPPE_CORE_QPADM_F3_TRIPLES_HPP
#define STEPPE_CORE_QPADM_F3_TRIPLES_HPP

#include <span>

#include "device/backend.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"

namespace steppe {
namespace device { class DeviceF2Blocks; }
namespace core::qpadm {

[[nodiscard]] inline F4Blocks assemble_f3_triples(ComputeBackend& be,
                                                  const device::DeviceF2Blocks& f2,
                                                  std::span<const int> triples,
                                                  const Precision& precision) {
    return be.assemble_f3_triples(f2, triples, precision);
}

[[nodiscard]] inline F4Blocks assemble_f3_triples(ComputeBackend& be,
                                                  const F2BlockTensor& f2,
                                                  std::span<const int> triples,
                                                  const Precision& precision) {
    return be.assemble_f3_triples(f2, triples, precision);
}

}  // namespace core::qpadm
}  // namespace steppe

#endif  // STEPPE_CORE_QPADM_F3_TRIPLES_HPP
