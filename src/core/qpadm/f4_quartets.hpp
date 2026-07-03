// src/core/qpadm/f4_quartets.hpp
//
// Host-pure, CUDA-free dispatch seam: two thin assemble_f4_quartets entries that
// forward a batch of population quartets to a ComputeBackend — one over device-resident
// f2 (the GPU-first path) and one over a host f2 tensor (the parity oracle).
//
// Reference: docs/reference/src_core_qpadm_f4_quartets.hpp.md
#ifndef STEPPE_CORE_QPADM_F4_QUARTETS_HPP
#define STEPPE_CORE_QPADM_F4_QUARTETS_HPP

#include <span>

#include "device/backend.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"

namespace steppe {
namespace device { class DeviceF2Blocks; }
namespace core::qpadm {

// assemble_f4_quartets overloads — reference §2
[[nodiscard]] inline F4Blocks assemble_f4_quartets(ComputeBackend& be,
                                                   const device::DeviceF2Blocks& f2,
                                                   std::span<const int> quartets,
                                                   const Precision& precision) {
    return be.assemble_f4_quartets(f2, quartets, precision);
}

[[nodiscard]] inline F4Blocks assemble_f4_quartets(ComputeBackend& be,
                                                   const F2BlockTensor& f2,
                                                   std::span<const int> quartets,
                                                   const Precision& precision) {
    return be.assemble_f4_quartets(f2, quartets, precision);
}

}  // namespace core::qpadm
}  // namespace steppe

#endif  // STEPPE_CORE_QPADM_F4_QUARTETS_HPP
