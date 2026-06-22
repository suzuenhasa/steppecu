// src/core/qpadm/f4_quartets.hpp
//
// run_f4 S3 driver (fit-engine §6): dispatch the per-quartet f4 assembly through the
// CUDA-free ComputeBackend::assemble_f4_quartets seam. HOST-PURE, CUDA-FREE — the SIBLING
// of f4_matrix.hpp (the qpAdm assemble_f4 driver), reusing the SAME pattern. Two thin
// entries: one for the device-resident DeviceF2Blocks (the GPU-first primary, zero D2H)
// and one for the host F2BlockTensor (the oracle/parity door). The per-quartet four-slab
// combine lives in the backend (CpuBackend reference / CudaBackend GPU path).
#ifndef STEPPE_CORE_QPADM_F4_QUARTETS_HPP
#define STEPPE_CORE_QPADM_F4_QUARTETS_HPP

#include <span>

#include "device/backend.hpp"  // ComputeBackend, F4Blocks
#include "steppe/config.hpp"   // Precision
#include "steppe/fstats.hpp"   // F2BlockTensor

namespace steppe {
namespace device { class DeviceF2Blocks; }
namespace core::qpadm {

/// run_f4 S3 over device-resident f2 (zero D2H on the CUDA path). `quartets` is the
/// flattened P-axis index quad array (length 4*N, quad k = {p1,p2,p3,p4}).
[[nodiscard]] inline F4Blocks assemble_f4_quartets(ComputeBackend& be,
                                                   const device::DeviceF2Blocks& f2,
                                                   std::span<const int> quartets,
                                                   const Precision& precision) {
    return be.assemble_f4_quartets(f2, quartets, precision);
}

/// run_f4 S3 over a host f2 tensor (the oracle/parity door).
[[nodiscard]] inline F4Blocks assemble_f4_quartets(ComputeBackend& be,
                                                   const F2BlockTensor& f2,
                                                   std::span<const int> quartets,
                                                   const Precision& precision) {
    return be.assemble_f4_quartets(f2, quartets, precision);
}

}  // namespace core::qpadm
}  // namespace steppe

#endif  // STEPPE_CORE_QPADM_F4_QUARTETS_HPP
