// src/core/qpadm/f4_matrix.hpp
//
// S3 driver (design docs/design/fit-engine.md §1.3): resolve the model's
// population indices and dispatch the per-block f4 assembly through the CUDA-free
// ComputeBackend seam (assemble_f4). HOST-PURE, CUDA-FREE. Two thin entries — one
// for the device-resident DeviceF2Blocks (the GPU-first primary) and one for the
// host F2BlockTensor (the M(fit-1) oracle/parity path). The actual four-slab AT2
// combine lives in the backend (CpuBackend reference / CudaBackend in M(fit-4)).
#ifndef STEPPE_CORE_QPADM_F4_MATRIX_HPP
#define STEPPE_CORE_QPADM_F4_MATRIX_HPP

#include <span>

#include "device/backend.hpp"  // ComputeBackend, F4Blocks
#include "steppe/config.hpp"   // Precision
#include "steppe/fstats.hpp"   // F2BlockTensor

namespace steppe {
namespace device { class DeviceF2Blocks; }
namespace core::qpadm {

/// S3 over device-resident f2 (zero D2H on the CUDA path).
[[nodiscard]] inline F4Blocks assemble_f4(ComputeBackend& be,
                                          const device::DeviceF2Blocks& f2,
                                          std::span<const int> left_idx,
                                          std::span<const int> right_idx,
                                          const Precision& precision) {
    return be.assemble_f4(f2, left_idx, right_idx, precision);
}

/// S3 over a host f2 tensor (the oracle/parity door).
[[nodiscard]] inline F4Blocks assemble_f4(ComputeBackend& be, const F2BlockTensor& f2,
                                          std::span<const int> left_idx,
                                          std::span<const int> right_idx,
                                          const Precision& precision) {
    return be.assemble_f4(f2, left_idx, right_idx, precision);
}

}  // namespace core::qpadm
}  // namespace steppe

#endif  // STEPPE_CORE_QPADM_F4_MATRIX_HPP
