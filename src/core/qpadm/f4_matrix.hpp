// src/core/qpadm/f4_matrix.hpp
//
// Two thin, host-pure overloads that forward per-block f4 assembly through the
// ComputeBackend seam — one over device-resident f2, one over a host f2 tensor.
// The actual four-slab combine lives in the backend, not here.
#ifndef STEPPE_CORE_QPADM_F4_MATRIX_HPP
#define STEPPE_CORE_QPADM_F4_MATRIX_HPP

#include <span>

#include "device/backend.hpp"
#include "steppe/config.hpp"
#include "steppe/fstats.hpp"

namespace steppe {
namespace device { class DeviceF2Blocks; }
namespace core::qpadm {

[[nodiscard]] inline F4Blocks assemble_f4(ComputeBackend& be,
                                          const device::DeviceF2Blocks& f2,
                                          std::span<const int> left_idx,
                                          std::span<const int> right_idx,
                                          const Precision& precision) {
    return be.assemble_f4(f2, left_idx, right_idx, precision);
}

[[nodiscard]] inline F4Blocks assemble_f4(ComputeBackend& be, const F2BlockTensor& f2,
                                          std::span<const int> left_idx,
                                          std::span<const int> right_idx,
                                          const Precision& precision) {
    return be.assemble_f4(f2, left_idx, right_idx, precision);
}

}  // namespace core::qpadm
}  // namespace steppe

#endif  // STEPPE_CORE_QPADM_F4_MATRIX_HPP
