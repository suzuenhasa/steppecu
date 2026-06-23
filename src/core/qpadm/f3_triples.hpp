// src/core/qpadm/f3_triples.hpp
//
// run_f3 S3 driver (fit-engine §6): dispatch the per-triple f3 assembly through the
// CUDA-free ComputeBackend::assemble_f3_triples seam. HOST-PURE, CUDA-FREE — the SIBLING
// of f4_quartets.hpp (the run_f4 assemble driver), reusing the SAME pattern with the
// three-slab TRIPLE identity instead of the four-slab quartet. Two thin entries: one for
// the device-resident DeviceF2Blocks (the GPU-first primary, zero D2H) and one for the host
// F2BlockTensor (the oracle/parity door). The per-triple three-slab combine lives in the
// backend (CpuBackend reference / CudaBackend GPU path).
#ifndef STEPPE_CORE_QPADM_F3_TRIPLES_HPP
#define STEPPE_CORE_QPADM_F3_TRIPLES_HPP

#include <span>

#include "device/backend.hpp"  // ComputeBackend, F4Blocks (the generic per-block X carrier)
#include "steppe/config.hpp"   // Precision
#include "steppe/fstats.hpp"   // F2BlockTensor

namespace steppe {
namespace device { class DeviceF2Blocks; }
namespace core::qpadm {

/// run_f3 S3 over device-resident f2 (zero D2H on the CUDA path). `triples` is the
/// flattened P-axis index triple array (length 3*N, triple k = {p1=C,p2=A,p3=B}).
[[nodiscard]] inline F4Blocks assemble_f3_triples(ComputeBackend& be,
                                                  const device::DeviceF2Blocks& f2,
                                                  std::span<const int> triples,
                                                  const Precision& precision) {
    return be.assemble_f3_triples(f2, triples, precision);
}

/// run_f3 S3 over a host f2 tensor (the oracle/parity door).
[[nodiscard]] inline F4Blocks assemble_f3_triples(ComputeBackend& be,
                                                  const F2BlockTensor& f2,
                                                  std::span<const int> triples,
                                                  const Precision& precision) {
    return be.assemble_f3_triples(f2, triples, precision);
}

}  // namespace core::qpadm
}  // namespace steppe

#endif  // STEPPE_CORE_QPADM_F3_TRIPLES_HPP
