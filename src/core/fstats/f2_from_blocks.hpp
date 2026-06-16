// src/core/fstats/f2_from_blocks.hpp
//
// Declaration of the `core`-owned f2 assembly orchestration (architecture.md §5
// S2; ROADMAP §2, M0). See f2_from_blocks.cpp for the rationale. Host-pure and
// CUDA-free: it names only the CUDA-free ComputeBackend seam and the shared
// Q/V/N views, so it compiles into `core` without dragging in the device toolkit
// (architecture.md §4 layering rule).
#ifndef STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP
#define STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP

#include "device/backend.hpp"       // steppe::ComputeBackend, steppe::F2Result
#include "core/internal/views.hpp"  // steppe::core::MatView (Q/V/N contract)
#include "steppe/config.hpp"        // steppe::Precision

namespace steppe::core {

/// Compute the bias-corrected f2 matrix + pairwise-valid counts for ONE SNP block
/// from the Q/V/N contract, dispatching through the injected `backend`
/// (architecture.md §5 S2, §8). The orchestration owns the policy; the backend
/// owns the implementation (CpuBackend oracle vs CudaBackend 3-GEMM). `precision`
/// governs only the GPU matmul-heavy GEMMs (architecture.md §12); the CPU oracle
/// ignores it. Returns f2 [P × P] + Vpair [P × P] (column-major), the latter
/// retained as the S4 jackknife weight (architecture.md §5 S2 caveat (a)).
[[nodiscard]] F2Result compute_f2_block(ComputeBackend& backend, const MatView& Q,
                                        const MatView& V, const MatView& N,
                                        const Precision& precision);

}  // namespace steppe::core

#endif  // STEPPE_CORE_FSTATS_F2_FROM_BLOCKS_HPP
