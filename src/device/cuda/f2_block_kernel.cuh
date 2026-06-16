// src/device/cuda/f2_block_kernel.cuh
//
// Narrow launch-wrapper declarations for the f2 3-GEMM reformulation
// (architecture.md §5 S2, §7 "host code never includes kernel bodies or
// <<<>>>"; ROADMAP M0/M4). Host orchestration (cuda_backend.cu) calls these
// `void launch_*` / `run_*` functions; the kernel bodies and `<<<>>>` live only
// in f2_block_kernel.cu.
//
// This header names CUDA types (cublasHandle_t) and so is PRIVATE to
// steppe_device (architecture.md §4) — it is the device-internal seam between
// the backend and the kernel TU, not the CUDA-free public ComputeBackend seam
// (that is device/backend.hpp).
#ifndef STEPPE_DEVICE_CUDA_F2_BLOCK_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_F2_BLOCK_KERNEL_CUH

#include <cublas_v2.h>

#include "steppe/config.hpp"  // steppe::Precision

namespace steppe::device {

/// FUSED elementwise pre-pass over one decoded SNP tile (architecture.md §5 S2
/// "single fused sweep", §11.3): from the Q/V/N contract (raw reference-allele
/// frequency, validity, non-missing haploid count, all column-major [P × M])
/// produce the three GEMM-input matrices in ONE sweep, never materializing the
/// [SNP × pop × pop] intermediate:
///   dQ_masked [P  × M] : Q⊙V             (zero-filled where invalid)
///   dV_out    [P  × M] : 1.0 valid / 0.0 missing
///   dS        [2P × M] : [ Qsq ; Hc ] stacked (lda = 2P), where
///                        Qsq = Q_masked², Hc = q(1-q)/max(N-1,floor)·V
/// Hc uses the SHARED `het_correction` primitive (core/internal/f2_estimator.hpp)
/// so the CPU oracle and this feeder cannot diverge on the formula
/// (architecture.md §13). All native FP64. `M` is `long` (SNP-count scale).
void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream);

/// The three f2 GEMMs (architecture.md §5 S2 line 236-240; spike run_f2_gemms),
/// all column-major, into pre-allocated outputs:
///   dG     [P  × P] = Q · Qᵀ
///   dVpair [P  × P] = V · Vᵀ   (RETAINED — the S4 jackknife weight)
///   dR     [2P × P] = S · Vᵀ   (top P rows = Σp², bottom P rows = Σhc)
/// `precision` governs ONLY these GEMMs (architecture.md §12): EmulatedFp64 ⇒
/// FIXED-slice Ozaki at `precision.mantissa_bits` (the cublasSet* sequence,
/// engaged under STEPPE_HAVE_EMU_TUNING); Fp64 ⇒ native CUBLAS_COMPUTE_64F;
/// Tf32 ⇒ screening compute type. The handle must already be created (RAII,
/// architecture.md §7) and have its workspace set for emulated-FP64
/// determinism (architecture.md §12).
void run_f2_gemms(cublasHandle_t handle, const Precision& precision,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR,
                  cudaStream_t stream);

/// Fused numerator + divide (architecture.md §5 S2, §11.3), native FP64 in every
/// precision mode. Assembles f2(i,j) from the three GEMM outputs using the SHARED
/// `assemble_f2_numerator` / `finalize_f2` primitives (core/internal/
/// f2_estimator.hpp): this is the catastrophic-cancellation step, held native
/// FP64 regardless of `Precision` (architecture.md §12). Writes dF2 [P × P];
/// dVpair is the GEMM output, carried through unchanged.
void launch_assemble_f2(const double* dG, const double* dVpair, const double* dR,
                        double* dF2, int P, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_F2_BLOCK_KERNEL_CUH
