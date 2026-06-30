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
/// PRECONDITION: `P >= 1, M >= 1` — a zero/negative extent yields a zero-extent
/// grid the CUDA driver rejects (cudaErrorInvalidConfiguration). The caller
/// (compute_f2) guards the degenerate case with an empty-result early return
/// (cleanup E-3/B12), so this wrapper is never reached with `P<=0 || M<=0`.
void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream);

/// THE ONE predicate gating whether an EmulatedFp64 request can be HONORED as the
/// FIXED-slice Ozaki path (cleanup X-6/B2). True only for `EmulatedFp64` AND a
/// build with the fixed-slice tuning API (STEPPE_HAVE_EMU_TUNING). Without the
/// tuning, cuBLAS would engage emulation under its DYNAMIC mantissa-control default
/// — the rejected ~60-bit parity trap (architecture.md §12; ROADMAP §0) — so this
/// returns false and the policy downgrades to native Fp64 (architecture.md §9
/// build() "fall back to native Fp64 or error"). BOTH `engage_f2_precision` (math
/// mode) and `f2_compute_type` (compute type) consult this, so the two halves of
/// the precision decision can never disagree. Exposed so callers/tests can observe
/// the honorability state directly without depending on the STEPPE_HAVE_EMU_TUNING
/// macro being visible in their TU (the macro is PRIVATE to steppe_device).
[[nodiscard]] bool emulation_honorable(const Precision& precision) noexcept;

/// Engage the precision policy on the handle for the f2 GEMMs (architecture.md
/// §12). For an HONORABLE EmulatedFp64 request (`emulation_honorable`) this is the
/// load-bearing cublasSet* sequence (FP64-emulated math mode + EAGER strategy +
/// FIXED mantissa control at `precision.mantissa_bits`); for Fp64 — and for an
/// EmulatedFp64 request the build cannot honor — it sets PEDANTIC native math and
/// emits a one-shot capability-tagged warning on the downgrade (cleanup X-6/B2),
/// so the path NEVER silently runs cuBLAS's DYNAMIC-default emulation. Exposed
/// (rather than inlined in run_f2_gemms) so the M4 batched/grouped path
/// (f2_batched_kernel.cu) engages the SAME policy ONCE before its group loop instead
/// of re-deriving it — the single source of the precision engagement (§2 DRY).
void engage_f2_precision(cublasHandle_t handle, const Precision& precision);

/// Map the typed Precision to the cuBLAS compute type for the f2 GEMMs. Routes the
/// EmulatedFp64-honorability decision through `emulation_honorable` so the compute
/// type and the math mode are always derived together (honorable EmulatedFp64 ⇒
/// CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT; Fp64, Tf32, AND an unhonorable
/// EmulatedFp64 request ⇒ native CUBLAS_COMPUTE_64F). Shared by the single-block
/// and batched f2 GEMM paths.
[[nodiscard]] cublasComputeType_t f2_compute_type(const Precision& precision);

/// The three f2 GEMMs (architecture.md §5 S2 line 236-240; spike run_f2_gemms),
/// all column-major, into pre-allocated outputs:
///   dG     [P  × P] = Q · Qᵀ
///   dVpair [P  × P] = V · Vᵀ   (RETAINED — the S4 jackknife weight)
///   dR     [2P × P] = S · Vᵀ   (top P rows = Σp², bottom P rows = Σhc)
/// `precision` governs ONLY these GEMMs (architecture.md §12): an HONORABLE
/// EmulatedFp64 request ⇒ FIXED-slice Ozaki at `precision.mantissa_bits` (the
/// cublasSet* sequence, engaged under STEPPE_HAVE_EMU_TUNING); Fp64, Tf32, AND an
/// EmulatedFp64 request the build cannot honor ⇒ native CUBLAS_COMPUTE_64F (the
/// last with a one-shot capability tag; cleanup X-6/B2). The handle must already be created (RAII,
/// architecture.md §7) and have its stream + emulated-FP64 determinism workspace
/// bound ONCE via CublasHandle::set_stream/set_workspace (architecture.md §12;
/// cleanup X-1/B1). This routine deliberately takes NO stream and never calls
/// `cublasSetStream` — doing so would reset the workspace to the default pool
/// (cuBLAS §2.4.7) and defeat the §12 determinism guarantee.
/// PRECONDITION: `P >= 1, M >= 1` — `M` becomes the GEMM contraction extent `k`,
/// and cuBLAS requires `k >= 0` (k==0 would be a degenerate no-op leaving dG/dR
/// unpopulated). The caller (compute_f2) guards `P<=0 || M<=0` with an empty
/// result before this is reached (cleanup E-3/B12).
void run_f2_gemms(cublasHandle_t handle, const Precision& precision,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR);

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
