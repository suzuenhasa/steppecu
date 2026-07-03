// src/device/cuda/f2_block_kernel.cu
//
// GPU implementation of the pairwise f2 statistic: three library GEMMs framed
// by two custom kernels (feeder + assemble), so the [SNP×pop×pop] block is
// never materialized. Private to steppe_device; owns three entry points —
// launch_f2_feeder, run_f2_gemms, launch_assemble_f2.
//
// Reference: docs/reference/src_device_cuda_f2_block_kernel.cu.md
#include "device/cuda/f2_block_kernel.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include <atomic>
#include <limits>

#include "core/internal/f2_estimator.hpp"
#include "core/internal/log.hpp"
#include "device/cuda/check.cuh"

// STEPPE_HAVE_EMU_TUNING build switch — reference §6
#ifndef STEPPE_HAVE_EMU_TUNING
#define STEPPE_HAVE_EMU_TUNING 0
#endif

namespace steppe::device {

using core::assemble_f2_numerator;
using core::finalize_f2;
using core::het_correction;
using core::kF2StackedBlocks;

namespace {

// Feeder kernel: fused elementwise pre-pass — reference §3
__global__ void f2_feeder_kernel(const double* __restrict__ Q_raw,
                                 const double* __restrict__ V_raw,
                                 const double* __restrict__ N_raw,
                                 double* __restrict__ Q_masked,
                                 double* __restrict__ V_out,
                                 double* __restrict__ S,
                                 int P, long M) {
    const long s = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.y;
    const long i = static_cast<long>(blockIdx.y) * blockDim.y + threadIdx.x;
    if (i >= P || s >= M) return;

    const long Pl = static_cast<long>(P);
    const long idx = i + Pl * s;
    const long stack_idx = i + (kF2StackedBlocks * Pl) * s;

    const bool valid = (V_raw[idx] != 0.0);
    const double qraw = Q_raw[idx];
    const double q = valid ? qraw : 0.0;
    const double hc = het_correction(qraw, N_raw[idx], valid);

    Q_masked[idx] = q;
    V_out[idx] = valid ? 1.0 : 0.0;
    S[stack_idx] = q * q;
    S[Pl + stack_idx] = hc;
}

// Assemble kernel: fused numerator + divide — reference §5
__global__ void assemble_f2_kernel(const double* __restrict__ G,
                                   const double* __restrict__ Vpair,
                                   const double* __restrict__ R,
                                   double* __restrict__ f2,
                                   int P) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= P || j >= P) return;

    const size_t Pz = static_cast<size_t>(P);
    const size_t twoP = kF2StackedBlocks * Pz;
    const size_t si = static_cast<size_t>(i);
    const size_t sj = static_cast<size_t>(j);

    const double Gij = G[si + sj * Pz];
    const double vp = Vpair[si + sj * Pz];
    const double sumsq_i = R[si + sj * twoP];
    const double sumsq_j = R[sj + si * twoP];
    const double hsum_i = R[(Pz + si) + sj * twoP];
    const double hsum_j = R[(Pz + sj) + si * twoP];

    const double num =
        assemble_f2_numerator(sumsq_i, sumsq_j, Gij, hsum_i, hsum_j);
    f2[si + sj * Pz] = finalize_f2(num, vp);
}

}  // namespace

// emulation_honorable — the one EmulatedFp64 honorability predicate — reference §7
[[nodiscard]] bool emulation_honorable(const Precision& precision) noexcept {
    if (precision.kind != Precision::Kind::EmulatedFp64) return false;
#if STEPPE_HAVE_EMU_TUNING
    return true;
#else
    return false;
#endif
}

#if !STEPPE_HAVE_EMU_TUNING
namespace {

// One-time EmulatedFp64 downgrade notice — reference §7
void warn_emulated_fp64_downgraded_once() {
    static std::atomic_flag emitted = ATOMIC_FLAG_INIT;
    if (!emitted.test_and_set(std::memory_order_relaxed)) {
        STEPPE_LOG_WARN(
            "[capability] EmulatedFp64 requested but the FIXED-slice Ozaki tuning is "
            "unavailable (built without -DSTEPPE_HAVE_EMU_TUNING) -> downgraded to "
            "native Fp64 [tag: emu_tuning_unavailable]. The reported precision is "
            "native FP64, NOT emulated (architecture.md §9, §12; cleanup X-6/B2).");
    }
}

}  // namespace
#endif  // !STEPPE_HAVE_EMU_TUNING

// f2_compute_type — typed Precision → cuBLAS compute type — reference §8
cublasComputeType_t f2_compute_type(const Precision& precision) {
    if (emulation_honorable(precision)) return CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
    return CUBLAS_COMPUTE_64F;
}

// engage_f2_precision — typed Precision → cuBLAS math mode — reference §8
void engage_f2_precision(cublasHandle_t handle, const Precision& precision) {
    if (emulation_honorable(precision)) {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
#if STEPPE_HAVE_EMU_TUNING
        CUBLAS_CHECK(cublasSetEmulationStrategy(handle, CUBLAS_EMULATION_STRATEGY_EAGER));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(
            handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
        CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(
            handle, precision.mantissa_bits));
#endif
    } else {
#if !STEPPE_HAVE_EMU_TUNING
        if (precision.kind == Precision::Kind::EmulatedFp64)
            warn_emulated_fp64_downgraded_once();
#endif
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    }
}

// launch_f2_feeder — feeder launch wrapper + grid geometry — reference §9
void launch_f2_feeder(const double* dQ_raw, const double* dV_raw, const double* dN_raw,
                      double* dQ_masked, double* dV_out, double* dS,
                      int P, long M, cudaStream_t stream) {
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::cdiv(M, static_cast<long>(steppe::kCdivBlock))),
                    static_cast<unsigned>(core::grid_for(P)));
    f2_feeder_kernel<<<grid, block, 0, stream>>>(dQ_raw, dV_raw, dN_raw,
                                                 dQ_masked, dV_out, dS, P, M);
    STEPPE_CUDA_CHECK_KERNEL();
}

// run_f2_gemms — the three library GEMMs under the precision policy — reference §4
void run_f2_gemms(cublasHandle_t handle, const Precision& precision,
                  int P, long M,
                  const double* dQ, const double* dV, const double* dS,
                  double* dG, double* dVpair, double* dR) {
    engage_f2_precision(handle, precision);

    const cublasComputeType_t ct = f2_compute_type(precision);
    const double one = 1.0;
    const double zero = 0.0;
    STEPPE_ASSERT(M <= static_cast<long>(std::numeric_limits<int>::max()),
                  "run_f2_gemms: M exceeds INT_MAX; cublasGemmEx k is a 32-bit int "
                  "(guarded by CudaBackend::compute_f2, B22)");
    const int Mi = static_cast<int>(M);
    const int twoP = kF2StackedBlocks * P;

    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, Mi,
                              &one, dQ, CUDA_R_64F, P, dQ, CUDA_R_64F, P,
                              &zero, dG, CUDA_R_64F, P, ct, CUBLAS_GEMM_DEFAULT));

    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, P, P, Mi,
                              &one, dV, CUDA_R_64F, P, dV, CUDA_R_64F, P,
                              &zero, dVpair, CUDA_R_64F, P, ct, CUBLAS_GEMM_DEFAULT));

    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, twoP, P, Mi,
                              &one, dS, CUDA_R_64F, twoP, dV, CUDA_R_64F, P,
                              &zero, dR, CUDA_R_64F, twoP, ct, CUBLAS_GEMM_DEFAULT));
}

// launch_assemble_f2 — assemble launch wrapper + grid geometry — reference §9
void launch_assemble_f2(const double* dG, const double* dVpair, const double* dR,
                        double* dF2, int P, cudaStream_t stream) {
    const dim3 block(steppe::kCdivBlock, steppe::kCdivBlock);
    const dim3 grid(static_cast<unsigned>(core::grid_for(P)),
                    static_cast<unsigned>(core::grid_for(P)));
    assemble_f2_kernel<<<grid, block, 0, stream>>>(dG, dVpair, dR, dF2, P);
    STEPPE_CUDA_CHECK_KERNEL();
}

}  // namespace steppe::device
