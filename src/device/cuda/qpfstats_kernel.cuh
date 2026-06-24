// src/device/cuda/qpfstats_kernel.cuh — the narrow launch seam for the qpfstats
// smoothing-solve prep kernels (the CUDA-private bodies live in qpfstats_kernel.cu;
// architecture.md §7). The backend reaches them ONLY through these wrappers.
#ifndef STEPPE_DEVICE_CUDA_QPFSTATS_KERNEL_CUH
#define STEPPE_DEVICE_CUDA_QPFSTATS_KERNEL_CUH

#include <cuda_runtime.h>

namespace steppe::device {

/// Zero the non-finite (NaN/Inf) entries of the COLUMN-MAJOR [npopcomb × n_block] ymat IN
/// PLACE, and write the per-block NaN-comb-row COUNT into d_nan_per_block[n_block] (the AT2
/// ymat_chunk[nan]=0 + the k_i = sum(nan_i) per block). The shared-factor Dtrsm solve runs
/// over the zeroed ymat; a block whose count == npopcomb (all-NaN) yields RHS=0 ⇒ b=0
/// (the AT2 all-NaN→b=0 policy is exact via the zero RHS), and a block whose count == 0
/// (no NaN) is the shared no-downdate path. A block with 0 < count < npopcomb needs the
/// per-row downdate (handled host-side by re-solving those few blocks with a downdated A).
void launch_qpfstats_zero_nan_ymat(double* d_ymat, int npopcomb, int n_block,
                                   int* d_nan_per_block, cudaStream_t stream);

/// Add `ridge` to the diagonal of the COLUMN-MAJOR [n × n] matrix d_A IN PLACE
/// (A_shared = x'x + ridge·I). One thread per diagonal entry.
void launch_qpfstats_add_ridge_diag(double* d_A, int n, double ridge, cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPFSTATS_KERNEL_CUH
