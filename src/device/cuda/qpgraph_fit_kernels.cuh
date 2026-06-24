// src/device/cuda/qpgraph_fit_kernels.cuh
//
// The qpGraph FLEET launch wrapper (the productized IDEA-1 optimizer spike). Declares
// the host-callable launcher the CudaBackend::qpgraph_fit_fleet override calls; the
// kernel + device-side objective live in qpgraph_fit_kernels.cu. CUDA-only TU (the §4
// layering: kernels in .cu, the seam in backend.hpp is CUDA-free).
#ifndef STEPPE_DEVICE_CUDA_QPGRAPH_FIT_KERNELS_CUH
#define STEPPE_DEVICE_CUDA_QPGRAPH_FIT_KERNELS_CUH

#include <cuda_runtime.h>

namespace steppe::device::cuda {

/// The flat device-arena pointers for the topology (uploaded ONCE per fit; the qpAdm
/// per-model index-arena pattern). All device pointers; the scalars are by value.
struct QpGraphDeviceTopo {
    int npop, nedge_norm, nadmix, npair, npath, base_leaf;
    int n_pe, n_pae;                 ///< path-edge / path-admixedge table lengths.
    int constrained;                 ///< 1 => the box-constrained (NNLS) edge solve.
    double fudge;                    ///< the cc trace-scaled ridge (AT2 `diag`).
    const double* pwts0;             ///< [nedge_norm x npop] col-major (device).
    const int* pe_edge;              ///< [n_pe] (device).
    const int* pe_leaf;              ///< [n_pe] (device).
    const int* pe_path;              ///< [n_pe] (device).
    const int* pae_path;             ///< [n_pae] (device).
    const int* pae_admixedge;        ///< [n_pae] (device).
    const int* cmb1;                 ///< [npair] centered-col pair index (device).
    const int* cmb2;                 ///< [npair] (device).
};

/// Launch the fleet: `numstart` restarts, ONE thread each, the WHOLE multistart x maxit
/// projected-Newton loop in-kernel (GPU-bound; NO host objective per iteration). f_obs
/// [npair] + qinv [npair*npair] col-major are device-resident. Per-restart outputs:
///   out_theta [numstart * nadmix], out_score [numstart].
/// (Best-of-restarts + the bracket + the final edge-length recovery are done host-side
/// over these small per-restart arrays by the caller.)
void launch_qpgraph_fleet(const QpGraphDeviceTopo& topo, int numstart, int maxit,
                          double tol, const double* d_fobs, const double* d_qinv,
                          double* d_out_theta, double* d_out_score, cudaStream_t stream);

}  // namespace steppe::device::cuda

#endif  // STEPPE_DEVICE_CUDA_QPGRAPH_FIT_KERNELS_CUH
