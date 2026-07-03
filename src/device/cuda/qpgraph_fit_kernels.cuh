// src/device/cuda/qpgraph_fit_kernels.cuh
//
// GPU-side interface for fitting admixture-graph topologies (qpGraph): the
// host-callable launchers the CUDA backend calls, plus the flat data structures they
// pass to the kernels. The kernels + device-side objective live in the paired
// qpgraph_fit_kernels.cu, keeping the backend interface header CUDA-free.
//
// Reference: docs/reference/src_device_cuda_qpgraph_fit_kernels.cuh.md
#ifndef STEPPE_DEVICE_CUDA_QPGRAPH_FIT_KERNELS_CUH
#define STEPPE_DEVICE_CUDA_QPGRAPH_FIT_KERNELS_CUH

#include <cuda_runtime.h>

#include "core/internal/host_device.hpp"

namespace steppe::device {

// Per-thread theta cap — reference §3
constexpr int kMaxThetaDev = 16;

// QpGraphDeviceTopo: one topology, uploaded once — reference §4
struct QpGraphDeviceTopo {
    int npop, nedge_norm, nadmix, npair, npath, base_leaf;
    int n_pe, n_pae;
    int constrained;
    double fudge;
    const double* pwts0;
    const int* pe_edge;
    const int* pe_leaf;
    const int* pe_path;
    const int* pae_path;
    const int* pae_admixedge;
    const int* cmb1;
    const int* cmb2;
};

// ScratchLayout: per-thread scratch slab — reference §5
struct ScratchLayout {
    int pwts_c, ppwts, Wm, cc, ccs, sc, q1, bl, res, path_w, qf;
    int nn_w, nn_Ap, nn_qp, nn_z, nn_lu, nn_y;
    long dbl_total;
    int nn_P, nn_piv, nn_pass;
    long int_total;
};

// make_layout: build the per-thread scratch layout — reference §5
STEPPE_HD inline ScratchLayout make_layout(int npop, int nedge, int npair, int npath) {
    ScratchLayout L{};
    long o = 0;
    auto take = [&](int sz) { long s = o; o += sz; return s; };
    L.pwts_c = take(nedge * (npop - 1));
    L.ppwts  = take(npair * nedge);
    L.Wm     = take(npair * nedge);
    L.cc     = take(nedge * nedge);
    L.ccs    = take(nedge * nedge);
    L.sc     = take(nedge);
    L.q1     = take(nedge);
    L.bl     = take(nedge);
    L.res    = take(npair);
    L.path_w = take(npath);
    L.qf     = take(npair);
    L.nn_w   = take(nedge);
    L.nn_Ap  = take(nedge * nedge);
    L.nn_qp  = take(nedge);
    L.nn_z   = take(nedge);
    L.nn_lu  = take(nedge * nedge);
    L.nn_y   = take(nedge);
    L.dbl_total = o;
    long oi = 0;
    auto takei = [&](int sz) { long s = oi; oi += sz; return s; };
    L.nn_P    = takei(nedge);
    L.nn_piv  = takei(nedge);
    L.nn_pass = takei(nedge);
    L.int_total = oi;
    return L;
}

// launch_qpgraph_fleet: multistart fleet for a single topology — reference §6
void launch_qpgraph_fleet(const QpGraphDeviceTopo& topo, int numstart, int maxit,
                          double tol, const double* d_fobs, const double* d_qinv,
                          double* d_out_theta, double* d_out_score, cudaStream_t stream);

// launch_qpgraph_eval_at_theta: evaluate and export at a chosen theta — reference §7
void launch_qpgraph_eval_at_theta(const QpGraphDeviceTopo& topo, const double* d_theta,
                                  const double* d_fobs, const double* d_qinv,
                                  double* d_out_bl, double* d_out_f3, double* d_out_score,
                                  cudaStream_t stream);

// QpGraphDeviceTopoView: one topology inside a packed batch — reference §8
struct QpGraphDeviceTopoView {
    int npop, nedge_norm, nadmix, npair, npath, base_leaf;
    int n_pe, n_pae;
    int constrained;
    double fudge;
    long off_pwts0;
    long off_pe;
    long off_pae;
    long off_cmb;
};

// launch_qpgraph_fleet_batch: fitting many topologies in one launch — reference §9
void launch_qpgraph_fleet_batch(const QpGraphDeviceTopoView* d_views, int ntopo,
                                int numstart, int maxit, double tol, int dbl_per_thread,
                                int int_per_thread, const double* d_pwts0,
                                const int* d_pe_edge, const int* d_pe_leaf, const int* d_pe_path,
                                const int* d_pae_path, const int* d_pae_admixedge,
                                const int* d_cmb1, const int* d_cmb2,
                                const double* d_fobs, const double* d_qinv,
                                double* d_g_dbl, int* d_g_int, double* d_out_score,
                                cudaStream_t stream);

}  // namespace steppe::device

#endif  // STEPPE_DEVICE_CUDA_QPGRAPH_FIT_KERNELS_CUH
