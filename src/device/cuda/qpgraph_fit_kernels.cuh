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

/// The device theta-stack cap. The per-restart fit holds the admixture weights theta (and
/// its forward-diff perturbations thp/thm/thn) in FIXED-SIZE per-thread stack arrays of this
/// length — so `nadmix` MUST be `<= kMaxThetaDev`. The host launchers reject an over-cap
/// topology before launch (a clear throw) and the kernel returns a 1e30 sentinel score on
/// the over-cap path, so an oversized topology fails LOUDLY instead of silently overrunning
/// the stack. 16 comfortably covers production nadmix (admixture-node counts are small).
/// Single-source: both the header (host reject) and the kernel (device guard) read this.
constexpr int kMaxThetaDev = 16;

/// The flat device-arena pointers for the topology (uploaded ONCE per fit; the qpAdm
/// per-model index-arena pattern). All device pointers; the scalars are by value.
/// PRECONDITION: `nadmix <= kMaxThetaDev` (the per-thread theta stack cap) — enforced by the
/// host launchers (throw) + the kernel (1e30 sentinel); see kMaxThetaDev.
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

/// The per-thread scratch slab layout (all doubles except the int sub-arrays). Sized to the
/// topology (or the BATCH-MAX) via make_layout; offsets computed on the host + passed in.
/// In the header so the backend TU (qpgraph_fit_fleet_batch) can size the batch-MAX slab.
struct ScratchLayout {
    int pwts_c, ppwts, Wm, cc, ccs, sc, q1, bl, res, path_w, qf;  // double sub-array offsets
    int nn_w, nn_Ap, nn_qp, nn_z, nn_lu, nn_y;                    // NNLS / solve scratch
    int dbl_total;                                                // total doubles per thread
    int nn_P, nn_piv, nn_pass;                                    // int sub-array offsets
    int int_total;                                                // total ints per thread
};

/// Build the per-thread scratch layout for a topology of (npop,nedge,npair,npath). __host__
/// __device__: the host sizes the slab + the kernel reconstructs per-topology offsets.
__host__ __device__ inline ScratchLayout make_layout(int npop, int nedge, int npair, int npath) {
    ScratchLayout L{};
    int o = 0;
    auto take = [&](int sz) { int s = o; o += sz; return s; };
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
    int oi = 0;
    auto takei = [&](int sz) { int s = oi; oi += sz; return s; };
    L.nn_P    = takei(nedge);
    L.nn_piv  = takei(nedge);
    L.nn_pass = takei(nedge);
    L.int_total = oi;
    return L;
}

/// Launch the fleet: `numstart` restarts, ONE thread each, the WHOLE multistart x maxit
/// projected-Newton loop in-kernel (GPU-bound; NO host objective per iteration). f_obs
/// [npair] + qinv [npair*npair] col-major are device-resident. Per-restart outputs:
///   out_theta [numstart * nadmix], out_score [numstart].
/// (Best-of-restarts + the bracket + the final edge-length recovery are done host-side
/// over these small per-restart arrays by the caller.)
void launch_qpgraph_fleet(const QpGraphDeviceTopo& topo, int numstart, int maxit,
                          double tol, const double* d_fobs, const double* d_qinv,
                          double* d_out_theta, double* d_out_score, cudaStream_t stream);

/// L3: evaluate the objective at a GIVEN theta and EXPORT the edge lengths + f3_fit
/// ON-DEVICE (was a host core::qpadm::qpgraph_score re-eval at the winning restart's
/// theta — the per-fit host objective, L3). Runs the IDENTICAL d_qpgraph_score body
/// the fleet runs, ONCE, at `d_theta` (the host-selected best restart's converged
/// theta), and writes out_bl[nedge_norm] (the fitted drift edge lengths) + out_f3[npair]
/// (the fitted f3 = ppwts·bl = f_obs - res) + out_score[1] (the objective there; the
/// host maps a 1e30 singular score to NonSpdCovariance). f_obs[npair] + qinv[npair²]
/// col-major are device-resident. Single thread, native FP64 (the objective carve-out).
void launch_qpgraph_eval_at_theta(const QpGraphDeviceTopo& topo, const double* d_theta,
                                  const double* d_fobs, const double* d_qinv,
                                  double* d_out_bl, double* d_out_f3, double* d_out_score,
                                  cudaStream_t stream);

/// The per-topology VIEW into the PACKED batch arenas (the topology-search heterogeneous
/// fleet). Scalars by value; the array fields are BASE OFFSETS (element counts) into the
/// single packed device buffers (d_pwts0 / d_pe_* / d_pae_* / d_cmb*). The kernel
/// reconstructs a QpGraphDeviceTopo for topology g by adding the base device pointers to
/// these offsets — so G heterogeneous topologies live in ONE buffer + one index table.
/// PRECONDITION: `nadmix <= kMaxThetaDev` (same per-thread theta stack cap as
/// QpGraphDeviceTopo) — enforced by the batch host launcher (throw) + the kernel (1e30
/// sentinel); see kMaxThetaDev.
struct QpGraphDeviceTopoView {
    int npop, nedge_norm, nadmix, npair, npath, base_leaf;
    int n_pe, n_pae;
    int constrained;
    double fudge;
    long off_pwts0;     ///< element offset into d_pwts0  (len nedge_norm*npop).
    long off_pe;        ///< element offset into d_pe_edge/leaf/path (len n_pe each).
    long off_pae;       ///< element offset into d_pae_path/admixedge (len n_pae each).
    long off_cmb;       ///< element offset into d_cmb1/d_cmb2 (len npair each).
};

/// HETEROGENEOUS-TOPOLOGY FLEET launch: fit ALL `ntopo` packed topologies in ONE launch
/// over the flattened (topo,restart) axis (inst = blockIdx*blockDim+tid; topo_id =
/// inst/numstart; restart = inst%numstart). Every thread reads the SAME resident d_fobs/
/// d_qinv (the pop-set-bound basis). Per-thread scratch is sized to the BATCH-MAX layout
/// the caller computes (make_layout at the batch-max npop/nedge/npair/npath), passed via
/// `dbl_per_thread`/`int_per_thread`. D==0 (tree) topologies do a single in-kernel objective
/// eval (no theta axis) — the host-side D==0 carve-out is dropped. Output: the per-(topo,
/// restart) score d_out_score[ntopo*numstart] (the host reduces to the per-topology best +
/// the global-best argmin — a reduction, NOT a fit). The host owns out_score's [G*numstart].
void launch_qpgraph_fleet_batch(const QpGraphDeviceTopoView* d_views, int ntopo,
                                int numstart, int maxit, double tol, int dbl_per_thread,
                                int int_per_thread, const double* d_pwts0,
                                const int* d_pe_edge, const int* d_pe_leaf, const int* d_pe_path,
                                const int* d_pae_path, const int* d_pae_admixedge,
                                const int* d_cmb1, const int* d_cmb2,
                                const double* d_fobs, const double* d_qinv,
                                double* d_g_dbl, int* d_g_int, double* d_out_score,
                                cudaStream_t stream);

}  // namespace steppe::device::cuda

#endif  // STEPPE_DEVICE_CUDA_QPGRAPH_FIT_KERNELS_CUH
