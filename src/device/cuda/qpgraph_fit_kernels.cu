// src/device/cuda/qpgraph_fit_kernels.cu
//
// GPU kernels + host launchers that fit a qpGraph model: an admixture-graph
// topology whose edge lengths and admixture weights best explain observed
// f-statistics. One GPU thread runs a whole multistart optimizer per restart;
// the host launches once and reads back only per-restart {theta, score}.
//
// Reference: docs/reference/src_device_cuda_qpgraph_fit_kernels.cu.md
#include "device/cuda/qpgraph_fit_kernels.cuh"

#include "core/internal/launch_config.hpp"
#include "core/qpadm/qpgraph_opt_constants.hpp"
#include "device/cuda/check.cuh"
#include "device/cuda/device_buffer.cuh"

#include <stdexcept>
#include <string>

namespace steppe::device {

namespace {

namespace opt = ::steppe::core::qpadm::qpgraph_opt;

// Native-FP64 dense LU solve (d_lu_factor/d_solve) — reference §9
__device__ inline bool d_lu_factor(double* a, int n, int* piv) {
    for (int i = 0; i < n; ++i) piv[i] = i;
    for (int k = 0; k < n; ++k) {
        int p = k;
        double best = fabs(a[k + n * k]);
        for (int i = k + 1; i < n; ++i) {
            const double v = fabs(a[i + n * k]);
            if (v > best) { best = v; p = i; }
        }
        if (best == 0.0) return false;
        if (p != k) {
            for (int j = 0; j < n; ++j) {
                const double t = a[k + n * j]; a[k + n * j] = a[p + n * j]; a[p + n * j] = t;
            }
            const int tp = piv[k]; piv[k] = piv[p]; piv[p] = tp;
        }
        const double inv_pivot = 1.0 / a[k + n * k];
        for (int i = k + 1; i < n; ++i) {
            const double f = a[i + n * k] * inv_pivot;
            a[i + n * k] = f;
            for (int j = k + 1; j < n; ++j) a[i + n * j] -= f * a[k + n * j];
        }
    }
    return true;
}
__device__ inline bool d_solve(const double* A, int n, const double* b, double* x,
                               double* lu, int* piv, double* y) {
    for (int i = 0; i < n * n; ++i) lu[i] = A[i];
    if (!d_lu_factor(lu, n, piv)) return false;
    for (int i = 0; i < n; ++i) y[i] = b[piv[i]];
    for (int i = 0; i < n; ++i) {
        double s = y[i];
        for (int j = 0; j < i; ++j) s -= lu[i + n * j] * y[j];
        y[i] = s;
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < n; ++j) s -= lu[i + n * j] * x[j];
        x[i] = s / lu[i + n * i];
    }
    return true;
}

// Box-constrained edge solve (Lawson-Hanson NNLS) — reference §9
__device__ inline bool d_nnls(const double* A, const double* q, int n, double* bl,
                              double* w, int* P_idx, double* Ap, double* qp, double* z,
                              double* lu, double* y, int* piv, int* passive) {
    for (int i = 0; i < n; ++i) { bl[i] = 0.0; passive[i] = 0; }
    const double eps = 1e-12;
    const int max_outer = 3 * n + 30;
    for (int outer = 0; outer < max_outer; ++outer) {
        for (int i = 0; i < n; ++i) {
            double s = q[i];
            for (int j = 0; j < n; ++j) s -= A[i + n * j] * bl[j];
            w[i] = s;
        }
        int t = -1; double best = eps;
        for (int i = 0; i < n; ++i)
            if (!passive[i] && w[i] > best) { best = w[i]; t = i; }
        if (t < 0) return true;
        passive[t] = 1;
        for (int inner = 0; inner < max_outer; ++inner) {
            int p = 0;
            for (int i = 0; i < n; ++i) if (passive[i]) P_idx[p++] = i;
            for (int a = 0; a < p; ++a) {
                qp[a] = q[P_idx[a]];
                for (int b = 0; b < p; ++b) Ap[a + p * b] = A[P_idx[a] + n * P_idx[b]];
            }
            if (!d_solve(Ap, p, qp, z, lu, piv, y)) return false;
            bool all_pos = true;
            for (int a = 0; a < p; ++a) if (z[a] <= eps) { all_pos = false; break; }
            if (all_pos) {
                for (int i = 0; i < n; ++i) bl[i] = 0.0;
                for (int a = 0; a < p; ++a) bl[P_idx[a]] = z[a];
                break;
            }
            double alpha = 1.0;
            for (int a = 0; a < p; ++a) {
                if (z[a] <= eps) {
                    const double blp = bl[P_idx[a]];
                    const double denom = blp - z[a];
                    if (denom > eps) { const double r = blp / denom; if (r < alpha) alpha = r; }
                }
            }
            for (int a = 0; a < p; ++a) {
                const int idx = P_idx[a];
                bl[idx] += alpha * (z[a] - bl[idx]);
            }
            for (int i = 0; i < n; ++i)
                if (passive[i] && bl[i] <= eps) { passive[i] = 0; bl[i] = 0.0; }
        }
    }
    return true;
}

// Centered path-weight fill — reference §8
__device__ inline void d_fill_pwts_centered(const QpGraphDeviceTopo& t, const double* theta,
                                            double* pwts_c, double* path_w) {
    const int ne = t.nedge_norm, np = t.npop;
    for (int p = 0; p < t.npath; ++p) path_w[p] = 1.0;
    for (int k = 0; k < t.n_pae; ++k) {
        const int pi = t.pae_path[k];
        const int id = t.pae_admixedge[k];
        const int j = (id - 1) / 2;
        const double w = theta[j];
        const double v = (id & 1) ? w : (1.0 - w);
        path_w[pi] *= v;
    }
    int col = 0;
    for (int leaf = 0; leaf < np; ++leaf) {
        if (leaf == t.base_leaf) continue;
        for (int e = 0; e < ne; ++e)
            pwts_c[e + ne * col] = t.pwts0[e + ne * leaf] - t.pwts0[e + ne * t.base_leaf];
        ++col;
    }
    for (int a = 0; a < t.n_pe; ++a) {
        const int e = t.pe_edge[a], leaf = t.pe_leaf[a];
        bool first = true;
        for (int b = 0; b < a; ++b)
            if (t.pe_edge[b] == e && t.pe_leaf[b] == leaf) { first = false; break; }
        if (!first) continue;
        double sum = 0.0;
        for (int b = a; b < t.n_pe; ++b)
            if (t.pe_edge[b] == e && t.pe_leaf[b] == leaf) sum += path_w[t.pe_path[b]];
        const double oldv = t.pwts0[e + ne * leaf];
        const double delta = sum - oldv;
        if (delta == 0.0) continue;
        if (leaf == t.base_leaf) {
            for (int c = 0; c < np - 1; ++c) pwts_c[e + ne * c] -= delta;
        } else {
            int cc2 = leaf < t.base_leaf ? leaf : leaf - 1;
            pwts_c[e + ne * cc2] += delta;
        }
    }
}

// The GLS objective — reference §7
__device__ inline double d_qpgraph_score(const QpGraphDeviceTopo& t, const double* theta,
                                         const double* f_obs, const double* qinv,
                                         const ScratchLayout& L, double* DB, int* IB) {
    const int ne = t.nedge_norm, np = t.npair;
    double* pwts_c = DB + L.pwts_c;
    double* ppwts  = DB + L.ppwts;
    double* Wm     = DB + L.Wm;
    double* cc     = DB + L.cc;
    double* ccs    = DB + L.ccs;
    double* sc     = DB + L.sc;
    double* q1     = DB + L.q1;
    double* bl     = DB + L.bl;
    double* res    = DB + L.res;
    double* path_w = DB + L.path_w;
    double* qf     = DB + L.qf;

    d_fill_pwts_centered(t, theta, pwts_c, path_w);
    for (int k = 0; k < np; ++k) {
        const int ca = t.cmb1[k], cb = t.cmb2[k];
        for (int e = 0; e < ne; ++e)
            ppwts[k + np * e] = pwts_c[e + ne * ca] * pwts_c[e + ne * cb];
    }
    for (int e = 0; e < ne; ++e)
        for (int r = 0; r < np; ++r) {
            double acc = 0.0;
            for (int c = 0; c < np; ++c) acc += qinv[r + np * c] * ppwts[c + np * e];
            Wm[r + np * e] = acc;
        }
    for (int r = 0; r < np; ++r) {
        double a = 0.0;
        for (int c = 0; c < np; ++c) a += qinv[r + np * c] * f_obs[c];
        qf[r] = a;
    }
    for (int e1 = 0; e1 < ne; ++e1) {
        for (int e2 = 0; e2 < ne; ++e2) {
            double acc = 0.0;
            for (int r = 0; r < np; ++r) acc += ppwts[r + np * e1] * Wm[r + np * e2];
            cc[e1 + ne * e2] = acc;
        }
        double rr = 0.0;
        for (int r = 0; r < np; ++r) rr += ppwts[r + np * e1] * qf[r];
        q1[e1] = rr;
    }
    double trm = 0.0;
    for (int e = 0; e < ne; ++e) trm += cc[e + ne * e];
    const double ridge = t.fudge * trm / static_cast<double>(ne);
    for (int e = 0; e < ne; ++e) cc[e + ne * e] += ridge;
    for (int e = 0; e < ne; ++e) sc[e] = sqrt(cc[e + ne * e]);
    for (int e1 = 0; e1 < ne; ++e1) {
        q1[e1] = q1[e1] / sc[e1];
        for (int e2 = 0; e2 < ne; ++e2) ccs[e1 + ne * e2] = cc[e1 + ne * e2] / (sc[e1] * sc[e2]);
    }
    double* nn_w  = DB + L.nn_w;
    double* nn_Ap = DB + L.nn_Ap;
    double* nn_qp = DB + L.nn_qp;
    double* nn_z  = DB + L.nn_z;
    double* nn_lu = DB + L.nn_lu;
    double* nn_y  = DB + L.nn_y;
    int* nn_P    = IB + L.nn_P;
    int* nn_piv  = IB + L.nn_piv;
    int* nn_pass = IB + L.nn_pass;
    if (t.constrained) {
        if (!d_nnls(ccs, q1, ne, bl, nn_w, nn_P, nn_Ap, nn_qp, nn_z, nn_lu, nn_y, nn_piv, nn_pass))
            return 1e30;
    } else {
        if (!d_solve(ccs, ne, q1, bl, nn_lu, nn_piv, nn_y)) return 1e30;
    }
    for (int e = 0; e < ne; ++e) bl[e] = bl[e] / sc[e];
    for (int r = 0; r < np; ++r) {
        double pb = 0.0;
        for (int e = 0; e < ne; ++e) pb += ppwts[r + np * e] * bl[e];
        res[r] = f_obs[r] - pb;
    }
    double score = 0.0;
    for (int aa = 0; aa < np; ++aa) {
        double row = 0.0;
        for (int bb = 0; bb < np; ++bb) row += qinv[aa + np * bb] * res[bb];
        score += res[aa] * row;
    }
    return score;
}

// Deterministic multistart theta init — reference §6
__device__ inline double d_init_theta(unsigned inst, int dim) {
    unsigned long long z = (static_cast<unsigned long long>(inst) * opt::kSplitmixInstMul) +
                           (static_cast<unsigned long long>(dim) * opt::kSplitmixDimMul) +
                           opt::kSplitmixSeedInc;
    z = (z ^ (z >> 30)) * opt::kSplitmixMix1;
    z = (z ^ (z >> 27)) * opt::kSplitmixMix2;
    z = z ^ (z >> 31);
    return static_cast<double>(z & opt::kMantissaMask) / opt::kMantissaDiv;
}
__device__ inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

// The projected-Newton per-restart fit — reference §10
__device__ inline double d_fit_one_restart(const QpGraphDeviceTopo& t, unsigned inst, int maxit,
                                           double tol, const double* f_obs, const double* qinv,
                                           const ScratchLayout& L, double* DB, int* IB,
                                           double* th) {
    const int D = t.nadmix;
    if (D > kMaxThetaDev) return 1e30;
    for (int d = 0; d < D; ++d) th[d] = clamp01(d_init_theta(inst, d));
    double s = d_qpgraph_score(t, th, f_obs, qinv, L, DB, IB);
    if (D == 0) return s;
    const double h = opt::kFdStep;
    for (int it = 0; it < maxit; ++it) {
        double max_dx = 0.0, max_ds = 0.0;
        for (int d = 0; d < D; ++d) {
            const double w = th[d];
            const double wp = fmin(1.0, w + h), wm = fmax(0.0, w - h);
            double thp[kMaxThetaDev], thm[kMaxThetaDev];
            for (int k = 0; k < D; ++k) { thp[k] = th[k]; thm[k] = th[k]; }
            thp[d] = wp; thm[d] = wm;
            const double sp = d_qpgraph_score(t, thp, f_obs, qinv, L, DB, IB);
            const double sm = d_qpgraph_score(t, thm, f_obs, qinv, L, DB, IB);
            const double dwp = wp - w, dwm = w - wm;
            double g, curv;
            if (dwp > 0.0 && dwm > 0.0) {
                g = (sp - sm) / (dwp + dwm);
                curv = (sp - 2.0 * s + sm) / (opt::kCurvHalf * (dwp + dwm) * (dwp + dwm) + opt::kCurvGuard);
            } else if (dwp > 0.0) { g = (sp - s) / dwp; curv = 1.0; }
            else { g = (s - sm) / dwm; curv = 1.0; }
            double step = (curv > opt::kCurvThresh) ? (g / curv) : (g * opt::kGradStepScale);
            if (step > opt::kTrustClamp) step = opt::kTrustClamp;
            if (step < -opt::kTrustClamp) step = -opt::kTrustClamp;
            double wn = clamp01(w - step);
            double thn[kMaxThetaDev];
            for (int k = 0; k < D; ++k) thn[k] = th[k];
            thn[d] = wn;
            double sn = d_qpgraph_score(t, thn, f_obs, qinv, L, DB, IB);
            int bt = 0;
            while (sn > s && bt < opt::kMaxBacktrack) {
                wn = opt::kBacktrackHalf * (wn + w); thn[d] = wn;
                sn = d_qpgraph_score(t, thn, f_obs, qinv, L, DB, IB); ++bt;
            }
            const double dx = fabs(wn - w), ds = fabs(sn - s);
            if (sn <= s) { th[d] = wn; s = sn; }
            if (dx > max_dx) max_dx = dx;
            if (ds > max_ds) max_ds = ds;
        }
        if (max_dx < tol * opt::kTolDxScale && max_ds < tol * opt::kTolDsScale) break;
    }
    return s;
}

// The single-topology fleet kernel — reference §11
__global__ void qpgraph_fleet_kernel(QpGraphDeviceTopo t, int numstart, int maxit, double tol,
                                     const double* f_obs, const double* qinv,
                                     ScratchLayout L, double* g_dbl, int* g_int,
                                     double* out_theta, double* out_score) {
    const int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= numstart) return;
    const int D = t.nadmix;
    double* DB = g_dbl + static_cast<long>(inst) * L.dbl_total;
    int* IB = g_int + static_cast<long>(inst) * L.int_total;
    double th[kMaxThetaDev];
    const double s = d_fit_one_restart(t, static_cast<unsigned>(inst), maxit, tol, f_obs, qinv,
                                       L, DB, IB, th);
    const int Dw = D < kMaxThetaDev ? D : kMaxThetaDev;
    for (int d = 0; d < Dw; ++d) out_theta[static_cast<long>(inst) * D + d] = th[d];
    out_score[inst] = s;
}

// Evaluate at a chosen theta, exporting bl + f3_fit — reference §11
__global__ void qpgraph_eval_at_theta_kernel(QpGraphDeviceTopo t, const double* theta,
                                             const double* f_obs, const double* qinv,
                                             ScratchLayout L, double* g_dbl, int* g_int,
                                             double* out_bl, double* out_f3, double* out_score) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int ne = t.nedge_norm, np = t.npair;
    const double s = d_qpgraph_score(t, theta, f_obs, qinv, L, g_dbl, g_int);
    *out_score = s;
    const double* bl  = g_dbl + L.bl;
    const double* res = g_dbl + L.res;
    for (int e = 0; e < ne; ++e) out_bl[e] = bl[e];
    for (int r = 0; r < np; ++r) out_f3[r] = f_obs[r] - res[r];
}

// The heterogeneous-topology batch fleet kernel — reference §12
__global__ void qpgraph_fleet_batch_kernel(const QpGraphDeviceTopoView* views, int ntopo,
                                           int numstart, int maxit, double tol,
                                           const double* g_pwts0, const int* g_pe_edge,
                                           const int* g_pe_leaf, const int* g_pe_path,
                                           const int* g_pae_path, const int* g_pae_admixedge,
                                           const int* g_cmb1, const int* g_cmb2,
                                           const double* f_obs, const double* qinv,
                                           ScratchLayout Lmax, double* g_dbl, int* g_int,
                                           double* out_score) {
    const long inst = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long total = static_cast<long>(ntopo) * numstart;
    if (inst >= total) return;
    const int topo_id = static_cast<int>(inst / numstart);
    const int restart = static_cast<int>(inst % numstart);
    const QpGraphDeviceTopoView v = views[topo_id];

    QpGraphDeviceTopo t{};
    t.npop = v.npop; t.nedge_norm = v.nedge_norm; t.nadmix = v.nadmix; t.npair = v.npair;
    t.npath = v.npath; t.base_leaf = v.base_leaf; t.n_pe = v.n_pe; t.n_pae = v.n_pae;
    t.constrained = v.constrained; t.fudge = v.fudge;
    t.pwts0 = g_pwts0 + v.off_pwts0;
    t.pe_edge = g_pe_edge + v.off_pe; t.pe_leaf = g_pe_leaf + v.off_pe; t.pe_path = g_pe_path + v.off_pe;
    t.pae_path = g_pae_path + v.off_pae; t.pae_admixedge = g_pae_admixedge + v.off_pae;
    t.cmb1 = g_cmb1 + v.off_cmb; t.cmb2 = g_cmb2 + v.off_cmb;

    double* DB = g_dbl + inst * Lmax.dbl_total;
    int* IB = g_int + (Lmax.int_total > 0 ? inst * Lmax.int_total : 0);
    const ScratchLayout L = make_layout(t.npop, t.nedge_norm, t.npair, t.npath);

    double th[kMaxThetaDev];
    const double s = d_fit_one_restart(t, static_cast<unsigned>(restart), maxit, tol, f_obs,
                                       qinv, L, DB, IB, th);
    out_score[inst] = s;
}

}  // namespace

// launch_qpgraph_fleet — reference §11
void launch_qpgraph_fleet(const QpGraphDeviceTopo& topo, int numstart, int maxit,
                          double tol, const double* d_fobs, const double* d_qinv,
                          double* d_out_theta, double* d_out_score, cudaStream_t stream) {
    if (topo.nadmix > kMaxThetaDev)
        throw std::runtime_error(
            "steppe::device::launch_qpgraph_fleet: topology nadmix=" +
            std::to_string(topo.nadmix) + " exceeds the per-thread theta-stack cap kMaxThetaDev=" +
            std::to_string(kMaxThetaDev) + "; the in-kernel fit holds theta in fixed-size stack "
            "arrays of that length. Reduce the admixture-node count or raise kMaxThetaDev.");
    const ScratchLayout L = make_layout(topo.npop, topo.nedge_norm, topo.npair, topo.npath);
    DeviceBuffer<double> g_dbl(static_cast<std::size_t>(numstart) * static_cast<std::size_t>(L.dbl_total));
    DeviceBuffer<int> g_int(static_cast<std::size_t>(numstart) * static_cast<std::size_t>(L.int_total > 0 ? L.int_total : 1));
    const int TPB = 64;
    const int blocks = core::cdiv(numstart, TPB);
    qpgraph_fleet_kernel<<<blocks, TPB, 0, stream>>>(topo, numstart, maxit, tol, d_fobs, d_qinv,
                                                     L, g_dbl.data(), g_int.data(),
                                                     d_out_theta, d_out_score);
    STEPPE_CUDA_CHECK_KERNEL();
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// launch_qpgraph_eval_at_theta — reference §11
void launch_qpgraph_eval_at_theta(const QpGraphDeviceTopo& topo, const double* d_theta,
                                  const double* d_fobs, const double* d_qinv,
                                  double* d_out_bl, double* d_out_f3, double* d_out_score,
                                  cudaStream_t stream) {
    if (topo.nadmix > kMaxThetaDev)
        throw std::runtime_error(
            "steppe::device::launch_qpgraph_eval_at_theta: topology nadmix=" +
            std::to_string(topo.nadmix) + " exceeds the per-thread theta-stack cap kMaxThetaDev=" +
            std::to_string(kMaxThetaDev) + " (the qpGraph fit's nadmix precondition).");
    const ScratchLayout L = make_layout(topo.npop, topo.nedge_norm, topo.npair, topo.npath);
    DeviceBuffer<double> g_dbl(static_cast<std::size_t>(L.dbl_total));
    DeviceBuffer<int> g_int(static_cast<std::size_t>(L.int_total > 0 ? L.int_total : 1));
    qpgraph_eval_at_theta_kernel<<<1, 1, 0, stream>>>(topo, d_theta, d_fobs, d_qinv, L,
                                                      g_dbl.data(), g_int.data(),
                                                      d_out_bl, d_out_f3, d_out_score);
    STEPPE_CUDA_CHECK_KERNEL();
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// launch_qpgraph_fleet_batch — reference §12
void launch_qpgraph_fleet_batch(const QpGraphDeviceTopoView* d_views, int ntopo,
                                int numstart, int maxit, double tol, int dbl_per_thread,
                                int int_per_thread, const double* d_pwts0,
                                const int* d_pe_edge, const int* d_pe_leaf, const int* d_pe_path,
                                const int* d_pae_path, const int* d_pae_admixedge,
                                const int* d_cmb1, const int* d_cmb2,
                                const double* d_fobs, const double* d_qinv,
                                double* d_g_dbl, int* d_g_int, double* d_out_score,
                                cudaStream_t stream) {
    ScratchLayout Lmax{};
    Lmax.dbl_total = dbl_per_thread;
    Lmax.int_total = int_per_thread;
    const long total = static_cast<long>(ntopo) * static_cast<long>(numstart);
    const int TPB = 64;
    const long blocks = core::cdiv(total, static_cast<long>(TPB));
    qpgraph_fleet_batch_kernel<<<static_cast<unsigned>(blocks), TPB, 0, stream>>>(
        d_views, ntopo, numstart, maxit, tol, d_pwts0, d_pe_edge, d_pe_leaf, d_pe_path,
        d_pae_path, d_pae_admixedge, d_cmb1, d_cmb2, d_fobs, d_qinv, Lmax, d_g_dbl, d_g_int,
        d_out_score);
    STEPPE_CUDA_CHECK_KERNEL();
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace steppe::device
