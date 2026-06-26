// src/device/cuda/qpgraph_fit_kernels.cu
//
// The qpGraph FLEET kernel — the PRODUCTIZED IDEA-1 optimizer spike (bench_optimizers.cu
// idea1_fleet_kernel:373-432 + d_graph_gls_score:295-351), GENERALIZED from the fixed-
// nadmix hard-coded fill_pwts to an ARBITRARY topology via the device path-table model,
// and given the AT2 box-constrained (NNLS) edge solve the golden needs (the spike's
// unconstrained d_solve was insufficient — the golden has a boundary edge at 0).
//
// ONE THREAD per restart (the FLEET is the parallel axis; the settled IDEA-1 decision,
// optimizer-comparison.md). Each thread runs the WHOLE multistart x maxit projected-
// Newton loop in-kernel — per-dim forward-diff gradient + diagonal 3-point curvature,
// projected trust-clamped step, 8-step backtracking — each objective eval the full inner
// GLS pipeline (fill_pwts -> ppwts_2d -> opt_edge_lengths (the cc design, the trace-
// scaled ridge, the AT2 scaling, the native SPD / NNLS solve) -> res' Qinv res). NO host
// objective per iteration (the AT2 optim() host-loop trap designed out): the host launches
// ONCE and gets back only the per-restart {theta, score}.
//
// PRECISION: the inner SPD edge solve + the GLS quadratic form are NATIVE FP64 (the
// cancellation carve-out, matching the host oracle). The objective is in-thread and small
// (nedge<=the per-thread scratch cap); the EmulatedFp64 GEMM seam is the production-scale
// cc-assembly path (batched cuBLAS), not this in-thread fleet body — the same shape the
// spike measured GPU-bound (>=90% util).
//
// SCRATCH: each thread's working arrays (pwts_c, ppwts, cc, ...) live in a per-thread slab
// of GLOBAL memory the launcher allocates (numstart slabs) — sized to the topology at
// runtime, so the fit scales past the per-thread register/local envelope (production
// graphs have many edges). The arena ints (path tables) are read-only resident.
#include "device/cuda/qpgraph_fit_kernels.cuh"

#include "core/qpadm/qpgraph_opt_constants.hpp" // core::qpadm::qpgraph_opt — the splitmix + projected-Newton constant set (single-sourced with the CpuBackend oracle, §12/§13 parity). CUDA-FREE constexpr scalars: usable in __device__ code per CUDA C++ PG §5.3.
#include "device/cuda/check.cuh"
#include "device/cuda/device_buffer.cuh"

#include <stdexcept>  // std::runtime_error — the nadmix > kMaxThetaDev reject-before-launch
#include <string>     // std::to_string — the over-cap diagnostic message

namespace steppe::device::cuda {

namespace {

// The single-sourced splitmix + projected-Newton constant set, shared bit-for-bit with
// the CpuBackend oracle (qpgraph_fit_fleet). CUDA-free constexpr scalars of built-in
// integral / floating-point type — usable directly in __device__ code (CUDA C++ PG §5.3).
namespace opt = ::steppe::core::qpadm::qpgraph_opt;

// ---- per-thread native-FP64 linear algebra (mirrors the spike d_lu_factor/d_solve) ----
// Column-major A(i,j) at i + n*j.
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
// Solve A x = b (A col-major n x n) into x. lu/y/piv are caller scratch (length n*n,n,n).
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

// ---- per-thread NNLS (Lawson-Hanson active set; mirrors the host nnls_active_set) ----
// min 0.5 bl' A bl - bl' q  s.t. bl >= 0. A SPD (n x n col-major), q [n]. bl OUT [n].
// scratch: w[n], P_idx[n], Ap[n*n], qp[n], z[n], lu[n*n], y[n], piv[n], passive[n(int)].
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
        if (t < 0) return true;  // KKT satisfied
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

// (ScratchLayout + make_layout moved to qpgraph_fit_kernels.cuh — shared by the backend TU
//  for the batch-MAX slab sizing.)

// ---- device fill_pwts (the general path-table model; mirrors fill_pwts_centered) ----
// theta [nadmix] -> pwts_c [nedge x (npop-1)] col-major into scratch.
__device__ inline void d_fill_pwts_centered(const QpGraphDeviceTopo& t, const double* theta,
                                            double* pwts_c, double* path_w) {
    const int ne = t.nedge_norm, np = t.npop;
    for (int p = 0; p < t.npath; ++p) path_w[p] = 1.0;
    for (int k = 0; k < t.n_pae; ++k) {
        const int pi = t.pae_path[k];
        const int id = t.pae_admixedge[k];        // 1-based
        const int j = (id - 1) / 2;
        const double w = theta[j];
        const double v = (id & 1) ? w : (1.0 - w);
        path_w[pi] *= v;
    }
    // pwts (uncentered, full npop cols) = pwts0, overwrite the (edge,leaf) cells with the
    // path-weight sum. We compute centered directly: for each non-base leaf column, value
    // = pwts[e,leaf] - pwts[e,base]. We need pwts[e,base] too (base leaf may have varying
    // cells). Build a small per-thread accumulation by scanning the path-edge table.
    // Step 1: start pwts_c columns from pwts0 (centered: col = pwts0[:,leaf]-pwts0[:,base]).
    int col = 0;
    for (int leaf = 0; leaf < np; ++leaf) {
        if (leaf == t.base_leaf) continue;
        for (int e = 0; e < ne; ++e)
            pwts_c[e + ne * col] = t.pwts0[e + ne * leaf] - t.pwts0[e + ne * t.base_leaf];
        ++col;
    }
    // Step 2: the path-edge table overwrites cells of the UNCENTERED pwts. Recompute the
    // affected (edge,leaf) cell sums, then apply the delta (newval - pwts0) to the centered
    // columns (leaf column gets +delta; if the BASE leaf cell changes, ALL columns get
    // -delta_base). Accumulate per (edge,leaf) into a sum first.
    // Since n_pe is small we do a direct two-pass: for each table entry add path_w to a
    // running per-cell sum keyed by (e,leaf); we detect a new cell by (e,leaf) change in
    // the (path-ordered) table — but the table is path-ordered, not cell-grouped, so we
    // instead accumulate by re-scanning: for each unique cell, sum over its entries.
    // Simpler robust form: for each entry, the cell's final value is sum over ALL entries
    // with the same (e,leaf). We compute that lazily: mark the FIRST occurrence and sum.
    for (int a = 0; a < t.n_pe; ++a) {
        const int e = t.pe_edge[a], leaf = t.pe_leaf[a];
        // first occurrence of (e,leaf)? (scan earlier entries)
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
            // base cell changed: every centered column at this edge shifts by -delta.
            for (int c = 0; c < np - 1; ++c) pwts_c[e + ne * c] -= delta;
        } else {
            // map leaf -> centered column index.
            int cc2 = leaf < t.base_leaf ? leaf : leaf - 1;
            pwts_c[e + ne * cc2] += delta;
        }
    }
}

// ---- the GLS objective (mirrors core::qpadm::qpgraph_score) ----
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
    // ppwts_2d[k,e] = pwts_c[e,cmb1[k]] * pwts_c[e,cmb2[k]]  (col-major k + np*e).
    for (int k = 0; k < np; ++k) {
        const int ca = t.cmb1[k], cb = t.cmb2[k];
        for (int e = 0; e < ne; ++e)
            ppwts[k + np * e] = pwts_c[e + ne * ca] * pwts_c[e + ne * cb];
    }
    // Wm = qinv * ppwts  (np x ne);  qf = qinv * f_obs (np).
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
    // cc = ppwts' Wm  (ne x ne);  q = ppwts' qf (ne).  (q stored in q1 temporarily.)
    for (int e1 = 0; e1 < ne; ++e1) {
        for (int e2 = 0; e2 < ne; ++e2) {
            double acc = 0.0;
            for (int r = 0; r < np; ++r) acc += ppwts[r + np * e1] * Wm[r + np * e2];
            cc[e1 + ne * e2] = acc;
        }
        double rr = 0.0;
        for (int r = 0; r < np; ++r) rr += ppwts[r + np * e1] * qf[r];
        q1[e1] = rr;  // unscaled q for now
    }
    // AT2 ridge: diag(cc) += fudge * mean(diag(cc)).
    double trm = 0.0;
    for (int e = 0; e < ne; ++e) trm += cc[e + ne * e];
    const double ridge = t.fudge * trm / static_cast<double>(ne);
    for (int e = 0; e < ne; ++e) cc[e + ne * e] += ridge;
    // AT2 scaling: sc=sqrt(diag(cc)); q1/=sc; ccs=cc/(sc⊗sc).
    for (int e = 0; e < ne; ++e) sc[e] = sqrt(cc[e + ne * e]);
    for (int e1 = 0; e1 < ne; ++e1) {
        q1[e1] = q1[e1] / sc[e1];
        for (int e2 = 0; e2 < ne; ++e2) ccs[e1 + ne * e2] = cc[e1 + ne * e2] / (sc[e1] * sc[e2]);
    }
    // solve (constrained NNLS or unconstrained LU); bl <- scaled solution / sc.
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
    // res = f_obs - ppwts*bl ; score = res' qinv res.
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

// Deterministic-multistart splitmix theta init. The constants are single-sourced in
// core/qpadm/qpgraph_opt_constants.hpp so this body stays bit-identical to the CpuBackend
// oracle (init_theta in qpgraph_fit_fleet) for the §12/§13 parity diff.
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

// kMaxThetaDev (the per-thread theta-stack cap) is single-sourced in the .cuh — the host
// launchers read it to reject an over-cap topology before launch; the device path guards on
// it below (1e30 sentinel) so an oversized nadmix fails LOUDLY instead of overrunning th[].

// ---- the SHARED per-restart fit (the whole multistart-instance loop in-kernel) ----
// Runs ONE restart `inst` of topology `t`: deterministic splitmix theta init -> the
// projected-Newton loop (per-dim forward-diff gradient + diagonal curvature, projected
// trust-clamped step, 8-step backtracking). D==0 ⇒ a single objective eval (no theta).
// Writes th[] (length D) on return; returns the converged score. DB/IB are this thread's
// scratch slab; L is the layout (sized to the topology, or the batch-max — only the cell
// OFFSETS that this topology uses are read, so a batch-max slab is safe). The shared body
// of BOTH the single-topology fleet and the heterogeneous-topology batch fleet (no rebuild).
__device__ inline double d_fit_one_restart(const QpGraphDeviceTopo& t, unsigned inst, int maxit,
                                           double tol, const double* f_obs, const double* qinv,
                                           const ScratchLayout& L, double* DB, int* IB,
                                           double* th) {
    const int D = t.nadmix;
    // Device guard (defense in depth; the host launchers reject this before launch). An
    // over-cap nadmix would overrun the fixed th/thp/thm/thn[kMaxThetaDev] stack arrays and
    // silently corrupt adjacent locals — fail LOUDLY with the singular sentinel instead.
    if (D > kMaxThetaDev) return 1e30;
    for (int d = 0; d < D; ++d) th[d] = clamp01(d_init_theta(inst, d));
    double s = d_qpgraph_score(t, th, f_obs, qinv, L, DB, IB);
    if (D == 0) return s;  // pure tree: one eval, no theta axis (the in-kernel D==0 path).
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

// ---- the FLEET kernel (one thread per restart; productized idea1_fleet_kernel) ----
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
    // th[] is only valid for d<min(D,kMaxThetaDev): on the over-cap path d_fit_one_restart
    // returns 1e30 WITHOUT filling th, so clamp the readback to the stack length (the host
    // rejects over-cap before launch, so the goldens never take this branch).
    const int Dw = D < kMaxThetaDev ? D : kMaxThetaDev;
    for (int d = 0; d < Dw; ++d) out_theta[static_cast<long>(inst) * D + d] = th[d];
    out_score[inst] = s;
}

// ---- L3: evaluate the objective at a GIVEN theta and EXPORT bl + f3_fit ----------
// The qpGraph fit's edge_length + f3_fit at the WINNING restart's theta, recovered
// ON-DEVICE (was a host core::qpadm::qpgraph_score re-eval after the fleet — the
// per-fit host-objective re-eval, L3). Runs the IDENTICAL d_qpgraph_score body the
// fleet runs (the fleet's per-restart scratch is freed before the host can pick the
// winner, and the scratch bl/res after d_fit_one_restart is STALE w.r.t. the returned
// theta — the inner Newton loop's last objective eval is a probe, not the accepted
// step — so a clean single re-eval at the converged theta is required; doing it on the
// SAME device body is at least as faithful as the host re-eval it replaces, and drops
// the host objective entirely). Single thread (one fit). bl = DB[L.bl]; f3_fit[r] =
// ppwts·bl = f_obs[r] - res[r] (res = DB[L.res]). On a singular inner solve the score
// is 1e30 — the host maps that to NonSpdCovariance (out.status), bl/f3_fit zeroed.
// Native FP64 (the in-thread objective carve-out).
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
    for (int r = 0; r < np; ++r) out_f3[r] = f_obs[r] - res[r];  // f3_fit = ppwts·bl
}

// ---- the HETEROGENEOUS-TOPOLOGY FLEET kernel (the topology-search killer app) ----
// inst flattens (topo,restart): topo_id = inst/numstart, restart = inst%numstart. Each
// thread reconstructs its topology's QpGraphDeviceTopo from the packed-arena base pointers
// + its view offsets, then runs the IDENTICAL d_fit_one_restart inner loop. Per-thread
// scratch is the batch-MAX slab (Lmax). Output: out_score[inst] (per topo,restart) — the
// host reduces to the per-topology best + the global argmin (a reduction, NOT a fit).
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

    // reconstruct this topology's device view from the packed-arena bases + the offsets.
    QpGraphDeviceTopo t{};
    t.npop = v.npop; t.nedge_norm = v.nedge_norm; t.nadmix = v.nadmix; t.npair = v.npair;
    t.npath = v.npath; t.base_leaf = v.base_leaf; t.n_pe = v.n_pe; t.n_pae = v.n_pae;
    t.constrained = v.constrained; t.fudge = v.fudge;
    t.pwts0 = g_pwts0 + v.off_pwts0;
    t.pe_edge = g_pe_edge + v.off_pe; t.pe_leaf = g_pe_leaf + v.off_pe; t.pe_path = g_pe_path + v.off_pe;
    t.pae_path = g_pae_path + v.off_pae; t.pae_admixedge = g_pae_admixedge + v.off_pae;
    t.cmb1 = g_cmb1 + v.off_cmb; t.cmb2 = g_cmb2 + v.off_cmb;

    // per-thread scratch slab (the batch-MAX layout — only the cells this topology uses are
    // touched; make_layout offsets are computed at the batch max, so the slab is large
    // enough for every topology). Build a per-topology layout (its OWN offsets) for the
    // device objective, but index into the batch-MAX-sized slab.
    double* DB = g_dbl + inst * Lmax.dbl_total;
    int* IB = g_int + (Lmax.int_total > 0 ? inst * Lmax.int_total : 0);
    const ScratchLayout L = make_layout(t.npop, t.nedge_norm, t.npair, t.npath);

    double th[kMaxThetaDev];
    const double s = d_fit_one_restart(t, static_cast<unsigned>(restart), maxit, tol, f_obs,
                                       qinv, L, DB, IB, th);
    out_score[inst] = s;
}

}  // namespace

void launch_qpgraph_fleet(const QpGraphDeviceTopo& topo, int numstart, int maxit,
                          double tol, const double* d_fobs, const double* d_qinv,
                          double* d_out_theta, double* d_out_score, cudaStream_t stream) {
    if (topo.nadmix > kMaxThetaDev)
        throw std::runtime_error(
            "steppe::device::cuda::launch_qpgraph_fleet: topology nadmix=" +
            std::to_string(topo.nadmix) + " exceeds the per-thread theta-stack cap kMaxThetaDev=" +
            std::to_string(kMaxThetaDev) + "; the in-kernel fit holds theta in fixed-size stack "
            "arrays of that length. Reduce the admixture-node count or raise kMaxThetaDev.");
    const ScratchLayout L = make_layout(topo.npop, topo.nedge_norm, topo.npair, topo.npath);
    // per-thread global scratch (numstart slabs). Allocated here, freed at scope end.
    DeviceBuffer<double> g_dbl(static_cast<std::size_t>(numstart) * static_cast<std::size_t>(L.dbl_total));
    DeviceBuffer<int> g_int(static_cast<std::size_t>(numstart) * static_cast<std::size_t>(L.int_total > 0 ? L.int_total : 1));
    const int TPB = 64;
    const int blocks = (numstart + TPB - 1) / TPB;
    qpgraph_fleet_kernel<<<blocks, TPB, 0, stream>>>(topo, numstart, maxit, tol, d_fobs, d_qinv,
                                                     L, g_dbl.data(), g_int.data(),
                                                     d_out_theta, d_out_score);
    STEPPE_CUDA_CHECK(cudaGetLastError());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void launch_qpgraph_eval_at_theta(const QpGraphDeviceTopo& topo, const double* d_theta,
                                  const double* d_fobs, const double* d_qinv,
                                  double* d_out_bl, double* d_out_f3, double* d_out_score,
                                  cudaStream_t stream) {
    if (topo.nadmix > kMaxThetaDev)
        throw std::runtime_error(
            "steppe::device::cuda::launch_qpgraph_eval_at_theta: topology nadmix=" +
            std::to_string(topo.nadmix) + " exceeds the per-thread theta-stack cap kMaxThetaDev=" +
            std::to_string(kMaxThetaDev) + " (the qpGraph fit's nadmix precondition).");
    const ScratchLayout L = make_layout(topo.npop, topo.nedge_norm, topo.npair, topo.npath);
    DeviceBuffer<double> g_dbl(static_cast<std::size_t>(L.dbl_total));
    DeviceBuffer<int> g_int(static_cast<std::size_t>(L.int_total > 0 ? L.int_total : 1));
    qpgraph_eval_at_theta_kernel<<<1, 1, 0, stream>>>(topo, d_theta, d_fobs, d_qinv, L,
                                                      g_dbl.data(), g_int.data(),
                                                      d_out_bl, d_out_f3, d_out_score);
    STEPPE_CUDA_CHECK(cudaGetLastError());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void launch_qpgraph_fleet_batch(const QpGraphDeviceTopoView* d_views, int ntopo,
                                int numstart, int maxit, double tol, int dbl_per_thread,
                                int int_per_thread, const double* d_pwts0,
                                const int* d_pe_edge, const int* d_pe_leaf, const int* d_pe_path,
                                const int* d_pae_path, const int* d_pae_admixedge,
                                const int* d_cmb1, const int* d_cmb2,
                                const double* d_fobs, const double* d_qinv,
                                double* d_g_dbl, int* d_g_int, double* d_out_score,
                                cudaStream_t stream) {
    // The batch-MAX layout the per-thread slab is sized to (dbl_per_thread/int_per_thread are
    // the caller's batch-max make_layout totals — passed so the kernel indexes the slab with
    // the SAME stride the caller allocated). We re-derive dbl_total/int_total here from the
    // passed totals (the layout's per-topology offsets are recomputed in-kernel).
    ScratchLayout Lmax{};
    Lmax.dbl_total = dbl_per_thread;
    Lmax.int_total = int_per_thread;
    const long total = static_cast<long>(ntopo) * static_cast<long>(numstart);
    const int TPB = 64;
    const long blocks = (total + TPB - 1) / TPB;
    qpgraph_fleet_batch_kernel<<<static_cast<unsigned>(blocks), TPB, 0, stream>>>(
        d_views, ntopo, numstart, maxit, tol, d_pwts0, d_pe_edge, d_pe_leaf, d_pe_path,
        d_pae_path, d_pae_admixedge, d_cmb1, d_cmb2, d_fobs, d_qinv, Lmax, d_g_dbl, d_g_int,
        d_out_score);
    STEPPE_CUDA_CHECK(cudaGetLastError());
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace steppe::device::cuda
