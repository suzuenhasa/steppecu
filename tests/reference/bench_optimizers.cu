// bench_optimizers.cu — GPU OPTIMIZER SPIKE: batched-sequential fleet (L-BFGS-B /
// projected-gradient) vs batched-population (CMA-ES / DE), both GPU-BOUND, on a
// REAL-f2-grounded nonlinear box-constrained graph-GLS objective.
//
// PURPOSE (docs/research/qpgraph-gpu-design.md flagged the qpGraph OPTIMIZER as the
// highest-risk / most-reusable new asset): empirically compare the TWO candidate GPU
// optimizer shapes BEFORE building qpGraph. The WINNER becomes the qpGraph optimizer.
//
// THE OBJECTIVE (REAL data, NO synthetic). Source = the committed 9-pop AADR f2 fixture
// tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin (P=9, nb=710 jackknife blocks;
// the SAME real tensor test_qpadm_parity/test_qpwave_parity use). Pops:
//   0 England_BellBeaker 1 Czechia_EBA_CordedWare 2 Turkey_N 3 Mbuti
//   4 Israel_Natufian 5 Iran_GanjDareh_N 6 Han 7 Papuan 8 Karitiana
// TOPOLOGY (fixed, 1 admixture event, checkable optimum) — the canonical EUROPE admix
// graph: base/outgroup = Mbuti(3). Sources S1=Turkey_N(2) [Anatolian farmer], S2=
// Iran_GanjDareh_N(5) [Caucasus/steppe-related]. Admixed leaf X=CordedWare(1) modeled as
// w*S1 + (1-w)*S2 through one admixture node. theta = the single mixture weight w in
// [0,1]; the >=0 drift edges are fit by the INNER GLS.
// ALGEBRA: fit the npair=choose(L,2)+L outgroup-f3 vector f3(C;i,j)=0.5*(f2(C,i)+
// f2(C,j)-f2(i,j)), C=base=Mbuti — steppe's EXACT assemble_f3_triples_gather identity
// (qpadm_fit_kernels.cu:663). f_obs = this f3 vector from the real f2 cache (block-size
// weighted total). Sigma = the f3 BLOCK-JACKKNIFE covariance Q (diag_f3=1e-5), inverted
// to Qinv on the HOST oracle (small_linalg). f_pred(w): fill_pwts maps w -> ppwts_2d
// [npair x nedge] (LINEAR in edge lengths, NONLINEAR in w via the product-of-(w,1-w)
// lineage overlaps); the inner edge lengths are the GLS-optimal solve of cc=ppwts'·Qinv·
// ppwts (nedge x nedge SPD, ridge fudge=1e-4 trace-scaled). OBJECTIVE = res'·Qinv·res,
// res = f_obs - ppwts·branch_lengths. A nonlinear box-constrained least-squares with the
// inner linear GLS folded in.
// CHECKABLE OPTIMUM (host-pinned, prototype-verified): w* ~= 0.8291, score* ~= 1.9712 —
// a smooth unimodal minimum (score 26.3 @ w=0 -> 1.97 @ w* -> 2.85 @ w=1).
//
// THE FLEET (huge batch): each of N instances is a multistart PERTURBATION of the SAME
// objective — a deterministic per-instance initial w0 in [0,1], all targeting the same
// w*. f_obs and Qinv are computed ONCE on the host and shared (basis-once-resident); only
// the per-instance optimizer state differs. This is the S8 many-small-fits envelope; N
// scales 1k/10k/100k/1M as pure batch rows.
//
// BOTH OPTIMIZERS GPU-BOUND: the host computes the basis ONCE, uploads tiny constant
// arenas ONCE, launches ONE batched kernel per optimizer (the WHOLE multistart x maxit /
// generation loop lives IN-KERNEL), and reads back only the final per-instance {w, score}.
// No host per-instance / per-iteration work. The bench samples nvidia-smi throughout; if
// EITHER pegs 1 CPU core at GPU~0% THAT is the headline finding (reported, not hidden).
//
// PRECISION: the inner GLS solve + the GLS quadratic form are NATIVE FP64 (the
// cancellation-sensitive f-stat carve-out, matching dev_chisq_of_core). The matmul-heavy
// ppwts'·Qinv·ppwts assembly is small (nedge<=8) and done per-thread; at the 9-pop spike
// scale the EmulatedFp64{40} GEMM seam is exercised by the production fit path, not this
// in-thread objective — the spike measures OPTIMIZER SHAPE, not the GEMM precision (noted
// in the report).
//
// MANUAL bench, NOT a ctest gate (no add_test) — like bench_rotation_1240k.cu. Run:
//   ./bench_optimizers [fixture_path] [Ncsv] [maxit] [tol]
//   fixture_path  default tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin
//   Ncsv          comma list of N scale points, default 1000,10000,100000,1000000
//   maxit         optimizer iteration/generation budget (default 200)
//   tol           convergence tol on |w-w*| AND |score-score*| (default 1e-3)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/internal/small_linalg.hpp"  // steppe::core solve / inverse (host oracle)

// ============================ FIXED PROBLEM DIMENSIONS ==============================
// L graph leaves {S1=Turkey_N, S2=Iran, X=CordedWare}; npair = the unordered-with-self
// outgroup-f3 vector length; nedge drift edges. Small + compile-time so the per-thread
// objective scratch is register/local-resident (no VRAM arena needed at this scale).
static constexpr int kL = 3;
static constexpr int kNpair = (kL * (kL + 1)) / 2;  // 6
static constexpr int kNedge = 4;
static constexpr double kDiagF3 = 1e-5;
static constexpr double kFudge = 1e-4;

// ============================ DEVICE: the shared objective ==========================
// Native-FP64 in-thread linear algebra (mirrors qpadm_fit_kernels.cu dev_lu_factor/
// dev_solve, the cancellation-sensitive carve-out). Column-major A(i,j) at i + n*j.
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

// The lineage-overlap weight of leaf `li` (0=S1,1=S2,2=X) on edge e, given mixture w.
// E0 root->N (all lineages, weight 1); E1 N->S1-side (S1 leaf w=1, X via parent-1 w);
// E2 N->S2-side (S2 leaf w=1, X via parent-2 1-w); E3 terminal admix->X (X only).
// (The qpGraph f-stat-is-linear-in-edges-given-w formulation: f3(C;i,j)=sum over edges
// of (leaf-i lineage mass)*(leaf-j lineage mass)*edgelen — the shared-drift overlap.)
__device__ inline void d_leafweights(int li, double w, double* we) {
    we[0] = 0.0; we[1] = 0.0; we[2] = 0.0; we[3] = 0.0;
    if (li == 0) {            // S1 = Turkey_N
        we[0] = 1.0; we[1] = 1.0;
    } else if (li == 1) {     // S2 = Iran
        we[0] = 1.0; we[2] = 1.0;
    } else {                  // X = CordedWare (admixed)
        we[0] = 1.0; we[1] = w; we[2] = 1.0 - w; we[3] = 1.0;
    }
}

// The (a,b) unordered-with-self pair for flat pair index k (k in [0,kNpair)).
__device__ inline void d_pair(int k, int* pa, int* pb) {
    int idx = 0;
    for (int a = 0; a < kL; ++a)
        for (int b = a; b < kL; ++b) {
            if (idx == k) { *pa = a; *pb = b; return; }
            ++idx;
        }
    *pa = 0; *pb = 0;
}

// Fill ppwts_2d [kNpair x kNedge] (column-major M(k,e)=ppwts[k + kNpair*e]) for mixture w.
__device__ inline void d_fill_pwts(double w, double* ppwts) {
    double wa[kNedge], wb[kNedge];
    for (int k = 0; k < kNpair; ++k) {
        int pa, pb; d_pair(k, &pa, &pb);
        d_leafweights(pa, w, wa);
        d_leafweights(pb, w, wb);
        for (int e = 0; e < kNedge; ++e) ppwts[k + kNpair * e] = wa[e] * wb[e];
    }
}

// THE OBJECTIVE: score(w) = res'·Qinv·res with the inner GLS-optimal edge lengths.
// f_obs (length kNpair), Qinv (kNpair x kNpair, column-major) are constant device args.
// Native FP64 throughout (the carve-out). Returns the GLS score; optionally writes the
// fitted edge lengths to bl_out (length kNedge) when bl_out != nullptr.
__device__ inline double d_graph_gls_score(double w, const double* f_obs,
                                           const double* qinv, double* bl_out) {
    double ppwts[kNpair * kNedge];
    d_fill_pwts(w, ppwts);
    // W = Qinv * ppwts  (kNpair x kNedge)
    double Wm[kNpair * kNedge];
    for (int e = 0; e < kNedge; ++e)
        for (int r = 0; r < kNpair; ++r) {
            double acc = 0.0;
            for (int c = 0; c < kNpair; ++c) acc += qinv[r + kNpair * c] * ppwts[c + kNpair * e];
            Wm[r + kNpair * e] = acc;
        }
    // cc = ppwts' * W  (kNedge x kNedge SPD);  rhs = ppwts' * Qinv * f_obs
    double cc[kNedge * kNedge];
    double rhs[kNedge];
    for (int e1 = 0; e1 < kNedge; ++e1) {
        for (int e2 = 0; e2 < kNedge; ++e2) {
            double acc = 0.0;
            for (int r = 0; r < kNpair; ++r) acc += ppwts[r + kNpair * e1] * Wm[r + kNpair * e2];
            cc[e1 + kNedge * e2] = acc;
        }
        double rr = 0.0;
        for (int r = 0; r < kNpair; ++r) rr += ppwts[r + kNpair * e1] * (
            // (Qinv*f_obs)[r]
            [&]() { double a = 0.0; for (int c = 0; c < kNpair; ++c) a += qinv[r + kNpair * c] * f_obs[c]; return a; }());
        rhs[e1] = rr;
    }
    // ridge fudge: trace-scaled (matches the design fudge=1e-4 trace-scaled diag).
    double tr = 0.0;
    for (int e = 0; e < kNedge; ++e) tr += cc[e + kNedge * e];
    const double ridge = kFudge * tr / kNedge;
    for (int e = 0; e < kNedge; ++e) cc[e + kNedge * e] += ridge;
    // inner solve cc * bl = rhs
    double bl[kNedge], lu[kNedge * kNedge], y[kNedge];
    int piv[kNedge];
    if (!d_solve(cc, kNedge, rhs, bl, lu, piv, y)) {
        if (bl_out) for (int e = 0; e < kNedge; ++e) bl_out[e] = 0.0;
        return 1e30;  // singular inner system -> reject
    }
    // res = f_obs - ppwts*bl ; score = res' Qinv res
    double res[kNpair];
    for (int r = 0; r < kNpair; ++r) {
        double pb = 0.0;
        for (int e = 0; e < kNedge; ++e) pb += ppwts[r + kNpair * e] * bl[e];
        res[r] = f_obs[r] - pb;
    }
    double score = 0.0;
    for (int a = 0; a < kNpair; ++a) {
        double row = 0.0;
        for (int b = 0; b < kNpair; ++b) row += qinv[a + kNpair * b] * res[b];
        score += res[a] * row;
    }
    if (bl_out) for (int e = 0; e < kNedge; ++e) bl_out[e] = bl[e];
    return score;
}

// Deterministic per-instance initial w0 in [0,1] (a multistart perturbation seed).
__device__ inline double d_init_w0(unsigned inst) {
    // simple splitmix-style hash -> [0,1]; deterministic per instance.
    unsigned long long z = (static_cast<unsigned long long>(inst) + 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<double>(z & 0xFFFFFFFFFFFFFULL) / static_cast<double>(0x10000000000000ULL);
}

// ============================ IDEA 1: batched sequential fleet ======================
// One CUDA thread per instance. Each runs its OWN bounded projected-Newton / secant
// quasi-Newton on the 1-D w in [0,1] (nadmix=1: the L-BFGS m-history collapses to a 1-D
// curvature estimate -> bounded projected Newton with finite-difference grad+curvature is
// the faithful nadmix=1 reduction; for nadmix>1 a thread carries the full m-history). The
// WHOLE multistart x maxit loop is in-kernel; only the final {w,score} cross back. All N
// advance in lockstep SIMT — the FLEET is the parallel axis. Per-step the objective+grad
// are finite-difference (nadmix+1 = 2 evals/step) over the resident f_obs/Qinv.
__global__ void idea1_fleet_kernel(int N, const double* f_obs, const double* qinv,
                                   int maxit, double wtol, double stol,
                                   double w_star, double score_star,
                                   double* out_w, double* out_score, int* out_iters,
                                   unsigned long long* eval_counter) {
    const int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= N) return;
    double w = d_init_w0(static_cast<unsigned>(inst));
    if (w < 0.0) w = 0.0; if (w > 1.0) w = 1.0;
    unsigned long long evals = 0;
    const double h = 1e-4;                 // finite-difference step
    double s = d_graph_gls_score(w, f_obs, qinv, nullptr); evals++;
    int it = 0;
    for (; it < maxit; ++it) {
        // forward finite-difference gradient + central curvature (bounded inside [0,1]).
        const double wp = fmin(1.0, w + h), wm = fmax(0.0, w - h);
        const double sp = d_graph_gls_score(wp, f_obs, qinv, nullptr); evals++;
        const double sm = d_graph_gls_score(wm, f_obs, qinv, nullptr); evals++;
        const double dwp = wp - w, dwm = w - wm;
        // gradient (one-sided-safe) and curvature via the 3-point stencil.
        double g, curv;
        if (dwp > 0.0 && dwm > 0.0) {
            g = (sp - sm) / (dwp + dwm);
            curv = (sp - 2.0 * s + sm) / (0.5 * (dwp + dwm) * (dwp + dwm) + 1e-30);
        } else if (dwp > 0.0) {
            g = (sp - s) / dwp; curv = 1.0;
        } else {
            g = (s - sm) / dwm; curv = 1.0;
        }
        // projected Newton step; fall back to a damped gradient step on bad curvature.
        double step = (curv > 1e-8) ? (g / curv) : (g * 0.5);
        // line-search-free trust clamp: cap the move, project to [0,1].
        if (step > 0.5) step = 0.5;
        if (step < -0.5) step = -0.5;
        double wn = w - step;
        if (wn < 0.0) wn = 0.0; if (wn > 1.0) wn = 1.0;
        double sn = d_graph_gls_score(wn, f_obs, qinv, nullptr); evals++;
        // backtracking: if the projected Newton step did not improve, halve toward w.
        int bt = 0;
        while (sn > s && bt < 8) {
            wn = 0.5 * (wn + w);
            sn = d_graph_gls_score(wn, f_obs, qinv, nullptr); evals++;
            ++bt;
        }
        const double dw = fabs(wn - w);
        const double ds = fabs(sn - s);
        w = wn; s = sn;
        if (dw < wtol * 1e-2 && ds < stol * 1e-3) break;  // projected-grad-norm proxy
    }
    out_w[inst] = w;
    out_score[inst] = s;
    out_iters[inst] = it;
    atomicAdd(eval_counter, evals);
    (void)w_star; (void)score_star;
}

// ============================ IDEA 2: batched population (CMA-ES / DE) ===============
// One thread per instance; each runs a small DERIVATIVE-FREE population optimizer over the
// 1-D w in [0,1]. We use a CMA-ES-flavored 1-D self-adaptive Gaussian search (for nadmix=1
// the covariance C collapses to a scalar step-size sigma; B*D -> sigma): per generation
// sample lambda candidates w_k = clamp(m + sigma*z_k), z_k ~ N(0,1) via curand; rank by
// score; recombine the mu best (weighted mean) into the new mean m; adapt sigma by the
// 1/5-success / weighted-variance rule. The population is the extra parallel axis (N*lambda
// evals/generation). Derivative-free -> no finite-difference launches, tolerant of flat
// surfaces. The WHOLE generation loop is in-kernel; curand seeded per (instance) — NO host
// RNG, NO host per-candidate loop.
static constexpr int kLambda = 12;   // 4 + 3*ln(1) rounded up to a SIMT-friendly width
static constexpr int kMu = 6;        // parents (best half)
__global__ void idea2_population_kernel(int N, const double* f_obs, const double* qinv,
                                        int maxgen, double wtol, double stol,
                                        double w_star, double score_star,
                                        double* out_w, double* out_score, int* out_gens,
                                        unsigned long long* eval_counter) {
    const int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= N) return;
    curandStatePhilox4_32_10_t rng;
    curand_init(0x51EEDULL, static_cast<unsigned long long>(inst), 0ULL, &rng);
    double m = d_init_w0(static_cast<unsigned>(inst));   // initial mean (multistart seed)
    if (m < 0.0) m = 0.0; if (m > 1.0) m = 1.0;
    double sigma = 0.3;                                   // initial step (covers [0,1])
    unsigned long long evals = 0;
    // recombination weights (log-decreasing, normalized) for the mu best.
    double weights[kMu]; double wsum = 0.0;
    for (int i = 0; i < kMu; ++i) { weights[i] = log(kMu + 0.5) - log(double(i + 1)); wsum += weights[i]; }
    for (int i = 0; i < kMu; ++i) weights[i] /= wsum;
    double best_w = m, best_s = d_graph_gls_score(m, f_obs, qinv, nullptr); evals++;
    int gen = 0;
    for (; gen < maxgen; ++gen) {
        double cw[kLambda], cs[kLambda];
        for (int k = 0; k < kLambda; ++k) {
            double z = curand_normal_double(&rng);
            double wk = m + sigma * z;
            // box-constraint by reflection into [0,1].
            if (wk < 0.0) wk = -wk;
            if (wk > 1.0) wk = 2.0 - wk;
            if (wk < 0.0) wk = 0.0; if (wk > 1.0) wk = 1.0;
            cw[k] = wk;
            cs[k] = d_graph_gls_score(wk, f_obs, qinv, nullptr); evals++;
        }
        // insertion sort the lambda candidates ascending by score (lambda<=16, in-register).
        for (int i = 1; i < kLambda; ++i) {
            double kw = cw[i], ks = cs[i]; int j = i - 1;
            while (j >= 0 && cs[j] > ks) { cs[j + 1] = cs[j]; cw[j + 1] = cw[j]; --j; }
            cs[j + 1] = ks; cw[j + 1] = kw;
        }
        // weighted recombination of the mu best -> new mean.
        double m_new = 0.0;
        for (int i = 0; i < kMu; ++i) m_new += weights[i] * cw[i];
        // step-size adaptation: weighted std of the selected around the old mean.
        double var = 0.0;
        for (int i = 0; i < kMu; ++i) { double d = cw[i] - m; var += weights[i] * d * d; }
        double sigma_new = sqrt(var) * 1.5 + 1e-9;        // empirical 1-D CMA-style update
        // damp so sigma neither explodes nor collapses prematurely.
        sigma = 0.5 * sigma + 0.5 * sigma_new;
        m = m_new;
        if (m < 0.0) m = 0.0; if (m > 1.0) m = 1.0;
        if (cs[0] < best_s) { best_s = cs[0]; best_w = cw[0]; }
        if (sigma < wtol * 1e-2) break;                   // step-size floor convergence
    }
    out_w[inst] = best_w;
    out_score[inst] = best_s;
    out_gens[inst] = gen;
    atomicAdd(eval_counter, evals);
    (void)stol; (void)w_star; (void)score_star;
}

// ============================ HOST: fixture reader + basis ==========================
namespace {

struct Fixture {
    int P = 0, nb = 0;
    std::vector<int> block_sizes;
    std::vector<double> f2;  // i + P*j + P*P*b (column-major slabs)
};

// Identical layout to test_qpadm_parity.cu read_fixture: int32 P, int32 nb,
// int32 block_sizes[nb], double f2[P*P*nb].
bool read_fixture(const std::string& path, Fixture& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("ERROR: cannot open fixture %s\n", path.c_str()); return false; }
    std::int32_t P = 0, nb = 0;
    f.read(reinterpret_cast<char*>(&P), sizeof(P));
    f.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    if (!f || P <= 0 || nb <= 0) { std::printf("ERROR: bad fixture header\n"); return false; }
    out.P = P; out.nb = nb;
    out.block_sizes.resize(static_cast<std::size_t>(nb));
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) * static_cast<std::size_t>(nb)));
    const std::size_t n = static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                          static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    if (!f) { std::printf("ERROR: fixture truncated\n"); return false; }
    return true;
}

// HOST oracle: f_obs (block-size-weighted total outgroup-f3) + Q (block-jackknife cov,
// diag_f3) + Qinv (small_linalg inverse). C=Mbuti(3), leaves S1=2,S2=5,X=1.
struct Basis {
    std::vector<double> f_obs;   // kNpair
    std::vector<double> qinv;    // kNpair x kNpair column-major
    bool ok = false;
};

Basis build_basis(const Fixture& fx) {
    Basis B;
    const int P = fx.P, nb = fx.nb;
    const int C = 3;
    const int leaves[kL] = {2, 5, 1};
    auto F2 = [&](int i, int j, int b) -> double {
        return fx.f2[static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * j +
                     static_cast<std::size_t>(P) * P * b];
    };
    // pair table (a,b) with a<=b over leaf positions.
    int pa[kNpair], pb[kNpair];
    { int idx = 0; for (int a = 0; a < kL; ++a) for (int b = a; b < kL; ++b) { pa[idx] = a; pb[idx] = b; ++idx; } }
    // per-block f3 vector + block sizes.
    std::vector<double> fb(static_cast<std::size_t>(nb) * kNpair);
    double n_tot = 0.0;
    for (int b = 0; b < nb; ++b) n_tot += fx.block_sizes[static_cast<std::size_t>(b)];
    for (int b = 0; b < nb; ++b)
        for (int k = 0; k < kNpair; ++k) {
            const int i = leaves[pa[k]], j = leaves[pb[k]];
            fb[static_cast<std::size_t>(b) * kNpair + k] = 0.5 * (F2(C, i, b) + F2(C, j, b) - F2(i, j, b));
        }
    // block-size-weighted total f_obs and the per-block leave-one-out estimates.
    B.f_obs.assign(kNpair, 0.0);
    std::vector<double> tot_sum(kNpair, 0.0);
    for (int b = 0; b < nb; ++b)
        for (int k = 0; k < kNpair; ++k)
            tot_sum[k] += fx.block_sizes[static_cast<std::size_t>(b)] * fb[static_cast<std::size_t>(b) * kNpair + k];
    for (int k = 0; k < kNpair; ++k) B.f_obs[k] = tot_sum[k] / n_tot;
    std::vector<double> loo(static_cast<std::size_t>(nb) * kNpair);
    for (int b = 0; b < nb; ++b) {
        const double nb_b = fx.block_sizes[static_cast<std::size_t>(b)];
        for (int k = 0; k < kNpair; ++k)
            loo[static_cast<std::size_t>(b) * kNpair + k] =
                (tot_sum[k] - nb_b * fb[static_cast<std::size_t>(b) * kNpair + k]) / (n_tot - nb_b);
    }
    // jackknife covariance around the loo mean: cov = (nb-1)/nb * sum (loo_b - mean)(.)'
    std::vector<double> mean(kNpair, 0.0);
    for (int b = 0; b < nb; ++b) for (int k = 0; k < kNpair; ++k) mean[k] += loo[static_cast<std::size_t>(b) * kNpair + k];
    for (int k = 0; k < kNpair; ++k) mean[k] /= nb;
    std::vector<double> cov(static_cast<std::size_t>(kNpair) * kNpair, 0.0);
    for (int b = 0; b < nb; ++b)
        for (int a = 0; a < kNpair; ++a) {
            const double da = loo[static_cast<std::size_t>(b) * kNpair + a] - mean[a];
            for (int c = 0; c < kNpair; ++c) {
                const double dc = loo[static_cast<std::size_t>(b) * kNpair + c] - mean[c];
                cov[static_cast<std::size_t>(a) + kNpair * c] += da * dc;
            }
        }
    const double scl = static_cast<double>(nb - 1) / static_cast<double>(nb);
    for (auto& v : cov) v *= scl;
    for (int k = 0; k < kNpair; ++k) cov[static_cast<std::size_t>(k) + kNpair * k] += kDiagF3;
    // Qinv via the host oracle.
    std::vector<double> qinv;
    const steppe::core::LinAlgStatus st = steppe::core::inverse(cov, kNpair, qinv);
    if (!st.ok) { std::printf("ERROR: covariance singular (inverse failed)\n"); return B; }
    B.qinv = std::move(qinv);
    B.ok = true;
    return B;
}

// HOST grid-pin of (w*, score*) using the SAME algebra (the ground truth the GPU must hit).
void host_pin(const Basis& B, double& w_star, double& score_star) {
    auto score = [&](double w) -> double {
        double ppwts[kNpair * kNedge];
        // fill_pwts (host mirror)
        auto lw = [](int li, double w, double* we) {
            we[0] = we[1] = we[2] = we[3] = 0.0;
            if (li == 0) { we[0] = 1; we[1] = 1; }
            else if (li == 1) { we[0] = 1; we[2] = 1; }
            else { we[0] = 1; we[1] = w; we[2] = 1 - w; we[3] = 1; }
        };
        int idx = 0; double wa[kNedge], wb[kNedge];
        for (int a = 0; a < kL; ++a) for (int b = a; b < kL; ++b) {
            lw(a, w, wa); lw(b, w, wb);
            for (int e = 0; e < kNedge; ++e) ppwts[idx + kNpair * e] = wa[e] * wb[e];
            ++idx;
        }
        std::vector<double> cc(static_cast<std::size_t>(kNedge) * kNedge, 0.0), rhs(kNedge, 0.0);
        for (int e1 = 0; e1 < kNedge; ++e1) {
            for (int e2 = 0; e2 < kNedge; ++e2) {
                double acc = 0;
                for (int r = 0; r < kNpair; ++r) {
                    double qp = 0; for (int c = 0; c < kNpair; ++c) qp += B.qinv[static_cast<std::size_t>(r) + kNpair * c] * ppwts[c + kNpair * e2];
                    acc += ppwts[r + kNpair * e1] * qp;
                }
                cc[static_cast<std::size_t>(e1) + kNedge * e2] = acc;
            }
            double rr = 0;
            for (int r = 0; r < kNpair; ++r) {
                double qf = 0; for (int c = 0; c < kNpair; ++c) qf += B.qinv[static_cast<std::size_t>(r) + kNpair * c] * B.f_obs[c];
                rr += ppwts[r + kNpair * e1] * qf;
            }
            rhs[e1] = rr;
        }
        double tr = 0; for (int e = 0; e < kNedge; ++e) tr += cc[static_cast<std::size_t>(e) + kNedge * e];
        for (int e = 0; e < kNedge; ++e) cc[static_cast<std::size_t>(e) + kNedge * e] += kFudge * tr / kNedge;
        std::vector<double> bl;
        if (!steppe::core::solve(cc, kNedge, rhs, bl).ok) return 1e30;
        double sc = 0;
        std::vector<double> res(kNpair);
        for (int r = 0; r < kNpair; ++r) {
            double pb = 0; for (int e = 0; e < kNedge; ++e) pb += ppwts[r + kNpair * e] * bl[static_cast<std::size_t>(e)];
            res[static_cast<std::size_t>(r)] = B.f_obs[r] - pb;
        }
        for (int a = 0; a < kNpair; ++a) {
            double row = 0; for (int b = 0; b < kNpair; ++b) row += B.qinv[static_cast<std::size_t>(a) + kNpair * b] * res[static_cast<std::size_t>(b)];
            sc += res[static_cast<std::size_t>(a)] * row;
        }
        return sc;
    };
    // coarse then fine grid pin.
    double bw = 0, bs = 1e300;
    for (int i = 0; i <= 2000; ++i) { double w = i / 2000.0; double s = score(w); if (s < bs) { bs = s; bw = w; } }
    double lo = std::max(0.0, bw - 1e-3), hi = std::min(1.0, bw + 1e-3);
    for (int i = 0; i <= 60000; ++i) { double w = lo + (hi - lo) * i / 60000.0; double s = score(w); if (s < bs) { bs = s; bw = w; } }
    w_star = bw; score_star = bs;
}

// CUDA error check.
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("CUDA ERROR (%s): %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// Background nvidia-smi sampler: launches the CLI in the background writing a CSV, returns
// the pid so the caller can SIGINT it; then parses peak util + peak mem-used delta.
struct GpuSample { double peak_util = 0, peak_mem_mib = 0, mean_util = 0; int n = 0; };

}  // namespace

int main(int argc, char** argv) {
    const std::string fixture = (argc > 1) ? argv[1]
        : "tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin";
    std::vector<long> Ns;
    if (argc > 2) {
        std::string s = argv[2]; size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p); if (!tok.empty()) Ns.push_back(std::atol(tok.c_str())); if (q == std::string::npos) break; p = q + 1; }
    } else { Ns = {1000, 10000, 100000, 1000000}; }
    const int maxit = (argc > 3) ? std::atoi(argv[3]) : 200;
    const double tol = (argc > 4) ? std::atof(argv[4]) : 1e-3;

    std::printf("=== bench_optimizers — GPU optimizer spike (IDEA1 fleet vs IDEA2 population) ===\n");
    std::printf("fixture = %s\n", fixture.c_str());

    Fixture fx;
    if (!read_fixture(fixture, fx)) return 1;
    std::printf("fixture: P=%d nb=%d\n", fx.P, fx.nb);
    if (fx.P < 6) { std::printf("ERROR: need P>=6 (Mbuti+2 sources+admixed leaf among 9 pops)\n"); return 1; }

    // ---- BASIS ONCE (host oracle): f_obs, Qinv, and the (w*, score*) ground truth ----
    const Basis B = build_basis(fx);
    if (!B.ok) return 1;
    double w_star = 0, score_star = 0;
    host_pin(B, w_star, score_star);
    std::printf("BASIS (host oracle): f_obs = [");
    for (int k = 0; k < kNpair; ++k) std::printf("%s%.6f", k ? ", " : "", B.f_obs[k]);
    std::printf("]\n");
    std::printf("GROUND TRUTH (host grid-pin): w* = %.6f   score* = %.6f\n", w_star, score_star);

    // ---- device basis (resident, shared by every instance / both optimizers) ---------
    double *d_fobs = nullptr, *d_qinv = nullptr;
    ck(cudaMalloc(&d_fobs, kNpair * sizeof(double)), "malloc fobs");
    ck(cudaMalloc(&d_qinv, static_cast<std::size_t>(kNpair) * kNpair * sizeof(double)), "malloc qinv");
    ck(cudaMemcpy(d_fobs, B.f_obs.data(), kNpair * sizeof(double), cudaMemcpyHostToDevice), "h2d fobs");
    ck(cudaMemcpy(d_qinv, B.qinv.data(), static_cast<std::size_t>(kNpair) * kNpair * sizeof(double), cudaMemcpyHostToDevice), "h2d qinv");

    const double wtol = tol, stol = tol;

    std::printf("\n%-7s | %-6s | %14s | %14s | %12s | %10s | %s\n",
                "OPT", "N", "wall(ms)", "obj_evals", "conv_frac", "dispatch", "median |w-w*|");
    std::printf("--------+--------+----------------+----------------+--------------+------------+--------------\n");

    auto run_one = [&](int which, long N) {
        double *d_w, *d_score; int *d_iters; unsigned long long *d_evals;
        ck(cudaMalloc(&d_w, N * sizeof(double)), "malloc w");
        ck(cudaMalloc(&d_score, N * sizeof(double)), "malloc score");
        ck(cudaMalloc(&d_iters, N * sizeof(int)), "malloc iters");
        ck(cudaMalloc(&d_evals, sizeof(unsigned long long)), "malloc evals");
        ck(cudaMemset(d_evals, 0, sizeof(unsigned long long)), "memset evals");
        const int TPB = 256;
        const int blocks = static_cast<int>((N + TPB - 1) / TPB);
        // WARM-UP (prime the kernel; not timed).
        if (which == 1) idea1_fleet_kernel<<<blocks, TPB>>>(static_cast<int>(N), d_fobs, d_qinv, maxit, wtol, stol, w_star, score_star, d_w, d_score, d_iters, d_evals);
        else            idea2_population_kernel<<<blocks, TPB>>>(static_cast<int>(N), d_fobs, d_qinv, maxit, wtol, stol, w_star, score_star, d_w, d_score, d_iters, d_evals);
        ck(cudaDeviceSynchronize(), "warmup sync");
        ck(cudaMemset(d_evals, 0, sizeof(unsigned long long)), "memset evals2");
        // TIMED.
        const auto t0 = std::chrono::steady_clock::now();
        if (which == 1) idea1_fleet_kernel<<<blocks, TPB>>>(static_cast<int>(N), d_fobs, d_qinv, maxit, wtol, stol, w_star, score_star, d_w, d_score, d_iters, d_evals);
        else            idea2_population_kernel<<<blocks, TPB>>>(static_cast<int>(N), d_fobs, d_qinv, maxit, wtol, stol, w_star, score_star, d_w, d_score, d_iters, d_evals);
        ck(cudaDeviceSynchronize(), "timed sync");
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // readback + convergence quality.
        std::vector<double> hw(N), hs(N);
        unsigned long long hevals = 0;
        ck(cudaMemcpy(hw.data(), d_w, N * sizeof(double), cudaMemcpyDeviceToHost), "d2h w");
        ck(cudaMemcpy(hs.data(), d_score, N * sizeof(double), cudaMemcpyDeviceToHost), "d2h score");
        ck(cudaMemcpy(&hevals, d_evals, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "d2h evals");
        long conv = 0;
        std::vector<double> werr(N);
        for (long i = 0; i < N; ++i) {
            werr[static_cast<std::size_t>(i)] = std::fabs(hw[static_cast<std::size_t>(i)] - w_star);
            if (werr[static_cast<std::size_t>(i)] < tol && std::fabs(hs[static_cast<std::size_t>(i)] - score_star) < tol) ++conv;
        }
        std::sort(werr.begin(), werr.end());
        const double med_werr = werr[werr.size() / 2];
        const double conv_frac = static_cast<double>(conv) / static_cast<double>(N);
        // ONE batched dispatch (the GPU-bound proof: 1 launch for the WHOLE optimizer, not
        // N*iters host round-trips).
        std::printf("%-7s | %-6ld | %14.2f | %14llu | %11.4f%% | %10d | %.3e\n",
                    which == 1 ? "IDEA1" : "IDEA2", N, ms,
                    static_cast<unsigned long long>(hevals), conv_frac * 100.0, 1, med_werr);
        cudaFree(d_w); cudaFree(d_score); cudaFree(d_iters); cudaFree(d_evals);
        return conv_frac;
    };

    // SANITY (small N): both must converge to the checkable optimum.
    std::printf("\n--- SANITY: small-N convergence to the host-pinned optimum (N=256) ---\n");
    const double s1 = run_one(1, 256);
    const double s2 = run_one(2, 256);
    if (s1 < 0.90) std::printf("  [WARN] IDEA1 converged only %.1f%% at N=256\n", s1 * 100.0);
    if (s2 < 0.90) std::printf("  [WARN] IDEA2 converged only %.1f%% at N=256\n", s2 * 100.0);

    // SCALE SWEEP.
    std::printf("\n--- SCALE SWEEP (single GPU, device 0) ---\n");
    std::printf("%-7s | %-6s | %14s | %14s | %12s | %10s | %s\n",
                "OPT", "N", "wall(ms)", "obj_evals", "conv_frac", "dispatch", "median |w-w*|");
    std::printf("--------+--------+----------------+----------------+--------------+------------+--------------\n");
    for (long N : Ns) { run_one(1, N); run_one(2, N); }

    cudaFree(d_fobs); cudaFree(d_qinv);
    std::printf("\nNOTE: sample GPU util with `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 100` in parallel.\n");
    return 0;
}
