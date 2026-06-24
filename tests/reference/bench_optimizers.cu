// bench_optimizers.cu — GPU OPTIMIZER SPIKE (FAIR RE-MATCH): batched-sequential fleet
// (IDEA1: L-BFGS-B / projected quasi-Newton fleet) vs PROPER batched-population (IDEA2:
// block/warp-cooperative CMA-ES and DE), both GPU-BOUND, on a REAL-f2-grounded nonlinear
// box-constrained admixture-graph GLS objective at nadmix = 1, 2 AND 3.
//
// PURPOSE. The earlier spike (docs/research/optimizer-comparison.md +
// qpgraph-optimizer-spike.md) found IDEA1 wins — BUT (caveat #2) IDEA2 was UNDER-
// implemented: a SERIAL in-thread lambda loop (one thread per instance), NOT lambda-
// across-threads, so it threw away the N*lambda parallel axis; and (caveat #1) it ran
// only nadmix=1 (a degenerate 1-D smooth unimodal surface the gradient trivially wins).
// THE FAIR RE-MATCH replaces the serial IDEA2 with the PROPER block-cooperative form
// (warp-per-instance, lambda candidates ACROSS lanes, block-cooperative CMA/DE update)
// and adds nadmix=2 / nadmix=3 (nested product-of-mixtures topologies -> ridges/saddles,
// a non-degenerate surface where the population method can show its robustness). IDEA1 is
// kept AS-IS (it was proper) but generalized to theta_dim D (the nadmix=1 path is
// bit-identical to before).
//
// THE OBJECTIVE (REAL data, NO synthetic; cuRAND is IDEA2 SEARCH sampling, not data).
// Source = the committed 9-pop AADR f2 fixture
// tests/reference/goldens/at2/fixtures/f2_fit0_9pop.bin (P=9, nb=710 jackknife blocks;
// the SAME real tensor test_qpadm_parity / test_qpwave_parity use). Pops:
//   0 England_BellBeaker 1 Czechia_EBA_CordedWare 2 Turkey_N 3 Mbuti
//   4 Israel_Natufian 5 Iran_GanjDareh_N 6 Han 7 Papuan 8 Karitiana
// Outgroup base C = Mbuti(3) throughout. ALGEBRA = steppe's EXACT
// assemble_f3_triples_gather identity f3(C;i,j)=0.5*(f2(C,i)+f2(C,j)-f2(i,j))
// (qpadm_fit_kernels.cu:663). f_obs = block-size-weighted total f3 vector; Sigma = the f3
// block-jackknife covariance Q (diag_f3=1e-5), inverted to Qinv on the HOST oracle
// (small_linalg). f_pred(theta) maps theta -> ppwts_2d [npair x nedge] (LINEAR in edge
// lengths, NONLINEAR in theta via the product-of-mixtures lineage overlaps); the inner
// edge lengths are the GLS-optimal solve of cc = ppwts'·Qinv·ppwts (nedge x nedge SPD,
// ridge fudge=1e-4 trace-scaled). OBJECTIVE = res'·Qinv·res, res = f_obs - ppwts·bl.
//
// TOPOLOGIES (host-pinned checkable optimum for each):
//  nadmix=1: L=3 {S1=Turkey_N(2), S2=Iran(5), X=CordedWare(1)=a*S1+(1-a)*S2}. theta=(a).
//            npair=6, nedge=4. SMOOTH UNIMODAL (the regression anchor; a*~0.829).
//  nadmix=2: L=5 add S3=Natufian(4) and X2=BellBeaker(0)=b*X1+(1-b)*S3 (the 2nd admixture
//            consumes the 1st admixed lineage X1, so X2's mass depends on BOTH a,b — the
//            nested product a*b is the source of the ridge/saddle). theta=(a,b). npair=15,
//            nedge=7.
//  nadmix=3: L=6 add S4=Han(6) and X3=Karitiana(8)=c*X2+(1-c)*S4 (a deep East-Eurasian
//            pulse). theta=(a,b,c). npair=21, nedge=10. Three nested products -> a
//            genuinely 3-D non-separable surface.
//
// THE TWO OPTIMIZER SHAPES (both fully on-device, ONE launch each, GPU-BOUND):
//  IDEA1 — batched-sequential fleet. One CUDA THREAD per instance; bounded projected
//          quasi-Newton on theta in [0,1]^D (per-dim finite-difference gradient + diagonal
//          3-point curvature, projected trust-clamped step, backtracking). The FLEET (N)
//          is the parallel axis. ~ (2D+1) evals/step. (theta_dim=1 reduces EXACTLY to the
//          prior IDEA1.)
//  IDEA2 — PROPER batched-population (block/warp-cooperative). ONE WARP per instance
//          (gridDim.x = N warps); the lambda candidates are evaluated ACROSS the warp's
//          lanes (lane t = candidate t) — the N*lambda parallel axis the serial version
//          threw away. The instance's evolving CMA state (mean m, step sigma, full
//          covariance C, RNG, best-so-far) lives in registers/shared OWNED by the warp,
//          mutated once per generation by a WARP-COOPERATIVE reduction:
//            1. SAMPLE   : lane t draws z_t~N(0,I) (curand Philox, distinct substream per
//                          (instance,gen,lane)); candidate theta_t = clamp(m + sigma*B*D*z_t)
//                          with B*D from the C eigendecomposition (rotation-invariant CMA).
//            2. EVALUATE : lane t computes score_t = d_graph_gls_score(theta_t) — the SAME
//                          objective body. This is the N*lambda eval parallelism.
//            3. RANK     : warp bitonic sort of (score_t) via __shfl_xor_sync (lambda<=32).
//            4. UPDATE   : weighted recombine of the mu best into m_new + rank-mu C update,
//                          accumulated by warp-shuffle reductions (cg::reduce-equivalent).
//                          Step-size by the weighted-variance rule. Lane 0 re-eigendecomposes
//                          the 3x3 C for next gen's B*D and broadcasts via shuffle.
//          NO serial in-thread lambda loop. The WHOLE generation loop stays in-kernel.
//  IDEA2-DE — alternative block-cooperative population: one WARP = a sub-population of
//          lambda candidates (one per lane), per-gen rand/1/bin mutation + binomial
//          crossover + greedy 1:1 select vs the lane's own parent (no global sort; the warp
//          IS the population). Cheaper sync, no covariance learning — benched as the
//          lower-overhead population shape.
//
// PRECISION. Inner GLS solve + the GLS quadratic form are NATIVE FP64 (the cancellation-
// sensitive carve-out, matching dev_chisq_of_core). The matmul-heavy ppwts'·Qinv·ppwts
// assembly is small even at nadmix=3 (nedge<=10) and in-register per thread; the
// EmulatedFp64{40} GEMM seam is the production-fit path, not this in-thread objective —
// the spike measures OPTIMIZER SHAPE (noted in the report).
//
// MANUAL bench, NOT a ctest gate (no add_test) — like bench_rotation_1240k.cu. Run:
//   ./bench_optimizers [fixture_path] [Ncsv] [maxit] [tol] [nadmixcsv] [opts]
//   nadmixcsv  comma list of nadmix in {1,2,3}, default 1,2,3
//   opts       comma list of optimizers in {1,2,3} (1=IDEA1, 2=IDEA2-CMA, 3=IDEA2-DE),
//              default 1,2,3
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

// ============================ COMPILE-TIME PROBLEM ENVELOPE =========================
// Max dims across nadmix 1..3 so the per-thread / per-warp scratch is a single fixed
// size (register/local resident at nadmix=3; warp-shared scratch is a few KB/warp).
static constexpr int kMaxL = 6;                              // nadmix=3 leaf count
static constexpr int kMaxNpair = (kMaxL * (kMaxL + 1)) / 2;  // 21
static constexpr int kMaxNedge = 10;                         // nadmix=3 edges
static constexpr int kMaxTheta = 3;                          // nadmix=3 mixture weights
static constexpr double kDiagF3 = 1e-5;
static constexpr double kFudge = 1e-4;

// IDEA2 population width: ONE WARP per instance, lambda = 32 candidates across the lanes
// (the full warp -> the warp-shuffle reductions are the cheapest sync; lambda well above
// the CMA default 4+3*ln(D) so the covariance estimate is well-sampled). mu = lambda/2.
static constexpr int kLambda = 32;
static constexpr int kMu = 16;

// ============================ TOPOLOGY DESCRIPTOR ==================================
// A nadmix level fully described by {L, npair, nedge, theta_dim, leaf pop indices}. The
// lineage-weight function (the product-of-mixtures) is dispatched by nadmix at compile
// time below. Edge layout (per nadmix) is documented at d_leafweights_n*.
struct Topo {
    int nadmix;
    int L;
    int npair;
    int nedge;
    int theta;
    int leaves[kMaxL];   // pop indices into the P=9 fixture
};

__host__ __device__ inline Topo make_topo(int nadmix) {
    Topo t{};
    t.nadmix = nadmix;
    if (nadmix == 1) {
        t.L = 3; t.npair = 6;  t.nedge = 4;  t.theta = 1;
        t.leaves[0] = 2; t.leaves[1] = 5; t.leaves[2] = 1;             // S1,S2,X
    } else if (nadmix == 2) {
        t.L = 5; t.npair = 15; t.nedge = 7;  t.theta = 2;
        t.leaves[0] = 2; t.leaves[1] = 5; t.leaves[2] = 4;             // S1,S2,S3
        t.leaves[3] = 1; t.leaves[4] = 0;                             // X1,X2
    } else {  // nadmix == 3
        // L=6 sampled leaves {S1,S2,S3,S4, X2,X3}. X1 is an INTERNAL admixed node (= a*S1
        // + (1-a)*S2), not sampled, consumed by X2. d_leafweights_n3 encodes the leaf
        // order: 0:S1 1:S2 2:S3 3:S4 4:X2 5:X3.
        t.L = 6; t.npair = 21; t.nedge = 10; t.theta = 3;
        t.leaves[0] = 2;  // S1 = Turkey_N
        t.leaves[1] = 5;  // S2 = Iran
        t.leaves[2] = 4;  // S3 = Natufian
        t.leaves[3] = 6;  // S4 = Han
        t.leaves[4] = 0;  // X2 = BellBeaker (= b*X1 + (1-b)*S3)
        t.leaves[5] = 8;  // X3 = Karitiana (= c*X2 + (1-c)*S4)
    }
    return t;
}

// ============================ DEVICE: native-FP64 linear algebra ====================
// Column-major A(i,j) at i + n*j. Mirrors qpadm_fit_kernels.cu dev_lu_factor / dev_solve
// (the cancellation-sensitive native-FP64 carve-out).
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

// ============================ TOPOLOGY: lineage weights ============================
// The lineage-mass of leaf `li` on each drift edge, given theta. ppwts[k,e] = wa[e]*wb[e]
// is then the shared-drift overlap of the pair (the qpGraph f-stat-is-linear-in-edges-
// given-theta formulation). Edge layouts (column-major M(k,e)=ppwts[k+npair*e]):
//
// nadmix=1 (nedge=4): e0 root(all); e1 S1-side; e2 S2-side; e3 X-terminal.
//   S1: e0,e1.  S2: e0,e2.  X=a*S1+(1-a)*S2: e0, e1*a, e2*(1-a), e3.
__device__ __host__ inline void d_leafweights_n1(int li, const double* th, double* we) {
    for (int e = 0; e < kMaxNedge; ++e) we[e] = 0.0;
    const double a = th[0];
    if (li == 0) { we[0] = 1.0; we[1] = 1.0; }                    // S1
    else if (li == 1) { we[0] = 1.0; we[2] = 1.0; }              // S2
    else { we[0] = 1.0; we[1] = a; we[2] = 1.0 - a; we[3] = 1.0; }  // X
}

// nadmix=2 (nedge=7): leaves {S1,S2,S3,X1,X2}. X1=a*S1+(1-a)*S2 (internal-consumed but
// also a sampled leaf =CordedWare); X2=b*X1+(1-b)*S3 (=BellBeaker). NESTED product b*a.
//   e0 root(all); e1 S1-side; e2 S2-side; e3 X1-terminal; e4 S3-side; e5 X2-terminal;
//   e6 S3 internal (extra drift edge so nedge=7, only on the S3 / X2-via-S3 lineage).
//   S1: e0,e1.  S2: e0,e2.  S3: e0,e4,e6.
//   X1: e0, e1*a, e2*(1-a), e3.
//   X2: e0, e1*(b*a), e2*(b*(1-a)), e3*b, e4*(1-b), e6*(1-b), e5.
__device__ __host__ inline void d_leafweights_n2(int li, const double* th, double* we) {
    for (int e = 0; e < kMaxNedge; ++e) we[e] = 0.0;
    const double a = th[0], b = th[1];
    if (li == 0) { we[0] = 1.0; we[1] = 1.0; }                                  // S1
    else if (li == 1) { we[0] = 1.0; we[2] = 1.0; }                            // S2
    else if (li == 2) { we[0] = 1.0; we[4] = 1.0; we[6] = 1.0; }               // S3
    else if (li == 3) { we[0] = 1.0; we[1] = a; we[2] = 1.0 - a; we[3] = 1.0; }  // X1
    else {                                                                      // X2
        we[0] = 1.0;
        we[1] = b * a; we[2] = b * (1.0 - a); we[3] = b;
        we[4] = 1.0 - b; we[6] = 1.0 - b; we[5] = 1.0;
    }
}

// nadmix=3 (nedge=10): leaves {S1,S2,S3,S4,X2,X3}. X1 internal = a*S1+(1-a)*S2;
// X2=b*X1+(1-b)*S3 (=BellBeaker, sampled); X3=c*X2+(1-c)*S4 (=Karitiana, sampled). THREE
// nested products c*b*a.
//   e0 root(all); e1 S1-side; e2 S2-side; e3 X1-terminal; e4 S3-side; e5 X2-terminal;
//   e6 S3 internal; e7 S4-side; e8 X3-terminal; e9 S4 internal.
//   S1: e0,e1.  S2: e0,e2.  S3: e0,e4,e6.  S4: e0,e7,e9.
//   X2: e0, e1*(b*a), e2*(b*(1-a)), e3*b, e4*(1-b), e6*(1-b), e5.
//   X3: e0, e1*(c*b*a), e2*(c*b*(1-a)), e3*(c*b), e4*(c*(1-b)), e6*(c*(1-b)), e5*c,
//       e7*(1-c), e9*(1-c), e8.
__device__ __host__ inline void d_leafweights_n3(int li, const double* th, double* we) {
    for (int e = 0; e < kMaxNedge; ++e) we[e] = 0.0;
    const double a = th[0], b = th[1], c = th[2];
    if (li == 0) { we[0] = 1.0; we[1] = 1.0; }                                  // S1
    else if (li == 1) { we[0] = 1.0; we[2] = 1.0; }                            // S2
    else if (li == 2) { we[0] = 1.0; we[4] = 1.0; we[6] = 1.0; }               // S3
    else if (li == 3) { we[0] = 1.0; we[7] = 1.0; we[9] = 1.0; }               // S4
    else if (li == 4) {                                                         // X2
        we[0] = 1.0;
        we[1] = b * a; we[2] = b * (1.0 - a); we[3] = b;
        we[4] = 1.0 - b; we[6] = 1.0 - b; we[5] = 1.0;
    } else {                                                                    // X3
        we[0] = 1.0;
        we[1] = c * b * a; we[2] = c * b * (1.0 - a); we[3] = c * b;
        we[4] = c * (1.0 - b); we[6] = c * (1.0 - b); we[5] = c;
        we[7] = 1.0 - c; we[9] = 1.0 - c; we[8] = 1.0;
    }
}

__device__ __host__ inline void d_leafweights(int nadmix, int li, const double* th, double* we) {
    if (nadmix == 1) d_leafweights_n1(li, th, we);
    else if (nadmix == 2) d_leafweights_n2(li, th, we);
    else d_leafweights_n3(li, th, we);
}

// The (a,b) unordered-with-self pair for flat pair index k.
__device__ __host__ inline void d_pair(int L, int k, int* pa, int* pb) {
    int idx = 0;
    for (int a = 0; a < L; ++a)
        for (int b = a; b < L; ++b) {
            if (idx == k) { *pa = a; *pb = b; return; }
            ++idx;
        }
    *pa = 0; *pb = 0;
}

// Fill ppwts [npair x nedge] (column-major) for theta.
__device__ inline void d_fill_pwts(const Topo& t, const double* th, double* ppwts) {
    double wa[kMaxNedge], wb[kMaxNedge];
    for (int k = 0; k < t.npair; ++k) {
        int pa, pb; d_pair(t.L, k, &pa, &pb);
        d_leafweights(t.nadmix, pa, th, wa);
        d_leafweights(t.nadmix, pb, th, wb);
        for (int e = 0; e < t.nedge; ++e) ppwts[k + t.npair * e] = wa[e] * wb[e];
    }
}

// THE OBJECTIVE: score(theta) = res'·Qinv·res with the inner GLS-optimal edge lengths.
// Native FP64 throughout (the carve-out). f_obs (npair), qinv (npair x npair col-major)
// are constant device args. theta is clamped into [0,1]^D by the caller.
__device__ inline double d_graph_gls_score(const Topo& t, const double* th,
                                           const double* f_obs, const double* qinv) {
    double ppwts[kMaxNpair * kMaxNedge];
    d_fill_pwts(t, th, ppwts);
    const int np = t.npair, ne = t.nedge;
    // W = Qinv * ppwts  (np x ne)
    double Wm[kMaxNpair * kMaxNedge];
    for (int e = 0; e < ne; ++e)
        for (int r = 0; r < np; ++r) {
            double acc = 0.0;
            for (int c = 0; c < np; ++c) acc += qinv[r + np * c] * ppwts[c + np * e];
            Wm[r + np * e] = acc;
        }
    // Qinv * f_obs  (np)
    double qf[kMaxNpair];
    for (int r = 0; r < np; ++r) {
        double a = 0.0;
        for (int c = 0; c < np; ++c) a += qinv[r + np * c] * f_obs[c];
        qf[r] = a;
    }
    // cc = ppwts' * W  (ne x ne SPD);  rhs = ppwts' * (Qinv*f_obs)
    double cc[kMaxNedge * kMaxNedge];
    double rhs[kMaxNedge];
    for (int e1 = 0; e1 < ne; ++e1) {
        for (int e2 = 0; e2 < ne; ++e2) {
            double acc = 0.0;
            for (int r = 0; r < np; ++r) acc += ppwts[r + np * e1] * Wm[r + np * e2];
            cc[e1 + ne * e2] = acc;
        }
        double rr = 0.0;
        for (int r = 0; r < np; ++r) rr += ppwts[r + np * e1] * qf[r];
        rhs[e1] = rr;
    }
    // ridge fudge: trace-scaled (matches the design fudge=1e-4 trace-scaled diag).
    double tr = 0.0;
    for (int e = 0; e < ne; ++e) tr += cc[e + ne * e];
    const double ridge = kFudge * tr / ne;
    for (int e = 0; e < ne; ++e) cc[e + ne * e] += ridge;
    // inner solve cc * bl = rhs
    double bl[kMaxNedge], lu[kMaxNedge * kMaxNedge], y[kMaxNedge];
    int piv[kMaxNedge];
    if (!d_solve(cc, ne, rhs, bl, lu, piv, y)) return 1e30;  // singular -> reject
    // res = f_obs - ppwts*bl ; score = res' Qinv res
    double res[kMaxNpair];
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

// Deterministic per-instance, per-dim multistart seed in [0,1] (a d-D splitmix sequence so
// different instances land in different basins — what distinguishes the two optimizers).
__device__ __host__ inline double d_init_theta(unsigned inst, int dim) {
    unsigned long long z = (static_cast<unsigned long long>(inst) * 0x100000001B3ULL)
                         + (static_cast<unsigned long long>(dim) * 0x9E3779B97F4A7C15ULL)
                         + 0xD1B54A32D192ED03ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<double>(z & 0xFFFFFFFFFFFFFULL) / static_cast<double>(0x10000000000000ULL);
}

static __device__ inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

// ============================ IDEA 1: batched-sequential fleet ======================
// One CUDA thread per instance. Bounded projected quasi-Newton on theta in [0,1]^D: per-
// dim forward-difference gradient + diagonal 3-point curvature, projected trust-clamped
// Newton step with backtracking. The WHOLE multistart x maxit loop is in-kernel; only the
// final {theta,score} cross back. The FLEET (N) is the parallel axis. theta_dim=1 reduces
// EXACTLY to the prior nadmix=1 IDEA1.
__global__ void idea1_fleet_kernel(Topo t, int N, const double* f_obs, const double* qinv,
                                   int maxit, double xtol, double stol,
                                   double* out_theta, double* out_score, int* out_iters,
                                   unsigned long long* eval_counter) {
    const int inst = blockIdx.x * blockDim.x + threadIdx.x;
    if (inst >= N) return;
    const int D = t.theta;
    double th[kMaxTheta];
    for (int d = 0; d < D; ++d) th[d] = clamp01(d_init_theta(static_cast<unsigned>(inst), d));
    unsigned long long evals = 0;
    const double h = 1e-4;
    double s = d_graph_gls_score(t, th, f_obs, qinv); ++evals;
    int it = 0;
    for (; it < maxit; ++it) {
        double max_dx = 0.0, max_ds = 0.0;
        // coordinate-wise projected Newton (diagonal curvature) — the faithful nadmix=1
        // reduction at D=1, a robust diagonal quasi-Newton at D>1.
        for (int d = 0; d < D; ++d) {
            const double w = th[d];
            const double wp = fmin(1.0, w + h), wm = fmax(0.0, w - h);
            double thp[kMaxTheta], thm[kMaxTheta];
            for (int k = 0; k < D; ++k) { thp[k] = th[k]; thm[k] = th[k]; }
            thp[d] = wp; thm[d] = wm;
            const double sp = d_graph_gls_score(t, thp, f_obs, qinv); ++evals;
            const double sm = d_graph_gls_score(t, thm, f_obs, qinv); ++evals;
            const double dwp = wp - w, dwm = w - wm;
            double g, curv;
            if (dwp > 0.0 && dwm > 0.0) {
                g = (sp - sm) / (dwp + dwm);
                curv = (sp - 2.0 * s + sm) / (0.5 * (dwp + dwm) * (dwp + dwm) + 1e-30);
            } else if (dwp > 0.0) { g = (sp - s) / dwp; curv = 1.0; }
            else { g = (s - sm) / dwm; curv = 1.0; }
            double step = (curv > 1e-8) ? (g / curv) : (g * 0.5);
            if (step > 0.5) step = 0.5;
            if (step < -0.5) step = -0.5;
            double wn = clamp01(w - step);
            double thn[kMaxTheta];
            for (int k = 0; k < D; ++k) thn[k] = th[k];
            thn[d] = wn;
            double sn = d_graph_gls_score(t, thn, f_obs, qinv); ++evals;
            int bt = 0;
            while (sn > s && bt < 8) {
                wn = 0.5 * (wn + w);
                thn[d] = wn;
                sn = d_graph_gls_score(t, thn, f_obs, qinv); ++evals;
                ++bt;
            }
            const double dx = fabs(wn - w);
            const double ds = fabs(sn - s);
            if (sn <= s) { th[d] = wn; s = sn; }    // accept improving coordinate move
            if (dx > max_dx) max_dx = dx;
            if (ds > max_ds) max_ds = ds;
        }
        if (max_dx < xtol * 1e-2 && max_ds < stol * 1e-3) break;
    }
    for (int d = 0; d < D; ++d) out_theta[static_cast<long>(inst) * kMaxTheta + d] = th[d];
    out_score[inst] = s;
    out_iters[inst] = it;
    atomicAdd(eval_counter, evals);
}

// ============================ WARP-COOPERATIVE PRIMITIVES ===========================
// Full warp (32 lanes). All reductions over the full warp via shuffle.
static constexpr unsigned kFullMask = 0xFFFFFFFFu;

__device__ inline double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFullMask, v, o);
    return v;
}

// Cooperative bitonic sort of the per-lane key `key` (ascending) carrying an int payload
// `idx` (the candidate's lane id), over the full 32-lane warp. After this every lane holds
// its rank-position's (key, idx). Standard shuffle-based bitonic network for 32 elements.
__device__ inline void warp_bitonic_sort(double& key, int& idx) {
    const int lane = threadIdx.x & 31;
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const double okey = __shfl_xor_sync(kFullMask, key, j);
            const int oidx = __shfl_xor_sync(kFullMask, idx, j);
            // canonical bitonic comparator: this lane keeps the MIN of the (lane, lane^j)
            // pair when ((lane&k)==0) == ((lane&j)==0), else keeps the MAX. (lane&j)==0
            // marks the low partner; (lane&k)==0 marks an ascending block. Tie-break on idx
            // keeps the sort deterministic.
            const bool keep_min = (((lane & k) == 0) == ((lane & j) == 0));
            const bool other_is_smaller = (okey < key) || (okey == key && oidx < idx);
            const bool take_other = keep_min ? other_is_smaller : !other_is_smaller;
            if (take_other) { key = okey; idx = oidx; }
        }
    }
}

// ============================ IDEA 2 (CMA): block-cooperative population =============
// ONE WARP per instance. blockDim.x = 32 (one warp/block) -> gridDim.x = N. Lane t is
// candidate t (lambda=32). The CMA state {m, sigma, C, RNG, best} is warp-owned; updated
// once per generation by warp-cooperative reductions. NO serial in-thread lambda loop.
//
// Per generation:
//  1. SAMPLE   : lane t draws z_t (curand Philox, substream = inst, offset = gen*lambda+t),
//                y_t = (B*D) z_t (B*D from the C eig, broadcast from lane 0), candidate
//                theta_t = clamp(m + sigma*y_t).
//  2. EVALUATE : score_t = d_graph_gls_score(theta_t)  (the N*lambda eval parallelism).
//  3. RANK     : warp bitonic sort of score_t -> each lane learns its sorted rank's source
//                lane; lane r<mu contributes weight w_r.
//  4. UPDATE   : m_new = sum_{r<mu} w_r * theta_(r) (warp_sum over the mu best);
//                rank-mu C update C <- (1-c_mu)C + c_mu sum_r w_r y_(r) y_(r)';
//                sigma by the weighted-variance rule. Lane 0 eigendecomposes the (<=3x3) C
//                for next gen's B*D.
//
// theta_dim<=3, so C is at most 3x3 and the eig is an in-thread cyclic Jacobi on lane 0.

// In-thread symmetric eigendecomposition (cyclic Jacobi) of an n x n (n<=3) matrix A
// (row-major, will be overwritten). Returns eigenvalues in d[], eigenvectors as columns of
// V (row-major). Tiny n -> a handful of sweeps converge to FP64.
__device__ inline void jacobi_eig(double* A, int n, double* d, double* V) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) V[i * n + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 16; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) off += A[p * n + q] * A[p * n + q];
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[p * n + q];
                if (fabs(apq) < 1e-300) continue;
                const double app = A[p * n + p], aqq = A[q * n + q];
                const double phi = 0.5 * atan2(2.0 * apq, aqq - app);
                const double c = cos(phi), s = sin(phi);
                for (int k = 0; k < n; ++k) {
                    const double akp = A[k * n + p], akq = A[k * n + q];
                    A[k * n + p] = c * akp - s * akq;
                    A[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A[p * n + k], aqk = A[q * n + k];
                    A[p * n + k] = c * apk - s * aqk;
                    A[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = V[k * n + p], vkq = V[k * n + q];
                    V[k * n + p] = c * vkp - s * vkq;
                    V[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) d[i] = A[i * n + i];
}

__global__ void __launch_bounds__(32)
idea2_cma_kernel(Topo t, int N, const double* f_obs, const double* qinv,
                 int maxgen, double xtol, double stol,
                 double* out_theta, double* out_score, int* out_gens,
                 unsigned long long* eval_counter) {
    const int inst = blockIdx.x;                 // ONE warp (block) per instance
    if (inst >= N) return;
    const int lane = threadIdx.x;                // 0..31
    const int D = t.theta;

    // --- warp-shared CMA state (one warp per block -> a single shared struct) ---
    __shared__ double s_m[kMaxTheta];            // mean
    __shared__ double s_BD[kMaxTheta * kMaxTheta];  // B*D (col-major: col j = j-th scaled eigvec)
    __shared__ double s_C[kMaxTheta * kMaxTheta];   // covariance (sym)
    __shared__ double s_sigma;
    __shared__ double s_best_s;
    __shared__ double s_best_th[kMaxTheta];
    __shared__ double s_theta_all[kLambda * kMaxTheta];  // candidate thetas (per lane)
    __shared__ double s_y_all[kLambda * kMaxTheta];      // candidate y = (B*D) z

    // recombination weights (log-decreasing, normalized over mu); mu effective.
    double w_r = 0.0;  // this lane's recomb weight if it is among the mu best (by RANK)

    if (lane == 0) {
        for (int d = 0; d < D; ++d) s_m[d] = clamp01(d_init_theta(static_cast<unsigned>(inst), d));
        s_sigma = 0.3;
        for (int i = 0; i < D * D; ++i) s_C[i] = 0.0;
        for (int d = 0; d < D; ++d) s_C[d * D + d] = 1.0;
        for (int i = 0; i < D * D; ++i) s_BD[i] = 0.0;
        for (int d = 0; d < D; ++d) s_BD[d * D + d] = 1.0;  // B*D = I initially (C=I)
        double th0[kMaxTheta];
        for (int d = 0; d < D; ++d) { th0[d] = s_m[d]; s_best_th[d] = s_m[d]; }
        s_best_s = d_graph_gls_score(t, th0, f_obs, qinv);
    }
    __syncwarp();

    // CMA rank-mu learning rate (standard heuristic, small D).
    double wsum = 0.0, wsq = 0.0;
    double wtab[kMu];
    for (int i = 0; i < kMu; ++i) { wtab[i] = log(kMu + 0.5) - log(double(i + 1)); wsum += wtab[i]; }
    for (int i = 0; i < kMu; ++i) { wtab[i] /= wsum; wsq += wtab[i] * wtab[i]; }
    const double mueff = 1.0 / wsq;
    const double c_mu = fmin(1.0 - 1e-3, mueff / ((double(D) + 2.0) * (double(D) + 2.0) + mueff));

    curandStatePhilox4_32_10_t rng;
    curand_init(0x51EEDULL, static_cast<unsigned long long>(inst),
                static_cast<unsigned long long>(lane) * 7919ULL, &rng);

    unsigned long long evals = 0;
    int gen = 0;
    for (; gen < maxgen; ++gen) {
        // 1. SAMPLE (lane = candidate). z ~ N(0,I_D); y = (B*D) z; theta = clamp(m+sigma*y).
        double z[kMaxTheta], y[kMaxTheta], th[kMaxTheta];
        for (int d = 0; d < D; ++d) {
            double2 g = curand_normal2_double(&rng);
            z[d] = g.x;  // second component discarded (D<=3, cheap; substream is per-lane)
        }
        for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int k = 0; k < D; ++k) acc += s_BD[d + D * k] * z[k];  // (B*D) z, col-major
            y[d] = acc;
            th[d] = clamp01(s_m[d] + s_sigma * acc);
        }
        // 2. EVALUATE.
        double score = d_graph_gls_score(t, th, f_obs, qinv); ++evals;
        // stash this lane's candidate + y for the cooperative recombination.
        for (int d = 0; d < D; ++d) { s_theta_all[lane * kMaxTheta + d] = th[d]; s_y_all[lane * kMaxTheta + d] = y[d]; }
        __syncwarp();
        // 3. RANK: bitonic sort (score) carrying source lane id.
        double key = score;
        int src = lane;
        warp_bitonic_sort(key, src);
        // now lane r holds the r-th smallest score and its source lane `src`. The mu best
        // are ranks 0..mu-1. Each such rank-lane contributes weight wtab[rank] to source.
        w_r = (lane < kMu) ? wtab[lane] : 0.0;
        // 4. UPDATE (warp-cooperative):
        //    m_new[d] = sum_r w_r * theta_src(r)[d]; broadcast source theta via shuffle from
        //    its home lane. We read s_theta_all[src*..] (shared) directly — src is this
        //    rank-lane's source candidate.
        double m_new[kMaxTheta];
        double Cacc[kMaxTheta * kMaxTheta];
        for (int i = 0; i < D * D; ++i) Cacc[i] = 0.0;
        for (int d = 0; d < D; ++d) {
            const double th_src = (lane < kMu) ? s_theta_all[src * kMaxTheta + d] : 0.0;
            m_new[d] = warp_sum(w_r * th_src);
        }
        // rank-mu C update accumulation: sum_r w_r * y_src y_src'  (y in the m-relative frame)
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) {
                const double yi = (lane < kMu) ? s_y_all[src * kMaxTheta + i] : 0.0;
                const double yj = (lane < kMu) ? s_y_all[src * kMaxTheta + j] : 0.0;
                Cacc[i * D + j] = warp_sum(w_r * yi * yj);
            }
        // step-size: weighted variance of the selected around the OLD mean (1-D-CMA-style,
        // generalized: sigma scales by the rms selected step length / expected).
        double var_acc = 0.0;
        for (int d = 0; d < D; ++d) {
            const double th_src = (lane < kMu) ? s_theta_all[src * kMaxTheta + d] : 0.0;
            const double dd = (lane < kMu) ? (th_src - s_m[d]) : 0.0;
            var_acc += warp_sum(w_r * dd * dd);
        }
        const double sigma_new = sqrt(var_acc / double(D)) * 1.5 + 1e-12;
        // best-so-far from rank 0 (lane 0 holds the global-min score+source).
        const double gbest_key = __shfl_sync(kFullMask, key, 0);
        const int gbest_src = __shfl_sync(kFullMask, src, 0);
        // commit (lane 0).
        if (lane == 0) {
            if (gbest_key < s_best_s) {
                s_best_s = gbest_key;
                for (int d = 0; d < D; ++d) s_best_th[d] = s_theta_all[gbest_src * kMaxTheta + d];
            }
            for (int d = 0; d < D; ++d) s_m[d] = clamp01(m_new[d]);
            // C <- (1-c_mu) C + c_mu * Cacc
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    s_C[i * D + j] = (1.0 - c_mu) * s_C[i * D + j] + c_mu * Cacc[i * D + j];
            // symmetrize + a tiny diagonal floor for numerical PD.
            for (int i = 0; i < D; ++i)
                for (int j = i + 1; j < D; ++j) {
                    const double avg = 0.5 * (s_C[i * D + j] + s_C[j * D + i]);
                    s_C[i * D + j] = avg; s_C[j * D + i] = avg;
                }
            for (int d = 0; d < D; ++d) s_C[d * D + d] += 1e-12;
            s_sigma = 0.5 * s_sigma + 0.5 * sigma_new;
            // eigendecompose C -> B*D for next gen's correlated sampling.
            double Ctmp[kMaxTheta * kMaxTheta], evals_eig[kMaxTheta], V[kMaxTheta * kMaxTheta];
            for (int i = 0; i < D * D; ++i) Ctmp[i] = s_C[i];
            jacobi_eig(Ctmp, D, evals_eig, V);
            for (int j = 0; j < D; ++j) {
                double ev = evals_eig[j];
                if (ev < 1e-20) ev = 1e-20;
                const double sq = sqrt(ev);
                for (int i = 0; i < D; ++i) s_BD[i + D * j] = V[i * D + j] * sq;  // col j scaled
            }
        }
        __syncwarp();
        if (s_sigma < xtol * 1e-2) { ++gen; break; }
    }

    if (lane == 0) {
        for (int d = 0; d < D; ++d) out_theta[static_cast<long>(inst) * kMaxTheta + d] = s_best_th[d];
        out_score[inst] = s_best_s;
        out_gens[inst] = gen;
    }
    // warp-total evals.
    unsigned long long warp_evals = static_cast<unsigned long long>(warp_sum(double(evals)) + 0.5);
    if (lane == 0) atomicAdd(eval_counter, warp_evals);
    (void)stol;
}

// ============================ IDEA 2 (DE): block-cooperative population ==============
// ONE WARP per instance; the warp IS the population (lambda candidates, one per lane). Per
// generation each lane does rand/1/bin: v = x_r0 + F*(x_r1 - x_r2) (r* drawn from the
// warp's own population, read via shuffle), binomial crossover with the lane's parent,
// evaluate, greedy 1:1 select vs the lane's OWN parent. No global sort — selection is
// per-lane-greedy. Cheaper sync than CMA, no covariance learning.
__global__ void __launch_bounds__(32)
idea2_de_kernel(Topo t, int N, const double* f_obs, const double* qinv,
                int maxgen, double xtol, double stol,
                double* out_theta, double* out_score, int* out_gens,
                unsigned long long* eval_counter) {
    const int inst = blockIdx.x;
    if (inst >= N) return;
    const int lane = threadIdx.x;
    const int D = t.theta;
    const double F = 0.6, CR = 0.9;

    __shared__ double s_pop[kLambda * kMaxTheta];

    curandStatePhilox4_32_10_t rng;
    curand_init(0xDE17ULL, static_cast<unsigned long long>(inst),
                static_cast<unsigned long long>(lane) * 6271ULL, &rng);

    // init population: per-lane perturbed multistart around the instance seed.
    double x[kMaxTheta];
    for (int d = 0; d < D; ++d) {
        const double base = d_init_theta(static_cast<unsigned>(inst), d);
        const double jitter = (curand_uniform_double(&rng) - 0.5) * 0.6;
        x[d] = clamp01(base + jitter);
        s_pop[lane * kMaxTheta + d] = x[d];
    }
    __syncwarp();
    unsigned long long evals = 0;
    double fit = d_graph_gls_score(t, x, f_obs, qinv); ++evals;
    __syncwarp();

    int gen = 0;
    for (; gen < maxgen; ++gen) {
        // pick three distinct random lanes != self.
        int r0, r1, r2;
        do { r0 = int(curand_uniform_double(&rng) * 32.0) & 31; } while (r0 == lane);
        do { r1 = int(curand_uniform_double(&rng) * 32.0) & 31; } while (r1 == lane || r1 == r0);
        do { r2 = int(curand_uniform_double(&rng) * 32.0) & 31; } while (r2 == lane || r2 == r0 || r2 == r1);
        // mutation + binomial crossover.
        double v[kMaxTheta];
        const int jrand = int(curand_uniform_double(&rng) * double(D)) % (D > 0 ? D : 1);
        for (int d = 0; d < D; ++d) {
            const double xr0 = s_pop[r0 * kMaxTheta + d];
            const double xr1 = s_pop[r1 * kMaxTheta + d];
            const double xr2 = s_pop[r2 * kMaxTheta + d];
            double trial = xr0 + F * (xr1 - xr2);
            const bool cross = (curand_uniform_double(&rng) < CR) || (d == jrand);
            v[d] = cross ? clamp01(trial) : x[d];
        }
        const double vf = d_graph_gls_score(t, v, f_obs, qinv); ++evals;
        // greedy 1:1 select.
        if (vf <= fit) { for (int d = 0; d < D; ++d) x[d] = v[d]; fit = vf; }
        __syncwarp();
        for (int d = 0; d < D; ++d) s_pop[lane * kMaxTheta + d] = x[d];
        __syncwarp();
        // convergence: warp-min vs warp-max spread of fitness (population collapse).
        double fmn = fit, fmx = fit;
        for (int o = 16; o > 0; o >>= 1) {
            fmn = fmin(fmn, __shfl_xor_sync(kFullMask, fmn, o));
            fmx = fmax(fmx, __shfl_xor_sync(kFullMask, fmx, o));
        }
        if ((fmx - fmn) < stol * 1e-3) { ++gen; break; }
    }

    // best of the population (warp-min fit + its lane's theta).
    double bkey = fit; int bsrc = lane;
    warp_bitonic_sort(bkey, bsrc);
    const double gbest = __shfl_sync(kFullMask, bkey, 0);
    const int gsrc = __shfl_sync(kFullMask, bsrc, 0);
    if (lane == 0) {
        for (int d = 0; d < D; ++d) out_theta[static_cast<long>(inst) * kMaxTheta + d] = s_pop[gsrc * kMaxTheta + d];
        out_score[inst] = gbest;
        out_gens[inst] = gen;
    }
    unsigned long long warp_evals = static_cast<unsigned long long>(warp_sum(double(evals)) + 0.5);
    if (lane == 0) atomicAdd(eval_counter, warp_evals);
    (void)xtol;
}

// ============================ HOST: fixture reader + basis ==========================
namespace {

struct Fixture {
    int P = 0, nb = 0;
    std::vector<int> block_sizes;
    std::vector<double> f2;  // i + P*j + P*P*b (column-major slabs)
};

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

// HOST oracle (per nadmix): f_obs (block-size-weighted total outgroup-f3) + Q (block-
// jackknife cov, diag_f3) + Qinv (small_linalg inverse). C=Mbuti(3).
struct Basis {
    int nadmix = 0, npair = 0;
    std::vector<double> f_obs;   // npair
    std::vector<double> qinv;    // npair x npair column-major
    bool ok = false;
};

Basis build_basis(const Fixture& fx, const Topo& t) {
    Basis B; B.nadmix = t.nadmix; B.npair = t.npair;
    const int P = fx.P, nb = fx.nb;
    const int C = 3;
    const int L = t.L, np = t.npair;
    auto F2 = [&](int i, int j, int b) -> double {
        return fx.f2[static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * j +
                     static_cast<std::size_t>(P) * P * b];
    };
    std::vector<int> pa(np), pb(np);
    { int idx = 0; for (int a = 0; a < L; ++a) for (int b = a; b < L; ++b) { pa[idx] = a; pb[idx] = b; ++idx; } }
    std::vector<double> fb(static_cast<std::size_t>(nb) * np);
    double n_tot = 0.0;
    for (int b = 0; b < nb; ++b) n_tot += fx.block_sizes[static_cast<std::size_t>(b)];
    for (int b = 0; b < nb; ++b)
        for (int k = 0; k < np; ++k) {
            const int i = t.leaves[pa[k]], j = t.leaves[pb[k]];
            fb[static_cast<std::size_t>(b) * np + k] = 0.5 * (F2(C, i, b) + F2(C, j, b) - F2(i, j, b));
        }
    B.f_obs.assign(np, 0.0);
    std::vector<double> tot_sum(np, 0.0);
    for (int b = 0; b < nb; ++b)
        for (int k = 0; k < np; ++k)
            tot_sum[k] += fx.block_sizes[static_cast<std::size_t>(b)] * fb[static_cast<std::size_t>(b) * np + k];
    for (int k = 0; k < np; ++k) B.f_obs[k] = tot_sum[k] / n_tot;
    std::vector<double> loo(static_cast<std::size_t>(nb) * np);
    for (int b = 0; b < nb; ++b) {
        const double nb_b = fx.block_sizes[static_cast<std::size_t>(b)];
        for (int k = 0; k < np; ++k)
            loo[static_cast<std::size_t>(b) * np + k] =
                (tot_sum[k] - nb_b * fb[static_cast<std::size_t>(b) * np + k]) / (n_tot - nb_b);
    }
    std::vector<double> mean(np, 0.0);
    for (int b = 0; b < nb; ++b) for (int k = 0; k < np; ++k) mean[k] += loo[static_cast<std::size_t>(b) * np + k];
    for (int k = 0; k < np; ++k) mean[k] /= nb;
    std::vector<double> cov(static_cast<std::size_t>(np) * np, 0.0);
    for (int b = 0; b < nb; ++b)
        for (int a = 0; a < np; ++a) {
            const double da = loo[static_cast<std::size_t>(b) * np + a] - mean[a];
            for (int c = 0; c < np; ++c) {
                const double dc = loo[static_cast<std::size_t>(b) * np + c] - mean[c];
                cov[static_cast<std::size_t>(a) + np * c] += da * dc;
            }
        }
    const double scl = static_cast<double>(nb - 1) / static_cast<double>(nb);
    for (auto& v : cov) v *= scl;
    for (int k = 0; k < np; ++k) cov[static_cast<std::size_t>(k) + np * k] += kDiagF3;
    std::vector<double> qinv;
    const steppe::core::LinAlgStatus st = steppe::core::inverse(cov, np, qinv);
    if (!st.ok) { std::printf("ERROR: covariance singular (inverse failed) nadmix=%d\n", t.nadmix); return B; }
    B.qinv = std::move(qinv);
    B.ok = true;
    return B;
}

// HOST objective mirror (identical algebra to d_graph_gls_score) — the ground-truth scorer.
double host_score(const Basis& B, const Topo& t, const double* th) {
    const int np = t.npair, ne = t.nedge, L = t.L;
    std::vector<double> ppwts(static_cast<std::size_t>(np) * ne, 0.0);
    double wa[kMaxNedge], wb[kMaxNedge];
    int idx = 0;
    for (int a = 0; a < L; ++a) for (int b = a; b < L; ++b) {
        d_leafweights(t.nadmix, a, th, wa);
        d_leafweights(t.nadmix, b, th, wb);
        for (int e = 0; e < ne; ++e) ppwts[static_cast<std::size_t>(idx) + np * e] = wa[e] * wb[e];
        ++idx;
    }
    std::vector<double> cc(static_cast<std::size_t>(ne) * ne, 0.0), rhs(ne, 0.0);
    for (int e1 = 0; e1 < ne; ++e1) {
        for (int e2 = 0; e2 < ne; ++e2) {
            double acc = 0;
            for (int r = 0; r < np; ++r) {
                double qp = 0; for (int c = 0; c < np; ++c) qp += B.qinv[static_cast<std::size_t>(r) + np * c] * ppwts[static_cast<std::size_t>(c) + np * e2];
                acc += ppwts[static_cast<std::size_t>(r) + np * e1] * qp;
            }
            cc[static_cast<std::size_t>(e1) + ne * e2] = acc;
        }
        double rr = 0;
        for (int r = 0; r < np; ++r) {
            double qf = 0; for (int c = 0; c < np; ++c) qf += B.qinv[static_cast<std::size_t>(r) + np * c] * B.f_obs[c];
            rr += ppwts[static_cast<std::size_t>(r) + np * e1] * qf;
        }
        rhs[e1] = rr;
    }
    double tr = 0; for (int e = 0; e < ne; ++e) tr += cc[static_cast<std::size_t>(e) + ne * e];
    for (int e = 0; e < ne; ++e) cc[static_cast<std::size_t>(e) + ne * e] += kFudge * tr / ne;
    std::vector<double> bl;
    if (!steppe::core::solve(cc, ne, rhs, bl).ok) return 1e30;
    std::vector<double> res(np);
    for (int r = 0; r < np; ++r) {
        double pb = 0; for (int e = 0; e < ne; ++e) pb += ppwts[static_cast<std::size_t>(r) + np * e] * bl[static_cast<std::size_t>(e)];
        res[static_cast<std::size_t>(r)] = B.f_obs[r] - pb;
    }
    double sc = 0;
    for (int a = 0; a < np; ++a) {
        double row = 0; for (int b = 0; b < np; ++b) row += B.qinv[static_cast<std::size_t>(a) + np * b] * res[static_cast<std::size_t>(b)];
        sc += res[static_cast<std::size_t>(a)] * row;
    }
    return sc;
}

// HOST grid-pin of (theta*, score*) — a coarse grid over [0,1]^D then a local Nelder-Mead
// refine. The ground truth the GPU must hit.
void host_pin(const Basis& B, const Topo& t, std::vector<double>& theta_star, double& score_star) {
    const int D = t.theta;
    theta_star.assign(D, 0.0);
    score_star = 1e300;
    // coarse grid.
    const int g = (D == 1) ? 2000 : (D == 2 ? 80 : 26);
    std::vector<double> th(D, 0.0);
    std::vector<int> ii(D, 0);
    const long total = [&]{ long p = 1; for (int d = 0; d < D; ++d) p *= (g + 1); return p; }();
    for (long lin = 0; lin < total; ++lin) {
        long r = lin;
        for (int d = 0; d < D; ++d) { ii[d] = int(r % (g + 1)); r /= (g + 1); th[d] = double(ii[d]) / double(g); }
        const double s = host_score(B, t, th.data());
        if (s < score_star) { score_star = s; theta_star = th; }
    }
    // local Nelder-Mead refine around the grid best.
    const int n = D;
    std::vector<std::vector<double>> simplex(n + 1, theta_star);
    for (int i = 0; i < n; ++i) simplex[i + 1][i] = std::min(1.0, std::max(0.0, theta_star[i] + 0.02));
    std::vector<double> fval(n + 1);
    auto clampv = [](std::vector<double>& v) { for (auto& x : v) x = std::min(1.0, std::max(0.0, x)); };
    for (int i = 0; i <= n; ++i) { clampv(simplex[i]); fval[i] = host_score(B, t, simplex[i].data()); }
    for (int it = 0; it < 400; ++it) {
        std::vector<int> ord(n + 1); for (int i = 0; i <= n; ++i) ord[i] = i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return fval[a] < fval[b]; });
        std::vector<std::vector<double>> sx(n + 1); std::vector<double> sf(n + 1);
        for (int i = 0; i <= n; ++i) { sx[i] = simplex[ord[i]]; sf[i] = fval[ord[i]]; }
        simplex = sx; fval = sf;
        if (std::fabs(fval[n] - fval[0]) < 1e-15) break;
        std::vector<double> cen(n, 0.0);
        for (int i = 0; i < n; ++i) for (int d = 0; d < n; ++d) cen[d] += simplex[i][d];
        for (int d = 0; d < n; ++d) cen[d] /= n;
        std::vector<double> refl(n); for (int d = 0; d < n; ++d) refl[d] = cen[d] + 1.0 * (cen[d] - simplex[n][d]);
        clampv(refl); double fr = host_score(B, t, refl.data());
        if (fr < fval[0]) {
            std::vector<double> exp(n); for (int d = 0; d < n; ++d) exp[d] = cen[d] + 2.0 * (cen[d] - simplex[n][d]);
            clampv(exp); double fe = host_score(B, t, exp.data());
            if (fe < fr) { simplex[n] = exp; fval[n] = fe; } else { simplex[n] = refl; fval[n] = fr; }
        } else if (fr < fval[n - 1]) { simplex[n] = refl; fval[n] = fr; }
        else {
            std::vector<double> con(n); for (int d = 0; d < n; ++d) con[d] = cen[d] + 0.5 * (simplex[n][d] - cen[d]);
            clampv(con); double fc = host_score(B, t, con.data());
            if (fc < fval[n]) { simplex[n] = con; fval[n] = fc; }
            else { for (int i = 1; i <= n; ++i) { for (int d = 0; d < n; ++d) simplex[i][d] = 0.5 * (simplex[0][d] + simplex[i][d]); clampv(simplex[i]); fval[i] = host_score(B, t, simplex[i].data()); } }
        }
    }
    int bi = 0; for (int i = 1; i <= n; ++i) if (fval[i] < fval[bi]) bi = i;
    if (fval[bi] < score_star) { score_star = fval[bi]; theta_star = simplex[bi]; }
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::printf("CUDA ERROR (%s): %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

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
    std::vector<int> nadmixes;
    if (argc > 5) {
        std::string s = argv[5]; size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p); if (!tok.empty()) nadmixes.push_back(std::atoi(tok.c_str())); if (q == std::string::npos) break; p = q + 1; }
    } else { nadmixes = {1, 2, 3}; }
    std::vector<int> opts;
    if (argc > 6) {
        std::string s = argv[6]; size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); std::string tok = s.substr(p, q == std::string::npos ? std::string::npos : q - p); if (!tok.empty()) opts.push_back(std::atoi(tok.c_str())); if (q == std::string::npos) break; p = q + 1; }
    } else { opts = {1, 2, 3}; }

    std::printf("=== bench_optimizers — FAIR RE-MATCH (IDEA1 fleet vs PROPER IDEA2 CMA/DE) ===\n");
    std::printf("fixture = %s\n", fixture.c_str());

    Fixture fx;
    if (!read_fixture(fixture, fx)) return 1;
    std::printf("fixture: P=%d nb=%d\n", fx.P, fx.nb);
    if (fx.P < 9) { std::printf("ERROR: need P>=9 (the 9-pop topologies)\n"); return 1; }

    const char* opt_name[4] = {"?", "IDEA1", "IDEA2-CMA", "IDEA2-DE"};

    for (int nadmix : nadmixes) {
        const Topo t = make_topo(nadmix);
        std::printf("\n#################### nadmix=%d  (L=%d npair=%d nedge=%d theta_dim=%d) ####################\n",
                    nadmix, t.L, t.npair, t.nedge, t.theta);

        const Basis B = build_basis(fx, t);
        if (!B.ok) return 1;
        std::vector<double> theta_star; double score_star = 0;
        host_pin(B, t, theta_star, score_star);
        std::printf("GROUND TRUTH (host grid-pin + Nelder-Mead): theta* = [");
        for (int d = 0; d < t.theta; ++d) std::printf("%s%.6f", d ? ", " : "", theta_star[d]);
        std::printf("]   score* = %.6f\n", score_star);

        // device basis (resident, shared by every instance / all optimizers).
        double *d_fobs = nullptr, *d_qinv = nullptr;
        ck(cudaMalloc(&d_fobs, t.npair * sizeof(double)), "malloc fobs");
        ck(cudaMalloc(&d_qinv, static_cast<std::size_t>(t.npair) * t.npair * sizeof(double)), "malloc qinv");
        ck(cudaMemcpy(d_fobs, B.f_obs.data(), t.npair * sizeof(double), cudaMemcpyHostToDevice), "h2d fobs");
        ck(cudaMemcpy(d_qinv, B.qinv.data(), static_cast<std::size_t>(t.npair) * t.npair * sizeof(double), cudaMemcpyHostToDevice), "h2d qinv");

        const double wtol = tol, stol = tol;

        auto run_one = [&](int which, long N) -> double {
            double *d_theta, *d_score; int *d_iters; unsigned long long *d_evals;
            ck(cudaMalloc(&d_theta, static_cast<std::size_t>(N) * kMaxTheta * sizeof(double)), "malloc theta");
            ck(cudaMalloc(&d_score, N * sizeof(double)), "malloc score");
            ck(cudaMalloc(&d_iters, N * sizeof(int)), "malloc iters");
            ck(cudaMalloc(&d_evals, sizeof(unsigned long long)), "malloc evals");
            ck(cudaMemset(d_evals, 0, sizeof(unsigned long long)), "memset evals");
            auto launch = [&]() {
                if (which == 1) {
                    const int TPB = 256;
                    const int blocks = static_cast<int>((N + TPB - 1) / TPB);
                    idea1_fleet_kernel<<<blocks, TPB>>>(t, static_cast<int>(N), d_fobs, d_qinv, maxit, wtol, stol, d_theta, d_score, d_iters, d_evals);
                } else if (which == 2) {
                    idea2_cma_kernel<<<static_cast<int>(N), 32>>>(t, static_cast<int>(N), d_fobs, d_qinv, maxit, wtol, stol, d_theta, d_score, d_iters, d_evals);
                } else {
                    idea2_de_kernel<<<static_cast<int>(N), 32>>>(t, static_cast<int>(N), d_fobs, d_qinv, maxit, wtol, stol, d_theta, d_score, d_iters, d_evals);
                }
            };
            // WARM-UP (not timed).
            launch();
            ck(cudaDeviceSynchronize(), "warmup sync");
            ck(cudaMemset(d_evals, 0, sizeof(unsigned long long)), "memset evals2");
            // TIMED.
            const auto t0 = std::chrono::steady_clock::now();
            launch();
            ck(cudaDeviceSynchronize(), "timed sync");
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::vector<double> hth(static_cast<std::size_t>(N) * kMaxTheta), hs(N);
            unsigned long long hevals = 0;
            ck(cudaMemcpy(hth.data(), d_theta, static_cast<std::size_t>(N) * kMaxTheta * sizeof(double), cudaMemcpyDeviceToHost), "d2h theta");
            ck(cudaMemcpy(hs.data(), d_score, N * sizeof(double), cudaMemcpyDeviceToHost), "d2h score");
            ck(cudaMemcpy(&hevals, d_evals, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "d2h evals");
            long conv = 0;
            std::vector<double> therr(N);
            for (long i = 0; i < N; ++i) {
                double e2 = 0.0;
                for (int d = 0; d < t.theta; ++d) { const double e = hth[static_cast<std::size_t>(i) * kMaxTheta + d] - theta_star[d]; e2 += e * e; }
                const double terr = std::sqrt(e2);
                therr[static_cast<std::size_t>(i)] = terr;
                if (terr < tol && std::fabs(hs[static_cast<std::size_t>(i)] - score_star) < tol) ++conv;
            }
            std::sort(therr.begin(), therr.end());
            const double med = therr[therr.size() / 2];
            const double conv_frac = static_cast<double>(conv) / static_cast<double>(N);
            std::printf("%-10s | %-7ld | %14.2f | %16llu | %11.4f%% | %10d | %.3e\n",
                        opt_name[which], N, ms, static_cast<unsigned long long>(hevals), conv_frac * 100.0, 1, med);
            cudaFree(d_theta); cudaFree(d_score); cudaFree(d_iters); cudaFree(d_evals);
            return conv_frac;
        };

        // SANITY (small N): each selected optimizer must converge to the checkable optimum.
        std::printf("\n--- SANITY (N=256): convergence to host-pinned optimum ---\n");
        std::printf("%-10s | %-7s | %14s | %16s | %12s | %10s | %s\n",
                    "OPT", "N", "wall(ms)", "obj_evals", "conv_frac", "dispatch", "median ||theta-theta*||");
        std::printf("-----------+---------+----------------+------------------+--------------+------------+--------------\n");
        for (int which : opts) {
            const double cf = run_one(which, 256);
            if (cf < 0.50) std::printf("  [WARN] %s converged only %.1f%% at N=256 (nadmix=%d)\n", opt_name[which], cf * 100.0, nadmix);
        }

        // SCALE SWEEP.
        std::printf("\n--- SCALE SWEEP (single GPU, device 0) ---\n");
        std::printf("%-10s | %-7s | %14s | %16s | %12s | %10s | %s\n",
                    "OPT", "N", "wall(ms)", "obj_evals", "conv_frac", "dispatch", "median ||theta-theta*||");
        std::printf("-----------+---------+----------------+------------------+--------------+------------+--------------\n");
        for (long N : Ns) for (int which : opts) run_one(which, N);

        cudaFree(d_fobs); cudaFree(d_qinv);
    }

    std::printf("\nNOTE: sample GPU util with `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 100` in parallel.\n");
    return 0;
}
