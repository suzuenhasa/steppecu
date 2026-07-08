// tests/reference/test_pca_truncated_eig.cu
//
// Direct GPU unit test of the randomized truncated top-K eigensolver
// steppe::device::pca_truncated_topk on PLANTED-SPECTRUM SPD matrices (no genotype path, no
// AADR). Builds C = W Λ Wᵀ with a known orthonormal W and a chosen Λ, uploads it column-major,
// and asserts the truncated solve recovers the top-K eigenpairs:
//   (A) WELL-SEPARATED spectrum (Λ top-4 = 100,50,25,10, decaying tail): the K=4 Ritz values
//       match the planted top-4 tightly, and each lifted Ritz vector is sign-aligned to its
//       planted eigenvector (|Pearson r| > 0.999) — the unique-eigenvector regime.
//   (B) DEGENERATE PAIR (Λ = 100,50,10,10,…): the eigenVECTORS in the 2-D degenerate eigenspace
//       are NOT unique (any orthonormal rotation is valid), so we assert the SPANNED SUBSPACE
//       instead — each computed degenerate vector lies in the planted 2-D eigenspace
//       (‖Pᵀv‖² > 0.999) — plus the (well-defined) eigenVALUES match.
//
// SKIPs cleanly (exit 0) when no CUDA device is visible. Self-checking main().
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "device/backend_factory.hpp"  // visible_device_count
#include "device/cuda/check.cuh"       // STEPPE_CUDA_CHECK
#include "device/cuda/device_buffer.cuh"
#include "device/cuda/handles.hpp"
#include "device/cuda/pca_truncated_eig.cuh"
#include "steppe/config.hpp"

namespace {

using steppe::Precision;
using steppe::Status;
using steppe::device::CublasHandle;
using steppe::device::CusolverDnHandle;
using steppe::device::DeviceBuffer;
using steppe::device::PcaTruncatedEig;
using steppe::device::pca_truncated_topk;

int g_failures = 0;
void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}
void check_rel(const char* what, double got, double want, double rtol) {
    const double tol = rtol * (std::fabs(want) + 1e-300);
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("  [FAIL] %-38s got=% .10e want=% .10e rel=% .2e\n", what, got, want,
                    std::fabs(got - want) / (std::fabs(want) + 1e-300));
        ++g_failures;
    }
}

constexpr int N = 64;

// Deterministic LCG in [-1,1) — reproducible planted-matrix seed material.
struct Lcg {
    unsigned long long s;
    double next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(s >> 11) / 9007199254740992.0) * 2.0 - 1.0;
    }
};

// Build an N x N orthonormal W (column-major: W[i + k*N] = component i of eigenvector k) via
// modified Gram-Schmidt on a deterministic random matrix.
std::vector<double> planted_orthonormal(unsigned long long seed) {
    std::vector<double> W(static_cast<std::size_t>(N) * N);
    Lcg rng{seed};
    for (auto& v : W) v = rng.next();
    for (int k = 0; k < N; ++k) {
        double* wk = &W[static_cast<std::size_t>(k) * N];
        for (int j = 0; j < k; ++j) {
            const double* wj = &W[static_cast<std::size_t>(j) * N];
            double dot = 0.0;
            for (int i = 0; i < N; ++i) dot += wk[i] * wj[i];
            for (int i = 0; i < N; ++i) wk[i] -= dot * wj[i];
        }
        double nrm = 0.0;
        for (int i = 0; i < N; ++i) nrm += wk[i] * wk[i];
        nrm = std::sqrt(nrm);
        for (int i = 0; i < N; ++i) wk[i] /= nrm;
    }
    return W;
}

// C = W diag(lambda) Wᵀ  (column-major N x N).
std::vector<double> planted_covariance(const std::vector<double>& W,
                                       const std::vector<double>& lambda) {
    std::vector<double> C(static_cast<std::size_t>(N) * N, 0.0);
    for (int k = 0; k < N; ++k) {
        const double lk = lambda[static_cast<std::size_t>(k)];
        const double* wk = &W[static_cast<std::size_t>(k) * N];
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                C[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N] +=
                    lk * wk[i] * wk[j];
    }
    return C;
}

double abs_pearson(const double* a, const double* b) {
    double ma = 0, mb = 0;
    for (int i = 0; i < N; ++i) { ma += a[i]; mb += b[i]; }
    ma /= N; mb /= N;
    double sab = 0, saa = 0, sbb = 0;
    for (int i = 0; i < N; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    if (saa <= 0 || sbb <= 0) return 1.0;
    return std::fabs(sab / std::sqrt(saa * sbb));
}

// Run the solver, return (evecL N x L column-major, evalL length L, actual L).
struct Result { std::vector<double> evec; std::vector<double> eval; int L; Status status; };

Result run(CublasHandle& blas, CusolverDnHandle& solver, cudaStream_t stream,
           const std::vector<double>& C, int K, int oversample, int iters) {
    DeviceBuffer<double> dC(static_cast<std::size_t>(N) * N);
    steppe::device::h2d_async(dC, C.data(), C.size(), stream);
    const int L = std::min(K + oversample, N);
    DeviceBuffer<double> dEvec(static_cast<std::size_t>(N) * L);
    DeviceBuffer<double> dEval(static_cast<std::size_t>(L));
    int L_used = 0;
    const PcaTruncatedEig te = pca_truncated_topk(
        blas.get(), solver.get(), stream, dC.data(), N, K, oversample, iters,
        Precision::emulated_fp64(), dEvec.data(), dEval.data(), &L_used);
    Result r;
    r.status = te.status;
    r.L = te.L;
    r.evec.assign(static_cast<std::size_t>(N) * te.L, 0.0);
    r.eval.assign(static_cast<std::size_t>(te.L), 0.0);
    steppe::device::d2h_async(r.evec.data(), dEvec,
                              static_cast<std::size_t>(N) * te.L, stream);
    steppe::device::d2h_async(r.eval.data(), dEval, static_cast<std::size_t>(te.L), stream);
    STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream));
    return r;
}

}  // namespace

int main() {
    std::printf("=== steppe pca truncated top-K eigensolver (planted spectrum) ===\n");
    if (steppe::device::visible_device_count() <= 0) {
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }
    STEPPE_CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream = nullptr;
    STEPPE_CUDA_CHECK(cudaStreamCreate(&stream));
    CublasHandle blas;
    CusolverDnHandle solver;
    blas.set_stream(stream);
    solver.set_stream(stream);

    const std::vector<double> W = planted_orthonormal(0xC0FFEEULL);

    // --- (A) Well-separated: top-4 = 100,50,25,10; decaying tail -------------------
    {
        std::vector<double> lambda(N);
        const double sep[4] = {100.0, 50.0, 25.0, 10.0};
        for (int k = 0; k < 4; ++k) lambda[static_cast<std::size_t>(k)] = sep[k];
        for (int k = 4; k < N; ++k)
            lambda[static_cast<std::size_t>(k)] = 5.0 * std::pow(0.85, k - 4);  // 5,4.25,… < 10
        const std::vector<double> C = planted_covariance(W, lambda);

        const int K = 4;
        const Result r = run(blas, solver, stream, C, K, /*oversample=*/16, /*iters=*/2);
        check_true("A: status ok", r.status == Status::Ok);
        check_true("A: L == 20", r.L == 20);
        for (int k = 0; k < K; ++k) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "A: PC%d Ritz value ~ planted", k + 1);
            check_rel(lbl, r.eval[static_cast<std::size_t>(r.L - 1 - k)], sep[k], 1e-6);
            // Sign-aligned |Pearson r| ~ 1 vs planted eigenvector k (unique here).
            std::snprintf(lbl, sizeof(lbl), "A: PC%d |r| vs planted ~ 1", k + 1);
            check_true(lbl, abs_pearson(&r.evec[static_cast<std::size_t>(r.L - 1 - k) * N],
                                        &W[static_cast<std::size_t>(k) * N]) > 0.999);
        }
    }

    // --- (B) Degenerate pair: 100, 50, 10, 10 (PC3/PC4 span a 2-D eigenspace) --------
    {
        std::vector<double> lambda(N);
        lambda[0] = 100.0; lambda[1] = 50.0; lambda[2] = 10.0; lambda[3] = 10.0;
        for (int k = 4; k < N; ++k)
            lambda[static_cast<std::size_t>(k)] = 3.0 * std::pow(0.85, k - 4);  // < 10
        const std::vector<double> C = planted_covariance(W, lambda);

        const int K = 4;
        const Result r = run(blas, solver, stream, C, K, /*oversample=*/16, /*iters=*/3);
        check_true("B: status ok", r.status == Status::Ok);
        // Eigenvalues are well-defined even where the eigenvectors are not.
        check_rel("B: PC1 eigenvalue", r.eval[static_cast<std::size_t>(r.L - 1)], 100.0, 1e-6);
        check_rel("B: PC2 eigenvalue", r.eval[static_cast<std::size_t>(r.L - 2)], 50.0, 1e-6);
        check_rel("B: PC3 eigenvalue", r.eval[static_cast<std::size_t>(r.L - 3)], 10.0, 1e-4);
        check_rel("B: PC4 eigenvalue", r.eval[static_cast<std::size_t>(r.L - 4)], 10.0, 1e-4);
        // SUBSPACE check: each computed degenerate vector lies in span{W col2, W col3}.
        for (int j = 2; j <= 3; ++j) {
            const double* v = &r.evec[static_cast<std::size_t>(r.L - 1 - j) * N];
            double c2 = 0.0, c3 = 0.0, vv = 0.0;
            for (int i = 0; i < N; ++i) {
                c2 += v[i] * W[static_cast<std::size_t>(2) * N + i];
                c3 += v[i] * W[static_cast<std::size_t>(3) * N + i];
                vv += v[i] * v[i];
            }
            const double frac = (c2 * c2 + c3 * c3) / (vv + 1e-300);  // ‖Pᵀv‖²/‖v‖²
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "B: PC%d in degenerate span", j + 1);
            check_true(lbl, frac > 0.999);
        }
    }

    cudaStreamDestroy(stream);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (truncated top-K: separated eigenpairs + degenerate "
                    "subspace)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
