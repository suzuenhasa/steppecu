// tests/reference/test_pca_eig.cu
//
// GPU reference test of CudaBackend::pca_covariance_eig on a TINY in-memory 2-cluster tile
// (no AADR, no golden files). It drives the standalone genotype-PCA device path
// (Patterson standardize -> cuBLAS SYRK covariance -> cuSOLVER Dsyevd eigen -> top-K
// projection) and asserts:
//   (1) the monomorphic SNP is excluded (n_snp_used / n_snp_monomorphic are exact);
//   (2) trace invariance — Σ eigenvalues == Σ_{i,s} Z_is^2 computed independently on the
//       host from the SAME Patterson standardization (a numpy-free absolute-scale anchor);
//   (3) the GPU coords reproduce the independent CpuBackend long-double Jacobi oracle
//       per-PC sign-aligned (|Pearson r| ~ 1), the arbitrary-sign / rotation-robust metric;
//   (4) PC1 separates the two clusters (samples 0,1 share a sign, 2,3 the opposite);
//   (5) var_explained is descending and sums to ~1.
//
// SKIPs cleanly (exit 0) when no CUDA device is visible. Self-checking main().
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "steppe/config.hpp"
#include "core/internal/pca_standardize.hpp"
#include "core/internal/decode_af.hpp"

namespace {

using steppe::DecodeTileView;
using steppe::PcaEig;
using steppe::Precision;

int g_failures = 0;
void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}
void check_close(const char* what, double got, double want, double atol) {
    if (!(std::fabs(got - want) <= atol)) {
        std::printf("  [FAIL] %-34s got=% .10e want=% .10e |d|=% .2e\n", what, got, want,
                    std::fabs(got - want));
        ++g_failures;
    }
}

constexpr int N = 4;   // samples
constexpr int M = 8;   // SNPs
constexpr std::uint8_t X = steppe::core::kMissingGenotypeCode;

// The 2-cluster genotype matrix G[g][s] (diploid dosage 0/1/2, X = missing). SNP6 is
// monomorphic (all 2 -> dropped); SNP7 carries a missing genotype at g0.
const std::uint8_t G[N][M] = {
    {0, 0, 0, 1, 2, 2, 2, X},   // g0 (cluster A)
    {0, 0, 0, 0, 2, 2, 2, 1},   // g1 (cluster A)
    {2, 2, 2, 1, 0, 0, 2, 2},   // g2 (cluster B)
    {2, 2, 2, 2, 0, 0, 2, 1},   // g3 (cluster B)
};

// |Pearson r| of two length-N vectors (columns extracted per PC).
double abs_pearson(const std::vector<double>& a, const std::vector<double>& b) {
    double ma = 0, mb = 0;
    for (int i = 0; i < N; ++i) { ma += a[i]; mb += b[i]; }
    ma /= N; mb /= N;
    double sab = 0, saa = 0, sbb = 0;
    for (int i = 0; i < N; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    if (saa <= 0 || sbb <= 0) return 1.0;  // a constant PC column trivially "aligns"
    return std::fabs(sab / std::sqrt(saa * sbb));
}

std::vector<double> pc_column(const PcaEig& e, int k) {
    std::vector<double> col(N);
    for (int i = 0; i < N; ++i) col[i] = e.coords[static_cast<std::size_t>(i) * e.K + k];
    return col;
}

}  // namespace

int main() {
    std::printf("=== steppe pca device eigen (tiny 2-cluster tile) ===\n");
    if (steppe::device::visible_device_count() <= 0) {
        std::printf("\nRESULT: SKIP (no CUDA device visible)\n");
        return 0;
    }

    // Pack the tile individual-major (2-bit codes, MSB-first, bytes_per_record = ceil(M/4)).
    const std::size_t bpr = (static_cast<std::size_t>(M) + 3) / 4;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(N) * bpr, 0);
    for (int g = 0; g < N; ++g)
        for (int s = 0; s < M; ++s) {
            const std::size_t byte = static_cast<std::size_t>(s) / 4;
            const int pos = s % 4;
            const int shift = (3 - pos) * 2;
            packed[static_cast<std::size_t>(g) * bpr + byte] |=
                static_cast<std::uint8_t>(G[g][s] << shift);
        }
    std::vector<std::size_t> pop_offsets = {0, static_cast<std::size_t>(N)};
    std::vector<int> ploidy(N, steppe::core::kPloidyDiploid);

    DecodeTileView view;
    view.packed = packed.data();
    view.bytes_per_record = bpr;
    view.n_snp = static_cast<std::size_t>(M);
    view.n_individuals = static_cast<std::size_t>(N);
    view.pop_offsets = pop_offsets.data();
    view.n_pop = 1;
    view.sample_ploidy = ploidy.data();
    view.ploidy = 2;

    const int K = N;  // full spectrum so we can check the trace
    const Precision precision = Precision::emulated_fp64();

    auto cuda = steppe::device::make_cuda_backend(0);
    auto cpu = steppe::device::make_cpu_backend();
    const PcaEig gpu = cuda->pca_covariance_eig(view, K, precision);
    const PcaEig ref = cpu->pca_covariance_eig(view, K, precision);

    check_true("gpu status ok", gpu.status == steppe::Status::Ok);
    check_true("gpu N==4", gpu.N == N);
    check_true("gpu K==4", gpu.K == K);
    check_true("n_snp_used == 7 (SNP6 monomorphic dropped)", gpu.n_snp_used == 7);
    check_true("n_snp_monomorphic == 1", gpu.n_snp_monomorphic == 1);

    // (2) Trace invariance: Σ eigenvalues == Σ_{i,s} Z_is^2 (independent host standardization).
    double host_trace = 0.0;
    for (int s = 0; s < M; ++s) {
        long ac = 0, nn = 0;
        for (int g = 0; g < N; ++g)
            if (G[g][s] != X) { ac += G[g][s]; ++nn; }
        const steppe::core::PcaSnpScale sc = steppe::core::pca_snp_scale(ac, nn);
        const double inv = sc.used ? (1.0 / std::sqrt(sc.pq)) : 0.0;
        for (int g = 0; g < N; ++g) {
            const double z = steppe::core::pca_standardize_one(G[g][s], sc.center, inv);
            host_trace += z * z;
        }
    }
    double eig_sum = 0.0;
    for (int k = 0; k < K; ++k) eig_sum += gpu.eigenvalues[static_cast<std::size_t>(k)];
    check_close("trace == Σ eigenvalues", eig_sum, host_trace, 1e-4 * (host_trace + 1.0));

    // (3) GPU vs CPU-oracle per-PC sign-aligned |r| (the top-3 informative PCs; PC4 ~ 0 lives
    // in the all-ones null space so it is skipped).
    for (int k = 0; k < K - 1; ++k) {
        const double r = abs_pearson(pc_column(gpu, k), pc_column(ref, k));
        char lbl[48];
        std::snprintf(lbl, sizeof(lbl), "PC%d |r| gpu~cpu-oracle ~ 1", k + 1);
        check_true(lbl, r > 0.999);
    }

    // (4) PC1 separates the two clusters.
    const std::vector<double> pc1 = pc_column(gpu, 0);
    check_true("PC1 cluster A shares a sign", (pc1[0] > 0) == (pc1[1] > 0));
    check_true("PC1 cluster B shares a sign", (pc1[2] > 0) == (pc1[3] > 0));
    check_true("PC1 separates A from B", (pc1[0] > 0) != (pc1[2] > 0));

    // (5) var_explained descending, sums to ~1.
    double ve_sum = 0.0;
    bool descending = true;
    for (int k = 0; k < K; ++k) {
        ve_sum += gpu.var_explained[static_cast<std::size_t>(k)];
        if (k > 0 && gpu.var_explained[static_cast<std::size_t>(k)] >
                         gpu.var_explained[static_cast<std::size_t>(k - 1)] + 1e-9)
            descending = false;
    }
    check_true("var_explained descending", descending);
    check_close("var_explained sums to 1", ve_sum, 1.0, 1e-6);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (device PCA eigen: monomorphic drop, trace, oracle |r|, "
                    "cluster split)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
