// tests/reference/test_li_stephens_localanc.cu
//
// Li-Stephens LOCAL-ANCESTRY adapter gate (the `steppe paint --face localanc` FACE,
// Phase 3). Validates that the fused per-SNP localanc sink (post[(r*M+l)*P+g] =
// sum_{k:group(k)==g} gamma_l(k)) is the correct deterministic reduction of a correct
// posterior. Arms:
//
//   (A) STRUCTURAL — every informative column sums to 1. For the CpuBackend sink over
//       each golden recipient with a synthetic partition group(k) = k % P (P=3; physical
//       meaning irrelevant, exactly as the coancestry test uses a synthetic w): each SNP
//       column of the posterior sums to ~1 (gamma is column-normalized and the labels
//       partition the donors). A degenerate column whose FB denominator underflows to 0
//       is deliberately zeroed by the engine, so the invariant is sum in {~0, ~1} — the
//       same guard the kernel/oracle apply (never a strict ~1 that a zeroed column trips).
//   (B) REFERENCE from the FROZEN kalis golden gamma — ref_post[l*P+g] = sum_{k:group==g}
//       golden_gamma[k*M+l] is a PURE function of the frozen kalis posterior + the same
//       synthetic partition. The CpuBackend sink (which recomputes its own gamma, matching
//       kalis near-bit) must reproduce it (<=1e-8, the coancestry golden-arm family).
//   (C) IN-TEST FB reproduces the golden gamma (reused from the coancestry test's
//       independent transcription; <=1e-9) — anchors that the sink's own gamma is kalis's.
//   (D) GPU-vs-CPU sink cross-check (<=1e-11) when a device is visible; SKIPPED cleanly on
//       CPU-only builds (mirrors the parity gate's cudaGetDeviceCount guard).
//   (E) ANALYTIC P-edge identities on a tiny K=2/M=2 fixture: P=1 (post_l(0) == 1 at every
//       informative column) and P=K (post_l(k) == gamma_l(k), localanc reduces to the raw
//       posterior). Both catch an off-by-one in the label indexing.
//
// REAL data only: the golden is the same real 1000G chr22 phased panel as the parity gate.
// Self-checking main(); CTest gates on the exit. argv[1] = the golden directory.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "steppe/config.hpp"

namespace {

struct Golden {
    int K = 0;
    long M = 0;
    int R = 0;
    std::vector<std::uint8_t> donors;  // K*M donor-major
    std::vector<double> rho;           // M (steppe convention)
    std::vector<double> mu;            // M
    struct Rec {
        int self = -1;
        std::vector<std::uint8_t> alleles;  // M
        std::vector<double> pi;             // K
        std::vector<double> gamma;          // K*M donor-major
    };
    std::vector<Rec> recs;
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path);
    if (!f) { std::printf("  [FAIL] cannot open golden %s\n", path.c_str()); return false; }
    std::string line;
    Golden::Rec* cur = nullptr;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "K") { ss >> g.K; }
        else if (tag == "M") { ss >> g.M; }
        else if (tag == "R") { ss >> g.R; }
        else if (tag == "mu_scalar") { /* informational */ }
        else if (tag == "D") {
            int v; while (ss >> v) g.donors.push_back(static_cast<std::uint8_t>(v));
        } else if (tag == "RHO") {
            double v; while (ss >> v) g.rho.push_back(v);
        } else if (tag == "MU") {
            double v; while (ss >> v) g.mu.push_back(v);
        } else if (tag == "REC") {
            g.recs.emplace_back(); cur = &g.recs.back(); ss >> cur->self;
        } else if (tag == "A") {
            int v; while (ss >> v) cur->alleles.push_back(static_cast<std::uint8_t>(v));
        } else if (tag == "PI") {
            double v; while (ss >> v) cur->pi.push_back(v);
        } else if (tag == "G") {
            double v; while (ss >> v) cur->gamma.push_back(v);
        }
    }
    return true;
}

// An independent transcription of the per-column-rescaled forward-backward — a SECOND
// implementation (not the CpuBackend) so arm (C) has genuine independence. Returns the
// posterior gamma[k*M+l] (alpha*beta normalized per column). Mirrors the coancestry test.
void reference_gamma(const std::uint8_t* recip, const std::uint8_t* donors, const double* pi,
                     const double* rho, const double* mu, int K, long M,
                     std::vector<double>& gamma) {
    const std::size_t Ks = static_cast<std::size_t>(K), Ms = static_cast<std::size_t>(M);
    std::vector<double> a(Ks * Ms, 0.0), b(Ks * Ms, 0.0);
    gamma.assign(Ks * Ms, 0.0);
    auto e = [&](long l, int k) -> double {
        const std::uint8_t rr = recip[l];
        const std::uint8_t d = donors[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
        if (rr > 1u || d > 1u) return 1.0;
        return (rr == d) ? (1.0 - mu[l]) : mu[l];
    };
    { double s = 0; for (int k = 0; k < K; ++k) { double v = pi[k] * e(0, k); a[k * Ms] = v; s += v; }
      double inv = s > 0 ? 1.0 / s : 0.0; for (int k = 0; k < K; ++k) a[k * Ms] *= inv; }
    for (long l = 1; l < M; ++l) {
        double ps = 0; for (int k = 0; k < K; ++k) ps += a[k * Ms + (l - 1)];
        double s = 0;
        for (int k = 0; k < K; ++k) {
            double v = e(l, k) * ((1.0 - rho[l]) * a[k * Ms + (l - 1)] + rho[l] * pi[k] * ps);
            a[k * Ms + l] = v; s += v;
        }
        double inv = s > 0 ? 1.0 / s : 0.0; for (int k = 0; k < K; ++k) a[k * Ms + l] *= inv;
    }
    for (int k = 0; k < K; ++k) b[k * Ms + (M - 1)] = 1.0;
    for (long l = M - 2; l >= 0; --l) {
        double r = rho[l + 1], T = 0;
        for (int k = 0; k < K; ++k) T += pi[k] * e(l + 1, k) * b[k * Ms + (l + 1)];
        double s = 0;
        for (int k = 0; k < K; ++k) {
            double v = (1.0 - r) * e(l + 1, k) * b[k * Ms + (l + 1)] + r * T;
            b[k * Ms + l] = v; s += v;
        }
        double inv = s > 0 ? 1.0 / s : 0.0; for (int k = 0; k < K; ++k) b[k * Ms + l] *= inv;
    }
    for (long l = 0; l < M; ++l) {
        double d = 0; for (int k = 0; k < K; ++k) d += a[k * Ms + l] * b[k * Ms + l];
        double inv = d > 0 ? 1.0 / d : 0.0;
        for (int k = 0; k < K; ++k) gamma[k * Ms + l] = a[k * Ms + l] * b[k * Ms + l] * inv;
    }
}

int g_fail = 0;
void check(bool ok, const char* what, double achieved, double tol) {
    std::printf("  [%s] %s (max diff %.3e, tol %.1e)\n", ok ? " ok " : "FAIL", what, achieved, tol);
    if (!ok) ++g_fail;
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("=== Li-Stephens localanc adapter gate (steppe paint --face localanc, Phase 3) ===\n");
    std::string dir = "tests/reference/goldens/li_stephens";
    if (argc >= 2) dir = argv[1];

    auto cpu = steppe::device::make_cpu_backend();
    const steppe::Precision prec = steppe::Precision::fp64();

    // ---- Arm (E): P-edge analytic identities on a tiny K=2, M=2 fixture ------------
    {
        const std::vector<std::uint8_t> donors = {0, 0, 1, 1};  // donor0=[0,0], donor1=[1,1]
        const std::vector<std::uint8_t> recip = {0, 0};
        const std::vector<double> pi = {0.5, 0.5};
        const std::vector<double> rho = {1.0, 0.2};
        const std::vector<double> mu = {0.1, 0.1};

        // P=1: every donor collapses to label 0 -> post_l(0) == 1 at every column.
        {
            const std::vector<int> grp = {0, 0};
            const steppe::LsLocalAncestry la = cpu->ls_localanc(
                recip.data(), donors.data(), pi.data(), rho.data(), mu.data(), grp.data(),
                2, 2, 1, 1, prec);
            double md = 0.0;
            for (long l = 0; l < 2; ++l) md = std::fmax(md, std::fabs(la.post[static_cast<std::size_t>(l)] - 1.0));
            check(md <= 1e-12, "P=1 identity: post_l(0) == 1 at every column", md, 1e-12);
        }
        // P=K: each donor its own label -> post_l(k) == gamma_l(k).
        {
            const std::vector<int> grp = {0, 1};
            const steppe::LsLocalAncestry la = cpu->ls_localanc(
                recip.data(), donors.data(), pi.data(), rho.data(), mu.data(), grp.data(),
                2, 2, 1, 2, prec);
            std::vector<double> gamma;
            reference_gamma(recip.data(), donors.data(), pi.data(), rho.data(), mu.data(), 2, 2, gamma);
            double md = 0.0;
            for (long l = 0; l < 2; ++l)
                for (int k = 0; k < 2; ++k)
                    md = std::fmax(md, std::fabs(la.post[static_cast<std::size_t>(l) * 2 +
                                                         static_cast<std::size_t>(k)] -
                                                 gamma[static_cast<std::size_t>(k) * 2 +
                                                       static_cast<std::size_t>(l)]));
            check(md <= 1e-12, "P=K identity: post_l(k) == gamma_l(k)", md, 1e-12);
        }
    }

    // ---- Golden arms (A), (B), (C), (D) -------------------------------------------
    Golden g;
    if (!load_golden(dir + "/golden.txt", g) || g.K <= 0 || g.M <= 0 || g.recs.empty()) {
        std::printf("  [FAIL] golden unreadable/empty\n\nRESULT: FAIL\n");
        return 1;
    }
    const std::size_t Ks = static_cast<std::size_t>(g.K), Ms = static_cast<std::size_t>(g.M);
    const std::size_t KM = Ks * Ms;

    // A synthetic non-trivial ancestry partition (physical meaning irrelevant — it must
    // only be consumed identically by the reference and the sink to validate the adapter).
    const int P = 3;
    const std::size_t Ps = static_cast<std::size_t>(P);
    std::vector<int> group(Ks);
    for (int k = 0; k < g.K; ++k) group[static_cast<std::size_t>(k)] = k % P;

    int nd = 0; (void)cudaGetDeviceCount(&nd);
    std::unique_ptr<steppe::ComputeBackend> gpu;
    if (nd > 0) { gpu = steppe::device::make_cuda_backend(0);
        std::printf("GPU present (%d device(s)): running the GPU localanc sink arm\n", nd); }
    else std::printf("no CUDA device visible: SKIPPING the GPU sink arm (CPU adapter only)\n");

    double max_sum1 = 0.0, max_ref_golden = 0.0, max_fb_gamma = 0.0, max_gpu_cpu = 0.0;
    for (int r = 0; r < g.R; ++r) {
        const Golden::Rec& rec = g.recs[static_cast<std::size_t>(r)];
        if (rec.alleles.size() != Ms || rec.pi.size() != Ks || rec.gamma.size() != KM) {
            std::printf("  [FAIL] recipient %d malformed golden row\n", r); ++g_fail; continue;
        }
        // The CpuBackend sink (recomputes its own gamma near-bit against kalis).
        const steppe::LsLocalAncestry sink = cpu->ls_localanc(
            rec.alleles.data(), g.donors.data(), rec.pi.data(), g.rho.data(), g.mu.data(),
            group.data(), g.K, g.M, 1, P, prec);

        // (A) Structural: each informative column sums to ~1; a zeroed (degenerate) column
        //     sums to ~0 — assert sum in {~0, ~1}.
        for (long l = 0; l < g.M; ++l) {
            double s = 0.0;
            for (int gc = 0; gc < P; ++gc)
                s += sink.post[(static_cast<std::size_t>(l)) * Ps + static_cast<std::size_t>(gc)];
            const double dev = std::fmin(std::fabs(s - 1.0), std::fabs(s));  // to nearest of {0,1}
            max_sum1 = std::fmax(max_sum1, dev);
        }

        // (B) Reference from the FROZEN golden gamma summed by the same partition.
        for (long l = 0; l < g.M; ++l) {
            double ref[16] = {0.0};  // P=3 < 16
            for (int k = 0; k < g.K; ++k) {
                const std::size_t o = static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                ref[group[static_cast<std::size_t>(k)]] += rec.gamma[o];  // frozen kalis posterior
            }
            for (int gc = 0; gc < P; ++gc)
                max_ref_golden = std::fmax(
                    max_ref_golden,
                    std::fabs(sink.post[static_cast<std::size_t>(l) * Ps +
                                        static_cast<std::size_t>(gc)] -
                              ref[gc]));
        }

        // (C) In-test independent FB -> gamma reproduces the golden gamma.
        std::vector<double> gamma;
        reference_gamma(rec.alleles.data(), g.donors.data(), rec.pi.data(), g.rho.data(),
                        g.mu.data(), g.K, g.M, gamma);
        for (std::size_t i = 0; i < KM; ++i)
            max_fb_gamma = std::fmax(max_fb_gamma, std::fabs(gamma[i] - rec.gamma[i]));

        // (D) GPU sink vs CPU sink.
        if (gpu) {
            const steppe::LsLocalAncestry gs = gpu->ls_localanc(
                rec.alleles.data(), g.donors.data(), rec.pi.data(), g.rho.data(), g.mu.data(),
                group.data(), g.K, g.M, 1, P, prec);
            for (std::size_t i = 0; i < Ms * Ps; ++i)
                max_gpu_cpu = std::fmax(max_gpu_cpu, std::fabs(gs.post[i] - sink.post[i]));
        }
    }

    check(max_sum1 <= 1e-12, "every column sums to 1 (or 0 for a degenerate column)", max_sum1, 1e-12);
    check(max_fb_gamma <= 1e-9, "in-test FB reproduces the kalis golden gamma", max_fb_gamma, 1e-9);
    check(max_ref_golden <= 1e-8, "localanc sink matches the kalis-golden-gamma reference",
          max_ref_golden, 1e-8);
    if (gpu)
        check(max_gpu_cpu <= 1e-11, "GPU localanc sink matches the CpuBackend reference",
              max_gpu_cpu, 1e-11);

    if (g_fail == 0) { std::printf("\nRESULT: PASS\n"); return 0; }
    std::printf("\nRESULT: FAIL (%d check(s))\n", g_fail);
    return 1;
}
