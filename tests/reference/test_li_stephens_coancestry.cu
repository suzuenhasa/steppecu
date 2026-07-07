// tests/reference/test_li_stephens_coancestry.cu
//
// Li-Stephens ChromoPainter COANCESTRY adapter gate (the `steppe paint` FACE, Phase 2).
// Validates that the fused coancestry sink (chunkcounts + chunklengths) is the correct
// deterministic function of a correct posterior. Four arms:
//
//   (A) ANALYTIC fixture — a tiny K=2, M=2 case whose switch_l(k), chunkcount_k and
//       chunklength_k are computed BY HAND and hard-coded, so a mis-derivation of the
//       switch/chunk formula (a factor error, dropping gamma_0, rho_{l+1} vs rho_l) is
//       caught independently of any golden or any other implementation.
//   (B) CHUNKLENGTHS from the frozen kalis golden gamma — chunklength_k = sum_l
//       golden_gamma[k*M+l]*w_l is a PURE function of the frozen kalis posterior + a
//       synthetic weight w; the CpuBackend sink (which recomputes its own gamma, matching
//       kalis near-bit) must reproduce it.
//   (C) CHUNKCOUNTS from the frozen golden gamma + an in-test normalized forward — a
//       SECOND, independent transcription of the FB forward recursion computes the
//       normalized forward a_l; its gamma is first checked near-bit against the golden
//       (proving the in-test FB reproduces kalis), then the reference chunkcount (golden
//       gamma + in-test a_l, §1b formula) is diffed against the CpuBackend sink.
//   (D) GPU-vs-CPU sink cross-check (~1e-11) when a device is visible; SKIPPED cleanly
//       on CPU-only builds (mirrors the parity gate's cudaGetDeviceCount guard).
//
// This deliberately does NOT freeze kalis's normalized forward into golden.txt (which
// would require regenerating the frozen kalis golden and risk its bit-identical gamma
// values): the in-test independent FB + the hand-computed analytic fixture together give
// the same coverage the frozen AHAT record was meant to provide. The 1e-11 GPU-vs-CPU
// tolerance is calibrated for the golden's M=256 native-FP64 accumulation.
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

#include "core/stats/li_stephens.hpp"
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

// An independent transcription of the per-column-rescaled forward recursion — a SECOND
// implementation (not the CpuBackend) so the chunkcount reference has genuine independence.
// Returns the normalized forward a[k*M+l] and gamma[k*M+l] (alpha*beta normalized).
void reference_fb(const std::uint8_t* recip, const std::uint8_t* donors, const double* pi,
                  const double* rho, const double* mu, int K, long M,
                  std::vector<double>& a, std::vector<double>& gamma) {
    const std::size_t Ks = static_cast<std::size_t>(K), Ms = static_cast<std::size_t>(M);
    a.assign(Ks * Ms, 0.0);
    std::vector<double> b(Ks * Ms, 0.0);
    gamma.assign(Ks * Ms, 0.0);
    auto e = [&](long l, int k) -> double {
        const std::uint8_t rr = recip[l];
        const std::uint8_t d = donors[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
        if (rr > 1u || d > 1u) return 1.0;
        return (rr == d) ? (1.0 - mu[l]) : mu[l];
    };
    // forward
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
    // backward
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
    std::printf("=== Li-Stephens coancestry adapter gate (steppe paint face, Phase 2) ===\n");
    std::string dir = "tests/reference/goldens/li_stephens";
    if (argc >= 2) dir = argv[1];

    auto cpu = steppe::device::make_cpu_backend();
    const steppe::Precision prec = steppe::Precision::fp64();

    // ---- Arm (A): the hand-computed analytic fixture (K=2, M=2) --------------------
    {
        const std::vector<std::uint8_t> donors = {0, 0, 1, 1};  // donor0=[0,0], donor1=[1,1]
        const std::vector<std::uint8_t> recip = {0, 0};
        const std::vector<double> pi = {0.5, 0.5};
        const std::vector<double> rho = {1.0, 0.2};
        const std::vector<double> mu = {0.1, 0.1};
        const std::vector<double> w = {1.0, 2.0};
        const steppe::LsCoancestry co = cpu->ls_paint_coancestry(
            recip.data(), donors.data(), pi.data(), rho.data(), mu.data(), w.data(), 2, 2, 1, prec);
        // By-hand (see header): exact fractions.
        const double exp_cnt[2] = {23.0 / 21.0, 1.0 / 27.0};
        const double exp_len[2] = {41.0 / 14.0, 1.0 / 14.0};
        double md = 0.0;
        for (int k = 0; k < 2; ++k) {
            md = std::fmax(md, std::fabs(co.chunkcounts[k] - exp_cnt[k]));
            md = std::fmax(md, std::fabs(co.chunklengths[k] - exp_len[k]));
        }
        check(md <= 1e-12, "analytic fixture: chunkcounts+chunklengths match hand values", md, 1e-12);
    }

    // ---- build_genetic_weights unit sanity -----------------------------------------
    {
        const std::vector<int> chrom = {1, 1, 1};
        const std::vector<double> gp = {0.0, 0.01, 0.03};
        const std::vector<double> w = steppe::core::build_genetic_weights(chrom, gp);
        double md = 0.0;
        md = std::fmax(md, std::fabs(w[0] - 0.005));   // 0.5*(0 + 0.01)
        md = std::fmax(md, std::fabs(w[1] - 0.015));   // 0.5*(0.01 + 0.02)
        md = std::fmax(md, std::fabs(w[2] - 0.010));   // 0.5*(0.02 + 0)
        check(md <= 1e-15, "build_genetic_weights midpoint rule", md, 1e-15);
    }

    // ---- Golden arms (B), (C), (D) -------------------------------------------------
    Golden g;
    if (!load_golden(dir + "/golden.txt", g) || g.K <= 0 || g.M <= 0 || g.recs.empty()) {
        std::printf("  [FAIL] golden unreadable/empty\n\nRESULT: FAIL\n");
        return 1;
    }
    const std::size_t Ks = static_cast<std::size_t>(g.K), Ms = static_cast<std::size_t>(g.M);
    const std::size_t KM = Ks * Ms;

    // A synthetic, non-trivial per-SNP weight (physical meaning irrelevant — it must only
    // be consumed identically by the reference and the sink to validate the adapter math).
    std::vector<double> w(Ms);
    for (long l = 0; l < g.M; ++l) w[static_cast<std::size_t>(l)] = 1.0e-3 * (1.0 + (l % 7));

    int nd = 0; (void)cudaGetDeviceCount(&nd);
    std::unique_ptr<steppe::ComputeBackend> gpu;
    if (nd > 0) { gpu = steppe::device::make_cuda_backend(0);
        std::printf("GPU present (%d device(s)): running the GPU coancestry sink arm\n", nd); }
    else std::printf("no CUDA device visible: SKIPPING the GPU sink arm (CPU adapter only)\n");

    double max_len_golden = 0.0, max_cnt_golden = 0.0, max_fb_gamma = 0.0;
    double max_gpu_cpu = 0.0;
    for (int r = 0; r < g.R; ++r) {
        const Golden::Rec& rec = g.recs[static_cast<std::size_t>(r)];
        if (rec.alleles.size() != Ms || rec.pi.size() != Ks || rec.gamma.size() != KM) {
            std::printf("  [FAIL] recipient %d malformed golden row\n", r); ++g_fail; continue;
        }
        // The CpuBackend sink (recomputes its own gamma near-bit against kalis).
        const steppe::LsCoancestry sink = cpu->ls_paint_coancestry(
            rec.alleles.data(), g.donors.data(), rec.pi.data(), g.rho.data(), g.mu.data(),
            w.data(), g.K, g.M, 1, prec);

        // In-test independent FB -> a_l + gamma; verify gamma reproduces the golden.
        std::vector<double> a, gamma;
        reference_fb(rec.alleles.data(), g.donors.data(), rec.pi.data(), g.rho.data(),
                     g.mu.data(), g.K, g.M, a, gamma);
        for (std::size_t i = 0; i < KM; ++i)
            max_fb_gamma = std::fmax(max_fb_gamma, std::fabs(gamma[i] - rec.gamma[i]));

        // (B) chunklength reference from the FROZEN golden gamma (pure gamma*w).
        // (C) chunkcount reference from golden gamma_0 + golden gamma_l * switch(in-test a).
        for (int k = 0; k < g.K; ++k) {
            double ref_len = 0.0, ref_cnt = 0.0;
            for (long l = 0; l < g.M; ++l) {
                const std::size_t o = static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                const double gk = rec.gamma[o];  // frozen kalis posterior
                ref_len += gk * w[static_cast<std::size_t>(l)];
                if (l == 0) {
                    ref_cnt += gk;
                } else {
                    const double rl = g.rho[static_cast<std::size_t>(l)];
                    const double pk = rec.pi[static_cast<std::size_t>(k)];
                    const double aprev = a[static_cast<std::size_t>(k) * Ms +
                                           static_cast<std::size_t>(l - 1)];
                    const double den = (1.0 - rl) * aprev + rl * pk;
                    ref_cnt += (den > 0.0) ? gk * rl * pk / den : 0.0;
                }
            }
            max_len_golden = std::fmax(max_len_golden,
                                       std::fabs(sink.chunklengths[static_cast<std::size_t>(k)] - ref_len));
            max_cnt_golden = std::fmax(max_cnt_golden,
                                       std::fabs(sink.chunkcounts[static_cast<std::size_t>(k)] - ref_cnt));
        }

        // (D) GPU sink vs CPU sink.
        if (gpu) {
            const steppe::LsCoancestry gs = gpu->ls_paint_coancestry(
                rec.alleles.data(), g.donors.data(), rec.pi.data(), g.rho.data(), g.mu.data(),
                w.data(), g.K, g.M, 1, prec);
            for (int k = 0; k < g.K; ++k) {
                max_gpu_cpu = std::fmax(max_gpu_cpu,
                    std::fabs(gs.chunkcounts[static_cast<std::size_t>(k)] -
                              sink.chunkcounts[static_cast<std::size_t>(k)]));
                max_gpu_cpu = std::fmax(max_gpu_cpu,
                    std::fabs(gs.chunklengths[static_cast<std::size_t>(k)] -
                              sink.chunklengths[static_cast<std::size_t>(k)]));
            }
        }
    }

    // The golden arms compare the sink (steppe's own near-bit gamma) against a reference
    // built from the kalis gamma, so the residual is the steppe-vs-kalis posterior gap
    // (~1e-13 achieved) folded through w/switch — pinned under a loose 1e-8.
    check(max_fb_gamma <= 1e-9, "in-test FB reproduces the kalis golden gamma", max_fb_gamma, 1e-9);
    check(max_len_golden <= 1e-8, "chunklengths match the kalis-golden-gamma reference",
          max_len_golden, 1e-8);
    check(max_cnt_golden <= 1e-8, "chunkcounts match the kalis-golden-gamma reference",
          max_cnt_golden, 1e-8);
    if (gpu)
        check(max_gpu_cpu <= 1e-11, "GPU coancestry sink matches the CpuBackend reference",
              max_gpu_cpu, 1e-11);

    if (g_fail == 0) { std::printf("\nRESULT: PASS\n"); return 0; }
    std::printf("\nRESULT: FAIL (%d check(s))\n", g_fail);
    return 1;
}
