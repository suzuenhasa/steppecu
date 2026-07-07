// tests/reference/test_li_stephens_parity.cu
//
// Li-Stephens forward-backward NUMERICS gate: steppe's CpuBackend REFERENCE copying
// forward-backward (ComputeBackend::ls_forward_backward — exact, per-column rescaled,
// native FP64) must reproduce kalis's exact Li & Stephens posterior NEAR-BIT on a
// tiny REAL phased panel. This pins the FB recursion before a line of the Phase-1 GPU
// kernel exists — the "ruler" the GPU forward-backward is later diffed against.
//
// The golden (tests/reference/goldens/li_stephens/golden.txt) was produced by kalis
// (Aslett & Christ 2024) on box5090 from 12 REAL 1000 Genomes phase3 chr22 samples
// (24 phased haplotypes), 256 biallelic polymorphic SNPs, with a REAL HapMap-
// interpolated cM map. kalis copies each recipient from all OTHER haplotypes (Pi
// uniform, self excluded = leave-one-out). The golden freezes kalis's OWN inputs
// (the donor allele matrix, rho in steppe's convention, mu, and each recipient's Pi)
// AND its full per-SNP posterior gamma (N donors x M SNPs); this gate feeds those
// identical inputs to steppe's FB and reports the max abs posterior difference.
//
// The panel is a dev PARITY ORACLE (a tiny real panel for numerics-pinning), like the
// existing CpuBackend parity fixtures — NOT a user-facing result. This test uses ONLY
// the CpuBackend reference; it needs NO CUDA device (built by nvcc for link parity
// with the other reference gates). Self-checking main(); CTest gates on the exit.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
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
    int donor_rows = 0;
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
            int v;
            while (ss >> v) g.donors.push_back(static_cast<std::uint8_t>(v));
            ++donor_rows;
        } else if (tag == "RHO") {
            double v; while (ss >> v) g.rho.push_back(v);
        } else if (tag == "MU") {
            double v; while (ss >> v) g.mu.push_back(v);
        } else if (tag == "REC") {
            g.recs.emplace_back();
            cur = &g.recs.back();
            ss >> cur->self;
        } else if (tag == "A") {
            int v; while (ss >> v) cur->alleles.push_back(static_cast<std::uint8_t>(v));
        } else if (tag == "PI") {
            double v; while (ss >> v) cur->pi.push_back(v);
        } else if (tag == "G") {
            double v; while (ss >> v) cur->gamma.push_back(v);
        }
    }
    (void)donor_rows;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("=== Li-Stephens FB parity gate (steppe CpuBackend reference vs kalis) ===\n");
    std::string dir = "tests/reference/goldens/li_stephens";
    if (argc >= 2) dir = argv[1];
    const std::string path = dir + "/golden.txt";

    Golden g;
    if (!load_golden(path, g)) { std::printf("\nRESULT: FAIL (golden unreadable)\n"); return 1; }

    const std::size_t KM = static_cast<std::size_t>(g.K) * static_cast<std::size_t>(g.M);
    bool shape_ok =
        g.K > 0 && g.M > 0 && g.R > 0 &&
        g.donors.size() == KM && g.rho.size() == static_cast<std::size_t>(g.M) &&
        g.mu.size() == static_cast<std::size_t>(g.M) &&
        g.recs.size() == static_cast<std::size_t>(g.R);
    if (!shape_ok) {
        std::printf("  [FAIL] golden shape: K=%d M=%ld R=%d donors=%zu rho=%zu mu=%zu recs=%zu\n",
                    g.K, g.M, g.R, g.donors.size(), g.rho.size(), g.mu.size(), g.recs.size());
        std::printf("\nRESULT: FAIL\n");
        return 1;
    }
    std::printf("golden: K=%d donors, M=%ld SNPs, R=%d recipients (real 1000G chr22 phased panel)\n",
                g.K, g.M, g.R);

    auto be = steppe::device::make_cpu_backend();
    const steppe::Precision prec = steppe::Precision::fp64();

    // The near-bit tolerance: report the achieved max abs posterior diff and gate on a
    // tight bound. Rescaling + summation-order differences vs kalis make a hard 1e-12
    // optimistic, so the gate uses a measured-and-pinned constant (§3, G5).
    const double kTol = 1e-9;

    int failures = 0;
    double global_max = 0.0;
    for (int r = 0; r < g.R; ++r) {
        const Golden::Rec& rec = g.recs[static_cast<std::size_t>(r)];
        if (rec.alleles.size() != static_cast<std::size_t>(g.M) ||
            rec.pi.size() != static_cast<std::size_t>(g.K) || rec.gamma.size() != KM) {
            std::printf("  [FAIL] recipient %d has a malformed golden row\n", r);
            ++failures;
            continue;
        }
        const steppe::LsPosterior post = be->ls_forward_backward(
            rec.alleles.data(), g.donors.data(), rec.pi.data(), g.rho.data(), g.mu.data(),
            g.K, g.M, prec);
        if (post.status != steppe::Status::Ok || post.gamma.size() != KM) {
            std::printf("  [FAIL] recipient %d: FB returned status/size error\n", r);
            ++failures;
            continue;
        }
        double max_abs = 0.0;
        for (std::size_t i = 0; i < KM; ++i) {
            const double d = std::fabs(post.gamma[i] - rec.gamma[i]);
            if (d > max_abs) max_abs = d;
        }
        if (max_abs > global_max) global_max = max_abs;
        const bool ok = max_abs <= kTol;
        std::printf("  [%s] recipient %d (self donor %d): max|gamma_steppe - gamma_kalis| = %.3e\n",
                    ok ? " ok " : "FAIL", r, rec.self, max_abs);
        if (!ok) ++failures;
    }

    std::printf("global max abs posterior diff: %.3e (tol %.1e)\n", global_max, kTol);
    if (failures == 0) {
        std::printf("\nRESULT: PASS (steppe CpuBackend FB reproduces the kalis posterior near-bit)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d recipient(s) exceeded tolerance)\n", failures);
    return 1;
}
