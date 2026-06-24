// tests/reference/test_qpfstats_smooth_parity.cu
//
// CpuBackend == CudaBackend parity for the qpfstats SMOOTHING SOLVE (backend.hpp
// qpfstats_smooth; the shared-factor batched least-squares the genotype-path joint f2
// smoother drives). This is a NUMERICAL-EQUIVALENCE unit test of the solve PRIMITIVE (the
// linear algebra: A_shared = x'x + ridge·I, the Cholesky factor, the multi-column Dtrsm
// solve, the NaN-row downdate, the all-NaN→b=0 policy), NOT a reported statistic — so a
// deterministic synthetic design/RHS is used to exercise EVERY branch (a no-NaN block, a
// PARTIAL-NaN block → the host downdate fallback, an ALL-NaN block → b=0). The genotype
// end-to-end statistic is gated separately by the real-AADR cli_qpfstats golden test.
//
// The CpuBackend (native small_linalg LU oracle) and the CudaBackend (cublasDsyrk +
// cusolverDnDpotrf + cublasDgemm + the cublasDtrsm pair + the downdate fallback) must agree
// to ~1e-9 (the EmulatedFp64 syrk/gemm + native potrf/trsm carve-out tier). The CudaBackend
// block SKIPs cleanly when no device is visible; the CpuBackend self-check ALWAYS runs.

#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

#include "device/backend.hpp"          // steppe::ComputeBackend, QpfstatsSmooth, Precision
#include "device/backend_factory.hpp"  // make_cpu_backend / make_cuda_backend / visible_device_count
#include "steppe/error.hpp"

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    if (diff > tol) {
        if (g_failures < 30)
            std::printf("  [FAIL] %-22s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                        what, got, want, diff, tol);
        ++g_failures;
    }
}

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

// A small but representative problem: npopcomb=20, npairs=6 (== C(4,2), a 4-pop basis),
// n_block=8. The design rows are the +/-0.5 f4-identity coefficients (a few f2/f3/f4 shapes),
// the RHS is a deterministic smooth field with one PARTIAL-NaN block and one ALL-NaN block.
struct Problem {
    int npopcomb = 20, npairs = 6, n_block = 8;
    double ridge = 1e-5;
    std::vector<double> x;     // [npopcomb × npairs] col-major
    std::vector<double> ymat;  // [npopcomb × n_block] col-major
    std::vector<double> y;     // [npopcomb]
};

Problem make_problem() {
    Problem p;
    const int nc = p.npopcomb, np = p.npairs, nb = p.n_block;
    p.x.assign(static_cast<std::size_t>(nc) * np, 0.0);
    auto xat = [&](int c, int pp) -> double& {
        return p.x[static_cast<std::size_t>(c) + static_cast<std::size_t>(nc) * pp];
    };
    // Deterministic design: each comb writes 4 coefficients into 4 distinct pairs (mod np),
    // /2 — the construct_fstat_matrix shape (well-conditioned with the ridge).
    for (int c = 0; c < nc; ++c) {
        const int a = c % np, b = (c + 1) % np, d = (c + 2) % np, e = (c + 3) % np;
        xat(c, a) += 0.5; xat(c, b) += 0.5; xat(c, d) -= 0.5; xat(c, e) -= 0.5;
    }
    // RHS: a smooth deterministic field; column 5 is PARTIAL-NaN (combs 0..3 NaN),
    // column 6 is ALL-NaN (every comb NaN → b must be 0).
    const double nan = std::numeric_limits<double>::quiet_NaN();
    p.ymat.assign(static_cast<std::size_t>(nc) * nb, 0.0);
    for (int blk = 0; blk < nb; ++blk)
        for (int c = 0; c < nc; ++c) {
            double v = 0.01 * (c + 1) + 0.001 * (blk + 1) - 0.0002 * c * blk;
            if (blk == 5 && c < 4) v = nan;        // PARTIAL-NaN block
            if (blk == 6) v = nan;                 // ALL-NaN block
            p.ymat[static_cast<std::size_t>(c) + static_cast<std::size_t>(nc) * blk] = v;
        }
    // Global y: a smooth field, with a couple of NaN combs (the bglob downdate path).
    p.y.assign(static_cast<std::size_t>(nc), 0.0);
    for (int c = 0; c < nc; ++c)
        p.y[static_cast<std::size_t>(c)] = (c == 2 || c == 7) ? nan : (0.02 * (c + 1) - 0.0003 * c * c);
    return p;
}

void run_backend(steppe::ComputeBackend& be, const Problem& p, steppe::QpfstatsSmooth& out) {
    steppe::Precision prec;  // default EmulatedFp64 for the matmul sub-steps
    out = be.qpfstats_smooth(std::span<const double>(p.x), std::span<const double>(p.ymat),
                             std::span<const double>(p.y), p.npopcomb, p.npairs, p.n_block,
                             p.ridge, prec);
}

}  // namespace

int main() {
    std::printf("=== qpfstats_smooth CpuBackend==CudaBackend parity ===\n");
    const Problem p = make_problem();

    // ---- CpuBackend oracle (ALWAYS runs) ----
    auto cpu = steppe::device::make_cpu_backend();
    steppe::QpfstatsSmooth bc;
    run_backend(*cpu, p, bc);
    check_true("cpu status Ok", bc.status == steppe::Status::Ok);
    check_true("cpu npairs", bc.npairs == p.npairs);
    check_true("cpu n_block", bc.n_block == p.n_block);
    check_true("cpu b size", static_cast<int>(bc.b.size()) == p.npairs * p.n_block);
    check_true("cpu bglob size", static_cast<int>(bc.bglob.size()) == p.npairs);
    // The ALL-NaN block (6) must be exactly 0.
    for (int pp = 0; pp < p.npairs; ++pp)
        check_close("cpu all-nan block==0",
                    bc.b[static_cast<std::size_t>(pp) + static_cast<std::size_t>(p.npairs) * 6],
                    0.0, 0.0, 0.0);

    // ---- CudaBackend (SKIP cleanly when no device) ----
    int gpu = 0;
    try { gpu = steppe::device::visible_device_count(); }
    catch (const std::exception& e) { std::printf("  [INFO] device probe threw: %s\n", e.what()); }
    if (gpu < 1) {
        std::printf("  [SKIP] no CUDA device — CpuBackend self-check only\n");
        if (g_failures == 0) { std::printf("\nRESULT: PASS (CpuBackend; GPU skipped)\n"); return 0; }
        std::printf("\nRESULT: FAIL (%d)\n", g_failures); return 1;
    }

    auto cuda = steppe::device::make_cuda_backend(0);
    steppe::QpfstatsSmooth bg;
    run_backend(*cuda, p, bg);
    check_true("cuda status Ok", bg.status == steppe::Status::Ok);

    for (int blk = 0; blk < p.n_block; ++blk)
        for (int pp = 0; pp < p.npairs; ++pp) {
            const std::size_t k = static_cast<std::size_t>(pp) +
                                  static_cast<std::size_t>(p.npairs) * blk;
            char nm[48];
            std::snprintf(nm, sizeof(nm), "b[p%d,b%d]", pp, blk);
            check_close(nm, bg.b[k], bc.b[k], 1e-9, 1e-11);
        }
    for (int pp = 0; pp < p.npairs; ++pp) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "bglob[%d]", pp);
        check_close(nm, bg.bglob[static_cast<std::size_t>(pp)],
                    bc.bglob[static_cast<std::size_t>(pp)], 1e-9, 1e-11);
    }

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (CpuBackend == CudaBackend; no-NaN + partial-NaN + all-NaN)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
