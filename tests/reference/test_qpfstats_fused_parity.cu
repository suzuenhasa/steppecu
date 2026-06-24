// tests/reference/test_qpfstats_fused_parity.cu
//
// CpuBackend == CudaBackend parity for the qpfstats FUSED reduce→jackknife→smooth→recenter
// (backend.hpp qpfstats_blocks_smooth; the PERF path). This DIRECTLY gates the ON-DEVICE
// block-JACKKNIFE kernels (qpfstats_jackknife_kernel.cu): the per-comb global jackknife y/ymat
// (matrix_jackknife_est_col, the bglob RHS source) AND the per-pair recenter shift
// (f2blocks_pair_est) now run on the GPU in native FP64, where the CpuBackend oracle uses long
// double. This is the cancellation-sensitive §12 carve-out — the test confirms the FP64 carry
// precision reproduces the long-double reference at the golden tier (rtol 1e-6) on a SYNTHETIC
// genotype field (a within-codebase numerical-equivalence test of the primitive; the real-AADR
// end-to-end statistic is gated by the cli_qpfstats genotype golden).
//
// The synthetic Q/V exercise the reduce mask (some invalid SNPs), the per-comb jackknife (all
// 666 f2/f3/f4 combs of a 9-pop basis over several blocks), and the recenter. The CudaBackend
// block SKIPs cleanly when no device is visible; the CpuBackend self-check ALWAYS runs.

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#include "device/backend.hpp"          // steppe::ComputeBackend, QpfstatsSmooth, Precision
#include "device/backend_factory.hpp"  // make_cpu_backend / make_cuda_backend / visible_device_count
#include "core/domain/block_partition_rule.hpp"  // core::assign_blocks (the SAME block primitive)
#include "steppe/error.hpp"

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    if (!std::isfinite(got) || diff > tol) {
        if (g_failures < 30)
            std::printf("  [FAIL] %-20s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                        what, got, want, diff, tol);
        ++g_failures;
    }
}

void check_true(const char* what, bool ok) {
    if (!ok) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

// --- the f2∪f3∪f4 popcomb + the construct_fstat_matrix design (the SAME shape run_qpfstats
// builds; reproduced locally so the test does not reach into the .cpp's anonymous namespace).
struct PopComb { int p1, p2, p3, p4; };

int pair_index(int i, int j, int n) {  // row-major upper-tri (i<j)
    return i * n - (i * (i + 1)) / 2 + (j - i - 1);
}

void build(int npop, std::vector<PopComb>& combs, std::vector<double>& x,
           int& npopcomb, int& npairs) {
    combs.clear();
    for (int i = 0; i < npop; ++i)
        for (int j = i + 1; j < npop; ++j) combs.push_back({i, j, i, j});          // f2
    for (int i = 0; i < npop; ++i)
        for (int j = i + 1; j < npop; ++j)
            for (int k = j + 1; k < npop; ++k) {                                   // f3 3-rot
                combs.push_back({i, j, i, k});
                combs.push_back({j, k, j, i});
                combs.push_back({k, i, k, j});
            }
    for (int i = 0; i < npop; ++i)
        for (int j = i + 1; j < npop; ++j)
            for (int k = j + 1; k < npop; ++k)
                for (int l = k + 1; l < npop; ++l) {                               // f4 3-rot
                    combs.push_back({i, j, k, l});
                    combs.push_back({i, k, j, l});
                    combs.push_back({i, l, j, k});
                }
    npopcomb = static_cast<int>(combs.size());
    npairs = npop * (npop - 1) / 2;
    x.assign(static_cast<std::size_t>(npopcomb) * static_cast<std::size_t>(npairs), 0.0);
    auto set = [&](int c, int a, int b, double v) {
        if (a == b) return;
        const int p = (a < b) ? pair_index(a, b, npop) : pair_index(b, a, npop);
        x[static_cast<std::size_t>(c) + static_cast<std::size_t>(npopcomb) *
                                            static_cast<std::size_t>(p)] += v;
    };
    for (int c = 0; c < npopcomb; ++c) {
        const PopComb& pc = combs[static_cast<std::size_t>(c)];
        set(c, pc.p1, pc.p3, 1.0); set(c, pc.p2, pc.p4, 1.0);
        set(c, pc.p1, pc.p4, -1.0); set(c, pc.p2, pc.p3, -1.0);
        if (pc.p1 == pc.p3 && pc.p2 == pc.p4)
            for (int p = 0; p < npairs; ++p)
                x[static_cast<std::size_t>(c) + static_cast<std::size_t>(npopcomb) *
                                                    static_cast<std::size_t>(p)] *= 2.0;
    }
    for (double& v : x) v *= 0.5;
}

}  // namespace

int main() {
    std::printf("=== qpfstats_blocks_smooth (fused) CpuBackend==CudaBackend parity ===\n");

    constexpr int npop = 9;
    constexpr long M = 4000;  // SNPs across several blocks
    const int P = npop;

    // --- synthetic Q/V [P × M] col-major: a smooth deterministic freq field + a sparse mask. ---
    std::vector<double> Q(static_cast<std::size_t>(P) * static_cast<std::size_t>(M), 0.0);
    std::vector<double> Vv(static_cast<std::size_t>(P) * static_cast<std::size_t>(M), 1.0);
    for (long s = 0; s < M; ++s)
        for (int p = 0; p < P; ++p) {
            const std::size_t idx = static_cast<std::size_t>(p) +
                                    static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
            // a deterministic freq in (0,1).
            double f = 0.5 + 0.35 * std::sin(0.013 * s + 0.7 * p) + 0.1 * std::cos(0.002 * s * (p + 1));
            if (f < 0.02) f = 0.02;
            if (f > 0.98) f = 0.98;
            Q[idx] = f;
            // sparse missingness: drop ~3% of cells (exercises the allsnps mask + NaN combs).
            if (((s * 7 + p * 13) % 31) == 0) Vv[idx] = 0.0;
        }

    // --- block layout via the SAME assign_blocks primitive (chrom 1, evenly spaced genpos). ---
    std::vector<int> chrom(static_cast<std::size_t>(M), 1);
    std::vector<double> genpos(static_cast<std::size_t>(M), 0.0);
    for (long s = 0; s < M; ++s) genpos[static_cast<std::size_t>(s)] = 1e-4 * static_cast<double>(s);
    const steppe::core::BlockPartition part = steppe::core::assign_blocks(
        std::span<const int>(chrom), std::span<const double>(genpos), 0.05);
    const int n_block = part.n_block;
    check_true("n_block > 1", n_block > 1);
    const std::vector<steppe::core::BlockRange> ranges =
        steppe::core::block_ranges(std::span<const int>(part.block_id), M, n_block);
    std::vector<int> block_sizes(static_cast<std::size_t>(n_block), 0);
    for (int b = 0; b < n_block; ++b)
        block_sizes[static_cast<std::size_t>(b)] =
            static_cast<int>(ranges[static_cast<std::size_t>(b)].size());

    // --- popcomb + design + the flat quad table. ---
    std::vector<PopComb> combs;
    std::vector<double> x;
    int npopcomb = 0, npairs = 0;
    build(npop, combs, x, npopcomb, npairs);
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(npopcomb) * 4);
    for (const PopComb& pc : combs) { flat.push_back(pc.p1); flat.push_back(pc.p2);
                                      flat.push_back(pc.p3); flat.push_back(pc.p4); }
    std::printf("  npop=%d npopcomb=%d npairs=%d n_block=%d M=%ld\n",
                npop, npopcomb, npairs, n_block, M);

    auto run = [&](steppe::ComputeBackend& be, steppe::QpfstatsSmooth& out) {
        steppe::Precision prec;  // default: matmul sub-steps emulated, jackknife/solve native
        out = be.qpfstats_blocks_smooth(
            Q.data(), Vv.data(), P, M, part.block_id.data(), n_block,
            std::span<const int>(flat), std::span<const double>(x), npopcomb, npairs,
            std::span<const int>(block_sizes), 1e-5, prec);
    };

    // ---- CpuBackend oracle (ALWAYS runs) ----
    auto cpu = steppe::device::make_cpu_backend();
    steppe::QpfstatsSmooth bc;
    run(*cpu, bc);
    check_true("cpu status Ok", bc.status == steppe::Status::Ok);
    check_true("cpu b size", static_cast<int>(bc.b.size()) ==
                                 npairs * n_block);
    check_true("cpu bglob size", static_cast<int>(bc.bglob.size()) == npairs);
    check_true("cpu recenter_shift size",
               static_cast<int>(bc.recenter_shift.size()) == npairs);

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
    run(*cuda, bg);
    check_true("cuda status Ok", bg.status == steppe::Status::Ok);

    // The on-device native-FP64 jackknife vs the long-double oracle: the golden tier (rtol 1e-6).
    for (int blk = 0; blk < n_block; ++blk)
        for (int pp = 0; pp < npairs; ++pp) {
            const std::size_t k = static_cast<std::size_t>(pp) +
                                  static_cast<std::size_t>(npairs) * static_cast<std::size_t>(blk);
            char nm[48];
            std::snprintf(nm, sizeof(nm), "b[p%d,b%d]", pp, blk);
            check_close(nm, bg.b[k], bc.b[k], 1e-6, 1e-12);
        }
    for (int pp = 0; pp < npairs; ++pp) {
        char nm[40];
        std::snprintf(nm, sizeof(nm), "bglob[%d]", pp);
        check_close(nm, bg.bglob[static_cast<std::size_t>(pp)],
                    bc.bglob[static_cast<std::size_t>(pp)], 1e-6, 1e-12);
        std::snprintf(nm, sizeof(nm), "recenter_shift[%d]", pp);
        check_close(nm, bg.recenter_shift[static_cast<std::size_t>(pp)],
                    bc.recenter_shift[static_cast<std::size_t>(pp)], 1e-6, 1e-12);
    }

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (CpuBackend == CudaBackend fused reduce/jackknife/smooth/recenter)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
