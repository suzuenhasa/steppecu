// tests/reference/test_dates_parity.cu
//
// WITHIN-CODEBASE parity gate for the DATES cuFFT autocorrelation LD engine
// (ComputeBackend::dates_curve): the CudaBackend cuFFT path == the CpuBackend FFT-FREE direct
// autocorrelation oracle, bit-comparably. This pins the GPU FFT engine against the obviously-
// correct scalar reference (Σ_g W[g]·W[g+lag] == IFFT(|FFT(W)|²)/n), the same role
// test_decode_equivalence plays for decode_af. The EXTERNAL date golden (cli_dates / py_dates
// on real AADR) pins the algebra to the DATES reference; this pins the two backends to each
// other on the moment sufficient statistics that drive the corr curve.
//
// CONSTRUCTED inputs (a tiny fine grid, a few chroms, a couple of admixed samples, small
// diffmax) — permitted for a within-codebase NUMERICAL-EQUIVALENCE test (NOT a reported
// result; the real-data result is the golden gate). The CpuBackend oracle's O(diffmax·len)
// direct autocorrelation is only tractable at this small scale; the real run is the cuFFT path.
//
// CUDA TU: drives the production backends (make_cpu_backend / make_cuda_backend). SKIPs/fails
// fast with no device (a GPU gate). Self-checking main().

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "device/backend.hpp"          // ComputeBackend, DatesMoments, Precision
#include "device/backend_factory.hpp"  // make_cpu_backend / make_cuda_backend
#include "io/eigenstrat_format.hpp"    // packed_bytes / kCodesPerByte / kBitsPerCode

using steppe::ComputeBackend;
using steppe::DatesMoments;
using steppe::Precision;

namespace {

int g_failures = 0;

void check_close(const char* what, double a, double b, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(b);
    const double diff = std::fabs(a - b);
    if (diff > tol) {
        std::printf("  [FAIL] %-12s cpu=% .9e gpu=% .9e |d|=% .3e tol=% .3e\n", what, a, b, diff,
                    tol);
        ++g_failures;
    }
}

// Pack a genotype code (0/1/2/3) into a dense per-individual record (MSB-first, 4 codes/byte).
void set_code(std::vector<std::uint8_t>& rec, long s, std::uint8_t code) {
    const std::size_t b = static_cast<std::size_t>(s) / steppe::io::kCodesPerByte;
    const int p = static_cast<int>(s % steppe::io::kCodesPerByte);
    const int shift = (steppe::io::kCodesPerByte - 1 - p) * steppe::io::kBitsPerCode;
    rec[b] = static_cast<std::uint8_t>(rec[b] | (code << shift));
}

}  // namespace

int main() {
    int dev_count = 0;
    const cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count < 1) {
        std::fprintf(stderr, "RESULT: FAIL — no CUDA device available (%s); this is a GPU gate.\n",
                     cudaGetErrorString(derr));
        return EXIT_FAILURE;
    }

    std::printf("\nDATES engine parity: CpuBackend (FFT-free oracle) == CudaBackend (cuFFT)\n");

    // ---- A small constructed scenario --------------------------------------------------
    // Two chromosomes, each with a handful of SNPs mapped onto a fine grid; 3 admixed samples.
    // binsize 0.001 M, qbin 5 -> dbinsize 0.0002 M; maxdis small so diffmax is small.
    const double binsize = 0.001;
    const int qbin = 5;
    const double maxdis = 0.01;  // 1 cM -> n_bin = 10, diffmax = qbin*maxdis/binsize = 50.
    const int n_bin = static_cast<int>(std::lround(maxdis / binsize));            // 10
    const int diffmax = static_cast<int>(std::lround(qbin * maxdis / binsize));   // 50

    // SNPs: file order, two chroms. grid_cell is the DATES setqbins cumulative-cell index.
    // We place SNPs across a fine grid so several lags fall in [1, diffmax].
    const int n_chrom = 2;
    // chrom 1 cells [0..29], chrom 2 cells [60..99] (a +gap separates them, as setqbins does).
    std::vector<int> grid_cell;
    std::vector<int> chrom_first = {0, 60};
    std::vector<int> chrom_last = {29, 99};
    std::vector<double> s1, s2, valid;
    auto add_snp = [&](int cell, double f1, double f2) {
        grid_cell.push_back(cell);
        s1.push_back(f1);
        s2.push_back(f2);
        valid.push_back(1.0);
    };
    // chrom 1: cells 0,3,6,...,27 (10 SNPs)
    for (int k = 0; k < 10; ++k) add_snp(3 * k, 0.10 + 0.05 * k, 0.80 - 0.04 * k);
    // chrom 2: cells 60,64,...,96 (10 SNPs)
    for (int k = 0; k < 10; ++k) add_snp(60 + 4 * k, 0.20 + 0.03 * k, 0.70 - 0.05 * k);
    // one masked SNP (valid 0) to exercise the mask
    grid_cell.push_back(15); s1.push_back(0.5); s2.push_back(0.5); valid.push_back(0.0);

    const long M = static_cast<long>(grid_cell.size());
    int numqbins = 0;
    for (int c : grid_cell) numqbins = std::max(numqbins, c + 1);

    // 3 admixed samples: dense packed genotype records (codes 0/1/2; one missing).
    const int n_target = 3;
    const std::size_t bpr = steppe::io::packed_bytes(static_cast<std::size_t>(M));
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(n_target) * bpr, 0);
    for (int i = 0; i < n_target; ++i) {
        std::vector<std::uint8_t> rec(bpr, 0);
        for (long s = 0; s < M; ++s) {
            std::uint8_t code = static_cast<std::uint8_t>((i + s) % 3);  // 0/1/2
            if (s == 5) code = steppe::io::kMissingCode;                 // a missing genotype
            set_code(rec, s, code);
        }
        for (std::size_t b = 0; b < bpr; ++b)
            packed[static_cast<std::size_t>(i) * bpr + b] = rec[b];
    }
    std::vector<int> ploidy(static_cast<std::size_t>(n_target), 2);

    auto cpu = steppe::device::make_cpu_backend();
    auto gpu = steppe::device::make_cuda_backend(0);
    const Precision prec;

    const DatesMoments mc = cpu->dates_curve(
        s1.data(), s2.data(), valid.data(), packed.data(), bpr, n_target, ploidy.data(),
        grid_cell.data(), M, chrom_first.data(), chrom_last.data(), n_chrom, numqbins, n_bin,
        diffmax, binsize, qbin, prec);
    const DatesMoments mg = gpu->dates_curve(
        s1.data(), s2.data(), valid.data(), packed.data(), bpr, n_target, ploidy.data(),
        grid_cell.data(), M, chrom_first.data(), chrom_last.data(), n_chrom, numqbins, n_bin,
        diffmax, binsize, qbin, prec);

    if (mc.n_chrom != mg.n_chrom || mc.n_bin != mg.n_bin) {
        std::printf("  [FAIL] shape mismatch: cpu %dx%d gpu %dx%d\n", mc.n_chrom, mc.n_bin,
                    mg.n_chrom, mg.n_bin);
        ++g_failures;
    } else {
        // The corr-driving moments (s0=count autocorr, s12=signal autocorr, s11/s22 variance).
        // cuFFT is FP64 but the FFT round-trip is not bit-identical to the direct sum — gate at
        // a tight rtol (well within the date tolerance; the algebra is the same).
        const std::size_t cb = mc.s0.size();
        double max_s0 = 0.0, max_s12 = 0.0;
        for (std::size_t o = 0; o < cb; ++o) {
            max_s0 = std::max(max_s0, std::fabs(mc.s0[o]));
            max_s12 = std::max(max_s12, std::fabs(mc.s12[o]));
        }
        for (std::size_t o = 0; o < cb; ++o) {
            check_close("s0",  mg.s0[o],  mc.s0[o],  1e-9, 1e-9 + 1e-9 * max_s0);
            check_close("s12", mg.s12[o], mc.s12[o], 1e-7, 1e-9 + 1e-9 * max_s12);
            check_close("s11", mg.s11[o], mc.s11[o], 1e-7, 1e-9 + 1e-9 * max_s12);
            check_close("s22", mg.s22[o], mc.s22[o], 1e-7, 1e-9 + 1e-9 * max_s12);
        }
        // sanity: at least one bin has a nonzero count (the engine actually ran).
        if (max_s0 < 0.5) {
            std::printf("  [FAIL] no pair counts produced (engine inert)\n");
            ++g_failures;
        }
    }

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (the cuFFT LD engine matches the FFT-free oracle)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
