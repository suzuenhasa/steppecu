// bench_f2_multigpu.cu — single-GPU (G==1) vs multi-GPU (G==2) wall-clock SWEEP over
// population count P for the SPMG per-device fan-out (cleanup B1). Loads derived_full
// (P0=768) ONCE and repacks the first P rows into a [P×M] Q/V/N for each requested P
// (no dataset regeneration), times core::compute_f2_blocks_multigpu G==1 vs G==2
// end-to-end (the host call is synchronous), reports the speedup, and catches the
// single-GPU OOM (the point at which large-P only fits when sharded on a 32 GB card).
//
// Run: ./bench_f2_multigpu [data_root] [P1 P2 ...]   (defaults 100 200 300 400 500 600 700 768)
// EmulatedFp64{40} (the production f2 path).
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "steppe/config.hpp"
#include "steppe/fstats.hpp"
#include "core/internal/views.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "core/fstats/f2_blocks_multigpu.hpp"
#include "device/resources.hpp"
#include "io/snp_reader.hpp"

using steppe::Precision;
using steppe::DeviceConfig;
using steppe::core::MatView;
using steppe::core::BlockPartition;

namespace {

bool read_f64(const std::string& path, std::vector<double>& out, std::size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.resize(count);
    const std::size_t got = std::fread(out.data(), sizeof(double), count, f);
    std::fclose(f);
    return got == count;
}

// First Pp rows of a column-major [P0 x M] matrix -> a fresh column-major [Pp x M]
// (element i + Pp*s = full[i + P0*s]). A valid Q/V/N for Pp populations.
std::vector<double> repack(const std::vector<double>& full, int P0, int Pp, long M) {
    std::vector<double> out(static_cast<std::size_t>(Pp) * static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s)
        for (int i = 0; i < Pp; ++i)
            out[static_cast<std::size_t>(i) + static_cast<std::size_t>(Pp) * s] =
                full[static_cast<std::size_t>(i) + static_cast<std::size_t>(P0) * s];
    return out;
}

double time_run(steppe::device::Resources& res, const MatView& Q, const MatView& V,
                const MatView& N, const BlockPartition& part, const Precision& prec,
                int iters, const char* label) {
    try {
        { auto warm = steppe::core::compute_f2_blocks_multigpu(res, Q, V, N, part, prec); (void)warm; }
        double best = 1e300;
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            auto r = steppe::core::compute_f2_blocks_multigpu(res, Q, V, N, part, prec);
            const auto t1 = std::chrono::steady_clock::now();
            (void)r;
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    } catch (const std::exception& e) {
        std::printf("    [%s] OOM/failed: %s\n", label, e.what());
        return -1.0;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string root = (argc > 1) ? argv[1] : "/workspace/data/aadr";
    const std::string dir  = root + "/derived_full";
    const std::string snp  = root + "/raw/v66.p1_HO.aadr.patch.PUB.snp";

    std::vector<int> Ps;
    for (int i = 2; i < argc; ++i) Ps.push_back(std::atoi(argv[i]));
    if (Ps.empty()) Ps = {100, 200, 300, 400, 500, 600, 700, 768};

    int devcount = 0;
    cudaGetDeviceCount(&devcount);

    int P0 = 0;
    long M = 0;
    FILE* sf = std::fopen((dir + "/shape.txt").c_str(), "r");
    if (!sf || std::fscanf(sf, "%d %ld", &P0, &M) != 2) {
        std::printf("ERROR: cannot read %s/shape.txt\n", dir.c_str());
        if (sf) std::fclose(sf);
        return 1;
    }
    std::fclose(sf);
    const std::size_t cnt = static_cast<std::size_t>(P0) * static_cast<std::size_t>(M);
    std::vector<double> Qf, Vf, Nf;
    if (!read_f64(dir + "/Q.f64", Qf, cnt) || !read_f64(dir + "/V.f64", Vf, cnt) ||
        !read_f64(dir + "/N.f64", Nf, cnt)) {
        std::printf("ERROR: failed to read derived_full Q/V/N (P0=%d M=%ld)\n", P0, M);
        return 1;
    }

    steppe::io::SnpTable snptab = steppe::io::read_snp(snp, static_cast<std::size_t>(M));
    const double bs = steppe::core::block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);
    const BlockPartition part = steppe::core::assign_blocks(snptab.chrom, snptab.genpos_morgans, bs);

    const Precision prec{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    const int ITERS = 2;
    std::printf("bench_f2_multigpu — %d GPUs, EmulatedFp64{%d}, M=%ld n_block=%d, min of %d runs\n",
                devcount, steppe::kDefaultMantissaBits, M, part.n_block, ITERS);
    std::printf("  P        G==1 (1 GPU)     G==2 (2 GPU)     speedup\n");

    for (int Pp : Ps) {
        if (Pp <= 0 || Pp > P0) { std::printf("  %-4d  (skipped: out of range 1..%d)\n", Pp, P0); continue; }
        std::vector<double> Qd = repack(Qf, P0, Pp, M);
        std::vector<double> Vd = repack(Vf, P0, Pp, M);
        std::vector<double> Nd = repack(Nf, P0, Pp, M);
        const MatView Q{Qd.data(), Pp, M};
        const MatView V{Vd.data(), Pp, M};
        const MatView N{Nd.data(), Pp, M};

        DeviceConfig c1; c1.devices = {0};
        steppe::device::Resources r1 = steppe::device::build_resources(c1);
        const double t1 = time_run(r1, Q, V, N, part, prec, ITERS, "G==1");

        double t2 = -1.0;
        if (devcount >= 2) {
            DeviceConfig c2; c2.devices = {0, 1};
            steppe::device::Resources r2 = steppe::device::build_resources(c2);
            t2 = time_run(r2, Q, V, N, part, prec, ITERS, "G==2");
        }

        char b1[32], b2[32], sp[32];
        if (t1 > 0) std::snprintf(b1, sizeof b1, "%10.1f ms", t1); else std::snprintf(b1, sizeof b1, "%13s", "OOM");
        if (t2 > 0) std::snprintf(b2, sizeof b2, "%10.1f ms", t2); else std::snprintf(b2, sizeof b2, "%13s", devcount >= 2 ? "OOM" : "-");
        if (t1 > 0 && t2 > 0) std::snprintf(sp, sizeof sp, "%.2fx", t1 / t2); else std::snprintf(sp, sizeof sp, "%s", "-");
        std::printf("  %-4d  %15s  %15s     %s\n", Pp, b1, b2, sp);
        std::fflush(stdout);
    }
    return 0;
}
