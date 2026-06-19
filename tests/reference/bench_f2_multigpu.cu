// bench_f2_multigpu.cu — single-GPU (G==1) vs multi-GPU (G==2) wall-clock SWEEP over
// population count P for the SPMG per-device fan-out (cleanup B1). Loads derived_full
// (P0=768) ONCE and repacks the first P rows into a [P×M] Q/V/N for each requested P
// (no dataset regeneration), times core::compute_f2_blocks_multigpu G==1 vs G==2
// end-to-end (the host call is synchronous), reports the speedup, and catches the
// single-GPU OOM (the point at which large-P only fits when sharded on a 32 GB card).
//
// M4.5 Item 4 (bench honesty): ITERS >= 10 with MEDIAN + p10/p90 (NOT min-of-2 — the
// G2 path has structurally higher per-iter variance and min-of-2 hid the steady
// state); plus the OUT-OF-BAND phase breakdown + measured H2D/D2H/peer byte totals
// read from res.last_multigpu_timings (a debug field on Resources, mirroring
// last_combine_path; NEVER on the numeric F2BlockTensor — timing is strictly
// out-of-band, the numeric result is untouched).
//
// Run: ./bench_f2_multigpu [data_root] [P1 P2 ...]   (defaults 100 200 300 400 500 600 700 768)
// EmulatedFp64{40} (the production f2 path).
#include <algorithm>
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

// End-to-end stats for one (P, G) cell: robust order statistics over `iters` samples
// (median + p10/p90, NOT min-of-2 — Item 4) plus a snapshot of the LAST run's
// out-of-band phase timing / byte totals (res.last_multigpu_timings, mirroring
// last_combine_path; the per-run byte totals are constant for a fixed (P,G) so the
// last-run snapshot is representative). `ok == false` ⇒ OOM/failed (the timing
// numeric path is untouched; this is purely observability).
struct RunStats {
    bool ok = false;
    double median = -1.0;
    double p10 = -1.0;
    double p90 = -1.0;
    steppe::device::MultiGpuTimings timings{};  // out-of-band snapshot of the last run
};

RunStats time_run(steppe::device::Resources& res, const MatView& Q, const MatView& V,
                  const MatView& N, const BlockPartition& part, const Precision& prec,
                  int iters, const char* label) {
    RunStats st;
    try {
        { auto warm = steppe::core::compute_f2_blocks_multigpu(res, Q, V, N, part, prec); (void)warm; }
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(iters));
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            auto r = steppe::core::compute_f2_blocks_multigpu(res, Q, V, N, part, prec);
            const auto t1 = std::chrono::steady_clock::now();
            (void)r;
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        const std::size_t n = samples.size();
        st.ok = n > 0;
        if (st.ok) {
            st.median = samples[n / 2];
            st.p10 = samples[n / 10];
            st.p90 = samples[(9 * n) / 10];
        }
        st.timings = res.last_multigpu_timings;  // out-of-band: the LAST run's phases
        return st;
    } catch (const std::exception& e) {
        std::printf("    [%s] OOM/failed: %s\n", label, e.what());
        return st;  // ok == false
    }
}

// GiB helper for the measured-byte print (out-of-band; bytes are arithmetic, not the
// numeric result).
double to_gib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// Per-(P,G) detail line: end-to-end p10/median/p90 dispersion (Item 4 — replaces the
// opaque min-of-2) + the OUT-OF-BAND phase breakdown and measured H2D/D2H/peer byte
// totals read from res.last_multigpu_timings. The finer device-internal combine
// fields (combine_peer_ms / combine_d2h_ms) print only when the orchestrator filled
// them (they stay 0 unless the combine reports them through the CUDA-free seam).
void print_run_detail(const char* label, const RunStats& s) {
    if (!s.ok) { std::printf("      %-5s  (no timing - OOM/failed)\n", label); return; }
    const steppe::device::MultiGpuTimings& t = s.timings;
    std::printf("      %-5s  median %8.1f ms  (p10 %8.1f  p90 %8.1f)\n",
                label, s.median, s.p10, s.p90);
    // Phase wall (host steady_clock brackets in the orchestrator). compute_wall is 0
    // on the G==1 fast path (no fan-out) and combine_wall is 0 when no combine ran.
    if (t.compute_wall_ms > 0.0 || t.combine_wall_ms > 0.0) {
        std::printf("             phase: compute-wall %8.1f ms  combine-wall %8.1f ms",
                    t.compute_wall_ms, t.combine_wall_ms);
        if (t.combine_peer_ms > 0.0 || t.combine_d2h_ms > 0.0)
            std::printf("  (peer %8.1f  final-D2H %8.1f)", t.combine_peer_ms, t.combine_d2h_ms);
        std::printf("\n");
    }
    // Measured bus traffic (arithmetic byte totals; self-documents the 2x bus story).
    std::printf("             bytes: H2D %6.2f GiB  D2H %6.2f GiB  peer %6.2f GiB\n",
                to_gib(t.h2d_bytes), to_gib(t.d2h_bytes), to_gib(t.peer_bytes));
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
    const int ITERS = 10;  // Item 4: >= 10 (min-of-2 hid the G2 steady state)
    std::printf("bench_f2_multigpu — %d GPUs, EmulatedFp64{%d}, M=%ld n_block=%d, "
                "median (p10/p90) of %d runs\n",
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
        const RunStats s1 = time_run(r1, Q, V, N, part, prec, ITERS, "G==1");

        RunStats s2;
        if (devcount >= 2) {
            DeviceConfig c2; c2.devices = {0, 1};
            steppe::device::Resources r2 = steppe::device::build_resources(c2);
            s2 = time_run(r2, Q, V, N, part, prec, ITERS, "G==2");
        }

        // ---- Summary row: MEDIAN (Item 4: median, not min-of-2) + speedup ----
        char b1[32], b2[32], sp[32];
        if (s1.ok) std::snprintf(b1, sizeof b1, "%10.1f ms", s1.median); else std::snprintf(b1, sizeof b1, "%13s", "OOM");
        if (s2.ok) std::snprintf(b2, sizeof b2, "%10.1f ms", s2.median); else std::snprintf(b2, sizeof b2, "%13s", devcount >= 2 ? "OOM" : "-");
        if (s1.ok && s2.ok) std::snprintf(sp, sizeof sp, "%.2fx", s1.median / s2.median); else std::snprintf(sp, sizeof sp, "%s", "-");
        std::printf("  %-4d  %15s  %15s     %s\n", Pp, b1, b2, sp);

        // ---- Per-G dispersion (p10/p90) + OUT-OF-BAND phase + measured bus bytes ----
        // (res.last_multigpu_timings, mirroring last_combine_path — never on the numeric
        // tensor). G==1 runs no combine, so its compute_wall/byte fields stay 0 there;
        // print the end-to-end p10/p90 for both, and the combine/peer breakdown for G2.
        print_run_detail("G==1", s1);
        if (devcount >= 2) print_run_detail("G==2", s2);
        std::fflush(stdout);
    }
    return 0;
}
