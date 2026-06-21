// bench_f2_multigpu.cu — OOM-TOLERANT ASCENDING SCALING SWEEP over population
// count P for the SPMG f2 precompute, with THREE cells per P:
//   (A) SINGLE-GPU            — Resources with device_count == 1 (the exact 1-GPU path)
//   (B) MULTI-GPU DEVICE-RESIDENT — G==2, prefer_p2p_combine=true (the fast P2P combine,
//       p2p_combine.cu; ROOT holds the full result + its own resident partial on-device)
//   (C) MULTI-GPU HOST-STAGED — G==2, prefer_p2p_combine=false (forces
//       combine_f2_partials_host; only per-device shards stay on-device, the full result
//       lives in HOST RAM — the path that scales furthest)
//
// WHY A SWEEP, NOT A BENCH (the memory reality): the f2/Vpair result is
// [P^2 * n_block] FP64 EACH; n_block ~ 748 (full v66, AT2 walk). result_GB grows as P^2, so:
//   * single-GPU needs the full result (~76 GB @ P=2500) + inputs on ONE device ⇒ OOM > 96 GB;
//   * the device-resident combine has the ROOT hold the full result + its own resident
//     partial ⇒ OOM at the root (tops out ~P2000);
//   * the host-staged combine keeps only per-device shards (~38 GB ea) on-device and the
//     ~76 GB result in HOST RAM — the path that scales, but at P=2500 host peak may brush
//     the 169 GB host ceiling.
// The sweep EMPIRICALLY finds each ceiling by CATCHING the OOM and CONTINUING.
//
// OOM TOLERANCE (the structural contract): each of the three cells is wrapped in its OWN
// try/catch around a FRESHLY-CONSTRUCTED Resources + the compute call. On a CUDA
// out-of-memory (steppe::device::CudaError carrying cudaErrorMemoryAllocation — it derives
// from std::exception) OR a host std::bad_alloc (the ~76 GB host result), the cell prints
// "OOM" and the sweep CONTINUES (it does not abort). Resources is built fresh PER CELL so
// a failed alloc does not poison the next cell, and after a caught failure we drain
// cudaGetLastError() to clear sticky device state and let RAII free the partial allocation.
//
// Data: loads a derived dir whose NATIVE P (shape.txt) may be up to 2500, then repacks the
// first Pp rows into a fresh [Pp x M] Q/V/N for each requested P (no dataset regeneration;
// the repack reads P0 from shape.txt and is correct for any P0 >= Pp). n_block + result_GB
// are computed from the loaded partition.
//
// Run: ./bench_f2_multigpu [data_root] [P1 P2 ...]
//      data_root defaults to /workspace/data/aadr; the derived subdir is the first of
//      {derived_2500, derived_full} that exists under it; the .snp is data_root/raw/*.snp.
//      Default ascending sweep: 256 512 768 1024 1536 2000 2500.
// ITERS = 3 (MEDIAN) to keep wall-clock sane at large P; a warm-up iter per cell.
// EmulatedFp64{40} (the production f2 path).
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <cuda_runtime.h>

#include "steppe/config.hpp"
#include "steppe/fstats.hpp"
#include "core/internal/views.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "core/fstats/f2_blocks_multigpu.hpp"
#include "device/resources.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/f2_blocks_out.hpp"        // M5: F2BlocksOut, OutputTier (the adaptive tiered result)
#include "device/tier_select.hpp"          // M5: select_output_tier, free_host_ram_bytes (auto-tier report)
#include "io/snp_reader.hpp"

using steppe::Precision;
using steppe::DeviceConfig;
using steppe::core::MatView;
using steppe::core::BlockPartition;

namespace {

// DEVICE-RESIDENT timing mode (the --resident flag, M4.5). When true, run_cell times
// compute_f2_blocks_multigpu_device and does NOT call .to_host(), so the ~1840ms host
// alloc/zero/D2H tail is EXCLUDED from the wall — directly demonstrating the CPU
// round-trip is gone. When false (default), the bench is host-returning (the baseline
// that INCLUDES the materialization tail), unchanged from before.
bool g_resident_mode = false;

// M5 TIERED mode (the --tiered flag). When true, main() runs a SINGLE-GPU tier-focused
// sweep instead of the G1/G2 sweep: for each P it reports (i) the AUTO-selected tier (the
// production policy from the runtime free-VRAM/free-host-RAM probes — confirms P=512
// stays Resident), (ii) the Resident-tier wall (device-resident, no streaming — the 3.9x
// baseline), and (iii) a FORCED-Disk-tier wall (block-stream + triple-buffer spill) so
// the disk wall ≈ the resident compute wall demonstrates the spill OVERLAPS compute.
bool g_tiered_mode = false;

bool read_f64(const std::string& path, std::vector<double>& out, std::size_t count) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.resize(count);
    const std::size_t got = std::fread(out.data(), sizeof(double), count, f);
    std::fclose(f);
    return got == count;
}

bool path_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

// First Pp rows of a column-major [P0 x M] matrix -> a fresh column-major [Pp x M]
// (element i + Pp*s = full[i + P0*s]). A valid Q/V/N for Pp populations. Correct for ANY
// native P0 >= Pp (the loaded shape.txt P0 may be up to 2500); the inner loop only ever
// touches the first Pp of the P0 rows per column, so a P0=2500 derived dir serves the whole
// ascending sweep down to any requested Pp.
std::vector<double> repack(const std::vector<double>& full, int P0, int Pp, long M) {
    std::vector<double> out(static_cast<std::size_t>(Pp) * static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s)
        for (int i = 0; i < Pp; ++i)
            out[static_cast<std::size_t>(i) + static_cast<std::size_t>(Pp) * s] =
                full[static_cast<std::size_t>(i) + static_cast<std::size_t>(P0) * s];
    return out;
}

// End-to-end stats for one (P, cell) measurement: median over `iters` samples plus a
// snapshot of the LAST run's out-of-band phase timing / byte totals
// (res.last_multigpu_timings, mirroring last_combine_path; NEVER on the numeric tensor).
// `ok == false` ⇒ OOM/failed.
struct RunStats {
    bool ok = false;
    double median = -1.0;
    steppe::device::MultiGpuTimings timings{};  // out-of-band snapshot of the last run
};

// Run ONE cell, OOM-tolerant. The try/catch wraps BOTH the FRESH Resources construction
// (which itself allocates the per-device cuBLAS handle / workspace and can OOM) AND the
// warm-up + timed compute calls, so a failure at ANY point in the cell is caught and
// reported as OOM without poisoning the next cell:
//   * Resources is constructed fresh INSIDE this function (per cell) and destroyed by RAII
//     on the way out (success OR throw), so a failed alloc's partial device state is freed
//     before the next cell builds its own Resources;
//   * on a caught exception we drain cudaGetLastError() to clear any sticky device error
//     state the failed alloc left, so the next cell's CUDA calls are not poisoned.
// Catches std::exception, which covers BOTH steppe::device::CudaError (the project's typed
// CUDA error — it derives from std::exception and carries cudaErrorMemoryAllocation for a
// device OOM) AND std::bad_alloc (the ~76 GB host-staged result allocation).
RunStats run_cell(const DeviceConfig& cfg, const MatView& Q, const MatView& V,
                  const MatView& N, const BlockPartition& part, const Precision& prec,
                  int iters, const char* label) {
    RunStats st;
    try {
        steppe::device::Resources res = steppe::device::build_resources(cfg);
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(iters));
        if (g_resident_mode) {
            // DEVICE-RESIDENT timing: the result STAYS in VRAM. NO .to_host(), so the
            // ~1840ms host alloc/zero/D2H tail is EXCLUDED — this wall is the
            // device-resident precompute only. The handle frees (cudaFree) at end of
            // scope; that is VRAM teardown, not a host round-trip.
            { auto warm = steppe::core::compute_f2_blocks_multigpu_device(res, Q, V, N, part, prec); (void)warm; }
            for (int i = 0; i < iters; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                auto r = steppe::core::compute_f2_blocks_multigpu_device(res, Q, V, N, part, prec);
                const auto t1 = std::chrono::steady_clock::now();
                (void)r;
                samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        } else {
            // HOST-RETURNING timing (the baseline): includes the to_host materialization.
            { auto warm = steppe::core::compute_f2_blocks_multigpu(res, Q, V, N, part, prec); (void)warm; }
            for (int i = 0; i < iters; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                auto r = steppe::core::compute_f2_blocks_multigpu(res, Q, V, N, part, prec);
                const auto t1 = std::chrono::steady_clock::now();
                (void)r;
                samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
        }
        std::sort(samples.begin(), samples.end());
        const std::size_t n = samples.size();
        st.ok = n > 0;
        if (st.ok) st.median = samples[n / 2];
        st.timings = res.last_multigpu_timings;  // out-of-band: the LAST run's phases
        return st;  // res RAII-frees here
    } catch (const std::exception& e) {
        // OOM (device CudaError or host bad_alloc) OR any other failure for this cell:
        // report and CONTINUE the sweep. Drain the sticky device error so the next cell
        // starts clean; the partial Resources (if any) already RAII-freed on the throw.
        std::printf("    [%-13s] OOM/failed: %s\n", label, e.what());
        cudaGetLastError();  // clear sticky state left by the failed alloc
        return st;           // ok == false
    }
}

// M5: run a forced-tier single-GPU cell (compute_f2_blocks_multigpu_tiered) and return
// the median wall. The result handle frees by RAII at end of each iter (no host
// materialization is timed — the wall is the tiered precompute itself, mirroring the
// resident-mode bench). OOM-tolerant like run_cell.
RunStats run_tier_cell(const MatView& Q, const MatView& V, const MatView& N,
                       const BlockPartition& part, const Precision& prec, int iters,
                       steppe::DeviceConfig::ForceTier ft, const std::string& disk_path,
                       const char* label) {
    RunStats st;
    try {
        DeviceConfig cfg;
        cfg.devices = {0};
        cfg.precision = prec;
        cfg.force_tier = ft;
        if (ft == steppe::DeviceConfig::ForceTier::Disk) cfg.disk_cache_path = disk_path;
        steppe::device::Resources res = steppe::device::build_resources(cfg);
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(iters));
        { auto warm = steppe::core::compute_f2_blocks_multigpu_tiered(res, Q, V, N, part, prec); (void)warm; }
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            auto r = steppe::core::compute_f2_blocks_multigpu_tiered(res, Q, V, N, part, prec);
            const auto t1 = std::chrono::steady_clock::now();
            (void)r;
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        const std::size_t n = samples.size();
        st.ok = n > 0;
        if (st.ok) st.median = samples[n / 2];
        return st;
    } catch (const std::exception& e) {
        std::printf("    [%-13s] OOM/failed: %s\n", label, e.what());
        cudaGetLastError();
        return st;
    }
}

// M5: time the EXISTING bulk host-returning single-GPU path (compute_f2_blocks_multigpu =
// compute_f2_blocks_device().to_host()) — the apples-to-apples baseline for the streamed
// HostRam tier (both land the full result in host RAM). OOM-tolerant like run_cell.
RunStats run_tohost_cell(const MatView& Q, const MatView& V, const MatView& N,
                         const BlockPartition& part, const Precision& prec, int iters) {
    RunStats st;
    try {
        DeviceConfig cfg;
        cfg.devices = {0};
        cfg.precision = prec;
        steppe::device::Resources res = steppe::device::build_resources(cfg);
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(iters));
        { auto warm = steppe::core::compute_f2_blocks_multigpu(res, Q, V, N, part, prec); (void)warm; }
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
        if (st.ok) st.median = samples[n / 2];
        return st;
    } catch (const std::exception& e) {
        std::printf("    [ToHost       ] OOM/failed: %s\n", e.what());
        cudaGetLastError();
        return st;
    }
}

// GiB helper for the measured-byte print (out-of-band; bytes are arithmetic, not numeric).
double to_gib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// Per-cell detail line: end-to-end median + the OUT-OF-BAND phase breakdown and measured
// H2D/D2H/peer byte totals from res.last_multigpu_timings (mirroring last_combine_path;
// never on the numeric tensor). The single-GPU cell runs no combine, so its compute/byte
// fields stay 0 there.
void print_cell_detail(const char* label, const RunStats& s) {
    if (!s.ok) { std::printf("      %-13s  (no timing - OOM/failed)\n", label); return; }
    const steppe::device::MultiGpuTimings& t = s.timings;
    std::printf("      %-13s  median %9.1f ms\n", label, s.median);
    if (t.compute_wall_ms > 0.0 || t.combine_wall_ms > 0.0) {
        std::printf("                   phase: compute-wall %9.1f ms  combine-wall %9.1f ms",
                    t.compute_wall_ms, t.combine_wall_ms);
        if (t.combine_peer_ms > 0.0 || t.combine_d2h_ms > 0.0)
            std::printf("  (peer %9.1f  final-D2H %9.1f)", t.combine_peer_ms, t.combine_d2h_ms);
        std::printf("\n");
    }
    std::printf("                   bytes: H2D %7.2f GiB  D2H %7.2f GiB  peer %7.2f GiB\n",
                to_gib(t.h2d_bytes), to_gib(t.d2h_bytes), to_gib(t.peer_bytes));
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1] is the data root unless it is a flag (--resident / --tiered; handled below).
    const std::string root =
        (argc > 1 && std::string(argv[1]) != "--resident" && std::string(argv[1]) != "--tiered")
            ? argv[1] : "/workspace/data/aadr";

    // Pick the derived subdir: prefer the at-scale derived_2500, else derived_full. The
    // repack handles any native P0 (it reads shape.txt), so a 2500-pop dir subsets DOWN to
    // every requested P in the sweep.
    std::string dir;
    for (const char* sub : {"/derived_2500", "/derived_full"}) {
        if (path_exists(root + sub + "/shape.txt")) { dir = root + sub; break; }
    }
    if (dir.empty()) {
        std::printf("ERROR: no derived_2500 or derived_full under %s\n", root.c_str());
        return 1;
    }
    const std::string snp = root + "/raw/v66.p1_HO.aadr.patch.PUB.snp";

    // --resident: the M4.5 DEVICE-RESIDENT timing mode (no host materialization). Scan
    // all argv for it (it may appear anywhere) and skip it when building the P-list.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--resident") { g_resident_mode = true; }
        if (std::string(argv[i]) == "--tiered")   { g_tiered_mode = true; }
    }

    std::vector<int> Ps;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--resident") continue;  // a flag, not a P
        if (std::string(argv[i]) == "--tiered")   continue;  // a flag, not a P
        Ps.push_back(std::atoi(argv[i]));
    }
    if (Ps.empty()) Ps = {256, 512, 768, 1024, 1536, 2000, 2500};
    std::sort(Ps.begin(), Ps.end());  // ASCENDING sweep (OOM-tolerant: a failed large P does
                                      // not block the smaller ones, and ascending shows the
                                      // ceiling crossing in order)

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
        std::printf("ERROR: failed to read %s Q/V/N (P0=%d M=%ld)\n", dir.c_str(), P0, M);
        return 1;
    }

    steppe::io::SnpTable snptab = steppe::io::read_snp(snp, static_cast<std::size_t>(M));
    const double bs = steppe::core::block_size_cm_to_morgans(steppe::kDefaultBlockSizeCm);
    const BlockPartition part = steppe::core::assign_blocks(snptab.chrom, snptab.genpos_morgans, bs);
    const int n_block = part.n_block;

    const Precision prec{Precision::Kind::EmulatedFp64, steppe::kDefaultMantissaBits};
    const int ITERS = 3;  // median of 3 (sweep: keeps wall-clock sane at large P)

    std::printf("bench_f2_multigpu — SCALING SWEEP, %d GPUs, EmulatedFp64{%d}\n",
                devcount, steppe::kDefaultMantissaBits);
    std::printf("  mode: %s\n", g_resident_mode
                ? "DEVICE-RESIDENT (no host materialization; the host round-trip is excluded)"
                : "HOST-RETURNING (includes the to_host() materialization tail)");
    std::printf("  data=%s  native P0=%d  M=%ld  n_block=%d  median of %d runs (warm-up per cell)\n",
                dir.c_str(), P0, M, n_block, ITERS);
    std::printf("  cells: G1=single-GPU | G2res=device-resident P2P combine | "
                "G2host=host-staged combine\n");
    std::printf("  result_GB = 2 * P^2 * n_block * 8 / 1e9 (f2 + Vpair, FP64)\n\n");

    // ---- M5 TIERED MODE (--tiered): single-GPU tier-focused sweep ------------------
    // For each P: the AUTO-selected tier (production policy from the runtime probes —
    // confirms P=512 stays Resident), the Resident-tier wall (no streaming, the 3.9x
    // baseline), and a FORCED-Disk-tier wall (block-stream + triple-buffer spill). The
    // disk wall ≈ the resident compute wall demonstrates the spill OVERLAPS compute.
    if (g_tiered_mode) {
        std::printf("  M5 TIERED MODE: single-GPU; auto-tier + the streamed tiers vs the right baselines.\n");
        std::printf("  ResCompute = device-resident compute (no materialization; data stays on GPU).\n");
        std::printf("  ToHost     = the EXISTING bulk compute_f2_blocks().to_host() (full result -> host RAM).\n");
        std::printf("  HostRam    = the M5 streamed host tier (block-by-block spill + triple-buffer overlap).\n");
        std::printf("  Disk       = the M5 streamed disk tier (+ the disk-write bandwidth, this fs ~0.85 GB/s).\n");
        std::printf("  OVERLAP CLAIM: HostRam ~ ToHost (streaming adds NO penalty over bulk materialization,\n");
        std::printf("  the spill hides behind compute+D2H); Disk = ToHost + the disk residual (partly hidden).\n");
        std::printf("  %-5s %7s %9s | %-9s | %11s %10s %10s %10s | %8s\n",
                    "P", "n_block", "result_GB", "auto-tier", "ResCompute", "ToHost",
                    "HostRam", "Disk", "Host/ToH");
        std::printf("  %s\n", "-------------------------------------------------------------"
                              "-----------------------------------------------------");
        std::fflush(stdout);
        const std::string disk_path = "/tmp/steppe_bench_f2.cache";
        for (int Pp : Ps) {
            if (Pp <= 0 || Pp > P0) {
                std::printf("  %-5d  (skipped: out of range 1..%d)\n", Pp, P0);
                std::fflush(stdout);
                continue;
            }
            const double result_gb =
                2.0 * static_cast<double>(Pp) * static_cast<double>(Pp) *
                static_cast<double>(n_block) * 8.0 / 1.0e9;
            std::vector<double> Qd = repack(Qf, P0, Pp, M);
            std::vector<double> Vd = repack(Vf, P0, Pp, M);
            std::vector<double> Nd = repack(Nf, P0, Pp, M);
            const MatView Q{Qd.data(), Pp, M};
            const MatView V{Vd.data(), Pp, M};
            const MatView N{Nd.data(), Pp, M};

            // Auto tier from the production policy + the device-0 runtime probes.
            std::size_t free_b = 0, total_b = 0;
            cudaSetDevice(0);
            cudaMemGetInfo(&free_b, &total_b);
            const std::size_t free_host = steppe::device::free_host_ram_bytes();
            const steppe::device::OutputTier at =
                steppe::device::select_output_tier(Pp, M, n_block, free_b, free_host);
            const char* atn = (at == steppe::device::OutputTier::Resident) ? "Resident"
                            : (at == steppe::device::OutputTier::HostRam)  ? "HostRam" : "Disk";

            const RunStats rRes = run_tier_cell(Q, V, N, part, prec, ITERS,
                                                steppe::DeviceConfig::ForceTier::Resident,
                                                disk_path, "ResCompute");
            const RunStats rToH = run_tohost_cell(Q, V, N, part, prec, ITERS);
            const RunStats rHost = run_tier_cell(Q, V, N, part, prec, ITERS,
                                                 steppe::DeviceConfig::ForceTier::HostRam,
                                                 disk_path, "HostRam");
            const RunStats rDisk = run_tier_cell(Q, V, N, part, prec, ITERS,
                                                 steppe::DeviceConfig::ForceTier::Disk,
                                                 disk_path, "Disk");
            auto ms = [](const RunStats& s, char* b, std::size_t n) {
                if (s.ok) std::snprintf(b, n, "%9.1f", s.median);
                else      std::snprintf(b, n, "%10s", "OOM");
            };
            char bRes[32], bToH[32], bHost[32], bDisk[32], rHostR[16];
            ms(rRes, bRes, sizeof bRes);
            ms(rToH, bToH, sizeof bToH);
            ms(rHost, bHost, sizeof bHost);
            ms(rDisk, bDisk, sizeof bDisk);
            if (rToH.ok && rHost.ok) std::snprintf(rHostR, sizeof rHostR, "%.2fx", rHost.median / rToH.median);
            else                     std::snprintf(rHostR, sizeof rHostR, "%s", "-");
            std::printf("  %-5d %7d %9.2f | %-9s | %11s %10s %10s %10s | %8s\n",
                        Pp, n_block, result_gb, atn, bRes, bToH, bHost, bDisk, rHostR);
            std::fflush(stdout);
        }
        return 0;
    }

    // ---- Table header --------------------------------------------------------
    std::printf("  %-5s %7s %9s | %13s %13s %13s | %10s %10s\n",
                "P", "n_block", "result_GB", "G1(ms)", "G2res(ms)", "G2host(ms)",
                "G2res/G1", "G2host/G1");
    std::printf("  %s\n", "------------------------------------------------------------"
                          "------------------------------------------------------");
    std::fflush(stdout);

    for (int Pp : Ps) {
        if (Pp <= 0 || Pp > P0) {
            std::printf("  %-5d  (skipped: out of range 1..%d)\n", Pp, P0);
            std::fflush(stdout);
            continue;
        }

        // result_GB from the loaded partition: f2 + Vpair are each [P^2 * n_block] FP64.
        const double result_gb =
            2.0 * static_cast<double>(Pp) * static_cast<double>(Pp) *
            static_cast<double>(n_block) * 8.0 / 1.0e9;

        // Repack the first Pp rows into fresh [Pp x M] Q/V/N (held for all three cells).
        std::vector<double> Qd = repack(Qf, P0, Pp, M);
        std::vector<double> Vd = repack(Vf, P0, Pp, M);
        std::vector<double> Nd = repack(Nf, P0, Pp, M);
        const MatView Q{Qd.data(), Pp, M};
        const MatView V{Vd.data(), Pp, M};
        const MatView N{Nd.data(), Pp, M};

        // ---- (A) SINGLE-GPU: Resources with device_count == 1 ----------------
        DeviceConfig c1;
        c1.devices = {0};
        c1.precision = prec;
        const RunStats g1 = run_cell(c1, Q, V, N, part, prec, ITERS, "G1");

        // ---- (B) MULTI-GPU DEVICE-RESIDENT: G==2, prefer_p2p_combine = true ---
        RunStats g2res;
        if (devcount >= 2) {
            DeviceConfig c2r;
            c2r.devices = {0, 1};
            c2r.precision = prec;
            c2r.enable_peer_access = true;
            c2r.prefer_p2p_combine = true;   // WHICH-PATH: device-resident P2P combine
            g2res = run_cell(c2r, Q, V, N, part, prec, ITERS, "G2res");
        }

        // ---- (C) MULTI-GPU HOST-STAGED: G==2, prefer_p2p_combine = false ------
        RunStats g2host;
        if (devcount >= 2) {
            DeviceConfig c2h;
            c2h.devices = {0, 1};
            c2h.precision = prec;
            c2h.prefer_p2p_combine = false;  // forces combine_f2_partials_host
            g2host = run_cell(c2h, Q, V, N, part, prec, ITERS, "G2host");
        }

        // ---- Summary row: median + speedups, OOM cells marked -----------------
        auto fmt_ms = [](const RunStats& s, bool have_dev, char* buf, std::size_t n) {
            if (s.ok) std::snprintf(buf, n, "%11.1f", s.median);
            else      std::snprintf(buf, n, "%13s", have_dev ? "OOM" : "-");
        };
        char b1[32], b2r[32], b2h[32], spr[16], sph[16];
        fmt_ms(g1, true, b1, sizeof b1);
        fmt_ms(g2res, devcount >= 2, b2r, sizeof b2r);
        fmt_ms(g2host, devcount >= 2, b2h, sizeof b2h);
        if (g1.ok && g2res.ok)  std::snprintf(spr, sizeof spr, "%.2fx", g1.median / g2res.median);
        else                    std::snprintf(spr, sizeof spr, "%s", "-");
        if (g1.ok && g2host.ok) std::snprintf(sph, sizeof sph, "%.2fx", g1.median / g2host.median);
        else                    std::snprintf(sph, sizeof sph, "%s", "-");

        std::printf("  %-5d %7d %9.2f | %13s %13s %13s | %10s %10s\n",
                    Pp, n_block, result_gb, b1, b2r, b2h, spr, sph);

        // ---- Per-cell detail: median + OUT-OF-BAND phase + measured bus bytes --
        print_cell_detail("G1", g1);
        if (devcount >= 2) {
            print_cell_detail("G2res", g2res);
            print_cell_detail("G2host", g2host);
        }
        std::fflush(stdout);
    }
    return 0;
}
