// bench_fstats_1240k.cu — SINGLE-GPU batched f-stats (f4 / f3 / f4-ratio) throughput on a
// REAL 1240K f2_blocks dir.
//
// PURPOSE: the CLI f4/f3/f4-ratio subcommands re-pay the CUDA-context init + f2.bin load
// PER PROCESS (the same ~1.0s floor the rotation bench documents) and ingest the item
// (quartet/triple/tuple) list via argv, which is ARG_MAX-capped (~33k quartets). This bench
// loads the REAL 1240K f2 ONCE, uploads it RESIDENT to device 0, and times the GENUINE
// batched engines — run_f4 / run_f3 / run_f4ratio (assemble_*_gather single launch over ALL
// items + on-device jackknife_cov for f4/f3; the host ratio_jackknife for f4-ratio) — over
// item batches up to 1,000,000. This is the floor-free device throughput (items/sec) the
// docs/perf/fstats-sweep.md table needs at PRODUCTION scale.
//
// It is a MANUAL bench, NOT a ctest gate (no golden assert; accuracy is gated by the
// reference parity tests). SINGLE-GPU --device 0 (multi-GPU parked). REAL AADR only.
//
// Run: ./bench_fstats_1240k <f2_dir> <feature> <N1> [N2 ...]
//   <f2_dir>   an f2_blocks dir (STPF2BK1 f2.bin + pops.txt), e.g. /workspace/data/1240k_sweep/f2_60
//   <feature>  f4 | f3 | f4ratio
//   <Nk>       one or more item counts to time (each: warm-up + median of 3)
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "device/backend.hpp"
#include "device/backend_factory.hpp"   // visible_device_count
#include "device/device_f2_blocks.hpp"  // DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/f2_disk_format.hpp"    // F2DiskHeader / kF2DiskMagic
#include "device/resources.hpp"         // Resources, build_resources
#include "steppe/config.hpp"            // DeviceConfig
#include "steppe/error.hpp"             // Status
#include "steppe/fstats.hpp"            // F2BlockTensor
#include "steppe/f4.hpp"                // run_f4
#include "steppe/f3.hpp"                // run_f3
#include "steppe/f4ratio.hpp"           // run_f4ratio
#include "steppe/qpadm.hpp"             // QpAdmOptions

namespace {

bool read_f2_bin(const std::string& path, steppe::F2BlockTensor& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("ERROR: cannot open %s\n", path.c_str()); return false; }
    steppe::device::F2DiskHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f) { std::printf("ERROR: short header %s\n", path.c_str()); return false; }
    if (std::memcmp(h.magic, steppe::device::kF2DiskMagic, 8) != 0) {
        std::printf("ERROR: bad magic in %s\n", path.c_str()); return false;
    }
    if (h.P <= 0 || h.n_block <= 0) {
        std::printf("ERROR: bad P/n_block (%d/%d)\n", h.P, h.n_block); return false;
    }
    out.P = h.P;
    out.n_block = h.n_block;
    const std::size_t n = static_cast<std::size_t>(h.P) * static_cast<std::size_t>(h.P) *
                          static_cast<std::size_t>(h.n_block);
    out.f2.resize(n);
    out.vpair.resize(n);
    out.block_sizes.resize(static_cast<std::size_t>(h.n_block));
    f.seekg(static_cast<std::streamoff>(h.f2_offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    f.seekg(static_cast<std::streamoff>(h.vpair_offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(out.vpair.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    f.seekg(static_cast<std::streamoff>(h.block_sizes_offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) *
                                        static_cast<std::size_t>(h.n_block)));
    if (!f) { std::printf("ERROR: truncated payload %s\n", path.c_str()); return false; }
    return true;
}

// Deterministic distinct-within-tuple index generator: enumerate G-combinations over [0,P),
// recycle with a rotation offset if N exceeds C(P,G) (still distinct within each tuple).
template <int G>
std::vector<std::array<int, G>> gen_tuples(int P, std::size_t N) {
    std::vector<std::array<int, G>> out;
    out.reserve(N);
    std::array<int, G> idx{};
    for (int i = 0; i < G; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::vector<std::array<int, G>> base;
    // enumerate combinations until we have N or exhaust C(P,G).
    auto next_comb = [&](std::array<int, G>& a) -> bool {
        int i = G - 1;
        while (i >= 0 && a[static_cast<std::size_t>(i)] == P - G + i) --i;
        if (i < 0) return false;
        ++a[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < G; ++j)
            a[static_cast<std::size_t>(j)] = a[static_cast<std::size_t>(j - 1)] + 1;
        return true;
    };
    base.push_back(idx);
    while (base.size() < N) {
        if (!next_comb(idx)) break;
        base.push_back(idx);
    }
    out = base;
    int rot = 1;
    while (out.size() < N) {
        for (const auto& c : base) {
            if (out.size() >= N) break;
            std::array<int, G> s{};
            bool ok = true;
            for (int i = 0; i < G; ++i) {
                s[static_cast<std::size_t>(i)] =
                    (c[static_cast<std::size_t>(i)] + rot) % P;
            }
            // distinct within tuple?
            for (int i = 0; i < G && ok; ++i)
                for (int j = i + 1; j < G; ++j)
                    if (s[static_cast<std::size_t>(i)] == s[static_cast<std::size_t>(j)]) ok = false;
            if (ok) out.push_back(s);
        }
        ++rot;
        if (rot > P + 2) break;  // safety
    }
    if (out.size() > N) out.resize(N);
    return out;
}

template <class RunFn>
void time_feature(const char* name, std::size_t N, RunFn&& run,
                  steppe::device::Resources& res) {
    const int ITERS = 3;
    // warm-up (primes handles + this N's kernel launch config).
    { auto w = run(); (void)w; }
    std::vector<double> ms;
    std::size_t ok = 0;
    for (int it = 0; it < ITERS; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        auto r = run();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (it == ITERS - 1) {
            ok = (r.status == steppe::Status::Ok) ? r.se.size() : 0;
        }
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    const double ips = static_cast<double>(N) / (med / 1000.0);
    const std::size_t ndispatch = res.gpus.at(0).backend->batched_dispatch_count();
    std::printf("%-8s N=%-9zu  wall(med)=%9.2f ms   %12.0f items/sec   %8.4f ms/item   "
                "dispatches=%zu   rows=%zu\n",
                name, N, med, ips, med / static_cast<double>(N), ndispatch, ok);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s <f2_dir> <f4|f3|f4ratio> <N1> [N2 ...]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const std::string feat = argv[2];

    steppe::F2BlockTensor f2;
    if (!read_f2_bin(dir + "/f2.bin", f2)) return 1;
    std::printf("=== bench_fstats_1240k (SINGLE-GPU device 0) ===\n");
    std::printf("f2_dir=%s  feature=%s  P=%d  n_block=%d\n",
                dir.c_str(), feat.c_str(), f2.P, f2.n_block);

    int gpu_count = 0;
    try { gpu_count = steppe::device::visible_device_count(); } catch (...) { gpu_count = 0; }
    if (gpu_count < 1) { std::printf("ERROR: no CUDA device visible\n"); return 1; }

    steppe::DeviceConfig cfg; cfg.devices = {0};
    steppe::device::Resources res = steppe::device::build_resources(cfg);
    const steppe::BackendCapabilities& caps = res.gpus.at(0).caps;
    std::printf("device 0: sm_%d%d\n", caps.compute_major, caps.compute_minor);

    steppe::F2BlockTensor up = f2;
    steppe::device::DeviceF2Blocks dev = steppe::device::upload_f2_blocks_to_device(up, 0);
    if (dev.f2_device() == nullptr || dev.empty()) {
        std::printf("ERROR: f2 not resident on device 0\n"); return 1;
    }
    std::printf("f2 RESIDENT on device 0\n\n");

    steppe::QpAdmOptions opts;  // struct default (fudge irrelevant to f4/f3 SE)
    const int P = f2.P;

    for (int ai = 3; ai < argc; ++ai) {
        const std::size_t N = static_cast<std::size_t>(std::atoll(argv[ai]));
        if (N == 0) continue;
        if (feat == "f4") {
            auto tup = gen_tuples<4>(P, N);
            time_feature("f4", N, [&]() {
                return steppe::run_f4(
                    dev, std::span<const std::array<int, 4>>(tup), opts, res);
            }, res);
        } else if (feat == "f3") {
            auto tup = gen_tuples<3>(P, N);
            time_feature("f3", N, [&]() {
                return steppe::run_f3(
                    dev, std::span<const std::array<int, 3>>(tup), opts, res);
            }, res);
        } else if (feat == "f4ratio") {
            auto tup = gen_tuples<5>(P, N);
            time_feature("f4ratio", N, [&]() {
                return steppe::run_f4ratio(
                    dev, std::span<const std::array<int, 5>>(tup), opts, res);
            }, res);
        } else {
            std::printf("ERROR: unknown feature '%s'\n", feat.c_str());
            return 2;
        }
    }
    return 0;
}
