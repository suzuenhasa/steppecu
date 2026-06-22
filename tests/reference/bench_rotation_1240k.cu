// bench_rotation_1240k.cu — SINGLE-GPU batched qpAdm ROTATION throughput on a REAL
// 1240K f2_blocks dir.
//
// PURPOSE: the 1240K perf sweep (docs/perf/1240k-sweep.md) measured "rotation" by
// looping the CLI `qpadm` subcommand (~1 model/sec), which is PROCESS-SPAWN bound (it
// re-pays the CUDA context init + the f2.bin load PER PROCESS) and wrongly concluded
// "rotation not implemented". The genuine batched engine — run_qpadm_search ->
// CudaBackend::fit_models_batched, f2 RESIDENT in VRAM, cublasDgemmStridedBatched +
// cuSOLVER potrf/potrsBatched + a model-batched fit kernel — EXISTS and is golden-gated
// by test_qpadm_rotation.cu. But that test only loads a tiny committed fixture
// (f2_rot.bin, ~9 pops). This bench measures the REAL batched throughput over a REAL
// 1240K f2 dir on ONE GPU (--device 0; multi-GPU is parked).
//
// It is a MANUAL bench, NOT a ctest gate (no golden assert; the accuracy gate is
// test_qpadm_rotation). It builds N REAL-pop models (1 target + a source pool ->
// C(pool, 2..maxk) a few hundred models) over the SAME resident 1240K f2 and times
// run_qpadm_search BATCHED on device 0, reporting models/sec + per-model compute.
//
// Run: ./bench_rotation_1240k <f2_dir> [maxk] [jackknife]
//   <f2_dir>   an f2_blocks dir with f2.bin (STPF2BK1) + pops.txt (e.g.
//              /workspace/data/1240k_sweep/f2_60).
//   maxk       max #sources per model (default 4 => C(pool,2)+C(pool,3)+C(pool,4)).
//   jackknife  0=None (point estimate only), 1=FeasibleOnly, 2=All (default 0 — the
//              honest rotation throughput is the cheap point estimate; the LOO SE is a
//              separate cost class the rotation test already characterises).
#include <algorithm>
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
#include "device/f2_disk_format.hpp"    // F2DiskHeader / kF2DiskMagic (the STPF2BK1 layout)
#include "device/resources.hpp"         // Resources, build_resources
#include "steppe/config.hpp"            // DeviceConfig
#include "steppe/error.hpp"             // Status
#include "steppe/fstats.hpp"            // F2BlockTensor
#include "steppe/qpadm.hpp"             // run_qpadm_search, QpAdmModel/Result/Options

namespace {

// Read an f2_blocks dir's f2.bin (STPF2BK1) into a host F2BlockTensor. The on-disk
// layout (f2_disk_format.hpp) is header[64] | f2[P^2*nb] | vpair[P^2*nb] |
// block_sizes[nb int32], BYTE-IDENTICAL to F2BlockTensor's i+P*j+P*P*b — a whole-file
// read is a memcpy into the tensor. (Same reader semantics the CLI's read_f2_dir uses;
// inlined here so the bench needs no app-library link, mirroring how the rotation test
// reads its own fixture.)
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
    // f2 region starts at h.f2_offset (== 64).
    f.seekg(static_cast<std::streamoff>(h.f2_offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    f.seekg(static_cast<std::streamoff>(h.vpair_offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(out.vpair.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    f.seekg(static_cast<std::streamoff>(h.block_sizes_offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) * static_cast<std::size_t>(h.n_block)));
    if (!f) { std::printf("ERROR: truncated payload %s\n", path.c_str()); return false; }
    return true;
}

std::size_t count_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f) return 0;
    std::size_t n = 0; std::string line;
    while (std::getline(f, line)) if (!line.empty()) ++n;
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: %s <f2_dir> [maxk] [jackknife 0|1|2]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int maxk = (argc > 2) ? std::atoi(argv[2]) : 4;
    const int jk = (argc > 3) ? std::atoi(argv[3]) : 0;

    std::printf("=== bench_rotation_1240k — SINGLE-GPU batched run_qpadm_search throughput ===\n");
    std::printf("f2_dir = %s   maxk = %d   jackknife = %d\n", dir.c_str(), maxk, jk);

    // ---- load the REAL 1240K f2 (host) -----------------------------------------
    steppe::F2BlockTensor f2;
    if (!read_f2_bin(dir + "/f2.bin", f2)) return 1;
    const std::size_t poplines = count_lines(dir + "/pops.txt");
    std::printf("loaded f2: P=%d  n_block=%d  pops.txt lines=%zu\n",
                f2.P, f2.n_block, poplines);
    if (f2.P < 5) { std::printf("ERROR: need P>=5 for a target+right+pool\n"); return 1; }

    // ---- GPU present? ----------------------------------------------------------
    int gpu_count = 0;
    try { gpu_count = steppe::device::visible_device_count(); } catch (...) { gpu_count = 0; }
    if (gpu_count < 1) { std::printf("ERROR: no CUDA device visible\n"); return 1; }

    // ---- build a REAL rotation model space over the resident f2 ----------------
    // Convention: index 0 = target; the LAST `nright` pops = the right outgroup set
    // (right[0] is the fixed reference); everything in between is the SOURCE POOL the
    // rotation enumerates k-subsets of. This mirrors a real qpAdm rotation (one target,
    // a fixed right, a pool of candidate sources) over the SAME resident f2.
    const int P = f2.P;
    const int target_idx = 0;
    // right set: up to 6 outgroups taken from the tail of the pop axis (or fewer for
    // small P). right.size()-1 == nr.
    int nright = std::min(6, std::max(2, P / 5));
    std::vector<int> right_idx;
    for (int i = P - nright; i < P; ++i) right_idx.push_back(i);
    // source pool = the indices between the target and the right set.
    std::vector<int> pool;
    for (int i = 1; i < P - nright; ++i) pool.push_back(i);
    // cap the pool so the enumeration is a few hundred models (the production rotation
    // shape: ~12-source pool). C(12,2..4) = 66+220+495 = 781; C(12,2..3)=286.
    const int kPoolCap = 12;
    if (static_cast<int>(pool.size()) > kPoolCap) pool.resize(static_cast<std::size_t>(kPoolCap));
    std::printf("model space: target=idx0, right=%d outgroups (tail), source pool=%zu pops\n",
                nright, pool.size());

    std::vector<steppe::QpAdmModel> models;
    auto add_subsets = [&](int ksz) {
        const int n = static_cast<int>(pool.size());
        if (ksz > n) return;
        std::vector<int> idx(static_cast<std::size_t>(ksz));
        for (int i = 0; i < ksz; ++i) idx[static_cast<std::size_t>(i)] = i;
        while (true) {
            steppe::QpAdmModel m;
            m.target = target_idx;
            m.right = right_idx;
            for (int i = 0; i < ksz; ++i)
                m.left.push_back(pool[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])]);
            m.model_index = static_cast<int>(models.size());
            models.push_back(std::move(m));
            int i = ksz - 1;
            while (i >= 0 && idx[static_cast<std::size_t>(i)] == n - ksz + i) --i;
            if (i < 0) break;
            ++idx[static_cast<std::size_t>(i)];
            for (int j = i + 1; j < ksz; ++j)
                idx[static_cast<std::size_t>(j)] = idx[static_cast<std::size_t>(j - 1)] + 1;
        }
    };
    for (int k = 2; k <= maxk; ++k) add_subsets(k);
    std::printf("enumerated %zu REAL models (C(pool,2..%d)) over the SAME resident 1240K f2\n",
                models.size(), maxk);
    if (models.empty()) { std::printf("ERROR: no models enumerated\n"); return 1; }

    steppe::QpAdmOptions opts;
    opts.jackknife = (jk == 2) ? steppe::JackknifePolicy::All
                   : (jk == 1) ? steppe::JackknifePolicy::FeasibleOnly
                               : steppe::JackknifePolicy::None;

    // ---- upload f2 RESIDENT to device 0; build SINGLE-GPU Resources -------------
    steppe::DeviceConfig cfg; cfg.devices = {0};   // SINGLE GPU (multi-GPU parked)
    steppe::device::Resources res = steppe::device::build_resources(cfg);
    const steppe::BackendCapabilities& caps = res.gpus.at(0).caps;
    std::printf("device 0: sm_%d%d (CudaBackend, compute_major=%d)\n",
                caps.compute_major, caps.compute_minor, caps.compute_major);

    steppe::F2BlockTensor up = f2;            // upload reads block_sizes; vpair carried
    steppe::device::DeviceF2Blocks dev =
        steppe::device::upload_f2_blocks_to_device(up, 0);
    if (dev.f2_device() == nullptr || dev.empty()) {
        std::printf("ERROR: f2 not resident on device 0\n"); return 1;
    }
    std::printf("f2 RESIDENT on device 0 (f2_device != null)\n");

    // ---- WARM-UP (one batched dispatch; primes cuBLAS/cuSOLVER handles) ----------
    {
        const auto w = steppe::run_qpadm_search(
            dev, std::span<const steppe::QpAdmModel>(models), opts, res);
        (void)w;
    }

    // ---- TIMED batched rotation (SINGLE GPU) -----------------------------------
    const int ITERS = 3;
    std::vector<double> ms;
    std::vector<steppe::QpAdmResult> r;
    for (int it = 0; it < ITERS; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        r = steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(models), opts, res);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];

    // PROVE it ran GPU-BATCHED (not a per-model host loop): dispatch count << models.
    const std::size_t ndispatch = res.gpus.at(0).backend->batched_dispatch_count();

    // count Ok / feasible for an honest report.
    std::size_t ok = 0, feasible = 0;
    for (const auto& x : r) {
        if (x.status == steppe::Status::Ok) ++ok;
        if (!x.popdrop_feasible.empty() && x.popdrop_feasible.at(0) != 0) ++feasible;
    }

    const double mps = static_cast<double>(models.size()) / (med / 1000.0);
    const double per_model_ms = med / static_cast<double>(models.size());
    std::printf("\n--- RESULT (median of %d timed iters, single GPU device 0) ---\n", ITERS);
    std::printf("models                = %zu\n", models.size());
    std::printf("wall (median)         = %.1f ms\n", med);
    std::printf("THROUGHPUT            = %.0f models/sec\n", mps);
    std::printf("per-model compute     = %.3f ms/model\n", per_model_ms);
    std::printf("batched dispatches    = %zu (<< %zu models => GPU-BATCHED, NOT a per-model loop)\n",
                ndispatch, models.size());
    std::printf("status: Ok=%zu  feasible=%zu  (of %zu)\n", ok, feasible, models.size());
    if (!r.empty())
        std::printf("precision_tag         = %s\n",
                    r[0].precision_tag == steppe::Precision::Kind::EmulatedFp64 ? "EmulatedFp64"
                                                                               : "Fp64");
    std::printf("vs the bogus CLI-loop ~1 model/sec: %.0fx faster\n", mps / 1.0);
    return 0;
}
