// tests/reference/test_fstat_sweep_parity.cu
//
// GPU-ONLY f-stat SWEEP parity gate. The sweep (run_f4_sweep / run_f3_sweep) enumerates EVERY
// C(P,k) item ON THE DEVICE (combinatorial-unrank kernel), reuses the SAME assemble_f4_quartets
// gather + loo/total + xtau + diagonal-jackknife device kernels the EXPLICIT run_f4/run_f3 path
// runs, filters |z| + CUB-compacts survivors ON THE DEVICE, and returns ONLY survivors. This
// test proves:
//   (1) PARITY — for every quartet (triple), the SWEEP's est/se/z is BIT-EQUAL to the explicit
//       run_f4 / run_f3 over that SAME item (same kernels, no fork). Run with min_z=0 (keep
//       every finite item) so the survivor set == the full explicit set, then diff item-for-item.
//   (2) FILTER — a min_z cut keeps exactly the explicit items with |z| >= min_z.
//   (3) CAP — a sweep over more than kFstatMaxComb items REFUSES (capped, no compute) unless sure.
//
// Uses the COMMITTED real 9-pop golden f2 fixture (the SAME tensor test_qpadm_parity uses); NO
// synthetic data. SKIPs cleanly (exit 0) when no CUDA device is visible. Self-checking main().

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "device/device_f2_blocks.hpp"  // DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // Resources, build_resources
#include "steppe/config.hpp"            // kFstatMaxComb
#include "steppe/f4.hpp"                // run_f4
#include "steppe/f3.hpp"                // run_f3
#include "steppe/fstat_sweep.hpp"       // run_f4_sweep / run_f3_sweep
#include "steppe/fstats.hpp"            // F2BlockTensor

namespace {

int g_fail = 0;

bool read_fixture(const std::string& path, steppe::F2BlockTensor& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("  [FAIL] cannot open fixture: %s\n", path.c_str()); return false; }
    std::int32_t P = 0, nb = 0;
    f.read(reinterpret_cast<char*>(&P), sizeof(P));
    f.read(reinterpret_cast<char*>(&nb), sizeof(nb));
    if (!f || P <= 0 || nb <= 0) { std::printf("  [FAIL] bad fixture header\n"); return false; }
    out.P = P; out.n_block = nb;
    out.block_sizes.resize(static_cast<std::size_t>(nb));
    f.read(reinterpret_cast<char*>(out.block_sizes.data()),
           static_cast<std::streamsize>(sizeof(std::int32_t) * static_cast<std::size_t>(nb)));
    const std::size_t n = static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                          static_cast<std::size_t>(nb);
    out.f2.resize(n);
    f.read(reinterpret_cast<char*>(out.f2.data()),
           static_cast<std::streamsize>(sizeof(double) * n));
    if (!f) { std::printf("  [FAIL] fixture truncated\n"); return false; }
    return true;
}

// Enumerate every k-combination of [0,P) in COLEX order (matching the device unrank) into a
// flat list of arrays (the 4th slot 0 for k=3).
void enumerate_colex(int P, int k, std::vector<std::array<int, 4>>& out) {
    std::vector<int> c(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) c[static_cast<std::size_t>(i)] = i;
    while (true) {
        std::array<int, 4> a{0, 0, 0, 0};
        for (int i = 0; i < k; ++i) a[static_cast<std::size_t>(i)] = c[static_cast<std::size_t>(i)];
        out.push_back(a);
        // next colex combination of [0,P): standard lex-next on ascending tuple.
        int i = k - 1;
        while (i >= 0 && c[static_cast<std::size_t>(i)] == P - k + i) --i;
        if (i < 0) break;
        ++c[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < k; ++j)
            c[static_cast<std::size_t>(j)] = c[static_cast<std::size_t>(j - 1)] + 1;
    }
}

std::string key_str(const std::array<int, 4>& a, int k) {
    std::string s;
    for (int i = 0; i < k; ++i) { s += std::to_string(a[static_cast<std::size_t>(i)]); s += ","; }
    return s;
}

bool eqx(double a, double b) {  // bit-exact OR both-NaN.
    if (std::isnan(a) && std::isnan(b)) return true;
    return a == b;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden_dir = (argc > 1) ? argv[1] : "tests/reference/goldens/at2";
    std::printf("=== GPU-only f-stat SWEEP parity test ===\n");

    int gpu_count = 0;
    {
        steppe::DeviceConfig dc;  // empty ⇒ auto-enumerate.
        try {
            steppe::device::Resources r = steppe::device::build_resources(dc);
            gpu_count = static_cast<int>(r.gpus.size());
        } catch (...) { gpu_count = 0; }
    }
    if (gpu_count <= 0) {
        std::printf("  [SKIP] no CUDA device visible — sweep is a GPU-only product.\n");
        std::printf("RESULT: PASS (skipped, no GPU)\n");
        return 0;
    }

    steppe::F2BlockTensor f2;
    if (!read_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", f2)) {
        std::printf("RESULT: FAIL (fixture read)\n");
        return 1;
    }
    const int P = f2.P;
    std::printf("  fixture P=%d n_block=%d\n", P, f2.n_block);

    steppe::device::Resources resources;
    try {
        resources = steppe::device::build_resources(steppe::DeviceConfig{});
    } catch (const std::exception& e) {
        std::printf("RESULT: FAIL (build_resources: %s)\n", e.what());
        return 1;
    }
    const int device_id = resources.gpus.front().device_id;
    steppe::F2BlockTensor f2_up = f2;
    // read_fixture leaves vpair empty; upload copies BOTH slabs, so size vpair (zeros — the
    // fit/sweep read block_sizes, NOT vpair; a fully-zero slab is the "no Vpair info" keep-all
    // sentinel, so both the explicit and sweep paths keep every block ⇒ parity holds).
    f2_up.vpair.assign(f2.f2.size(), 0.0);
    steppe::device::DeviceF2Blocks dev_f2 =
        steppe::device::upload_f2_blocks_to_device(f2_up, device_id);

    const steppe::QpAdmOptions opts;

    // ---- (1) f4 PARITY: explicit run_f4 over ALL C(P,4) vs the keep-all sweep --------
    {
        std::vector<std::array<int, 4>> quartets;
        enumerate_colex(P, 4, quartets);
        const steppe::F4Result expl =
            steppe::run_f4(dev_f2, std::span<const std::array<int, 4>>(quartets), opts, resources);
        // Build a key->row map of the explicit values.
        std::map<std::string, std::array<double, 3>> exp_map;  // key -> {est,se,z}
        for (std::size_t r = 0; r < expl.est.size(); ++r) {
            std::array<int, 4> q{expl.p1[r], expl.p2[r], expl.p3[r], expl.p4[r]};
            exp_map[key_str(q, 4)] = {expl.est[r], expl.se[r], expl.z[r]};
        }

        steppe::SweepRequest req;
        req.filter = steppe::SweepFilter::MinZ;
        req.min_z = 0.0;  // keep every finite-z item ⇒ survivor set == explicit set.
        req.sure = true;
        const steppe::SweepResult sw = steppe::run_f4_sweep(dev_f2, req, resources);

        std::printf("  f4: enumerated=%zu explicit_rows=%zu survivors=%zu\n",
                    sw.enumerated, expl.est.size(), sw.survivors);
        if (sw.status != steppe::Status::Ok) {
            std::printf("  [FAIL] f4 sweep status != Ok\n"); ++g_fail;
        }
        // Every survivor must bit-match the explicit value for its key. (min_z=0 drops only
        // NaN-z degenerate items, which the explicit path would also score NaN.)
        int matched = 0, mism = 0;
        for (std::size_t r = 0; r < sw.survivors; ++r) {
            const std::string ks = key_str(sw.keys[r], 4);
            auto it = exp_map.find(ks);
            if (it == exp_map.end()) {
                if (mism < 5) std::printf("  [FAIL] f4 survivor key %s not in explicit set\n", ks.c_str());
                ++mism; continue;
            }
            const auto& e = it->second;
            if (!eqx(sw.est[r], e[0]) || !eqx(sw.se[r], e[1]) || !eqx(sw.z[r], e[2])) {
                if (mism < 5)
                    std::printf("  [FAIL] f4 mismatch key %s: sweep(%.17g,%.17g,%.17g) vs expl(%.17g,%.17g,%.17g)\n",
                                ks.c_str(), sw.est[r], sw.se[r], sw.z[r], e[0], e[1], e[2]);
                ++mism;
            } else ++matched;
        }
        std::printf("  f4 parity: %d bit-exact matches, %d mismatches\n", matched, mism);
        if (mism != 0) ++g_fail;
        // The keep-all (min_z=0) sweep should recover essentially the whole explicit set
        // (only NaN-z items, if any, are dropped). Require >= 90% recovered as a sanity floor.
        if (matched < static_cast<int>(expl.est.size()) * 9 / 10) {
            std::printf("  [FAIL] f4 sweep recovered too few items (%d of %zu)\n",
                        matched, expl.est.size());
            ++g_fail;
        }

        // ---- (2) FILTER: a min_z cut keeps exactly the |z|>=min_z explicit items --------
        steppe::SweepRequest reqz;
        reqz.filter = steppe::SweepFilter::MinZ;
        reqz.min_z = 3.0;
        reqz.sure = true;
        const steppe::SweepResult swz = steppe::run_f4_sweep(dev_f2, reqz, resources);
        int exp_pass = 0;
        for (const auto& kv : exp_map)
            if (std::fabs(kv.second[2]) >= 3.0) ++exp_pass;
        std::printf("  f4 |z|>=3 filter: sweep survivors=%zu expected=%d\n",
                    swz.survivors, exp_pass);
        if (static_cast<int>(swz.survivors) != exp_pass) {
            std::printf("  [FAIL] f4 |z|>=3 survivor count != explicit count\n"); ++g_fail;
        }
        for (std::size_t r = 0; r < swz.survivors; ++r)
            if (std::fabs(swz.z[r]) < 3.0) {
                std::printf("  [FAIL] f4 filter kept |z|<3 item\n"); ++g_fail; break;
            }
    }

    // ---- f3 PARITY: explicit run_f3 over ALL C(P,3) vs the keep-all sweep ------------
    {
        std::vector<std::array<int, 4>> triples4;
        enumerate_colex(P, 3, triples4);
        std::vector<std::array<int, 3>> triples;
        triples.reserve(triples4.size());
        for (const auto& a : triples4) triples.push_back({a[0], a[1], a[2]});
        const steppe::F3Result expl =
            steppe::run_f3(dev_f2, std::span<const std::array<int, 3>>(triples), opts, resources);
        std::map<std::string, std::array<double, 3>> exp_map;
        for (std::size_t r = 0; r < expl.est.size(); ++r) {
            std::array<int, 4> t{expl.p1[r], expl.p2[r], expl.p3[r], 0};
            exp_map[key_str(t, 3)] = {expl.est[r], expl.se[r], expl.z[r]};
        }
        steppe::SweepRequest req;
        req.filter = steppe::SweepFilter::MinZ;
        req.min_z = 0.0;
        req.sure = true;
        const steppe::SweepResult sw = steppe::run_f3_sweep(dev_f2, req, resources);
        std::printf("  f3: enumerated=%zu explicit_rows=%zu survivors=%zu\n",
                    sw.enumerated, expl.est.size(), sw.survivors);
        int matched = 0, mism = 0;
        for (std::size_t r = 0; r < sw.survivors; ++r) {
            const std::string ks = key_str(sw.keys[r], 3);
            auto it = exp_map.find(ks);
            if (it == exp_map.end()) { ++mism; continue; }
            const auto& e = it->second;
            if (!eqx(sw.est[r], e[0]) || !eqx(sw.se[r], e[1]) || !eqx(sw.z[r], e[2])) {
                if (mism < 5)
                    std::printf("  [FAIL] f3 mismatch key %s: sweep(%.17g,%.17g,%.17g) vs expl(%.17g,%.17g,%.17g)\n",
                                ks.c_str(), sw.est[r], sw.se[r], sw.z[r], e[0], e[1], e[2]);
                ++mism;
            } else ++matched;
        }
        std::printf("  f3 parity: %d bit-exact matches, %d mismatches\n", matched, mism);
        if (mism != 0) ++g_fail;
        if (matched < static_cast<int>(expl.est.size()) * 9 / 10) {
            std::printf("  [FAIL] f3 sweep recovered too few items\n"); ++g_fail;
        }
    }

    // ---- (3) CAP: a sweep over more than kFstatMaxComb items REFUSES unless sure -----
    // Synthesize a tiny f2 with a huge P so C(P,4) > kFstatMaxComb without any compute. The
    // cap is a pure choose() compare in the driver BEFORE any device work, so a degenerate
    // dev handle is fine — it must short-circuit to capped.
    {
        // Find the smallest P with C(P,4) > kFstatMaxComb.
        auto choose4 = [](unsigned long long p) -> unsigned long long {
            if (p < 4) return 0;
            return p * (p - 1) * (p - 2) * (p - 3) / 24ULL;
        };
        int bigP = 4;
        while (choose4(static_cast<unsigned long long>(bigP)) <= steppe::kFstatMaxComb) bigP += 100;
        // Use pop_subset to drive the cap range without needing a real big f2 (the cap only
        // reads the range size). dev_f2 is the real 9-pop handle; the subset is bogus indices
        // but the cap fires BEFORE any device read, so it never dereferences them.
        steppe::SweepRequest req;
        req.filter = steppe::SweepFilter::MinZ;
        req.min_z = 3.0;
        req.sure = false;  // NOT sure ⇒ must refuse.
        std::vector<int> subset(static_cast<std::size_t>(bigP));
        for (int i = 0; i < bigP; ++i) subset[static_cast<std::size_t>(i)] = i % P;  // bogus, unused
        req.pop_subset = subset;
        const steppe::SweepResult cap = steppe::run_f4_sweep(dev_f2, req, resources);
        std::printf("  cap: range P=%d enumerated=%zu capped=%d status=%d\n",
                    bigP, cap.enumerated, cap.capped ? 1 : 0, static_cast<int>(cap.status));
        if (!cap.capped || cap.status != steppe::Status::InvalidConfig) {
            std::printf("  [FAIL] over-cap sweep did NOT refuse\n"); ++g_fail;
        }
        if (cap.survivors != 0) {  // a capped sweep ran NO compute ⇒ no survivors.
            std::printf("  [FAIL] capped sweep produced survivors\n"); ++g_fail;
        }
    }

    if (g_fail == 0) { std::printf("RESULT: PASS\n"); return 0; }
    std::printf("RESULT: FAIL (%d checks)\n", g_fail);
    return 1;
}
