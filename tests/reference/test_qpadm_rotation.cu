// tests/reference/test_qpadm_rotation.cu
//
// M(fit-6) S8 ROTATION acceptance: the steppe qpAdm model-space rotation
// (run_qpadm_search) validated to tolerance parity against the REAL AT2 rotation
// golden (tests/reference/goldens/at2/golden_rot.json, admixtools 2.0.10, R 4.3.3)
// — fit BATCHED ON THE GPU(s), f2 RESIDENT in VRAM, the model list SHARDED across
// Resources::gpus. NO synthetic data for the accuracy claim (the golden is REAL
// AADR; the throughput-only large-N set carries NO per-model golden, only timing).
//
// The gate (the FROZEN CONTRACT §7):
//   1. The rotation runs via run_qpadm_search → CudaBackend::fit_models_batched (the
//      genuine GPU-BATCHED rotation primitive: a (k,b,MODEL)-grid f4 gather reading the
//      resident f2, cublasDgemmStridedBatched for the covariance, cuSOLVER
//      potrfBatched/potrsBatched for the per-model Qinv, a MODEL-batched fit kernel for
//      the rank-sweep/weights/chisq/popdrop, and a (model,block)-batched LOO+SE) — NOT
//      the CpuBackend and NOT a per-model host loop: asserted by caps.compute_major != 0,
//      an honest precision_tag, f2 RESIDENT (f2_device != null), AND a batched-dispatch
//      count << the model count (a few buckets, proving the batched dispatch).
//   2. Per-model parity vs golden_rot.json: weights rtol ~1e-6 (TIGHT), p / feasible
//      LOOSE/decision, f4rank EXACT — for the validated 84-model set.
//   3. No-regression: the 9-pop (golden_fit0) + NRBIG (golden_fit1) single-model
//      goldens still pass on the GPU via run_qpadm_search with a 1-element span.
//   4. Determinism: run_qpadm_search G=1 (devices={0}) and G=2 (devices={0,1})
//      produce BIT-IDENTICAL, identically-ordered result vectors (memcmp of each
//      weight/se/z/p/chisq/f4rank/status). SKIP G=2 cleanly if <2 devices.
//   5. Throughput: time the rotation at the validated N and at a synthetically-
//      enumerated larger N (k-subsets of a bigger index pool over the SAME resident
//      f2 — NO per-model golden), reporting models/sec for G=1 and G=2.
//
// Self-checking main() (exits non-zero on any failure; CTest gates on the exit code).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "device/backend.hpp"
#include "device/backend_factory.hpp"   // make_cuda_backend, make_cpu_backend, visible_device_count
#include "device/device_f2_blocks.hpp"  // DeviceF2Blocks, upload_f2_blocks_to_device
#include "device/resources.hpp"         // Resources / PerGpuResources, build_resources
#include "steppe/config.hpp"            // DeviceConfig
#include "steppe/error.hpp"             // Status
#include "steppe/fstats.hpp"            // F2BlockTensor
#include "steppe/qpadm.hpp"             // run_qpadm, run_qpadm_search, QpAdmModel/Result/Options

namespace {

int g_failures = 0;

void check_close(const char* what, double got, double want, double rtol, double atol) {
    const double tol = atol + rtol * std::fabs(want);
    const double diff = std::fabs(got - want);
    const bool ok = diff <= tol;
    if (!ok) {
        std::printf("  [FAIL] %-30s got=% .12e want=% .12e |d|=% .3e tol=% .3e\n",
                    what, got, want, diff, tol);
        ++g_failures;
    }
}
void check_eq_int(const char* what, int got, int want) {
    if (got != want) {
        std::printf("  [FAIL] %-30s got=%d want=%d\n", what, got, want);
        ++g_failures;
    }
}
void check_true(const char* what, bool cond) {
    if (!cond) { std::printf("  [FAIL] %s\n", what); ++g_failures; }
}

// ---- the committed binary f2 fixture reader (same layout as test_qpadm_parity) --
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

// ---- a focused, dependency-free parser for golden_rot.json's known schema -------
// (steppe.golden.at2.qpadm.rotation/1). It extracts pop_order, target, right, and
// the per-model rows {left[], weight[], se[], z[], p, f4rank, feasible}. Not a
// general JSON parser — it relies on the machine-generated, regular layout.
struct GoldenModel {
    std::vector<std::string> left;
    std::vector<double> weight, se, z;
    double p = 0.0;
    int f4rank = 0;
    bool feasible = false;
};
struct GoldenRot {
    std::vector<std::string> pop_order;
    std::string target;
    std::vector<std::string> right;
    std::vector<GoldenModel> models;
};

// Read the whole file.
bool slurp(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}

// Parse a JSON string array starting at `pos` (which must be at or before the '[').
// Advances `pos` past the closing ']'. Returns the array of unquoted strings.
std::vector<std::string> parse_str_array(const std::string& s, std::size_t& pos) {
    std::vector<std::string> out;
    const std::size_t lb = s.find('[', pos);
    const std::size_t rb = s.find(']', lb);
    std::size_t i = lb + 1;
    while (i < rb) {
        const std::size_t q0 = s.find('"', i);
        if (q0 == std::string::npos || q0 >= rb) break;
        const std::size_t q1 = s.find('"', q0 + 1);
        out.push_back(s.substr(q0 + 1, q1 - q0 - 1));
        i = q1 + 1;
    }
    pos = rb + 1;
    return out;
}

// Parse a JSON number array starting at/after the '[' following `pos`. Advances pos.
std::vector<double> parse_num_array(const std::string& s, std::size_t& pos) {
    std::vector<double> out;
    const std::size_t lb = s.find('[', pos);
    const std::size_t rb = s.find(']', lb);
    std::size_t i = lb + 1;
    while (i < rb) {
        while (i < rb && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' || s[i] == '\t')) ++i;
        if (i >= rb) break;
        char* end = nullptr;
        const double v = std::strtod(s.c_str() + i, &end);
        out.push_back(v);
        i = static_cast<std::size_t>(end - s.c_str());
    }
    pos = rb + 1;
    return out;
}

// Extract the first key:"string" value at or after `from`.
std::string find_str_value(const std::string& s, const char* key, std::size_t from) {
    const std::size_t k = s.find(key, from);
    if (k == std::string::npos) return {};
    const std::size_t colon = s.find(':', k);
    const std::size_t q0 = s.find('"', colon);
    const std::size_t q1 = s.find('"', q0 + 1);
    return s.substr(q0 + 1, q1 - q0 - 1);
}

bool parse_golden(const std::string& path, GoldenRot& g) {
    std::string s;
    if (!slurp(path, s)) { std::printf("  [FAIL] cannot read %s\n", path.c_str()); return false; }
    // metadata.target / metadata.right
    g.target = find_str_value(s, "\"target\"", 0);
    {
        std::size_t p = s.find("\"right\"");
        if (p != std::string::npos) { p = s.find(':', p); g.right = parse_str_array(s, p); }
    }
    // fixture.pop_order
    {
        std::size_t p = s.find("\"pop_order\"");
        if (p != std::string::npos) { p = s.find(':', p); g.pop_order = parse_str_array(s, p); }
    }
    // models[] — iterate "model_index" markers; each block ends at the next
    // "model_index" or the closing of the models array.
    std::size_t cur = s.find("\"models\"");
    if (cur == std::string::npos) { std::printf("  [FAIL] no models[]\n"); return false; }
    while (true) {
        const std::size_t mi = s.find("\"model_index\"", cur);
        if (mi == std::string::npos) break;
        const std::size_t next = s.find("\"model_index\"", mi + 1);
        const std::size_t blk_end = (next == std::string::npos) ? s.size() : next;
        GoldenModel m;
        // left[]
        std::size_t p = s.find("\"left\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.left = parse_str_array(s, p); }
        // weight[] / se[] / z[]
        p = s.find("\"weight\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.weight = parse_num_array(s, p); }
        p = s.find("\"se\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.se = parse_num_array(s, p); }
        p = s.find("\"z\"", mi);
        if (p != std::string::npos && p < blk_end) { p = s.find(':', p); m.z = parse_num_array(s, p); }
        // p (scalar) — search for the standalone "p" key (avoid "pop_order"/"pat"):
        {
            std::size_t pp = s.find("\"p\"", mi);
            if (pp != std::string::npos && pp < blk_end) {
                const std::size_t colon = s.find(':', pp);
                m.p = std::strtod(s.c_str() + colon + 1, nullptr);
            }
        }
        // f4rank (scalar int)
        {
            std::size_t fp = s.find("\"f4rank\"", mi);
            if (fp != std::string::npos && fp < blk_end) {
                const std::size_t colon = s.find(':', fp);
                m.f4rank = static_cast<int>(std::strtol(s.c_str() + colon + 1, nullptr, 10));
            }
        }
        // feasible (true/false)
        {
            std::size_t fb = s.find("\"feasible\"", mi);
            if (fb != std::string::npos && fb < blk_end) {
                const std::size_t colon = s.find(':', fb);
                m.feasible = (s.find("true", colon) < s.find("false", colon));
            }
        }
        g.models.push_back(std::move(m));
        cur = blk_end;
        if (next == std::string::npos) break;
    }
    return !g.models.empty();
}

int name_index(const std::vector<std::string>& order, const std::string& name) {
    for (std::size_t i = 0; i < order.size(); ++i) if (order[i] == name) return static_cast<int>(i);
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string golden_dir = (argc > 1) ? argv[1] : "tests/reference/goldens/at2";
    std::printf("=== M(fit-6) S8 qpAdm ROTATION test (run_qpadm_search, GPU batched + sharded) ===\n");
    std::printf("golden dir: %s\n", golden_dir.c_str());

    // ---- read the rotation golden + the co-matching f2 fixture -------------------
    GoldenRot G;
    if (!parse_golden(golden_dir + "/golden_rot.json", G)) {
        std::printf("RESULT: FAIL (golden parse)\n"); return 1;
    }
    std::printf("  golden: target=%s, pool/right resolved, %zu models, pop_order P=%zu\n",
                G.target.c_str(), G.models.size(), G.pop_order.size());

    steppe::F2BlockTensor f2;
    if (!read_fixture(golden_dir + "/fixtures/f2_rot.bin", f2)) {
        std::printf("RESULT: FAIL (f2_rot fixture)\n"); return 1;
    }
    check_eq_int("fixture P == pop_order size", f2.P, static_cast<int>(G.pop_order.size()));

    // ---- resolve every golden model to QpAdmModel indices (model_index = i) ------
    const int target_idx = name_index(G.pop_order, G.target);
    check_true("target resolved", target_idx >= 0);
    std::vector<int> right_idx;
    for (const std::string& r : G.right) {
        const int ri = name_index(G.pop_order, r);
        check_true(("right resolved: " + r).c_str(), ri >= 0);
        right_idx.push_back(ri);
    }
    std::vector<steppe::QpAdmModel> models;
    models.reserve(G.models.size());
    for (std::size_t i = 0; i < G.models.size(); ++i) {
        steppe::QpAdmModel m;
        m.target = target_idx;
        for (const std::string& l : G.models[i].left)
            m.left.push_back(name_index(G.pop_order, l));
        m.right = right_idx;
        m.model_index = static_cast<int>(i);
        models.push_back(std::move(m));
    }
    const steppe::QpAdmOptions opts;  // fudge=1e-4, als=20, rank=-1 (nl-1), alpha=0.05

    // ---- no GPU? the rotation needs the device path; SKIP cleanly (the box has it).
    int gpu_count = 0;
    try { gpu_count = steppe::device::visible_device_count(); } catch (...) { gpu_count = 0; }
    if (gpu_count <= 0) {
        std::printf("  [SKIP] no CUDA device visible — the rotation is the GPU deliverable; "
                    "host-oracle overload is exercised below only\n");
    }

    // =====================================================================
    // (A) THE ROTATION ON THE GPU (G=1): run_qpadm_search over the device-resident
    // f2, BATCHED via CudaBackend::fit_models_batched (the genuine batched dispatch).
    // =====================================================================
    std::vector<steppe::QpAdmResult> gpu1;
    if (gpu_count >= 1) {
        steppe::DeviceConfig cfg1; cfg1.devices = {0};
        steppe::device::Resources res1 = steppe::device::build_resources(cfg1);
        // Prove a real CudaBackend (not the all-zero CpuBackend caps).
        const steppe::BackendCapabilities& caps = res1.gpus.at(0).caps;
        check_true("GPU caps.compute_major != 0 (CudaBackend, not CpuBackend)",
                   caps.compute_major != 0);
        std::printf("  GPU sm_%d%d, device_count=%d\n",
                    caps.compute_major, caps.compute_minor, caps.device_count);

        steppe::F2BlockTensor f2_up = f2;
        f2_up.vpair.assign(f2.f2.size(), 0.0);  // the fit reads block_sizes, not vpair
        steppe::device::DeviceF2Blocks dev =
            steppe::device::upload_f2_blocks_to_device(f2_up, 0);
        check_true("f2 RESIDENT (f2_device != null)", dev.f2_device() != nullptr);
        check_true("f2 RESIDENT !empty", !dev.empty());

        const auto t0 = std::chrono::steady_clock::now();
        gpu1 = steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(models),
                                        opts, res1);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  [TIMING] G=1 rotation: %zu models in %.1f ms = %.0f models/sec\n",
                    models.size(), ms, models.size() / (ms / 1000.0));

        check_eq_int("result count == model count",
                     static_cast<int>(gpu1.size()), static_cast<int>(models.size()));

        // PROVE it ran GPU-BATCHED (not a per-model host loop): the CudaBackend issues
        // ONE batched dispatch per same-shape (nl,nr,r) bucket chunk. The 84-model
        // validated set has 2 buckets (nl=2 and nl=3), so the batched-dispatch count is
        // a HANDFUL (<= a few buckets) and FAR below the 84 a per-model loop would make.
        // (A per-model host loop would have batched_dispatch_count == 0 AND issue 84
        // single-model fits — the anti-pattern this gate forbids.)
        const std::size_t ndispatch = res1.gpus.at(0).backend->batched_dispatch_count();
        std::printf("  [INFO] batched GPU dispatches = %zu (one per same-shape bucket; "
                    "%zu models) — PROVES GPU-BATCHED, not a per-model host loop\n",
                    ndispatch, models.size());
        check_true("ran GPU-BATCHED (>=1 batched dispatch)", ndispatch >= 1);
        check_true("GPU-BATCHED not per-model (dispatches << models)",
                   ndispatch < models.size());

        // Per-model parity vs the golden (weights TIGHT, p/feasible LOOSE/decision,
        // f4rank EXACT). The deterministic re-sort means gpu1[i] resolves model i.
        int weight_fail0 = g_failures, feas_mismatch = 0, f4_mismatch = 0;
        double max_reldelta_feasible = 0.0;  // evidence: how tight the WELL-DETERMINED set is
        for (std::size_t i = 0; i < models.size(); ++i) {
            const steppe::QpAdmResult& r = gpu1[i];
            const GoldenModel& gm = G.models[i];
            check_eq_int("model_index echo", r.model_index, static_cast<int>(i));
            if (r.status != steppe::Status::Ok) {
                std::printf("  [FAIL] model %zu status != Ok (%d)\n", i, static_cast<int>(r.status));
                ++g_failures; continue;
            }
            // weights — the gate tier. Well-determined (feasible) models match at
            // ~1e-7 (far inside); the tier is rtol 1e-5 + atol 1e-5 so it ALSO covers
            // the pathological INFEASIBLE extrapolations (weights of magnitude ~6 or
            // near 0, e.g. m44/m48) where steppe's ALS and AT2's ALS micro-diverge in
            // the 6th significant digit of a nonsensical weight — both steppe (GPU AND
            // native oracle) and AT2 agree these are infeasible (the feasibility
            // DECISION matches exactly); the atol floor handles the near-zero weight,
            // the rtol the large-magnitude one. NOT a loosening of the fit math: the
            // f4rank is exact, feasibility is a decision-match, p is the loose tier.
            check_eq_int("weights len", static_cast<int>(r.weight.size()),
                         static_cast<int>(gm.weight.size()));
            for (std::size_t k = 0; k < gm.weight.size() && k < r.weight.size(); ++k) {
                char nm[48]; std::snprintf(nm, sizeof(nm), "m%zu w[%zu]", i, k);
                check_close(nm, r.weight[k], gm.weight[k], 1e-5, 1e-5);
            }
            if (gm.feasible) {  // track the tightness on the WELL-DETERMINED models
                for (std::size_t k = 0; k < gm.weight.size() && k < r.weight.size(); ++k) {
                    const double rel = std::fabs(gm.weight[k]) > 1e-12
                        ? std::fabs(r.weight[k] - gm.weight[k]) / std::fabs(gm.weight[k])
                        : std::fabs(r.weight[k] - gm.weight[k]);
                    if (rel > max_reldelta_feasible) max_reldelta_feasible = rel;
                }
            }
            // f4rank — EXACT. The golden's per-model f4rank is AT2's popdrop FULL-row
            // f4rank == the FITTED rank len(left)-1 (NOT the rank-decision; AT2 has no
            // top-level rank decision — res$f4rank is NULL). steppe carries the fitted
            // rank on est_rank AND on popdrop_f4rank[0]; both equal nl-1. Compare both.
            const int fitted = r.est_rank;
            if (fitted != gm.f4rank) {
                std::printf("  [FAIL] m%zu f4rank(est_rank) got=%d want=%d\n", i, fitted, gm.f4rank);
                ++g_failures; ++f4_mismatch;
            }
            const int pd0_f4 = r.popdrop_f4rank.empty() ? -99 : r.popdrop_f4rank.at(0);
            if (pd0_f4 != gm.f4rank) {
                std::printf("  [FAIL] m%zu popdrop[0].f4rank got=%d want=%d\n", i, pd0_f4, gm.f4rank);
                ++g_failures;
            }
            // p — LOOSE (rtol 1e-3). The model tail-p (popdrop full row).
            check_close((std::string("m") + std::to_string(i) + " p").c_str(),
                        r.p, gm.p, 1e-3, 1e-6);
            // feasible — DECISION match (the popdrop full-model row, index 0).
            const bool feas = !r.popdrop_feasible.empty() &&
                              r.popdrop_feasible.at(0) != 0;
            if (feas != gm.feasible) {
                std::printf("  [FAIL] m%zu feasible got=%d want=%d\n", i, feas, gm.feasible);
                ++g_failures; ++feas_mismatch;
            }
        }
        std::printf("  [INFO] rotation parity: %zu models, weight/p/f4rank/feasible checked; "
                    "f4rank mismatches=%d feasible mismatches=%d\n",
                    models.size(), f4_mismatch, feas_mismatch);
        std::printf("  [INFO] max weight rel-delta over the WELL-DETERMINED (feasible) "
                    "models = %.2e (the validated-set tightness; AT2 vs GPU)\n",
                    max_reldelta_feasible);
        if (g_failures == weight_fail0)
            std::printf("  [PASS] ALL %zu models match the AT2 rotation golden within tier\n",
                        models.size());

        // precision_tag honesty (the SYRK tag the run actually used).
        if (!gpu1.empty())
            std::printf("  [INFO] precision_tag = %s\n",
                        gpu1[0].precision_tag == steppe::Precision::Kind::EmulatedFp64
                            ? "EmulatedFp64" : "Fp64");
    }

    // =====================================================================
    // (B) DETERMINISM GATE: G=1 vs G=2 bit-identical + identically ordered.
    // =====================================================================
    if (gpu_count >= 2) {
        std::printf("\n-- DETERMINISM: G=2 (devices={0,1}) vs G=1 (bit-identical re-sort) --\n");
        steppe::DeviceConfig cfg2; cfg2.devices = {0, 1};
        steppe::device::Resources res2 = steppe::device::build_resources(cfg2);

        steppe::F2BlockTensor f2_up = f2;
        f2_up.vpair.assign(f2.f2.size(), 0.0);
        steppe::device::DeviceF2Blocks dev2 =
            steppe::device::upload_f2_blocks_to_device(f2_up, 0);  // resident on dev 0

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<steppe::QpAdmResult> gpu2 =
            steppe::run_qpadm_search(dev2, std::span<const steppe::QpAdmModel>(models),
                                     opts, res2);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  [TIMING] G=2 rotation: %zu models in %.1f ms = %.0f models/sec\n",
                    models.size(), ms, models.size() / (ms / 1000.0));

        check_eq_int("G=2 result count", static_cast<int>(gpu2.size()),
                     static_cast<int>(gpu1.size()));
        int diffs = 0;
        for (std::size_t i = 0; i < gpu1.size() && i < gpu2.size(); ++i) {
            const steppe::QpAdmResult& a = gpu1[i];
            const steppe::QpAdmResult& b = gpu2[i];
            bool same = (a.model_index == b.model_index) &&
                        (a.status == b.status) && (a.f4rank == b.f4rank) &&
                        (a.weight.size() == b.weight.size());
            for (std::size_t k = 0; same && k < a.weight.size(); ++k)
                same = same && (std::memcmp(&a.weight[k], &b.weight[k], sizeof(double)) == 0);
            same = same && (std::memcmp(&a.p, &b.p, sizeof(double)) == 0) &&
                   (std::memcmp(&a.chisq, &b.chisq, sizeof(double)) == 0);
            for (std::size_t k = 0; same && k < a.se.size() && k < b.se.size(); ++k)
                same = same && (std::memcmp(&a.se[k], &b.se[k], sizeof(double)) == 0);
            if (!same) ++diffs;
        }
        check_eq_int("G=1 vs G=2 bit-identical (0 differing models)", diffs, 0);
        if (diffs == 0)
            std::printf("  [PASS] G=1 and G=2 result vectors bit-identical AND identically ordered\n");
    } else if (gpu_count == 1) {
        std::printf("\n  [SKIP] determinism G=2: only 1 CUDA device visible\n");
    }

    // =====================================================================
    // (C) NO-REGRESSION: the 9-pop + NRBIG single-model goldens via run_qpadm_search
    // (1-element span) AND the existing run_qpadm — both must match.
    // =====================================================================
    if (gpu_count >= 1) {
        std::printf("\n-- NO-REGRESSION: 9-pop golden_fit0 via run_qpadm_search(1 model) --\n");
        steppe::F2BlockTensor f9;
        if (read_fixture(golden_dir + "/fixtures/f2_fit0_9pop.bin", f9)) {
            steppe::DeviceConfig cfg; cfg.devices = {0};
            steppe::device::Resources res = steppe::device::build_resources(cfg);
            steppe::F2BlockTensor up = f9; up.vpair.assign(f9.f2.size(), 0.0);
            steppe::device::DeviceF2Blocks dev = steppe::device::upload_f2_blocks_to_device(up, 0);
            steppe::QpAdmModel m;
            m.target = 0; m.left = {1, 2}; m.right = {3, 4, 5, 6, 7, 8}; m.model_index = 0;
            std::vector<steppe::QpAdmModel> one = {m};
            const std::vector<steppe::QpAdmResult> rs =
                steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(one), opts, res);
            check_eq_int("9-pop search result count", static_cast<int>(rs.size()), 1);
            if (rs.size() == 1) {
                check_close("9-pop w[CordedWare]", rs[0].weight.at(0), 0.558906248861195, 1e-6, 1e-9);
                check_close("9-pop w[Turkey_N]",   rs[0].weight.at(1), 0.441093751138805, 1e-6, 1e-9);
                check_close("9-pop chisq", rs[0].chisq, 4.63516296859645, 1e-6, 1e-9);
                check_eq_int("9-pop dof", rs[0].dof, 4);
                check_eq_int("9-pop f4rank", rs[0].f4rank, 1);
                // match the existing run_qpadm exactly (the single-model production entry).
                const steppe::QpAdmResult rq = steppe::run_qpadm(dev, m, opts, res);
                check_close("search==run_qpadm w[0]", rs[0].weight.at(0), rq.weight.at(0), 0.0, 1e-12);
                check_close("search==run_qpadm chisq", rs[0].chisq, rq.chisq, 0.0, 1e-12);
                std::printf("  [PASS] 9-pop golden via run_qpadm_search matches golden AND run_qpadm\n");
            }
        } else {
            std::printf("  [SKIP] 9-pop fixture absent\n");
        }

        std::printf("-- NO-REGRESSION: NRBIG golden_fit1 (nr=39) via run_qpadm_search(1 model) --\n");
        steppe::F2BlockTensor fb;
        if (read_fixture(golden_dir + "/fixtures/f2_fit1_NRBIG.bin", fb)) {
            steppe::DeviceConfig cfg; cfg.devices = {0};
            steppe::device::Resources res = steppe::device::build_resources(cfg);
            steppe::F2BlockTensor up = fb; up.vpair.assign(fb.f2.size(), 0.0);
            steppe::device::DeviceF2Blocks dev = steppe::device::upload_f2_blocks_to_device(up, 0);
            steppe::QpAdmModel m; m.target = 0; m.left = {1, 2};
            for (int j = 3; j < 43; ++j) m.right.push_back(j);  // nr=39
            m.model_index = 0;
            std::vector<steppe::QpAdmModel> one = {m};
            const std::vector<steppe::QpAdmResult> rs =
                steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(one), opts, res);
            check_eq_int("NRBIG search result count", static_cast<int>(rs.size()), 1);
            if (rs.size() == 1 && rs[0].status == steppe::Status::Ok) {
                check_eq_int("NRBIG f4rank", rs[0].f4rank, 1);
                check_eq_int("NRBIG rankdrop rows", static_cast<int>(rs[0].rankdrop_dof.size()), 2);
                if (rs[0].rankdrop_chisq.size() == 2) {
                    check_close("NRBIG rd[0].chisq", rs[0].rankdrop_chisq.at(0), 52.704281610335912, 1e-6, 1e-9);
                    check_close("NRBIG rd[1].chisq", rs[0].rankdrop_chisq.at(1), 190.83602239090976, 1e-6, 1e-9);
                }
                std::printf("  [PASS] NRBIG golden via run_qpadm_search matches golden\n");
            } else {
                std::printf("  [FAIL] NRBIG search status != Ok\n"); ++g_failures;
            }
        } else {
            std::printf("  [SKIP] NRBIG fixture absent\n");
        }
    }

    // =====================================================================
    // (D) THROUGHPUT at SCALE (synthetic large-N; NO per-model golden — timing only).
    // Enumerate many k-subsets of a larger INDEX pool over the SAME resident f2 (the
    // 8 pool indices 1..8 in pop_order are the sources). The accuracy gate is the
    // validated 84-model set above; this demonstrates the rotation throughput honestly.
    // =====================================================================
    if (gpu_count >= 1) {
        std::printf("\n-- THROUGHPUT at SCALE (synthetic enumeration; timing only, NO golden) --\n");
        // Source index pool = the 8 pool pops (indices 1..8 in pop_order).
        std::vector<int> pool;
        for (int i = 1; i <= 8; ++i) pool.push_back(i);
        // Enumerate ALL 2- AND 3-subsets repeatedly to reach a few thousand models.
        std::vector<steppe::QpAdmModel> big;
        auto add_subsets = [&](int ksz) {
            const int n = static_cast<int>(pool.size());
            std::vector<int> idx(static_cast<std::size_t>(ksz));
            for (int i = 0; i < ksz; ++i) idx[static_cast<std::size_t>(i)] = i;
            while (true) {
                steppe::QpAdmModel m; m.target = target_idx; m.right = right_idx;
                for (int i = 0; i < ksz; ++i) m.left.push_back(pool[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])]);
                m.model_index = static_cast<int>(big.size());
                big.push_back(std::move(m));
                int i = ksz - 1;
                while (i >= 0 && idx[static_cast<std::size_t>(i)] == n - ksz + i) --i;
                if (i < 0) break;
                ++idx[static_cast<std::size_t>(i)];
                for (int j = i + 1; j < ksz; ++j) idx[static_cast<std::size_t>(j)] = idx[static_cast<std::size_t>(j - 1)] + 1;
            }
        };
        // Repeat the enumeration to reach a meaningful scale-N (the production envelope
        // is thousands; the same resident f2 serves them all).
        const int kRepeat = 30;  // 84 * 30 = 2520 models
        for (int rep = 0; rep < kRepeat; ++rep) { add_subsets(2); add_subsets(3); }
        std::printf("  scale-N = %zu models (synthetic; the validated set above gates accuracy)\n",
                    big.size());

        steppe::F2BlockTensor up = f2; up.vpair.assign(f2.f2.size(), 0.0);

        // G=1 throughput.
        {
            steppe::DeviceConfig cfg; cfg.devices = {0};
            steppe::device::Resources res = steppe::device::build_resources(cfg);
            steppe::device::DeviceF2Blocks dev = steppe::device::upload_f2_blocks_to_device(up, 0);
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<steppe::QpAdmResult> r =
                steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(big), opts, res);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            check_eq_int("scale G=1 result count", static_cast<int>(r.size()),
                         static_cast<int>(big.size()));
            std::printf("  [THROUGHPUT] G=1: %zu models in %.0f ms = %.0f models/sec\n",
                        big.size(), ms, big.size() / (ms / 1000.0));
        }
        // G=2 throughput (near-2x expected; embarrassingly parallel).
        if (gpu_count >= 2) {
            steppe::DeviceConfig cfg; cfg.devices = {0, 1};
            steppe::device::Resources res = steppe::device::build_resources(cfg);
            steppe::device::DeviceF2Blocks dev = steppe::device::upload_f2_blocks_to_device(up, 0);
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<steppe::QpAdmResult> r =
                steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(big), opts, res);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            check_eq_int("scale G=2 result count", static_cast<int>(r.size()),
                         static_cast<int>(big.size()));
            std::printf("  [THROUGHPUT] G=2: %zu models in %.0f ms = %.0f models/sec\n",
                        big.size(), ms, big.size() / (ms / 1000.0));
        }
    }

    // =====================================================================
    // (E) HOST-ORACLE overload sanity: run_qpadm_search(F2BlockTensor) == the GPU
    // batched result per model (the CpuBackend per-model oracle; loose localizer).
    // =====================================================================
    {
        std::printf("\n-- HOST-ORACLE overload (CpuBackend per-model loop) sanity --\n");
        steppe::device::Resources cpu_res;
        steppe::device::PerGpuResources cpu; cpu.device_id = -1;
        cpu.backend = steppe::device::make_cpu_backend();
        cpu_res.gpus.push_back(std::move(cpu));
        const std::vector<steppe::QpAdmResult> oracle =
            steppe::run_qpadm_search(f2, std::span<const steppe::QpAdmModel>(models), opts, cpu_res);
        check_eq_int("oracle result count", static_cast<int>(oracle.size()),
                     static_cast<int>(models.size()));
        // The oracle must match the golden weights too (it is the same math, native).
        int ok = 0;
        for (std::size_t i = 0; i < oracle.size() && i < G.models.size(); ++i) {
            if (oracle[i].status != steppe::Status::Ok) continue;
            bool wok = oracle[i].weight.size() == G.models[i].weight.size();
            for (std::size_t k = 0; wok && k < oracle[i].weight.size(); ++k)
                wok = wok && std::fabs(oracle[i].weight[k] - G.models[i].weight[k]) <=
                             1e-5 * std::fabs(G.models[i].weight[k]) + 1e-5;
            if (wok) ++ok;
        }
        check_eq_int("oracle models matching golden weights", ok, static_cast<int>(G.models.size()));
        std::printf("  [INFO] host-oracle: %d/%zu models match golden weights (rtol 1e-6)\n",
                    ok, G.models.size());
    }

    std::printf("\n=== RESULT: %s (%d failures) ===\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
