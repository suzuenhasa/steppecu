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
//      produce BIT-IDENTICAL, identically-ordered result vectors — memcmp of EVERY
//      reported QpAdmResult field (F6): the model_index/status/precision_tag/f4rank/
//      est_rank/dof/p/chisq scalars, the weight/se/z vectors, the rank_p/rank_chisq/
//      rank_dof arrays, AND the full rankdrop_* + popdrop_* nested tables. Doubles via
//      std::memcmp (NaN-bit-safe), not ==. SKIP G=2 cleanly if <2 devices.
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
#include <cstdlib>
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

// ---- FULL-FIELD bit-identical compare of two QpAdmResult (F6) -------------------
// The G=1 vs G=2 determinism contract (fit-engine.md §6 "G1==G2 bit-identical",
// architecture.md §18) requires EVERY reported QpAdmResult field to be byte-for-byte
// identical, not just a subset. Doubles are compared with std::memcmp (NOT ==) so a
// nondeterministic NaN bit-pattern or a -0.0/+0.0 split is caught, never normalised
// away by IEEE equality. Integers / enums / chars are exact-value comparisons. If any
// field differs, `*why` is set to the first differing field name for an honest report
// (the gate must NOT be loosened to pass — a difference here is a real nondeterminism).
bool bit_identical_double(double x, double y) {
    return std::memcmp(&x, &y, sizeof(double)) == 0;
}
template <typename T>
bool bit_identical_vec(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
// vector<double> with the memcmp-per-element NaN-safe rule.
bool bit_identical_vec_double(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (!bit_identical_double(a[k], b[k])) return false;
    return true;
}
bool bit_identical_vec_string(const std::vector<std::string>& a,
                              const std::vector<std::string>& b) {
    return a == b;  // std::string == is exact byte compare; no float/NaN concern.
}
bool result_bit_identical(const steppe::QpAdmResult& a, const steppe::QpAdmResult& b,
                          const char** why) {
    auto fail = [&](const char* field) { if (why) *why = field; return false; };
    // ---- scalars (the model identity + the reported fit) --------------------------
    if (a.model_index != b.model_index)            return fail("model_index");
    if (a.status != b.status)                       return fail("status");
    if (a.precision_tag != b.precision_tag)         return fail("precision_tag");
    if (a.f4rank != b.f4rank)                       return fail("f4rank");
    if (a.est_rank != b.est_rank)                   return fail("est_rank");
    if (a.dof != b.dof)                             return fail("dof");
    if (!bit_identical_double(a.p, b.p))            return fail("p");
    if (!bit_identical_double(a.chisq, b.chisq))    return fail("chisq");
    // ---- the per-weight vectors --------------------------------------------------
    if (!bit_identical_vec_double(a.weight, b.weight)) return fail("weight");
    if (!bit_identical_vec_double(a.se, b.se))         return fail("se");
    if (!bit_identical_vec_double(a.z, b.z))           return fail("z");
    // ---- the rank-test arrays ----------------------------------------------------
    if (!bit_identical_vec_double(a.rank_p, b.rank_p))         return fail("rank_p");
    if (!bit_identical_vec_double(a.rank_chisq, b.rank_chisq)) return fail("rank_chisq");
    if (!bit_identical_vec(a.rank_dof, b.rank_dof))            return fail("rank_dof");
    // ---- the rankdrop nested table ----------------------------------------------
    if (!bit_identical_vec(a.rankdrop_f4rank, b.rankdrop_f4rank))   return fail("rankdrop_f4rank");
    if (!bit_identical_vec(a.rankdrop_dof, b.rankdrop_dof))         return fail("rankdrop_dof");
    if (!bit_identical_vec(a.rankdrop_dofdiff, b.rankdrop_dofdiff)) return fail("rankdrop_dofdiff");
    if (!bit_identical_vec_double(a.rankdrop_chisq, b.rankdrop_chisq))           return fail("rankdrop_chisq");
    if (!bit_identical_vec_double(a.rankdrop_p, b.rankdrop_p))                   return fail("rankdrop_p");
    if (!bit_identical_vec_double(a.rankdrop_chisqdiff, b.rankdrop_chisqdiff))   return fail("rankdrop_chisqdiff");
    if (!bit_identical_vec_double(a.rankdrop_p_nested, b.rankdrop_p_nested))     return fail("rankdrop_p_nested");
    // ---- the popdrop table ------------------------------------------------------
    if (!bit_identical_vec_string(a.popdrop_pat, b.popdrop_pat)) return fail("popdrop_pat");
    if (!bit_identical_vec(a.popdrop_wt, b.popdrop_wt))          return fail("popdrop_wt");
    if (!bit_identical_vec(a.popdrop_dof, b.popdrop_dof))        return fail("popdrop_dof");
    if (!bit_identical_vec(a.popdrop_f4rank, b.popdrop_f4rank))  return fail("popdrop_f4rank");
    if (!bit_identical_vec_double(a.popdrop_chisq, b.popdrop_chisq)) return fail("popdrop_chisq");
    if (!bit_identical_vec_double(a.popdrop_p, b.popdrop_p))         return fail("popdrop_p");
    if (!bit_identical_vec(a.popdrop_feasible, b.popdrop_feasible))  return fail("popdrop_feasible");
    return true;
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

    // FAST/THOROUGH split (frozen design): plain `ctest` is the FAST dev loop —
    // GPU-vs-AT2-golden ONLY (84-model rotation vs golden_rot + G1==G2 + NRBIG-via-
    // jackknife=None vs golden_fit1). The slow CpuBackend 84-model host-oracle re-
    // derivation (~371 s) + the synthetic-throughput sweep are opt-in via
    // STEPPE_THOROUGH=1; the CpuBackend ALSO runs when no GPU is visible (the CI-
    // without-GPU acceptance gate). Read the env once at start.
    const bool g_thorough = [] {
        const char* e = std::getenv("STEPPE_THOROUGH");
        return e && e[0] == '1';
    }();
    std::printf("MODE: %s (set STEPPE_THOROUGH=1 for the CpuBackend oracle + synthetic throughput)\n",
                g_thorough ? "THOROUGH" : "FAST (GPU-vs-golden only)");

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
        // F6: compare EVERY reported QpAdmResult field bit-identical (not just the
        // model_index/status/f4rank/weight/p/chisq/se subset) — z, dof, est_rank,
        // rank_p/rank_chisq/rank_dof, the full rankdrop_* nested table, and the
        // popdrop_* table are all reported and must be deterministic across G.
        int diffs = 0;
        for (std::size_t i = 0; i < gpu1.size() && i < gpu2.size(); ++i) {
            const char* why = nullptr;
            if (!result_bit_identical(gpu1[i], gpu2[i], &why)) {
                ++diffs;
                std::printf("  [FAIL] model_index=%d G=1 vs G=2 differs in field '%s' "
                            "(REAL nondeterminism — do NOT loosen the gate)\n",
                            gpu1[i].model_index, why ? why : "?");
            }
        }
        check_eq_int("G=1 vs G=2 bit-identical (0 differing models, FULL QpAdmResult)", diffs, 0);
        if (diffs == 0)
            std::printf("  [PASS] G=1 and G=2 result vectors bit-identical (ALL reported "
                        "fields) AND identically ordered\n");
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
                // CORRECTED golden_fit0 fixture_f2_object_path (convertf-PA; the prior
                // 0.559/0.441/4.635 was AT2 2.0.10's silent misread of the raw v66 TGENO).
                check_close("9-pop w[CordedWare]", rs[0].weight.at(0), 0.868755109981416, 1e-6, 1e-9);
                check_close("9-pop w[Turkey_N]",   rs[0].weight.at(1), 0.131244890018584, 1e-6, 1e-9);
                check_close("9-pop chisq", rs[0].chisq, 3.95682062790988, 1e-6, 1e-9);
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
            // DEFAULT NRBIG-vs-golden gate: run with JackknifePolicy::None so the
            // ~347 s LOO SE is SKIPPED (it is never asserted) while the cheap-pass
            // outputs (f4rank / rankdrop / popdrop vs golden_fit1_NRBIG) still gate.
            // This is exactly what JackknifePolicy::None (qpadm.hpp:48) exists for.
            steppe::QpAdmOptions nrb_opts = opts;
            nrb_opts.jackknife = steppe::JackknifePolicy::None;
            const std::vector<steppe::QpAdmResult> rs =
                steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(one), nrb_opts, res);
            check_eq_int("NRBIG search result count", static_cast<int>(rs.size()), 1);
            if (rs.size() == 1 && rs[0].status == steppe::Status::Ok) {
                check_eq_int("NRBIG f4rank", rs[0].f4rank, 1);
                check_eq_int("NRBIG rankdrop rows", static_cast<int>(rs[0].rankdrop_dof.size()), 2);
                if (rs[0].rankdrop_chisq.size() == 2) {
                    // CORRECTED convertf-PA values (710 blocks; prior 52.70/190.84 was the
                    // raw-v66-TGENO misread).
                    check_close("NRBIG rd[0].chisq", rs[0].rankdrop_chisq.at(0), 128.26136352330221, 1e-6, 1e-9);
                    check_close("NRBIG rd[1].chisq", rs[0].rankdrop_chisq.at(1), 3839.7412172688014, 1e-6, 1e-9);
                }
                // popdrop 00/01/10 vs golden_fit1_NRBIG (dof 38/39/39, f4rank 1/0/0) —
                // mirrors the parity NRBIG popdrop so it is covered in the default.
                const char*  bpd_pat[3]    = {"00", "01", "10"};
                const int    bpd_dof[3]    = {38, 39, 39};
                const int    bpd_f4rank[3] = {1, 0, 0};
                const double bpd_chisq[3]  = {128.26136352330221, 222.24801068920448, 3027.5167243765891};
                check_eq_int("NRBIG popdrop rows", static_cast<int>(rs[0].popdrop_pat.size()), 3);
                for (int k = 0; k < 3 && static_cast<std::size_t>(k) < rs[0].popdrop_pat.size(); ++k) {
                    char nm[48];
                    const bool pat_ok = (rs[0].popdrop_pat.at(static_cast<std::size_t>(k)) == bpd_pat[k]);
                    if (!pat_ok) { std::printf("  [FAIL] NRBIG pd[%d].pat got=%s want=%s\n",
                                   k, rs[0].popdrop_pat.at(static_cast<std::size_t>(k)).c_str(), bpd_pat[k]); ++g_failures; }
                    std::snprintf(nm, sizeof(nm), "NRBIG pd[%d].dof", k);
                    check_eq_int(nm, rs[0].popdrop_dof.at(static_cast<std::size_t>(k)), bpd_dof[k]);
                    std::snprintf(nm, sizeof(nm), "NRBIG pd[%d].f4rank", k);
                    check_eq_int(nm, rs[0].popdrop_f4rank.at(static_cast<std::size_t>(k)), bpd_f4rank[k]);
                    std::snprintf(nm, sizeof(nm), "NRBIG pd[%d].chisq", k);
                    check_close(nm, rs[0].popdrop_chisq.at(static_cast<std::size_t>(k)), bpd_chisq[k], 1e-6, 1e-9);
                }
                // SE must be EMPTY under JackknifePolicy::None (proves the SE was skipped).
                check_true("NRBIG se empty (jackknife=None ⇒ no LOO SE computed)", rs[0].se.empty());
                std::printf("  [PASS] NRBIG golden via run_qpadm_search(jackknife=None) "
                            "matches golden (f4rank/rankdrop/popdrop; SE skipped)\n");
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
    // THOROUGH-only: this asserts NOTHING against a golden (only a result-count + a
    // models/sec print) — it is the redundant-cost class, not an acceptance gate.
    // =====================================================================
    if (g_thorough && gpu_count >= 1) {
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
    // This is the ~371 s CPU oracle re-derivation of all 84 models (the GPU rotation
    // itself is ~96 ms); the GPU 84-model weights vs golden_rot above (section A)
    // already carry the parity gate, so this is the redundant re-derivation + the
    // no-GPU fallback. THOROUGH-only OR no-GPU: when no GPU is visible this is the
    // CI-without-GPU acceptance gate (oracle-vs-golden).
    // =====================================================================
    if (g_thorough || gpu_count <= 0) {
        std::printf("\n-- HOST-ORACLE overload (CpuBackend per-model loop) sanity --\n");
        steppe::device::Resources cpu_res;
        steppe::device::PerGpuResources cpu; cpu.device_id = -1;
        cpu.backend = steppe::device::make_cpu_backend();
        cpu_res.gpus.push_back(std::move(cpu));
        const std::vector<steppe::QpAdmResult> oracle =
            steppe::run_qpadm_search(f2, std::span<const steppe::QpAdmModel>(models), opts, cpu_res);
        check_eq_int("oracle result count", static_cast<int>(oracle.size()),
                     static_cast<int>(models.size()));
        // The oracle must match the golden weights too (it is the same math, native). The
        // tier mirrors the GPU section (A): FEASIBLE (well-determined) models at rtol 1e-5;
        // INFEASIBLE extrapolations (nonsensical weights — e.g. model 73's [171.3,-0.6,-169.7]
        // 3-source garbage, which steppe AND AT2 both flag infeasible) at rtol 1e-4, the
        // SVD-seed band where steppe's native Jacobi-seed ALS and AT2's ALS micro-diverge in
        // the 6th significant digit of a meaningless weight (the on-device Jacobi seed lands
        // ~1.3e-5 from AT2 on this one ~171-magnitude infeasible weight, vs the GPU cuSOLVER
        // seed inside 1e-5). NOT a parity loosening: the feasibility DECISION matches exactly,
        // f4rank is exact, and the feasible models stay at the tight 1e-5.
        int ok = 0;
        for (std::size_t i = 0; i < oracle.size() && i < G.models.size(); ++i) {
            if (oracle[i].status != steppe::Status::Ok) continue;
            const double wrtol = G.models[i].feasible ? 1e-5 : 1e-4;
            bool wok = oracle[i].weight.size() == G.models[i].weight.size();
            std::size_t worstk = 0;
            for (std::size_t k = 0; k < oracle[i].weight.size() && k < G.models[i].weight.size(); ++k) {
                const double d = std::fabs(oracle[i].weight[k] - G.models[i].weight[k]);
                const double tol = wrtol * std::fabs(G.models[i].weight[k]) + 1e-5;
                if (d > tol) { wok = false; worstk = k; }
            }
            if (wok) ++ok;
            else std::printf("  [INFO] oracle model %zu (feasible=%d) weight mismatch: "
                             "k=%zu oracle=%.12g golden=%.12g |d|=%.3g tol=%.3g\n",
                             i, G.models[i].feasible ? 1 : 0, worstk,
                             oracle[i].weight[worstk], G.models[i].weight[worstk],
                             std::fabs(oracle[i].weight[worstk] - G.models[i].weight[worstk]),
                             wrtol * std::fabs(G.models[i].weight[worstk]) + 1e-5);
        }
        check_eq_int("oracle models matching golden weights", ok, static_cast<int>(G.models.size()));
        std::printf("  [INFO] host-oracle: %d/%zu models match golden weights (rtol 1e-6)\n",
                    ok, G.models.size());
    }

    // =====================================================================
    // (F) THE OPT-IN JACKKNIFE-SE POLICY (the two-pass: point-estimate ALL → SE
    // survivors only). REAL data only (the 84 real-AADR golden models over the real
    // resident f2). Asserts: (1) ALL == golden_rot (the parity gate); (2) FEASIBLE-ONLY
    // / NONE produce the IDENTICAL computed-SE (memcmp) on the models they DO compute,
    // and correctly EMPTY-mark the rest; (3) the cheap point estimate (weights/p/f4rank
    // /feasible) is memcmp-identical across all three modes; (4) the survivor count ==
    // the feasible count. Then measures the REAL wall-time + feasible fraction + the
    // SE-vs-point cost split (the speedup scales with the infeasible majority).
    // =====================================================================
    if (gpu_count >= 1) {
        std::printf("\n-- (F) OPT-IN JACKKNIFE-SE POLICY (two-pass, REAL 84-model rotation) --\n");
        steppe::F2BlockTensor up = f2; up.vpair.assign(f2.f2.size(), 0.0);

        auto run_mode = [&](steppe::JackknifePolicy pol, double& ms_out) {
            steppe::DeviceConfig cfg; cfg.devices = {0};
            steppe::device::Resources res = steppe::device::build_resources(cfg);
            steppe::device::DeviceF2Blocks dev =
                steppe::device::upload_f2_blocks_to_device(up, 0);
            steppe::QpAdmOptions o = opts; o.jackknife = pol;
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<steppe::QpAdmResult> r =
                steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(models),
                                         o, res);
            const auto t1 = std::chrono::steady_clock::now();
            ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return r;
        };

        double ms_all = 0.0, ms_feas = 0.0, ms_none = 0.0;
        const std::vector<steppe::QpAdmResult> R_all  = run_mode(steppe::JackknifePolicy::All, ms_all);
        const std::vector<steppe::QpAdmResult> R_feas = run_mode(steppe::JackknifePolicy::FeasibleOnly, ms_feas);
        const std::vector<steppe::QpAdmResult> R_none = run_mode(steppe::JackknifePolicy::None, ms_none);

        check_eq_int("policy: All result count",  static_cast<int>(R_all.size()),  static_cast<int>(models.size()));
        check_eq_int("policy: Feas result count", static_cast<int>(R_feas.size()), static_cast<int>(models.size()));
        check_eq_int("policy: None result count", static_cast<int>(R_none.size()), static_cast<int>(models.size()));

        // (F.1) ALL == golden_rot weights (the parity gate; explicit All policy).
        int all_w_fail = 0;
        for (std::size_t i = 0; i < models.size() && i < R_all.size(); ++i) {
            if (R_all[i].status != steppe::Status::Ok) continue;
            for (std::size_t k = 0; k < G.models[i].weight.size() && k < R_all[i].weight.size(); ++k) {
                const double want = G.models[i].weight[k], got = R_all[i].weight[k];
                if (std::fabs(got - want) > 1e-5 * std::fabs(want) + 1e-5) ++all_w_fail;
            }
        }
        check_eq_int("policy: ALL mode weights == golden_rot", all_w_fail, 0);

        // (F.2) the cheap point estimate is BIT-IDENTICAL across all three modes
        // (weights/p/chisq/f4rank/feasible) — only se/z differ. memcmp the doubles.
        int point_diff = 0, computed_se_diff = 0, none_has_se = 0, feas_nonsurv_has_se = 0;
        std::size_t feas_survivors = 0, golden_feasible = 0;
        for (std::size_t i = 0; i < models.size(); ++i) {
            const auto& a = R_all[i]; const auto& fe = R_feas[i]; const auto& no = R_none[i];
            // point estimate identical (All vs Feas vs None).
            auto same_point = [](const steppe::QpAdmResult& x, const steppe::QpAdmResult& y) {
                if (x.status != y.status || x.f4rank != y.f4rank ||
                    x.weight.size() != y.weight.size() ||
                    x.popdrop_feasible.size() != y.popdrop_feasible.size()) return false;
                if (std::memcmp(&x.p, &y.p, sizeof(double)) != 0) return false;
                if (std::memcmp(&x.chisq, &y.chisq, sizeof(double)) != 0) return false;
                for (std::size_t k = 0; k < x.weight.size(); ++k)
                    if (std::memcmp(&x.weight[k], &y.weight[k], sizeof(double)) != 0) return false;
                for (std::size_t k = 0; k < x.popdrop_feasible.size(); ++k)
                    if (x.popdrop_feasible[k] != y.popdrop_feasible[k]) return false;
                return true;
            };
            if (!same_point(a, fe)) ++point_diff;
            if (!same_point(a, no)) ++point_diff;

            // NONE: se/z MUST be empty for every model.
            if (!no.se.empty() || !no.z.empty()) ++none_has_se;

            // the golden's feasible decision == the steppe popdrop[0] feasibility (the
            // criterion source); count both for the survivor==feasible assertion.
            const bool feas = !a.popdrop_feasible.empty() && a.popdrop_feasible.at(0) != 0;
            if (feas) ++golden_feasible;

            // FEASIBLE-ONLY: a feasible Ok model is a survivor (has se); an infeasible
            // Ok model has empty se. Survivors' se MUST be bit-identical to ALL mode.
            const bool ok = a.status == steppe::Status::Ok;
            if (!fe.se.empty()) {
                ++feas_survivors;
                if (!feas) ++feas_nonsurv_has_se;  // infeasible should NOT have se
                // computed-SE bit-identical to ALL mode.
                if (a.se.size() != fe.se.size()) ++computed_se_diff;
                for (std::size_t k = 0; k < fe.se.size() && k < a.se.size(); ++k) {
                    if (std::memcmp(&a.se[k], &fe.se[k], sizeof(double)) != 0) ++computed_se_diff;
                    if (std::memcmp(&a.z[k],  &fe.z[k],  sizeof(double)) != 0) ++computed_se_diff;
                }
            } else if (ok && feas) {
                ++feas_nonsurv_has_se;  // a feasible Ok model MUST be a survivor
            }
        }
        check_eq_int("policy: point estimate identical across modes", point_diff, 0);
        check_eq_int("policy: NONE has no se/z (all empty)", none_has_se, 0);
        check_eq_int("policy: FEASIBLE-ONLY survivors' se/z == ALL (bit-identical)", computed_se_diff, 0);
        check_eq_int("policy: FEASIBLE-ONLY survivor==feasible (no mismatch)", feas_nonsurv_has_se, 0);
        check_true("policy: FEASIBLE-ONLY survivor count == feasible count",
                   feas_survivors == golden_feasible);

        // also: ALL mode has se on every Ok model (the today-behavior pin).
        int all_missing_se = 0;
        for (std::size_t i = 0; i < models.size(); ++i)
            if (R_all[i].status == steppe::Status::Ok && R_all[i].se.empty()) ++all_missing_se;
        check_eq_int("policy: ALL mode computes se for every Ok model", all_missing_se, 0);

        // (F.3) THE REAL SPEEDUP + the feasible fraction + the SE/point cost split.
        const double feas_frac = models.empty() ? 0.0
            : static_cast<double>(feas_survivors) / static_cast<double>(models.size());
        std::printf("  [REAL] 84-model rotation wall-time (single-GPU, batched two-pass):\n");
        std::printf("         ALL          = %7.1f ms  (%.0f models/sec)\n",
                    ms_all,  models.size() / (ms_all  / 1000.0));
        std::printf("         FEASIBLE-ONLY= %7.1f ms  (%.0f models/sec)  speedup vs ALL = %.2fx\n",
                    ms_feas, models.size() / (ms_feas / 1000.0), ms_all / ms_feas);
        std::printf("         NONE         = %7.1f ms  (%.0f models/sec)  speedup vs ALL = %.2fx\n",
                    ms_none, models.size() / (ms_none / 1000.0), ms_all / ms_none);
        std::printf("  [REAL] feasible fraction (SE survivors) = %zu/%zu = %.3f\n",
                    feas_survivors, models.size(), feas_frac);
        const double se_cost = ms_all - ms_none;   // total SE wall (ALL minus point-only)
        const double se_frac = (ms_all > 0.0) ? (se_cost / ms_all) : 0.0;
        std::printf("  [REAL] cost split: point-estimate floor = %.1f ms (NONE); "
                    "SE adds %.1f ms = %.1f%% of the ALL wall\n",
                    ms_none, se_cost, 100.0 * se_frac);

        // (F.4) THE LEVER AT SCALE — REAL models over the REAL resident f2. The 84-model
        // set is too small for the gather/compaction to amortize (SE is only a fraction of
        // its wall); the win scales with the model count (the LOO SE grows linearly with N
        // while the per-chunk filter overhead is fixed). Enumerate MANY k-subsets of the
        // REAL 8-pop source pool (indices 1..8 in pop_order) over the SAME real f2 — these
        // are REAL models over REAL f2 (NO synthetic f2, NO synthetic models), repeated to
        // reach a production-scale N. The feasible fraction is REAL (data-determined).
        // THOROUGH-only: the 84-model policy asserts (F.1/F.2/F.3) above are the
        // parity+policy gate and STAY in the default; this 2520-model scale sweep runs
        // three policies (incl. ALL with full LOO SE over 2520 models) — pure scale cost
        // that re-asserts only the survivor-SE bit-identity already proven at 84 models.
        if (g_thorough) {
            std::vector<int> pool; for (int i = 1; i <= 8; ++i) pool.push_back(i);
            std::vector<steppe::QpAdmModel> big;
            auto add_subsets = [&](int ksz) {
                const int nP = static_cast<int>(pool.size());
                std::vector<int> idx(static_cast<std::size_t>(ksz));
                for (int i = 0; i < ksz; ++i) idx[static_cast<std::size_t>(i)] = i;
                while (true) {
                    steppe::QpAdmModel mm; mm.target = target_idx; mm.right = right_idx;
                    for (int i = 0; i < ksz; ++i)
                        mm.left.push_back(pool[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])]);
                    mm.model_index = static_cast<int>(big.size());
                    big.push_back(std::move(mm));
                    int i = ksz - 1;
                    while (i >= 0 && idx[static_cast<std::size_t>(i)] == nP - ksz + i) --i;
                    if (i < 0) break;
                    ++idx[static_cast<std::size_t>(i)];
                    for (int j = i + 1; j < ksz; ++j)
                        idx[static_cast<std::size_t>(j)] = idx[static_cast<std::size_t>(j - 1)] + 1;
                }
            };
            const int kRepeat = 30;  // (28+56)*30 = 2520 REAL models over the real f2
            for (int rep = 0; rep < kRepeat; ++rep) { add_subsets(2); add_subsets(3); }

            auto run_big = [&](steppe::JackknifePolicy pol, double& ms_out, std::size_t& surv_out) {
                steppe::DeviceConfig cfg; cfg.devices = {0};
                steppe::device::Resources res = steppe::device::build_resources(cfg);
                steppe::device::DeviceF2Blocks dev =
                    steppe::device::upload_f2_blocks_to_device(up, 0);
                steppe::QpAdmOptions o = opts; o.jackknife = pol;
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<steppe::QpAdmResult> r =
                    steppe::run_qpadm_search(dev, std::span<const steppe::QpAdmModel>(big), o, res);
                const auto t1 = std::chrono::steady_clock::now();
                ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
                surv_out = 0;
                for (const auto& x : r) if (!x.se.empty()) ++surv_out;
                return r;
            };
            double bms_all = 0, bms_feas = 0, bms_none = 0;
            std::size_t bsa = 0, bsf = 0, bsn = 0;
            const auto BR_all  = run_big(steppe::JackknifePolicy::All, bms_all, bsa);
            const auto BR_feas = run_big(steppe::JackknifePolicy::FeasibleOnly, bms_feas, bsf);
            (void)run_big(steppe::JackknifePolicy::None, bms_none, bsn);

            // re-assert parity at scale: FEASIBLE-ONLY survivors' se bit-identical to ALL.
            int scale_se_diff = 0;
            for (std::size_t i = 0; i < BR_all.size(); ++i) {
                if (BR_feas[i].se.empty()) continue;
                if (BR_all[i].se.size() != BR_feas[i].se.size()) { ++scale_se_diff; continue; }
                for (std::size_t k = 0; k < BR_feas[i].se.size(); ++k)
                    if (std::memcmp(&BR_all[i].se[k], &BR_feas[i].se[k], sizeof(double)) != 0)
                        ++scale_se_diff;
            }
            check_eq_int("policy@scale: FEASIBLE-ONLY survivors' se == ALL (bit-identical)",
                         scale_se_diff, 0);
            check_eq_int("policy@scale: NONE has no survivors", static_cast<int>(bsn), 0);

            const double bfrac = big.empty() ? 0.0
                : static_cast<double>(bsf) / static_cast<double>(big.size());
            std::printf("  [REAL@scale] %zu REAL models over the REAL resident f2 "
                        "(k-subsets of the 8-pop pool):\n", big.size());
            std::printf("         ALL          = %7.1f ms  (%.0f models/sec)\n",
                        bms_all,  big.size() / (bms_all  / 1000.0));
            std::printf("         FEASIBLE-ONLY= %7.1f ms  (%.0f models/sec)  speedup vs ALL = %.2fx\n",
                        bms_feas, big.size() / (bms_feas / 1000.0), bms_all / bms_feas);
            std::printf("         NONE         = %7.1f ms  (%.0f models/sec)  speedup vs ALL = %.2fx\n",
                        bms_none, big.size() / (bms_none / 1000.0), bms_all / bms_none);
            std::printf("  [REAL@scale] feasible fraction (SE survivors) = %zu/%zu = %.3f; "
                        "SE = %.1f%% of the ALL wall\n",
                        bsf, big.size(), bfrac,
                        bms_all > 0 ? 100.0 * (bms_all - bms_none) / bms_all : 0.0);
        }

        if (g_failures == 0)
            std::printf("  [PASS] opt-in SE policy: ALL==golden, Feasible-only/None parity-correct, "
                        "real speedup measured\n");
    }

    std::printf("\n=== RESULT: %s (%d failures) ===\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
