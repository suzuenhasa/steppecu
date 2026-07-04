// src/app/cmd_scan.cpp
//
// The `steppe scan` command — Phase 0 of the proxy/model scanner. It enumerates the
// candidate pool exactly like qpadm-rotate (no guided search yet), fits the whole model
// list in one batched, f2-resident engine call, then applies the scanner objective and
// emits a BEST-FIRST table — the thing plain rotate lacks.
//
// Objective (docs/planning/proxy-scanner-scope.md §1a):
//   HARD GATE  — status==Ok AND feasible (weights in [0,1]) AND tail p >= alpha (--p-min).
//                No multiplicity correction on the gate: success here is *non-rejection*,
//                so shrinking alpha would admit MORE models. Power, not alpha, is the lever.
//   RANK       — survivors first, then parsimony (fewest sources) -> weight stability
//                (higher min |z|) -> robustness (more leave-one-out-feasible). NEVER by p.
//   CENSUS     — report models-tested / feasible; the top survivor is SELECTED, not confirmed.
//
// The within-model rank-drop over-parameterization strike and the held-out / right-set
// machinery are later phases; Phase 0 surfaces est_rank so the user can see it. App-layer
// and CUDA-free — the GPU is reached only through the library's CUDA-free device seams.
//
// Reference template: src/app/cmd_rotate.cpp.
#include "app/cmd_scan.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_io.hpp"
#include "app/pop_resolver.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/device_f2_blocks.hpp"
#include "device/resources.hpp"
#include "steppe/error.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Pool-subset enumerator (Phase-0 scaffold: identical to qpadm-rotate's — Phase 1 replaces
// this exhaustive enumeration with the guided greedy/beam search).
[[nodiscard]] std::vector<QpAdmModel> enumerate_pool_subsets(
    int target_idx, const std::vector<int>& pool_idx, const std::vector<int>& right_idx,
    int lo, int hi) {
    std::vector<QpAdmModel> models;
    const int n = static_cast<int>(pool_idx.size());
    int counter = 0;
    for (int k = lo; k <= hi; ++k) {
        if (k < 1 || k > n) continue;
        std::vector<int> c(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) c[static_cast<std::size_t>(i)] = i;
        while (true) {
            QpAdmModel m;
            m.target = target_idx;
            m.left.reserve(static_cast<std::size_t>(k));
            for (int i = 0; i < k; ++i)
                m.left.push_back(pool_idx[static_cast<std::size_t>(c[static_cast<std::size_t>(i)])]);
            m.right = right_idx;
            m.model_index = counter++;
            models.push_back(std::move(m));

            int i = k - 1;
            while (i >= 0 && c[static_cast<std::size_t>(i)] == n - k + i) --i;
            if (i < 0) break;
            ++c[static_cast<std::size_t>(i)];
            for (int j = i + 1; j < k; ++j)
                c[static_cast<std::size_t>(j)] = c[static_cast<std::size_t>(j - 1)] + 1;
        }
    }
    return models;
}

// Feasibility, mirroring the rotation path: prefer the popdrop full-model flag, else all
// weights in [0,1] (matches result_emit's rotation_feasible / model_feasible).
[[nodiscard]] bool scan_feasible(const QpAdmResult& r) {
    if (!r.popdrop_feasible.empty()) return r.popdrop_feasible[0] != 0;
    if (r.weight.empty()) return false;
    for (double w : r.weight)
        if (!std::isfinite(w) || w < 0.0 || w > 1.0) return false;   // NaN weight -> infeasible
    return true;
}

// Status label (result_emit's status_str is file-local; keep a small local copy).
[[nodiscard]] const char* scan_status_label(Status s) {
    switch (s) {
        case Status::Ok:               return "ok";
        case Status::DeviceOom:        return "device_oom";
        case Status::RankDeficient:    return "rank_deficient";
        case Status::NonSpdCovariance: return "non_spd_covariance";
        case Status::ChisqUndefined:   return "chisq_undefined";
        case Status::InvalidConfig:    return "invalid_config";
        default:                       return "error";
    }
}

// Stability score: the smallest |z| across the model's INTERIOR weights (higher = weights
// better determined, none pinned near a bound). Weights pinned at a bound (<=0 or >=1) are
// skipped — their z is degenerate (a 1-source model forces weight=1, SE~0, z->inf, which is
// not "stable" but trivial). NaN when no weight is interior (e.g. a 1-source model), so the
// caller can rank it below any model with a genuine interior-weight stability signal.
[[nodiscard]] double min_abs_z(const QpAdmResult& r) {
    double m = std::numeric_limits<double>::infinity();
    bool any = false;
    const std::size_t n = std::min(r.weight.size(), r.z.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double w = r.weight[i];
        if (w <= 0.0 || w >= 1.0) continue;               // pinned at a bound: z not meaningful
        if (std::isfinite(r.z[i])) { m = std::min(m, std::fabs(r.z[i])); any = true; }
    }
    return any ? m : std::numeric_limits<double>::quiet_NaN();
}

// Robustness score: number of leave-one-source-out refits that stay feasible.
[[nodiscard]] int popdrop_feasible_count(const QpAdmResult& r) {
    int n = 0;
    for (char f : r.popdrop_feasible) n += (f != 0);
    return n;
}

// Over-parameterization strike (docs §1a / the Gap-2 correction): a feasible k-source model
// whose top mixture dimension is not justified by the WITHIN-model rank-drop test is
// over-specified — a source is redundant. Test the sweep entry at the fit rank (nl-1):
// rankdrop_p_nested >= rank_alpha means adding that last dimension did not significantly
// improve the fit.
//   IMPORTANT: this is a WITHIN-model rank test ("are the k sources distinct waves for THIS
//   set"), NOT a between-model "k vs k+1 sources" LRT. 1-source models (rank 0) are never
//   over-parameterized. When no rank-drop entry exists at the fit rank — the |right| < nsource
//   edge (§3.8) — the test is indeterminate and the model is NOT struck.
[[nodiscard]] bool over_parameterized(const QpAdmResult& r, double rank_alpha) {
    const int fit_rank = static_cast<int>(r.weight.size()) - 1;      // nl - 1
    if (fit_rank < 1) return false;                                  // 1-source: nothing to strike
    const std::size_t n = std::min(r.rankdrop_f4rank.size(), r.rankdrop_p_nested.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (r.rankdrop_f4rank[i] == fit_rank) {
            const double pn = r.rankdrop_p_nested[i];
            if (!std::isfinite(pn)) return false;                    // indeterminate
            return pn >= rank_alpha;                                 // top dim not justified
        }
    }
    return false;                                                    // no entry (§3.8): indeterminate
}

// One ranked output row.
struct ScanRow {
    std::string target;
    std::string left;      // sources joined "A+B+C"
    int         nsource = 0;
    double      p = 0.0;
    bool        feasible = false;
    bool        passes = false;    // survives the full hard gate
    bool        selected = false;  // the single best eligible survivor
    bool        over_param = false;  // top mixture dimension not justified (within-model)
    double      min_z = 0.0;
    int         popdrop_feas = 0;
    int         f4rank = 0;        // data-driven first-accepted rank (NOT the vacuous est_rank)
    const char* status = "error";
};

// Column semantics (kept distinct on purpose): `feasible` = weights in [0,1] only; `passes` =
// the full hard gate (status Ok AND feasible AND p>=alpha); the header census counts `passes`.
void emit_scan_csv(std::ostream& os, const std::vector<ScanRow>& rows,
                   std::size_t n_tested, std::size_t n_pass, char sep) {
    os << "# steppe scan: " << n_tested << " models tested, " << n_pass
       << " pass the gate; best is SELECTED (not confirmed)\n";
    os << "rank" << sep << "target" << sep << "left" << sep << "nsource" << sep << "p"
       << sep << "feasible" << sep << "passes" << sep << "selected" << sep << "over_param"
       << sep << "min_abs_z" << sep << "popdrop_feasible" << sep << "f4rank" << sep << "status"
       << '\n';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const ScanRow& r = rows[i];
        os << i << sep << csv_field(r.target, sep) << sep << csv_field(r.left, sep) << sep
           << r.nsource << sep << fmt_double(r.p) << sep << (r.feasible ? "TRUE" : "FALSE") << sep
           << (r.passes ? "TRUE" : "FALSE") << sep << (r.selected ? "TRUE" : "FALSE") << sep
           << (r.over_param ? "TRUE" : "FALSE") << sep << fmt_double(r.min_z) << sep
           << r.popdrop_feas << sep << r.f4rank << sep << r.status << '\n';
    }
}

void emit_scan_json(std::ostream& os, const std::vector<ScanRow>& rows,
                    std::size_t n_tested, std::size_t n_pass) {
    os << "{\n";
    os << "  \"n_tested\": " << n_tested << ",\n";
    os << "  \"n_pass\": " << n_pass << ",\n";
    os << "  \"selected_not_confirmed\": true,\n";
    os << "  \"models\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const ScanRow& r = rows[i];
        os << "    {\"rank\": " << i << ", \"target\": " << json_quote(r.target)
           << ", \"left\": " << json_quote(r.left) << ", \"nsource\": " << r.nsource
           << ", \"p\": " << json_double(r.p) << ", \"feasible\": " << (r.feasible ? "true" : "false")
           << ", \"passes\": " << (r.passes ? "true" : "false")
           << ", \"selected\": " << (r.selected ? "true" : "false")
           << ", \"over_param\": " << (r.over_param ? "true" : "false")
           << ", \"min_abs_z\": " << json_double(r.min_z)
           << ", \"popdrop_feasible\": " << r.popdrop_feas
           << ", \"f4rank\": " << r.f4rank << ", \"status\": " << json_quote(r.status)
           << "}" << (i + 1 < rows.size() ? "," : "") << '\n';
    }
    os << "  ]\n}\n";
}

// ---- Phase 1: guided greedy/beam search ---------------------------------------------

// Infeasibility margin: how far the WORST weight sits outside [0,1] (0 == feasible). This is
// the off-simplex direction the greedy/beam frontier descends — it is NOT tail p (a broken or
// empty fit is +inf, i.e. maximally infeasible, so it sinks in the frontier).
[[nodiscard]] double infeasibility_margin(const QpAdmResult& r) {
    if (r.status != Status::Ok || r.weight.empty())
        return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (double w : r.weight) {
        if (!std::isfinite(w)) return std::numeric_limits<double>::infinity();  // degenerate: sink it
        m = std::max(m, std::max(-w, w - 1.0));
    }
    return m > 0.0 ? m : 0.0;
}

// Nested rank-drop chi-square gain at the fit rank — the frontier tie-break (prefer growing the
// branch whose last mixture dimension is most justified). Not tail p; 0 when indeterminate.
[[nodiscard]] double chisq_gain(const QpAdmResult& r) {
    const int fit_rank = static_cast<int>(r.weight.size()) - 1;
    if (fit_rank < 1) return 0.0;
    const std::size_t n = std::min(r.rankdrop_f4rank.size(), r.rankdrop_chisqdiff.size());
    for (std::size_t i = 0; i < n; ++i)
        if (r.rankdrop_f4rank[i] == fit_rank)
            return std::isfinite(r.rankdrop_chisqdiff[i]) ? r.rankdrop_chisqdiff[i] : 0.0;
    return 0.0;
}

// Count in-band models sum_{k=max(lo,1)}^{min(hi,n)} C(n,k), saturating at cap+1. Each term is
// computed independently via the smaller side C(n,min(k,n-k)) so a large MIDDLE binomial never
// false-triggers the cap for a genuinely small high-lo band.
[[nodiscard]] long band_count(int n, int lo, int hi, long cap) {
    long total = 0;
    const int mlo = std::max(lo, 1);
    const int mhi = std::min(hi, n);
    for (int k = mlo; k <= mhi; ++k) {
        long c = 1;
        const int kk = std::min(k, n - k);
        for (int i = 0; i < kk; ++i) {
            if (c > cap) break;                  // saturate; keep the multiply below in range
            c = c * (n - i) / (i + 1);           // exact: running binomial is integer-divisible
        }
        total += (c > cap) ? cap + 1 : c;
        if (total > cap) return cap + 1;
    }
    return total;
}

// Pools whose in-band enumeration is at most this small run exhaustively regardless of
// --strategy: greedy's blind spots vanish for free when everything can just be fit.
constexpr long kScanExhaustiveCap = 4096;

// Hard ceiling for an EXPLICIT --strategy exhaustive enumeration: bounds memory and keeps the
// per-batch model_index (int) well in range. Lift with --sure. Reference: scope §3.6.
constexpr long kScanMaxModels = 2'000'000;

struct SearchOutcome {
    std::vector<QpAdmModel>  models;      // every model fit, in fit order
    std::vector<QpAdmResult> results;     // parallel to models
    bool        truncated = false;        // greedy/beam ended with no eligible feasible in-band model
    std::string strategy_used;            // "greedy" | "beam" | "exhaustive" (may be auto-selected)
    int         rounds = 0;               // search levels fit (0 for exhaustive)
};

// The guided search. Every round fits ONE batch against the SAME resident dev_f2 (no re-upload),
// exactly like qpadm-rotate. greedy/beam descend the infeasibility margin (never p) and
// early-stop at the smallest ELIGIBLE feasible model in the band; small pools auto-run
// exhaustively. Reference: docs/planning/proxy-scanner-scope.md §4 (Phase 1), Gap-4 surrogate.
[[nodiscard]] SearchOutcome guided_search(
    const device::DeviceF2Blocks& dev_f2, const QpAdmOptions& opts, device::Resources& resources,
    int target_idx, const std::vector<int>& pool_idx, const std::vector<int>& right_idx,
    int lo, int hi, const std::string& strategy, int beam_width,
    const std::vector<int>& base_idx, double alpha, bool allow_clade) {

    SearchOutcome out;
    const int pool_n = static_cast<int>(pool_idx.size());

    // Exhaustive: explicit, or auto when the in-band enumeration is small.
    const bool exhaustive = (strategy == "exhaustive")
        || (band_count(pool_n, lo, hi, kScanExhaustiveCap) <= kScanExhaustiveCap);
    if (exhaustive) {
        out.models = enumerate_pool_subsets(target_idx, pool_idx, right_idx, lo, hi);
        out.results =
            run_qpadm_search(dev_f2, std::span<const QpAdmModel>(out.models), opts, resources);
        out.strategy_used = "exhaustive";
        return out;
    }

    out.strategy_used = strategy;
    const int width = (strategy == "greedy") ? 1 : beam_width;
    const int eff_min = allow_clade ? lo : std::max(lo, 2);   // smallest ELIGIBLE winning size

    std::set<std::vector<int>> seen;                          // dedup by sorted source-set
    std::vector<std::vector<int>> level;                      // source-sets to fit this round
    if (!base_idx.empty()) {
        std::vector<int> b = base_idx;
        std::sort(b.begin(), b.end());
        level.push_back(std::move(b));
    } else {
        for (int s : pool_idx) level.push_back(std::vector<int>{s});   // seed: every 1-source model
    }

    bool found_eligible = false;
    while (!level.empty() && !found_eligible) {
        std::vector<QpAdmModel> batch;                        // this round's fits (seen-deduped)
        for (const std::vector<int>& ss : level) {
            if (!seen.insert(ss).second) continue;
            QpAdmModel m;
            m.target = target_idx;
            m.left = ss;
            m.right = right_idx;
            m.model_index = static_cast<int>(batch.size());   // batch-local: engine needs 0..n-1
            batch.push_back(std::move(m));
        }
        if (batch.empty()) break;
        ++out.rounds;
        std::vector<QpAdmResult> res =
            run_qpadm_search(dev_f2, std::span<const QpAdmModel>(batch), opts, resources);

        struct Parent { std::vector<int> ss; double margin; double gain; };
        std::vector<Parent> parents;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            const QpAdmResult& r = res[i];
            const int ns = static_cast<int>(batch[i].left.size());
            if (r.status == Status::Ok && scan_feasible(r) && r.p >= alpha
                && ns >= eff_min && ns <= hi) {
                found_eligible = true;
            }
            if (ns < hi)
                parents.push_back({batch[i].left, infeasibility_margin(r), chisq_gain(r)});
            // Only in-band models enter the output (matching the exhaustive path); sub-`lo`
            // seeds/intermediates are frontier scaffolding only — never emitted or crowned.
            if (ns >= lo && ns <= hi) {
                out.models.push_back(std::move(batch[i]));
                out.results.push_back(std::move(res[i]));
            }
        }
        if (found_eligible) break;

        // Prune the frontier: keep the top-`width` parents by smallest margin (tie: larger gain).
        std::sort(parents.begin(), parents.end(), [](const Parent& a, const Parent& b) {
            if (a.margin != b.margin) return a.margin < b.margin;
            return a.gain > b.gain;
        });
        if (static_cast<int>(parents.size()) > width)
            parents.resize(static_cast<std::size_t>(width));

        std::vector<std::vector<int>> next;                   // expand each kept parent by one source
        for (const Parent& p : parents) {
            for (int s : pool_idx) {
                if (std::find(p.ss.begin(), p.ss.end(), s) != p.ss.end()) continue;
                std::vector<int> child = p.ss;
                child.push_back(s);
                std::sort(child.begin(), child.end());
                if (seen.find(child) == seen.end()) next.push_back(std::move(child));
            }
        }
        level = std::move(next);
    }
    out.truncated = !found_eligible;   // greedy/beam pruned without reaching an eligible feasible model
    return out;
}

}  // namespace

// The scan command pipeline — Phase 0.
int run_scan_command(const cfg::RunConfig& config) {
    if (config.f2_dir().empty()) {
        std::fprintf(stderr, "steppe scan: --f2-dir is required\n");
        return cfg::kExitInvalidConfig;
    }
    const F2DirResult dir = read_f2_dir(config.f2_dir());
    if (!dir.ok) {
        std::fprintf(stderr, "steppe scan: %s\n", dir.error.c_str());
        return cfg::kExitIoError;
    }
    const PopResolver resolver(dir.dir.pop_labels);
    if (!resolver.valid()) {
        std::fprintf(stderr, "steppe scan: %s\n", resolver.error().c_str());
        return cfg::kExitIoError;
    }

    if (config.target().empty()) {
        std::fprintf(stderr, "steppe scan: --target is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.right().empty()) {
        std::fprintf(stderr, "steppe scan: --right needs at least one outgroup population\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.pool().empty()) {
        std::fprintf(stderr, "steppe scan: --pool needs at least one source population\n");
        return cfg::kExitInvalidConfig;
    }

    const ResolveResult t = resolver.resolve(config.target());
    if (!t.ok) { std::fprintf(stderr, "steppe scan: %s\n", t.error.c_str()); return cfg::kExitInvalidConfig; }
    const ResolveListResult rt = resolver.resolve_all(config.right());
    if (!rt.ok) { std::fprintf(stderr, "steppe scan: %s\n", rt.error.c_str()); return cfg::kExitInvalidConfig; }
    const ResolveListResult pl = resolver.resolve_all(config.pool());
    if (!pl.ok) { std::fprintf(stderr, "steppe scan: %s\n", pl.error.c_str()); return cfg::kExitInvalidConfig; }

    const int target_idx = t.index;
    const std::vector<int>& right_idx = rt.indices;
    const std::vector<int>& pool_idx = pl.indices;

    const int pool_n = static_cast<int>(pool_idx.size());
    const int lo = config.min_sources();
    int hi = (config.max_sources() == -1) ? pool_n : config.max_sources();
    if (hi > pool_n) hi = pool_n;
    if (lo > pool_n) {
        std::fprintf(stderr,
                     "steppe scan: --min-sources (%d) exceeds the pool size (%d) — "
                     "no models to fit\n", lo, pool_n);
        return cfg::kExitInvalidConfig;
    }

    // Cost guard (scope §3.6): an explicit exhaustive enumeration can be astronomically large;
    // refuse a runaway without --sure (mirrors the f-stat sweep's cap + --sure gate).
    if (config.scan_strategy() == "exhaustive"
        && band_count(pool_n, lo, hi, kScanMaxModels) > kScanMaxModels && !config.sweep_sure()) {
        std::fprintf(stderr,
                     "steppe scan: --strategy exhaustive over this pool enumerates > %ld models "
                     "(C(pool, %d..%d)); pass --sure to proceed, or use --strategy beam/greedy or a "
                     "smaller --max-sources.\n", kScanMaxModels, lo, hi);
        return cfg::kExitInvalidConfig;
    }

    // Optional seed model for the guided search (--base): resolve its labels to indices.
    std::vector<int> base_idx;
    if (!config.scan_base().empty()) {
        const ResolveListResult b = resolver.resolve_all(config.scan_base());
        if (!b.ok) {
            std::fprintf(stderr, "steppe scan: %s\n", b.error.c_str());
            return cfg::kExitInvalidConfig;
        }
        base_idx = b.indices;
    }

    if (config.device().devices.size() >= 2) {
        std::fprintf(stderr,
                     "steppe scan: WARNING: %zu devices requested; scan runs single-GPU "
                     "(multi-GPU parked). Use --device 0.\n",
                     config.device().devices.size());
    }

    const double alpha = config.scan_p_min();
    const double rank_alpha = config.qpadm_options().rank_alpha;
    const bool allow_clade = config.scan_allow_clade();
    const std::string target_label = resolver.label_at(target_idx);

    const QpAdmOptions opts = config.qpadm_options();
    SearchOutcome search;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe scan: no CUDA device available (steppe is a GPU product; "
                         "a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        const int device_id = resources.gpus.front().device_id;
        device::DeviceF2Blocks dev_f2 =
            device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
        search = guided_search(dev_f2, opts, resources, target_idx, pool_idx, right_idx,
                               lo, hi, config.scan_strategy(), config.scan_beam_width(),
                               base_idx, alpha, allow_clade);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe scan: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    const std::vector<QpAdmModel>& models = search.models;
    const std::vector<QpAdmResult>& results = search.results;
    if (results.empty()) {
        std::fprintf(stderr, "steppe scan: the search fit zero models "
                             "(check --min/--max-sources, --pool, --base)\n");
        return cfg::kExitInvalidConfig;
    }

    // Apply the objective: gate every fit, then rank survivors-first by parsimony ->
    // over-param strike -> stability -> robustness, over the models the search fit.
    std::vector<std::size_t> order(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) order[i] = i;

    const auto passes_gate = [&](const QpAdmResult& r) {
        return r.status == Status::Ok && scan_feasible(r) && r.p >= alpha;
    };

    std::size_t n_pass = 0;   // full-gate survivors (status Ok + feasible + p >= alpha)
    for (const QpAdmResult& r : results) n_pass += passes_gate(r) ? 1 : 0;

    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const QpAdmResult& ra = results[a];
        const QpAdmResult& rb = results[b];
        const bool pa = passes_gate(ra), pb = passes_gate(rb);
        if (pa != pb) return pa;                                  // survivors first
        const int na = static_cast<int>(ra.weight.size());
        const int nb = static_cast<int>(rb.weight.size());
        if (na != nb) return na < nb;                            // parsimony: fewer sources
        const bool oa = over_parameterized(ra, rank_alpha);      // strike over-specified models
        const bool ob = over_parameterized(rb, rank_alpha);
        if (oa != ob) return !oa;                                // right-sized before over-param
        const double za = min_abs_z(ra), zb = min_abs_z(rb);     // stability: higher min|z|
        const bool fa = std::isfinite(za), fb = std::isfinite(zb);
        if (fa != fb) return fa;                                 // a real stability signal beats NA
        if (fa && fb && za != zb) return za > zb;
        const int da = popdrop_feasible_count(ra), db = popdrop_feasible_count(rb);
        if (da != db) return da > db;                            // robustness: more LOO-feasible
        return a < b;   // global fit-order: unique + deterministic (model_index is batch-local)
    });

    // Crown the best-ranked survivor eligible under --allow-clade: with --no-allow-clade a
    // 1-source (clade) model still appears in the table but is never the winner (the crown
    // goes to the best genuine >=2-source mixture, if any).
    std::size_t selected_rank = order.size();   // sentinel: nothing selected
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        const QpAdmResult& r = results[order[rank]];
        if (!passes_gate(r)) break;             // survivors sort first; stop at first non-survivor
        if (allow_clade || r.weight.size() >= 2) { selected_rank = rank; break; }
    }

    std::vector<ScanRow> rows;
    rows.reserve(order.size());
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        const std::size_t mi = order[rank];      // index into both results[] and models[]
        const QpAdmResult& r = results[mi];
        ScanRow row;
        row.target = target_label;
        std::string joined;
        for (std::size_t k = 0; k < models[mi].left.size(); ++k) {
            if (k) joined += "+";
            joined += resolver.label_at(models[mi].left[static_cast<std::size_t>(k)]);
        }
        row.left = joined;
        row.nsource = static_cast<int>(r.weight.size());
        row.p = r.p;
        row.feasible = scan_feasible(r);
        row.passes = passes_gate(r);
        row.selected = (rank == selected_rank);
        row.over_param = over_parameterized(r, rank_alpha);
        row.min_z = min_abs_z(r);
        row.popdrop_feas = popdrop_feasible_count(r);
        row.f4rank = r.f4rank;
        row.status = scan_status_label(r.status);
        rows.push_back(std::move(row));
    }

    if (selected_rank < rows.size()) {
        std::fprintf(stderr,
                     "steppe scan: %zu models tested, %zu pass the gate (status ok, weights "
                     "in [0,1], p >= %.4g). Best: %s = %s (SELECTED, not confirmed).\n",
                     results.size(), n_pass, alpha, target_label.c_str(),
                     rows[selected_rank].left.c_str());
    } else {
        std::fprintf(stderr,
                     "steppe scan: %zu models tested, %zu pass the gate (status ok, weights "
                     "in [0,1], p >= %.4g). No eligible model selected%s.\n",
                     results.size(), n_pass, alpha,
                     allow_clade ? "" : " (--no-allow-clade: no feasible >=2-source mixture)");
    }

    if (search.truncated) {
        std::fprintf(stderr,
                     "steppe scan: NOTE: --strategy %s explored %d level(s) and may have missed a "
                     "feasible model reachable only by a non-greedy step — rerun --strategy "
                     "exhaustive to rule that out.\n",
                     search.strategy_used.c_str(), search.rounds);
    }

    if (const auto rc = emit_to_destination(
            config, "scan", [&](std::ostream& os, OutputFormat fmt) {
                switch (fmt) {
                    case OutputFormat::Csv: emit_scan_csv(os, rows, results.size(), n_pass, ','); break;
                    case OutputFormat::Tsv: emit_scan_csv(os, rows, results.size(), n_pass, '\t'); break;
                    case OutputFormat::Json: emit_scan_json(os, rows, results.size(), n_pass); break;
                }
            })) {
        return *rc;
    }

    return cfg::kExitOk;
}

}  // namespace steppe::app
