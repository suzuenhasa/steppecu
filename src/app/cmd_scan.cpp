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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <ostream>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
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
#include "steppe/f3.hpp"
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
    // --suggest-swaps (populated only for infeasible models that have a suggestion)
    bool        swap_has = false;
    std::string swap_drop, swap_add;
    bool        swap_feasible = false;
    double      swap_p = std::numeric_limits<double>::quiet_NaN();
};

// Column semantics (kept distinct on purpose): `feasible` = weights in [0,1] only; `passes` =
// the full hard gate (status Ok AND feasible AND p>=alpha); the header census counts `passes`.
void emit_scan_csv(std::ostream& os, const std::vector<ScanRow>& rows,
                   std::size_t n_tested, std::size_t n_pass, bool show_swaps, char sep) {
    os << "# steppe scan: " << n_tested << " models tested, " << n_pass
       << " pass the gate; best is SELECTED (not confirmed)\n";
    os << "rank" << sep << "target" << sep << "left" << sep << "nsource" << sep << "p"
       << sep << "feasible" << sep << "passes" << sep << "selected" << sep << "over_param"
       << sep << "min_abs_z" << sep << "popdrop_feasible" << sep << "f4rank" << sep << "status";
    if (show_swaps)
        os << sep << "swap_drop" << sep << "swap_add" << sep << "swap_feasible" << sep << "swap_p";
    os << '\n';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const ScanRow& r = rows[i];
        os << i << sep << csv_field(r.target, sep) << sep << csv_field(r.left, sep) << sep
           << r.nsource << sep << fmt_double(r.p) << sep << (r.feasible ? "TRUE" : "FALSE") << sep
           << (r.passes ? "TRUE" : "FALSE") << sep << (r.selected ? "TRUE" : "FALSE") << sep
           << (r.over_param ? "TRUE" : "FALSE") << sep << fmt_double(r.min_z) << sep
           << r.popdrop_feas << sep << r.f4rank << sep << r.status;
        if (show_swaps) {
            os << sep << csv_field(r.swap_drop, sep) << sep << csv_field(r.swap_add, sep) << sep
               << (r.swap_has ? (r.swap_feasible ? "TRUE" : "FALSE") : "") << sep
               << (r.swap_has ? fmt_double(r.swap_p) : std::string());
        }
        os << '\n';
    }
}

void emit_scan_json(std::ostream& os, const std::vector<ScanRow>& rows,
                    std::size_t n_tested, std::size_t n_pass, bool show_swaps) {
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
           << ", \"f4rank\": " << r.f4rank << ", \"status\": " << json_quote(r.status);
        if (show_swaps) {
            os << ", \"swap\": ";
            if (r.swap_has)
                os << "{\"drop\": " << json_quote(r.swap_drop) << ", \"add\": "
                   << json_quote(r.swap_add) << ", \"feasible\": "
                   << (r.swap_feasible ? "true" : "false") << ", \"p\": " << json_double(r.swap_p)
                   << "}";
            else
                os << "null";
        }
        os << "}" << (i + 1 < rows.size() ? "," : "") << '\n';
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

// ---- Phase 2: relatedness pre-rank + swap suggestions -------------------------------

// Per-pool-source relatedness to the target, aligned to pool_idx.
struct Relatedness { std::vector<double> est, se, z; };

// Outgroup-f3 relatedness of each pool source X to the target, AVERAGED over the whole right set:
// mean_k f3(R_k; target, X). Higher = more shared drift with the target = a better proxy candidate.
// Averaging over ALL outgroups (not just right[0]) makes the ranking independent of --right ordering
// and consistent with the fits, which use the full right set. (se is the mean per-outgroup se — a
// directional noise indicator, not a rigorous SE of the mean.)
[[nodiscard]] Relatedness relatedness_f3(
    const device::DeviceF2Blocks& dev_f2, const QpAdmOptions& opts, device::Resources& resources,
    int target_idx, const std::vector<int>& pool_idx, const std::vector<int>& right_idx) {
    const std::size_t np = pool_idx.size();
    const std::size_t nr = right_idx.size();
    std::vector<std::array<int, 3>> triples;
    triples.reserve(np * nr);
    for (std::size_t j = 0; j < np; ++j)              // triple t = j*nr + k  ->  f3(R_k; target, X_j)
        for (int rk : right_idx) triples.push_back({rk, target_idx, pool_idx[j]});
    const F3Result f3 = run_f3(dev_f2, std::span<const std::array<int, 3>>(triples), opts, resources);

    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    Relatedness rel;
    rel.est.assign(np, kNaN);
    rel.se.assign(np, kNaN);
    rel.z.assign(np, kNaN);
    for (std::size_t j = 0; j < np; ++j) {
        double sum_est = 0.0, sum_se = 0.0;
        int n_est = 0, n_se = 0;
        for (std::size_t k = 0; k < nr; ++k) {
            const std::size_t t = j * nr + k;
            if (t < f3.est.size() && std::isfinite(f3.est[t])) { sum_est += f3.est[t]; ++n_est; }
            if (t < f3.se.size() && std::isfinite(f3.se[t]))   { sum_se += f3.se[t];   ++n_se; }
        }
        if (n_est) rel.est[j] = sum_est / n_est;
        if (n_se)  rel.se[j] = sum_se / n_se;
        if (std::isfinite(rel.est[j]) && std::isfinite(rel.se[j]) && rel.se[j] > 0.0)
            rel.z[j] = rel.est[j] / rel.se[j];
    }
    return rel;
}

// Pool positions sorted by DESCENDING relatedness (NaN sinks to the end).
[[nodiscard]] std::vector<std::size_t> relatedness_order(const Relatedness& rel) {
    std::vector<std::size_t> ord(rel.est.size());
    for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t b) {
        const double ea = rel.est[a], eb = rel.est[b];
        const bool fa = std::isfinite(ea), fb = std::isfinite(eb);
        if (fa != fb) return fa;                 // finite relatedness before NaN
        if (fa && fb && ea != eb) return ea > eb;
        return a < b;
    });
    return ord;
}

// Swap culprit: the position (in `left`) of the LEAST-related source to the target — the likeliest
// one to drop. Uses the per-source outgroup-f3 relatedness; a source absent from the map (e.g. an
// off-pool --base source) is treated as maximally suspect. It is a heuristic — the swap refit is
// what verifies whether the suggestion actually helps. Only meaningful for nl >= 2.
[[nodiscard]] int swap_culprit(const std::vector<int>& left,
                               const std::unordered_map<int, double>& est_by_source) {
    if (left.size() < 2) return -1;
    int pos = -1;
    double worst = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < left.size(); ++i) {
        const auto it = est_by_source.find(left[i]);
        const double e = (it != est_by_source.end() && std::isfinite(it->second))
                             ? it->second : -std::numeric_limits<double>::infinity();
        if (e < worst) { worst = e; pos = static_cast<int>(i); }
    }
    return pos;
}

// Highest-relatedness pool source not already in `left` (the swap replacement). -1 if none.
[[nodiscard]] int swap_replacement(const std::vector<std::size_t>& rel_order,
                                   const std::vector<int>& pool_idx,
                                   const std::vector<int>& left) {
    for (std::size_t pos : rel_order) {
        const int cand = pool_idx[pos];
        if (std::find(left.begin(), left.end(), cand) == left.end()) return cand;
    }
    return -1;
}

// Per-output-model swap suggestion (drop the culprit, add a related source, refit).
struct SwapInfo {
    bool        has = false;
    std::string drop, add;
    bool        feasible = false;
    double      p = std::numeric_limits<double>::quiet_NaN();
};

// For every model that FAILS the gate, propose "drop the least-related source / add the most-related
// unused source", then refit all the swapped models in ONE batched call against the resident dev_f2.
// Aligned to `models`.
[[nodiscard]] std::vector<SwapInfo> compute_swaps(
    const device::DeviceF2Blocks& dev_f2, const QpAdmOptions& opts, device::Resources& resources,
    const std::vector<QpAdmModel>& models, const std::vector<QpAdmResult>& results,
    const PopResolver& resolver, const Relatedness& rel, const std::vector<std::size_t>& rel_order,
    const std::vector<int>& pool_idx, const std::vector<int>& right_idx, double alpha) {

    std::unordered_map<int, double> est_by_source;   // pool source index -> outgroup-f3 relatedness
    for (std::size_t j = 0; j < pool_idx.size() && j < rel.est.size(); ++j)
        est_by_source[pool_idx[j]] = rel.est[j];

    std::vector<SwapInfo> swaps(models.size());
    std::vector<QpAdmModel> batch;
    std::vector<std::size_t> batch_to_model;
    for (std::size_t i = 0; i < models.size(); ++i) {
        const QpAdmResult& r = results[i];
        // Suggest a swap for any model that FAILS the gate (weight-infeasible OR p-rejected);
        // skip the ones that already pass.
        if (r.status == Status::Ok && scan_feasible(r) && r.p >= alpha) continue;
        const int culprit_pos = swap_culprit(models[i].left, est_by_source);
        if (culprit_pos < 0) continue;
        const int repl_idx = swap_replacement(rel_order, pool_idx, models[i].left);
        if (repl_idx < 0) continue;
        swaps[i].has = true;
        swaps[i].drop = resolver.label_at(models[i].left[static_cast<std::size_t>(culprit_pos)]);
        swaps[i].add = resolver.label_at(repl_idx);
        QpAdmModel sm;
        sm.target = models[i].target;
        sm.left = models[i].left;
        sm.left[static_cast<std::size_t>(culprit_pos)] = repl_idx;       // drop culprit, add replacement
        std::sort(sm.left.begin(), sm.left.end());
        sm.right = right_idx;
        sm.model_index = static_cast<int>(batch.size());                 // batch-local (0..n-1)
        batch_to_model.push_back(i);
        batch.push_back(std::move(sm));
    }
    if (!batch.empty()) {
        const std::vector<QpAdmResult> sr =
            run_qpadm_search(dev_f2, std::span<const QpAdmModel>(batch), opts, resources);
        for (std::size_t k = 0; k < sr.size() && k < batch_to_model.size(); ++k) {
            SwapInfo& s = swaps[batch_to_model[k]];
            s.feasible = sr[k].status == Status::Ok && scan_feasible(sr[k]);
            s.p = sr[k].p;
        }
    }
    return swaps;
}

// The --prerank shortlist emitter (per-source, best-first).
struct RelatedRow { std::string source; double est, se, z; };

void emit_prerank(std::ostream& os, OutputFormat fmt, const std::string& target_label,
                  const std::vector<RelatedRow>& rows) {
    if (fmt == OutputFormat::Json) {
        os << "{\n  \"target\": " << json_quote(target_label) << ",\n  \"related\": [\n";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const RelatedRow& r = rows[i];
            os << "    {\"rank\": " << i << ", \"source\": " << json_quote(r.source)
               << ", \"f3\": " << json_double(r.est) << ", \"se\": " << json_double(r.se)
               << ", \"z\": " << json_double(r.z) << "}" << (i + 1 < rows.size() ? "," : "") << '\n';
        }
        os << "  ]\n}\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "# steppe scan --prerank: pool by mean outgroup-f3 relatedness to " << target_label
       << " (mean_k f3(right_k; target, X) over the full right set; higher = more related)\n";
    os << "rank" << sep << "source" << sep << "f3" << sep << "se" << sep << "z" << '\n';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const RelatedRow& r = rows[i];
        os << i << sep << csv_field(r.source, sep) << sep << fmt_double(r.est) << sep
           << fmt_double(r.se) << sep << fmt_double(r.z) << '\n';
    }
}

// ---- Phase 3: right-set (outgroup) admissibility + optimization ---------------------

// Sources-only qpWave admissibility — the §1a anti-circularity gate. A right set R distinguishes
// the `sources` iff qpWave(sources, R) REJECTS every tested lower rank: the sources then span full
// rank (nsource-1), i.e. R has the power to tell them apart. Target-fit-BLIND by construction (the
// target never enters the block, so the fit p cannot leak in). Reads the explicit rank_p vector,
// NEVER est_rank (which is a thresholded first-accepted rank and would re-import the fit). Needs
// |R| >= nsource outgroups. `accepted_rank` >= 0 names the first non-rejected rank (why it failed).
struct Admissibility {
    bool   ok = false;
    int    accepted_rank = -1;       // first rank NOT rejected (>=0 => inadmissible); -1 if admissible
    bool   enough_outgroups = true;  // |R| >= nsource
    Status status = Status::Ok;
};

[[nodiscard]] Admissibility qpwave_admissible(
    const device::DeviceF2Blocks& dev_f2, const QpAdmOptions& opts, device::Resources& resources,
    const std::vector<int>& sources, const std::vector<int>& right, double rank_alpha) {
    Admissibility a;
    if (right.size() < sources.size()) { a.enough_outgroups = false; return a; }
    const QpWaveResult w = run_qpwave(dev_f2, std::span<const int>(sources),
                                      std::span<const int>(right), opts, resources);
    a.status = w.status;
    if (w.status != Status::Ok || w.rank_p.empty()) return a;   // fail-closed on non-Ok / empty
    for (std::size_t r = 0; r < w.rank_p.size(); ++r) {
        if (!(w.rank_p[r] < rank_alpha)) {   // this lower rank is ACCEPTED -> sources not fully distinct
            a.accepted_rank = static_cast<int>(r);
            return a;
        }
    }
    a.ok = true;                             // every tested rank rejected -> sources span full rank
    return a;
}

// Minimal sufficient right set by ADD-DROP: R0 pinned; test the whole {R0} ∪ curated-pool universe,
// then greedily drop non-R0 outgroups while the set stays admissible AND keeps at least `min_size`
// outgroups. Returns the minimal sufficient right set (R0 first), or empty if the universe is not
// admissible or has fewer than `min_size` outgroups. `universe_out` reports WHY it is empty (status /
// enough_outgroups / accepted_rank) so the caller can give an accurate diagnostic. "Sufficient" =
// distinguishes the sources (sources-only qpWave) AND has enough outgroups to power a non-degenerate
// fit (min_size = nsource+1, so nr >= nsource, dof > 0). fit-p / feasibility NEVER enter this —
// admissibility + a size floor only (§1a Gap-3 contract: outgroups are never chosen to make the
// model pass; the floor is a dof requirement, not a p threshold).
//   NOTE: "a superset is at least as distinguishing" is exact for the deterministic matrix rank but a
//   HEURISTIC for the finite-sample chi-square rank test (a noisy outgroup can raise the dof faster
//   than the signal), so the early-bail below can rarely miss an admissible subset. Every set actually
//   RETURNED is re-verified admissible, so soundness and anti-circularity always hold regardless.
[[nodiscard]] std::vector<int> minimal_admissible_right(
    const device::DeviceF2Blocks& dev_f2, const QpAdmOptions& opts, device::Resources& resources,
    const std::vector<int>& sources, int r0, const std::vector<int>& right_pool, double rank_alpha,
    std::size_t min_size, Admissibility& universe_out) {
    std::vector<int> universe{r0};                       // R0 pinned + dedup the curated pool
    for (int x : right_pool)
        if (x != r0 && std::find(universe.begin(), universe.end(), x) == universe.end())
            universe.push_back(x);
    universe_out = qpwave_admissible(dev_f2, opts, resources, sources, universe, rank_alpha);
    if (!universe_out.ok) return {};                     // inadmissible / qpWave-failed / too-few
    if (universe.size() < min_size) return {};           // admissible, but below the powered-fit floor
    std::vector<int> cur = universe;
    for (bool dropped = true; dropped;) {
        dropped = false;
        for (std::size_t i = 1; i < cur.size(); ++i) {   // i>=1: never drop R0
            if (cur.size() <= min_size) break;           // keep enough outgroups to power the fit
            std::vector<int> trial = cur;
            trial.erase(trial.begin() + static_cast<std::ptrdiff_t>(i));
            if (qpwave_admissible(dev_f2, opts, resources, sources, trial, rank_alpha).ok) {
                cur = std::move(trial);
                dropped = true;
                break;                                   // restart the sweep after a successful drop
            }
        }
    }
    return cur;
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

    // --prerank: emit the pool ranked by outgroup-f3 relatedness to the target, then exit.
    if (config.scan_prerank()) {
        const QpAdmOptions popts = config.qpadm_options();
        Relatedness rel;
        try {
            device::Resources resources = device::build_resources(config.device());
            if (resources.gpus.empty()) {
                std::fprintf(stderr, "steppe scan: no CUDA device available (steppe is a GPU "
                                     "product; a CUDA-capable GPU is required)\n");
                return cfg::kExitRuntimeError;
            }
            const int device_id = resources.gpus.front().device_id;
            device::DeviceF2Blocks dev_f2 =
                device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
            rel = relatedness_f3(dev_f2, popts, resources, target_idx, pool_idx, right_idx);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe scan: device error: %s\n", e.what());
            return exit_code_for_caught(e);
        }
        const std::string target_label = resolver.label_at(target_idx);
        const std::vector<std::size_t> ord = relatedness_order(rel);
        const double kNaN = std::numeric_limits<double>::quiet_NaN();
        std::vector<RelatedRow> rrows;
        rrows.reserve(ord.size());
        for (std::size_t pos : ord) {
            RelatedRow rr;
            rr.source = resolver.label_at(pool_idx[pos]);
            rr.est = pos < rel.est.size() ? rel.est[pos] : kNaN;
            rr.se  = pos < rel.se.size()  ? rel.se[pos]  : kNaN;
            rr.z   = pos < rel.z.size()   ? rel.z[pos]   : kNaN;
            rrows.push_back(std::move(rr));
        }
        std::fprintf(stderr, "steppe scan --prerank: %zu pool sources ranked by relatedness to %s\n",
                     rrows.size(), target_label.c_str());
        if (const auto rc = emit_to_destination(config, "scan",
                [&](std::ostream& os, OutputFormat fmt) { emit_prerank(os, fmt, target_label, rrows); })) {
            return *rc;
        }
        return cfg::kExitOk;
    }

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
    std::vector<SwapInfo> swaps;
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
        if (config.scan_suggest_swaps()) {   // Phase 2: drop-culprit/add-related refits (batched)
            const Relatedness rel =
                relatedness_f3(dev_f2, opts, resources, target_idx, pool_idx, right_idx);
            const std::vector<std::size_t> rel_ord = relatedness_order(rel);
            swaps = compute_swaps(dev_f2, opts, resources, search.models, search.results, resolver,
                                  rel, rel_ord, pool_idx, right_idx, alpha);
        }
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
        if (mi < swaps.size() && swaps[mi].has) {
            row.swap_has = true;
            row.swap_drop = swaps[mi].drop;
            row.swap_add = swaps[mi].add;
            row.swap_feasible = swaps[mi].feasible;
            row.swap_p = swaps[mi].p;
        }
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

    // Phase 3: right-set admissibility (--right-search) on the SELECTED model. The gate is the
    // target-fit-blind sources-only qpWave (§1a); fit-p never chooses outgroups.
    if (config.scan_right_search() != "none") {
        if (selected_rank >= rows.size()) {
            std::fprintf(stderr, "steppe scan --right-search: no model passed the gate, so there is "
                                 "no selected model to check outgroups for.\n");
        } else {
            const std::vector<int>& sel_sources = models[order[selected_rank]].left;
            const std::string sel_label = rows[selected_rank].left;
            const double ra = config.qpadm_options().rank_alpha;
            const int k = static_cast<int>(sel_sources.size());
            std::vector<int> right_pool_idx;
            bool pool_ok = true;
            if (k < 2) {
                std::fprintf(stderr, "steppe scan --right-search: selected model %s = %s has a single "
                    "source — outgroup admissibility is vacuous (nothing to distinguish).\n",
                    target_label.c_str(), sel_label.c_str());
                pool_ok = false;
            }
            if (pool_ok && !config.scan_right_pool().empty()) {
                const ResolveListResult rp = resolver.resolve_all(config.scan_right_pool());
                if (!rp.ok) {
                    std::fprintf(stderr, "steppe scan --right-search: %s\n", rp.error.c_str());
                    pool_ok = false;
                } else {
                    right_pool_idx = rp.indices;
                }
            }
            if (pool_ok) try {
                device::Resources resources = device::build_resources(config.device());
                if (resources.gpus.empty()) {
                    std::fprintf(stderr, "steppe scan --right-search: no CUDA device available\n");
                } else {
                    const int device_id = resources.gpus.front().device_id;
                    device::DeviceF2Blocks dev_f2 =
                        device::upload_f2_blocks_to_device(dir.dir.f2, device_id);
                    if (config.scan_right_search() == "check") {
                        const Admissibility a =
                            qpwave_admissible(dev_f2, opts, resources, sel_sources, right_idx, ra);
                        if (!a.enough_outgroups) {
                            std::fprintf(stderr, "steppe scan --right-search check: %s = %s — too few "
                                "outgroups (|right|=%zu < nsource=%d) to distinguish the sources.\n",
                                target_label.c_str(), sel_label.c_str(), right_idx.size(), k);
                        } else if (a.status != Status::Ok) {
                            std::fprintf(stderr, "steppe scan --right-search check: qpWave failed (%s) "
                                "on the source set.\n", scan_status_label(a.status));
                        } else if (a.ok) {
                            std::fprintf(stderr, "steppe scan --right-search check: %s = %s — right set "
                                "is ADMISSIBLE: the outgroups distinguish all %d sources (sources-only "
                                "qpWave rejects every lower rank).\n",
                                target_label.c_str(), sel_label.c_str(), k);
                        } else {
                            std::fprintf(stderr, "steppe scan --right-search check: %s = %s — right set "
                                "is NOT ADMISSIBLE: rank %d is not rejected, so the outgroups cannot "
                                "tell the sources apart at that level. The qpAdm fit is not "
                                "identifiable here — treat the model as unproven.\n",
                                target_label.c_str(), sel_label.c_str(), a.accepted_rank);
                        }
                    } else {   // add-drop
                        std::vector<int> uni_pool = right_pool_idx;
                        for (std::size_t j = 1; j < right_idx.size(); ++j)   // seed non-R0 outgroups too
                            uni_pool.push_back(right_idx[j]);
                        Admissibility uadm;
                        const std::vector<int> minr = minimal_admissible_right(
                            dev_f2, opts, resources, sel_sources, right_idx.front(), uni_pool, ra,
                            static_cast<std::size_t>(k) + 1, uadm);   // floor: nsource+1 for a powered fit
                        if (!minr.empty()) {
                            std::string ml;
                            for (std::size_t j = 0; j < minr.size(); ++j) {
                                if (j) ml += ",";
                                ml += resolver.label_at(minr[j]);
                            }
                            QpAdmModel m;   // robustness: refit under the minimal set — REPORTED only
                            m.target = target_idx;
                            m.left = sel_sources;
                            m.right = minr;
                            m.model_index = 0;
                            const std::vector<QpAdmModel> one{m};
                            const std::vector<QpAdmResult> rr = run_qpadm_search(
                                dev_f2, std::span<const QpAdmModel>(one), opts, resources);
                            const bool feas =
                                !rr.empty() && rr[0].status == Status::Ok && scan_feasible(rr[0]);
                            const double pp =
                                rr.empty() ? std::numeric_limits<double>::quiet_NaN() : rr[0].p;
                            std::fprintf(stderr, "steppe scan --right-search add-drop: %s = %s — minimal "
                                "sufficient right set (%zu outgroups, R0 pinned): %s\n  robustness under "
                                "it: feasible=%s, p=%.4g (reported, NOT used to pick outgroups).\n",
                                target_label.c_str(), sel_label.c_str(), minr.size(), ml.c_str(),
                                feas ? "TRUE" : "FALSE", pp);
                        } else if (!uadm.enough_outgroups) {
                            std::fprintf(stderr, "steppe scan --right-search add-drop: %s = %s — the "
                                "candidate outgroups ({R0} + --right-pool) are too few to distinguish "
                                "%d sources; add more to --right-pool.\n",
                                target_label.c_str(), sel_label.c_str(), k);
                        } else if (uadm.status != Status::Ok) {
                            std::fprintf(stderr, "steppe scan --right-search add-drop: %s = %s — qpWave "
                                "failed (%s) on the source set; cannot assess outgroups.\n",
                                target_label.c_str(), sel_label.c_str(), scan_status_label(uadm.status));
                        } else if (!uadm.ok) {
                            std::fprintf(stderr, "steppe scan --right-search add-drop: %s = %s — no "
                                "admissible outgroup set: rank %d is not rejected even with all "
                                "candidates, so these outgroups cannot separate the sources.\n",
                                target_label.c_str(), sel_label.c_str(), uadm.accepted_rank);
                        } else {
                            std::fprintf(stderr, "steppe scan --right-search add-drop: %s = %s — the "
                                "outgroups distinguish the sources, but there are fewer than the "
                                "nsource+1 (%d) outgroups needed to power a non-degenerate fit; add "
                                "more to --right-pool.\n",
                                target_label.c_str(), sel_label.c_str(), k + 1);
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "steppe scan --right-search: device error: %s\n", e.what());
            }
        }
    }

    if (const auto rc = emit_to_destination(
            config, "scan", [&](std::ostream& os, OutputFormat fmt) {
                const bool sw = config.scan_suggest_swaps();
                switch (fmt) {
                    case OutputFormat::Csv: emit_scan_csv(os, rows, results.size(), n_pass, sw, ','); break;
                    case OutputFormat::Tsv: emit_scan_csv(os, rows, results.size(), n_pass, sw, '\t'); break;
                    case OutputFormat::Json: emit_scan_json(os, rows, results.size(), n_pass, sw); break;
                }
            })) {
        return *rc;
    }

    return cfg::kExitOk;
}

}  // namespace steppe::app
