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
        if (w < 0.0 || w > 1.0) return false;
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

    const std::vector<QpAdmModel> models =
        enumerate_pool_subsets(target_idx, pool_idx, right_idx, lo, hi);
    if (models.empty()) {
        std::fprintf(stderr,
                     "steppe scan: the [min,max]-sources band over the pool enumerated "
                     "zero models\n");
        return cfg::kExitInvalidConfig;
    }

    if (config.device().devices.size() >= 2) {
        std::fprintf(stderr,
                     "steppe scan: WARNING: %zu devices requested; scan runs single-GPU "
                     "(multi-GPU parked). Use --device 0.\n",
                     config.device().devices.size());
    }

    const QpAdmOptions opts = config.qpadm_options();
    std::vector<QpAdmResult> results;
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
        results = run_qpadm_search(
            dev_f2, std::span<const QpAdmModel>(models), opts, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe scan: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    // Apply the objective: gate every fit, then rank survivors-first by parsimony ->
    // stability -> robustness. Build a parallel per-model metric view over `results`.
    const double alpha = config.scan_p_min();
    const double rank_alpha = config.qpadm_options().rank_alpha;
    const bool allow_clade = config.scan_allow_clade();
    const std::string target_label = resolver.label_at(target_idx);

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
        return ra.model_index < rb.model_index;                  // stable, deterministic
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
