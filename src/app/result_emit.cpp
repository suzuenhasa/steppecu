// src/app/result_emit.cpp
//
// QpAdmResult -> CSV / TSV / JSON serialization (cli-bindings.md §4.4). PLAIN C++20,
// app-only, NO CUDA header.
#include "app/result_emit.hpp"

#include <climits>   // INT_MIN (the rankdrop dofdiff NA sentinel)
#include <cmath>     // std::isnan
#include <cstddef>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "steppe/error.hpp"  // steppe::Status

namespace steppe::app {

namespace {

// 17 significant digits round-trips an IEEE-754 double exactly (max precision), so the
// emitted CSV/JSON re-parses to bit-identical values — required for the test's TIGHT
// tier (weights/chisq rtol 1e-6, but the round-trip itself must lose nothing).
[[nodiscard]] std::string fmt_double(double v) {
    if (std::isnan(v)) return "NA";  // §4.4: NaN sentinel (e.g. ChisqUndefined p) -> NA
    std::ostringstream o;
    o.precision(17);
    o << v;
    return o.str();
}

// JSON variant: NaN is not valid JSON, so a NaN double becomes the JSON null literal
// (the diff-able "not computed / undefined" marker; §4.4 filter-on-status contract).
[[nodiscard]] std::string json_double(double v) {
    if (std::isnan(v)) return "null";
    std::ostringstream o;
    o.precision(17);
    o << v;
    return o.str();
}

// The status -> human string the `status` column / field carries (cli-bindings.md
// §4.4). A per-model domain outcome is a VALUE here, never an error.
[[nodiscard]] const char* status_str(Status s) {
    switch (s) {
        case Status::Ok:               return "ok";
        case Status::DeviceOom:        return "device_oom";
        case Status::RankDeficient:    return "rank_deficient";
        case Status::NonSpdCovariance: return "non_spd_covariance";
        case Status::ChisqUndefined:   return "chisq_undefined";
        case Status::InvalidConfig:    return "invalid_config";
    }
    return "unknown";
}

// AT2 res$ "feasible" for the WHOLE model = both weights in [0,1] (the canonical qpAdm
// feasibility screen; cli-bindings.md §4.4 summary `feasible`). Empty weights (a
// domain-failed model) ⇒ not feasible.
[[nodiscard]] bool model_feasible(const QpAdmResult& r) {
    if (r.weight.empty()) return false;
    for (double w : r.weight) {
        if (w < 0.0 || w > 1.0) return false;
    }
    return true;
}

// A rankdrop dofdiff NA sentinel is INT_MIN (matches the test's
// res.rankdrop_dofdiff == INT_MIN NA check, test_qpadm_parity.cu).
[[nodiscard]] std::string fmt_dofdiff(int v) {
    return (v == INT_MIN) ? std::string("NA") : std::to_string(v);
}

// CSV cell quoting for a label (mirrors the committed golden CSVs, which quote string
// columns). A label has no embedded quote in practice; we still escape defensively.
[[nodiscard]] std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

// JSON string escaping (labels + status). Minimal: quotes + backslash + control.
[[nodiscard]] std::string json_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    out += "\"";
    return out;
}

// ---- CSV / TSV ----------------------------------------------------------------
void emit_csv(std::ostream& os, const QpAdmResult& r, const std::string& target,
              const std::vector<std::string>& left, char sep) {
    const bool have_se = !r.se.empty();

    // weights section (target,left,weight,se,z) — golden_fit0_weights.csv columns.
    os << "# section: weights\n";
    os << "\"target\"" << sep << "\"left\"" << sep << "\"weight\"" << sep
       << "\"se\"" << sep << "\"z\"\n";
    for (std::size_t i = 0; i < r.weight.size(); ++i) {
        const std::string lab = (i < left.size()) ? left[i] : std::string("left") + std::to_string(i);
        os << csv_quote(target) << sep << csv_quote(lab) << sep
           << fmt_double(r.weight[i]) << sep
           << (have_se && i < r.se.size() ? fmt_double(r.se[i]) : std::string("NA")) << sep
           << (have_se && i < r.z.size()  ? fmt_double(r.z[i])  : std::string("NA")) << "\n";
    }

    // summary section (scalars).
    os << "# section: summary\n";
    os << "\"p\"" << sep << "\"chisq\"" << sep << "\"dof\"" << sep << "\"f4rank\""
       << sep << "\"est_rank\"" << sep << "\"feasible\"" << sep << "\"status\""
       << sep << "\"precision\"\n";
    os << fmt_double(r.p) << sep << fmt_double(r.chisq) << sep << r.dof << sep
       << r.f4rank << sep << r.est_rank << sep
       << (model_feasible(r) ? "TRUE" : "FALSE") << sep
       << csv_quote(status_str(r.status)) << sep
       << csv_quote(r.precision_tag == Precision::Kind::EmulatedFp64 ? "emu"
                    : r.precision_tag == Precision::Kind::Tf32        ? "tf32"
                                                                      : "fp64")
       << "\n";

    // rankdrop section (AT2 res$rankdrop order) — golden_fit0_rankdrop.csv columns.
    os << "# section: rankdrop\n";
    os << "\"f4rank\"" << sep << "\"dof\"" << sep << "\"chisq\"" << sep << "\"p\""
       << sep << "\"dofdiff\"" << sep << "\"chisqdiff\"" << sep << "\"p_nested\"\n";
    for (std::size_t k = 0; k < r.rankdrop_f4rank.size(); ++k) {
        os << r.rankdrop_f4rank[k] << sep << r.rankdrop_dof[k] << sep
           << fmt_double(r.rankdrop_chisq[k]) << sep << fmt_double(r.rankdrop_p[k]) << sep
           << fmt_dofdiff(r.rankdrop_dofdiff[k]) << sep
           << fmt_double(r.rankdrop_chisqdiff[k]) << sep
           << fmt_double(r.rankdrop_p_nested[k]) << "\n";
    }

    // popdrop section (AT2 res$popdrop) — golden_fit0_popdrop.csv core columns.
    os << "# section: popdrop\n";
    os << "\"pat\"" << sep << "\"wt\"" << sep << "\"dof\"" << sep << "\"chisq\""
       << sep << "\"p\"" << sep << "\"f4rank\"" << sep << "\"feasible\"\n";
    for (std::size_t k = 0; k < r.popdrop_pat.size(); ++k) {
        os << csv_quote(r.popdrop_pat[k]) << sep << r.popdrop_wt[k] << sep
           << r.popdrop_dof[k] << sep << fmt_double(r.popdrop_chisq[k]) << sep
           << fmt_double(r.popdrop_p[k]) << sep << r.popdrop_f4rank[k] << sep
           << (r.popdrop_feasible[k] != 0 ? "TRUE" : "FALSE") << "\n";
    }
}

// ---- JSON (mirrors golden_fit0.json's weights/rankdrop/popdrop block shape) ----
void emit_json(std::ostream& os, const QpAdmResult& r, const std::string& target,
               const std::vector<std::string>& left) {
    const bool have_se = !r.se.empty();

    os << "{\n";

    // weights block: parallel arrays (target,left,weight,se,z), golden_fit0 shape.
    os << "  \"weights\": {\n";
    os << "    \"target\": [";
    for (std::size_t i = 0; i < r.weight.size(); ++i)
        os << (i ? ", " : "") << json_quote(target);
    os << "],\n";
    os << "    \"left\": [";
    for (std::size_t i = 0; i < r.weight.size(); ++i) {
        const std::string lab = (i < left.size()) ? left[i] : std::string("left") + std::to_string(i);
        os << (i ? ", " : "") << json_quote(lab);
    }
    os << "],\n";
    os << "    \"weight\": [";
    for (std::size_t i = 0; i < r.weight.size(); ++i)
        os << (i ? ", " : "") << json_double(r.weight[i]);
    os << "],\n";
    os << "    \"se\": [";
    for (std::size_t i = 0; i < r.weight.size(); ++i)
        os << (i ? ", " : "") << (have_se && i < r.se.size() ? json_double(r.se[i]) : "null");
    os << "],\n";
    os << "    \"z\": [";
    for (std::size_t i = 0; i < r.weight.size(); ++i)
        os << (i ? ", " : "") << (have_se && i < r.z.size() ? json_double(r.z[i]) : "null");
    os << "]\n";
    os << "  },\n";

    // summary block: scalars.
    os << "  \"summary\": {\n";
    os << "    \"p\": " << json_double(r.p) << ",\n";
    os << "    \"chisq\": " << json_double(r.chisq) << ",\n";
    os << "    \"dof\": " << r.dof << ",\n";
    os << "    \"f4rank\": " << r.f4rank << ",\n";
    os << "    \"est_rank\": " << r.est_rank << ",\n";
    os << "    \"feasible\": " << (model_feasible(r) ? "true" : "false") << ",\n";
    os << "    \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "    \"precision\": " << json_quote(
        r.precision_tag == Precision::Kind::EmulatedFp64 ? "emu"
        : r.precision_tag == Precision::Kind::Tf32        ? "tf32"
                                                          : "fp64") << "\n";
    os << "  },\n";

    // rankdrop block: parallel arrays (golden_fit0 res$rankdrop shape).
    os << "  \"rankdrop\": {\n";

    auto emit_int_arr = [&](const char* name, const std::vector<int>& v, bool last) {
        os << "    " << json_quote(name) << ": [";
        for (std::size_t k = 0; k < v.size(); ++k) os << (k ? ", " : "") << v[k];
        os << "]" << (last ? "\n" : ",\n");
    };
    auto emit_dbl_arr = [&](const char* name, const std::vector<double>& v, bool last) {
        os << "    " << json_quote(name) << ": [";
        for (std::size_t k = 0; k < v.size(); ++k) os << (k ? ", " : "") << json_double(v[k]);
        os << "]" << (last ? "\n" : ",\n");
    };
    auto emit_dofdiff_arr = [&](const char* name, const std::vector<int>& v, bool last) {
        os << "    " << json_quote(name) << ": [";
        for (std::size_t k = 0; k < v.size(); ++k)
            os << (k ? ", " : "") << (v[k] == INT_MIN ? std::string("null") : std::to_string(v[k]));
        os << "]" << (last ? "\n" : ",\n");
    };
    emit_int_arr("f4rank", r.rankdrop_f4rank, false);
    emit_int_arr("dof", r.rankdrop_dof, false);
    emit_dbl_arr("chisq", r.rankdrop_chisq, false);
    emit_dbl_arr("p", r.rankdrop_p, false);
    emit_dofdiff_arr("dofdiff", r.rankdrop_dofdiff, false);
    emit_dbl_arr("chisqdiff", r.rankdrop_chisqdiff, false);
    emit_dbl_arr("p_nested", r.rankdrop_p_nested, true);
    os << "  },\n";

    // popdrop block: parallel arrays (golden_fit0 res$popdrop shape).
    os << "  \"popdrop\": {\n";
    os << "    \"pat\": [";
    for (std::size_t k = 0; k < r.popdrop_pat.size(); ++k)
        os << (k ? ", " : "") << json_quote(r.popdrop_pat[k]);
    os << "],\n";
    emit_int_arr("wt", r.popdrop_wt, false);
    emit_int_arr("dof", r.popdrop_dof, false);
    emit_dbl_arr("chisq", r.popdrop_chisq, false);
    emit_dbl_arr("p", r.popdrop_p, false);
    emit_int_arr("f4rank", r.popdrop_f4rank, false);
    os << "    \"feasible\": [";
    for (std::size_t k = 0; k < r.popdrop_feasible.size(); ++k)
        os << (k ? ", " : "") << (r.popdrop_feasible[k] != 0 ? "true" : "false");
    os << "]\n";
    os << "  }\n";

    os << "}\n";
}

// ---- ROTATION (M(cli-3)): per-model table -------------------------------------
// The per-model feasibility decision the rotation row carries. The engine's own
// decision lives on the popdrop FULL-model row (index 0); prefer it (byte-faithful to
// what run_qpadm_search recorded, matching the engine gate test_qpadm_rotation.cu:474),
// falling back to the canonical model_feasible (weights in [0,1]) when popdrop is empty
// (a domain-failed model that produced no popdrop). Both sources agree for these models
// (the same canonical screen), so either matches the golden.
[[nodiscard]] bool rotation_feasible(const QpAdmResult& r) {
    if (!r.popdrop_feasible.empty()) return r.popdrop_feasible[0] != 0;
    return model_feasible(r);
}

// Join the model's left labels into one field (semicolon-separated; §4.4 `left` as one
// field). Quoting is applied by the caller (csv_quote / json_quote).
[[nodiscard]] std::string join_left(const std::vector<std::string>& labels) {
    std::string s;
    for (std::size_t i = 0; i < labels.size(); ++i) { if (i) s += ";"; s += labels[i]; }
    return s;
}

void emit_rotation_csv(std::ostream& os, std::span<const QpAdmResult> results,
                       const std::string& target,
                       const std::vector<std::vector<std::string>>& left_labels, int right_n,
                       char sep) {
    // One section, one row per model (cli-bindings.md §4.4 qpadm-rotate row schema).
    os << "# section: rotation\n";
    os << "\"model_index\"" << sep << "\"target\"" << sep << "\"left\"" << sep
       << "\"right_n\"" << sep << "\"p\"" << sep << "\"chisq\"" << sep << "\"dof\""
       << sep << "\"f4rank\"" << sep << "\"feasible\"" << sep << "\"status\"" << sep
       << "\"weights\"" << sep << "\"se\"\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const QpAdmResult& r = results[i];
        const std::string left =
            (i < left_labels.size()) ? join_left(left_labels[i]) : std::string();
        // weights / se as semicolon-joined per-row sub-fields (se = "NA" when absent).
        std::string wjoin, sjoin;
        for (std::size_t k = 0; k < r.weight.size(); ++k) {
            if (k) { wjoin += ";"; sjoin += ";"; }
            wjoin += fmt_double(r.weight[k]);
            sjoin += (k < r.se.size() ? fmt_double(r.se[k]) : std::string("NA"));
        }
        if (r.se.empty()) sjoin = "NA";  // policy sentinel: SE not computed for this model
        os << r.model_index << sep << csv_quote(target) << sep << csv_quote(left) << sep
           << right_n << sep << fmt_double(r.p) << sep << fmt_double(r.chisq) << sep
           << r.dof << sep << r.est_rank << sep
           << (rotation_feasible(r) ? "TRUE" : "FALSE") << sep
           << csv_quote(status_str(r.status)) << sep
           << csv_quote(wjoin) << sep << csv_quote(sjoin) << "\n";
    }
}

void emit_rotation_json(std::ostream& os, std::span<const QpAdmResult> results,
                        const std::string& target,
                        const std::vector<std::vector<std::string>>& left_labels,
                        int right_n) {
    // Mirror golden_rot.json's models[] shape so a run diffs directly against the golden.
    os << "{\n";
    os << "  \"target\": " << json_quote(target) << ",\n";
    os << "  \"right_n\": " << right_n << ",\n";
    os << "  \"models\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const QpAdmResult& r = results[i];
        const bool have_se = !r.se.empty();
        os << "    {\n";
        os << "      \"model_index\": " << r.model_index << ",\n";
        os << "      \"left\": [";
        if (i < left_labels.size())
            for (std::size_t k = 0; k < left_labels[i].size(); ++k)
                os << (k ? ", " : "") << json_quote(left_labels[i][k]);
        os << "],\n";
        os << "      \"weight\": [";
        for (std::size_t k = 0; k < r.weight.size(); ++k)
            os << (k ? ", " : "") << json_double(r.weight[k]);
        os << "],\n";
        os << "      \"se\": [";
        for (std::size_t k = 0; k < r.weight.size(); ++k)
            os << (k ? ", " : "") << (have_se && k < r.se.size() ? json_double(r.se[k]) : "null");
        os << "],\n";
        os << "      \"z\": [";
        for (std::size_t k = 0; k < r.weight.size(); ++k)
            os << (k ? ", " : "") << (have_se && k < r.z.size() ? json_double(r.z[k]) : "null");
        os << "],\n";
        os << "      \"p\": " << json_double(r.p) << ",\n";
        os << "      \"chisq\": " << json_double(r.chisq) << ",\n";
        os << "      \"dof\": " << r.dof << ",\n";
        os << "      \"f4rank\": " << r.est_rank << ",\n";
        os << "      \"feasible\": " << (rotation_feasible(r) ? "true" : "false") << ",\n";
        os << "      \"status\": " << json_quote(status_str(r.status)) << "\n";
        os << "    }" << (i + 1 < results.size() ? ",\n" : "\n");
    }
    os << "  ]\n";
    os << "}\n";
}

// ---- qpWave (M(cli-2)): the rank-sweep result ---------------------------------
// qpWave has NO target -> no admixture weights / popdrop. The result is the per-rank
// rank-sufficiency sweep (rank_chisq/rank_dof/rank_p, ASCENDING r) + the AT2-shaped
// rankdrop table (f4rank-DESCENDING) + the f4rank/est_rank/status summary. Both emitters
// REUSE the file-static format primitives verbatim (fmt_double/json_double/fmt_dofdiff/
// csv_quote/json_quote/status_str + the rankdrop loop body / parallel-array lambdas), so
// the rankdrop output is byte-shaped exactly like the qpadm rankdrop section / golden_qpwave.

void emit_qpwave_csv(std::ostream& os, const QpWaveResult& r,
                     const std::vector<std::string>& left, int right_n, char sep) {
    // rankdrop section (AT2 res$rankdrop order, f4rank DESCENDING) — the SAME columns and
    // loop body as the qpadm rankdrop section (emit_csv above).
    os << "# section: rankdrop\n";
    os << "\"f4rank\"" << sep << "\"dof\"" << sep << "\"chisq\"" << sep << "\"p\""
       << sep << "\"dofdiff\"" << sep << "\"chisqdiff\"" << sep << "\"p_nested\"\n";
    for (std::size_t k = 0; k < r.rankdrop_f4rank.size(); ++k) {
        os << r.rankdrop_f4rank[k] << sep << r.rankdrop_dof[k] << sep
           << fmt_double(r.rankdrop_chisq[k]) << sep << fmt_double(r.rankdrop_p[k]) << sep
           << fmt_dofdiff(r.rankdrop_dofdiff[k]) << sep
           << fmt_double(r.rankdrop_chisqdiff[k]) << sep
           << fmt_double(r.rankdrop_p_nested[k]) << "\n";
    }

    // per_rank section (the ASCENDING-r sweep; `rank` column == the index r, 0-based).
    os << "# section: per_rank\n";
    os << "\"rank\"" << sep << "\"chisq\"" << sep << "\"dof\"" << sep << "\"p\"\n";
    for (std::size_t rr = 0; rr < r.rank_chisq.size(); ++rr) {
        os << rr << sep << fmt_double(r.rank_chisq[rr]) << sep
           << (rr < r.rank_dof.size() ? std::to_string(r.rank_dof[rr]) : std::string("NA"))
           << sep
           << (rr < r.rank_p.size() ? fmt_double(r.rank_p[rr]) : std::string("NA")) << "\n";
    }

    // summary section (scalars). Echo the reference label + right_n for readability; the
    // minimal schema-gated columns are f4rank/est_rank/status/precision (cli-bindings.md §357).
    os << "# section: summary\n";
    os << "\"f4rank\"" << sep << "\"est_rank\"" << sep << "\"status\"" << sep
       << "\"precision\"" << sep << "\"reference\"" << sep << "\"right_n\"\n";
    os << r.f4rank << sep << r.est_rank << sep
       << csv_quote(status_str(r.status)) << sep
       << csv_quote(r.precision_tag == Precision::Kind::EmulatedFp64 ? "emu"
                    : r.precision_tag == Precision::Kind::Tf32        ? "tf32"
                                                                      : "fp64")
       << sep << csv_quote(left.empty() ? std::string() : left.front()) << sep
       << right_n << "\n";
}

void emit_qpwave_json(std::ostream& os, const QpWaveResult& r,
                      const std::vector<std::string>& left, int right_n) {
    os << "{\n";

    // left[] (left[0] is the reference) + right_n for human readability / round-trip.
    os << "  \"left\": [";
    for (std::size_t i = 0; i < left.size(); ++i)
        os << (i ? ", " : "") << json_quote(left[i]);
    os << "],\n";
    os << "  \"right_n\": " << right_n << ",\n";

    // rankdrop block: parallel arrays (golden_qpwave.json res$rankdrop shape) — the SAME
    // lambdas as the qpadm rankdrop block (NaN->null via json_double; dofdiff INT_MIN->null).
    os << "  \"rankdrop\": {\n";
    auto emit_int_arr = [&](const char* name, const std::vector<int>& v, bool last) {
        os << "    " << json_quote(name) << ": [";
        for (std::size_t k = 0; k < v.size(); ++k) os << (k ? ", " : "") << v[k];
        os << "]" << (last ? "\n" : ",\n");
    };
    auto emit_dbl_arr = [&](const char* name, const std::vector<double>& v, bool last) {
        os << "    " << json_quote(name) << ": [";
        for (std::size_t k = 0; k < v.size(); ++k) os << (k ? ", " : "") << json_double(v[k]);
        os << "]" << (last ? "\n" : ",\n");
    };
    auto emit_dofdiff_arr = [&](const char* name, const std::vector<int>& v, bool last) {
        os << "    " << json_quote(name) << ": [";
        for (std::size_t k = 0; k < v.size(); ++k)
            os << (k ? ", " : "") << (v[k] == INT_MIN ? std::string("null") : std::to_string(v[k]));
        os << "]" << (last ? "\n" : ",\n");
    };
    emit_int_arr("f4rank", r.rankdrop_f4rank, false);
    emit_int_arr("dof", r.rankdrop_dof, false);
    emit_dbl_arr("chisq", r.rankdrop_chisq, false);
    emit_dbl_arr("p", r.rankdrop_p, false);
    emit_dofdiff_arr("dofdiff", r.rankdrop_dofdiff, false);
    emit_dbl_arr("chisqdiff", r.rankdrop_chisqdiff, false);
    emit_dbl_arr("p_nested", r.rankdrop_p_nested, true);
    os << "  },\n";

    // per_rank block: the ASCENDING-r sweep (rank == the 0-based index r).
    os << "  \"per_rank\": {\n";
    os << "    \"rank\": [";
    for (std::size_t rr = 0; rr < r.rank_chisq.size(); ++rr) os << (rr ? ", " : "") << rr;
    os << "],\n";
    emit_dbl_arr("chisq", r.rank_chisq, false);
    emit_int_arr("dof", r.rank_dof, false);
    emit_dbl_arr("p", r.rank_p, true);
    os << "  },\n";

    // summary block: scalars.
    os << "  \"summary\": {\n";
    os << "    \"f4rank\": " << r.f4rank << ",\n";
    os << "    \"est_rank\": " << r.est_rank << ",\n";
    os << "    \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "    \"precision\": " << json_quote(
        r.precision_tag == Precision::Kind::EmulatedFp64 ? "emu"
        : r.precision_tag == Precision::Kind::Tf32        ? "tf32"
                                                          : "fp64") << "\n";
    os << "  }\n";

    os << "}\n";
}

// ---- STANDALONE f4 (the `steppe f4` command) ----------------------------------
// ONE ROW PER QUARTET in input order, exactly the regenerated golden schema
// (golden_fit0_f4_readf2.csv): pop1,pop2,pop3,pop4,est,se,z,p. REUSES the file-static
// format primitives verbatim (fmt_double/json_double/csv_quote/json_quote), so a NaN est/
// se/z/p (a degenerate quartet) emits the SAME NA/null sentinel the qpadm/qpwave emitters
// use. No `# section:` prefix — the f4 output is a single flat table (the golden has the
// bare header + data rows), so a row-for-row CSV diff against the golden is direct.
void emit_f4_csv(std::ostream& os, const F4Result& r,
                 const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                 const std::vector<std::string>& p3, const std::vector<std::string>& p4,
                 char sep) {
    os << "\"pop1\"" << sep << "\"pop2\"" << sep << "\"pop3\"" << sep << "\"pop4\""
       << sep << "\"est\"" << sep << "\"se\"" << sep << "\"z\"" << sep << "\"p\"\n";
    for (std::size_t k = 0; k < r.est.size(); ++k) {
        const auto at = [k](const std::vector<std::string>& v) {
            return k < v.size() ? v[k] : std::string();
        };
        os << csv_quote(at(p1)) << sep << csv_quote(at(p2)) << sep
           << csv_quote(at(p3)) << sep << csv_quote(at(p4)) << sep
           << fmt_double(r.est[k]) << sep << fmt_double(r.se[k]) << sep
           << fmt_double(r.z[k]) << sep << fmt_double(r.p[k]) << "\n";
    }
}

void emit_f4_json(std::ostream& os, const F4Result& r,
                  const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                  const std::vector<std::string>& p3, const std::vector<std::string>& p4) {
    os << "{\n";
    os << "  \"quartets\": [\n";
    for (std::size_t k = 0; k < r.est.size(); ++k) {
        const auto at = [k](const std::vector<std::string>& v) {
            return k < v.size() ? v[k] : std::string();
        };
        os << "    { \"pop1\": " << json_quote(at(p1))
           << ", \"pop2\": " << json_quote(at(p2))
           << ", \"pop3\": " << json_quote(at(p3))
           << ", \"pop4\": " << json_quote(at(p4))
           << ", \"est\": " << json_double(r.est[k])
           << ", \"se\": " << json_double(r.se[k])
           << ", \"z\": " << json_double(r.z[k])
           << ", \"p\": " << json_double(r.p[k]) << " }"
           << (k + 1 < r.est.size() ? ",\n" : "\n");
    }
    os << "  ],\n";
    os << "  \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "  \"precision\": " << json_quote(
        r.precision_tag == Precision::Kind::EmulatedFp64 ? "emu"
        : r.precision_tag == Precision::Kind::Tf32        ? "tf32"
                                                          : "fp64") << "\n";
    os << "}\n";
}

// STANDALONE f3 table emitter — the THREE-column clone of emit_f4_csv (drop pop4). The
// fixture-matched golden schema (golden_fit0_f3_readf2.csv): pop1,pop2,pop3,est,se,z,p.
// REUSES the file-static format primitives verbatim (fmt_double/json_double/csv_quote/
// json_quote), so a NaN est/se/z/p (a degenerate triple) emits the SAME NA/null sentinel.
// No `# section:` prefix — a single flat table (bare header + data rows), so a row-for-row
// CSV diff against the golden is direct.
void emit_f3_csv(std::ostream& os, const F3Result& r,
                 const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                 const std::vector<std::string>& p3, char sep) {
    os << "\"pop1\"" << sep << "\"pop2\"" << sep << "\"pop3\""
       << sep << "\"est\"" << sep << "\"se\"" << sep << "\"z\"" << sep << "\"p\"\n";
    for (std::size_t k = 0; k < r.est.size(); ++k) {
        const auto at = [k](const std::vector<std::string>& v) {
            return k < v.size() ? v[k] : std::string();
        };
        os << csv_quote(at(p1)) << sep << csv_quote(at(p2)) << sep
           << csv_quote(at(p3)) << sep
           << fmt_double(r.est[k]) << sep << fmt_double(r.se[k]) << sep
           << fmt_double(r.z[k]) << sep << fmt_double(r.p[k]) << "\n";
    }
}

void emit_f3_json(std::ostream& os, const F3Result& r,
                  const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                  const std::vector<std::string>& p3) {
    os << "{\n";
    os << "  \"triples\": [\n";
    for (std::size_t k = 0; k < r.est.size(); ++k) {
        const auto at = [k](const std::vector<std::string>& v) {
            return k < v.size() ? v[k] : std::string();
        };
        os << "    { \"pop1\": " << json_quote(at(p1))
           << ", \"pop2\": " << json_quote(at(p2))
           << ", \"pop3\": " << json_quote(at(p3))
           << ", \"est\": " << json_double(r.est[k])
           << ", \"se\": " << json_double(r.se[k])
           << ", \"z\": " << json_double(r.z[k])
           << ", \"p\": " << json_double(r.p[k]) << " }"
           << (k + 1 < r.est.size() ? ",\n" : "\n");
    }
    os << "  ],\n";
    os << "  \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "  \"precision\": " << json_quote(
        r.precision_tag == Precision::Kind::EmulatedFp64 ? "emu"
        : r.precision_tag == Precision::Kind::Tf32        ? "tf32"
                                                          : "fp64") << "\n";
    os << "}\n";
}

}  // namespace

bool parse_output_format(const std::string& token, OutputFormat& out) {
    if (token == "csv")  { out = OutputFormat::Csv;  return true; }
    if (token == "tsv")  { out = OutputFormat::Tsv;  return true; }
    if (token == "json") { out = OutputFormat::Json; return true; }
    return false;
}

void emit_qpadm_result(std::ostream& os, OutputFormat fmt,
                       const QpAdmResult& result,
                       const std::string& target_label,
                       const std::vector<std::string>& left_labels) {
    switch (fmt) {
        case OutputFormat::Csv:  emit_csv(os, result, target_label, left_labels, ','); break;
        case OutputFormat::Tsv:  emit_csv(os, result, target_label, left_labels, '\t'); break;
        case OutputFormat::Json: emit_json(os, result, target_label, left_labels); break;
    }
}

void emit_rotation_table(std::ostream& os, OutputFormat fmt,
                         std::span<const QpAdmResult> results,
                         const std::string& target_label,
                         const std::vector<std::vector<std::string>>& left_labels_per_model,
                         int right_n) {
    switch (fmt) {
        case OutputFormat::Csv:
            emit_rotation_csv(os, results, target_label, left_labels_per_model, right_n, ',');
            break;
        case OutputFormat::Tsv:
            emit_rotation_csv(os, results, target_label, left_labels_per_model, right_n, '\t');
            break;
        case OutputFormat::Json:
            emit_rotation_json(os, results, target_label, left_labels_per_model, right_n);
            break;
    }
}

void emit_qpwave_result(std::ostream& os, OutputFormat fmt,
                        const QpWaveResult& result,
                        const std::vector<std::string>& left_labels,
                        int right_n) {
    switch (fmt) {
        case OutputFormat::Csv:  emit_qpwave_csv(os, result, left_labels, right_n, ','); break;
        case OutputFormat::Tsv:  emit_qpwave_csv(os, result, left_labels, right_n, '\t'); break;
        case OutputFormat::Json: emit_qpwave_json(os, result, left_labels, right_n); break;
    }
}

void emit_f4_result(std::ostream& os, OutputFormat fmt,
                    const F4Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels,
                    const std::vector<std::string>& p4_labels) {
    switch (fmt) {
        case OutputFormat::Csv:
            emit_f4_csv(os, result, p1_labels, p2_labels, p3_labels, p4_labels, ',');
            break;
        case OutputFormat::Tsv:
            emit_f4_csv(os, result, p1_labels, p2_labels, p3_labels, p4_labels, '\t');
            break;
        case OutputFormat::Json:
            emit_f4_json(os, result, p1_labels, p2_labels, p3_labels, p4_labels);
            break;
    }
}

void emit_f3_result(std::ostream& os, OutputFormat fmt,
                    const F3Result& result,
                    const std::vector<std::string>& p1_labels,
                    const std::vector<std::string>& p2_labels,
                    const std::vector<std::string>& p3_labels) {
    switch (fmt) {
        case OutputFormat::Csv:
            emit_f3_csv(os, result, p1_labels, p2_labels, p3_labels, ',');
            break;
        case OutputFormat::Tsv:
            emit_f3_csv(os, result, p1_labels, p2_labels, p3_labels, '\t');
            break;
        case OutputFormat::Json:
            emit_f3_json(os, result, p1_labels, p2_labels, p3_labels);
            break;
    }
}

}  // namespace steppe::app
