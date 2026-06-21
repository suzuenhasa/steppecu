// src/app/result_emit.cpp
//
// QpAdmResult -> CSV / TSV / JSON serialization (cli-bindings.md §4.4). PLAIN C++20,
// app-only, NO CUDA header.
#include "app/result_emit.hpp"

#include <climits>   // INT_MIN (the rankdrop dofdiff NA sentinel)
#include <cmath>     // std::isnan
#include <cstddef>
#include <ostream>
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

}  // namespace steppe::app
