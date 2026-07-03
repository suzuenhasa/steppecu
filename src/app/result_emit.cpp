// src/app/result_emit.cpp
//
// Serializes every steppe command result (qpAdm fit, rotation, qpWave, f4/f3/
// f4-ratio) to CSV/TSV/JSON for stdout. App-only and CUDA-free; every emitter is
// total — a domain-failed result still serializes via sentinels, never throws.
//
// Reference: docs/reference/src_app_result_emit.cpp.md
#include "app/result_emit.hpp"

#include <climits>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "steppe/error.hpp"

namespace steppe::app {

namespace {

// Number formatters (17-digit exact round-trip) — reference §2
[[nodiscard]] std::string fmt_double(double v) {
    if (std::isnan(v)) return "NA";
    std::ostringstream o;
    o.precision(17);
    o << v;
    return o.str();
}

[[nodiscard]] std::string json_double(double v) {
    if (std::isnan(v)) return "null";
    std::ostringstream o;
    o.precision(17);
    o << v;
    return o.str();
}

// Status and precision labels — reference §4
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

[[nodiscard]] const char* precision_str(Precision::Kind k) {
    switch (k) {
        case Precision::Kind::EmulatedFp64: return "emu";
        case Precision::Kind::Tf32:         return "tf32";
        case Precision::Kind::Fp64:         return "fp64";
    }
    return "fp64";
}

// Whole-model feasibility (weights in [0,1]) — reference §5
[[nodiscard]] bool model_feasible(const QpAdmResult& r) {
    if (r.weight.empty()) return false;
    for (double w : r.weight) {
        if (w < 0.0 || w > 1.0) return false;
    }
    return true;
}

// "Not computed" sentinels: INT_MIN dofdiff -> NA — reference §3
[[nodiscard]] std::string fmt_dofdiff(int v) {
    return (v == INT_MIN) ? std::string("NA") : std::to_string(v);
}

// Always-quote CSV cell — reference §6
[[nodiscard]] std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}


// Shared JSON parallel-array primitives — reference §9
void emit_int_arr(std::ostream& os, const char* name, const std::vector<int>& v, bool last) {
    os << "    " << json_quote(name) << ": [";
    for (std::size_t k = 0; k < v.size(); ++k) os << (k ? ", " : "") << v[k];
    os << "]" << (last ? "\n" : ",\n");
}
void emit_dbl_arr(std::ostream& os, const char* name, const std::vector<double>& v, bool last) {
    os << "    " << json_quote(name) << ": [";
    for (std::size_t k = 0; k < v.size(); ++k) os << (k ? ", " : "") << json_double(v[k]);
    os << "]" << (last ? "\n" : ",\n");
}
void emit_dofdiff_arr(std::ostream& os, const char* name, const std::vector<int>& v, bool last) {
    os << "    " << json_quote(name) << ": [";
    for (std::size_t k = 0; k < v.size(); ++k)
        os << (k ? ", " : "") << (v[k] == INT_MIN ? std::string("null") : std::to_string(v[k]));
    os << "]" << (last ? "\n" : ",\n");
}

// Shared rankdrop CSV body — reference §9
template <class Rankdrop>
void emit_rankdrop_csv(std::ostream& os, const Rankdrop& r, char sep) {
    for (std::size_t k = 0; k < r.rankdrop_f4rank.size(); ++k) {
        os << r.rankdrop_f4rank[k] << sep << r.rankdrop_dof[k] << sep
           << fmt_double(r.rankdrop_chisq[k]) << sep << fmt_double(r.rankdrop_p[k]) << sep
           << fmt_dofdiff(r.rankdrop_dofdiff[k]) << sep
           << fmt_double(r.rankdrop_chisqdiff[k]) << sep
           << fmt_double(r.rankdrop_p_nested[k]) << "\n";
    }
}

// Output schemas per command: single qpAdm fit — reference §7
void emit_csv(std::ostream& os, const QpAdmResult& r, const std::string& target,
              const std::vector<std::string>& left, char sep) {
    const bool have_se = !r.se.empty();

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

    os << "# section: summary\n";
    os << "\"p\"" << sep << "\"chisq\"" << sep << "\"dof\"" << sep << "\"f4rank\""
       << sep << "\"est_rank\"" << sep << "\"feasible\"" << sep << "\"status\""
       << sep << "\"precision\"\n";
    os << fmt_double(r.p) << sep << fmt_double(r.chisq) << sep << r.dof << sep
       << r.f4rank << sep << r.est_rank << sep
       << (model_feasible(r) ? "TRUE" : "FALSE") << sep
       << csv_quote(status_str(r.status)) << sep
       << csv_quote(precision_str(r.precision_tag))
       << "\n";

    os << "# section: rankdrop\n";
    os << "\"f4rank\"" << sep << "\"dof\"" << sep << "\"chisq\"" << sep << "\"p\""
       << sep << "\"dofdiff\"" << sep << "\"chisqdiff\"" << sep << "\"p_nested\"\n";
    emit_rankdrop_csv(os, r, sep);

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

void emit_json(std::ostream& os, const QpAdmResult& r, const std::string& target,
               const std::vector<std::string>& left) {
    const bool have_se = !r.se.empty();

    os << "{\n";

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

    os << "  \"summary\": {\n";
    os << "    \"p\": " << json_double(r.p) << ",\n";
    os << "    \"chisq\": " << json_double(r.chisq) << ",\n";
    os << "    \"dof\": " << r.dof << ",\n";
    os << "    \"f4rank\": " << r.f4rank << ",\n";
    os << "    \"est_rank\": " << r.est_rank << ",\n";
    os << "    \"feasible\": " << (model_feasible(r) ? "true" : "false") << ",\n";
    os << "    \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "    \"precision\": " << json_quote(precision_str(r.precision_tag)) << "\n";
    os << "  },\n";

    os << "  \"rankdrop\": {\n";
    emit_int_arr(os, "f4rank", r.rankdrop_f4rank, false);
    emit_int_arr(os, "dof", r.rankdrop_dof, false);
    emit_dbl_arr(os, "chisq", r.rankdrop_chisq, false);
    emit_dbl_arr(os, "p", r.rankdrop_p, false);
    emit_dofdiff_arr(os, "dofdiff", r.rankdrop_dofdiff, false);
    emit_dbl_arr(os, "chisqdiff", r.rankdrop_chisqdiff, false);
    emit_dbl_arr(os, "p_nested", r.rankdrop_p_nested, true);
    os << "  },\n";

    os << "  \"popdrop\": {\n";
    os << "    \"pat\": [";
    for (std::size_t k = 0; k < r.popdrop_pat.size(); ++k)
        os << (k ? ", " : "") << json_quote(r.popdrop_pat[k]);
    os << "],\n";
    emit_int_arr(os, "wt", r.popdrop_wt, false);
    emit_int_arr(os, "dof", r.popdrop_dof, false);
    emit_dbl_arr(os, "chisq", r.popdrop_chisq, false);
    emit_dbl_arr(os, "p", r.popdrop_p, false);
    emit_int_arr(os, "f4rank", r.popdrop_f4rank, false);
    os << "    \"feasible\": [";
    for (std::size_t k = 0; k < r.popdrop_feasible.size(); ++k)
        os << (k ? ", " : "") << (r.popdrop_feasible[k] != 0 ? "true" : "false");
    os << "]\n";
    os << "  }\n";

    os << "}\n";
}

// Rotation feasibility (prefer engine's recorded flag) — reference §5
[[nodiscard]] bool rotation_feasible(const QpAdmResult& r) {
    if (!r.popdrop_feasible.empty()) return r.popdrop_feasible[0] != 0;
    return model_feasible(r);
}

// join_left shared helper — reference §9
[[nodiscard]] std::string join_left(const std::vector<std::string>& labels) {
    std::string s;
    for (std::size_t i = 0; i < labels.size(); ++i) { if (i) s += ";"; s += labels[i]; }
    return s;
}

// Rotation table emitters: the deliberately-mislabeled f4rank column — reference §8
void emit_rotation_csv(std::ostream& os, std::span<const QpAdmResult> results,
                       const std::string& target,
                       const std::vector<std::vector<std::string>>& left_labels, int right_n,
                       char sep) {
    os << "# section: rotation\n";
    os << "\"model_index\"" << sep << "\"target\"" << sep << "\"left\"" << sep
       << "\"right_n\"" << sep << "\"p\"" << sep << "\"chisq\"" << sep << "\"dof\""
       << sep << "\"f4rank\"" << sep << "\"feasible\"" << sep << "\"status\"" << sep
       << "\"weights\"" << sep << "\"se\"\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const QpAdmResult& r = results[i];
        const std::string left =
            (i < left_labels.size()) ? join_left(left_labels[i]) : std::string();
        std::string wjoin, sjoin;
        for (std::size_t k = 0; k < r.weight.size(); ++k) {
            if (k) { wjoin += ";"; sjoin += ";"; }
            wjoin += fmt_double(r.weight[k]);
            sjoin += (k < r.se.size() ? fmt_double(r.se[k]) : std::string("NA"));
        }
        if (r.se.empty()) sjoin = "NA";
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

// Output schemas per command: qpWave rank sweep — reference §7

void emit_qpwave_csv(std::ostream& os, const QpWaveResult& r,
                     const std::vector<std::string>& left, int right_n, char sep) {
    os << "# section: rankdrop\n";
    os << "\"f4rank\"" << sep << "\"dof\"" << sep << "\"chisq\"" << sep << "\"p\""
       << sep << "\"dofdiff\"" << sep << "\"chisqdiff\"" << sep << "\"p_nested\"\n";
    emit_rankdrop_csv(os, r, sep);

    os << "# section: per_rank\n";
    os << "\"rank\"" << sep << "\"chisq\"" << sep << "\"dof\"" << sep << "\"p\"\n";
    for (std::size_t rr = 0; rr < r.rank_chisq.size(); ++rr) {
        os << rr << sep << fmt_double(r.rank_chisq[rr]) << sep
           << (rr < r.rank_dof.size() ? std::to_string(r.rank_dof[rr]) : std::string("NA"))
           << sep
           << (rr < r.rank_p.size() ? fmt_double(r.rank_p[rr]) : std::string("NA")) << "\n";
    }

    os << "# section: summary\n";
    os << "\"f4rank\"" << sep << "\"est_rank\"" << sep << "\"status\"" << sep
       << "\"precision\"" << sep << "\"reference\"" << sep << "\"right_n\"\n";
    os << r.f4rank << sep << r.est_rank << sep
       << csv_quote(status_str(r.status)) << sep
       << csv_quote(precision_str(r.precision_tag))
       << sep << csv_quote(left.empty() ? std::string() : left.front()) << sep
       << right_n << "\n";
}

void emit_qpwave_json(std::ostream& os, const QpWaveResult& r,
                      const std::vector<std::string>& left, int right_n) {
    os << "{\n";

    os << "  \"left\": [";
    for (std::size_t i = 0; i < left.size(); ++i)
        os << (i ? ", " : "") << json_quote(left[i]);
    os << "],\n";
    os << "  \"right_n\": " << right_n << ",\n";

    os << "  \"rankdrop\": {\n";
    emit_int_arr(os, "f4rank", r.rankdrop_f4rank, false);
    emit_int_arr(os, "dof", r.rankdrop_dof, false);
    emit_dbl_arr(os, "chisq", r.rankdrop_chisq, false);
    emit_dbl_arr(os, "p", r.rankdrop_p, false);
    emit_dofdiff_arr(os, "dofdiff", r.rankdrop_dofdiff, false);
    emit_dbl_arr(os, "chisqdiff", r.rankdrop_chisqdiff, false);
    emit_dbl_arr(os, "p_nested", r.rankdrop_p_nested, true);
    os << "  },\n";

    os << "  \"per_rank\": {\n";
    os << "    \"rank\": [";
    for (std::size_t rr = 0; rr < r.rank_chisq.size(); ++rr) os << (rr ? ", " : "") << rr;
    os << "],\n";
    emit_dbl_arr(os, "chisq", r.rank_chisq, false);
    emit_int_arr(os, "dof", r.rank_dof, false);
    emit_dbl_arr(os, "p", r.rank_p, true);
    os << "  },\n";

    os << "  \"summary\": {\n";
    os << "    \"f4rank\": " << r.f4rank << ",\n";
    os << "    \"est_rank\": " << r.est_rank << ",\n";
    os << "    \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "    \"precision\": " << json_quote(precision_str(r.precision_tag)) << "\n";
    os << "  }\n";

    os << "}\n";
}

// label_at shared helper — reference §9
[[nodiscard]] std::string label_at(const std::vector<std::string>& v, std::size_t k) {
    return k < v.size() ? v[k] : std::string();
}

// Output schemas per command: standalone f4 — reference §7
void emit_f4_csv(std::ostream& os, const F4Result& r,
                 const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                 const std::vector<std::string>& p3, const std::vector<std::string>& p4,
                 char sep) {
    os << "\"pop1\"" << sep << "\"pop2\"" << sep << "\"pop3\"" << sep << "\"pop4\""
       << sep << "\"est\"" << sep << "\"se\"" << sep << "\"z\"" << sep << "\"p\"\n";
    for (std::size_t k = 0; k < r.est.size(); ++k) {
        os << csv_quote(label_at(p1, k)) << sep << csv_quote(label_at(p2, k)) << sep
           << csv_quote(label_at(p3, k)) << sep << csv_quote(label_at(p4, k)) << sep
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
        os << "    { \"pop1\": " << json_quote(label_at(p1, k))
           << ", \"pop2\": " << json_quote(label_at(p2, k))
           << ", \"pop3\": " << json_quote(label_at(p3, k))
           << ", \"pop4\": " << json_quote(label_at(p4, k))
           << ", \"est\": " << json_double(r.est[k])
           << ", \"se\": " << json_double(r.se[k])
           << ", \"z\": " << json_double(r.z[k])
           << ", \"p\": " << json_double(r.p[k]) << " }"
           << (k + 1 < r.est.size() ? ",\n" : "\n");
    }
    os << "  ],\n";
    os << "  \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "  \"precision\": " << json_quote(precision_str(r.precision_tag)) << "\n";
    os << "}\n";
}

// Output schemas per command: standalone f3 — reference §7
void emit_f3_csv(std::ostream& os, const F3Result& r,
                 const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                 const std::vector<std::string>& p3, char sep) {
    os << "\"pop1\"" << sep << "\"pop2\"" << sep << "\"pop3\""
       << sep << "\"est\"" << sep << "\"se\"" << sep << "\"z\"" << sep << "\"p\"\n";
    for (std::size_t k = 0; k < r.est.size(); ++k) {
        os << csv_quote(label_at(p1, k)) << sep << csv_quote(label_at(p2, k)) << sep
           << csv_quote(label_at(p3, k)) << sep
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
        os << "    { \"pop1\": " << json_quote(label_at(p1, k))
           << ", \"pop2\": " << json_quote(label_at(p2, k))
           << ", \"pop3\": " << json_quote(label_at(p3, k))
           << ", \"est\": " << json_double(r.est[k])
           << ", \"se\": " << json_double(r.se[k])
           << ", \"z\": " << json_double(r.z[k])
           << ", \"p\": " << json_double(r.p[k]) << " }"
           << (k + 1 < r.est.size() ? ",\n" : "\n");
    }
    os << "  ],\n";
    os << "  \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "  \"precision\": " << json_quote(precision_str(r.precision_tag)) << "\n";
    os << "}\n";
}

// Output schemas per command: standalone f4-ratio — reference §7
void emit_f4ratio_csv(std::ostream& os, const F4RatioResult& r,
                      const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                      const std::vector<std::string>& p3, const std::vector<std::string>& p4,
                      const std::vector<std::string>& p5, char sep) {
    os << "\"pop1\"" << sep << "\"pop2\"" << sep << "\"pop3\"" << sep << "\"pop4\""
       << sep << "\"pop5\"" << sep << "\"alpha\"" << sep << "\"se\"" << sep << "\"z\"\n";
    for (std::size_t k = 0; k < r.alpha.size(); ++k) {
        os << csv_quote(label_at(p1, k)) << sep << csv_quote(label_at(p2, k)) << sep
           << csv_quote(label_at(p3, k)) << sep << csv_quote(label_at(p4, k)) << sep
           << csv_quote(label_at(p5, k)) << sep
           << fmt_double(r.alpha[k]) << sep << fmt_double(r.se[k]) << sep
           << fmt_double(r.z[k]) << "\n";
    }
}

void emit_f4ratio_json(std::ostream& os, const F4RatioResult& r,
                       const std::vector<std::string>& p1, const std::vector<std::string>& p2,
                       const std::vector<std::string>& p3, const std::vector<std::string>& p4,
                       const std::vector<std::string>& p5) {
    os << "{\n";
    os << "  \"tuples\": [\n";
    for (std::size_t k = 0; k < r.alpha.size(); ++k) {
        os << "    { \"pop1\": " << json_quote(label_at(p1, k))
           << ", \"pop2\": " << json_quote(label_at(p2, k))
           << ", \"pop3\": " << json_quote(label_at(p3, k))
           << ", \"pop4\": " << json_quote(label_at(p4, k))
           << ", \"pop5\": " << json_quote(label_at(p5, k))
           << ", \"alpha\": " << json_double(r.alpha[k])
           << ", \"se\": " << json_double(r.se[k])
           << ", \"z\": " << json_double(r.z[k]) << " }"
           << (k + 1 < r.alpha.size() ? ",\n" : "\n");
    }
    os << "  ],\n";
    os << "  \"status\": " << json_quote(status_str(r.status)) << ",\n";
    os << "  \"precision\": " << json_quote(precision_str(r.precision_tag)) << "\n";
    os << "}\n";
}

}  // namespace

// Conditional RFC-4180 CSV field (public) — reference §6
std::string csv_field(const std::string& s, char sep) {
    bool needs_quote = false;
    for (char c : s) {
        if (c == sep || c == '"' || c == '\n' || c == '\r') { needs_quote = true; break; }
    }
    if (!needs_quote) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

// JSON string escaping (public) — reference §6
std::string json_quote(const std::string& s) {
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

// Public entry points: format selector + per-command dispatchers — reference §9
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

void emit_f4ratio_result(std::ostream& os, OutputFormat fmt,
                         const F4RatioResult& result,
                         const std::vector<std::string>& p1_labels,
                         const std::vector<std::string>& p2_labels,
                         const std::vector<std::string>& p3_labels,
                         const std::vector<std::string>& p4_labels,
                         const std::vector<std::string>& p5_labels) {
    switch (fmt) {
        case OutputFormat::Csv:
            emit_f4ratio_csv(os, result, p1_labels, p2_labels, p3_labels, p4_labels,
                             p5_labels, ',');
            break;
        case OutputFormat::Tsv:
            emit_f4ratio_csv(os, result, p1_labels, p2_labels, p3_labels, p4_labels,
                             p5_labels, '\t');
            break;
        case OutputFormat::Json:
            emit_f4ratio_json(os, result, p1_labels, p2_labels, p3_labels, p4_labels,
                              p5_labels);
            break;
    }
}

}  // namespace steppe::app
