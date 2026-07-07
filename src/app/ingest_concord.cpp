// src/app/ingest_concord.cpp
//
// Reference: docs/reference/src_app_ingest_concord.cpp.md
//
// The VCF-ingest concordance validator arithmetic (host-only, CUDA-free). See the
// header for the schema + the rsID join over valid-pos38 oracle rows. Exact
// per-site match of {call, dosage, source, drop_reason}, with the ref-block
// hom-ref subset surfaced on its own line (the carry-forward requirement).
#include "app/ingest_concord.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace steppe::app {

namespace {

[[nodiscard]] std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

// Split on TAB (the oracle/report schema is tab-separated); keeps empty cells
// (an empty drop_reason field is meaningful and must compare equal to "").
[[nodiscard]] std::vector<std::string> split_tab(const std::string& line) {
    std::vector<std::string> cells;
    std::string cur;
    for (char c : line) {
        if (c == '\t') { cells.push_back(cur); cur.clear(); }
        else if (c != '\r') { cur += c; }
    }
    cells.push_back(cur);
    return cells;
}

struct Row {
    std::string call, dosage, source, drop_reason, pos38;
};

[[nodiscard]] bool read_table(const std::string& path, std::map<std::string, Row>& out,
                              bool drop_null_pos38, long long& rows_raw, long long& rows_kept,
                              std::string& err, IngestConcordStatus& st) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { st = IngestConcordStatus::kIoError; err = "cannot open table: " + path; return false; }

    std::string header;
    while (std::getline(f, header)) if (!trim(header).empty()) break;
    if (trim(header).empty()) {
        st = IngestConcordStatus::kIoError; err = "empty table (no header): " + path; return false;
    }
    const std::vector<std::string> head = split_tab(header);
    std::map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < head.size(); ++i) col[trim(head[i])] = i;
    for (const char* req : {"rsID", "call", "dosage", "source", "drop_reason", "pos38"}) {
        if (!col.count(req)) {
            st = IngestConcordStatus::kBadInput;
            err = std::string("required column missing: '") + req + "' in " + path;
            return false;
        }
    }
    const std::size_t ci_rs = col["rsID"], ci_call = col["call"], ci_dos = col["dosage"];
    const std::size_t ci_src = col["source"], ci_dr = col["drop_reason"], ci_p38 = col["pos38"];
    std::size_t need = 0;
    for (std::size_t v : {ci_rs, ci_call, ci_dos, ci_src, ci_dr, ci_p38}) need = std::max(need, v);

    rows_raw = 0; rows_kept = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (trim(line).empty()) continue;
        const std::vector<std::string> c = split_tab(line);
        if (c.size() <= need) {
            st = IngestConcordStatus::kBadInput;
            err = "short row (fewer columns than the header requires) in " + path;
            return false;
        }
        ++rows_raw;
        Row r;
        r.call = trim(c[ci_call]);
        r.dosage = trim(c[ci_dos]);
        r.source = trim(c[ci_src]);
        r.drop_reason = trim(c[ci_dr]);
        r.pos38 = trim(c[ci_p38]);
        // Skip the orchestrated lift-stage drops (null pos38) when requested.
        if (drop_null_pos38 && (r.pos38.empty() || r.pos38 == "None" || r.pos38 == "NA")) continue;
        const std::string key = trim(c[ci_rs]);
        if (!out.emplace(key, r).second) {
            st = IngestConcordStatus::kBadInput;
            err = "duplicate rsID key '" + key + "' in " + path;
            return false;
        }
        ++rows_kept;
    }
    st = IngestConcordStatus::kOk;
    return true;
}

[[nodiscard]] std::string g6(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

}  // namespace

IngestConcordResult run_ingest_concordance(const std::string& a_path, const std::string& b_path,
                                           const IngestThresholds& thr) {
    IngestConcordResult res;
    std::map<std::string, Row> A, B;
    long long a_raw = 0, a_kept = 0, b_raw = 0, b_kept = 0;

    // steppe's report has no null-pos38 rows; keep them all. The oracle carries
    // no_lift/dup rows with pos38 == "None" — drop them from the join.
    if (!read_table(a_path, A, /*drop_null_pos38=*/false, a_raw, a_kept, res.error, res.status))
        return res;
    if (!read_table(b_path, B, /*drop_null_pos38=*/true, b_raw, b_kept, res.error, res.status))
        return res;

    IngestReport& rep = res.report;
    rep.a_rows = a_kept;
    rep.b_rows = b_kept;
    rep.b_rows_raw = b_raw;

    long long common = 0;
    for (const auto& [key, brow] : B) {
        const auto it = A.find(key);
        if (it == A.end()) continue;  // b_only
        ++common;
        const Row& arow = it->second;
        const bool match = (arow.call == brow.call) && (arow.dosage == brow.dosage) &&
                           (arow.source == brow.source) && (arow.drop_reason == brow.drop_reason);
        if (match) ++rep.match_all;
        else {
            ++rep.mismatch;
            if (rep.first_mismatch_key.empty()) {
                rep.first_mismatch_key = key;
                rep.first_mismatch_detail =
                    "steppe{call=" + arow.call + ",dos=" + arow.dosage + ",src=" + arow.source +
                    ",dr=" + arow.drop_reason + "} oracle{call=" + brow.call + ",dos=" + brow.dosage +
                    ",src=" + brow.source + ",dr=" + brow.drop_reason + "}";
            }
        }
        // Ref-block hom-ref novel-surface subset (oracle-defined).
        if (brow.source == "refblock" && brow.call == "homref") {
            ++rep.refblock_total;
            if (match) ++rep.refblock_match;
        }
        // Explicit-variant subset (oracle-defined).
        if (brow.source == "variant") {
            ++rep.variant_total;
            if (match) ++rep.variant_match;
        }
    }

    rep.common = common;
    rep.a_only = rep.a_rows - common;
    rep.b_only = rep.b_rows - common;
    rep.coverage = (rep.b_rows > 0) ? static_cast<double>(common) / static_cast<double>(rep.b_rows) : 0.0;
    rep.overall_match = (common > 0) ? static_cast<double>(rep.match_all) / static_cast<double>(common) : 0.0;
    rep.refblock_match_frac =
        (rep.refblock_total > 0) ? static_cast<double>(rep.refblock_match) / static_cast<double>(rep.refblock_total) : 0.0;
    rep.variant_match_frac =
        (rep.variant_total > 0) ? static_cast<double>(rep.variant_match) / static_cast<double>(rep.variant_total) : 0.0;

    rep.pass = (rep.coverage >= thr.coverage_min) && (common > 0) &&
               (rep.overall_match >= thr.overall_match_min) &&
               (rep.refblock_match_frac >= thr.refblock_match_min) &&
               (rep.variant_match_frac >= thr.variant_match_min);
    return res;
}

void write_ingest_report(std::ostream& os, const IngestReport& rep, const IngestThresholds& thr,
                         const std::string& format) {
    if (format == "json") {
        os << "{\n";
        os << "  \"a_rows\": " << rep.a_rows << ",\n";
        os << "  \"b_rows\": " << rep.b_rows << ",\n";
        os << "  \"b_rows_raw\": " << rep.b_rows_raw << ",\n";
        os << "  \"common\": " << rep.common << ",\n";
        os << "  \"a_only\": " << rep.a_only << ",\n";
        os << "  \"b_only\": " << rep.b_only << ",\n";
        os << "  \"coverage\": " << g6(rep.coverage) << ",\n";
        os << "  \"match_all\": " << rep.match_all << ",\n";
        os << "  \"mismatch\": " << rep.mismatch << ",\n";
        os << "  \"overall_match\": " << g6(rep.overall_match) << ",\n";
        os << "  \"refblock_total\": " << rep.refblock_total << ",\n";
        os << "  \"refblock_match\": " << rep.refblock_match << ",\n";
        os << "  \"refblock_match_frac\": " << g6(rep.refblock_match_frac) << ",\n";
        os << "  \"variant_total\": " << rep.variant_total << ",\n";
        os << "  \"variant_match\": " << rep.variant_match << ",\n";
        os << "  \"variant_match_frac\": " << g6(rep.variant_match_frac) << ",\n";
        os << "  \"result\": \"" << (rep.pass ? "PASS" : "FAIL") << "\"\n";
        os << "}\n";
        return;
    }

    os << "VCF-ingest concordance (A = steppe/test, B = Stage-0 oracle/ruler)\n";
    os << "  rows: A=" << rep.a_rows << "  B(valid-pos38)=" << rep.b_rows
       << "  B(raw)=" << rep.b_rows_raw << "  common=" << rep.common
       << "  a_only=" << rep.a_only << "  b_only=" << rep.b_only << "\n\n";
    {
        const bool ok = rep.coverage >= thr.coverage_min;
        os << "  coverage:              " << g6(rep.coverage) << " (min " << g6(thr.coverage_min)
           << ") " << (ok ? "PASS" : "FAIL") << "\n";
    }
    {
        const bool ok = rep.overall_match >= thr.overall_match_min;
        os << "  overall exact match:   " << rep.match_all << "/" << rep.common << " = "
           << g6(rep.overall_match) << " (min " << g6(thr.overall_match_min) << ") "
           << (ok ? "PASS" : "FAIL") << "\n";
    }
    {
        const bool ok = rep.refblock_match_frac >= thr.refblock_match_min;
        os << "  refblock hom-ref [*]:  " << rep.refblock_match << "/" << rep.refblock_total << " = "
           << g6(rep.refblock_match_frac) << " (min " << g6(thr.refblock_match_min) << ") "
           << (ok ? "PASS" : "FAIL") << "   [*] novel surface: NO external decoder\n";
    }
    {
        const bool ok = rep.variant_match_frac >= thr.variant_match_min;
        os << "  explicit-variant:      " << rep.variant_match << "/" << rep.variant_total << " = "
           << g6(rep.variant_match_frac) << " (min " << g6(thr.variant_match_min) << ") "
           << (ok ? "PASS" : "FAIL") << "\n";
    }
    if (rep.mismatch > 0) {
        os << "\n  first mismatch @ " << rep.first_mismatch_key << ": " << rep.first_mismatch_detail
           << "\n";
    }
    os << "\n";

    os << "iconcord_a_rows: " << rep.a_rows << "\n";
    os << "iconcord_b_rows: " << rep.b_rows << "\n";
    os << "iconcord_b_rows_raw: " << rep.b_rows_raw << "\n";
    os << "iconcord_common: " << rep.common << "\n";
    os << "iconcord_a_only: " << rep.a_only << "\n";
    os << "iconcord_b_only: " << rep.b_only << "\n";
    os << "iconcord_coverage: " << g6(rep.coverage) << "\n";
    os << "iconcord_match_all: " << rep.match_all << "\n";
    os << "iconcord_mismatch: " << rep.mismatch << "\n";
    os << "iconcord_overall_match: " << g6(rep.overall_match) << "\n";
    os << "iconcord_refblock_total: " << rep.refblock_total << "\n";
    os << "iconcord_refblock_match: " << rep.refblock_match << "\n";
    os << "iconcord_refblock_match_frac: " << g6(rep.refblock_match_frac) << "\n";
    os << "iconcord_variant_total: " << rep.variant_total << "\n";
    os << "iconcord_variant_match: " << rep.variant_match << "\n";
    os << "iconcord_variant_match_frac: " << g6(rep.variant_match_frac) << "\n";
    os << "RESULT: " << (rep.pass ? "PASS" : "FAIL") << "\n";
}

}  // namespace steppe::app
