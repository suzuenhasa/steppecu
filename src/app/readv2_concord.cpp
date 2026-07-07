// src/app/readv2_concord.cpp
//
// Reference: docs/reference/src_app_readv2_concord.cpp.md
//
// The READv2 concordance validator arithmetic (host-only, CUDA-free). See the header for
// the frozen schema + degree enum. This TU: sniff-parses a READv2 table (CSV or TSV) keyed
// by column NAME, canonicalizes the unordered pair, builds the 4x4 degree confusion
// (oracle rows x steppe cols), computes per-pair P0_norm concordance with the combined
// abs/rel tolerance, and renders the human block + stable machine trailer.
#include "app/readv2_concord.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace steppe::app {

namespace {

// The frozen degree tokens, fixed index order (identical=0 .. unrelated=3).
const std::array<const char*, kNumDegrees> kDegreeTokens = {
    "identical", "first", "second", "unrelated"};

// Map a raw degree token to its fixed index, or -1 if it is outside the enum.
[[nodiscard]] int degree_index(const std::string& tok) {
    for (int i = 0; i < kNumDegrees; ++i)
        if (tok == kDegreeTokens[static_cast<std::size_t>(i)]) return i;
    return -1;
}

// Trim ASCII whitespace + a trailing CR from a field.
[[nodiscard]] std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

// Split one line on `delim`. CSV double-quote handling for the comma case; TSV never quotes.
[[nodiscard]] std::vector<std::string> split_line(const std::string& line, char delim) {
    std::vector<std::string> cells;
    std::string cur;
    bool in_q = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_q) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
                else in_q = false;
            } else {
                cur += c;
            }
        } else if (c == '"') {
            in_q = true;
        } else if (c == delim) {
            cells.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    cells.push_back(cur);
    return cells;
}

// Parse a P0_norm cell. NA / null / empty / non-numeric -> NaN (a real failure, never a
// silent pass; the compute side counts a NaN as NOT within tolerance).
[[nodiscard]] double parse_p0(const std::string& raw) {
    const std::string s = trim(raw);
    if (s.empty() || s == "NA" || s == "null" || s == "NaN" || s == "nan") return std::nan("");
    const char* start = s.c_str();
    char* end = nullptr;
    const double v = std::strtod(start, &end);
    if (end == start) return std::nan("");  // no digits consumed
    return v;
}

// Read one READv2 table into a key->row map. Returns false + sets (err, st) on IO error,
// a missing required column, a degree token outside the enum, or a duplicate pair key.
[[nodiscard]] bool read_table(const std::string& path, std::map<std::string, Readv2Row>& out,
                              std::string& err, ConcordStatus& st) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        st = ConcordStatus::kIoError;
        err = "cannot open table: " + path;
        return false;
    }

    std::string header;
    // Skip blank leading lines to reach the header.
    while (std::getline(f, header)) {
        if (!trim(header).empty()) break;
    }
    if (trim(header).empty()) {
        st = ConcordStatus::kIoError;
        err = "empty table (no header): " + path;
        return false;
    }

    // Sniff the delimiter from the header line: a TAB means TSV, else CSV.
    const char delim = (header.find('\t') != std::string::npos) ? '\t' : ',';
    const std::vector<std::string> head = split_line(header, delim);

    std::map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < head.size(); ++i) col[trim(head[i])] = i;

    for (const char* req : {"sampleA", "sampleB", "degree", "P0_norm"}) {
        if (!col.count(req)) {
            st = ConcordStatus::kBadInput;
            err = std::string("required column missing: '") + req + "' in " + path;
            return false;
        }
    }
    const std::size_t ci_a = col["sampleA"], ci_b = col["sampleB"];
    const std::size_t ci_deg = col["degree"], ci_p0 = col["P0_norm"];
    const std::size_t need = std::max(std::max(ci_a, ci_b), std::max(ci_deg, ci_p0));

    std::string line;
    while (std::getline(f, line)) {
        if (trim(line).empty()) continue;
        const std::vector<std::string> cells = split_line(line, delim);
        if (cells.size() <= need) {
            st = ConcordStatus::kBadInput;
            err = "short row (fewer columns than the header requires) in " + path;
            return false;
        }
        Readv2Row r;
        r.sampleA = trim(cells[ci_a]);
        r.sampleB = trim(cells[ci_b]);
        r.degree = trim(cells[ci_deg]);
        if (degree_index(r.degree) < 0) {
            st = ConcordStatus::kBadInput;
            err = "degree token outside the enum {identical,first,second,unrelated}: '" +
                  r.degree + "' in " + path;
            return false;
        }
        r.p0_norm = parse_p0(cells[ci_p0]);
        const std::string key = pair_key(r.sampleA, r.sampleB);
        if (!out.emplace(key, r).second) {
            st = ConcordStatus::kBadInput;
            err = "duplicate pair key (" + r.sampleA + "," + r.sampleB + ") in " + path;
            return false;
        }
    }
    st = ConcordStatus::kOk;
    return true;
}

[[nodiscard]] std::string g6(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

}  // namespace

const std::array<const char*, kNumDegrees>& degree_tokens() { return kDegreeTokens; }

std::string pair_key(const std::string& a, const std::string& b) {
    const std::string& lo = (a < b) ? a : b;
    const std::string& hi = (a < b) ? b : a;
    return lo + "\x1f" + hi;
}

ConcordResult run_concordance(const std::string& a_path, const std::string& b_path,
                              const ConcordanceThresholds& thr) {
    ConcordResult res;
    std::map<std::string, Readv2Row> A, B;

    if (!read_table(a_path, A, res.error, res.status)) return res;
    if (!read_table(b_path, B, res.error, res.status)) return res;

    ConcordanceReport& rep = res.report;
    rep.a_rows = static_cast<int>(A.size());
    rep.b_rows = static_cast<int>(B.size());

    int common = 0;
    for (const auto& [key, brow] : B) {
        const auto it = A.find(key);
        if (it == A.end()) continue;  // b_only, handled below
        ++common;
        const Readv2Row& arow = it->second;

        // read_table already validated both tokens are in-enum, so the index is >= 0.
        const int b_deg = degree_index(brow.degree);
        const int a_deg = degree_index(arow.degree);
        rep.confusion[static_cast<std::size_t>(b_deg)][static_cast<std::size_t>(a_deg)] += 1;
        if (a_deg == b_deg) ++rep.degree_agree_num;

        // Per-pair P0_norm concordance.
        const double a_p0 = arow.p0_norm, b_p0 = brow.p0_norm;
        if (std::isnan(a_p0) || std::isnan(b_p0)) {
            // A NaN in either table is a real failure — counted NOT within tol, never a
            // silent pass, and excluded from the finite max trackers.
            continue;
        }
        const double abs_dev = std::fabs(a_p0 - b_p0);
        const bool within = abs_dev <= thr.p0_atol + thr.p0_rtol * std::fabs(b_p0);
        if (within) ++rep.p0_within_tol_num;
        if (abs_dev > rep.p0_max_abs_dev) {
            rep.p0_max_abs_dev = abs_dev;
            rep.p0_max_abs_dev_key = key;
        }
        if (b_p0 != 0.0) {
            const double rel_dev = abs_dev / std::fabs(b_p0);
            if (rel_dev > rep.p0_max_rel_dev) {
                rep.p0_max_rel_dev = rel_dev;
                rep.p0_max_rel_dev_key = key;
            }
        }
    }

    rep.common_pairs = common;
    rep.a_only = rep.a_rows - common;
    rep.b_only = rep.b_rows - common;
    rep.coverage = (rep.b_rows > 0) ? static_cast<double>(common) / rep.b_rows : 0.0;

    rep.degree_agree_den = common;
    rep.degree_agreement =
        (common > 0) ? static_cast<double>(rep.degree_agree_num) / common : 0.0;

    rep.p0_within_tol_den = common;
    rep.p0_within_tol_frac =
        (common > 0) ? static_cast<double>(rep.p0_within_tol_num) / common : 0.0;

    // An empty intersection is never a pass; degree_agreement/frac already fall to 0 there.
    rep.pass = (rep.coverage >= thr.coverage_min) &&
               (rep.degree_agreement >= thr.degree_agreement_min) &&
               (rep.p0_within_tol_frac >= thr.p0_within_tol_min);
    return res;
}

void write_report(std::ostream& os, const ConcordanceReport& rep,
                  const ConcordanceThresholds& thr, const std::string& format) {
    const auto& tok = kDegreeTokens;

    if (format == "json") {
        os << "{\n";
        os << "  \"a_rows\": " << rep.a_rows << ",\n";
        os << "  \"b_rows\": " << rep.b_rows << ",\n";
        os << "  \"common_pairs\": " << rep.common_pairs << ",\n";
        os << "  \"a_only\": " << rep.a_only << ",\n";
        os << "  \"b_only\": " << rep.b_only << ",\n";
        os << "  \"coverage\": " << g6(rep.coverage) << ",\n";
        os << "  \"degree_agree_num\": " << rep.degree_agree_num << ",\n";
        os << "  \"degree_agree_den\": " << rep.degree_agree_den << ",\n";
        os << "  \"degree_agreement\": " << g6(rep.degree_agreement) << ",\n";
        os << "  \"confusion\": [";
        for (int r = 0; r < kNumDegrees; ++r) {
            os << (r ? ", [" : "[");
            for (int c = 0; c < kNumDegrees; ++c)
                os << (c ? ", " : "") << rep.confusion[static_cast<std::size_t>(r)]
                                                       [static_cast<std::size_t>(c)];
            os << "]";
        }
        os << "],\n";
        os << "  \"p0_max_abs_dev\": " << g6(rep.p0_max_abs_dev) << ",\n";
        os << "  \"p0_max_rel_dev\": " << g6(rep.p0_max_rel_dev) << ",\n";
        os << "  \"p0_within_tol_num\": " << rep.p0_within_tol_num << ",\n";
        os << "  \"p0_within_tol_den\": " << rep.p0_within_tol_den << ",\n";
        os << "  \"p0_within_tol_frac\": " << g6(rep.p0_within_tol_frac) << ",\n";
        os << "  \"result\": \"" << (rep.pass ? "PASS" : "FAIL") << "\"\n";
        os << "}\n";
        return;
    }

    // --- Human block (for eyeballing; not test-load-bearing) -----------------
    os << "READv2 concordance (A = steppe/test, B = reference/oracle)\n";
    os << "  rows: A=" << rep.a_rows << "  B=" << rep.b_rows
       << "  common=" << rep.common_pairs << "  a_only=" << rep.a_only
       << "  b_only=" << rep.b_only << "\n\n";
    os << "  Degree confusion (rows = oracle truth, cols = steppe test):\n";
    os << "                 identical      first     second  unrelated\n";
    for (int r = 0; r < kNumDegrees; ++r) {
        char row[128];
        std::snprintf(row, sizeof(row), "    %-10s %9d %10d %10d %10d\n",
                      tok[static_cast<std::size_t>(r)],
                      rep.confusion[static_cast<std::size_t>(r)][0],
                      rep.confusion[static_cast<std::size_t>(r)][1],
                      rep.confusion[static_cast<std::size_t>(r)][2],
                      rep.confusion[static_cast<std::size_t>(r)][3]);
        os << row;
    }
    os << "\n";
    {
        const bool ok = rep.degree_agreement >= thr.degree_agreement_min;
        os << "  degree agreement:   " << rep.degree_agree_num << "/" << rep.degree_agree_den
           << " = " << g6(rep.degree_agreement) << " (min " << g6(thr.degree_agreement_min)
           << ") " << (ok ? "PASS" : "FAIL") << "\n";
    }
    {
        const bool ok = rep.p0_within_tol_frac >= thr.p0_within_tol_min;
        os << "  P0_norm within tol: " << rep.p0_within_tol_num << "/" << rep.p0_within_tol_den
           << " = " << g6(rep.p0_within_tol_frac) << " (min " << g6(thr.p0_within_tol_min)
           << ") " << (ok ? "PASS" : "FAIL")
           << "   max|d|=" << g6(rep.p0_max_abs_dev)
           << (rep.p0_max_abs_dev_key.empty() ? "" : " @" ) << rep.p0_max_abs_dev_key
           << " maxrel=" << g6(rep.p0_max_rel_dev)
           << (rep.p0_max_rel_dev_key.empty() ? "" : " @") << rep.p0_max_rel_dev_key << "\n";
    }
    {
        const bool ok = rep.coverage >= thr.coverage_min;
        os << "  coverage:           " << g6(rep.coverage) << " (min " << g6(thr.coverage_min)
           << ") " << (ok ? "PASS" : "FAIL") << "\n";
    }
    os << "\n";

    // --- Stable machine trailer (the test greps these) -----------------------
    os << "concord_a_rows: " << rep.a_rows << "\n";
    os << "concord_b_rows: " << rep.b_rows << "\n";
    os << "concord_common_pairs: " << rep.common_pairs << "\n";
    os << "concord_a_only: " << rep.a_only << "\n";
    os << "concord_b_only: " << rep.b_only << "\n";
    os << "concord_coverage: " << g6(rep.coverage) << "\n";
    os << "concord_degree_agree_num: " << rep.degree_agree_num << "\n";
    os << "concord_degree_agree_den: " << rep.degree_agree_den << "\n";
    os << "concord_degree_agreement: " << g6(rep.degree_agreement) << "\n";
    os << "concord_confusion:";
    for (int r = 0; r < kNumDegrees; ++r)
        for (int c = 0; c < kNumDegrees; ++c)
            os << " " << rep.confusion[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
    os << "\n";
    os << "concord_p0_max_abs_dev: " << g6(rep.p0_max_abs_dev) << "\n";
    os << "concord_p0_max_rel_dev: " << g6(rep.p0_max_rel_dev) << "\n";
    os << "concord_p0_within_tol_num: " << rep.p0_within_tol_num << "\n";
    os << "concord_p0_within_tol_den: " << rep.p0_within_tol_den << "\n";
    os << "concord_p0_within_tol_frac: " << g6(rep.p0_within_tol_frac) << "\n";
    os << "RESULT: " << (rep.pass ? "PASS" : "FAIL") << "\n";
}

}  // namespace steppe::app
