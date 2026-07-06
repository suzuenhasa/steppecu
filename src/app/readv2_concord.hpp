// src/app/readv2_concord.hpp
//
// The READv2 concordance validator — the Phase-0 "ruler". Pure host-only, CUDA-free
// arithmetic (no device, no RunConfig): it reads TWO READv2 output tables in the frozen
// steppe schema (steppe's --a TEST table vs the reference tool's --b ORACLE table), keys
// them on the unordered sample pair, and computes (a) the relatedness-degree confusion
// matrix + overall classification agreement and (b) per-pair P0_norm numeric concordance.
// It never classifies — it only diffs the `degree` strings and the P0_norm values the two
// tables already carry. This module is parse-agnostic and directly unit-testable; the thin
// CLI shim lives in cmd_readv2_concord.{hpp,cpp}.
//
// Frozen schema (scope §1a): one row per unordered pair, header-driven by column NAME —
//   sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z
// Degree enum, exactly these four lowercase tokens: identical | first | second | unrelated.
#ifndef STEPPE_APP_READV2_CONCORD_HPP
#define STEPPE_APP_READV2_CONCORD_HPP

#include <array>
#include <iosfwd>
#include <map>
#include <string>

namespace steppe::app {

// The four frozen READv2 relatedness-degree tokens (fixed index order 0..3).
inline constexpr int kNumDegrees = 4;
// Order: identical, first, second, unrelated.
const std::array<const char*, kNumDegrees>& degree_tokens();

// A single parsed READv2 row. Only the four columns the validator reads carry real values;
// the rest of the schema is tolerated but ignored.
struct Readv2Row {
    std::string sampleA;
    std::string sampleB;
    std::string degree;    // raw token, validated against the four-enum
    double p0_norm = 0.0;  // NaN when the source cell was NA / null / empty / non-numeric
};

// The diff of two READv2 tables (test A vs oracle B), all metrics over the intersection.
struct ConcordanceReport {
    int a_rows = 0;
    int b_rows = 0;
    int common_pairs = 0;
    int a_only = 0;  // pairs steppe emitted that the oracle lacks (reported, not fatal)
    int b_only = 0;  // oracle pairs steppe missed (drive coverage down)
    double coverage = 0.0;

    // Degree confusion: rows = oracle B (truth), cols = steppe A (test),
    // order identical/first/second/unrelated.
    std::array<std::array<int, kNumDegrees>, kNumDegrees> confusion{};
    int degree_agree_num = 0;
    int degree_agree_den = 0;
    double degree_agreement = 0.0;

    double p0_max_abs_dev = 0.0;
    std::string p0_max_abs_dev_key;
    double p0_max_rel_dev = 0.0;
    std::string p0_max_rel_dev_key;
    int p0_within_tol_num = 0;
    int p0_within_tol_den = 0;
    double p0_within_tol_frac = 0.0;

    bool pass = false;
};

// The PASS floors + P0_norm tolerances.
struct ConcordanceThresholds {
    double p0_atol = 5e-3;
    double p0_rtol = 1e-2;
    double degree_agreement_min = 0.95;
    double p0_within_tol_min = 0.90;
    double coverage_min = 1.0;
};

// The outcome lane — a malformed table / bad enum / duplicate key is DISTINCT from a clean
// concordance FAIL so a disagreement is never confused with a broken input.
enum class ConcordStatus {
    kOk,        // computed cleanly; report.pass carries PASS vs FAIL
    kBadInput,  // missing required column, degree token outside the enum, or a duplicate pair key
    kIoError,   // a table file could not be opened / read
};

struct ConcordResult {
    ConcordStatus status = ConcordStatus::kOk;
    std::string error;         // human message when status != kOk
    ConcordanceReport report;  // valid only when status == kOk
};

// Canonical unordered-pair key: min(a,b) + '\x1f' + max(a,b).
[[nodiscard]] std::string pair_key(const std::string& a, const std::string& b);

// The entry point: read both tables, compute every metric, gate PASS/FAIL against `thr`.
[[nodiscard]] ConcordResult run_concordance(const std::string& a_path, const std::string& b_path,
                                            const ConcordanceThresholds& thr);

// Render the report to `os`. format: "text" (human block + stable machine `concord_*` trailer
// + final `RESULT: PASS|FAIL`) or "json" (one object, confusion as a nested 4x4 array).
void write_report(std::ostream& os, const ConcordanceReport& rep,
                  const ConcordanceThresholds& thr, const std::string& format);

}  // namespace steppe::app

#endif  // STEPPE_APP_READV2_CONCORD_HPP
