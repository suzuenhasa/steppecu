// src/app/ingest_concord.hpp
//
// Reference: docs/reference/src_app_ingest_concord.hpp.md
//
// The VCF-ingest concordance validator — the Stage-1 gVCF block-correctness gate
// arithmetic. Pure host-only, CUDA-free (no device, no RunConfig): it reads TWO
// per-site report TSVs in the oracle schema (steppe's --a TEST table vs the
// Stage-0 oracle's --b RULER table), joins them on rsID over the oracle rows with
// a valid pos38 (the reader cannot reproduce the orchestrated no_lift/dup_rsid
// lift-stage drops, which carry pos38 == "None"), and asserts EXACT per-site
// agreement of the tuple {call, dosage, source, drop_reason}.
//
// It reports the match rate SEPARATELY for the ref-block hom-ref subset — the
// carry-forward / verify-LOW-#2 requirement: those sites have NO external decoder,
// so steppe's reader IS their independent check and the subset must be surfaced on
// its own line, not folded into the overall. Mirrors readv2_concord.{hpp,cpp};
// the thin CLI shim lives in cmd_ingest_concord.{hpp,cpp}.
//
// Oracle schema (stage-0 spec §9), header-driven by column NAME:
//   rsID  chrom  pos37  pos38  A1  A2  call  dosage  source  flip  drop_reason
#ifndef STEPPE_APP_INGEST_CONCORD_HPP
#define STEPPE_APP_INGEST_CONCORD_HPP

#include <iosfwd>
#include <string>

namespace steppe::app {

// The diff of two ingest report tables (test A vs oracle B), over the rsID join.
struct IngestReport {
    long long a_rows = 0;          // steppe rows
    long long b_rows = 0;          // oracle rows (valid-pos38 only)
    long long b_rows_raw = 0;      // oracle rows incl. null-pos38 (no_lift/dup)
    long long common = 0;          // rsIDs present in both (over valid-pos38 B)
    long long a_only = 0;
    long long b_only = 0;          // oracle valid-pos38 rows steppe missed
    double coverage = 0.0;

    long long match_all = 0;       // all four fields equal
    long long mismatch = 0;
    double overall_match = 0.0;

    // The novel-surface subset: oracle source==refblock && call==homref.
    long long refblock_total = 0;
    long long refblock_match = 0;
    double refblock_match_frac = 0.0;

    // The explicit-variant subset: oracle source==variant.
    long long variant_total = 0;
    long long variant_match = 0;
    double variant_match_frac = 0.0;

    // First few mismatch keys (for eyeballing; not gate-load-bearing).
    std::string first_mismatch_key;
    std::string first_mismatch_detail;

    bool pass = false;
};

// PASS floors.
struct IngestThresholds {
    double overall_match_min = 1.0;
    double refblock_match_min = 1.0;
    double variant_match_min = 1.0;
    double coverage_min = 1.0;
};

enum class IngestConcordStatus {
    kOk,        // computed cleanly; report.pass carries PASS vs FAIL
    kBadInput,  // missing required column or a duplicate rsID
    kIoError,   // a table file could not be opened / read
};

struct IngestConcordResult {
    IngestConcordStatus status = IngestConcordStatus::kOk;
    std::string error;
    IngestReport report;
};

// The entry point: read both tables, compute every metric, gate PASS/FAIL.
[[nodiscard]] IngestConcordResult run_ingest_concordance(const std::string& a_path,
                                                         const std::string& b_path,
                                                         const IngestThresholds& thr);

// Render the report: "text" (human block + stable machine `iconcord_*` trailer +
// final RESULT) or "json".
void write_ingest_report(std::ostream& os, const IngestReport& rep,
                         const IngestThresholds& thr, const std::string& format);

}  // namespace steppe::app

#endif  // STEPPE_APP_INGEST_CONCORD_HPP
