// src/app/cmd_ingest_concord.hpp
//
// Reference: docs/reference/src_app_cmd_ingest_concord.hpp.md
//
// The thin CLI shim for `steppe ingest-concord` — the host-only VCF-ingest
// concordance validator (the Stage-1 gVCF block-correctness gate; no GPU, no
// RunConfig, no device). Parses flags into IngestConcordArgs, calls the
// parse-agnostic arithmetic in ingest_concord.{hpp,cpp}, renders the report, and
// maps the outcome to a process exit code:
//   0 = PASS (all floors met), 1 = ran cleanly but concordance FAIL,
//   2 = bad args / missing column / duplicate rsID, 4 = a table could not be read.
#ifndef STEPPE_APP_CMD_INGEST_CONCORD_HPP
#define STEPPE_APP_CMD_INGEST_CONCORD_HPP

#include <string>

namespace steppe::app {

struct IngestConcordArgs {
    std::string a_path;                   // REQUIRED: steppe's ingest report (TEST)
    std::string b_path;                   // REQUIRED: the Stage-0 oracle dosage TSV (RULER)
    double overall_match_min = 1.0;       // PASS floor: overall exact-match fraction
    double refblock_match_min = 1.0;      // PASS floor: ref-block hom-ref subset match
    double variant_match_min = 1.0;       // PASS floor: explicit-variant subset match
    double coverage_min = 1.0;            // PASS floor: oracle-row coverage
    std::string format = "text";          // text | json
    std::string out_path;                 // OPTIONAL; default stdout
};

[[nodiscard]] int run_ingest_concord(const IngestConcordArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_INGEST_CONCORD_HPP
