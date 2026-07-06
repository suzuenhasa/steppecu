// src/app/cmd_readv2_concord.hpp
//
// The thin CLI shim for `steppe readv2-concord` — the host-only READv2 concordance
// validator (the Phase-0 "ruler"; no GPU, no RunConfig, no device). It parses the flags
// into Readv2ConcordArgs, calls the parse-agnostic arithmetic in readv2_concord.{hpp,cpp},
// renders the report, and maps the outcome to a process exit code:
//   0 = PASS (all floors met), 1 = ran cleanly but concordance FAIL (a floor not met),
//   2 = bad args / missing column / bad degree token / duplicate pair key,
//   4 = a table file could not be opened / read.
#ifndef STEPPE_APP_CMD_READV2_CONCORD_HPP
#define STEPPE_APP_CMD_READV2_CONCORD_HPP

#include <string>

namespace steppe::app {

// Owned at run_cli scope so it outlives CLI11's parse (member initializers = the flag
// defaults documented in scope §B.2).
struct Readv2ConcordArgs {
    std::string a_path;                 // REQUIRED: steppe's READv2 output (TEST table)
    std::string b_path;                 // REQUIRED: the reference tool's output (ORACLE table)
    double p0_atol = 5e-3;              // P0_norm abs tolerance
    double p0_rtol = 1e-2;             // P0_norm rel tolerance
    double degree_agreement_min = 0.95;  // PASS floor: degree-match fraction
    double p0_within_tol_min = 0.90;    // PASS floor: P0_norm within-tol fraction
    double coverage_min = 1.0;          // PASS floor: oracle-pair coverage
    std::string format = "text";        // text | json
    std::string out_path;               // OPTIONAL; default stdout
};

// Runs the validator; returns the process exit code (see the header banner).
[[nodiscard]] int run_readv2_concord(const Readv2ConcordArgs& args);

}  // namespace steppe::app

#endif  // STEPPE_APP_CMD_READV2_CONCORD_HPP
