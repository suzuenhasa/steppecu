// Reference: docs/reference/src_app_cmd_readv2_concord.cpp.md
// src/app/cmd_readv2_concord.cpp
//
// The `steppe readv2-concord` CLI shim (host-only): parse flags -> call the arithmetic in
// readv2_concord.cpp -> render -> exit code. No CUDA, no device, no RunConfig.
#include "app/cmd_readv2_concord.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "app/readv2_concord.hpp"
#include "core/config/exit_code.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

int run_readv2_concord(const Readv2ConcordArgs& args) {
    if (args.format != "text" && args.format != "json") {
        std::fprintf(stderr,
                     "steppe readv2-concord: --format must be 'text' or 'json' (got '%s')\n",
                     args.format.c_str());
        return cfg::kExitInvalidConfig;
    }

    ConcordanceThresholds thr;
    thr.p0_atol = args.p0_atol;
    thr.p0_rtol = args.p0_rtol;
    thr.degree_agreement_min = args.degree_agreement_min;
    thr.p0_within_tol_min = args.p0_within_tol_min;
    thr.coverage_min = args.coverage_min;

    const ConcordResult res = run_concordance(args.a_path, args.b_path, thr);
    if (res.status == ConcordStatus::kIoError) {
        std::fprintf(stderr, "steppe readv2-concord: %s\n", res.error.c_str());
        return cfg::kExitIoError;
    }
    if (res.status == ConcordStatus::kBadInput) {
        std::fprintf(stderr, "steppe readv2-concord: %s\n", res.error.c_str());
        return cfg::kExitInvalidConfig;
    }

    if (args.out_path.empty()) {
        write_report(std::cout, res.report, thr, args.format);
    } else {
        std::ofstream of(args.out_path, std::ios::trunc);
        if (!of) {
            std::fprintf(stderr, "steppe readv2-concord: cannot write --out %s\n",
                         args.out_path.c_str());
            return cfg::kExitIoError;
        }
        write_report(of, res.report, thr, args.format);
    }

    // 0 = PASS (all floors met); a clean concordance FAIL is a DISTINCT lane (literal 1)
    // so a disagreement is never confused with the kExitInvalidConfig=2 bad-input lane.
    return res.report.pass ? cfg::kExitOk : 1;
}

}  // namespace steppe::app
