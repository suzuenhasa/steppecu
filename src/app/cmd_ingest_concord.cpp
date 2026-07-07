// src/app/cmd_ingest_concord.cpp
//
// The `steppe ingest-concord` CLI shim (host-only): parse flags -> call the
// arithmetic in ingest_concord.cpp -> render -> exit code. No CUDA, no device,
// no RunConfig. Mirrors cmd_readv2_concord.
#include "app/cmd_ingest_concord.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>

#include "app/ingest_concord.hpp"
#include "core/config/exit_code.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

int run_ingest_concord(const IngestConcordArgs& args) {
    if (args.format != "text" && args.format != "json") {
        std::fprintf(stderr,
                     "steppe ingest-concord: --format must be 'text' or 'json' (got '%s')\n",
                     args.format.c_str());
        return cfg::kExitInvalidConfig;
    }

    IngestThresholds thr;
    thr.overall_match_min = args.overall_match_min;
    thr.refblock_match_min = args.refblock_match_min;
    thr.variant_match_min = args.variant_match_min;
    thr.coverage_min = args.coverage_min;

    const IngestConcordResult res = run_ingest_concordance(args.a_path, args.b_path, thr);
    if (res.status == IngestConcordStatus::kIoError) {
        std::fprintf(stderr, "steppe ingest-concord: %s\n", res.error.c_str());
        return cfg::kExitIoError;
    }
    if (res.status == IngestConcordStatus::kBadInput) {
        std::fprintf(stderr, "steppe ingest-concord: %s\n", res.error.c_str());
        return cfg::kExitInvalidConfig;
    }

    if (args.out_path.empty()) {
        write_ingest_report(std::cout, res.report, thr, args.format);
    } else {
        std::ofstream of(args.out_path, std::ios::trunc);
        if (!of) {
            std::fprintf(stderr, "steppe ingest-concord: cannot write --out %s\n",
                         args.out_path.c_str());
            return cfg::kExitIoError;
        }
        write_ingest_report(of, res.report, thr, args.format);
    }

    // 0 = PASS; a clean concordance FAIL is the distinct literal-1 lane.
    return res.report.pass ? cfg::kExitOk : 1;
}

}  // namespace steppe::app
