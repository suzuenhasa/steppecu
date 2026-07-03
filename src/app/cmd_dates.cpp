// src/app/cmd_dates.cpp
//
// The `steppe dates` command — admixture dating via the weighted ancestry-covariance
// decay (the DATES tool). App-only and CUDA-free: the GPU is reached only through the
// run_dates seam.
#include "app/cmd_dates.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "core/config/exit_code.hpp"
#include "app/cmd_emit.hpp"
#include "app/result_emit.hpp"
#include "device/resources.hpp"
#include "io/genotype_source.hpp"
#include "steppe/dates.hpp"
#include "steppe/error.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

const char* status_text(steppe::Status s) {
    switch (s) {
        case steppe::Status::Ok: return "ok";
        case steppe::Status::RankDeficient: return "rank_deficient";
        case steppe::Status::NonSpdCovariance: return "non_spd";
        case steppe::Status::ChisqUndefined: return "chisq_undefined";
        case steppe::Status::DeviceOom: return "device_oom";
        case steppe::Status::InvalidConfig: return "invalid_config";
    }
    return "unknown";
}

void emit_dates(std::ostream& os, OutputFormat fmt, const steppe::DatesResult& r,
                const std::string& target, const std::string& src1, const std::string& src2) {
    auto d = [](double v) -> std::string {
        if (std::isnan(v)) return "NA";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", v);
        return std::string(buf);
    };
    if (fmt == OutputFormat::Json) {
        os << "{\n";
        os << "  \"target\": " << json_quote(target) << ",\n";
        os << "  \"source1\": " << json_quote(src1) << ",\n";
        os << "  \"source2\": " << json_quote(src2) << ",\n";
        os << "  \"date_gen\": " << (std::isnan(r.date_gen) ? "null" : d(r.date_gen)) << ",\n";
        os << "  \"se\": " << (std::isnan(r.se) ? "null" : d(r.se)) << ",\n";
        os << "  \"fit_error_sd\": " << (std::isnan(r.fit_error_sd) ? "null" : d(r.fit_error_sd))
           << ",\n";
        os << "  \"status\": \"" << status_text(r.status) << "\"\n";
        os << "}\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "target" << sep << "source1" << sep << "source2" << sep << "date_gen" << sep << "se"
       << sep << "fit_error_sd" << sep << "status" << "\n";
    os << csv_field(target, sep) << sep << csv_field(src1, sep) << sep << csv_field(src2, sep)
       << sep << d(r.date_gen) << sep << d(r.se) << sep << d(r.fit_error_sd) << sep
       << status_text(r.status) << "\n";
}

}  // namespace

int run_dates_command(const cfg::RunConfig& config) {
    if (config.qpdstat_prefix().empty()) {
        std::fprintf(stderr, "steppe dates: --prefix PREFIX.{geno,snp,ind} is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.target().empty()) {
        std::fprintf(stderr, "steppe dates: --target (the admixed population) is required\n");
        return cfg::kExitInvalidConfig;
    }
    const std::vector<std::string>& sources = config.left();
    if (sources.size() != 2) {
        std::fprintf(stderr,
                     "steppe dates: --left must name EXACTLY two reference sources (the two "
                     "ancestral populations); got %zu\n", sources.size());
        return cfg::kExitInvalidConfig;
    }

    const std::string& prefix = config.qpdstat_prefix();
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);
    const std::string& geno = triple.geno;
    const std::string& snp = triple.snp;
    const std::string& ind = triple.ind;

    steppe::DatesOptions opts;

    steppe::DatesResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe dates: no CUDA device available (steppe is a GPU product; a "
                         "CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        result = run_dates(geno, snp, ind, config.target(), sources[0], sources[1], opts,
                           resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe dates: input/device error: %s\n", e.what());
        return cfg::kExitIoError;
    }

    if (const auto rc = emit_to_destination(
            config, "dates", [&](std::ostream& os, OutputFormat fmt) {
                emit_dates(os, fmt, result, config.target(), sources[0], sources[1]);
            })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
