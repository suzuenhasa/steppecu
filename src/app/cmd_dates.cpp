// src/app/cmd_dates.cpp
//
// The `steppe dates` command — admixture DATING via the weighted ancestry-covariance decay
// (the DATES tool). Reads PREFIX.{geno,snp,ind} (--prefix), the admixed --target, and the two
// reference sources (--left, exactly two), and reports the date (generations) + the
// leave-one-chromosome block-jackknife SE through run_dates (the cuFFT autocorrelation LD
// engine; NEVER the f2 cache, NEVER a host O(M²) SNP-pair loop). Mirrors cmd_qpdstat.cpp's
// --prefix Part-B shape: build_resources -> run_dates -> emit. PLAIN C++20, app-only, NO CUDA
// header (the §4 layering); the GPU is reached ONLY through the CUDA-free run_dates seam.
// main() owns stdout/stderr (architecture.md §10). A degenerate run is a NaN date + exit 0
// (record-and-continue); only faults return nonzero.
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
#include "app/result_emit.hpp"           // OutputFormat, parse_output_format
#include "device/resources.hpp"          // CUDA-FREE: Resources, build_resources
#include "io/genotype_source.hpp"        // io::resolve_genotype_triple (EIGENSTRAT-family vs PLINK --prefix)
#include "steppe/dates.hpp"              // steppe::run_dates + DatesResult/DatesOptions
#include "steppe/error.hpp"             // steppe::Status

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
        os << "  \"target\": \"" << target << "\",\n";
        os << "  \"source1\": \"" << src1 << "\",\n";
        os << "  \"source2\": \"" << src2 << "\",\n";
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
    os << target << sep << src1 << sep << src2 << sep << d(r.date_gen) << sep << d(r.se) << sep
       << d(r.fit_error_sd) << sep << status_text(r.status) << "\n";
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
    // The two reference sources come from --left (exactly two: the ancestral pops). DATES uses
    // wt = freq(source1) - freq(source2); the order sets the weight sign (date-neutral — the
    // decay rate is symmetric in source order, the amplitude flips sign only).
    const std::vector<std::string>& sources = config.left();
    if (sources.size() != 2) {
        std::fprintf(stderr,
                     "steppe dates: --left must name EXACTLY two reference sources (the two "
                     "ancestral populations); got %zu\n", sources.size());
        return cfg::kExitInvalidConfig;
    }

    const std::string& prefix = config.qpdstat_prefix();
    // Format-aware --prefix expansion (M-FR PLINK): EIGENSTRAT family -> P.{geno,snp,ind};
    // PLINK -> P.{bed,bim,fam}. run_dates pins the parser via the GenoReader ctor.
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);
    const std::string& geno = triple.geno;
    const std::string& snp = triple.snp;
    const std::string& ind = triple.ind;

    steppe::DatesOptions opts;  // defaults == the reference par.dates the goldens used.

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

    OutputFormat fmt = OutputFormat::Csv;
    if (!parse_output_format(config.format(), fmt)) {
        std::fprintf(stderr, "steppe dates: unknown --format '%s' (csv|tsv|json)\n",
                     config.format().c_str());
        return cfg::kExitInvalidConfig;
    }

    if (config.out_file().empty()) {
        emit_dates(std::cout, fmt, result, config.target(), sources[0], sources[1]);
    } else {
        std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "steppe dates: cannot open --out file: %s\n",
                         config.out_file().c_str());
            return cfg::kExitIoError;
        }
        emit_dates(out, fmt, result, config.target(), sources[0], sources[1]);
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
