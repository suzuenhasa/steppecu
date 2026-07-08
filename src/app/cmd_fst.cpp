// src/app/cmd_fst.cpp
//
// The `steppe fst` command — standalone per-SNP Weir & Cockerham 1984 FST over a
// population pair, computed on the GPU straight from a genotype triple. App-only and
// CUDA-free: the GPU is reached only through the run_fst seam. Emits a per-SNP TSV/CSV/
// JSON table (with --per-snp) or the genome-wide summary row.
//
// v1 SCOPE GUARD: pairwise (2-pop) WC only. --all-pairs / Hudson / windowed / PBS /
// block-jackknife SE are documented follow-ups.
#include "app/cmd_fst.hpp"

#include <cstdio>
#include <exception>
#include <ostream>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/resources.hpp"
#include "io/genotype_source.hpp"
#include "steppe/error.hpp"
#include "steppe/fst.hpp"

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

std::string char_field(char c) { return std::string(1, c); }

// The genome-wide summary row (popA popB n_valid fst_ratio fst_mean).
void emit_summary(std::ostream& os, OutputFormat fmt, const steppe::FstResult& r) {
    if (fmt == OutputFormat::Json) {
        os << "{\n";
        os << "  \"popA\": " << json_quote(r.popA) << ",\n";
        os << "  \"popB\": " << json_quote(r.popB) << ",\n";
        os << "  \"n_valid\": " << r.n_valid << ",\n";
        os << "  \"fst_ratio\": " << json_double(r.fst_ratio) << ",\n";
        os << "  \"fst_mean\": " << json_double(r.fst_mean) << ",\n";
        os << "  \"status\": \"" << status_text(r.status) << "\"\n";
        os << "}\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "popA" << sep << "popB" << sep << "n_valid" << sep << "fst_ratio" << sep
       << "fst_mean" << "\n";
    os << csv_field(r.popA, sep) << sep << csv_field(r.popB, sep) << sep << r.n_valid << sep
       << fmt_double(r.fst_ratio) << sep << fmt_double(r.fst_mean) << "\n";
}

// The per-SNP FST table (snp_id chrom pos_bp a1 a2 popA popB fst_num fst_den fst valid).
void emit_per_snp(std::ostream& os, OutputFormat fmt, const steppe::FstResult& r) {
    const std::size_t M = r.fst.size();
    if (fmt == OutputFormat::Json) {
        os << "[\n";
        for (std::size_t s = 0; s < M; ++s) {
            os << "  {\"snp_id\": " << json_quote(r.snp_id[s])
               << ", \"chrom\": " << r.chrom[s]
               << ", \"pos_bp\": " << json_double(r.physpos[s])
               << ", \"a1\": " << json_quote(char_field(r.a1[s]))
               << ", \"a2\": " << json_quote(char_field(r.a2[s]))
               << ", \"popA\": " << json_quote(r.popA)
               << ", \"popB\": " << json_quote(r.popB)
               << ", \"fst_num\": " << json_double(r.num[s])
               << ", \"fst_den\": " << json_double(r.den[s])
               << ", \"fst\": " << json_double(r.fst[s])
               << ", \"valid\": " << static_cast<int>(r.valid[s]) << "}";
            os << (s + 1 < M ? ",\n" : "\n");
        }
        os << "]\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "snp_id" << sep << "chrom" << sep << "pos_bp" << sep << "a1" << sep << "a2" << sep
       << "popA" << sep << "popB" << sep << "fst_num" << sep << "fst_den" << sep << "fst"
       << sep << "valid" << "\n";
    for (std::size_t s = 0; s < M; ++s) {
        os << csv_field(r.snp_id[s], sep) << sep << r.chrom[s] << sep << fmt_double(r.physpos[s])
           << sep << char_field(r.a1[s]) << sep << char_field(r.a2[s]) << sep
           << csv_field(r.popA, sep) << sep << csv_field(r.popB, sep) << sep
           << fmt_double(r.num[s]) << sep << fmt_double(r.den[s]) << sep << fmt_double(r.fst[s])
           << sep << static_cast<int>(r.valid[s]) << "\n";
    }
}

// The all-pairs (P x P) genome-wide WC FST matrix: a pop-label header row + a leading label
// column, symmetric with a zero diagonal (JSON = {pops:[...], fst:[[...]]}). A cell is the
// ratio-of-averages Σnum/Σden for that pair (NaN when every shared site is invalid).
void emit_matrix(std::ostream& os, OutputFormat fmt, const steppe::FstMatrixResult& r) {
    const std::size_t P = r.pops.size();
    if (fmt == OutputFormat::Json) {
        os << "{\n  \"pops\": [";
        for (std::size_t i = 0; i < P; ++i)
            os << (i ? ", " : "") << json_quote(r.pops[i]);
        os << "],\n  \"fst\": [\n";
        for (std::size_t i = 0; i < P; ++i) {
            os << "    [";
            for (std::size_t j = 0; j < P; ++j)
                os << (j ? ", " : "") << json_double(r.fst[i * P + j]);
            os << "]" << (i + 1 < P ? ",\n" : "\n");
        }
        os << "  ]\n}\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "pop";
    for (std::size_t j = 0; j < P; ++j) os << sep << csv_field(r.pops[j], sep);
    os << "\n";
    for (std::size_t i = 0; i < P; ++i) {
        os << csv_field(r.pops[i], sep);
        for (std::size_t j = 0; j < P; ++j) os << sep << fmt_double(r.fst[i * P + j]);
        os << "\n";
    }
}

// The all-pairs matrix path (`steppe fst --all-pairs`). Sits ABOVE the single-pair 2-pop
// guard so single-pair semantics are unchanged. wc only in v1 (hudson is a follow-up).
int run_fst_all_pairs_command(const cfg::RunConfig& config) {
    const std::string method = config.fst_method();
    if (method != "wc") {
        std::fprintf(stderr,
                     "steppe fst: --method '%s' is not available (v1 ships wc = Weir-Cockerham "
                     "1984; hudson is a follow-up)\n",
                     method.c_str());
        return cfg::kExitInvalidConfig;
    }

    const std::string& prefix = config.qpdstat_prefix();
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);
    const std::vector<std::string>& pops = config.pops();
    const io::PopSelection& sel = config.pop_selection();
    const int min_n =
        (sel.mode == io::PopSelection::Mode::MinN) ? static_cast<int>(sel.min_n) : 1;

    steppe::FstMatrixResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "fst")) return cfg::kExitRuntimeError;
        result = run_fst_all_pairs(triple.geno, triple.snp, triple.ind, pops, min_n,
                                   config.sweep_sure(), resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe fst: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (result.capped) {
        std::fprintf(stderr,
                     "steppe fst --all-pairs: refusing to enumerate %zu population pairs "
                     "(> the maxcomb cap). Pass --sure to override, or restrict --pops / raise "
                     "--min-n. (the cap is on the PAIR COUNT; note total work is C(P,2)*M "
                     "accumulations — a very large P at full SNPs can still be a long run.)\n",
                     result.enumerated);
        return cfg::kExitInvalidConfig;
    }
    if (result.status != Status::Ok) {
        std::fprintf(stderr,
                     "steppe fst --all-pairs: %s (need >= 2 populations; check --prefix and the "
                     "--pops / --min-n selection)\n",
                     status_text(result.status));
        return cfg::exit_code_for(result.status);
    }

    std::fprintf(stderr,
                 "steppe fst --all-pairs: %zu populations, %zu pairs -> %zux%zu WC FST matrix "
                 "(genome-wide ratio-of-averages over autosomes)\n",
                 result.pops.size(), result.enumerated, result.pops.size(), result.pops.size());

    if (const auto rc = emit_to_destination(
            config, "fst", [&](std::ostream& os, OutputFormat fmt) {
                emit_matrix(os, fmt, result);
            })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace

int run_fst_command(const cfg::RunConfig& config) {
    if (config.qpdstat_prefix().empty()) {
        std::fprintf(stderr,
                     "steppe fst: --prefix PREFIX.{geno,snp,ind} is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.fst_all_pairs()) {
        return run_fst_all_pairs_command(config);
    }
    const std::vector<std::string>& pops = config.pops();
    if (pops.size() != 2) {
        std::fprintf(stderr,
                     "steppe fst: --pops must name EXACTLY two populations A,B to differentiate "
                     "(v1 is pairwise WC FST; --all-pairs is a follow-up); got %zu\n",
                     pops.size());
        return cfg::kExitInvalidConfig;
    }
    const std::string method = config.fst_method();
    if (method != "wc") {
        std::fprintf(stderr,
                     "steppe fst: --method '%s' is not available (v1 ships wc = Weir-Cockerham "
                     "1984; hudson is a follow-up)\n",
                     method.c_str());
        return cfg::kExitInvalidConfig;
    }
    if (pops[0] == pops[1]) {
        std::fprintf(stderr, "steppe fst: --pops A,B must name two DIFFERENT populations\n");
        return cfg::kExitInvalidConfig;
    }

    const std::string& prefix = config.qpdstat_prefix();
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);

    steppe::FstResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "fst")) return cfg::kExitRuntimeError;
        result = run_fst(triple.geno, triple.snp, triple.ind, pops[0], pops[1], resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe fst: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (result.status != Status::Ok) {
        std::fprintf(stderr, "steppe fst: %s (check the two --pops labels exist in the .ind)\n",
                     status_text(result.status));
        return cfg::exit_code_for(result.status);
    }

    // Always echo the genome-wide summary to stderr as a human diagnostic.
    std::fprintf(stderr,
                 "steppe fst: %s vs %s — WC FST(ratio-of-averages) = %.6f over %ld valid "
                 "autosomal sites (mean per-SNP = %.6f)\n",
                 result.popA.c_str(), result.popB.c_str(), result.fst_ratio, result.n_valid,
                 result.fst_mean);

    const bool per_snp = config.fst_per_snp();
    if (const auto rc = emit_to_destination(
            config, "fst", [&](std::ostream& os, OutputFormat fmt) {
                if (per_snp) emit_per_snp(os, fmt, result);
                else emit_summary(os, fmt, result);
            })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
