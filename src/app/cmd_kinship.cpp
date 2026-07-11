// src/app/cmd_kinship.cpp
//
// The `steppe kinship` command — KING-robust between-family kinship (Manichaikul et al. 2010)
// over a diploid genotype triple, on the GPU. App-only and CUDA-free: the GPU is reached only
// through the run_kinship_* seam. Emits a plink2 .kin0-shaped per-pair table (id1 id2 nsnp
// hethet ibs0 phi degree) as TSV (default) / CSV / JSON.
//
// The diploid counterpart of `steppe readv2`: unlike readv2 (pseudo-haploid, rejects het),
// kinship REQUIRES diploid and consumes the het signal; unlike `steppe fst` (population
// units) the unit is a per-INDIVIDUAL Genetic ID (singleton partition).
#include "app/cmd_kinship.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/resources.hpp"
#include "io/genotype_source.hpp"
#include "steppe/error.hpp"
#include "steppe/kinship.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Read a --samples file of Genetic IDs (one per line; blanks/whitespace ignored).
[[nodiscard]] bool read_samples_file(const std::string& path, std::vector<std::string>& out,
                                     std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open --samples file: " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tok;
        if (ls >> tok) out.push_back(tok);
    }
    if (out.empty()) { err = "--samples file is empty: " + path; return false; }
    return true;
}

// Read a --pairs file of "id1<ws>id2" per line (blank / short lines ignored).
[[nodiscard]] bool read_pairs_file(const std::string& path,
                                   std::vector<std::pair<std::string, std::string>>& out,
                                   std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open --pairs file: " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string a, b;
        if (ls >> a >> b) out.emplace_back(a, b);
    }
    if (out.empty()) { err = "--pairs file has no id1<ws>id2 rows: " + path; return false; }
    return true;
}

// One-pair-per-row table (id1 id2 nsnp hethet ibs0 phi degree). JSON = array of pair objects.
void emit_pairs(std::ostream& os, OutputFormat fmt, const steppe::KinshipResult& r) {
    const std::size_t n = r.id1.size();
    if (fmt == OutputFormat::Json) {
        os << "[\n";
        for (std::size_t k = 0; k < n; ++k) {
            os << "  {\"id1\": " << json_quote(r.id1[k])
               << ", \"id2\": " << json_quote(r.id2[k])
               << ", \"nsnp\": " << r.nsnp[k]
               << ", \"hethet\": " << r.hethet[k]
               << ", \"ibs0\": " << r.ibs0[k]
               << ", \"phi\": " << json_double(r.phi[k])
               << ", \"degree\": " << json_quote(r.degree[k]) << "}";
            os << (k + 1 < n ? ",\n" : "\n");
        }
        os << "]\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Csv) ? ',' : '\t';
    os << "id1" << sep << "id2" << sep << "nsnp" << sep << "hethet" << sep << "ibs0" << sep
       << "phi" << sep << "degree" << "\n";
    for (std::size_t k = 0; k < n; ++k) {
        os << csv_field(r.id1[k], sep) << sep << csv_field(r.id2[k], sep) << sep << r.nsnp[k]
           << sep << r.hethet[k] << sep << r.ibs0[k] << sep << fmt_double(r.phi[k]) << sep
           << csv_field(r.degree[k], sep) << "\n";
    }
}

// Write one Genetic ID per line (no header) — the .king.cutoff.in / .out sample-set files.
[[nodiscard]] bool write_id_list(const std::string& path, const std::vector<std::string>& ids,
                                 std::string& err) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { err = "cannot open output file: " + path; return false; }
    for (const std::string& id : ids) out << id << "\n";
    out.flush();
    if (!out.good()) { err = "write failed (disk full / short write): " + path; return false; }
    return true;
}

// Write the above-cutoff relatedness graph as a plink2-compatible .kin0 (tab-separated, columns
// #IID1 IID2 NSNP HETHET IBS0 KINSHIP). KINSHIP (phi) is written round-trippable (%.17g) so
// `plink2 --king-cutoff-table` reads the exact double and reproduces the identical edge set.
[[nodiscard]] bool write_cutoff_kin0(const std::string& path,
                                     const steppe::KinshipCutoffResult& r, std::string& err) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { err = "cannot open output file: " + path; return false; }
    out << "#IID1\tIID2\tNSNP\tHETHET\tIBS0\tKINSHIP\n";
    char buf[64];
    for (std::size_t e = 0; e < r.edge_id1.size(); ++e) {
        std::snprintf(buf, sizeof(buf), "%.17g", r.edge_phi[e]);
        out << r.edge_id1[e] << '\t' << r.edge_id2[e] << '\t' << r.edge_nsnp[e] << '\t'
            << r.edge_hethet[e] << '\t' << r.edge_ibs0[e] << '\t' << buf << '\n';
    }
    out.flush();
    if (!out.good()) { err = "write failed (disk full / short write): " + path; return false; }
    return true;
}

// Run `steppe kinship --king-cutoff`: the greedy relatedness prune -> PREFIX.king.cutoff.in/.out
// (+ the .king.cutoff.kin0 edge table). PREFIX = --out if given, else "steppe".
int run_kinship_cutoff_command(const cfg::RunConfig& config, const io::GenotypeTriple& triple,
                               const std::optional<std::vector<std::string>>& samples) {
    steppe::KinshipCutoffResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "kinship")) return cfg::kExitRuntimeError;
        result = run_kinship_cutoff(triple.geno, triple.snp, triple.ind, samples,
                                    config.king_cutoff(), resources, config.filter());
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "steppe kinship: %s\n", e.what());
        return cfg::kExitInvalidConfig;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe kinship: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }
    if (result.status != Status::Ok) {
        std::fprintf(stderr,
                     "steppe kinship: could not run --king-cutoff (need >= 2 diploid individuals; "
                     "check --prefix and --samples)\n");
        return cfg::exit_code_for(result.status);
    }

    if (!write_kept_snps(config.emit_kept_snps(), result.kept_snp_ids, "kinship")) {
        return cfg::kExitIoError;
    }

    const std::string prefix = config.out_file().empty() ? std::string("steppe")
                                                         : config.out_file();
    const std::string in_path = prefix + ".king.cutoff.in";
    const std::string out_path = prefix + ".king.cutoff.out";
    const std::string kin0_path = prefix + ".king.cutoff.kin0";
    std::string werr;
    if (!write_id_list(in_path, result.retained, werr) ||
        !write_id_list(out_path, result.removed, werr) ||
        !write_cutoff_kin0(kin0_path, result, werr)) {
        std::fprintf(stderr, "steppe kinship: %s\n", werr.c_str());
        return cfg::kExitIoError;
    }

    std::fprintf(stderr,
                 "steppe kinship --king-cutoff %.6g: %d individuals, %zu above-cutoff pairs, "
                 "%zu retained, %zu removed, %ld autosomal SNPs used\n"
                 "  wrote %s (%zu), %s (%zu), %s (%zu edges)\n",
                 result.cutoff, result.N, result.n_edges, result.retained.size(),
                 result.removed.size(), result.autosomal_snps, in_path.c_str(),
                 result.retained.size(), out_path.c_str(), result.removed.size(),
                 kin0_path.c_str(), result.n_edges);
    return cfg::kExitOk;
}

}  // namespace

int run_kinship_command(const cfg::RunConfig& config) {
    if (config.qpdstat_prefix().empty()) {
        std::fprintf(stderr, "steppe kinship: --prefix PREFIX.{geno,snp,ind} is required\n");
        return cfg::kExitInvalidConfig;
    }
    const io::GenotypeTriple triple = io::resolve_genotype_triple(config.qpdstat_prefix());

    // Optional --samples restriction (a file of Genetic IDs).
    std::optional<std::vector<std::string>> samples;
    if (!config.samples_file().empty()) {
        std::vector<std::string> ids;
        std::string serr;
        if (!read_samples_file(config.samples_file(), ids, serr)) {
            std::fprintf(stderr, "steppe kinship: %s\n", serr.c_str());
            return cfg::kExitInvalidConfig;
        }
        samples = std::move(ids);
    }

    // --king-cutoff: the greedy relatedness prune (all-pairs only) -> .in/.out/.kin0 ID files, a
    // different output shape than the per-pair table, so it has its own dispatch + writers.
    if (config.has_king_cutoff()) {
        return run_kinship_cutoff_command(config, triple, samples);
    }

    // Mode: an explicit --pairs list (the biobank-scale path) or the full C(N,2) sweep.
    std::vector<std::pair<std::string, std::string>> pairs;
    const bool pairs_mode = !config.pairs_file().empty();
    if (pairs_mode) {
        std::string perr;
        if (!read_pairs_file(config.pairs_file(), pairs, perr)) {
            std::fprintf(stderr, "steppe kinship: %s\n", perr.c_str());
            return cfg::kExitInvalidConfig;
        }
    }

    steppe::KinshipResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "kinship")) return cfg::kExitRuntimeError;
        if (pairs_mode) {
            result = run_kinship_pairs(triple.geno, triple.snp, triple.ind, samples, pairs,
                                       config.min_kinship(), resources, config.filter());
        } else {
            // A finite --min-kinship routes through the STREAMED backend (no 5*C(N,2) allocation,
            // the maxcomb cap does not bind). No threshold -> the dense capped sweep (small-N +
            // goldens). STEPPE_KING_FORCE_DENSE forces the dense path with the host filter (the
            // values-gate oracle: dense filtered to phi>=min-kinship == the streamed emission).
            const bool has_threshold =
                config.min_kinship() > -std::numeric_limits<double>::infinity();
            const bool force_dense = std::getenv("STEPPE_KING_FORCE_DENSE") != nullptr;
            if (has_threshold && !force_dense) {
                result = run_kinship_streamed(triple.geno, triple.snp, triple.ind, samples,
                                              config.min_kinship(), resources, config.filter());
            } else {
                result = run_kinship_all_pairs(triple.geno, triple.snp, triple.ind, samples,
                                               config.min_kinship(), config.sweep_sure(), resources,
                                               config.filter());
            }
        }
    } catch (const std::invalid_argument& e) {
        // Fail-fast INPUT reject (unknown --pairs id, self-pair) -> invalid config.
        std::fprintf(stderr, "steppe kinship: %s\n", e.what());
        return cfg::kExitInvalidConfig;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe kinship: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (result.capped) {
        std::fprintf(stderr,
                     "steppe kinship: refusing to enumerate %zu pairs (> the maxcomb cap). "
                     "Pass --sure to override, restrict --samples, or use --pairs FILE.\n",
                     result.enumerated);
        return cfg::kExitInvalidConfig;
    }
    if (result.status != Status::Ok) {
        std::fprintf(stderr,
                     "steppe kinship: could not run (need >= 2 diploid individuals; check "
                     "--prefix, --samples, and --pairs)\n");
        return cfg::exit_code_for(result.status);
    }

    std::fprintf(stderr,
                 "steppe kinship: %d individuals, %zu pairs, %zu emitted, %ld autosomal SNPs used\n",
                 result.N, result.enumerated, result.emitted, result.autosomal_snps);

    if (!write_kept_snps(config.emit_kept_snps(), result.kept_snp_ids, "kinship")) {
        return cfg::kExitIoError;
    }

    if (const auto rc = emit_to_destination(
            config, "kinship", [&](std::ostream& os, OutputFormat fmt) {
                emit_pairs(os, fmt, result);
            })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
