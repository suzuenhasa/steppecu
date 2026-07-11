// src/app/cmd_pca.cpp
//
// The `steppe pca` command — standalone genotype PCA over a genotype triple, computed on the
// GPU (Patterson-2006 standardize -> cuBLAS SYRK covariance -> cuSOLVER eigen -> top-K
// projection). App-only and CUDA-free: the GPU is reached only through the run_pca seam.
// Emits a per-sample PC-coordinate table (sample, pop, PC1..PCk) by default, the scree table
// (pc_index, eigenvalue, var_explained) with --eigenvalues, and — with --emit-html — a
// separate self-contained interactive scatter HTML artifact.
//
// v1 SCOPE GUARD: the exact dense N x N covariance + symmetric eigensolve, gated vs
// scikit-allel/sklearn PCA (NOT ADMIXTOOLS2). A nonlinear UMAP embedding (--embed umap), EMU
// imputation, randomized SVD, and projection of new samples are documented follow-ups.
#include "app/cmd_pca.hpp"

#include <cstdio>
#include <exception>
#include <fstream>
#include <ostream>
#include <sstream>
#include <span>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/pca_html_writer.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/resources.hpp"
#include "io/genotype_source.hpp"
#include "steppe/error.hpp"
#include "steppe/pca.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Read a --project-samples file of Genetic IDs (one per line; blanks/whitespace ignored).
[[nodiscard]] bool read_id_list_file(const std::string& path, std::vector<std::string>& out,
                                     std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open --project-samples file: " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tok;
        if (ls >> tok) out.push_back(tok);
    }
    if (out.empty()) { err = "--project-samples file is empty: " + path; return false; }
    return true;
}

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

// The per-sample PC-coordinate table: sample<sep>pop<sep>PC1..PCk (one row per sample, in
// tile order), or a JSON object with the coords + the eigen spectrum.
void emit_coords(std::ostream& os, OutputFormat fmt, const steppe::PcaResult& r) {
    const int K = r.K;
    // Back-compat guard: the is_projected column/field appears ONLY when projection was
    // requested (some sample is a target). With no --project-* set the output is byte-identical
    // to the pre-projection schema.
    const bool has_proj = r.n_ref > 0 && r.n_ref < r.N;
    const auto projected = [&](int i) -> int {
        return (static_cast<std::size_t>(i) < r.is_projected.size())
                   ? (r.is_projected[static_cast<std::size_t>(i)] ? 1 : 0)
                   : 0;
    };
    if (fmt == OutputFormat::Json) {
        os << "{\n  \"eigenvalues\": [";
        for (int k = 0; k < K; ++k)
            os << (k ? ", " : "") << json_double(r.eigenvalues[static_cast<std::size_t>(k)]);
        os << "],\n  \"var_explained\": [";
        for (int k = 0; k < K; ++k)
            os << (k ? ", " : "") << json_double(r.var_explained[static_cast<std::size_t>(k)]);
        os << "],\n  \"samples\": [\n";
        for (int i = 0; i < r.N; ++i) {
            os << "    {\"sample\": " << json_quote(r.sample_id[static_cast<std::size_t>(i)])
               << ", \"pop\": " << json_quote(r.sample_pop[static_cast<std::size_t>(i)]);
            if (has_proj) os << ", \"projected\": " << (projected(i) ? "true" : "false");
            os << ", \"coords\": [";
            for (int k = 0; k < K; ++k) {
                const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(K) +
                                        static_cast<std::size_t>(k);
                os << (k ? ", " : "") << json_double(r.coords[off]);
            }
            os << "]}" << (i + 1 < r.N ? ",\n" : "\n");
        }
        os << "  ]\n}\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "sample" << sep << "pop";
    if (has_proj) os << sep << "is_projected";
    for (int k = 0; k < K; ++k) os << sep << "PC" << (k + 1);
    os << "\n";
    for (int i = 0; i < r.N; ++i) {
        os << csv_field(r.sample_id[static_cast<std::size_t>(i)], sep) << sep
           << csv_field(r.sample_pop[static_cast<std::size_t>(i)], sep);
        if (has_proj) os << sep << projected(i);
        for (int k = 0; k < K; ++k) {
            const std::size_t off = static_cast<std::size_t>(i) * static_cast<std::size_t>(K) +
                                    static_cast<std::size_t>(k);
            os << sep << fmt_double(r.coords[off]);
        }
        os << "\n";
    }
}

// The scree table: pc_index<sep>eigenvalue<sep>var_explained (one row per PC).
void emit_scree(std::ostream& os, OutputFormat fmt, const steppe::PcaResult& r) {
    const int K = r.K;
    if (fmt == OutputFormat::Json) {
        os << "[\n";
        for (int k = 0; k < K; ++k) {
            os << "  {\"pc_index\": " << (k + 1)
               << ", \"eigenvalue\": " << json_double(r.eigenvalues[static_cast<std::size_t>(k)])
               << ", \"var_explained\": " << json_double(r.var_explained[static_cast<std::size_t>(k)])
               << "}" << (k + 1 < K ? ",\n" : "\n");
        }
        os << "]\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "pc_index" << sep << "eigenvalue" << sep << "var_explained" << "\n";
    for (int k = 0; k < K; ++k) {
        os << (k + 1) << sep << fmt_double(r.eigenvalues[static_cast<std::size_t>(k)]) << sep
           << fmt_double(r.var_explained[static_cast<std::size_t>(k)]) << "\n";
    }
}

}  // namespace

int run_pca_command(const cfg::RunConfig& config) {
    const bool use_bgen = !config.pca_bgen().empty();
    if (config.qpdstat_prefix().empty() && !use_bgen) {
        std::fprintf(stderr,
                     "steppe pca: one of --prefix PREFIX.{geno,snp,ind} or --bgen FILE.bgen is "
                     "required\n");
        return cfg::kExitInvalidConfig;
    }
    if (!config.qpdstat_prefix().empty() && use_bgen) {
        std::fprintf(stderr, "steppe pca: --prefix and --bgen are mutually exclusive\n");
        return cfg::kExitInvalidConfig;
    }
    const int k = config.pca_k();
    if (k < 1) {
        std::fprintf(stderr, "steppe pca: -k must be >= 1 (number of principal components)\n");
        return cfg::kExitInvalidConfig;
    }

    // BGEN v1.2 dosage path: read real-valued ALT dosages and run the Patterson PCA over them.
    // The result schema is identical to the genotype-triple path, so every emit below is reused.
    if (use_bgen) {
        steppe::PcaResult bres;
        try {
            device::Resources resources = device::build_resources(config.device());
            if (!require_first_gpu(resources, "pca")) return cfg::kExitRuntimeError;
            bres = run_pca_bgen(config.pca_bgen(), k, config.device().precision, resources);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe pca: BGEN/device error: %s\n", e.what());
            return exit_code_for_caught(e);
        }
        if (bres.status != Status::Ok) {
            std::fprintf(stderr,
                         "steppe pca: %s (check --bgen file and that -k <= min(samples, usable "
                         "SNPs))\n",
                         status_text(bres.status));
            return cfg::exit_code_for(bres.status);
        }
        const double bv1 = bres.K > 0 ? bres.var_explained[0] * 100.0 : 0.0;
        const double bv2 = bres.K > 1 ? bres.var_explained[1] * 100.0 : 0.0;
        std::fprintf(stderr,
                     "steppe pca: %d samples x %ld SNPs used (%ld dropped monomorphic) from BGEN "
                     "dosages, top-%d PCs; PC1 var=%.2f%%, PC2 var=%.2f%%\n",
                     bres.N, bres.n_snp_used, bres.n_snp_monomorphic, bres.K, bv1, bv2);
        if (!config.pca_emit_html().empty()) {
            if (!write_pca_html(config.pca_emit_html(), bres, "pca")) return cfg::kExitIoError;
            std::fprintf(stderr, "steppe pca: wrote self-contained scatter -> %s\n",
                         config.pca_emit_html().c_str());
        }
        const bool bscree = config.pca_eigenvalues();
        if (const auto rc = emit_to_destination(
                config, "pca", [&](std::ostream& os, OutputFormat fmt) {
                    if (bscree) emit_scree(os, fmt, bres);
                    else emit_coords(os, fmt, bres);
                })) {
            return *rc;
        }
        return cfg::exit_code_for(bres.status);
    }

    const std::string& prefix = config.qpdstat_prefix();
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);
    const std::vector<std::string>& pops = config.pops();
    const std::vector<std::string>& project_pops = config.project_pops();

    // --project-samples FILE -> the list of projected-only Genetic IDs.
    std::vector<std::string> project_samples;
    if (!config.project_samples_file().empty()) {
        std::string serr;
        if (!read_id_list_file(config.project_samples_file(), project_samples, serr)) {
            std::fprintf(stderr, "steppe pca: %s\n", serr.c_str());
            return cfg::kExitInvalidConfig;
        }
    }
    const steppe::PcaProjectMode project_mode = (config.project_mode() == "scaled")
                                                    ? steppe::PcaProjectMode::Scaled
                                                    : steppe::PcaProjectMode::Lsq;
    const steppe::PcaSolver pca_solver =
        (config.pca_solver() == "exact")        ? steppe::PcaSolver::Exact
        : (config.pca_solver() == "randomized") ? steppe::PcaSolver::Randomized
                                                : steppe::PcaSolver::Auto;

    steppe::PcaResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "pca")) return cfg::kExitRuntimeError;
        result = run_pca(triple.geno, triple.snp, triple.ind,
                         std::span<const std::string>(pops), k, config.device().precision,
                         resources, std::span<const std::string>(project_pops),
                         std::span<const std::string>(project_samples), project_mode,
                         config.filter(), pca_solver);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe pca: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (result.status != Status::Ok) {
        std::fprintf(stderr,
                     "steppe pca: %s (check --prefix, the --pops labels, and that -k <= "
                     "min(samples, usable SNPs))\n",
                     status_text(result.status));
        return cfg::exit_code_for(result.status);
    }

    // Human diagnostic to stderr (out of the parsed coordinate stream).
    const double v1 = result.K > 0 ? result.var_explained[0] * 100.0 : 0.0;
    const double v2 = result.K > 1 ? result.var_explained[1] * 100.0 : 0.0;
    std::fprintf(stderr,
                 "steppe pca: %d samples x %ld SNPs used (%ld dropped monomorphic), top-%d PCs; "
                 "PC1 var=%.2f%%, PC2 var=%.2f%%\n",
                 result.N, result.n_snp_used, result.n_snp_monomorphic, result.K, v1, v2);
    if (result.n_ref > 0 && result.n_ref < result.N) {
        std::fprintf(stderr,
                     "steppe pca: %d reference + %d projected (lsqproject %s) samples\n",
                     result.n_ref, result.N - result.n_ref,
                     project_mode == steppe::PcaProjectMode::Scaled ? "scaled" : "lsq");
    }

    if (!write_kept_snps(config.emit_kept_snps(), result.kept_snp_ids, "pca")) {
        return cfg::kExitIoError;
    }

    // The self-contained interactive HTML artifact (separate write path; combinable with --out).
    if (!config.pca_emit_html().empty()) {
        if (!write_pca_html(config.pca_emit_html(), result, "pca")) {
            return cfg::kExitIoError;
        }
        std::fprintf(stderr, "steppe pca: wrote self-contained scatter -> %s\n",
                     config.pca_emit_html().c_str());
    }

    const bool scree = config.pca_eigenvalues();
    if (const auto rc = emit_to_destination(
            config, "pca", [&](std::ostream& os, OutputFormat fmt) {
                if (scree) emit_scree(os, fmt, result);
                else emit_coords(os, fmt, result);
            })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
