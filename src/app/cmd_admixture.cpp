// src/app/cmd_admixture.cpp
//
// The `steppe admixture` handler. Resolves the triple + selection, builds the mode inputs
// (supervised labels / projection F table), runs steppe::run_admixture on a backend (GPU when
// visible, else the CpuBackend reference oracle), and writes out-dir/{Q.tsv, F.tsv (--emit-F),
// loglik.txt, meta.json}. Mode exclusivity (--supervised XOR --project-onto) and the K/seed
// legality are enforced pre-flight. Mirrors the cmd_pcangsd self-contained shape.
#include "app/cmd_admixture.hpp"

#include <cctype>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "app/exit_code_for_caught.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "io/filter/include_exclude.hpp"
#include "io/genotype_source.hpp"
#include "steppe/admixture.hpp"
#include "steppe/config.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

[[nodiscard]] bool parse_device(const std::string& raw, int& dev, std::string& err) {
    std::string s;
    for (char c : raw)
        if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty() || s == "auto") { dev = 0; return true; }
    try { dev = std::stoi(s); } catch (...) {
        err = "--device ordinal '" + raw + "' is not an integer";
        return false;
    }
    return true;
}

// Read a whitespace-delimited token list file (one label / id per line; blanks ignored).
[[nodiscard]] bool read_token_file(const std::string& path, std::vector<std::string>& out,
                                   std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open file: " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tok;
        if (ls >> tok) out.push_back(tok);
    }
    return true;
}

// Read an F.tsv reference table: M rows, each K numeric columns, optionally preceded by a
// SNP-id first column (auto-detected when the first row has a non-numeric leading field).
[[nodiscard]] bool read_f_table(const std::string& path, std::vector<double>& F, long& Mrows,
                                int& Kcols, std::vector<std::string>& snp_ids, std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open --project-onto file: " + path; return false; }
    F.clear();
    snp_ids.clear();
    Mrows = 0;
    Kcols = -1;
    bool have_id = false;
    bool first = true;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::vector<std::string> toks;
        std::string t;
        while (ls >> t) toks.push_back(t);
        if (toks.empty()) continue;
        if (first) {
            // Detect a leading SNP-id column: the first token fails to parse as a double.
            try { std::size_t p; (void)std::stod(toks[0], &p); if (p != toks[0].size()) throw 0; }
            catch (...) { have_id = true; }
            Kcols = static_cast<int>(toks.size()) - (have_id ? 1 : 0);
            if (Kcols < 1) { err = "--project-onto file has no numeric columns"; return false; }
            first = false;
        }
        const int expect = Kcols + (have_id ? 1 : 0);
        if (static_cast<int>(toks.size()) != expect) {
            err = "--project-onto file has a ragged row (expected " + std::to_string(expect) +
                  " fields)";
            return false;
        }
        int c0 = 0;
        if (have_id) { snp_ids.push_back(toks[0]); c0 = 1; }
        for (int c = 0; c < Kcols; ++c) {
            try { F.push_back(std::stod(toks[static_cast<std::size_t>(c0 + c)])); }
            catch (...) { err = "--project-onto file has a non-numeric value"; return false; }
        }
        ++Mrows;
    }
    if (Mrows == 0) { err = "--project-onto file is empty"; return false; }
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

[[nodiscard]] bool write_q_table(const std::string& path, const steppe::AdmixtureResult& r,
                                 char sep) {
    std::ofstream os(path, std::ios::trunc);
    if (!os) return false;
    os << "sample" << sep << "pop";
    for (int k = 0; k < r.K; ++k) os << sep << "K" << (k + 1);
    os << "\n";
    for (int i = 0; i < r.N; ++i) {
        os << csv_field(r.sample_id[static_cast<std::size_t>(i)], sep) << sep
           << csv_field(r.sample_pop[static_cast<std::size_t>(i)], sep);
        for (int k = 0; k < r.K; ++k)
            os << sep
               << fmt_double(r.Q[static_cast<std::size_t>(i) * static_cast<std::size_t>(r.K) +
                                 static_cast<std::size_t>(k)]);
        os << "\n";
    }
    os.flush();
    return os.good();
}

[[nodiscard]] bool write_f_table(const std::string& path, const steppe::AdmixtureResult& r,
                                 char sep) {
    std::ofstream os(path, std::ios::trunc);
    if (!os) return false;
    for (long s = 0; s < r.M; ++s) {
        for (int k = 0; k < r.K; ++k) {
            if (k) os << sep;
            os << fmt_double(r.F[static_cast<std::size_t>(s) * static_cast<std::size_t>(r.K) +
                                 static_cast<std::size_t>(k)]);
        }
        os << "\n";
    }
    os.flush();
    return os.good();
}

const char* mode_name(steppe::AdmixtureMode m) {
    switch (m) {
        case steppe::AdmixtureMode::Unsupervised: return "unsupervised";
        case steppe::AdmixtureMode::Supervised: return "supervised";
        case steppe::AdmixtureMode::Projection: return "projection";
    }
    return "unknown";
}

}  // namespace

int run_admixture_cmd(const AdmixtureArgs& args) {
    if (args.prefix.empty()) {
        std::fprintf(stderr, "steppe admixture: --prefix PREFIX.{geno,snp,ind} is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.out_dir.empty()) {
        std::fprintf(stderr, "steppe admixture: --out-dir <dir> is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.precision != "emu" && args.precision != "fp64") {
        std::fprintf(stderr, "steppe admixture: --precision must be emu | fp64 (got '%s')\n",
                     args.precision.c_str());
        return cfg::kExitInvalidConfig;
    }
    if (args.accel != "squarem" && args.accel != "em") {
        std::fprintf(stderr, "steppe admixture: --accel must be squarem | em (got '%s')\n",
                     args.accel.c_str());
        return cfg::kExitInvalidConfig;
    }
    const bool supervised = !args.supervised.empty();
    const bool projection = !args.project_onto.empty();
    if (supervised && projection) {
        std::fprintf(stderr,
                     "steppe admixture: --supervised and --project-onto are mutually exclusive\n");
        return cfg::kExitInvalidConfig;
    }

    steppe::AdmixtureParams params;
    steppe::AdmixtureInputs inputs;
    params.seed = args.seed;
    params.seeds = args.seeds < 1 ? 1 : args.seeds;
    params.max_iter = args.max_iter;
    params.tol = args.tol;
    params.init = (args.init == "svd") ? steppe::AdmixtureInit::Svd : steppe::AdmixtureInit::Random;
    params.accel =
        (args.accel == "em") ? steppe::AdmixtureAccel::Em : steppe::AdmixtureAccel::Squarem;
    params.precision =
        (args.precision == "fp64") ? steppe::Precision::fp64() : steppe::Precision::emulated_fp64();

    // Per-SNP QC filter (Phase-0 common-variant front-end). Strand-ambiguous SNPs are kept.
    if (args.maf < 0.0 || args.maf > 0.5) {
        std::fprintf(stderr, "steppe admixture: --maf must lie in [0, 0.5] (got %g)\n", args.maf);
        return cfg::kExitInvalidConfig;
    }
    if (args.geno_max_miss < 0.0 || args.geno_max_miss > 1.0) {
        std::fprintf(stderr, "steppe admixture: --geno-max-miss must lie in [0, 1] (got %g)\n",
                     args.geno_max_miss);
        return cfg::kExitInvalidConfig;
    }
    params.filter.maf_min = args.maf;
    params.filter.geno_max_missing = args.geno_max_miss;
    params.filter.drop_monomorphic = args.drop_mono;
    params.filter.autosomes_only = args.autosomes_only;
    params.filter.strand_mode = steppe::StrandMode::Keep;
    params.filter.allow_mixed_ascertainment = args.allow_mixed_ascertainment;
    if (!args.keep_snps.empty()) params.filter.prune_in_path = args.keep_snps;
    if (!args.exclude_snps.empty()) {
        try {
            steppe::io::filter::read_snp_id_list(args.exclude_snps, params.filter.exclude_snp_ids);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe admixture: --exclude-snps: %s\n", e.what());
            return cfg::kExitInvalidConfig;
        }
    }
    if (!args.ld_prune.empty()) {
        std::string lderr;
        if (!steppe::io::filter::parse_ld_prune_spec(args.ld_prune, params.filter, lderr)) {
            std::fprintf(stderr, "steppe admixture: %s\n", lderr.c_str());
            return cfg::kExitInvalidConfig;
        }
    }

    if (projection) {
        params.mode = steppe::AdmixtureMode::Projection;
        std::string err;
        long Mrows = 0;
        int Kcols = 0;
        if (!read_f_table(args.project_onto, inputs.fixed_F, Mrows, Kcols, inputs.fixed_F_snps,
                          err)) {
            std::fprintf(stderr, "steppe admixture: %s\n", err.c_str());
            return cfg::kExitInvalidConfig;
        }
        inputs.fixed_F_M = Mrows;
        inputs.fixed_F_K = Kcols;
        params.K = Kcols;  // K comes from F's columns; -K guessing forbidden
    } else if (supervised) {
        params.mode = steppe::AdmixtureMode::Supervised;
        std::string err;
        if (!read_token_file(args.supervised, inputs.supervised_labels, err) ||
            inputs.supervised_labels.empty()) {
            std::fprintf(stderr, "steppe admixture: cannot read --supervised labels: %s\n",
                         err.c_str());
            return cfg::kExitInvalidConfig;
        }
        params.K = static_cast<int>(inputs.supervised_labels.size());
    } else {
        params.mode = steppe::AdmixtureMode::Unsupervised;
        if (args.K < 1) {
            std::fprintf(stderr,
                         "steppe admixture: -K >= 1 is required for unsupervised mode\n");
            return cfg::kExitInvalidConfig;
        }
        params.K = args.K;
    }

    // Population selection file (one label per line; empty = all).
    std::vector<std::string> pops;
    if (!args.pops.empty()) {
        std::string err;
        if (!read_token_file(args.pops, pops, err)) {
            std::fprintf(stderr, "steppe admixture: %s\n", err.c_str());
            return cfg::kExitInvalidConfig;
        }
    }

    const io::GenotypeTriple triple = io::resolve_genotype_triple(args.prefix);
    const char sep = (args.format == "csv") ? ',' : '\t';

    // Backend: GPU when visible, else the CpuBackend reference oracle.
    std::unique_ptr<ComputeBackend> be;
    try {
        int dev = 0;
        std::string derr;
        if (!parse_device(args.device, dev, derr)) {
            std::fprintf(stderr, "steppe admixture: %s\n", derr.c_str());
            return cfg::kExitInvalidConfig;
        }
        be = (device::visible_device_count() > 0) ? device::make_cuda_backend(dev)
                                                   : device::make_cpu_backend();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe admixture: device init failed: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    steppe::AdmixtureResult r;
    try {
        r = steppe::run_admixture(triple.geno, triple.snp, triple.ind, pops, params, inputs, *be);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe admixture: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }
    if (r.status != Status::Ok) {
        std::fprintf(stderr,
                     "steppe admixture: %s (check --prefix, --pops, -K/-project-onto, and that "
                     "K < samples)\n",
                     status_text(r.status));
        return cfg::exit_code_for(r.status);
    }

    // Write outputs into --out-dir.
    std::error_code ec;
    std::filesystem::create_directories(args.out_dir, ec);
    const std::string base = args.out_dir + "/";
    if (!args.emit_kept_snps.empty()) {
        std::ofstream ks(args.emit_kept_snps, std::ios::trunc);
        for (const std::string& id : r.kept_snp_ids) ks << id << "\n";
        ks.flush();
        if (!ks.good()) {
            std::fprintf(stderr, "steppe admixture: failed to write --emit-kept-snps file: %s\n",
                         args.emit_kept_snps.c_str());
            return cfg::kExitIoError;
        }
    }
    if (!write_q_table(base + "Q.tsv", r, sep)) {
        std::fprintf(stderr, "steppe admixture: failed to write Q.tsv\n");
        return cfg::kExitIoError;
    }
    if (args.emit_F) {
        if (!write_f_table(base + "F.tsv", r, sep)) {
            std::fprintf(stderr, "steppe admixture: failed to write F.tsv\n");
            return cfg::kExitIoError;
        }
    }
    {
        std::ofstream os(base + "loglik.txt", std::ios::trunc);
        os << "seed_index\tloglik\titers\tconverged\twinner\n";
        for (std::size_t si = 0; si < r.seed_loglik.size(); ++si)
            os << si << "\t" << fmt_double(r.seed_loglik[si]) << "\t" << r.seed_iters[si] << "\t"
               << (r.seed_converged[si] ? 1 : 0) << "\t"
               << (static_cast<int>(si) == r.best_seed ? 1 : 0) << "\n";
        if (!os.good()) {
            std::fprintf(stderr, "steppe admixture: failed to write loglik.txt\n");
            return cfg::kExitIoError;
        }
    }
    {
        std::ofstream os(base + "meta.json", std::ios::trunc);
        os << "{\n";
        os << "  \"mode\": \"" << mode_name(r.mode) << "\",\n";
        os << "  \"K\": " << r.K << ",\n";
        os << "  \"seed\": " << args.seed << ",\n";
        os << "  \"seeds\": " << params.seeds << ",\n";
        os << "  \"init\": \"" << args.init << "\",\n";
        os << "  \"accel\": \"" << args.accel << "\",\n";
        os << "  \"max_iter\": " << args.max_iter << ",\n";
        os << "  \"tol\": " << fmt_double(args.tol) << ",\n";
        os << "  \"precision\": \"" << args.precision << "\",\n";
        os << "  \"n_indiv\": " << r.N << ",\n";
        os << "  \"n_snp\": " << r.M << ",\n";
        os << "  \"best_seed\": " << r.best_seed << ",\n";
        os << "  \"best_loglik\": " << fmt_double(r.best_loglik) << ",\n";
        os << "  \"iters_run\": " << r.iters_run << ",\n";
        os << "  \"base_map_evals\": " << r.base_map_evals << ",\n";
        os << "  \"converged\": " << (r.converged ? "true" : "false") << "\n";
        os << "}\n";
        if (!os.good()) {
            std::fprintf(stderr, "steppe admixture: failed to write meta.json\n");
            return cfg::kExitIoError;
        }
    }

    std::fprintf(stderr,
                 "steppe admixture: %s mode (accel=%s), %d samples x %ld SNPs, K=%d; %s after %d %s "
                 "(%d base-EM-map evals; loglik=%s, best seed %d/%d) -> %sQ.tsv%s\n",
                 mode_name(r.mode), args.accel.c_str(), r.N, r.n_snp_total, r.K,
                 r.converged ? "converged" : "stopped at --max-iter", r.iters_run,
                 (args.accel == "em") ? "EM iters" : "accel steps", r.base_map_evals,
                 fmt_double(r.best_loglik).c_str(), r.best_seed, params.seeds, base.c_str(),
                 args.emit_F ? " F.tsv" : "");
    return cfg::kExitOk;
}

}  // namespace steppe::app
