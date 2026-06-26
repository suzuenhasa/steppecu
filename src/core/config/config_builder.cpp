// src/core/config/config_builder.cpp
//
// ConfigBuilder implementation — the §9 layered merge + the validating build()
// (architecture.md §9; cli-bindings.md §4.5). CUDA-FREE host-pure: no device header,
// no CUDA call, no printf/cout (the failure reason is RETURNED via error_message();
// the app prints it — architecture.md §10).
#include "core/config/config_builder.hpp"

#include "io/genotype_source.hpp"  // io::resolve_genotype_triple (--prefix EIGENSTRAT-family vs PLINK)

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>   // std::getenv
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace steppe::config {

namespace {

// ---- small CUDA-free string helpers (no <regex>, no locale) ------------------

[[nodiscard]] std::string to_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

[[nodiscard]] std::string trim(std::string_view s) {
    const auto is_ws = [](unsigned char c) { return std::isspace(c) != 0; };
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && is_ws(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && is_ws(static_cast<unsigned char>(s[e - 1]))) --e;
    return std::string{s.substr(b, e - b)};
}

// Split a comma-separated list, trimming each token; empty tokens are dropped
// (so a trailing comma or "0,,1" does not yield a spurious empty ordinal/label).
[[nodiscard]] std::vector<std::string> split_csv(std::string_view s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::string_view tok =
            comma == std::string_view::npos ? s.substr(start) : s.substr(start, comma - start);
        std::string t = trim(tok);
        if (!t.empty()) out.push_back(std::move(t));
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return out;
}

// Parse a base-10 int from a fully-consumed token (no trailing junk). Returns false
// on any non-numeric / partial / overflowing input — the fail-fast input gate.
[[nodiscard]] bool parse_int(std::string_view tok, int& out) {
    const std::string t = trim(tok);
    if (t.empty()) return false;
    int v = 0;
    const char* first = t.data();
    const char* last = t.data() + t.size();
    const auto [ptr, ec] = std::from_chars(first, last, v);
    if (ec != std::errc{} || ptr != last) return false;
    out = v;
    return true;
}

// An env var as an optional (unset OR empty ⇒ nullopt, so STEPPE_FOO= is "unset").
[[nodiscard]] std::optional<std::string> env(const char* key) {
    const char* v = std::getenv(key);
    if (v == nullptr) return std::nullopt;
    std::string s{v};
    if (s.empty()) return std::nullopt;
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layer folds (precedence: compiled defaults < TOML < env STEPPE_* < CLI).
// merged_ accumulates the RAW (still-string) state; build() parses it ONCE.
// ---------------------------------------------------------------------------

ConfigBuilder& ConfigBuilder::with_defaults() {
    // The compiled defaults are the std-struct defaults already baked into RunConfig
    // (DeviceConfig{}, QpAdmOptions{}, FilterConfig{}, format=csv, 5 cM). At the RAW
    // layer that is simply "nothing set" — every CliArgs optional stays nullopt, so
    // build() falls through to the struct defaults. Reset merged_ so a reused builder
    // starts clean (idempotent seeding).
    merged_ = CliArgs{};
    toml_path_.clear();
    toml_requested_ = false;
    return *this;
}

ConfigBuilder& ConfigBuilder::merge_file(const std::filesystem::path& path) {
    // Record the requested TOML path (BELOW env + CLI). M(cli-0) has no TOML parser
    // compiled in; build() rejects a non-empty path rather than silently dropping it,
    // so a user who passes --config gets a clear error, not a confusing no-op. An
    // empty path is a genuine no-op (no file requested).
    if (!path.empty()) {
        toml_path_ = path;
        toml_requested_ = true;
    }
    return *this;
}

ConfigBuilder& ConfigBuilder::merge_env() {
    // STEPPE_* env layer (ABOVE TOML, BELOW CLI). Only override a field the env SETS;
    // an unset/empty var leaves the lower layer (here: the defaults / TOML) intact.
    if (auto v = env("STEPPE_DEVICE"))    merged_.device = std::move(v);
    if (auto v = env("STEPPE_PRECISION")) merged_.precision = std::move(v);
    if (auto v = env("STEPPE_FORMAT"))    merged_.format = std::move(v);
    if (auto v = env("STEPPE_F2_DIR"))    merged_.f2_dir = std::move(v);
    if (auto v = env("STEPPE_CONFIG"))    { toml_path_ = *v; toml_requested_ = true; }
    return *this;
}

ConfigBuilder& ConfigBuilder::merge_cli(const CliArgs& args) {
    // CLI layer (HIGHEST precedence). Each SET optional overrides the lower layer;
    // each non-empty list overrides. command always carries (the parser sets it).
    merged_.command = args.command;

    const auto take = [](std::optional<std::string>& dst, const std::optional<std::string>& src) {
        if (src.has_value()) dst = src;
    };
    take(merged_.device,      args.device);
    take(merged_.precision,   args.precision);
    take(merged_.f2_dir,      args.f2_dir);
    take(merged_.target,      args.target);
    take(merged_.out_file,    args.out_file);
    take(merged_.format,      args.format);
    take(merged_.prefix,      args.prefix);
    take(merged_.qpdstat_prefix, args.qpdstat_prefix);
    take(merged_.graph,       args.graph);
    take(merged_.geno,        args.geno);
    take(merged_.snp,         args.snp);
    take(merged_.ind,         args.ind);
    take(merged_.out_dir,     args.out_dir);
    take(merged_.tier,        args.tier);

    if (!args.left.empty())  merged_.left  = args.left;
    if (!args.right.empty()) merged_.right = args.right;
    if (!args.pool.empty())  merged_.pool  = args.pool;
    if (!args.pops.empty())  merged_.pops  = args.pops;
    if (!args.pop1.empty())  merged_.pop1  = args.pop1;
    if (!args.pop2.empty())  merged_.pop2  = args.pop2;
    if (!args.pop3.empty())  merged_.pop3  = args.pop3;
    if (!args.pop4.empty())  merged_.pop4  = args.pop4;
    if (!args.pop5.empty())  merged_.pop5  = args.pop5;

    // Scalar option overrides (numeric / bool sentinels).
    const auto take_d = [](std::optional<double>& d, const std::optional<double>& s) { if (s) d = s; };
    const auto take_i = [](std::optional<int>& d, const std::optional<int>& s) { if (s) d = s; };
    const auto take_b = [](std::optional<bool>& d, const std::optional<bool>& s) { if (s) d = s; };
    take_d(merged_.fudge, args.fudge);
    take_i(merged_.als_iterations, args.als_iterations);
    take_i(merged_.rank, args.rank);
    take_d(merged_.rank_alpha, args.rank_alpha);
    take_b(merged_.allow_negative_weights, args.allow_negative_weights);
    take_i(merged_.jackknife, args.jackknife);
    take_d(merged_.p_se_threshold, args.p_se_threshold);
    take_b(merged_.se_require_p, args.se_require_p);
    take_i(merged_.min_sources, args.min_sources);
    take_i(merged_.max_sources, args.max_sources);
    take_d(merged_.sweep_min_z, args.sweep_min_z);
    take_i(merged_.sweep_top_k, args.sweep_top_k);
    take_b(merged_.sweep_sure, args.sweep_sure);
    take_b(merged_.sweep_all_combinations, args.sweep_all_combinations);
    take(merged_.shard_dir, args.shard_dir);
    take_i(merged_.auto_top_k, args.auto_top_k);
    take_i(merged_.min_n, args.min_n);
    take_d(merged_.blgsize, args.blgsize);
    take_d(merged_.maf, args.maf);
    take_d(merged_.geno_max_missing, args.geno_max_missing);
    take_d(merged_.mind_max_missing, args.mind_max_missing);
    take_b(merged_.autosomes_only, args.autosomes_only);
    take_b(merged_.drop_monomorphic, args.drop_monomorphic);
    take_b(merged_.transversions_only, args.transversions_only);
    take_b(merged_.dry_run, args.dry_run);
    take_b(merged_.hash_source, args.hash_source);
    take_i(merged_.numstart, args.numstart);
    take_d(merged_.diag_f3, args.diag_f3);
    take_b(merged_.constrained, args.constrained);
    take_i(merged_.max_nadmix, args.max_nadmix);
    if (args.ploidy.has_value()) merged_.ploidy = args.ploidy;

    // A --config on the CLI is the highest-precedence TOML request.
    if (args.config_path.has_value() && !args.config_path->empty()) {
        toml_path_ = *args.config_path;
        toml_requested_ = true;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// build() — validate ONCE + freeze. Returns InvalidConfig (with a reason in
// error_message_) on any static violation; the LIVE device/VRAM probe is the app's
// job through build_resources (cli-bindings.md §4.5), not this CUDA-free layer.
// ---------------------------------------------------------------------------

BuildResult<RunConfig> ConfigBuilder::build() const {
    error_message_.clear();
    const auto fail = [this](std::string msg) -> BuildResult<RunConfig> {
        error_message_ = std::move(msg);
        return unexpected(Status::InvalidConfig);
    };

    RunConfig cfg;
    cfg.command_ = merged_.command;

    // ---- TOML: no parser compiled in this milestone -> a requested file is a fault.
    if (toml_requested_) {
        return fail("config file '" + toml_path_.string() +
                    "' requested via --config/STEPPE_CONFIG, but TOML config is not "
                    "supported in this build (no TOML parser is compiled in yet)");
    }

    // ---- --device -> DeviceConfig::devices (GPU-ONLY; cli-bindings.md §5.4) -------
    if (merged_.device.has_value()) {
        const std::string raw = trim(*merged_.device);
        const std::string lower = to_lower(raw);
        if (lower == "cpu") {
            return fail("--device cpu is not supported: steppe is a GPU product "
                        "(the CpuBackend is a dev/test oracle only). Use --device "
                        "auto | <ordinal> | <ordinal>,<ordinal> (cli-bindings.md §5.4)");
        }
        if (!raw.empty() && lower != "auto") {
            const std::vector<std::string> toks = split_csv(raw);
            if (toks.empty()) {
                return fail("--device '" + raw + "' parsed to no ordinals "
                            "(use auto | <ordinal> | <ordinal>,<ordinal>)");
            }
            std::set<int> seen;
            std::vector<int> ordinals;
            ordinals.reserve(toks.size());
            for (const std::string& t : toks) {
                int ord = 0;
                if (!parse_int(t, ord)) {
                    return fail("--device ordinal '" + t + "' is not an integer "
                                "(use auto | <ordinal> | <ordinal>,<ordinal>)");
                }
                if (ord < 0) {
                    return fail("--device ordinal '" + t + "' is negative "
                                "(CUDA device ordinals are >= 0)");
                }
                if (!seen.insert(ord).second) {
                    return fail("--device ordinal " + std::to_string(ord) +
                                " is duplicated (the device set/order must be distinct, "
                                "architecture.md §11.4/§12)");
                }
                ordinals.push_back(ord);
            }
            cfg.device_.devices = std::move(ordinals);
        }
        // "auto"/"" -> leave devices empty (auto-enumerate; §9 DeviceConfig contract).
    }

    // ---- --precision -> Precision (cli-bindings.md §4.1) -------------------------
    if (merged_.precision.has_value()) {
        const std::string p = to_lower(trim(*merged_.precision));
        if (p == "emu40" || p == "emu") {
            cfg.device_.precision = Precision{Precision::Kind::EmulatedFp64, 40};
        } else if (p == "emu32") {
            cfg.device_.precision = Precision{Precision::Kind::EmulatedFp64, 32};
        } else if (p == "fp64") {
            cfg.device_.precision = Precision{Precision::Kind::Fp64, kDefaultMantissaBits};
        } else if (p == "tf32") {
            cfg.device_.precision = Precision{Precision::Kind::Tf32, kDefaultMantissaBits};
        } else {
            return fail("--precision '" + *merged_.precision +
                        "' is unknown (use emu40 | emu32 | fp64 | tf32)");
        }
    }

    // ---- --tier auto|resident|host|disk -> DeviceConfig::force_tier (M5) ---------
    // The f2_blocks OUTPUT-tier override (cli-bindings.md §4.1). This is the SINGLE,
    // VALIDATED merge site that sets DeviceConfig::force_tier (it had no setter before —
    // the field defaulted to Auto and only the STEPPE_FORCE_TIER env / the test-set
    // config field reached resolve_output_tier). Mapped here so the precedence is the
    // §9 chain: a --tier on the CLI wins, else force_tier stays Auto and
    // resolve_output_tier falls back to the STEPPE_FORCE_TIER env, then the automatic
    // select_output_tier policy. The CLI token spellings (resident/host/disk) mirror the
    // STEPPE_FORCE_TIER env tokens (tier_select.hpp kForceTierToken*); they are repeated
    // here as plain literals to keep this CUDA-free config layer free of the device
    // header (architecture.md §4). An unknown token is InvalidConfig (never coerced).
    if (merged_.tier.has_value()) {
        const std::string t = to_lower(trim(*merged_.tier));
        if (t == "auto") {
            cfg.device_.force_tier = DeviceConfig::ForceTier::Auto;
        } else if (t == "resident") {
            cfg.device_.force_tier = DeviceConfig::ForceTier::Resident;
        } else if (t == "host") {
            cfg.device_.force_tier = DeviceConfig::ForceTier::HostRam;
        } else if (t == "disk") {
            cfg.device_.force_tier = DeviceConfig::ForceTier::Disk;
        } else {
            return fail("--tier '" + *merged_.tier +
                        "' is unknown (use auto | resident | host | disk)");
        }
    }

    // ---- --format csv|tsv|json (cli-bindings.md §4.4) ---------------------------
    if (merged_.format.has_value()) {
        const std::string f = to_lower(trim(*merged_.format));
        if (f != "csv" && f != "tsv" && f != "json") {
            return fail("--format '" + *merged_.format + "' is unknown (use csv | tsv | json)");
        }
        cfg.format_ = f;
    }

    // ---- QpAdmOptions overrides + range checks ----------------------------------
    QpAdmOptions& opt = cfg.qpadm_options_;
    if (merged_.fudge.has_value()) {
        if (*merged_.fudge < 0.0) return fail("--fudge must be >= 0");
        opt.fudge = *merged_.fudge;
    }
    if (merged_.als_iterations.has_value()) {
        if (*merged_.als_iterations < 1) return fail("--als-iters must be >= 1");
        opt.als_iterations = *merged_.als_iterations;
    }
    if (merged_.rank.has_value()) {
        // -1 ⇒ default (nl-1); any other negative is illegal.
        if (*merged_.rank < -1) return fail("--rank must be -1 (auto) or >= 0");
        opt.rank = *merged_.rank;
    }
    if (merged_.rank_alpha.has_value()) {
        if (*merged_.rank_alpha <= 0.0 || *merged_.rank_alpha >= 1.0) {
            return fail("--rank-alpha must lie in (0, 1)");
        }
        opt.rank_alpha = *merged_.rank_alpha;
    }
    if (merged_.allow_negative_weights.has_value()) {
        opt.allow_negative_weights = *merged_.allow_negative_weights;
    }
    if (merged_.jackknife.has_value()) {
        const int j = *merged_.jackknife;
        if (j < 0 || j > 2) return fail("--jackknife must be 0 (none), 1 (feasible-only), or 2 (all)");
        opt.jackknife = static_cast<JackknifePolicy>(j);
    }
    if (merged_.p_se_threshold.has_value()) {
        if (*merged_.p_se_threshold < 0.0 || *merged_.p_se_threshold > 1.0) {
            return fail("--p-se-threshold must lie in [0, 1]");
        }
        opt.p_se_threshold = *merged_.p_se_threshold;
    }
    if (merged_.se_require_p.has_value()) opt.se_require_p = *merged_.se_require_p;

    // ---- FilterConfig overrides + range checks (extract-f2; M(cli-4)) ------------
    FilterConfig& flt = cfg.filter_;
    if (merged_.maf.has_value()) {
        if (*merged_.maf < 0.0 || *merged_.maf > 0.5) return fail("--maf must lie in [0, 0.5]");
        flt.maf_min = *merged_.maf;
    }
    if (merged_.geno_max_missing.has_value()) {
        if (*merged_.geno_max_missing < 0.0 || *merged_.geno_max_missing > 1.0) {
            return fail("--geno-max-miss must lie in [0, 1]");
        }
        flt.geno_max_missing = *merged_.geno_max_missing;
    }
    if (merged_.mind_max_missing.has_value()) {
        if (*merged_.mind_max_missing < 0.0 || *merged_.mind_max_missing > 1.0) {
            return fail("--mind-max-miss must lie in [0, 1]");
        }
        flt.mind_max_missing = *merged_.mind_max_missing;
    }
    // autosomes_only: extract-f2 defaults this ON (AT2 extract_f2 restricts to autosomes
    // 1..22 by default; cli_args.hpp "extract-f2 default ON, AT2 parity"). An explicit
    // --auto-only/--no-auto-only overrides. Other commands keep the struct default (off).
    //
    // drop_monomorphic: extract-f2 ALSO defaults this ON (AT2 extract_f2 builds f2 on the
    // POLYMORPHIC subset only — its `poly_only` default — dropping every SNP monomorphic
    // across the analysis pop set before the block partition). steppe matches that by
    // dropping monomorphic SNPs by default in extract-f2, which changes the kept-SNP count
    // and the per-block SNP counts (hence the jackknife SEs) to AT2 parity. A monomorphic
    // SNP contributes 0 to every f2 difference, so this never moves the f2 point estimates;
    // it only aligns the SNP set / block partition with AT2. An explicit
    // --drop-mono / --no-drop-mono overrides; other commands keep the struct default (off).
    if (merged_.command == Command::ExtractF2) {
        flt.autosomes_only = true;
        flt.drop_monomorphic = true;
    }
    if (merged_.autosomes_only.has_value())    flt.autosomes_only = *merged_.autosomes_only;
    if (merged_.drop_monomorphic.has_value())  flt.drop_monomorphic = *merged_.drop_monomorphic;
    if (merged_.transversions_only.has_value()) flt.transversions_only = *merged_.transversions_only;

    // ---- PopSelection (extract-f2; M(cli-4)) — the three modes are mutually
    // exclusive. Default stays AutoTopK{k=0} (the "no selection requested" state the
    // app surfaces as a fail-fast when extract-f2 actually needs one).
    {
        int modes_set = 0;
        if (!merged_.pops.empty())           ++modes_set;
        if (merged_.auto_top_k.has_value())  ++modes_set;
        if (merged_.min_n.has_value())       ++modes_set;
        if (modes_set > 1) {
            return fail("--pops, --auto-top-k, and --min-n are mutually exclusive "
                        "(choose one population-selection mode)");
        }
        if (!merged_.pops.empty()) {
            cfg.pop_selection_.mode = io::PopSelection::Mode::Explicit;
            cfg.pop_selection_.labels = merged_.pops;
        } else if (merged_.auto_top_k.has_value()) {
            if (*merged_.auto_top_k < 1) return fail("--auto-top-k must be >= 1");
            cfg.pop_selection_.mode = io::PopSelection::Mode::AutoTopK;
            cfg.pop_selection_.k = static_cast<std::size_t>(*merged_.auto_top_k);
        } else if (merged_.min_n.has_value()) {
            if (*merged_.min_n < 1) return fail("--min-n must be >= 1");
            cfg.pop_selection_.mode = io::PopSelection::Mode::MinN;
            cfg.pop_selection_.min_n = static_cast<std::size_t>(*merged_.min_n);
        }
    }

    // ---- blgsize (MORGANS on the CLI, AT2 convention) ---------------------------
    // ADMIXTOOLS 2's `blgsize` is in MORGANS (its default 0.05 == 5 cM), so the
    // steppe `--blgsize` flag speaks Morgans too — a bare `--blgsize 0.05` reproduces
    // AT2's block partition (same physical 5 cM block width). The RunConfig stores cM
    // (the kDefaultBlockSizeCm convention; the block math converts cM->Morgans at the
    // single block_size_cm_to_morgans site), so the Morgans->cM conversion lives HERE,
    // at the CLI/config seam, via the single kCentimorgansPerMorgan constant. Do not
    // reinterpret the stored field as Morgans — only the flag's input unit changed.
    if (merged_.blgsize.has_value()) {
        if (!(*merged_.blgsize > 0.0)) return fail("--blgsize must be > 0 (Morgans)");
        cfg.blgsize_cm_ = *merged_.blgsize * kCentimorgansPerMorgan;
    }

    // ---- ploidy policy (extract-f2; the f2 pseudo-haploid fix) ------------------
    // Default Auto (= AT2 adjust_pseudohaploid=TRUE per-sample detection); --ploidy
    // 1/2 force a uniform pseudo-haploid/diploid ploidy. The detection/forcing
    // happens in cmd_extract_f2 against the gathered tile (the genotypes).
    if (merged_.ploidy.has_value()) cfg.ploidy_ = *merged_.ploidy;

    // ---- rotate pool-enumeration bounds ----------------------------------------
    if (merged_.min_sources.has_value()) {
        if (*merged_.min_sources < 1) return fail("--min-sources must be >= 1");
        cfg.min_sources_ = *merged_.min_sources;
    }
    if (merged_.max_sources.has_value()) {
        // -1 ⇒ up to the whole pool; otherwise >= min_sources.
        if (*merged_.max_sources != -1 && *merged_.max_sources < 1) {
            return fail("--max-sources must be -1 (whole pool) or >= 1");
        }
        cfg.max_sources_ = *merged_.max_sources;
    }
    if (cfg.max_sources_ != -1 && cfg.max_sources_ < cfg.min_sources_) {
        return fail("--max-sources must be >= --min-sources");
    }

    // ---- f4-sweep / f3-sweep filter knobs --------------------------------------
    if (merged_.sweep_min_z.has_value()) {
        if (*merged_.sweep_min_z < 0.0) return fail("--min-z must be >= 0");
        cfg.sweep_min_z_ = *merged_.sweep_min_z;
    }
    if (merged_.sweep_top_k.has_value()) {
        if (*merged_.sweep_top_k < 1) return fail("--top-k must be >= 1");
        cfg.sweep_top_k_ = *merged_.sweep_top_k;
    }
    if (merged_.sweep_min_z.has_value() && merged_.sweep_top_k.has_value()) {
        return fail("--min-z and --top-k are mutually exclusive (pick one sweep filter)");
    }
    if (merged_.sweep_sure.has_value()) cfg.sweep_sure_ = *merged_.sweep_sure;
    if (merged_.sweep_all_combinations.has_value())
        cfg.sweep_all_combinations_ = *merged_.sweep_all_combinations;
    if (merged_.shard_dir) cfg.shard_dir_ = *merged_.shard_dir;

    // ---- carry the verbatim I/O strings ----------------------------------------
    if (merged_.f2_dir)   cfg.f2_dir_ = *merged_.f2_dir;
    if (merged_.target)   cfg.target_ = *merged_.target;
    cfg.left_  = merged_.left;
    cfg.right_ = merged_.right;
    cfg.pool_  = merged_.pool;
    cfg.pop1_  = merged_.pop1;   // f4 row-aligned quartet columns (--pop1..--pop4)
    cfg.pop2_  = merged_.pop2;
    cfg.pop3_  = merged_.pop3;
    cfg.pop4_  = merged_.pop4;
    cfg.pop5_  = merged_.pop5;   // f4-ratio 5th row-aligned column (--pop5)
    cfg.pops_  = merged_.pops;   // f4 --pops 4-tuple convenience (raw labels, carried verbatim)
    if (merged_.out_file) cfg.out_file_ = *merged_.out_file;
    // qpdstat's --prefix (the Part-B normalized-D genotype prefix) is carried VERBATIM and
    // NOT expanded into geno/snp/ind — it is only the fail-fast sentinel the qpdstat command
    // reads (Part B not yet implemented). Distinct from extract-f2's --prefix below.
    if (merged_.qpdstat_prefix) cfg.qpdstat_prefix_ = *merged_.qpdstat_prefix;
    // --prefix P expands to the genotype triple (cli-bindings.md §4.2). For the
    // EIGENSTRAT family (TGENO/GENO/EIGENSTRAT) that is P.{geno,snp,ind}; for PLINK it is
    // P.{bed,bim,fam} — resolve_genotype_triple chooses the extensions by a filesystem
    // probe (.geno present wins; else a P.bed -> PLINK). The authoritative on-disk format
    // is still pinned later by the GenoReader ctor; this only picks the sibling paths so a
    // PLINK prefix opens .bed/.bim/.fam (M-FR PLINK). An explicit --geno/--snp/--ind
    // OVERRIDES the corresponding prefix-derived path (cli_parse documents --geno overrides
    // --prefix).
    if (merged_.prefix && !merged_.prefix->empty()) {
        const io::GenotypeTriple triple = io::resolve_genotype_triple(*merged_.prefix);
        if (!merged_.geno) cfg.geno_ = triple.geno;
        if (!merged_.snp)  cfg.snp_  = triple.snp;
        if (!merged_.ind)  cfg.ind_  = triple.ind;
    }
    if (merged_.geno)     cfg.geno_ = *merged_.geno;
    if (merged_.snp)      cfg.snp_ = *merged_.snp;
    if (merged_.ind)      cfg.ind_ = *merged_.ind;
    if (merged_.out_dir)  cfg.out_dir_ = *merged_.out_dir;
    if (merged_.dry_run)     cfg.dry_run_ = *merged_.dry_run;
    if (merged_.hash_source) cfg.hash_source_ = *merged_.hash_source;
    // qpgraph (single-graph fit) inputs/options.
    if (merged_.graph)       cfg.graph_file_ = *merged_.graph;
    if (merged_.numstart) {
        if (*merged_.numstart < 1) return fail("--numstart must be >= 1");
        cfg.qpgraph_numstart_ = *merged_.numstart;
    }
    if (merged_.diag_f3)     cfg.qpgraph_diag_f3_ = *merged_.diag_f3;
    if (merged_.constrained) cfg.qpgraph_constrained_ = *merged_.constrained;
    if (merged_.max_nadmix) {
        // v1 supports the {0,1} admixture-node ceiling (run_config.hpp:178).
        if (*merged_.max_nadmix < 0) return fail("--max-nadmix must be >= 0");
        cfg.qpgraph_max_nadmix_ = *merged_.max_nadmix;
    }

    return cfg;
}

}  // namespace steppe::config
