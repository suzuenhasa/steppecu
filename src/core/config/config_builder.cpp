// src/core/config/config_builder.cpp
//
// ConfigBuilder implementation: folds the config layers (defaults < TOML < env <
// CLI) into raw string state, then build() parses, range-checks, and freezes it
// once into an immutable RunConfig. Host-pure and CUDA-free — a failure is
// returned via error_message(), never printed.
//
// Reference: docs/reference/src_core_config_config_builder.cpp.md
#include "core/config/config_builder.hpp"

#include "core/internal/index_cast.hpp"
#include "io/filter/include_exclude.hpp"
#include "io/genotype_source.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace steppe::config {

namespace {

// String-parsing helpers — reference §3

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

// Case-folded token -> enum lookup over a fixed {token, value} table. The token
// is expected already to_lower(trim())'d; per-flag help text stays at the call
// site. Reference §3
template <typename E>
[[nodiscard]] std::optional<E> parse_enum(
    std::string_view token, std::initializer_list<std::pair<std::string_view, E>> table) {
    for (const auto& [name, value] : table) {
        if (token == name) return value;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> env(const char* key) {
    const char* v = std::getenv(key);
    if (v == nullptr) return std::nullopt;
    std::string s{v};
    if (s.empty()) return std::nullopt;
    return s;
}

}  // namespace

// Layer folds: defaults < TOML < env < CLI — reference §2
ConfigBuilder& ConfigBuilder::with_defaults() {
    merged_ = CliArgs{};
    toml_path_.clear();
    toml_requested_ = false;
    return *this;
}

ConfigBuilder& ConfigBuilder::merge_file(const std::filesystem::path& path) {
    if (!path.empty()) {
        toml_path_ = path;
        toml_requested_ = true;
    }
    return *this;
}

ConfigBuilder& ConfigBuilder::merge_env() {
    if (auto v = env("STEPPE_DEVICE"))    merged_.device = std::move(v);
    if (auto v = env("STEPPE_PRECISION")) merged_.precision = std::move(v);
    if (auto v = env("STEPPE_FORMAT"))    merged_.format = std::move(v);
    if (auto v = env("STEPPE_F2_DIR"))    merged_.f2_dir = std::move(v);
    if (auto v = env("STEPPE_CONFIG"))    { toml_path_ = *v; toml_requested_ = true; }
    return *this;
}

ConfigBuilder& ConfigBuilder::merge_cli(const CliArgs& args) {
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
    take(merged_.samples, args.samples);
    take(merged_.norm, args.norm);
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
    if (!args.project_pops.empty()) merged_.project_pops = args.project_pops;
    if (!args.pop1.empty())  merged_.pop1  = args.pop1;
    if (!args.pop2.empty())  merged_.pop2  = args.pop2;
    if (!args.pop3.empty())  merged_.pop3  = args.pop3;
    if (!args.pop4.empty())  merged_.pop4  = args.pop4;
    if (!args.pop5.empty())  merged_.pop5  = args.pop5;

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
    take_d(merged_.scan_p_min, args.scan_p_min);
    take_b(merged_.scan_allow_clade, args.scan_allow_clade);
    take(merged_.scan_strategy, args.scan_strategy);
    take_i(merged_.scan_beam_width, args.scan_beam_width);
    if (!args.scan_base.empty()) merged_.scan_base = args.scan_base;
    take_b(merged_.scan_prerank, args.scan_prerank);
    take_b(merged_.scan_suggest_swaps, args.scan_suggest_swaps);
    take(merged_.scan_right_search, args.scan_right_search);
    if (!args.scan_right_pool.empty()) merged_.scan_right_pool = args.scan_right_pool;
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
    take(merged_.strand_mode, args.strand_mode);
    take(merged_.keep_snps, args.keep_snps);
    take(merged_.exclude_snps, args.exclude_snps);
    take_b(merged_.allow_mixed_ascertainment, args.allow_mixed_ascertainment);
    take(merged_.emit_kept_snps, args.emit_kept_snps);
    take(merged_.ld_prune, args.ld_prune);
    take_b(merged_.dry_run, args.dry_run);
    take_b(merged_.hash_source, args.hash_source);
    take_i(merged_.numstart, args.numstart);
    take_d(merged_.diag_f3, args.diag_f3);
    take_b(merged_.constrained, args.constrained);
    take_i(merged_.max_nadmix, args.max_nadmix);
    take_i(merged_.window_snps, args.window_snps);
    take_d(merged_.min_overlap, args.min_overlap);
    take(merged_.donors, args.donors);
    take(merged_.labels, args.labels);
    take(merged_.face, args.face);
    take(merged_.ls_theta, args.ls_theta);
    take_d(merged_.ls_ne, args.ls_ne);
    take_b(merged_.self_copy, args.self_copy);
    take_i(merged_.recip_batch, args.recip_batch);
    take_b(merged_.bp_fallback, args.bp_fallback);
    take_b(merged_.paint_full, args.paint_full);
    take(merged_.recip_vcf, args.recip_vcf);
    take(merged_.donor_vcf, args.donor_vcf);
    take(merged_.vcf_map, args.vcf_map);
    take(merged_.vcf_region, args.vcf_region);
    take_d(merged_.vcf_unphased_max, args.vcf_unphased_max);
    take(merged_.fst_method, args.fst_method);
    take_b(merged_.fst_per_snp, args.fst_per_snp);
    take_b(merged_.fst_all_pairs, args.fst_all_pairs);
    take(merged_.fst_windowed, args.fst_windowed);
    take(merged_.fst_pbs, args.fst_pbs);
    take_b(merged_.kinship_all_pairs, args.kinship_all_pairs);
    take(merged_.pairs, args.pairs);
    take_d(merged_.min_kinship, args.min_kinship);
    take_b(merged_.sfs_fold, args.sfs_fold);
    take_i(merged_.pca_k, args.pca_k);
    take_b(merged_.pca_eigenvalues, args.pca_eigenvalues);
    take(merged_.pca_emit_html, args.pca_emit_html);
    take(merged_.project_samples, args.project_samples);
    take(merged_.project_mode, args.project_mode);
    take(merged_.bgen, args.bgen);
    take(merged_.pca_solver, args.pca_solver);
    if (args.ploidy.has_value()) merged_.ploidy = args.ploidy;

    if (args.config_path.has_value() && !args.config_path->empty()) {
        toml_path_ = *args.config_path;
        toml_requested_ = true;
    }
    return *this;
}

// build(): parse the raw layers once, range-check, and freeze — reference §4-§14
BuildResult<RunConfig> ConfigBuilder::build() {
    error_message_.clear();
    const auto fail = [this](std::string msg) -> BuildResult<RunConfig> {
        error_message_ = std::move(msg);
        return unexpected(Status::InvalidConfig);
    };

    RunConfig cfg;
    cfg.command_ = merged_.command;

    if (toml_requested_) {
        return fail("config file '" + toml_path_.string() +
                    "' requested via --config/STEPPE_CONFIG, but TOML config is not "
                    "supported in this build (no TOML parser is compiled in yet)");
    }

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
    }

    if (merged_.precision.has_value()) {
        const std::string p = to_lower(trim(*merged_.precision));
        if (p == "emu40" || p == "emu" || p == "emulated_fp64") {
            cfg.device_.precision = Precision{Precision::Kind::EmulatedFp64, 40};
        } else if (p == "emu32" || p == "emulated_fp64_32") {
            cfg.device_.precision = Precision{Precision::Kind::EmulatedFp64, 32};
        } else if (p == "fp64" || p == "native") {
            cfg.device_.precision = Precision{Precision::Kind::Fp64, kDefaultMantissaBits};
        } else if (p == "tf32") {
            cfg.device_.precision = Precision{Precision::Kind::Tf32, kDefaultMantissaBits};
        } else {
            return fail("--precision '" + *merged_.precision +
                        "' is unknown (use emu40 | emu32 | fp64 | tf32; aliases "
                        "emu=emu40, emulated_fp64=emu40, emulated_fp64_32=emu32, "
                        "native=fp64)");
        }
    }

    if (merged_.tier.has_value()) {
        const std::string t = to_lower(trim(*merged_.tier));
        const auto tier = parse_enum<DeviceConfig::ForceTier>(
            t, {{"auto", DeviceConfig::ForceTier::Auto},
                {"resident", DeviceConfig::ForceTier::Resident},
                {"host", DeviceConfig::ForceTier::HostRam},
                {"disk", DeviceConfig::ForceTier::Disk}});
        if (!tier) {
            return fail("--tier '" + *merged_.tier +
                        "' is unknown (use auto | resident | host | disk)");
        }
        cfg.device_.force_tier = *tier;
    }

    if (merged_.format.has_value()) {
        const std::string f = to_lower(trim(*merged_.format));
        if (f != "csv" && f != "tsv" && f != "json") {
            return fail("--format '" + *merged_.format + "' is unknown (use csv | tsv | json)");
        }
        cfg.format_ = f;
    }

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

    FilterConfig& flt = cfg.filter_;
    // Common-variant defaults for the standalone population-genetic tools (pca/fst/kinship;
    // gated vs scikit-allel/plink2/PCAngsd, NOT AT2): a MAF floor collapses the rare/monomorphic
    // tail that makes these noisy on modern dense data, and strand-ambiguous SNPs are KEPT
    // (single-file analyses have no cross-panel merge to protect). The explicit --maf /
    // --strand-mode below still win; autosome-only stays opt-in (--auto-only) since the tools
    // already layer their own autosome summary masks.
    if (merged_.command == Command::Pca || merged_.command == Command::Fst ||
        merged_.command == Command::Kinship) {
        flt.maf_min = 0.05;
        flt.strand_mode = StrandMode::Keep;
    }
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
    if (merged_.command == Command::ExtractF2) {
        flt.autosomes_only = true;
        flt.drop_monomorphic = true;
    }
    if (merged_.command == Command::Readv2) {
        flt.autosomes_only = true;  // READv2 convention: exclude sex chromosomes (--no-auto-only off)
    }
    if (merged_.autosomes_only.has_value())    flt.autosomes_only = *merged_.autosomes_only;
    if (merged_.drop_monomorphic.has_value())  flt.drop_monomorphic = *merged_.drop_monomorphic;
    if (merged_.transversions_only.has_value()) flt.transversions_only = *merged_.transversions_only;
    if (merged_.strand_mode.has_value()) {
        const std::string s = to_lower(trim(*merged_.strand_mode));
        const auto mode = parse_enum<StrandMode>(
            s, {{"drop", StrandMode::Drop},
                {"keep", StrandMode::Keep},
                {"flip", StrandMode::Flip}});
        if (!mode) {
            return fail("--strand-mode '" + *merged_.strand_mode +
                        "' is unknown (use drop | keep | flip)");
        }
        flt.strand_mode = *mode;
    }
    // Membership + same-ascertainment guard. --keep-snps reuses the lazy prune.in reader;
    // --exclude-snps is read eagerly into the drop set here.
    if (merged_.keep_snps.has_value() && !merged_.keep_snps->empty()) {
        flt.prune_in_path = *merged_.keep_snps;
    }
    if (merged_.exclude_snps.has_value() && !merged_.exclude_snps->empty()) {
        try {
            io::filter::read_snp_id_list(*merged_.exclude_snps, flt.exclude_snp_ids);
        } catch (const std::exception& e) {
            return fail(std::string("--exclude-snps: ") + e.what());
        }
    }
    if (merged_.allow_mixed_ascertainment.has_value()) {
        flt.allow_mixed_ascertainment = *merged_.allow_mixed_ascertainment;
    }
    if (merged_.emit_kept_snps.has_value()) cfg.emit_kept_snps_ = *merged_.emit_kept_snps;
    // Windowed-r2 LD prune "WIN:STEP:R2" (variant-count window; plink2 --indep-pairwise).
    if (merged_.ld_prune.has_value() && !merged_.ld_prune->empty()) {
        std::string lderr;
        if (!io::filter::parse_ld_prune_spec(*merged_.ld_prune, flt, lderr)) return fail(lderr);
    }

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
            cfg.pop_selection_.k = core::idx(*merged_.auto_top_k);
        } else if (merged_.min_n.has_value()) {
            if (*merged_.min_n < 1) return fail("--min-n must be >= 1");
            cfg.pop_selection_.mode = io::PopSelection::Mode::MinN;
            cfg.pop_selection_.min_n = core::idx(*merged_.min_n);
        }
    }

    if (merged_.blgsize.has_value()) {
        if (!(*merged_.blgsize > 0.0)) return fail("--blgsize must be > 0 (Morgans)");
        cfg.blgsize_cm_ = *merged_.blgsize * kCentimorgansPerMorgan;
    }

    if (merged_.ploidy.has_value()) cfg.ploidy_ = *merged_.ploidy;

    if (merged_.min_sources.has_value()) {
        if (*merged_.min_sources < 1) return fail("--min-sources must be >= 1");
        cfg.min_sources_ = *merged_.min_sources;
    }
    if (merged_.max_sources.has_value()) {
        if (*merged_.max_sources != -1 && *merged_.max_sources < 1) {
            return fail("--max-sources must be -1 (whole pool) or >= 1");
        }
        cfg.max_sources_ = *merged_.max_sources;
    }
    if (merged_.scan_p_min.has_value()) {
        if (*merged_.scan_p_min <= 0.0 || *merged_.scan_p_min >= 1.0) {
            return fail("--p-min must be in (0,1)");
        }
        cfg.scan_p_min_ = *merged_.scan_p_min;
    }
    if (merged_.scan_allow_clade.has_value()) cfg.scan_allow_clade_ = *merged_.scan_allow_clade;
    if (merged_.scan_strategy.has_value()) {
        const std::string& s = *merged_.scan_strategy;
        if (s != "greedy" && s != "beam" && s != "exhaustive") {
            return fail("--strategy must be greedy | beam | exhaustive");
        }
        cfg.scan_strategy_ = s;
    }
    if (merged_.scan_beam_width.has_value()) {
        if (*merged_.scan_beam_width < 1) return fail("--beam-width must be >= 1");
        cfg.scan_beam_width_ = *merged_.scan_beam_width;
    }
    cfg.scan_base_ = merged_.scan_base;
    if (merged_.scan_prerank.has_value()) cfg.scan_prerank_ = *merged_.scan_prerank;
    if (merged_.scan_suggest_swaps.has_value()) cfg.scan_suggest_swaps_ = *merged_.scan_suggest_swaps;
    if (merged_.scan_right_search.has_value()) {
        const std::string& s = *merged_.scan_right_search;
        if (s != "none" && s != "check" && s != "add-drop") {
            return fail("--right-search must be none | check | add-drop");
        }
        cfg.scan_right_search_ = s;
    }
    cfg.scan_right_pool_ = merged_.scan_right_pool;
    if (cfg.max_sources_ != -1 && cfg.max_sources_ < cfg.min_sources_) {
        return fail("--max-sources must be >= --min-sources");
    }

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

    if (merged_.f2_dir)   cfg.f2_dir_ = *merged_.f2_dir;
    if (merged_.target)   cfg.target_ = *merged_.target;
    cfg.left_  = merged_.left;
    cfg.right_ = merged_.right;
    cfg.pool_  = merged_.pool;
    cfg.pop1_  = merged_.pop1;
    cfg.pop2_  = merged_.pop2;
    cfg.pop3_  = merged_.pop3;
    cfg.pop4_  = merged_.pop4;
    cfg.pop5_  = merged_.pop5;
    cfg.pops_  = merged_.pops;
    if (merged_.out_file) cfg.out_file_ = *merged_.out_file;
    if (merged_.qpdstat_prefix) cfg.qpdstat_prefix_ = *merged_.qpdstat_prefix;

    // READv2 controls.
    if (merged_.samples) cfg.samples_file_ = *merged_.samples;
    if (merged_.window_snps.has_value()) {
        if (*merged_.window_snps < 1) return fail("--window-snps must be >= 1");
        cfg.window_snps_ = *merged_.window_snps;
    }
    if (merged_.norm.has_value()) {
        const std::string n = to_lower(trim(*merged_.norm));
        if (n != "median" && n != "mean") {
            return fail("--norm '" + *merged_.norm + "' is unknown (use median | mean)");
        }
        cfg.norm_mode_ = n;
    }
    if (merged_.min_overlap.has_value()) {
        if (*merged_.min_overlap < 0.0 || *merged_.min_overlap > 1.0) {
            return fail("--min-overlap must lie in [0, 1]");
        }
        cfg.min_overlap_ = *merged_.min_overlap;
    }

    // Li-Stephens `paint` controls.
    if (merged_.donors) cfg.donors_prefix_ = *merged_.donors;
    if (merged_.labels) cfg.labels_file_ = *merged_.labels;
    if (merged_.face) {
        const std::string f = to_lower(trim(*merged_.face));
        if (f != "paint" && f != "localanc" && f != "impute" && f != "roh" && f != "contam") {
            return fail("--face '" + *merged_.face +
                        "' is unknown (paint | localanc | impute | roh | contam)");
        }
        if (f != "paint" && f != "localanc") {
            return fail("--face '" + f + "' is not yet available (v1 ships paint + localanc; "
                        "impute/roh/contam are later)");
        }
        cfg.face_ = f;
    }
    if (merged_.ls_ne.has_value()) {
        if (!(*merged_.ls_ne > 0.0) || !std::isfinite(*merged_.ls_ne)) {
            return fail("--Ne must be a finite value > 0");
        }
        cfg.ls_ne_ = *merged_.ls_ne;
    }
    if (merged_.ls_theta.has_value()) {
        const std::string t = to_lower(trim(*merged_.ls_theta));
        if (t == "auto" || t.empty()) {
            cfg.ls_theta_ = std::numeric_limits<double>::quiet_NaN();
        } else {
            char* end = nullptr;
            const double v = std::strtod(t.c_str(), &end);
            if (end == t.c_str() || *end != '\0' || !std::isfinite(v) || v < 0.0 || v > 1.0) {
                return fail("--theta must be 'auto' or a finite value in [0, 1]; got '" +
                            *merged_.ls_theta + "'");
            }
            cfg.ls_theta_ = v;
        }
    }
    if (merged_.self_copy) cfg.ls_self_copy_ = *merged_.self_copy;
    if (merged_.bp_fallback) cfg.ls_bp_fallback_ = *merged_.bp_fallback;
    if (merged_.paint_full) cfg.paint_full_ = *merged_.paint_full;
    if (merged_.recip_vcf) cfg.recip_vcf_ = *merged_.recip_vcf;
    if (merged_.donor_vcf) cfg.donor_vcf_ = *merged_.donor_vcf;
    if (merged_.vcf_map) cfg.vcf_map_ = *merged_.vcf_map;
    if (merged_.vcf_region) cfg.vcf_region_ = *merged_.vcf_region;
    if (merged_.vcf_unphased_max.has_value()) {
        if (!(*merged_.vcf_unphased_max >= 0.0) || !std::isfinite(*merged_.vcf_unphased_max)) {
            return fail("--unphased-max must be a finite value >= 0");
        }
        cfg.vcf_unphased_max_ = *merged_.vcf_unphased_max;
    }
    if (merged_.fst_method) {
        const std::string m = to_lower(trim(*merged_.fst_method));
        if (m != "wc" && m != "hudson") {
            return fail("--method '" + *merged_.fst_method + "' is unknown (wc | hudson)");
        }
        cfg.fst_method_ = m;
    }
    if (merged_.fst_per_snp) cfg.fst_per_snp_ = *merged_.fst_per_snp;
    if (merged_.fst_all_pairs) cfg.fst_all_pairs_ = *merged_.fst_all_pairs;
    if (merged_.fst_windowed) cfg.fst_windowed_ = *merged_.fst_windowed;
    if (merged_.fst_pbs) cfg.fst_pbs_ = *merged_.fst_pbs;
    if (merged_.kinship_all_pairs) cfg.kinship_all_pairs_ = *merged_.kinship_all_pairs;
    if (merged_.pairs) cfg.pairs_file_ = *merged_.pairs;
    if (merged_.min_kinship) cfg.min_kinship_ = *merged_.min_kinship;
    if (merged_.command == Command::Kinship && !cfg.pairs_file_.empty() &&
        cfg.kinship_all_pairs_) {
        return fail("steppe kinship: --pairs and --all-pairs are mutually exclusive");
    }
    if (merged_.sfs_fold) cfg.sfs_fold_ = *merged_.sfs_fold;
    if (merged_.pca_k.has_value()) {
        if (*merged_.pca_k < 1) return fail("--k must be >= 1 (number of principal components)");
        cfg.pca_k_ = *merged_.pca_k;
    }
    if (merged_.pca_eigenvalues) cfg.pca_eigenvalues_ = *merged_.pca_eigenvalues;
    if (merged_.pca_emit_html) cfg.pca_emit_html_ = *merged_.pca_emit_html;
    if (merged_.bgen) cfg.pca_bgen_ = *merged_.bgen;
    if (!merged_.project_pops.empty()) cfg.project_pops_ = merged_.project_pops;
    if (merged_.project_samples) cfg.project_samples_file_ = *merged_.project_samples;
    if (merged_.project_mode) {
        const std::string m = to_lower(trim(*merged_.project_mode));
        if (m != "lsq" && m != "scaled")
            return fail("--project-mode '" + *merged_.project_mode + "' is unknown (lsq | scaled)");
        cfg.project_mode_ = m;
    }
    if (merged_.pca_solver) {
        const std::string s = to_lower(trim(*merged_.pca_solver));
        if (s != "exact" && s != "randomized" && s != "auto")
            return fail("--pca-solver '" + *merged_.pca_solver +
                        "' is unknown (exact | randomized | auto)");
        cfg.pca_solver_ = s;
    }
    if (merged_.recip_batch.has_value()) {
        if (*merged_.recip_batch < 1) return fail("--recip-batch must be >= 1");
        cfg.ls_recip_batch_ = *merged_.recip_batch;
    }
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
    if (merged_.graph)       cfg.graph_file_ = *merged_.graph;
    if (merged_.numstart) {
        if (*merged_.numstart < 1) return fail("--numstart must be >= 1");
        cfg.qpgraph_numstart_ = *merged_.numstart;
    }
    if (merged_.diag_f3)     cfg.qpgraph_diag_f3_ = *merged_.diag_f3;
    if (merged_.constrained) cfg.qpgraph_constrained_ = *merged_.constrained;
    if (merged_.max_nadmix) {
        if (*merged_.max_nadmix < 0) return fail("--max-nadmix must be >= 0");
        cfg.qpgraph_max_nadmix_ = *merged_.max_nadmix;
    }

    return cfg;
}

}  // namespace steppe::config
