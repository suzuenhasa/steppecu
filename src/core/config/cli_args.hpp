// src/core/config/cli_args.hpp
//
// CliArgs — the flat, parse-time command-line surface: one raw field per flag,
// nothing resolved or validated. The CLI parser fills it in and ConfigBuilder
// folds it over the lower config layers (compiled defaults < TOML < env < CLI),
// then build() freezes an immutable RunConfig. CUDA-free by contract. Each field
// is a std::optional "was this flag set?" sentinel — an unset field leaves the
// lower layer intact; a few vector/string fields use emptiness as that sentinel.
//
// Reference: docs/reference/src_core_config_cli_args.hpp.md
#ifndef STEPPE_CORE_CONFIG_CLI_ARGS_HPP
#define STEPPE_CORE_CONFIG_CLI_ARGS_HPP

#include <optional>
#include <string>
#include <vector>

namespace steppe::config {

// Command — the selected subcommand — reference §3
enum class Command {
    None,
    ExtractF2,
    QpAdm,
    QpWave,
    QpAdmRotate,
    F4,
    F3,
    F4Ratio,
    Qpdstat,
    F4Sweep,
    F3Sweep,
    Qpfstats,
    QpGraph,
    QpGraphSearch,
    Dates,
    Scan,
};

// PloidyMode — extract-f2 ploidy policy — reference §4
enum class PloidyMode {
    Auto,
    Diploid,
    PseudoHaploid,
};

struct CliArgs {
    Command command = Command::None;

    // Global resource and precision knobs — reference §5
    std::optional<std::string> device;
    std::optional<std::string> precision;
    std::optional<std::string> config_path;

    // qpAdm and qpwave inputs — reference §6
    std::optional<std::string> f2_dir;
    std::optional<std::string> target;
    std::vector<std::string> left;
    std::vector<std::string> right;
    std::vector<std::string> pool;

    // Standalone f-statistic inputs — reference §7
    std::vector<std::string> pop1, pop2, pop3, pop4;

    // qpGraph and qpgraph-search options — reference §8
    std::optional<std::string> graph;
    std::optional<int> numstart;
    std::optional<double> diag_f3;
    std::optional<bool> constrained;
    std::optional<int> max_nadmix;

    // qpdstat magnitude prefix and f4-ratio 5th column — reference §7
    std::optional<std::string> qpdstat_prefix;
    std::vector<std::string> pop5;

    // qpAdm option overrides — reference §9
    std::optional<double> fudge;
    std::optional<int>    als_iterations;
    std::optional<int>    rank;
    std::optional<double> rank_alpha;
    std::optional<bool>   allow_negative_weights;
    std::optional<int>    jackknife;
    std::optional<double> p_se_threshold;
    std::optional<bool>   se_require_p;

    // qpadm-rotate enumeration bounds — reference §10
    std::optional<int> min_sources;
    std::optional<int> max_sources;

    // steppe scan (proxy/model scanner) objective — hard-gate p cutoff (α), default 0.05
    std::optional<double> scan_p_min;
    // steppe scan: may a 1-source (clade) model be crowned the winner? default true
    std::optional<bool> scan_allow_clade;
    // steppe scan Phase-1 search: strategy (greedy|beam|exhaustive), beam width, optional seed
    std::optional<std::string> scan_strategy;
    std::optional<int>         scan_beam_width;
    std::vector<std::string>   scan_base;
    // steppe scan Phase-2: relatedness shortlist mode, and drop/replace swap suggestions
    std::optional<bool> scan_prerank;
    std::optional<bool> scan_suggest_swaps;

    // f4-sweep and f3-sweep controls — reference §11
    std::optional<bool>   sweep_all_combinations;
    std::optional<double> sweep_min_z;
    std::optional<int>    sweep_top_k;
    std::optional<bool>   sweep_sure;
    std::optional<std::string> shard_dir;

    // extract-f2 inputs — reference §12
    std::optional<std::string> prefix;
    std::optional<std::string> geno;
    std::optional<std::string> snp;
    std::optional<std::string> ind;
    std::optional<std::string> out_dir;
    std::vector<std::string>   pops;
    std::optional<int>         auto_top_k;
    std::optional<int>         min_n;
    std::optional<double>      blgsize;
    std::optional<PloidyMode>  ploidy;

    // Filter overrides — reference §13
    std::optional<double> maf;
    std::optional<double> geno_max_missing;
    std::optional<double> mind_max_missing;
    std::optional<bool>   autosomes_only;
    std::optional<bool>   drop_monomorphic;
    std::optional<bool>   transversions_only;
    std::optional<std::string> strand_mode;

    // extract-f2 run controls — reference §14
    std::optional<std::string> tier;
    std::optional<bool>   dry_run;
    std::optional<bool>   hash_source;

    // Output — reference §15
    std::optional<std::string> out_file;
    std::optional<std::string> format;
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_CLI_ARGS_HPP
