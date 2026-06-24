// src/core/config/cli_args.hpp
//
// CliArgs — the FLAT, parse-time-mutable command-line surface (architecture.md §9
// "ConfigBuilder& merge_cli(const CliArgs&)"). It is the highest-precedence config
// LAYER (compiled defaults < TOML < env STEPPE_* < CLI; architecture.md §9, ROADMAP
// §4 / cli-bindings.md §4.5). The CLI11 parser (src/app, plain CXX) binds flags into
// this struct, then ConfigBuilder::merge_cli folds it over the lower layers and
// build() validates + freezes into an immutable RunConfig.
//
// CUDA-FREE BY CONTRACT — like steppe/config.hpp, it depends only on the C++ standard
// library, so it compiles into core, the CLI, and (later) the bindings without the
// device toolkit (architecture.md §4). It names ONLY std types and the CUDA-free
// public enums; the app->struct mapping (e.g. "--device 0,1" string -> devices) is
// done in ConfigBuilder, the single validated merge site, not scattered in the parser.
//
// std::optional is the "was this flag SET on the command line?" sentinel: an UNSET
// optional leaves the lower-precedence layer (env / TOML / compiled default) intact;
// a SET optional overrides it (the §9 precedence rule made explicit per field). A bare
// value (no optional) would conflate "user passed the default" with "user passed
// nothing", silently clobbering a TOML/env override with the compiled default.
#ifndef STEPPE_CORE_CONFIG_CLI_ARGS_HPP
#define STEPPE_CORE_CONFIG_CLI_ARGS_HPP

#include <optional>
#include <string>
#include <vector>

namespace steppe::config {

/// Which subcommand was selected (mirrors cli-bindings.md §4.1 command set). `None`
/// is the no-subcommand case (bare `steppe` -> help). The mapping to the real fit
/// entry points (run_qpadm / run_qpwave / run_qpadm_search) is the app's job; this
/// enum is only the parsed selection so build() can apply command-specific validation.
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
    F4Sweep,  ///< GPU-only all-combinations f4 sweep (every C(P,4); on-device filter).
    F3Sweep,  ///< GPU-only all-combinations f3 sweep (every C(P,3); on-device filter).
    Qpfstats, ///< genotype-path joint f2 smoother (--prefix + --pops -> a smoothed f2 dir).
};

/// extract-f2 ploidy policy (cli-bindings.md §4.1; the f2-estimator pseudo-haploid
/// fix). `Auto` (the DEFAULT, matching AT2 adjust_pseudohaploid=TRUE) auto-detects
/// each sample's ploidy from its genotypes (a het call ⇒ diploid, none ⇒ pseudo-
/// haploid); `Diploid`/`PseudoHaploid` force a uniform ploidy for every sample
/// (the --ploidy 2 / --ploidy 1 overrides; the legacy hardcoded-2 behavior is
/// `Diploid`).
enum class PloidyMode {
    Auto,
    Diploid,
    PseudoHaploid,
};

/// The flat, parse-time CLI surface. Every field is the std::optional "was it set?"
/// sentinel for the §9 precedence merge (an UNSET field does NOT override the lower
/// layer). The vector/string fields that default to empty use emptiness as the
/// "unset" sentinel where an empty value is itself the no-op (e.g. an empty pop list).
struct CliArgs {
    /// Selected subcommand (None ⇒ no subcommand; the app prints help).
    Command command = Command::None;

    // ---- Global resource / precision knobs (every fit + extract command) -------

    /// Raw `--device` argument string, UNPARSED ("auto" | "0" | "0,1"). Parsed +
    /// validated into DeviceConfig::devices by ConfigBuilder (the single merge site),
    /// NOT in the parser. GPU-ONLY: there is no "cpu" value (cli-bindings.md §5.4);
    /// "cpu" is rejected at build() as InvalidConfig.
    std::optional<std::string> device;

    /// Raw `--precision` argument ("emu40" | "emu32" | "fp64" | "tf32"). Mapped to
    /// steppe::Precision by ConfigBuilder; an unknown token is InvalidConfig.
    std::optional<std::string> precision;

    /// `--config FILE` — a TOML config to merge BELOW the CLI but ABOVE env/defaults
    /// (architecture.md §9). Empty ⇒ no file. The app passes this to merge_file().
    std::optional<std::string> config_path;

    // ---- qpAdm / qpwave inputs + options (cli-bindings.md §4.1) ----------------

    /// `--f2-dir DIR` — the precompute-once/fit-many f2_blocks dir (cli-bindings.md
    /// §4.3). Resolved + uploaded by the app (M(cli-1)); here it is only carried.
    std::optional<std::string> f2_dir;

    /// `--target` pop label (qpadm/qpadm-rotate). Resolved to an index against
    /// pops.txt by the app, never here (the engine is index-only).
    std::optional<std::string> target;

    /// `--left a,b,...` source labels (qpadm) / the full left set (qpwave, left[0]=ref).
    std::vector<std::string> left;

    /// `--right r0,r1,...` outgroup labels; right[0] == R0.
    std::vector<std::string> right;

    /// `--pool a,b,c,...` (qpadm-rotate) source pool the app enumerates subsets of.
    std::vector<std::string> pool;

    /// `--pop1`/`--pop2`/`--pop3`/`--pop4` (the `f4` command) — ROW-ALIGNED quartet
    /// columns (admixtools::f4 comb=FALSE): quartet k = (pop1[k], pop2[k], pop3[k],
    /// pop4[k]). All four must have the same length (the app validates). The single-
    /// quartet convenience `--pops A,B,C,D` (a 4-name --pops) is ALSO accepted by the f4
    /// command (it reuses the `pops` field below, read in groups of 4), so a one-line f4
    /// needs no four flags.
    std::vector<std::string> pop1, pop2, pop3, pop4;

    /// `--prefix PATH` (the `qpdstat` command ONLY) — the genotype prefix for the
    /// normalized-D MAGNITUDE (Part B; not yet implemented). DISTINCT from `prefix` below
    /// (extract-f2's geno/snp/ind triple prefix) so the qpdstat guard checks it WITHOUT
    /// triggering ConfigBuilder's extract-f2 prefix->geno/snp/ind expansion. When SET, the
    /// qpdstat command fails fast with a "Part B not yet implemented" message; the --f2-dir
    /// path reports f4 (the AT2 f2-path convention, proven byte-identical to qpdstat f4mode).
    std::optional<std::string> qpdstat_prefix;

    /// `--pop5` (the `f4-ratio` command only) — the 5th ROW-ALIGNED column (admixtools
    /// ::qpf4ratio): tuple k = (pop1[k], pop2[k], pop3[k], pop4[k], pop5[k]); alpha =
    /// f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4). All five columns must have the same length (the app
    /// validates). The `--pops A,B,C,D,E` 5-tuple convenience (the `pops` field below, read in
    /// groups of 5) is ALSO accepted by the f4-ratio command.
    std::vector<std::string> pop5;

    // ---- QpAdmOptions overrides (default to the struct defaults; cli-bindings.md
    // §4.1 — flag names mirror QpAdmOptions so a bare invocation reproduces goldens).

    std::optional<double> fudge;            ///< --fudge        (QpAdmOptions::fudge)
    std::optional<int>    als_iterations;   ///< --als-iters    (QpAdmOptions::als_iterations)
    std::optional<int>    rank;             ///< --rank         (QpAdmOptions::rank)
    std::optional<double> rank_alpha;       ///< --rank-alpha   (QpAdmOptions::rank_alpha)
    std::optional<bool>   allow_negative_weights;  ///< --allow-neg / --no-allow-neg
    std::optional<int>    jackknife;        ///< --jackknife 0|1|2 (JackknifePolicy)
    std::optional<double> p_se_threshold;   ///< --p-se-threshold
    std::optional<bool>   se_require_p;     ///< --se-require-p

    // ---- qpadm-rotate pool-enumeration bounds (cli-bindings.md §4.1) -----------
    std::optional<int> min_sources;         ///< --min-sources
    std::optional<int> max_sources;         ///< --max-sources

    // ---- f4-sweep / f3-sweep (GPU-only all-combinations f-stat sweep) ----------
    /// --all-quartets (f4 / qpdstat) / --all-triples (f3) : ENABLE the sweep mode on the
    /// standalone-stat commands. When set, the command enumerates EVERY C(P,k) combination
    /// over the --pops SUBSET (empty ⇒ the whole f2 dir) instead of the explicit row-aligned
    /// --pop1..--popK list. Unset ⇒ the byte-identical explicit-list path (the goldens).
    std::optional<bool>   sweep_all_combinations;
    /// --min-z Z : keep items with |z| >= Z (the on-device filter; default 3.0). Mutually
    /// exclusive with --top-k.
    std::optional<double> sweep_min_z;
    /// --top-k K : keep the K items with the largest |z| (bounded device-side reservoir, ~K resident).
    std::optional<int>    sweep_top_k;
    /// --sure : lift the maxcomb cap (a sweep over more than kFstatMaxComb items refuses
    /// without it — the cap guards COMPUTE TIME, every item is computed to test the filter).
    std::optional<bool>   sweep_sure;
    /// --shard-dir DIR : write the survivor table to a CSV file under DIR (created if absent),
    /// instead of stdout/--out. At sweep scale the survivor set (post |z| / top-k filter) is
    /// the small output; the full C(P,k) table is NEVER materialized (it stays on-device).
    std::optional<std::string> shard_dir;

    // ---- extract-f2 inputs (cli-bindings.md §4.1; consumed in M(cli-4)) --------
    std::optional<std::string> prefix;      ///< --prefix (sets geno/snp/ind = PREFIX.{geno,snp,ind})
    std::optional<std::string> geno;        ///< --geno
    std::optional<std::string> snp;         ///< --snp
    std::optional<std::string> ind;         ///< --ind
    std::optional<std::string> out_dir;     ///< --out (extract-f2 dir)
    std::vector<std::string>   pops;        ///< --pops (PopSelection::Explicit)
    std::optional<int>         auto_top_k;  ///< --auto-top-k (PopSelection::AutoTopK)
    std::optional<int>         min_n;       ///< --min-n (PopSelection::MinN)
    std::optional<double>      blgsize;     ///< --blgsize (MORGANS, AT2 convention; default 0.05 = 5 cM). ConfigBuilder converts Morgans->cM (×kCentimorgansPerMorgan) into the cM-stored RunConfig::blgsize_cm_.
    std::optional<PloidyMode>  ploidy;      ///< --ploidy auto|1|2 (default Auto = AT2 adjust_pseudohaploid; the f2 pseudo-haploid fix)

    // ---- FilterConfig overrides (cli-bindings.md §4.1; M(cli-4)) ---------------
    std::optional<double> maf;              ///< --maf
    std::optional<double> geno_max_missing; ///< --geno-max-miss
    std::optional<double> mind_max_missing; ///< --mind-max-miss
    std::optional<bool>   autosomes_only;   ///< --auto-only / --no-auto-only (extract-f2 default ON, AT2 parity)
    std::optional<bool>   drop_monomorphic; ///< --drop-mono / --no-drop-mono (extract-f2 default ON, AT2 poly_only parity)
    std::optional<bool>   transversions_only; ///< --transversions

    // ---- extract-f2 run controls (cli-bindings.md §4.1; M(cli-4)) -------------
    /// --tier auto|resident|host|disk (the M5 f2_blocks OUTPUT-tier override). Raw,
    /// UNPARSED token; ConfigBuilder maps it to DeviceConfig::force_tier at the single
    /// merge site (an unknown token is InvalidConfig). `auto` (the default) = the
    /// select_output_tier runtime policy; resident/host/disk PIN the tier. It is the
    /// higher-precedence twin of STEPPE_FORCE_TIER (the config field wins; the env stays
    /// the fallback resolve_output_tier honors). PARITY-NEUTRAL (the tier moves bytes to
    /// a different store, never a reported number; architecture.md §12).
    std::optional<std::string> tier;
    std::optional<bool>   dry_run;          ///< --dry-run (report tiers/sizes, no compute)
    /// --hash (default OFF): compute the source-dataset provenance SHA-256s
    /// (geno/snp/ind). OFF by default because the whole-.geno SHA is a ~tens-of-seconds
    /// whole-file read+compress that dominated extract-f2 yet only yields a provenance
    /// value; when ON it is overlapped on a background thread with the GPU pipeline.
    /// When OFF, meta.json records source_hash_computed:false + empty *_sha256.
    std::optional<bool>   hash_source;

    // ---- Output (cli-bindings.md §4.4) ----------------------------------------
    std::optional<std::string> out_file;    ///< --out FILE (stdout if unset)
    std::optional<std::string> format;      ///< --format csv|tsv|json (default csv)
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_CLI_ARGS_HPP
