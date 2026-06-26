// src/core/config/run_config.hpp
//
// RunConfig — the IMMUTABLE, VALIDATED config the CLI hands to the compute seam
// (architecture.md §9 "validates once ... and freezes into an immutable RunConfig.
// After construction, config is const."). It is the FROZEN result of
// ConfigBuilder::build(): const accessors only, no mutators.
//
// SHAPE DECISION (cli-bindings.md §1, §9.1): the architecture §9 sketch shows a
// RunConfig with abstract accessors (block_size_cm/resampling/seed) over a future
// GenotypeDataset entry; the REAL fit entry points take DeviceConfig / QpAdmOptions /
// PopSelection / FilterConfig DIRECTLY. So this RunConfig is exactly those four real
// structs, already mapped + validated, plus the carried I/O paths the app needs — the
// "map flags straight onto the real structs" decision (cli-bindings.md §1, §9.1).
// build() is the ONE site that turns the raw CLI/env/TOML layers into these typed
// structs and fail-fasts on bad input; everything downstream sees only valid config.
//
// CUDA-FREE BY CONTRACT (architecture.md §4): it composes only the CUDA-free public
// structs (steppe/config.hpp) + the io PopSelection + the qpadm QpAdmOptions + std
// types. It does NOT include any device header — the VRAM/precision/device validation
// in build() routes through the CUDA-free build_resources / BackendCapabilities seam
// from the APP layer at run time, not from this value object (the value object only
// HOLDS the resolved DeviceConfig; the live probe is the app's call, cli-bindings.md
// §4.5). So core/config compiles and is unit-tested with NO GPU and NO CUDA toolkit.
#ifndef STEPPE_CORE_CONFIG_RUN_CONFIG_HPP
#define STEPPE_CORE_CONFIG_RUN_CONFIG_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"     // DeviceConfig, FilterConfig, Precision (CUDA-free)
#include "steppe/qpadm.hpp"      // QpAdmOptions, JackknifePolicy (CUDA-free)
#include "io/ind_reader.hpp"     // io::PopSelection (the extract-f2 pop selection)
#include "core/config/cli_args.hpp"  // Command (echoed onto the frozen config)

namespace steppe::config {

/// The immutable, validated run configuration. Constructed ONLY by ConfigBuilder::
/// build() (which validates); the public ctor is intentionally the
/// aggregate-from-builder path. Const accessors only — no field is publicly mutable.
class RunConfig {
public:
    RunConfig() = default;

    /// The selected subcommand (echoed from CliArgs so the app dispatches on the
    /// frozen config, not the raw args).
    [[nodiscard]] Command command() const noexcept { return command_; }

    /// The resolved device/precision policy (devices parsed from "--device",
    /// precision from "--precision"). Empty `devices` ⇒ auto-enumerate (§9). This is
    /// what the app feeds to device::build_resources at run time.
    [[nodiscard]] const DeviceConfig& device() const noexcept { return device_; }

    /// The resolved qpAdm per-call options (fudge/als/rank/rank_alpha/jackknife/...),
    /// defaulted to the QpAdmOptions struct defaults so a bare invocation reproduces
    /// the goldens (cli-bindings.md §4.1).
    [[nodiscard]] const QpAdmOptions& qpadm_options() const noexcept { return qpadm_options_; }

    /// The resolved on-the-fly QC filters (extract-f2; M(cli-4)). Defaults are no-ops.
    [[nodiscard]] const FilterConfig& filter() const noexcept { return filter_; }

    /// The resolved population selection (extract-f2; M(cli-4)). Default AutoTopK with
    /// k unset is the "no selection requested" state — the app surfaces that as a
    /// fail-fast when extract-f2 needs an explicit selection.
    [[nodiscard]] const io::PopSelection& pop_selection() const noexcept { return pop_selection_; }

    // ---- Carried I/O strings (resolved verbatim; the app consumes them) ----------
    [[nodiscard]] const std::string& f2_dir()   const noexcept { return f2_dir_; }
    [[nodiscard]] const std::string& target()   const noexcept { return target_; }
    [[nodiscard]] const std::vector<std::string>& left()  const noexcept { return left_; }
    [[nodiscard]] const std::vector<std::string>& right() const noexcept { return right_; }
    [[nodiscard]] const std::vector<std::string>& pool()  const noexcept { return pool_; }
    /// The `f4` command's ROW-ALIGNED quartet columns (--pop1/--pop2/--pop3/--pop4). Also
    /// carries the `--pops` 4-tuple convenience verbatim via pops() below; the app picks
    /// whichever is non-empty (cmd_f4.cpp). Default empty.
    [[nodiscard]] const std::vector<std::string>& pop1()  const noexcept { return pop1_; }
    [[nodiscard]] const std::vector<std::string>& pop2()  const noexcept { return pop2_; }
    [[nodiscard]] const std::vector<std::string>& pop3()  const noexcept { return pop3_; }
    [[nodiscard]] const std::vector<std::string>& pop4()  const noexcept { return pop4_; }
    /// The `f4-ratio` command's 5th ROW-ALIGNED column (--pop5). tuple k = (pop1[k]..pop5[k]);
    /// alpha = f4(p1,p2;p3,p4)/f4(p1,p2;p5,p4). The `--pops` 5-tuple convenience rides on
    /// pops() (read in groups of 5). Default empty.
    [[nodiscard]] const std::vector<std::string>& pop5()  const noexcept { return pop5_; }
    /// The raw `--pops` labels (the f4 single-quartet convenience: 4 names = one quartet).
    /// Distinct from pop_selection() (which extract-f2 maps it into) — the f4 command reads
    /// these names directly, in groups of 4.
    [[nodiscard]] const std::vector<std::string>& pops()  const noexcept { return pops_; }
    [[nodiscard]] const std::string& out_file() const noexcept { return out_file_; }
    [[nodiscard]] const std::string& format()   const noexcept { return format_; }
    /// qpdstat's `--prefix` (the Part-B normalized-D genotype prefix). Empty ⇒ unset (the
    /// --f2-dir f4-reporting path); non-empty ⇒ the qpdstat command fails fast (Part B not
    /// yet implemented). Carried verbatim, NOT expanded into geno/snp/ind (extract's prefix).
    [[nodiscard]] const std::string& qpdstat_prefix() const noexcept { return qpdstat_prefix_; }
    [[nodiscard]] const std::string& geno()     const noexcept { return geno_; }
    [[nodiscard]] const std::string& snp()      const noexcept { return snp_; }
    [[nodiscard]] const std::string& ind()      const noexcept { return ind_; }
    [[nodiscard]] const std::string& out_dir()  const noexcept { return out_dir_; }
    [[nodiscard]] double blgsize_cm()           const noexcept { return blgsize_cm_; }
    [[nodiscard]] int min_sources()             const noexcept { return min_sources_; }

    /// f4-sweep / f3-sweep filter knobs. sweep_top_k() > 0 ⇒ TopK mode (keep the K largest
    /// |z|); else MinZ mode (keep |z| >= sweep_min_z()). sweep_sure() lifts the maxcomb cap.
    [[nodiscard]] double sweep_min_z()          const noexcept { return sweep_min_z_; }
    [[nodiscard]] int    sweep_top_k()          const noexcept { return sweep_top_k_; }
    [[nodiscard]] bool   sweep_sure()           const noexcept { return sweep_sure_; }
    /// --all-quartets / --all-triples: when true, the f4/f3/qpdstat command routes to the
    /// GPU sweep (every C(P,k) over the --pops subset) instead of the explicit-list path.
    [[nodiscard]] bool   sweep_all_combinations() const noexcept { return sweep_all_combinations_; }
    /// --shard-dir: directory the sweep writes the survivor CSV into (empty ⇒ stdout/--out).
    [[nodiscard]] const std::string& shard_dir() const noexcept { return shard_dir_; }

    /// extract-f2 ploidy policy (--ploidy auto|1|2). Default Auto = AT2
    /// adjust_pseudohaploid per-sample auto-detection (the f2 pseudo-haploid fix).
    [[nodiscard]] PloidyMode ploidy()           const noexcept { return ploidy_; }
    [[nodiscard]] int max_sources()             const noexcept { return max_sources_; }

    /// --dry-run (extract-f2): report tiers/sizes/precision and exit without compute
    /// (cli-bindings.md §4.5 planning aid). Defaults to false (a real run).
    [[nodiscard]] bool dry_run()                const noexcept { return dry_run_; }

    /// --hash (extract-f2): compute the source-dataset provenance SHA-256s. Default
    /// false — the whole-.geno SHA dominated extract-f2 (a provenance value, not
    /// correctness), so it is skipped unless explicitly requested. When true the
    /// command overlaps the big geno hash with the GPU pipeline on a background thread.
    [[nodiscard]] bool hash_source()            const noexcept { return hash_source_; }

    // ---- qpgraph (single-graph fit) -------------------------------------------
    /// --graph FILE: the admixture-graph edge-list file path (empty ⇒ unset).
    [[nodiscard]] const std::string& graph_file() const noexcept { return graph_file_; }
    /// --numstart / --diag-f3 / --constrained (qpgraph; defaults == QpGraphOptions / AT2).
    [[nodiscard]] int    qpgraph_numstart()    const noexcept { return qpgraph_numstart_; }
    [[nodiscard]] double qpgraph_diag_f3()     const noexcept { return qpgraph_diag_f3_; }
    [[nodiscard]] bool   qpgraph_constrained() const noexcept { return qpgraph_constrained_; }
    /// --max-nadmix (qpgraph-search): the bounded admixture-node ceiling (v1 supports {0,1}).
    [[nodiscard]] int    qpgraph_max_nadmix()  const noexcept { return qpgraph_max_nadmix_; }

private:
    // ConfigBuilder is the ONLY constructor of a validated RunConfig (it sets these
    // fields after build()-validation). Friendship keeps the fields const-after-build
    // without a public mutating surface (architecture.md §9 "config is const").
    friend class ConfigBuilder;

    Command         command_ = Command::None;
    DeviceConfig    device_{};
    QpAdmOptions    qpadm_options_{};
    FilterConfig    filter_{};
    io::PopSelection pop_selection_{};

    std::string f2_dir_;
    std::string target_;
    std::vector<std::string> left_;
    std::vector<std::string> right_;
    std::vector<std::string> pool_;
    std::vector<std::string> pop1_, pop2_, pop3_, pop4_;  // f4 row-aligned quartet columns
    std::vector<std::string> pop5_;                       // f4-ratio 5th row-aligned column (--pop5)
    std::vector<std::string> pops_;                       // f4 --pops 4-tuple convenience (raw labels)
    std::string out_file_;
    std::string format_ = "csv";   // cli-bindings.md §4.4 default
    std::string qpdstat_prefix_;   // qpdstat --prefix (Part-B normalized-D; carried verbatim)
    std::string geno_;
    std::string snp_;
    std::string ind_;
    std::string out_dir_;
    double blgsize_cm_ = kDefaultBlockSizeCm;  // cli-bindings.md §4.1 default (5 cM)
    PloidyMode ploidy_ = PloidyMode::Auto;     // --ploidy default (AT2 adjust_pseudohaploid)
    int min_sources_ = 1;
    int max_sources_ = -1;          // -1 ⇒ "up to the whole pool" (app default)
    double sweep_min_z_ = 3.0;      // f4/f3-sweep --min-z (MinZ filter threshold; default 3.0)
    int sweep_top_k_ = -1;          // f4/f3-sweep --top-k (>0 ⇒ TopK mode; -1 ⇒ MinZ mode)
    bool sweep_sure_ = false;       // f4/f3-sweep --sure (lift the maxcomb cap)
    bool sweep_all_combinations_ = false;  // --all-quartets / --all-triples (route f4/f3 to the sweep)
    std::string shard_dir_;         // --shard-dir (survivor CSV destination; empty ⇒ stdout/--out)
    bool dry_run_ = false;          // --dry-run (extract-f2 planning aid, §4.5)
    bool hash_source_ = false;      // --hash (extract-f2): source-provenance SHA opt-in (default OFF)
    std::string graph_file_;        // qpgraph --graph (the edge-list file path)
    int    qpgraph_numstart_   = 10;     // qpgraph --numstart (QpGraphOptions default)
    double qpgraph_diag_f3_    = 1e-5;   // qpgraph --diag-f3 (AT2 default)
    bool   qpgraph_constrained_ = true;  // qpgraph --constrained (AT2 default)
    int    qpgraph_max_nadmix_ = 1;      // qpgraph-search --max-nadmix (bounded; v1 {0,1})
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_RUN_CONFIG_HPP
