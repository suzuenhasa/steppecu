// src/core/config/run_config.hpp
//
// RunConfig — the immutable, validated configuration the CLI freezes and hands to
// the compute layer. Built only by ConfigBuilder::build() (const accessors, no
// mutators) and CUDA-free by contract, so core/config compiles and unit-tests
// with no GPU toolkit.
//
// Reference: docs/reference/src_core_config_run_config.hpp.md
#ifndef STEPPE_CORE_CONFIG_RUN_CONFIG_HPP
#define STEPPE_CORE_CONFIG_RUN_CONFIG_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/qpadm.hpp"
#include "io/ind_reader.hpp"
#include "core/config/cli_args.hpp"

namespace steppe::config {

// The frozen, validated run config — reference §1–§4
class RunConfig {
public:
    RunConfig() = default;

    // Selected command — reference §5
    [[nodiscard]] Command command() const noexcept { return command_; }

    // The four resolved config structs — reference §6
    [[nodiscard]] const DeviceConfig& device() const noexcept { return device_; }

    [[nodiscard]] const QpAdmOptions& qpadm_options() const noexcept { return qpadm_options_; }

    [[nodiscard]] const FilterConfig& filter() const noexcept { return filter_; }

    [[nodiscard]] const io::PopSelection& pop_selection() const noexcept { return pop_selection_; }

    // Carried input and output paths — reference §7
    [[nodiscard]] const std::string& f2_dir()   const noexcept { return f2_dir_; }
    [[nodiscard]] const std::string& target()   const noexcept { return target_; }
    [[nodiscard]] const std::vector<std::string>& left()  const noexcept { return left_; }
    [[nodiscard]] const std::vector<std::string>& right() const noexcept { return right_; }
    [[nodiscard]] const std::vector<std::string>& pool()  const noexcept { return pool_; }
    // f4 and f4-ratio population columns — reference §8
    [[nodiscard]] const std::vector<std::string>& pop1()  const noexcept { return pop1_; }
    [[nodiscard]] const std::vector<std::string>& pop2()  const noexcept { return pop2_; }
    [[nodiscard]] const std::vector<std::string>& pop3()  const noexcept { return pop3_; }
    [[nodiscard]] const std::vector<std::string>& pop4()  const noexcept { return pop4_; }
    [[nodiscard]] const std::vector<std::string>& pop5()  const noexcept { return pop5_; }
    [[nodiscard]] const std::vector<std::string>& pops()  const noexcept { return pops_; }
    // Carried input and output paths (continued) — reference §7
    [[nodiscard]] const std::string& out_file() const noexcept { return out_file_; }
    [[nodiscard]] const std::string& format()   const noexcept { return format_; }
    [[nodiscard]] const std::string& qpdstat_prefix() const noexcept { return qpdstat_prefix_; }
    [[nodiscard]] const std::string& geno()     const noexcept { return geno_; }
    [[nodiscard]] const std::string& snp()      const noexcept { return snp_; }
    [[nodiscard]] const std::string& ind()      const noexcept { return ind_; }
    [[nodiscard]] const std::string& out_dir()  const noexcept { return out_dir_; }

    // extract-f2 controls — reference §9
    [[nodiscard]] double blgsize_cm()           const noexcept { return blgsize_cm_; }
    [[nodiscard]] int min_sources()             const noexcept { return min_sources_; }
    [[nodiscard]] double scan_p_min()           const noexcept { return scan_p_min_; }
    [[nodiscard]] bool scan_allow_clade()       const noexcept { return scan_allow_clade_; }
    [[nodiscard]] const std::string& scan_strategy() const noexcept { return scan_strategy_; }
    [[nodiscard]] int scan_beam_width()          const noexcept { return scan_beam_width_; }
    [[nodiscard]] const std::vector<std::string>& scan_base() const noexcept { return scan_base_; }
    [[nodiscard]] bool scan_prerank()            const noexcept { return scan_prerank_; }
    [[nodiscard]] bool scan_suggest_swaps()      const noexcept { return scan_suggest_swaps_; }

    // f-statistic sweep controls — reference §10
    [[nodiscard]] double sweep_min_z()          const noexcept { return sweep_min_z_; }
    [[nodiscard]] int    sweep_top_k()          const noexcept { return sweep_top_k_; }
    [[nodiscard]] bool   sweep_sure()           const noexcept { return sweep_sure_; }
    [[nodiscard]] bool   sweep_all_combinations() const noexcept { return sweep_all_combinations_; }
    [[nodiscard]] const std::string& shard_dir() const noexcept { return shard_dir_; }

    // extract-f2 controls (continued) — reference §9
    [[nodiscard]] PloidyMode ploidy()           const noexcept { return ploidy_; }
    [[nodiscard]] int max_sources()             const noexcept { return max_sources_; }

    [[nodiscard]] bool dry_run()                const noexcept { return dry_run_; }

    [[nodiscard]] bool hash_source()            const noexcept { return hash_source_; }

    // qpGraph controls — reference §11
    [[nodiscard]] const std::string& graph_file() const noexcept { return graph_file_; }
    [[nodiscard]] int    qpgraph_numstart()    const noexcept { return qpgraph_numstart_; }
    [[nodiscard]] double qpgraph_diag_f3()     const noexcept { return qpgraph_diag_f3_; }
    [[nodiscard]] bool   qpgraph_constrained() const noexcept { return qpgraph_constrained_; }
    [[nodiscard]] int    qpgraph_max_nadmix()  const noexcept { return qpgraph_max_nadmix_; }

private:
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
    std::vector<std::string> pop1_, pop2_, pop3_, pop4_;
    std::vector<std::string> pop5_;
    std::vector<std::string> pops_;
    std::string out_file_;
    std::string format_ = "csv";
    std::string qpdstat_prefix_;
    std::string geno_;
    std::string snp_;
    std::string ind_;
    std::string out_dir_;
    double blgsize_cm_ = kDefaultBlockSizeCm;
    PloidyMode ploidy_ = PloidyMode::Auto;
    int min_sources_ = 1;
    int max_sources_ = -1;
    double scan_p_min_ = 0.05;
    bool scan_allow_clade_ = true;
    std::string scan_strategy_ = "beam";
    int scan_beam_width_ = 3;
    std::vector<std::string> scan_base_;
    bool scan_prerank_ = false;
    bool scan_suggest_swaps_ = false;
    double sweep_min_z_ = 3.0;
    int sweep_top_k_ = -1;
    bool sweep_sure_ = false;
    bool sweep_all_combinations_ = false;
    std::string shard_dir_;
    bool dry_run_ = false;
    bool hash_source_ = false;
    std::string graph_file_;
    int    qpgraph_numstart_   = 10;
    double qpgraph_diag_f3_    = 1e-5;
    bool   qpgraph_constrained_ = true;
    int    qpgraph_max_nadmix_ = 1;
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_RUN_CONFIG_HPP
