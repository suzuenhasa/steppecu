// include/steppe/qpadm.hpp
//
// Public, CUDA-free value types and entry points for the qpAdm fit engine —
// run_qpadm / run_qpwave / run_qpadm_search over QpAdmModel. Populations are
// referenced by index into the f2_blocks population axis, and every entry point
// has a GPU-resident primary overload plus a host-memory parity overload.
//
// Reference: docs/reference/include_steppe_qpadm.hpp.md
#ifndef STEPPE_QPADM_HPP
#define STEPPE_QPADM_HPP

#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"

namespace steppe {

namespace device {
class DeviceF2Blocks;
struct Resources;
}  // namespace device

// Jackknife SE policy for the model-space search — reference §2
enum class JackknifePolicy : int {
    None         = 0,
    FeasibleOnly = 1,
    All          = 2,
};

// Per-call qpAdm options — reference §3
struct QpAdmOptions {
    double fudge = 1e-4;
    int als_iterations = 20;
    int rank = -1;
    bool allow_negative_weights = true;
    double rank_alpha = 0.05;
    JackknifePolicy jackknife = JackknifePolicy::All;
    double p_se_threshold = 0.05;
    bool se_require_p = false;
};

// The model to fit (population indices) — reference §4
struct QpAdmModel {
    int target = -1;
    std::vector<int> left;
    std::vector<int> right;
    int model_index = -1;
};

// Single-model fit result — reference §5
struct QpAdmResult {
    std::vector<double> weight;
    std::vector<double> se;
    std::vector<double> z;
    double p = 0.0;
    double chisq = 0.0;
    int dof = 0;
    std::vector<double> rank_p;
    int est_rank = 0;
    std::vector<double> rank_chisq;
    std::vector<int>    rank_dof;
    int                 f4rank = 0;
    std::vector<int>    rankdrop_f4rank, rankdrop_dof, rankdrop_dofdiff;
    std::vector<double> rankdrop_chisq, rankdrop_p, rankdrop_chisqdiff, rankdrop_p_nested;
    std::vector<std::string> popdrop_pat;
    std::vector<int>         popdrop_wt, popdrop_dof, popdrop_f4rank;
    std::vector<double>      popdrop_chisq, popdrop_p;
    std::vector<char>        popdrop_feasible;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
    int model_index = -1;
};

// Single-model entry points: run_qpadm — reference §6
[[nodiscard]] QpAdmResult run_qpadm(const device::DeviceF2Blocks& f2,
                                    const QpAdmModel& model,
                                    const QpAdmOptions& opts,
                                    device::Resources& resources);

[[nodiscard]] QpAdmResult run_qpadm(const F2BlockTensor& f2_host,
                                    const QpAdmModel& model,
                                    const QpAdmOptions& opts,
                                    device::Resources& resources);

// Model-space search: run_qpadm_search — reference §7
[[nodiscard]] std::vector<QpAdmResult> run_qpadm_search(
    const device::DeviceF2Blocks& f2,
    std::span<const QpAdmModel> models,
    const QpAdmOptions& opts,
    device::Resources& resources);

[[nodiscard]] std::vector<QpAdmResult> run_qpadm_search(
    const F2BlockTensor& f2_host,
    std::span<const QpAdmModel> models,
    const QpAdmOptions& opts,
    device::Resources& resources);

// qpWave: run_qpwave and QpWaveResult — reference §8
struct QpWaveResult {
    std::vector<double> rank_chisq;
    std::vector<int>    rank_dof;
    std::vector<double> rank_p;
    std::vector<int>    rankdrop_f4rank, rankdrop_dof, rankdrop_dofdiff;
    std::vector<double> rankdrop_chisq, rankdrop_p, rankdrop_chisqdiff, rankdrop_p_nested;
    int                 f4rank = 0;
    int                 est_rank = 0;
    Status              status = Status::Ok;
    Precision::Kind     precision_tag = Precision::Kind::Fp64;
};

[[nodiscard]] QpWaveResult run_qpwave(const device::DeviceF2Blocks& f2,
                                      std::span<const int> left,
                                      std::span<const int> right,
                                      const QpAdmOptions& opts,
                                      device::Resources& resources);

[[nodiscard]] QpWaveResult run_qpwave(const F2BlockTensor& f2_host,
                                      std::span<const int> left,
                                      std::span<const int> right,
                                      const QpAdmOptions& opts,
                                      device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPADM_HPP
