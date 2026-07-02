// include/steppe/qpgraph.hpp
//
// Public value types and entry points for fitting one fixed admixture graph to
// observed f-statistics (the qpGraph operation; topology SEARCH is deferred).
// Host-only and CUDA-free, so it compiles into the core, CLI, and Python
// bindings without pulling in the device layer; the two GPU-side types it
// references are forward-declared below.
//
// Reference: docs/reference/include_steppe_qpgraph.hpp.md
#ifndef STEPPE_QPGRAPH_HPP
#define STEPPE_QPGRAPH_HPP

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

// One directed parent->child edge of the admixture graph — reference §2
struct QpGraphEdge {
    std::string from;
    std::string to;
};

// Per-call fit settings (AT2 qpgraph() parity defaults) — reference §3
struct QpGraphOptions {
    double fudge = 1e-4;

    double diag_f3 = 1e-5;

    int numstart = 10;

    bool constrained = true;

    int maxit = 200;

    double tol = 1e-9;
};

// Single-graph fit result — reference §4
struct QpGraphResult {
    std::vector<double> weight;
    std::vector<double> weight_lo;
    std::vector<double> weight_hi;
    std::vector<std::string> admix_from;
    std::vector<std::string> admix_to;

    std::vector<double> edge_length;
    std::vector<std::string> edge_from;
    std::vector<std::string> edge_to;

    double score = 0.0;
    double restart_spread = 0.0;

    double worst_residual_z = 0.0;
    std::string worst_pop2;
    std::string worst_pop3;

    std::vector<std::string> leaves;

    Status status = Status::Ok;

    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// Entry points — reference §5
[[nodiscard]] QpGraphResult run_qpgraph(const device::DeviceF2Blocks& f2,
                                        const std::vector<QpGraphEdge>& edges,
                                        const std::vector<std::string>& leaf_names,
                                        const QpGraphOptions& opts,
                                        device::Resources& resources);

[[nodiscard]] QpGraphResult run_qpgraph(const F2BlockTensor& f2_host,
                                        const std::vector<QpGraphEdge>& edges,
                                        const std::vector<std::string>& leaf_names,
                                        const QpGraphOptions& opts,
                                        device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPGRAPH_HPP
