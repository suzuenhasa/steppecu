// include/steppe/fstat_sweep.hpp
//
// Public, CUDA-free entry point for the all-combinations f-stat sweep — the
// production-scale sibling of run_f4 / run_f3. A sweep enumerates every C(P,k)
// quartet/triple, scores and filters on the GPU, and returns only the survivors;
// the full result table is never materialized on the host.
//
// Reference: docs/reference/include_steppe_fstat_sweep.hpp.md
#ifndef STEPPE_FSTAT_SWEEP_HPP
#define STEPPE_FSTAT_SWEEP_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/qpadm.hpp"

namespace steppe {

// SweepFilter — reference §4
enum class SweepFilter {
    MinZ,
    TopK,
};

// SweepRequest — reference §5
struct SweepRequest {
    SweepFilter filter = SweepFilter::MinZ;
    double min_z = 3.0;
    std::size_t top_k = 100;
    std::vector<int> pop_subset;
    bool sure = false;
};

// SweepResult — reference §6
struct SweepResult {
    std::vector<std::array<int, 4>> keys;
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;
    std::vector<double> p;

    std::size_t enumerated = 0;
    std::size_t survivors = 0;
    bool capped = false;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// Sweep entry points — reference §7
[[nodiscard]] SweepResult run_f4_sweep(const device::DeviceF2Blocks& f2,
                                       const SweepRequest& req,
                                       device::Resources& resources);

[[nodiscard]] SweepResult run_f3_sweep(const device::DeviceF2Blocks& f2,
                                       const SweepRequest& req,
                                       device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_FSTAT_SWEEP_HPP
