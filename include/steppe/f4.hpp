// include/steppe/f4.hpp
//
// Public, CUDA-free entry point for the standalone f4 statistic — a sibling of
// the qpAdm entry point (qpadm.hpp), not a fork of it. Populations are P-axis
// indices into the device-resident f2 blocks; the GPU types are forward-declared
// so the header stays CUDA-free.
//
// Reference: docs/reference/include_steppe_f4.hpp.md
#ifndef STEPPE_F4_HPP
#define STEPPE_F4_HPP

#include <array>
#include <span>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpadm.hpp"

namespace steppe {

// f4 result table — reference §3
struct F4Result {
    std::vector<int>    p1, p2, p3, p4;
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;
    std::vector<double> p;

    Status status = Status::Ok;

    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// Two-sided normal tail p-value — reference §4
[[nodiscard]] double f4_two_sided_p(double z);

// run_f4 entry points — reference §5
[[nodiscard]] F4Result run_f4(const device::DeviceF2Blocks& f2,
                              std::span<const std::array<int, 4>> quartets,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

[[nodiscard]] F4Result run_f4(const F2BlockTensor& f2_host,
                              std::span<const std::array<int, 4>> quartets,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_F4_HPP
