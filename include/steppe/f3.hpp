// include/steppe/f3.hpp
//
// Public, CUDA-free entry point for the standalone f3 statistic. For each input
// triple (C; A, B) it returns the block-jackknife point estimate plus its
// standard error, z-score, and p-value. The device types appear only as forward
// declarations, so the header includes without pulling in the GPU toolchain.
//
// Reference: docs/reference/include_steppe_f3.hpp.md
#ifndef STEPPE_F3_HPP
#define STEPPE_F3_HPP

#include <array>
#include <span>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/f4.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpadm.hpp"

namespace steppe {

// f3 result table — reference §5
struct F3Result {
    std::vector<int>    p1, p2, p3;
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;
    std::vector<double> p;

    Status status = Status::Ok;

    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// f3 entry points: device-resident primary + host-oracle parity overload — reference §6
[[nodiscard]] F3Result run_f3(const device::DeviceF2Blocks& f2,
                              std::span<const std::array<int, 3>> triples,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

[[nodiscard]] F3Result run_f3(const F2BlockTensor& f2_host,
                              std::span<const std::array<int, 3>> triples,
                              const QpAdmOptions& opts,
                              device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_F3_HPP
