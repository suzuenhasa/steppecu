// include/steppe/f4ratio.hpp
//
// Public, CUDA-free entry point for f4-ratios (AT2 qpf4ratio): alpha =
// f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4). Sibling of f4.hpp / f3.hpp — a small result
// struct plus two run_f4ratio overloads; no ALS solve and no rank test.
//
// Reference: docs/reference/include_steppe_f4ratio.hpp.md
#ifndef STEPPE_F4RATIO_HPP
#define STEPPE_F4RATIO_HPP

#include <array>
#include <span>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"
#include "steppe/qpadm.hpp"

namespace steppe {

// F4RatioResult — reference §4
struct F4RatioResult {
    std::vector<int>    p1, p2, p3, p4, p5;
    std::vector<double> alpha;
    std::vector<double> se;
    std::vector<double> z;

    Status status = Status::Ok;

    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// The two entry points — reference §5
[[nodiscard]] F4RatioResult run_f4ratio(const device::DeviceF2Blocks& f2,
                                        std::span<const std::array<int, 5>> tuples,
                                        const QpAdmOptions& opts,
                                        device::Resources& resources);

[[nodiscard]] F4RatioResult run_f4ratio(const F2BlockTensor& f2_host,
                                        std::span<const std::array<int, 5>> tuples,
                                        const QpAdmOptions& opts,
                                        device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_F4RATIO_HPP
