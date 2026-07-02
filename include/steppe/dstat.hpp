// include/steppe/dstat.hpp
//
// Public, CUDA-free entry point for the genotype-path normalized D statistic
// (qpDstat Part B): reads the .geno/.snp/.ind triple through the extract-f2
// decode front-end, then branches into its own per-SNP D kernel and never
// touches the f2 cache. Declares DstatResult and run_dstat.
//
// Reference: docs/reference/include_steppe_dstat.hpp.md
#ifndef STEPPE_DSTAT_HPP
#define STEPPE_DSTAT_HPP

#include <array>
#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

// DstatResult — reference §5
struct DstatResult {
    std::vector<int>    p1, p2, p3, p4;
    std::vector<double> est;
    std::vector<double> se;
    std::vector<double> z;
    std::vector<double> p;

    Status status = Status::Ok;

    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_dstat — reference §6
[[nodiscard]] DstatResult run_dstat(const std::string& geno,
                                    const std::string& snp,
                                    const std::string& ind,
                                    std::span<const std::string> pop_union,
                                    std::span<const std::array<int, 4>> quadruples,
                                    double blgsize_morgans,
                                    device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_DSTAT_HPP
