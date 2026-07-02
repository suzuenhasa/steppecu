// include/steppe/qpfstats.hpp
//
// Public, CUDA-free entry point for qpfstats — the joint f-statistic smoother.
// Reads the genotype triple directly, jointly fits every f2/f3/f4 over the given
// populations, and emits a smoothed per-block f2 tensor in the f2-cache format that
// qpAdm / f4 / qpGraph consume. Only forward-declares the GPU resource type, so the
// header stays standard-C++ and needs no device layer.
//
// Reference: docs/reference/include_steppe_qpfstats.hpp.md
#ifndef STEPPE_QPFSTATS_HPP
#define STEPPE_QPFSTATS_HPP

#include <span>
#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/fstats.hpp"

namespace steppe {

namespace device {
struct Resources;
}  // namespace device

// QpfstatsResult — reference §7
struct QpfstatsResult {
    F2BlockTensor f2;
    std::vector<std::string> pop_labels;
    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// run_qpfstats — reference §8
[[nodiscard]] QpfstatsResult run_qpfstats(const std::string& geno,
                                          const std::string& snp,
                                          const std::string& ind,
                                          std::span<const std::string> pops,
                                          double blgsize_morgans,
                                          const Precision& precision,
                                          device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_QPFSTATS_HPP
