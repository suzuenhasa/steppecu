// include/steppe/dates.hpp
//
// Public, CUDA-free entry point for admixture dating — the DATES tool. Given a
// target (admixed) population and two source populations, estimate how many
// generations ago they mixed. Unlike the f-statistic tools it reads the genotype
// triple directly and runs an FFT-based decay-curve engine; it never touches the
// f2 cache.
//
// Reference: docs/reference/include_steppe_dates.hpp.md
#ifndef STEPPE_DATES_HPP
#define STEPPE_DATES_HPP

#include <string>
#include <vector>

#include "steppe/config.hpp"
#include "steppe/error.hpp"

namespace steppe {

// CUDA-free device seam — reference §6
namespace device {
struct Resources;
}  // namespace device

// The run knobs — reference §3
struct DatesOptions {
    double binsize_morgans = 0.001;
    int qbin = 10;
    double maxdis_morgans = 1.0;
    double lovalfit_cm = 0.45;
    bool affine = true;
    long seed = 77;
};

// The run output — reference §4
struct DatesResult {
    double date_gen = 0.0;
    double se = 0.0;
    double fit_error_sd = 0.0;
    std::vector<double> curve_cm;
    std::vector<double> curve_corr;
    std::vector<double> loo_date_gen;
    std::vector<double> loo_weight;

    Status status = Status::Ok;
    Precision::Kind precision_tag = Precision::Kind::Fp64;
};

// The dating entry point — reference §5
[[nodiscard]] DatesResult run_dates(const std::string& geno, const std::string& snp,
                                    const std::string& ind, const std::string& target,
                                    const std::string& source1, const std::string& source2,
                                    const DatesOptions& opts, device::Resources& resources);

}  // namespace steppe

#endif  // STEPPE_DATES_HPP
