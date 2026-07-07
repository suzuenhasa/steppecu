// Reference: docs/reference/src_core_readv2_readv2_classify.cpp.md
// src/core/readv2/readv2_classify.cpp — the pure-host degree classifier + z statistic.
// No device, no I/O; the single native-FP64 arithmetic the concord gate measures.
#include "core/readv2/readv2_classify.hpp"

#include <cmath>
#include <cstring>

namespace steppe::core::readv2 {

const char* degree_from_p0norm(double p0_norm) noexcept {
    if (std::isnan(p0_norm)) return kDegreeUnrelated;  // undefined -> most-distant token
    if (p0_norm < kCutIdentical) return kDegreeIdentical;
    if (p0_norm < kCutFirst) return kDegreeFirst;
    if (p0_norm < kCutSecond) return kDegreeSecond;
    return kDegreeUnrelated;
}

double boundary_for_degree(const char* degree) noexcept {
    if (std::strcmp(degree, kDegreeIdentical) == 0) return kCutIdentical;
    if (std::strcmp(degree, kDegreeFirst) == 0) return kCutFirst;
    if (std::strcmp(degree, kDegreeSecond) == 0) return kCutSecond;
    return kCutSecond;  // unrelated: nearest boundary (no more-distant class)
}

double readv2_z(double p0_mean, double p0_norm, double background, int n_windows,
                double sum_p0_sq) noexcept {
    if (n_windows < 2 || !(background > 0.0) || std::isnan(p0_mean) || std::isnan(p0_norm)) {
        return std::nan("");
    }
    // Window block-jackknife-style variance of P0_mean (windows are the blocks).
    const double n = static_cast<double>(n_windows);
    double var_windows = (sum_p0_sq - n * p0_mean * p0_mean) / (n - 1.0);
    if (var_windows < 0.0) var_windows = 0.0;  // guard tiny negative from cancellation
    const double se_p0_mean = std::sqrt(var_windows / n);
    const double se_p0_norm = se_p0_mean / background;
    if (!(se_p0_norm > 0.0)) return std::nan("");

    const char* degree = degree_from_p0norm(p0_norm);
    const double boundary = boundary_for_degree(degree);
    return (boundary - p0_norm) / se_p0_norm;  // Z_upper: >0 => P0_norm sits below the boundary
}

}  // namespace steppe::core::readv2
