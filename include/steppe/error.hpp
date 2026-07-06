// include/steppe/error.hpp
//
// The public outcome codes steppe returns from its early layers — a single
// strongly-typed Status enum. CUDA-free and standard-library-free, and the
// stand-in for the full C ABI that is still deferred.
//
// Reference: docs/reference/include_steppe_error.hpp.md
#ifndef STEPPE_ERROR_HPP
#define STEPPE_ERROR_HPP

namespace steppe {

// Status taxonomy — reference §3
enum class Status {
    Ok,

    DeviceOom,

    RankDeficient,

    NonSpdCovariance,

    ChisqUndefined,

    InvalidConfig
};

// Stable snake_case label for a Status — reference §3
[[nodiscard]] inline const char* status_str(Status s) {
    switch (s) {
        case Status::Ok:               return "ok";
        case Status::DeviceOom:        return "device_oom";
        case Status::RankDeficient:    return "rank_deficient";
        case Status::NonSpdCovariance: return "non_spd_covariance";
        case Status::ChisqUndefined:   return "chisq_undefined";
        case Status::InvalidConfig:    return "invalid_config";
    }
    return "unknown";
}

}  // namespace steppe

#endif  // STEPPE_ERROR_HPP
