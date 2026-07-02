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

}  // namespace steppe

#endif  // STEPPE_ERROR_HPP
