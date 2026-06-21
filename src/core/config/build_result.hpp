// src/core/config/build_result.hpp
//
// BuildResult<T> — a minimal, CUDA-FREE, C++20 stand-in for std::expected<T, Status>
// (which is C++23 and the project compiles -std=c++20; the architecture §9 sketch uses
// std::expected as design intent). It carries EITHER a value T OR a steppe::Status
// error, with the small slice of the std::expected interface the access layer uses:
//   has_value() / explicit operator bool   — did build() succeed?
//   value() / operator* / operator->        — the RunConfig (UB if !has_value, like expected)
//   error()                                 — the Status (the InvalidConfig fault category)
// plus an `unexpected(Status)` free function mirroring std::unexpected so build()'s
// failure sites read identically to the §9 std::expected sketch. When the toolchain
// moves to C++23 this can be retired for std::expected with no call-site churn.
//
// It is header-only and depends only on <steppe/error.hpp> + the standard library, so
// it compiles into core, the CLI, and the bindings without the device toolkit (§4).
#ifndef STEPPE_CORE_CONFIG_BUILD_RESULT_HPP
#define STEPPE_CORE_CONFIG_BUILD_RESULT_HPP

#include <optional>
#include <utility>

#include "steppe/error.hpp"  // steppe::Status

namespace steppe::config {

/// The error-carrying tag (mirrors std::unexpected<Status>) so a build() failure site
/// reads `return unexpected(Status::InvalidConfig);` exactly as the §9 sketch does.
struct Unexpected {
    Status status;
};

/// Construct the error arm of a BuildResult (the std::unexpected analogue).
[[nodiscard]] inline Unexpected unexpected(Status s) noexcept { return Unexpected{s}; }

/// A value-or-Status result. Default-constructed-from-value or from `unexpected(...)`.
template <typename T>
class BuildResult {
public:
    // Value arm (implicit, like std::expected's value-converting ctor).
    BuildResult(T value) : value_(std::move(value)) {}  // NOLINT(google-explicit-constructor)

    // Error arm (from `unexpected(Status)`).
    BuildResult(Unexpected u) : error_(u.status) {}  // NOLINT(google-explicit-constructor)

    [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }
    explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] const T& value() const& noexcept { return *value_; }
    [[nodiscard]] T& value() & noexcept { return *value_; }
    [[nodiscard]] T&& value() && noexcept { return std::move(*value_); }

    [[nodiscard]] const T& operator*() const& noexcept { return *value_; }
    [[nodiscard]] T& operator*() & noexcept { return *value_; }
    [[nodiscard]] T&& operator*() && noexcept { return std::move(*value_); }
    [[nodiscard]] const T* operator->() const noexcept { return &*value_; }
    [[nodiscard]] T* operator->() noexcept { return &*value_; }

    /// The Status on the error arm (valid only when !has_value(); like expected::error).
    [[nodiscard]] Status error() const noexcept { return error_; }

private:
    std::optional<T> value_;
    Status error_ = Status::Ok;  // meaningful only when value_ is empty
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_BUILD_RESULT_HPP
