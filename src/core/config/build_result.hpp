// src/core/config/build_result.hpp
//
// BuildResult<T> — a CUDA-free, C++20, header-only stand-in for
// std::expected<T, Status>: it carries either a built value or a steppe::Status
// error. Retire it for std::expected once the toolchain moves to C++23.
//
// Reference: docs/reference/src_core_config_build_result.hpp.md
#ifndef STEPPE_CORE_CONFIG_BUILD_RESULT_HPP
#define STEPPE_CORE_CONFIG_BUILD_RESULT_HPP

#include <optional>
#include <utility>

#include "steppe/error.hpp"

namespace steppe::config {

// The error arm: Unexpected + unexpected() — reference §2
struct Unexpected {
    Status status;
};

[[nodiscard]] inline Unexpected unexpected(Status s) noexcept { return Unexpected{s}; }

// BuildResult<T> — reference §3
template <typename T>
class BuildResult {
public:
    BuildResult(T value) : value_(std::move(value)) {}  // NOLINT(google-explicit-constructor)

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

    [[nodiscard]] Status error() const noexcept { return error_; }

private:
    std::optional<T> value_;
    Status error_ = Status::Ok;
};

}  // namespace steppe::config

#endif  // STEPPE_CORE_CONFIG_BUILD_RESULT_HPP
