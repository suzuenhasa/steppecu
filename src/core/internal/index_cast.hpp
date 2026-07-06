// src/core/internal/index_cast.hpp
//
// Reference: docs/reference/src_core_internal_index_cast.hpp.md
//
// idx() — the single signed→size_t index-cast helper. Lets the flat index math in
// core's host-reference headers read as idx(i) + idx(n) * idx(j) instead of spelling
// out static_cast<std::size_t> at every subscript. Host-pure, constexpr.
#ifndef STEPPE_CORE_INTERNAL_INDEX_CAST_HPP
#define STEPPE_CORE_INTERNAL_INDEX_CAST_HPP

#include <cstddef>

namespace steppe::core {

[[nodiscard]] constexpr std::size_t idx(long i) noexcept {
    return static_cast<std::size_t>(i);
}

// ld() — the single long-double promotion helper. Lets the host long-double
// accumulation in core's CPU-oracle reference paths read as ld(x) instead of
// spelling out static_cast<long double> at every term. Every arg type wrapped
// (double / int / long / std::uint8_t) converts to long double value-preserving,
// so this is a bit-identical wrap of the explicit cast. Host-pure, constexpr.
template <class T>
[[nodiscard]] constexpr long double ld(T x) noexcept {
    return static_cast<long double>(x);
}

// nonneg_count() — idx() of a signed count, clamping a negative to 0. The size_t
// analogue of idx() for lengths/counts that must never wrap on an uninitialized
// (negative) shape. Host-pure, constexpr.
[[nodiscard]] constexpr std::size_t nonneg_count(int n) noexcept {
    return idx(n < 0 ? 0 : n);
}
[[nodiscard]] constexpr std::size_t nonneg_count(long n) noexcept {
    return idx(n < 0 ? 0 : n);
}

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_INDEX_CAST_HPP
