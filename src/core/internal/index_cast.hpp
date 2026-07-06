// src/core/internal/index_cast.hpp
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

}  // namespace steppe::core

#endif  // STEPPE_CORE_INTERNAL_INDEX_CAST_HPP
