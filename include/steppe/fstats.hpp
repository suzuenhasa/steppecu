// include/steppe/fstats.hpp
//
// The public f-statistics result type F2BlockTensor: the per-block f2 tensor
// (plus its retained per-block pairwise-valid count) that steppe's GPU precompute
// engine hands to the fit engine. A CUDA-free, host-accessible interchange
// artifact whose storage is always FP64.
//
// Reference: docs/reference/include_steppe_fstats.hpp.md
#ifndef STEPPE_FSTATS_HPP
#define STEPPE_FSTATS_HPP

#include <cstddef>
#include <span>
#include <vector>

namespace steppe {

// F2BlockTensor: per-block f2 tensor, layout + diagonal convention — reference §4, §6
struct F2BlockTensor {
    // Fields: f2, vpair, block metadata — reference §5
    std::vector<double> f2;
    std::vector<double> vpair;
    std::vector<int> block_sizes;
    int P = 0;
    int n_block = 0;

    // Accessors: size, f2_at, vpair_at, block — reference §7
    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
               static_cast<std::size_t>(n_block);
    }

    [[nodiscard]] double f2_at(int i, int j, int b) const noexcept {
        return f2[flat_index(i, j, b)];
    }
    double& f2_at(int i, int j, int b) noexcept { return f2[flat_index(i, j, b)]; }

    [[nodiscard]] double vpair_at(int i, int j, int b) const noexcept {
        return vpair[flat_index(i, j, b)];
    }
    double& vpair_at(int i, int j, int b) noexcept {
        return vpair[flat_index(i, j, b)];
    }

    [[nodiscard]] std::span<const double> block(int b) const noexcept {
        const std::size_t slab =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
        return std::span<const double>{
            f2.data() + slab * static_cast<std::size_t>(b), slab};
    }

  private:
    // flat_index: the one canonical index formula — reference §7
    [[nodiscard]] std::size_t flat_index(int i, int j, int b) const noexcept {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(P) * static_cast<std::size_t>(j) +
               static_cast<std::size_t>(P) * static_cast<std::size_t>(P) *
                   static_cast<std::size_t>(b);
    }
};

}  // namespace steppe

#endif  // STEPPE_FSTATS_HPP
