// src/io/ploidy_detect.cpp
//
// AT2 pseudo-haploid per-sample ploidy auto-detection: scans each gathered
// sample's packed SNP-prefix bytes for a het call within the AT2 window.
#include "io/ploidy_detect.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "io/eigenstrat_format.hpp"

namespace steppe::io {

std::vector<int> detect_sample_ploidy(const GenotypeTile& tile) {
    constexpr int kPloidyPseudoHaploid = 1;
    constexpr int kPloidyDiploid = 2;
    std::vector<int> ploidy(tile.n_individuals, kPloidyPseudoHaploid);
    const std::size_t window = std::min(kPloidyDetectSnps, tile.n_snp);
    if (window == 0 || tile.n_individuals == 0) return ploidy;

    for (std::size_t g = 0; g < tile.n_individuals; ++g) {
        const std::uint8_t* rec = tile.packed.data() + g * tile.bytes_per_record;
        for (std::size_t s = 0; s < window; ++s) {
            const std::size_t byte_in_rec =
                s / static_cast<std::size_t>(kCodesPerByte);
            const int pos_in_byte =
                static_cast<int>(s % static_cast<std::size_t>(kCodesPerByte));
            if (code_in_byte(rec[byte_in_rec], pos_in_byte) == kHetCode) {
                ploidy[g] = kPloidyDiploid;
                break;
            }
        }
    }
    return ploidy;
}

}  // namespace steppe::io
