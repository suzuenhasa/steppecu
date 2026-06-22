// src/io/ploidy_detect.cpp
//
// Implementation of the AT2 pseudo-haploid per-sample ploidy auto-detection
// (ploidy_detect.hpp). Scans each gathered sample's packed SNP-prefix bytes for a
// heterozygous call within the AT2 detection window.
#include "io/ploidy_detect.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "io/eigenstrat_format.hpp"  // code_in_byte, kHetCode, kPloidyDetectSnps, kCodesPerByte

namespace steppe::io {

std::vector<int> detect_sample_ploidy(const GenotypeTile& tile) {
    // Initialize every sample to PSEUDO-HAPLOID (ploidy 1), the AT2 default; a het
    // call in the window bumps it to DIPLOID (ploidy 2). {1, 2} are the only AT2
    // ploidy values (mirrors core::kPloidy{PseudoHaploid,Diploid}; the io leaf does
    // not depend on core, so the values are domain literals here).
    constexpr int kPloidyPseudoHaploid = 1;
    constexpr int kPloidyDiploid = 2;
    std::vector<int> ploidy(tile.n_individuals, kPloidyPseudoHaploid);
    // AT2 ntest, capped at the SNPs this tile actually carries (a shorter record
    // scans its whole prefix — exactly what AT2 does when ntest > nsnp).
    const std::size_t window = std::min(kPloidyDetectSnps, tile.n_snp);
    if (window == 0 || tile.n_individuals == 0) return ploidy;

    for (std::size_t g = 0; g < tile.n_individuals; ++g) {
        const std::uint8_t* rec = tile.packed.data() + g * tile.bytes_per_record;
        // Bump to diploid (ploidy 2) on the FIRST het call (code == 1) in the window;
        // a haploid genome cannot be heterozygous, so any het ⇒ diploid (AT2 rule).
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
