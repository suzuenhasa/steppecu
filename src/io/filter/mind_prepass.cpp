// src/io/filter/mind_prepass.cpp
//
// The conditional --mind streaming pre-pass: one host pass over the packed
// records that counts per-sample non-missing SNPs and resolves the kept-sample
// set. Reuses io/eigenstrat_format.hpp's byte helpers so "non-missing" matches
// the decode path exactly.
#include "io/filter/mind_prepass.hpp"

#include <cstddef>
#include <cstdint>

#include "io/eigenstrat_format.hpp"
#include "io/filter/filter_decision.hpp"

namespace steppe::io::filter {

constexpr double kNoMissingFrac = 0.0;

MindSummary run_mind_prepass(const MindPrepassInput& in, const FilterConfig& cfg) {
    const std::size_t n_ind = in.n_individuals;
    const std::size_t n_snp = in.n_snp;

    MindSummary out;
    out.nonmissing.assign(n_ind, 0);
    out.missing_frac.assign(n_ind, kNoMissingFrac);
    out.kept.reserve(n_ind);

    const bool active = cfg.mind_max_missing < kMindFilterInactiveThreshold;

    if (in.packed != nullptr && n_snp > 0) {
        const double n_snp_d = static_cast<double>(n_snp);
        constexpr auto kPerByte = static_cast<std::size_t>(io::kCodesPerByte);
        for (std::size_t ind = 0; ind < n_ind; ++ind) {
            const std::uint8_t* rec = in.packed + ind * in.bytes_per_record;
            std::size_t nonmissing_count = 0;
            for (std::size_t s = 0; s < n_snp; ++s) {
                const std::uint8_t byte = rec[s / kPerByte];
                const std::uint8_t code =
                    code_in_byte(byte, static_cast<int>(s % kPerByte));
                if (code != kMissingCode) ++nonmissing_count;
            }
            out.nonmissing[ind] = nonmissing_count;
            out.missing_frac[ind] =
                1.0 - static_cast<double>(nonmissing_count) / n_snp_d;
        }
    }

    for (std::size_t ind = 0; ind < n_ind; ++ind) {
        const double frac = active ? out.missing_frac[ind] : kNoMissingFrac;
        if (sample_passes_mind(frac, cfg.mind_max_missing)) {
            out.kept.push_back(ind);
        }
    }
    return out;
}

}  // namespace steppe::io::filter
