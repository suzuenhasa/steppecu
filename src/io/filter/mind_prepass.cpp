// src/io/filter/mind_prepass.cpp
//
// The conditional S-1 --mind streaming pre-pass (architecture.md §5 S-1; ROADMAP
// M2). Accumulates per-sample non-missing counts over ALL SNPs and resolves the
// kept-sample set. Host-pure `io`-leaf TU; reuses io/eigenstrat_format.hpp's byte
// helpers (the same bit order as the M1 decode path).
#include "io/filter/mind_prepass.hpp"

#include <cstddef>
#include <cstdint>

#include "io/eigenstrat_format.hpp"       // code_in_byte, kMissingCode (io-side byte path)
#include "io/filter/filter_decision.hpp"  // sample_passes_mind (the shared predicate)

namespace steppe::io::filter {

MindSummary run_mind_prepass(const MindPrepassInput& in, const FilterConfig& cfg) {
    const std::size_t n_ind = in.n_individuals;
    const std::size_t n_snp = in.n_snp;

    MindSummary out;
    out.nonmissing.assign(n_ind, 0);
    out.missing_frac.assign(n_ind, 0.0);
    out.kept.reserve(n_ind);

    // No-op fast path: when --mind is not requested (mind_max_missing >= 1.0) keep
    // every sample (architecture.md §5 S-1: the pre-pass is skipped entirely). We
    // still report the missing fractions if we have data, but do not stream when
    // there is nothing to decide.
    const bool active = cfg.mind_max_missing < 1.0;

    if (in.packed != nullptr && n_snp > 0) {
        // One streaming pass over the packed records, counting non-missing SNPs per
        // sample. Same bit extraction as the decode front-end (io::code_in_byte,
        // MSB-first) and the same missing sentinel (io::kMissingCode == code 3), so
        // "non-missing" here is identical to the decode's notion.
        for (std::size_t g = 0; g < n_ind; ++g) {
            const std::uint8_t* rec = in.packed + g * in.bytes_per_record;
            std::size_t nm = 0;
            for (std::size_t s = 0; s < n_snp; ++s) {
                const std::uint8_t byte = rec[s / 4u];
                const std::uint8_t code = code_in_byte(byte, static_cast<int>(s % 4u));
                if (code != kMissingCode) ++nm;
            }
            out.nonmissing[g] = nm;
            out.missing_frac[g] =
                1.0 - static_cast<double>(nm) / static_cast<double>(n_snp);
        }
    } else {
        // No SNPs (or no data): missing fraction is undefined; treat as 0 missing so
        // the no-op default keeps everyone, and an active filter sees frac 0 too
        // (nothing to base a drop on without SNPs).
        for (std::size_t g = 0; g < n_ind; ++g) out.missing_frac[g] = 0.0;
    }

    // Resolve the kept-sample set via the shared predicate. At the no-op default
    // (not active) every sample passes (missing_frac <= 1.0 always); when active,
    // a sample is dropped iff its missing fraction exceeds the threshold.
    for (std::size_t g = 0; g < n_ind; ++g) {
        const double frac = active ? out.missing_frac[g] : 0.0;
        if (sample_passes_mind(frac, cfg.mind_max_missing)) {
            out.kept.push_back(g);
        }
    }
    return out;
}

}  // namespace steppe::io::filter
