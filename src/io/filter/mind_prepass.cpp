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

    // `active` is the --mind request flag: when --mind is not requested
    // (mind_max_missing >= 1.0) every sample is kept. Note `active` gates ONLY the
    // drop decision (consulted at the kept-set resolution below) — NOT the streaming
    // pass: the per-SNP missing-fraction count loop runs whenever packed data is
    // present (in.packed != nullptr && n_snp > 0), regardless of `active`, so the
    // missing_frac report is always populated when there is data to measure (the
    // pass is not short-circuited on the no-op default — doing so would drop that
    // reporting and change behavior; architecture.md §5 S-1).
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
                // Byte index s/kCodesPerByte and in-byte position s%kCodesPerByte:
                // the packing radix is single-homed in io::kCodesPerByte (the same 4
                // that packed_bytes / code_in_byte derive from), never re-spelled here.
                constexpr auto kPerByte = static_cast<std::size_t>(io::kCodesPerByte);
                const std::uint8_t byte = rec[s / kPerByte];
                const std::uint8_t code =
                    code_in_byte(byte, static_cast<int>(s % kPerByte));
                if (code != kMissingCode) ++nm;
            }
            out.nonmissing[g] = nm;
            out.missing_frac[g] =
                1.0 - static_cast<double>(nm) / static_cast<double>(n_snp);
        }
    } else {
        // No SNPs (or no data): the missing fraction is UNDEFINED. We treat every
        // sample as 0-missing so the no-op default keeps everyone AND an active
        // filter still keeps everyone (frac 0 <= any threshold) — the no-data
        // fail-safe is keep-all, never drop-all: with zero SNPs there is no evidence
        // to drop a sample on. This is the OPPOSITE of snp_filter's empty-denominator
        // convention (frac 1.0 ⇒ drop) and the divergence is intentional (the header
        // documents why). No write is needed here: `nonmissing` is already 0 and
        // `missing_frac` is already the default 0.0 from the assign() above — the
        // keep-all convention is carried entirely by those defaults. [3.4]
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
