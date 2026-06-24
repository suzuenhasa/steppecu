// src/io/filter/snp_filter.cpp
//
// SNP-level keep-mask producer (architecture.md §1, §5 S0'; ROADMAP M2). Derives
// the per-SNP pooled folded MAF + per-SNP missing fraction from the decoded per-pop
// Q/V/N, then applies the SHARED filter_decision.hpp predicates + membership + the
// flag-gated options into one SNP-global keep mask. Host-pure `io`-leaf TU.
#include "io/filter/snp_filter.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "io/filter/filter_decision.hpp"     // the shared predicates (single source)
#include "io/filter/snp_summary_reduce.hpp"  // derive_pooled_summary_one (the shared __host__ __device__ reduction)

namespace steppe::io::filter {

std::vector<PerSnpSummary> derive_per_snp_summary(const DecodedTileSummaryInput& in) {
    const int P = in.P;
    const long M = in.M;

    std::vector<PerSnpSummary> out(static_cast<std::size_t>(M < 0 ? 0 : M));
    if (P <= 0 || M <= 0) return out;

    // Fail-fast on a nonsensical ploidy (architecture.md §2 fail-fast; cleanup
    // B10/F5/X-11). Ploidy is METADATA, never auto-detected, and {1, 2} are the
    // only meaningful values (decode_af.hpp). A non-positive / out-of-range ploidy
    // is a config/wiring error: the previous silent clamp-to-1 produced a
    // plausible-but-wrong missing fraction (it doubles nonmissing_indiv vs the
    // diploid truth) rather than surfacing the bug. This RECONCILES the illegal-
    // ploidy contract with the decode primitive: the device-side finalize_af has
    // no throw path on the GPU so it masks the cell OUT ({0,0,0}); this host `io`
    // leaf has a throw path, so it rejects up front — both refuse to fabricate a
    // trustworthy N from a bad ploidy. std::invalid_argument is the io-leaf idiom
    // (cf. include_exclude.cpp's runtime_error on a bad read).
    if (in.ploidy != 1 && in.ploidy != 2) {
        throw std::invalid_argument(
            "snp_filter: ploidy must be 1 (pseudo-haploid) or 2 (diploid); got " +
            std::to_string(in.ploidy));
    }

    // Fail-fast on a short `pop_individuals` (cleanup B20/F1). The header documents
    // it as "length P"; the prior code guarded only the DENOMINATOR loop below
    // (`p < pop_individuals.size()`) while the per-SNP NUMERATOR loop ran over ALL
    // P pops. A short vector then understated `total_indiv` (the denominator)
    // without dropping numerator terms, so `nonmissing_indiv > total_indiv` and
    // `missing_frac = 1 - nonmissing/total` went NEGATIVE — and snp_passes_geno
    // (negative <= geno_max_missing) is TRUE, spuriously KEEPING a missing-heavy
    // SNP. A SNP-axis sharding bug must surface here, not silently corrupt the geno
    // filter. (This also subsumes the static_cast<int>(size()) INT_MAX-truncation
    // footgun B20/F22: with size() == P the denominator loop is unconditional.)
    if (in.pop_individuals.size() != static_cast<std::size_t>(P)) {
        throw std::invalid_argument(
            "snp_filter: pop_individuals.size() (" +
            std::to_string(in.pop_individuals.size()) +
            ") must equal P (" + std::to_string(P) + ")");
    }

    // Fail-fast on null Q/N (cleanup B20/F6). The reduction dereferences both per
    // (pop, snp) cell; with P>0 && M>0 a null pointer (the struct's nullptr
    // defaults make this an easy caller mistake) would SIGSEGV three frames deep
    // rather than surface a diagnosable error (architecture.md §2). `v` is NOT
    // required: N==0 ⇔ V==0 ⇔ missing by the decode contract (finalize_af), so the
    // reduction uses N and never reads V.
    if (in.q == nullptr || in.n == nullptr) {
        throw std::invalid_argument(
            "snp_filter: q and n must be non-null when P>0 && M>0");
    }

    // Total kept individuals across the kept pops (the missing-fraction
    // denominator). SNP-independent: the sample axis is fixed for the tile.
    // `pop_individuals.size() == P` is pinned above, so this loop is unconditional.
    std::size_t total_indiv = 0;
    for (int p = 0; p < P; ++p) {
        total_indiv += in.pop_individuals[static_cast<std::size_t>(p)];
    }
    const double total_indiv_d = static_cast<double>(total_indiv);
    const double ploidy_d = static_cast<double>(in.ploidy);

    for (long s = 0; s < M; ++s) {
        // Reduce ACROSS the kept populations into SNP-global scalars (the §1, §5 S2
        // invariant) via the SHARED __host__ __device__ primitive (snp_summary_reduce
        // .hpp), so this host loop and the regime-B device keep-mask kernel cannot
        // diverge on a boundary — incl. the FFMA-immune Σ Q·N (the bit-exact pin).
        const PooledSnpSummary ps =
            derive_pooled_summary_one(in.q, in.n, P, s, ploidy_d, total_indiv_d);
        PerSnpSummary& sm = out[static_cast<std::size_t>(s)];
        sm.pooled_allele_count = ps.pooled_allele_count;
        sm.pooled_ref_af = ps.pooled_ref_af;
        sm.pooled_minor_af = ps.pooled_minor_af;
        sm.missing_frac = ps.missing_frac;
    }
    return out;
}

std::vector<bool> build_snp_keep_mask(const DecodedTileSummaryInput& in,
                                      const SnpTable& snps,
                                      const FilterConfig& cfg,
                                      const SnpMembership& mem) {
    const long M = in.M;
    std::vector<bool> keep(static_cast<std::size_t>(M < 0 ? 0 : M), false);
    if (M <= 0) return keep;

    const std::size_t Mu = static_cast<std::size_t>(M);

    // Fail-fast on a too-short SnpTable (cleanup B20/F3). The allele pair drives the
    // unconditional class drop for EVERY SNP, so ref/alt must cover all M; chrom and
    // id are required only when their filter / membership is active. The prior code
    // silently substituted 'N'/0/empty-id for a short table — which made
    // is_multiallelic('N','N') true and DROPPED the tile tail (or altered
    // membership) for a mismatched .snp/decode length, a silent wrong answer for a
    // wiring/data bug that must abort with context (architecture.md §2 fail-fast).
    const bool mem_noop = mem.is_noop();

    // Fold the per-field "requires this column to cover all M" preconditions: each is
    // the same throw shape gated on whether the consuming filter is active, differing
    // only by `what` (cleanup 7.4). Distinct diagnostics preserved via `what`.
    const auto require_at_least = [&](const char* what, std::size_t have, bool active) {
        if (active && have < Mu) {
            throw std::invalid_argument(
                std::string("snp_filter: ") + what + " >= M (" + std::to_string(M) +
                "); got " + std::to_string(have));
        }
    };

    // ref/alt drive the unconditional class drop for EVERY SNP (always active) and the
    // throw reports BOTH fields, so it keeps its dedicated two-field shape.
    if (snps.ref.size() < Mu || snps.alt.size() < Mu) {
        throw std::invalid_argument(
            "snp_filter: snps.ref/alt must have >= M (" + std::to_string(M) +
            ") entries; got ref=" + std::to_string(snps.ref.size()) +
            " alt=" + std::to_string(snps.alt.size()));
    }
    require_at_least("autosomes_only requires snps.chrom", snps.chrom.size(),
                     cfg.autosomes_only);
    require_at_least("active membership requires snps.id", snps.id.size(), !mem_noop);

    const std::vector<PerSnpSummary> summary = derive_per_snp_summary(in);

    for (long s = 0; s < M; ++s) {
        const std::size_t si = static_cast<std::size_t>(s);

        // chrom is read only when autosomes_only is active (and then guaranteed
        // present by the precondition above); 0 is an inert placeholder otherwise.
        const int chrom = cfg.autosomes_only ? snps.chrom[si] : 0;
        // Membership is precomputed here (the decision primitive is pure and takes a
        // bool) so the future M4.5 device kernel can share snp_keep_decision without
        // a SnpMembership dependency. A no-op membership ⇒ always passes.
        const bool membership_ok = mem_noop ? true : mem.passes(snps.id[si]);

        // The SINGLE shared per-SNP decision (drop-not-flip; §1, §8 single source).
        keep[si] = snp_keep_decision(summary[si], snps.ref[si], snps.alt[si], chrom,
                                     cfg, membership_ok);
    }
    return keep;
}

}  // namespace steppe::io::filter
