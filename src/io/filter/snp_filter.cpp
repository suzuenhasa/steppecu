// src/io/filter/snp_filter.cpp
//
// SNP-level keep-mask producer (architecture.md §1, §5 S0'; ROADMAP M2). Derives
// the per-SNP pooled folded MAF + per-SNP missing fraction from the decoded per-pop
// Q/V/N, then applies the SHARED filter_decision.hpp predicates + membership + the
// flag-gated options into one SNP-global keep mask. Host-pure `io`-leaf TU.
#include "io/filter/snp_filter.hpp"

#include <cstddef>
#include <string>

#include "io/filter/filter_decision.hpp"  // the shared predicates (single source)

namespace steppe::io::filter {

std::vector<PerSnpSummary> derive_per_snp_summary(const DecodedTileSummaryInput& in) {
    const int P = in.P;
    const long M = in.M;

    std::vector<PerSnpSummary> out(static_cast<std::size_t>(M < 0 ? 0 : M));
    if (P <= 0 || M <= 0) return out;

    // Total kept individuals across the kept pops (the missing-fraction
    // denominator). SNP-independent: the sample axis is fixed for the tile.
    std::size_t total_indiv = 0;
    for (int p = 0; p < P && p < static_cast<int>(in.pop_individuals.size()); ++p) {
        total_indiv += in.pop_individuals[static_cast<std::size_t>(p)];
    }
    const double total_indiv_d = static_cast<double>(total_indiv);
    const double ploidy_d = static_cast<double>(in.ploidy > 0 ? in.ploidy : 1);

    for (long s = 0; s < M; ++s) {
        // Reduce ACROSS the kept populations into SNP-global scalars (the §1, §5
        // S2 invariant): one pooled ref count, one pooled allele count, one
        // non-missing-individual sum per SNP — never a per-(pop, SNP) value.
        double pooled_ref_count = 0.0;     // Σ_pop Q·N  (== Σ ssum_pop)
        double pooled_allele_count = 0.0;  // Σ_pop N    (== Σ 2·scnt_pop)
        double nonmissing_indiv = 0.0;     // Σ_pop N/ploidy
        for (int p = 0; p < P; ++p) {
            const std::size_t off =
                static_cast<std::size_t>(p) + static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
            const double npn = in.n[off];  // non-missing haploid count for (pop, snp)
            // Q is zero-filled where invalid, and N is 0 there, so Q·N and N both
            // contribute 0 from a missing (pop, snp) cell — no branch needed.
            pooled_ref_count += in.q[off] * npn;
            pooled_allele_count += npn;
            nonmissing_indiv += npn / ploidy_d;
        }

        PerSnpSummary& sm = out[static_cast<std::size_t>(s)];
        sm.pooled_allele_count = pooled_allele_count;
        sm.pooled_ref_af =
            (pooled_allele_count > 0.0) ? (pooled_ref_count / pooled_allele_count) : 0.0;
        sm.pooled_minor_af = folded_maf(sm.pooled_ref_af);
        sm.missing_frac =
            (total_indiv_d > 0.0) ? (1.0 - nonmissing_indiv / total_indiv_d) : 1.0;
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

    const std::vector<PerSnpSummary> summary = derive_per_snp_summary(in);
    const bool mem_noop = mem.is_noop();

    for (long s = 0; s < M; ++s) {
        const std::size_t si = static_cast<std::size_t>(s);
        const PerSnpSummary& sm = summary[si];

        const char ref = (si < snps.ref.size()) ? snps.ref[si] : 'N';
        const char alt = (si < snps.alt.size()) ? snps.alt[si] : 'N';

        // ---- Unconditional DROP-NOT-FLIP class drops (architecture.md §1). A
        // strand-ambiguous (A/T, C/G palindrome) or multiallelic / non-clean-ACGT
        // SNP is dropped regardless of any flag and is NEVER strand-flipped. -----
        if (is_multiallelic(ref, alt) || is_strand_ambiguous(ref, alt)) {
            continue;  // keep[s] stays false
        }

        // ---- Threshold filters (thresholds FROM cfg, predicates SHARED). -------
        if (!snp_passes_maf(sm.pooled_minor_af, cfg.maf_min)) continue;
        if (!snp_passes_geno(sm.missing_frac, cfg.geno_max_missing)) continue;

        // ---- Flag-gated additions (each off by default ⇒ no-op). ---------------
        if (cfg.drop_monomorphic && is_monomorphic(sm.pooled_minor_af)) continue;
        if (cfg.transversions_only && !is_transversion(ref, alt)) continue;
        if (cfg.autosomes_only) {
            const int chrom = (si < snps.chrom.size()) ? snps.chrom[si] : 0;
            if (!is_autosome(chrom)) continue;
        }

        // ---- Include/exclude + prune.in membership. ----------------------------
        if (!mem_noop) {
            static const std::string kEmptyId;  // id for an alleles-only/short record
            const std::string& id = (si < snps.id.size()) ? snps.id[si] : kEmptyId;
            if (!mem.passes(id)) continue;
        }

        keep[si] = true;  // survived every active filter
    }
    return keep;
}

}  // namespace steppe::io::filter
