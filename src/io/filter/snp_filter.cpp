// src/io/filter/snp_filter.cpp
//
// SNP-level keep-mask producer: derives the per-SNP pooled folded MAF and
// missing fraction from the decoded per-pop Q/N, then applies the shared
// filter_decision predicates into one SNP-global keep mask. Host-pure io-leaf.
#include "io/filter/snp_filter.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "io/filter/filter_decision.hpp"
#include "io/filter/snp_summary_reduce.hpp"

namespace steppe::io::filter {

std::vector<PerSnpSummary> derive_per_snp_summary(const DecodedTileSummaryInput& in) {
    const int P = in.P;
    const long M = in.M;

    std::vector<PerSnpSummary> out(static_cast<std::size_t>(M < 0 ? 0 : M));
    if (P <= 0 || M <= 0) return out;

    if (in.ploidy != 1 && in.ploidy != 2) {
        throw std::invalid_argument(
            "snp_filter: ploidy must be 1 (pseudo-haploid) or 2 (diploid); got " +
            std::to_string(in.ploidy));
    }

    if (in.pop_individuals.size() != static_cast<std::size_t>(P)) {
        throw std::invalid_argument(
            "snp_filter: pop_individuals.size() (" +
            std::to_string(in.pop_individuals.size()) +
            ") must equal P (" + std::to_string(P) + ")");
    }

    if (in.q == nullptr || in.n == nullptr) {
        throw std::invalid_argument(
            "snp_filter: q and n must be non-null when P>0 && M>0");
    }

    std::size_t total_indiv = 0;
    for (int p = 0; p < P; ++p) {
        total_indiv += in.pop_individuals[static_cast<std::size_t>(p)];
    }
    const double total_indiv_d = static_cast<double>(total_indiv);
    const double ploidy_d = static_cast<double>(in.ploidy);

    for (long s = 0; s < M; ++s) {
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

namespace {

// SnpTable size guard shared by the two keep-mask builders (bit-identical messages/order to
// the pre-refactor build_snp_keep_mask): ref/alt must cover M, chrom must cover M when
// autosomes-only is active, id must cover M when a membership predicate is active.
void validate_snp_table(const SnpTable& snps, long M, const FilterConfig& cfg,
                        const SnpMembership& mem) {
    const std::size_t Mu = static_cast<std::size_t>(M);
    const bool mem_noop = mem.is_noop();
    const auto require_at_least = [&](const char* what, std::size_t have, bool active) {
        if (active && have < Mu) {
            throw std::invalid_argument(
                std::string("snp_filter: ") + what + " >= M (" + std::to_string(M) +
                "); got " + std::to_string(have));
        }
    };
    if (snps.ref.size() < Mu || snps.alt.size() < Mu) {
        throw std::invalid_argument(
            "snp_filter: snps.ref/alt must have >= M (" + std::to_string(M) +
            ") entries; got ref=" + std::to_string(snps.ref.size()) +
            " alt=" + std::to_string(snps.alt.size()));
    }
    require_at_least("autosomes_only requires snps.chrom", snps.chrom.size(),
                     cfg.autosomes_only);
    require_at_least("active membership requires snps.id", snps.id.size(), !mem_noop);
}

// The per-SNP keep decision over an already-derived summary (validation done by the caller).
// The single decision loop both public builders share, so they cannot drift.
std::vector<bool> decide_keep(const std::vector<PerSnpSummary>& summary, const SnpTable& snps,
                              const FilterConfig& cfg, const SnpMembership& mem) {
    const long M = static_cast<long>(summary.size());
    std::vector<bool> keep(static_cast<std::size_t>(M < 0 ? 0 : M), false);
    const bool mem_noop = mem.is_noop();
    for (long s = 0; s < M; ++s) {
        const std::size_t si = static_cast<std::size_t>(s);
        const int chrom = cfg.autosomes_only ? snps.chrom[si] : 0;
        const bool membership_ok = mem_noop ? true : mem.passes(snps.id[si]);
        keep[si] = snp_keep_decision(summary[si], snps.ref[si], snps.alt[si], chrom, cfg,
                                     membership_ok);
    }
    return keep;
}

}  // namespace

std::vector<bool> build_snp_keep_mask(const DecodedTileSummaryInput& in,
                                      const SnpTable& snps,
                                      const FilterConfig& cfg,
                                      const SnpMembership& mem) {
    const long M = in.M;
    if (M <= 0) return std::vector<bool>(static_cast<std::size_t>(M < 0 ? 0 : M), false);

    validate_snp_table(snps, M, cfg, mem);
    const std::vector<PerSnpSummary> summary = derive_per_snp_summary(in);
    return decide_keep(summary, snps, cfg, mem);
}

std::vector<bool> build_snp_keep_mask_from_summary(const std::vector<PerSnpSummary>& summary,
                                                   const SnpTable& snps,
                                                   const FilterConfig& cfg,
                                                   const SnpMembership& mem) {
    const long M = static_cast<long>(summary.size());
    if (M <= 0) return std::vector<bool>(0, false);

    validate_snp_table(snps, M, cfg, mem);
    return decide_keep(summary, snps, cfg, mem);
}

bool filter_is_active(const FilterConfig& cfg) noexcept {
    return cfg.maf_min > 0.0 || cfg.geno_max_missing < 1.0 || cfg.autosomes_only ||
           cfg.drop_monomorphic || cfg.transversions_only || !cfg.include_snp_ids.empty() ||
           !cfg.exclude_snp_ids.empty() || !cfg.prune_in_path.empty() || cfg.ld_prune_active();
}

}  // namespace steppe::io::filter
