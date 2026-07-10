// src/core/stats/apply_snp_filter.cpp — the shared per-SNP QC-filter seam. See the header for
// the design (bit-exact subset-in-place, same-ascertainment guard, host repack rationale).
#include "core/stats/apply_snp_filter.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/internal/decode_af.hpp"          // genotype_code, kCodesPerByte, kPloidyDiploid
#include "core/stats/decode_keep_autosomes.hpp"  // make_decode_tile_view
#include "device/backend.hpp"                    // ComputeBackend, DecodeResult, DecodeTileView
#include "io/filter/ascertainment.hpp"
#include "io/filter/include_exclude.hpp"
#include "io/filter/snp_filter.hpp"

namespace steppe {
namespace core {

namespace {

namespace flt = steppe::io::filter;

// Gather the external "keep panel" ids (include list ∪ prune.in file), used only by the
// same-ascertainment guard. The exclude list is a drop set, not a panel, so it is left out.
std::vector<std::string> gather_external_keep_ids(const FilterConfig& cfg) {
    std::vector<std::string> ids = cfg.include_snp_ids;
    if (!cfg.prune_in_path.empty()) {
        flt::read_snp_id_list(cfg.prune_in_path, ids);  // throws on a missing/unreadable file
    }
    return ids;
}

// Build the SNP-global keep mask over the tile's population partition. Decodes per-pop allele
// frequencies on the backend only when a threshold that needs them is active (MAF / per-SNP
// missing / drop-monomorphic); otherwise the class/membership predicates are evaluated
// directly from the SnpTable with a summary that trivially clears the numeric thresholds.
std::vector<bool> build_keep_mask(const io::GenotypeTile& tile, const io::SnpTable& snptab,
                                  const FilterConfig& cfg, ComputeBackend& backend, int P, long M) {
    const bool needs_decode =
        cfg.maf_min > 0.0 || cfg.geno_max_missing < 1.0 || cfg.drop_monomorphic;

    if (needs_decode) {
        std::vector<std::size_t> pop_individuals(static_cast<std::size_t>(P));
        for (int p = 0; p < P; ++p) {
            pop_individuals[static_cast<std::size_t>(p)] =
                tile.pop_offsets[static_cast<std::size_t>(p) + 1] -
                tile.pop_offsets[static_cast<std::size_t>(p)];
        }
        // Force diploid: the four tools all read each 2-bit code as a diploid dosage (the
        // ploidy vector only sizes the view). The pooled MAF/missing definition is per-SNP
        // per-individual, pooled across the tile's populations (documented in the header).
        std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);
        const DecodeTileView view = make_decode_tile_view(tile, sample_ploidy, P);
        const DecodeResult dec = backend.decode_af(view);

        flt::DecodedTileSummaryInput fin;
        fin.q = dec.q.data();
        fin.v = dec.v.data();
        fin.n = dec.n.data();
        fin.P = P;
        fin.M = M;
        fin.pop_individuals = std::move(pop_individuals);
        fin.ploidy = kPloidyDiploid;

        const flt::SnpMembership mem(cfg);
        return flt::build_snp_keep_mask(fin, snptab, cfg, mem);
    }

    // No numeric threshold active: only membership + class predicates (autosome, strand,
    // multiallelic, transversion) apply. A summary that clears MAF/missing/monomorphic lets
    // us reuse the exact keep_decision so the two paths cannot drift.
    const flt::SnpMembership mem(cfg);
    const bool mem_noop = mem.is_noop();
    flt::PerSnpSummary pass;
    pass.pooled_ref_af = 0.5;
    pass.pooled_minor_af = 0.5;
    pass.missing_frac = 0.0;
    pass.pooled_allele_count = 1.0;

    std::vector<bool> keep(static_cast<std::size_t>(M), false);
    for (long s = 0; s < M; ++s) {
        const std::size_t si = static_cast<std::size_t>(s);
        const int chrom = cfg.autosomes_only && si < snptab.chrom.size() ? snptab.chrom[si] : 0;
        const bool membership_ok = mem_noop ? true : mem.passes(snptab.id[si]);
        const char ref = si < snptab.ref.size() ? snptab.ref[si] : '\0';
        const char alt = si < snptab.alt.size() ? snptab.alt[si] : '\0';
        keep[si] = flt::snp_keep_decision(pass, ref, alt, chrom, cfg, membership_ok);
    }
    return keep;
}

// Repack the individual-major 2-bit tile down to the kept SNP columns (kept order preserved).
// Pure gather: each surviving code is copied bit-for-bit into the smaller record.
void repack_tile_columns(io::GenotypeTile& tile, const std::vector<long>& kept_cols) {
    const std::size_t N = tile.n_individuals;
    const std::size_t oldbpr = tile.bytes_per_record;
    const std::size_t n_kept = kept_cols.size();
    const std::size_t newbpr =
        (n_kept + static_cast<std::size_t>(kCodesPerByte) - 1) / static_cast<std::size_t>(kCodesPerByte);

    std::vector<std::uint8_t> out(N * newbpr, 0);
    for (std::size_t r = 0; r < N; ++r) {
        const std::uint8_t* rec = tile.packed.data() + r * oldbpr;
        std::uint8_t* orec = out.data() + r * newbpr;
        for (std::size_t j = 0; j < n_kept; ++j) {
            const long s = kept_cols[j];
            const std::uint8_t code =
                genotype_code(rec[static_cast<std::size_t>(s) / kCodesPerByte], static_cast<int>(s));
            const int shift =
                (kCodesPerByte - 1 - static_cast<int>(j % static_cast<std::size_t>(kCodesPerByte))) *
                kBitsPerCode;
            orec[j / static_cast<std::size_t>(kCodesPerByte)] |=
                static_cast<std::uint8_t>((code & kCodeMask) << shift);
        }
    }
    tile.packed = std::move(out);
    tile.bytes_per_record = newbpr;
    tile.n_snp = n_kept;
}

// Subset a SnpTable to the kept columns, in lockstep with the tile repack.
void subset_snptab(io::SnpTable& snptab, const std::vector<long>& kept_cols) {
    io::SnpTable out;
    const std::size_t n_kept = kept_cols.size();
    out.id.reserve(n_kept);
    out.chrom.reserve(n_kept);
    out.genpos_morgans.reserve(n_kept);
    out.physpos.reserve(n_kept);
    out.ref.reserve(n_kept);
    out.alt.reserve(n_kept);
    for (const long s : kept_cols) {
        const std::size_t si = static_cast<std::size_t>(s);
        out.id.push_back(si < snptab.id.size() ? snptab.id[si] : std::string());
        out.chrom.push_back(si < snptab.chrom.size() ? snptab.chrom[si] : 0);
        out.genpos_morgans.push_back(si < snptab.genpos_morgans.size() ? snptab.genpos_morgans[si] : 0.0);
        out.physpos.push_back(si < snptab.physpos.size() ? snptab.physpos[si] : 0.0);
        out.ref.push_back(si < snptab.ref.size() ? snptab.ref[si] : '\0');
        out.alt.push_back(si < snptab.alt.size() ? snptab.alt[si] : '\0');
    }
    out.count = n_kept;
    snptab = std::move(out);
}

}  // namespace

SnpFilterOutcome apply_snp_filter(io::GenotypeTile& tile, io::SnpTable& snptab,
                                  const FilterConfig& cfg, ComputeBackend& backend) {
    SnpFilterOutcome out;
    out.n_in = static_cast<long>(tile.n_snp);
    out.n_kept = out.n_in;

    if (!flt::filter_is_active(cfg)) return out;  // byte-identical: nothing requested

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) return out;

    // Same-ascertainment guard: a supplied keep/prune list drawn from a different panel than
    // the target is a silent, biasing intersection. Fire BEFORE any decode (fail fast).
    const std::vector<std::string> external_ids = gather_external_keep_ids(cfg);
    if (!external_ids.empty() && !cfg.allow_mixed_ascertainment) {
        const flt::AscertainmentVerdict v =
            flt::check_same_ascertainment(snptab.id, external_ids);
        if (v.mixed) throw std::invalid_argument("same-ascertainment guard: " + v.reason);
    }

    const std::vector<bool> keep = build_keep_mask(tile, snptab, cfg, backend, P, M);

    std::vector<long> kept_cols;
    kept_cols.reserve(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        if (keep[static_cast<std::size_t>(s)]) kept_cols.push_back(s);
    }
    const long n_kept = static_cast<long>(kept_cols.size());
    out.n_kept = n_kept;

    if (n_kept <= 0) {
        throw std::invalid_argument(
            "per-SNP QC filter kept 0 of " + std::to_string(M) +
            " SNPs — relax --maf / --geno-max-miss / the keep list");
    }

    // Always report the retained ids (for --emit-kept-snps), whether or not a repack was needed.
    out.kept_ids.reserve(static_cast<std::size_t>(n_kept));
    for (const long s : kept_cols) {
        const std::size_t si = static_cast<std::size_t>(s);
        out.kept_ids.push_back(si < snptab.id.size() ? snptab.id[si] : std::string());
    }

    if (n_kept == M) return out;  // mask kept everything: leave the tile/SnpTable untouched

    repack_tile_columns(tile, kept_cols);
    subset_snptab(snptab, kept_cols);
    out.applied = true;
    return out;
}

}  // namespace core
}  // namespace steppe
