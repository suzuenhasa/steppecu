// src/core/stats/apply_snp_filter.cpp — the shared per-SNP QC-filter seam. See the header for
// the design (bit-exact subset-in-place, same-ascertainment guard, host repack rationale).
#include "core/stats/apply_snp_filter.hpp"

#include <algorithm>
#include <cstddef>
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

// Choose a SNP-tile width (a multiple of kCodesPerByte, so each tile start stays byte-aligned
// in the packed 2-bit tile) that bounds the streamed per-pop decode's transient — 3 arrays
// (Q/V/N) of P*tm doubles, on the device and copied back to host — to ~a quarter of free VRAM.
// This is what lets PCA (and the other genotype tools that share this seam) run the QC filter
// at biobank cohort sizes: the whole-tile O(P*M) decode (3 * P * M doubles, ~35 GB on the HO
// top-2500-pop cohort) is never materialized; it is walked SNP-tile by SNP-tile instead. When
// free VRAM is unknown (the CPU oracle / no-GPU path reports 0) the whole SNP axis is used, so
// that path is byte-identical to the pre-streaming behavior.
long choose_decode_snp_tile(int P, long M, std::size_t free_vram_bytes) {
    if (P <= 0 || M <= 0) return M > 0 ? M : 1;
    if (free_vram_bytes == 0) return M;  // unknown free VRAM (CPU oracle) -> whole axis
    const std::size_t budget = free_vram_bytes / 4u;
    const std::size_t per_snp = static_cast<std::size_t>(P) * 3u * sizeof(double);
    long tm = static_cast<long>(budget / (per_snp == 0 ? 1u : per_snp));
    // Round DOWN to a multiple of kCodesPerByte: only the tile START must be byte-aligned (the
    // packed sub-tile begins at s_lo/kCodesPerByte); the final short tile is fine.
    tm -= tm % static_cast<long>(kCodesPerByte);
    if (tm < static_cast<long>(kCodesPerByte)) tm = static_cast<long>(kCodesPerByte);
    if (tm > M) tm = M;
    return tm;
}

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

        // The pooled reduction's two tile-wide constants, computed EXACTLY as
        // derive_per_snp_summary does (integer sum of pop sizes cast once; diploid ploidy).
        std::size_t total_indiv = 0;
        for (int p = 0; p < P; ++p) total_indiv += pop_individuals[static_cast<std::size_t>(p)];
        const double total_indiv_d = static_cast<double>(total_indiv);
        const double ploidy_d = static_cast<double>(kPloidyDiploid);

        // Stream the per-pop decode over SNP tiles, pooling each tile to the O(M) per-SNP
        // summary on the host as it goes. This is BIT-IDENTICAL to decoding the whole tile
        // then pooling: the decode is per-(pop, SNP) independent, so a byte-aligned SNP
        // sub-view (packed offset s_lo/kCodesPerByte, n_snp = tm) yields the same Q/N bits as
        // the matching columns of the whole decode, and derive_pooled_summary_one is the same
        // reduction in the same p = 0..P-1 order. Only the O(P*M) transient is removed — the
        // whole-tile decode that cudaMalloc-fails past ~32 GB on a large-P cohort.
        const long tileM = choose_decode_snp_tile(P, M, backend.capabilities().free_vram_bytes);
        std::vector<flt::PerSnpSummary> summary(static_cast<std::size_t>(M));
        for (long s_lo = 0; s_lo < M; s_lo += tileM) {
            const long tm = std::min<long>(tileM, M - s_lo);
            DecodeTileView sub = view;
            sub.packed = view.packed +
                         static_cast<std::size_t>(s_lo) / static_cast<std::size_t>(kCodesPerByte);
            sub.n_snp = static_cast<std::size_t>(tm);
            const DecodeResult dec = backend.decode_af(sub);
            for (long j = 0; j < tm; ++j) {
                const flt::PooledSnpSummary ps = flt::derive_pooled_summary_one(
                    dec.q.data(), dec.n.data(), P, j, ploidy_d, total_indiv_d);
                flt::PerSnpSummary& sm = summary[static_cast<std::size_t>(s_lo + j)];
                sm.pooled_ref_af = ps.pooled_ref_af;
                sm.pooled_minor_af = ps.pooled_minor_af;
                sm.missing_frac = ps.missing_frac;
                sm.pooled_allele_count = ps.pooled_allele_count;
            }
        }

        const flt::SnpMembership mem(cfg);
        return flt::build_snp_keep_mask_from_summary(summary, snptab, cfg, mem);
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

// Subset `tile` + `snptab` in lockstep to the kept SNP columns. A no-op (returns false) when the
// mask keeps every column; otherwise repacks the 2-bit tile + subsets the table and returns true.
bool subset_to_mask(io::GenotypeTile& tile, io::SnpTable& snptab,
                    const std::vector<std::uint8_t>& keep, const char* what) {
    const long M = static_cast<long>(tile.n_snp);
    std::vector<long> kept_cols;
    kept_cols.reserve(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        if (keep[static_cast<std::size_t>(s)]) kept_cols.push_back(s);
    }
    if (kept_cols.empty()) {
        throw std::invalid_argument(std::string(what) + " kept 0 of " + std::to_string(M) +
                                    " SNPs — relax --maf / --geno-max-miss / --ld-prune / the keep list");
    }
    if (static_cast<long>(kept_cols.size()) == M) return false;  // kept everything: untouched
    repack_tile_columns(tile, kept_cols);
    subset_snptab(snptab, kept_cols);
    return true;
}

// Windowed-r2 LD prune (--ld-prune) over the CURRENT (post-QC) tile: decode + pairwise r^2 on the
// backend, greedy plink2 selection, returned as a per-SNP keep mask. See ld_prune_windowed.
std::vector<std::uint8_t> ld_prune_mask(io::GenotypeTile& tile, io::SnpTable& snptab,
                                        const FilterConfig& cfg, ComputeBackend& backend) {
    const long M = static_cast<long>(tile.n_snp);
    std::vector<int> chrom(static_cast<std::size_t>(M), 0);
    for (long s = 0; s < M; ++s) {
        const std::size_t si = static_cast<std::size_t>(s);
        chrom[si] = si < snptab.chrom.size() ? snptab.chrom[si] : 0;
    }
    std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);
    const DecodeTileView view = make_decode_tile_view(tile, sample_ploidy, static_cast<int>(tile.n_pop()));
    return backend.ld_prune_windowed(view, chrom, cfg.ld_prune_window, cfg.ld_prune_step,
                                     cfg.ld_prune_r2);
}

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

    // Phase 1 — per-SNP QC keep mask (MAF / missing / monomorphic / class / membership). Applied
    // FIRST so the LD pruner (Phase 2) runs on the QC survivors, matching the usual QC-then-prune
    // order and keeping each stage's math a pure SNP-subset of the tool's kernels.
    const std::vector<bool> keep_bool = build_keep_mask(tile, snptab, cfg, backend, P, M);
    std::vector<std::uint8_t> keep(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        keep[static_cast<std::size_t>(s)] = keep_bool[static_cast<std::size_t>(s)] ? 1u : 0u;
    }
    if (subset_to_mask(tile, snptab, keep, "per-SNP QC filter")) out.applied = true;

    // Phase 2 — windowed-r2 LD prune over the survivors (the one genuinely-new GPU kernel).
    if (cfg.ld_prune_active()) {
        const std::vector<std::uint8_t> ld_keep = ld_prune_mask(tile, snptab, cfg, backend);
        if (subset_to_mask(tile, snptab, ld_keep, "--ld-prune")) out.applied = true;
    }

    // The surviving SnpTable ids ARE the final kept set (used for --emit-kept-snps).
    out.n_kept = static_cast<long>(tile.n_snp);
    out.kept_ids.assign(snptab.id.begin(), snptab.id.end());
    return out;
}

}  // namespace core
}  // namespace steppe
