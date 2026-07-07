// src/core/stats/li_stephens_validate.cpp — the host-pure `steppe paint` validator.
//
// Fail-fast, CUDA-free: enforce the five §3 contracts before any compute launches.
// Reference: docs/reference/src_core_stats_li_stephens_validate.cpp.md
// Contract source: docs/planning/li-stephens-engine-scope.md §3

#include "core/stats/li_stephens_validate.hpp"

#include <cmath>
#include <cstddef>
#include <string>

#include "steppe/config.hpp"

namespace steppe::core {

namespace {

// Is the .snp genetic map effectively absent? A file with no genpos column parses
// to an all-zero genpos_morgans (the io leaf derives nothing). A real map is a
// strictly increasing cM run per chromosome, so all-zero == "no map".
[[nodiscard]] bool map_absent(const io::SnpTable& snp) {
    for (double g : snp.genpos_morgans) {
        if (g != 0.0) return false;
    }
    return true;
}

}  // namespace

Status validate_paint_request(const PaintRequest& req, const io::SnpTable& snp,
                              std::string& err) {
    err.clear();
    const auto fail = [&err](std::string msg) -> Status {
        err = std::move(msg);
        return Status::InvalidConfig;
    };

    // --- 1. Phased / haploid input -------------------------------------------
    // The copying model consumes phased haplotypes; steppe builds no phaser. A
    // diploid triple (any heterozygous call) must be phased into haploid columns
    // upstream (SHAPEIT/Beagle -> two haploid columns per diploid).
    if (req.n_diploid_samples > 0) {
        return fail("paint requires PRE-PHASED haploid input, but " +
                    std::to_string(req.n_diploid_samples) +
                    " sample(s) decode as diploid (a heterozygous call was seen). "
                    "Phase first (SHAPEIT/Beagle) so each haplotype is a haploid column "
                    "of the genotype triple.");
    }

    // --- 4. Panel geometry + numeric knobs (checked early so later math is safe) --
    if (req.n_recipients <= 0) {
        return fail("paint needs at least one recipient haplotype (--prefix resolved 0).");
    }
    if (req.n_donors <= 0) {
        return fail("paint needs at least one donor haplotype (--donors resolved 0).");
    }
    if (!(req.Ne > 0.0) || !std::isfinite(req.Ne)) {
        return fail("--Ne must be a finite value > 0.");
    }
    if (!req.theta_auto) {
        if (!std::isfinite(req.theta) || req.theta < 0.0 || req.theta > 1.0) {
            return fail("--theta must be 'auto' or a finite value in [0, 1].");
        }
    }
    if (req.recip_batch < 1) {
        return fail("--recip-batch must be >= 1.");
    }

    // --- 2. Genetic map present + monotonic within each chromosome -----------
    if (snp.count == 0 || snp.genpos_morgans.empty()) {
        return fail("the .snp table carries no SNPs (empty genetic map).");
    }
    if (map_absent(snp)) {
        if (!req.allow_bp_fallback) {
            return fail("the .snp genetic map (genpos, column 3, Morgans) is absent. "
                        "paint needs a real cM map by default; pass --bp-fallback to "
                        "approximate recombination from a fixed base-pair window instead.");
        }
        // bp-fallback opted in: physpos must then be usable as the map surrogate.
        if (snp.physpos.empty()) {
            return fail("--bp-fallback requested but the .snp has no physical positions "
                        "either — no recombination scale can be derived.");
        }
    } else {
        // A present cM map must be monotonic non-decreasing WITHIN each chromosome
        // (the classic map-merge bug produces a decreasing run at a join). The map
        // resets across a chromosome boundary, so only compare consecutive same-chrom
        // SNPs.
        const std::size_t n = snp.count;
        const bool have_chrom = snp.chrom.size() == n;
        for (std::size_t s = 1; s < n; ++s) {
            const bool same_chrom = !have_chrom || (snp.chrom[s] == snp.chrom[s - 1]);
            if (!same_chrom) continue;
            if (snp.genpos_morgans[s] < snp.genpos_morgans[s - 1]) {
                std::string where = have_chrom ? (" on chromosome " + std::to_string(snp.chrom[s]))
                                               : std::string();
                return fail("the .snp genetic map is non-monotonic" + where +
                            " at SNP index " + std::to_string(s) + " (" +
                            std::to_string(snp.genpos_morgans[s - 1]) + " -> " +
                            std::to_string(snp.genpos_morgans[s]) +
                            " Morgans). A cM run must be non-decreasing within a chromosome "
                            "(likely a map-merge bug).");
            }
        }
    }

    // --- 3. Self-copy / leave-one-out coherence ------------------------------
    // When the donor panel is a superset of the recipients (panel-vs-self painting,
    // the ChromoPainter all-vs-all shape), a haplotype must NOT copy itself or the
    // copying diagonal is a degenerate self-match. Locked decision: leave-one-out ON
    // in that case, i.e. --self-copy must be off.
    if (req.donors_superset_recipients && req.self_copy) {
        return fail("--self-copy is on but the donor panel is a superset of the recipients "
                    "(panel-vs-self painting). A haplotype copying itself is a degenerate "
                    "self-match; leave-one-out is required — turn --self-copy off.");
    }

    // --- 5. Cost guard: O(N·K·M) work + per-wave forward-table footprint ------
    const double work = static_cast<double>(req.n_recipients) *
                        static_cast<double>(req.n_donors) *
                        static_cast<double>(snp.count);
    if (work > kLsMaxWorkStates && !req.sure) {
        return fail("paint would do ~" + std::to_string(work) +
                    " forward-backward state-updates (N=" + std::to_string(req.n_recipients) +
                    " x K=" + std::to_string(req.n_donors) + " x M=" +
                    std::to_string(snp.count) + "), past the safety cap of " +
                    std::to_string(kLsMaxWorkStates) +
                    ". Re-run with --sure to proceed with a long job.");
    }
    // The per-wave forward table is recip_batch x K doubles resident at once.
    const long batch = req.recip_batch < req.n_recipients ? req.recip_batch : req.n_recipients;
    const double alpha_bytes = static_cast<double>(batch) * static_cast<double>(req.n_donors) *
                              static_cast<double>(sizeof(double));
    if (alpha_bytes > static_cast<double>(kLsMaxAlphaFootprintBytes)) {
        return fail("the per-wave forward table (recip-batch=" + std::to_string(batch) +
                    " x K=" + std::to_string(req.n_donors) + " doubles = ~" +
                    std::to_string(alpha_bytes) + " bytes) exceeds the resident cap of " +
                    std::to_string(kLsMaxAlphaFootprintBytes) +
                    " bytes. Lower --recip-batch.");
    }

    return Status::Ok;
}

}  // namespace steppe::core
