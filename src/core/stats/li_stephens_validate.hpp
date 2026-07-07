// src/core/stats/li_stephens_validate.hpp
//
// Reference: docs/reference/src_core_stats_li_stephens_validate.hpp.md
//
// The host-pure, CUDA-free up-front validator for a `steppe paint` (Li-Stephens
// haplotype-copying) request. It runs BEFORE any compute is launched — the same
// validate-once, fail-fast posture ConfigBuilder::build() and the f-stat sweep's
// maxcomb cap use — and enforces the five §3 contracts the copying model needs:
//
//   1. Input is phased/haploid — a diploid triple (any heterozygous sample) is
//      rejected with a "phase first" error (the load-bearing dependency: the model
//      copies phased haplotypes, and steppe builds no phaser).
//   2. The .snp genetic map is present and monotonic within each chromosome; a
//      missing map is a hard error unless the bp-fallback is explicitly opted into.
//   3. The self-copy / leave-one-out policy is coherent: when the donor panel is a
//      superset of the recipients (panel-vs-self painting), self-copy MUST be off
//      or the copying diagonal is a degenerate self-match.
//   4. Ne > 0, theta legal (or auto), recip-batch >= 1.
//   5. Cost guard: the O(N·K·M) work and the per-wave forward-table footprint are
//      computed up front and refused past a cap without an explicit override.
//
// The check operates on already-parsed, plain-data inputs (a SnpTable plus scalar
// request facts), so it is trivially unit-testable with no files and no GPU; the
// `paint` command adapts a RunConfig + the read SNP/individual tables onto it.
//
// Reference: docs/planning/li-stephens-engine-scope.md §3
#ifndef STEPPE_CORE_STATS_LI_STEPHENS_VALIDATE_HPP
#define STEPPE_CORE_STATS_LI_STEPHENS_VALIDATE_HPP

#include <string>

#include "io/snp_reader.hpp"
#include "steppe/error.hpp"

namespace steppe::core {

// The scalar facts a paint request must satisfy, gathered by the command after it
// has resolved the two triples' individual/SNP tables. Kept plain-data so the
// validator unit-tests with no io and no device.
struct PaintRequest {
    double Ne = 20000.0;
    // NaN => "auto" (Watterson theta over K); otherwise a fixed emission rate in [0,1].
    double theta = 0.0;
    bool theta_auto = true;
    // The user's --self-copy: true = a haplotype may copy itself (NO leave-one-out).
    bool self_copy = false;
    long recip_batch = 256;
    // Whether the recombination map may fall back to a bp window when genpos is absent.
    bool allow_bp_fallback = false;
    // Panel geometry.
    long n_recipients = 0;
    long n_donors = 0;
    // #samples across BOTH triples that decode as diploid (a heterozygous code seen);
    // must be zero — the input must be pre-phased into haploid columns.
    long n_diploid_samples = 0;
    // True when the donor set is a superset of the recipient set (all-vs-self painting):
    // the case decision (4) governs.
    bool donors_superset_recipients = false;
    // Lifts the work cost cap (the --sure posture).
    bool sure = false;
};

// validate_paint_request — returns Status::Ok on a well-formed request, else
// Status::InvalidConfig with a clear one-line reason in `err`. Never throws.
[[nodiscard]] Status validate_paint_request(const PaintRequest& req,
                                            const io::SnpTable& snp,
                                            std::string& err);

}  // namespace steppe::core

#endif  // STEPPE_CORE_STATS_LI_STEPHENS_VALIDATE_HPP
