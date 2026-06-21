// src/device/cpu/cpu_backend.cpp
//
// CpuBackend — the CPU REFERENCE backend (the correctness ORACLE).
//
// Implements `ComputeBackend::compute_f2` with the CANCELLATION-FREE long-double
// per-SNP reference lifted (logic, not structure) from the validated spike
// (experiments/f2_emu_spike/f2_prec_acc.cu, the long-double block at lines
// 105-112; experiments/f2_emu_spike/f2_emu_spike.cu reference_f2). It is the
// obviously-correct scalar oracle the GPU's 3-GEMM reformulation is continuously
// diffed against (architecture.md §2 "Testability", §13; ROADMAP §5).
//
// WHY THIS IS THE ORACLE, NOT A TEMPLATE THE GPU MIMICS (architecture.md §2):
// the GPU production hot path is three batched GEMMs + two fused elementwise
// kernels; this reference walks the exact ADMIXTOOLS-2 pairwise-complete scalar
// path. The two have entirely different control structure but MUST agree at the
// f2 / Vpair seam within the tight tolerance tier — which holds because the
// per-element FORMULA is shared, not copied: both consume the SAME
// `core/internal/f2_estimator.hpp` primitives (`het_correction`, `f2_term`,
// `finalize_f2`), so the het-correction denominator, the unbiased summand, and
// the Vpair==0 guard cannot drift between the oracle and the GPU feeder
// (architecture.md §5 S2 caveat (b), §13).
//
// CANCELLATION-FREE PROPERTY (the load-bearing oracle property): for each pop
// pair (i, j) we form the per-SNP difference (p_i - p_j) and square it DIRECTLY,
// never the expanded p_i² - 2·p_i·p_j + p_j² the GEMM path builds. Each per-SNP
// summand (p_i - p_j)² - hc_i - hc_j is therefore already small; we then sum
// those summands across SNPs in LONG DOUBLE using pairwise (cascade) summation,
// so the cross-SNP accumulator error is ~log2(M)·u_ld — far below the f2
// magnitude. This sidesteps the catastrophic cancellation that lives in the
// GEMM path's Σp_i² + Σp_j² - 2·Σp_i·p_j (which the GPU keeps in native FP64 for
// exactly this reason; architecture.md §12). The reference owes its accuracy to
// the per-SNP difference + long-double accumulation, NOT to a different formula.
//
// PRECISION POLICY: the `Precision` argument governs only the matmul-heavy GPU
// GEMMs (architecture.md §12). This CPU oracle computes the native (better than
// native: long-double) reference unconditionally and IGNORES the matmul precision
// mode — it is the gold reference every other mode is validated against
// (config.hpp Precision::Kind::Fp64 doc; ROADMAP §0).
//
// LAYERING: pure host C++20. NO CUDA header is included here (architecture.md
// §2, §4: this is the `src/device/cpu/cpu_backend.cpp` reference backend, one of
// the two implementations of the CUDA-free `ComputeBackend` interface in
// device/backend.hpp). It compiles into steppe_device alongside the CUDA backend
// but pulls in no CUDA toolkit headers — it is the GPU-free reference that makes
// `import steppe` work without a GPU and is the correctness anchor the GPU is
// diffed against (architecture.md §8, §13).

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "core/domain/block_partition_rule.hpp" // core::block_ranges, core::BlockRange (the X-3/B3 single-source inverse)
#include "core/internal/decode_af.hpp"     // genotype_code, accumulate_genotype, finalize_af (shared)
#include "core/internal/f2_estimator.hpp"  // het_correction, f2_term, finalize_f2 (shared primitive)
#include "core/internal/pchisq.hpp"        // core::internal::pchisq_upper (M(fit-2) rank-test p-value)
#include "core/internal/small_linalg.hpp"  // core::solve/inverse/jacobi_svd (the qpAdm reference solvers)
#include "core/internal/views.hpp"         // steppe::core::MatView (Q/V/N contract)
#include "core/qpadm/qpadm_bounds.hpp"     // core::qpadm::qpadm_dof — the single-source (nl-r)*(nr-r) dof
#include "device/backend.hpp"              // steppe::ComputeBackend, steppe::F2Result, DecodeResult, F4Blocks/...
#include "device/backend_factory.hpp"      // steppe::device::make_cpu_backend (the single-source decl, X-9/B8)
#include "device/device_f2_blocks.hpp"     // steppe::device::DeviceF2Blocks (the S3 device-resident input)
#include "steppe/config.hpp"               // steppe::Precision
#include "steppe/error.hpp"                // steppe::Status
#include "steppe/qpadm.hpp"                // steppe::QpAdmOptions

// CpuBackend + make_cpu_backend live in steppe::device alongside the GPU backend:
// both implement the ONE CUDA-free ComputeBackend interface and both compile into
// steppe_device, so they share ONE namespace (cleanup X-9/B8 — the old steppe::core
// placement was a namespace/layer mismatch: this TU never depended on CUDA, it just
// builds into the CUDA-private target). The shared host primitives it consumes still
// live in steppe::core (block_ranges, genotype_code, finalize_af, …) and resolve via
// the explicit `core::` qualifications below.
namespace steppe::device {

namespace {

/// Pairwise (cascade) summation of long-double summands — the cross-SNP
/// accumulator that makes the reference cancellation-free.
///
/// Lifted from the spike's `pairwise_sum` (f2_emu_spike.cu:332-342). Halving the
/// range recursively reduces the summation error factor from O(n) to O(log2 n),
/// so the accumulated error is ~log2(n)·u_ld — negligible against the f2
/// magnitude. Does not mutate the input. The 128-element straight-sum base case
/// matches the spike exactly so the oracle reproduces the validated reference.
[[nodiscard]] long double pairwise_sum(const long double* a, std::size_t n) noexcept {
    constexpr std::size_t kPairwiseBaseCase = 128;  // straight-sum block size (spike base case)
    if (n == 0) return 0.0L;
    if (n <= kPairwiseBaseCase) {
        long double s = 0.0L;
        for (std::size_t k = 0; k < n; ++k) s += a[k];
        return s;
    }
    const std::size_t h = n / 2;
    return pairwise_sum(a, h) + pairwise_sum(a + h, n - h);
}

/// The per-pair f2 ORACLE body — the cancellation-free long-double reference for ONE
/// (i, j) population pair over the SNP range [s0, s1). The SINGLE source shared by
/// compute_f2 (range [0, M)) and compute_f2_blocks (per-block range [begin, end)) so
/// the validity check / het correction / unbiased summand / long-double pairwise
/// accumulation / finalize CANNOT drift between the whole-tensor oracle and the
/// per-block oracle ([7.1] dedup; the header @12-21 promises they can't diverge).
///
/// PARITY (load-bearing, architecture.md §12/§13): the accumulation is byte-identical
/// to the two former inline copies — the SAME jointly-valid filter (V.element != 0 in
/// BOTH i and j), the SAME SHARED het_correction / f2_term primitives, the SAME
/// long-double `terms` fill in SNP order, and the SAME long-double pairwise_sum
/// numerator over `count` summands. `terms` is the caller's scratch (sized to the SNP
/// count / largest block) so no [P×P×M] intermediate is materialized. Returns
/// {f2_ij, vpair}; the caller does the symmetric column-major mirror-write.
struct F2Pair { double f2 = 0.0; double vpair = 0.0; };

[[nodiscard]] F2Pair f2_pair_over_range(const core::MatView& Q, const core::MatView& V,
                                        const core::MatView& N, int i, int j,
                                        long s0, long s1, long double* terms) {
    long count = 0;  // pairwise-valid SNP count == Vpair(i, j)
    for (long s = s0; s < s1; ++s) {
        const bool vi = V.element(i, s) != 0.0;
        const bool vj = V.element(j, s) != 0.0;
        if (!vi || !vj) continue;  // not jointly valid: contributes nothing
        const double p_i = Q.element(i, s);
        const double p_j = Q.element(j, s);
        const double hc_i = core::het_correction(p_i, N.element(i, s), true);
        const double hc_j = core::het_correction(p_j, N.element(j, s), true);
        terms[static_cast<std::size_t>(count)] =
            static_cast<long double>(core::f2_term(p_i, p_j, hc_i, hc_j));
        ++count;
    }
    const long double numerator = pairwise_sum(terms, static_cast<std::size_t>(count));
    const double vpair = static_cast<double>(count);
    return F2Pair{core::finalize_f2(static_cast<double>(numerator), vpair), vpair};
}

/// The CPU reference backend: the scalar / long-double correctness oracle.
///
/// Implements the full S0–S7 oracle surface against which the GPU backend is
/// diffed: compute_f2 / compute_f2_blocks, decode_af, assemble_f4, jackknife_cov,
/// rank_test, rank_sweep, gls_weights, and gls_weights_loo_batched. Move-only
/// ownership via the base class's deleted copy/move (architecture.md §9).
class CpuBackend final : public ComputeBackend {
public:
    [[nodiscard]] F2Result compute_f2(const core::MatView& Q,
                                      const core::MatView& V,
                                      const core::MatView& N,
                                      const Precision& precision) override {
        // The matmul precision mode governs the GPU GEMMs only; the oracle is
        // always the (long-double-accumulated) native reference (architecture.md
        // §12; config.hpp Precision::Kind::Fp64). Acknowledge the parameter so
        // the signature matches the interface without an unused-parameter warning
        // under -Werror.
        (void)precision;

        const int P = Q.P;     // number of populations == leading dimension
        const long M = Q.M;    // number of SNPs in this block

        F2Result out;
        out.P = P;
        // Column-major [P × P] outputs: element (i, j) at i + P·j. Zero-filled so
        // any entry left untouched on an empty block defaults to 0; the loop below
        // writes EVERY (i, j) including the diagonal (the F2Result diagonal
        // convention pinned in backend.hpp — see the loop comment).
        out.f2.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(P), 0.0);
        out.vpair.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(P), 0.0);

        if (P <= 0 || M <= 0) return out;  // empty block: nothing to accumulate

        // Per-SNP summand scratch, reused across pairs. Sized to the SNP count so
        // we never materialize the [P × P × M] intermediate (architecture.md
        // §11.1): the reference is O(P²·M) time, O(M) extra space.
        std::vector<long double> terms(static_cast<std::size_t>(M));

        // Walk the upper triangle INCLUDING the diagonal (j = i); f2 is symmetric,
        // so mirror each off-diagonal result to (j, i). The diagonal carries the
        // full (i, i) computation: f2_term(p_i, p_i, hc_i, hc_i) = (p_i − p_i)²
        // − hc_i − hc_i = −2·hc_i, so f2(i,i) = −2·mean within-pop het correction
        // and vpair(i,i) = i's valid-SNP count. This MATCHES the GPU M0 path
        // (assemble_f2_kernel writes every (i,j) with no i==j guard) and the
        // per-block oracle (compute_f2_blocks, j = i), so the F2Result diagonal is
        // consistent across backends — the convention pinned in backend.hpp
        // (cleanup X-2/B4; the diagonal is never consumed downstream, f3/f4 read
        // off-diagonal f2 only, but it must not spuriously differ across paths).
        // This is the AT2 pairwise-complete path: a SNP contributes to pair (i, j)
        // only when it is valid in BOTH populations (for the diagonal, that is
        // simply "valid in i").
        for (int i = 0; i < P; ++i) {
            for (int j = i; j < P; ++j) {
                // The cancellation-free per-pair oracle over the whole tensor's SNP
                // range [0, M) — the SHARED f2_pair_over_range body (het correction,
                // unbiased summand, long-double pairwise accumulation, finalize with the
                // Vpair==0 ⇒ 0 guard; [7.1] dedup). The numerator is held wider than
                // native (long double) — the CPU analogue of the GPU's native-FP64
                // numerator/divide step (architecture.md §12).
                const F2Pair pr = f2_pair_over_range(Q, V, N, i, j, 0, M, terms.data());

                // Mirror into both triangles of the symmetric column-major outputs.
                const std::size_t ij =
                    static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(j);
                const std::size_t ji =
                    static_cast<std::size_t>(j) + static_cast<std::size_t>(P) * static_cast<std::size_t>(i);
                out.f2[ij] = pr.f2;
                out.f2[ji] = pr.f2;
                out.vpair[ij] = pr.vpair;
                out.vpair[ji] = pr.vpair;
            }
        }

        return out;
    }

    /// The CPU REFERENCE per-block f2 oracle (architecture.md §5 S2, §11.1;
    /// ROADMAP M4). The obviously-correct scalar/long-double transcription the GPU
    /// grouped-batched path is diffed against. For each block (the contiguous run
    /// of SNPs with the same `block_id`) it runs the SAME cancellation-free
    /// per-SNP-difference + long-double pairwise accumulation as `compute_f2`,
    /// using the SHARED `het_correction` / `f2_term` / `finalize_f2` primitives so
    /// the oracle and the GPU feeder cannot diverge on the formula
    /// (architecture.md §13). `precision` governs the GPU GEMMs only and is ignored
    /// here (this is the native/long-double reference; architecture.md §12).
    ///
    /// Output layout matches F2BlockTensor: f2/vpair are n_block stacked
    /// column-major [P × P] slabs, entry (i,j,b) at `i + P·j + P·P·b`.
    [[nodiscard]] F2BlockTensor compute_f2_blocks(const core::MatView& Q, const core::MatView& V,
                                                  const core::MatView& N, const int* block_id,
                                                  int n_block,
                                                  const Precision& precision) override {
        (void)precision;  // oracle is always the long-double native reference (§12)

        const int P = Q.P;
        const long M = Q.M;

        F2BlockTensor out;
        out.P = P;
        out.n_block = n_block;
        out.block_sizes.assign(static_cast<std::size_t>(n_block < 0 ? 0 : n_block), 0);
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
        const std::size_t total = slab * static_cast<std::size_t>(n_block < 0 ? 0 : n_block);
        out.f2.assign(total, 0.0);
        out.vpair.assign(total, 0.0);
        if (P <= 0 || M <= 0 || n_block <= 0) return out;

        // Block columns are CONTIGUOUS in file order (assign_blocks is
        // non-decreasing), so a block is the half-open SNP range [begin, end). The
        // SINGLE-SOURCE inverse of assign_blocks (core::block_ranges) validates the
        // partition contract ONCE (0 <= id < n_block, non-decreasing, block_id long
        // enough) and returns those ranges — closing the OOB write/read this scan
        // used to risk on a malformed partition (cleanup X-3/B3). The per-block SNP
        // count is the S4 metadata (F2BlockTensor::block_sizes).
        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                               M, n_block);
        for (int b = 0; b < n_block; ++b) {
            out.block_sizes[static_cast<std::size_t>(b)] =
                static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
        }

        // Per-block, per-pair: the cancellation-free long-double reference, sharing
        // the exact primitives `compute_f2` uses (header rationale). Scratch sized
        // to the largest block so the [P×P×M] intermediate is never materialized.
        //
        // FULL i,j (INCLUDING the diagonal): the GPU grouped path computes every
        // (i,j) entry via the GEMM reformulation, and BOTH M0 compute_f2 paths
        // (GPU assemble_f2_kernel and the CPU oracle, j = i since cleanup X-2/B4)
        // likewise fill the diagonal (f2(i,i) = -hc_i - hc_i ⇒ -2·mean_het,
        // vpair(i,i) = i's valid-SNP count). We MATCH that here so the oracle, the
        // GPU grouped path, the M0 GPU/CPU pair, and the single-block == M0 check
        // all agree bit-consistently (the diagonal is never consumed downstream —
        // f3/f4 read off-diagonal f2 only — but it must not spuriously differ
        // across paths).
        std::vector<long double> terms(static_cast<std::size_t>(M));
        for (int b = 0; b < n_block; ++b) {
            const long s0 = ranges[static_cast<std::size_t>(b)].begin;
            const long s1 = ranges[static_cast<std::size_t>(b)].end;
            const std::size_t base = slab * static_cast<std::size_t>(b);
            for (int i = 0; i < P; ++i) {
                for (int j = i; j < P; ++j) {
                    // The SAME cancellation-free per-pair oracle as compute_f2, here over
                    // this block's SNP range [s0, s1) — the SHARED f2_pair_over_range body
                    // ([7.1] dedup), so the per-block oracle cannot drift from the
                    // whole-tensor oracle on the formula or accumulation order (§12/§13).
                    // Only the SNP range and the slab base offset differ.
                    const F2Pair pr = f2_pair_over_range(Q, V, N, i, j, s0, s1, terms.data());
                    const std::size_t ij = base +
                        static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(j);
                    const std::size_t ji = base +
                        static_cast<std::size_t>(j) + static_cast<std::size_t>(P) * static_cast<std::size_t>(i);
                    out.f2[ij] = pr.f2;
                    out.f2[ji] = pr.f2;
                    out.vpair[ij] = pr.vpair;
                    out.vpair[ji] = pr.vpair;
                }
            }
        }
        return out;
    }

    /// The CPU REFERENCE genotype decode (architecture.md §5 S0/S1; ROADMAP M1):
    /// the obviously-correct scalar transcription of the oracle math, the
    /// bit-for-bit target the GPU decode is diffed against. For each population p
    /// (segment of gathered individuals) and SNP s, sum the ref-allele codes (AC)
    /// and count non-missing individuals (AN) over the segment, then finalize via
    /// the SHARED primitive (core/internal/decode_af.hpp `genotype_code`,
    /// `accumulate_genotype`, `finalize_af`) so the GPU kernel and this oracle
    /// cannot diverge on the unpack / accumulation / missing-handling / divide.
    /// Integer accumulators, single FP64 divide ⇒ Q exact; N, V integer-valued ⇒ exact.
    [[nodiscard]] DecodeResult decode_af(const DecodeTileView& tile) override {
        const int P = tile.n_pop;
        const long M = static_cast<long>(tile.n_snp);

        DecodeResult out;
        out.P = P;
        out.M = M;
        const std::size_t pm =
            static_cast<std::size_t>(P) * static_cast<std::size_t>(M);
        out.q.assign(pm, 0.0);
        out.v.assign(pm, 0.0);
        out.n.assign(pm, 0.0);

        if (P <= 0 || M <= 0) return out;

        // One (population, SNP) entry at a time; reduce over the population's
        // individual segment. Column-major [P × M]: element (i, s) at i + P·s.
        for (int i = 0; i < P; ++i) {
            const std::size_t seg_begin = tile.pop_offsets[static_cast<std::size_t>(i)];
            const std::size_t seg_end = tile.pop_offsets[static_cast<std::size_t>(i) + 1];

            for (long s = 0; s < M; ++s) {
                const std::size_t byte_in_rec =
                    static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
                const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);
                std::int64_t ac = 0;  // Σ ref-allele copies over non-missing individuals
                std::int64_t an = 0;  // count of non-missing individuals

                for (std::size_t g = seg_begin; g < seg_end; ++g) {
                    const std::uint8_t byte =
                        tile.packed[g * tile.bytes_per_record + byte_in_rec];
                    const std::uint8_t code = core::genotype_code(byte, pos_in_byte);
                    core::accumulate_genotype(code, ac, an);  // shared inner step (A-1/B27)
                }

                const core::AfResult r = core::finalize_af(ac, an, tile.ploidy);
                const std::size_t off =
                    static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
                out.q[off] = r.q;
                out.v[off] = r.v;
                out.n[off] = r.n;
            }
        }
        return out;
    }

    // =====================================================================
    // qpAdm fit-engine reference (S3/S4/S5/S6) — native FP64, AT2-exact.
    // The math reproduces ADMIXTOOLS 2 R/qpadm.R + R/resampling.R verbatim
    // (design docs/design/fit-engine.md §4; the FROZEN CONTRACT §4) and was
    // validated bit-for-bit against the af6a8c2 golden via the R prototype. The
    // vectorization order is the AT2 `c(t(xmat))` ROW-MAJOR order k = j + nr*i
    // (design §2 F4Blocks doc) — Q is built in this same order so opt_A/opt_B's
    // c(t(xmat)) indexing of qinv lines up.
    // =====================================================================

    /// S3 — host-oracle f4 assembly (design §4 S3). The AT2 four-slab combine
    /// per block: X[i,j,b] = (f2(Li,R0)+f2(L0,Rj)-f2(L0,R0)-f2(Li,Rj))/2, where
    /// left_idx = [target] ++ sources, right_idx[0] = R0. Flatten k = j + nr*i.
    /// Native FP64 (ignores `precision`, OQ-5). Also computes the AT2 jackknife
    /// point estimate x_total + the est_to_loo replicate array x_loo (carried for
    /// S4/S7) so the totals→LOO conversion happens once.
    [[nodiscard]] F4Blocks assemble_f4(const F2BlockTensor& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override {
        (void)precision;  // native FP64 reference

        const int nl = static_cast<int>(left_idx.size()) - 1;  // sources (target prepended)
        const int nr = static_cast<int>(right_idx.size()) - 1;  // rights minus R0
        const int nb = f2.n_block;
        const int P = f2.P;
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

        F4Blocks out;
        out.nl = nl;
        out.nr = nr;
        out.n_block = nb;
        const std::size_t m = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
        out.x_blocks.assign(m * static_cast<std::size_t>(nb), 0.0);
        if (nl <= 0 || nr <= 0 || nb <= 0) return out;

        const int L0 = left_idx[0];
        const int R0 = right_idx[0];
        // f2 accessor: f2[i + P*j + P*P*b].
        const auto f2at = [&](int i, int j, int b) -> double {
            return f2.f2[static_cast<std::size_t>(i) +
                         static_cast<std::size_t>(P) * static_cast<std::size_t>(j) +
                         slab * static_cast<std::size_t>(b)];
        };
        for (int i = 0; i < nl; ++i) {
            const int Li = left_idx[static_cast<std::size_t>(i) + 1];
            for (int j = 0; j < nr; ++j) {
                const int Rj = right_idx[static_cast<std::size_t>(j) + 1];
                const std::size_t k = static_cast<std::size_t>(j) +
                                      static_cast<std::size_t>(nr) * static_cast<std::size_t>(i);
                for (int b = 0; b < nb; ++b) {
                    const double x = 0.5 * (f2at(Li, R0, b) + f2at(L0, Rj, b) -
                                            f2at(L0, R0, b) - f2at(Li, Rj, b));
                    out.x_blocks[k + m * static_cast<std::size_t>(b)] = x;
                }
            }
        }

        // est_to_loo + the AT2 jackknife point estimate, computed once here so S4
        // (Q) and S7 (LOO re-fits) share the SAME loo array. block_sizes are read
        // from the host tensor (OQ-3: AT2 block_lengths, NOT Vpair).
        compute_loo_and_total(out, f2.block_sizes);
        return out;
    }

    /// S4 — weighted block-jackknife covariance (design §4 S4; AT2
    /// jack_pairarr_stats). Consumes the per-entry LOO replicates carried on
    /// F4Blocks (x_loo) and the AT2 point estimate (x_total). Builds the xtau
    /// pseudo-values, Q = xtau·xtauᵀ/n_block (UNFUDGED, golden convention), then
    /// Qinv = inverse(Q with diag += fudge·tr(Q)). Native FP64.
    [[nodiscard]] JackknifeCov jackknife_cov(const F4Blocks& x,
                                             std::span<const int> block_sizes,
                                             double fudge,
                                             const Precision& precision) override {
        (void)precision;  // native FP64

        const int m = x.nl * x.nr;
        const int nb = x.n_block;
        JackknifeCov out;
        out.m = m;
        if (m <= 0 || nb <= 0) { out.status = Status::Ok; return out; }

        // n = Σ block_sizes; h_b = n / bl_b. AT2 xtau (design §4 S4):
        //   xtau[k,b] = ( est[k]*h_b - loo[k,b]*(h_b-1) - tot_line[k] ) / sqrt(h_b-1)
        // where est[k] = x_total[k] (the AT2 jackknife $est), and tot_line[k] is the
        // weighted.mean(loo, 1 - bl/n) line (computed in compute_loo_and_total).
        long double n_ld = 0.0L;
        for (int b = 0; b < nb; ++b) n_ld += static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
        const double n = static_cast<double>(n_ld);

        // m_sz: the flattened model dim (nl*nr) widened for indexing. Capital M is
        // reserved for the SNP axis file-wide ([6.3]; standard §4 row 2); this alias
        // names the model-dim widening so the two never read as the same letter.
        const std::size_t m_sz = static_cast<std::size_t>(m);
        std::vector<double> xtau(m_sz * static_cast<std::size_t>(nb), 0.0);
        for (int b = 0; b < nb; ++b) {
            const double bl = static_cast<double>(block_sizes[static_cast<std::size_t>(b)]);
            const double h = n / bl;
            const double sqrt_h_minus_1 = std::sqrt(h - 1.0);
            for (int k = 0; k < m; ++k) {
                const double loo = x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)];
                const double est = x.x_total[static_cast<std::size_t>(k)];
                const double totline = tot_line_[static_cast<std::size_t>(k)];
                xtau[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)] =
                    (est * h - loo * (h - 1.0) - totline) / sqrt_h_minus_1;
            }
        }
        // Q = xtau · xtauᵀ / numblocks  (m×m, symmetric, column-major).
        out.Q.assign(m_sz * m_sz, 0.0);
        for (int kk = 0; kk < m; ++kk) {
            for (int ll = kk; ll < m; ++ll) {
                long double acc = 0.0L;
                for (int b = 0; b < nb; ++b) {
                    acc += static_cast<long double>(
                               xtau[static_cast<std::size_t>(kk) + m_sz * static_cast<std::size_t>(b)]) *
                           static_cast<long double>(
                               xtau[static_cast<std::size_t>(ll) + m_sz * static_cast<std::size_t>(b)]);
                }
                const double v = static_cast<double>(acc / static_cast<long double>(nb));
                out.Q[static_cast<std::size_t>(kk) + m_sz * static_cast<std::size_t>(ll)] = v;
                out.Q[static_cast<std::size_t>(ll) + m_sz * static_cast<std::size_t>(kk)] = v;
            }
        }
        // Fudge (OQ-4): Qf = Q; diag(Qf) += fudge * tr(Q); Qinv = inverse(Qf).
        // The fudge-ridge idiom is single-sourced in ridge_diagonal ([7.4]); the
        // trace + diagonal-add op order is byte-identical to the former inline copy.
        std::vector<double> Qf = out.Q;
        ridge_diagonal(Qf, m, fudge);
        const core::LinAlgStatus st = core::inverse(Qf, m, out.Qinv);
        // Producer-side "always sized" contract ([10.2][MED]): core::inverse early-
        // returns on a singular Qf BEFORE zeroing its out-param, leaving out.Qinv
        // default-constructed EMPTY. Size+zero it here so EVERY downstream consumer
        // (als_weights/gls_weights -> als_ridge_solve, which indexes qinv[kr+m*kc])
        // sees a well-formed m×m buffer even on the non-SPD path. No parity impact:
        // the in-SPD-contract path already filled it, and NonSpdCovariance is gated
        // upstream (qpadm_fit.cpp) — this only removes the latent OOB read.
        if (!st.ok) out.Qinv.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
        out.status = st.ok ? Status::Ok : Status::NonSpdCovariance;
        return out;
    }

    /// S5 — rank test / SVD seed (design §4 S5). Seeds A,B from svd(x_total) at
    /// rank r and returns chisq = vec(E)'·Qinv·vec(E) for E = X - A·B with the
    /// seed factors. In M(fit-1) this is the ALS seed; the rank sweep is M(fit-2).
    [[nodiscard]] GlsWeights rank_test(const F4Blocks& x,
                                       const JackknifeCov& cov,
                                       int r,
                                       const Precision& precision) override {
        (void)precision;
        GlsWeights gw;
        gw.r = r;
        const int nl = x.nl, nr = x.nr;
        // xmat as nl×nr column-major from x_total (k = j + nr*i row-major source).
        std::vector<double> xmat = xmat_from_total(x);
        seed_AB(xmat, nl, nr, r, gw.A, gw.B);
        gw.chisq = chisq_of(xmat, gw.A, gw.B, nl, nr, r, cov.Qinv);
        gw.status = Status::Ok;
        return gw;
    }

    /// S5 SWEEP — the qpWave / qpAdm RANK TEST over r = 0..rmax (rmax =
    /// min(nl,nr)-1). The native ORACLE the GPU rank_sweep is diffed against
    /// (M(fit-2)). Per r: chisq(r) = the rank-r ALS-refined residual quadratic form
    /// (REUSES `als_weights`@808 — the exact M(fit-1)/M(fit-4) chisq machinery,
    /// including the r==0 trivial branch); dof(r) = (nl-r)*(nr-r); p(r) =
    /// pchisq_upper. Then the AT2 res$rankdrop nested table (rows f4rank DESCENDING
    /// rmax..0; the nested diff compares each row to the next-lower rank, the last
    /// row NA) and f4rank (the smallest non-rejected rank: ASCENDING scan, the first
    /// r with p(r) > alpha). Native FP64 throughout (CpuBackend ignores precision).
    [[nodiscard]] RankSweep rank_sweep(const F4Blocks& x,
                                       const JackknifeCov& cov,
                                       double alpha,
                                       const QpAdmOptions& opts,
                                       const Precision& precision) override {
        (void)precision;  // the oracle is ALWAYS native FP64 (§4 carve-out)
        RankSweep rs;
        const int nl = x.nl, nr = x.nr;
        const int m = nl * nr;
        const int rmax = (nl < nr ? nl : nr) - 1;
        if (rmax < 0) {  // a degenerate (0-row/0-col) model — no candidate rank
            rs.status = Status::RankDeficient;
            // SVD dispatch report (1=gesvdjBatched, 2=per-model gesvd); see the
            // full normal-path note below for the kGesvdjMaxDim crossover.
            rs.svd_path = (nl <= kGesvdjMaxDim && nr <= kGesvdjMaxDim) ? 1 : 2;
            return rs;
        }
        std::vector<double> xmat = xmat_from_total(x);

        // ---- per-rank sweep (chisq via the ALS-refined rank-r fit) ----
        rs.chisq.assign(static_cast<std::size_t>(rmax) + 1, 0.0);
        rs.dof.assign(static_cast<std::size_t>(rmax) + 1, 0);
        rs.p.assign(static_cast<std::size_t>(rmax) + 1, 0.0);
        bool degenerate = false;
        for (int r = 0; r <= rmax; ++r) {
            const GlsWeights gw = als_weights(xmat, nl, nr, r, cov.Qinv, opts);
            if (gw.status != Status::Ok) degenerate = true;
            rs.chisq[static_cast<std::size_t>(r)] = gw.chisq;
            rs.dof[static_cast<std::size_t>(r)] = core::qpadm::qpadm_dof(nl, nr, r);
            rs.p[static_cast<std::size_t>(r)] =
                core::internal::pchisq_upper(gw.chisq, rs.dof[static_cast<std::size_t>(r)]);
        }

        // ---- AT2 res$rankdrop nested table (rows: rank rmax, rmax-1, ..., 0) ----
        const std::size_t n = static_cast<std::size_t>(rmax) + 1;
        rs.rd_f4rank.resize(n); rs.rd_dof.resize(n); rs.rd_chisq.resize(n);
        rs.rd_p.resize(n); rs.rd_dofdiff.resize(n);
        rs.rd_chisqdiff.resize(n); rs.rd_p_nested.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const int r = rmax - static_cast<int>(k);  // this row's rank (descending)
            rs.rd_f4rank[k] = r;
            rs.rd_dof[k] = rs.dof[static_cast<std::size_t>(r)];
            rs.rd_chisq[k] = rs.chisq[static_cast<std::size_t>(r)];
            rs.rd_p[k] = rs.p[static_cast<std::size_t>(r)];
            if (r - 1 >= 0) {  // nested diff to the next-lower rank (r-1)
                const int dof_diff = rs.dof[static_cast<std::size_t>(r - 1)] -
                                     rs.dof[static_cast<std::size_t>(r)];
                const double chisq_diff = rs.chisq[static_cast<std::size_t>(r - 1)] -
                                          rs.chisq[static_cast<std::size_t>(r)];
                rs.rd_dofdiff[k] = dof_diff;
                rs.rd_chisqdiff[k] = chisq_diff;
                rs.rd_p_nested[k] = core::internal::pchisq_upper(chisq_diff, dof_diff);
            } else {  // the last row (rank 0): NA
                rs.rd_dofdiff[k] = INT_MIN;
                rs.rd_chisqdiff[k] = std::numeric_limits<double>::quiet_NaN();
                rs.rd_p_nested[k] = std::numeric_limits<double>::quiet_NaN();
            }
        }

        // ---- f4rank: the smallest non-rejected rank (p(r) > alpha, ASCENDING) ----
        rs.f4rank = rmax;  // fallback: the highest rank if all rejected
        for (int r = 0; r <= rmax; ++r) {
            if (rs.p[static_cast<std::size_t>(r)] > alpha) { rs.f4rank = r; break; }
        }

        // ---- rank_Q: numerical rank of the (unfudged) covariance Q (m×m) ----
        // AT2 model_well_determined$rank_Q: the count of singular values of Q above
        // a relative tolerance (here ~ m·eps·σ_max). Observability (not gated).
        if (!cov.Q.empty() && cov.m == m) {
            const core::SvdResult sq = core::jacobi_svd(cov.Q, m, m);
            const double smax = sq.S.empty() ? 0.0 : sq.S.front();
            const double tol = smax * static_cast<double>(m) *
                               std::numeric_limits<double>::epsilon();
            int rk = 0;
            for (double s : sq.S) if (s > tol) ++rk;
            rs.rank_Q = rk;
        } else {
            rs.rank_Q = m;
        }

        // ---- dispatch report (SPEC §5:231): which SVD path WOULD be selected ----
        // 1 = gesvdjBatched (both dims <= kGesvdjMaxDim); 2 = per-model gesvd (else).
        // The EXECUTED SVD here is the deterministic Jacobi seed at all sizes (M(fit-2)
        // decision); the report uses the SAME kGesvdjMaxDim crossover as the CudaBackend.
        rs.svd_path = (nl <= kGesvdjMaxDim && nr <= kGesvdjMaxDim) ? 1 : 2;

        rs.status = degenerate ? Status::RankDeficient
                               : (cov.status != Status::Ok ? cov.status : Status::Ok);
        return rs;
    }

    /// S6 — GLS weights via AT2 ALS (design §4 S6; AT2 qpadm_weights). svd seed →
    /// opt_A/opt_B for opts.als_iterations with the fudge ridge → the constrained
    /// weight solve (the literal "single Cholesky" — here an LU solve matching R's
    /// solve()) → normalize Σw=1. Native FP64.
    [[nodiscard]] GlsWeights gls_weights(const F4Blocks& x,
                                         const JackknifeCov& cov,
                                         int r,
                                         const QpAdmOptions& opts,
                                         const Precision& precision) override {
        (void)precision;
        const int nl = x.nl, nr = x.nr;
        std::vector<double> xmat = xmat_from_total(x);
        return als_weights(xmat, nl, nr, r, cov.Qinv, opts);
    }

    /// S7 — leave-one-block-out weight re-fits (the ORACLE host loop). This is the
    /// per-block re-solve body that se_from_loo used to run inline (nested_models.cpp
    /// host loop); moving it here behind the batched-capable virtual makes se_from_loo
    /// backend-agnostic (the CUDA backend runs the SAME re-fits BATCHED on-device; the
    /// FROZEN CONTRACT §2e). For each block b: build a one-block F4Blocks whose
    /// x_total is the LOO replicate slice loo[:,:,b] (gls_weights reads x_total →
    /// xmat), REUSING cov.Qinv unchanged (the AT2 parity pin). Returns wmat[nb*nl]
    /// row-major (b*nl + i). Native FP64.
    ///
    /// The per-block AT2 ALS is shared with S7: this seam reuses gls_weights per LOO
    /// block (design §3 nested_models), so no extra ALS virtual is needed — the S7
    /// driver in core/qpadm reaches the same solve through gls_weights with a
    /// one-block F4Blocks.
    [[nodiscard]] std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override {
        const int nl = x.nl, nr = x.nr, nb = x.n_block;
        const int m = nl * nr;
        // m_sz: the flattened model dim (nl*nr) widened for indexing; capital M is
        // reserved for the SNP axis file-wide ([6.3]; standard §4 row 2).
        const std::size_t m_sz = static_cast<std::size_t>(m);
        std::vector<double> wmat(static_cast<std::size_t>(nb < 0 ? 0 : nb) *
                                 static_cast<std::size_t>(nl), 0.0);
        if (m <= 0 || nb <= 0 || nl <= 0) return wmat;
        for (int b = 0; b < nb; ++b) {
            F4Blocks rep;
            rep.nl = nl;
            rep.nr = nr;
            rep.n_block = 1;
            rep.x_total.assign(m_sz, 0.0);
            for (int k = 0; k < m; ++k)
                rep.x_total[static_cast<std::size_t>(k)] =
                    x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)];
            const GlsWeights gw = gls_weights(rep, cov, r, opts, precision);
            for (int i = 0; i < nl; ++i)
                wmat[static_cast<std::size_t>(b) * static_cast<std::size_t>(nl) +
                     static_cast<std::size_t>(i)] =
                    (i < static_cast<int>(gw.w.size())) ? gw.w[static_cast<std::size_t>(i)] : 0.0;
        }
        return wmat;
    }

private:
    // tot_line_ caches the AT2 weighted.mean(loo, 1 - bl/n) line vector (length m)
    // computed in assemble_f4's compute_loo_and_total and consumed by jackknife_cov
    // (the xtau centering term). One model is fit at a time on this backend
    // instance (single-model M(fit-1)); the cache is rebuilt per assemble_f4 call.
    std::vector<double> tot_line_{};

    /// est_to_loo (AT2 R/resampling.R) + the AT2 jackknife point estimate
    /// (jack_pairarr_stats `est`) + the centering line `tot` — all in one pass so
    /// S4/S7 reuse the SAME loo array. With no missing blocks (M(fit-1), OQ-12):
    ///   tot_ij    = weighted.mean(X[i,j,:], bl)
    ///   loo[k,b]  = (tot_ij - X[k,b]*rel_b) / (1 - rel_b),  rel_b = bl_b / Σbl
    ///   tot_line  = weighted.mean(loo, 1 - bl/n)
    ///   est[k]    = mean(tot_line - loo[k,:])*nb + weighted.mean(loo[k,:], bl)
    void compute_loo_and_total(F4Blocks& x, const std::vector<int>& block_sizes) {
        const int nl = x.nl, nr = x.nr, nb = x.n_block;
        const int m = nl * nr;
        // m_sz: the flattened model dim (nl*nr) widened for indexing; capital M is
        // reserved for the SNP axis file-wide ([6.3]; standard §4 row 2).
        const std::size_t m_sz = static_cast<std::size_t>(m);
        x.x_loo.assign(m_sz * static_cast<std::size_t>(nb), 0.0);
        x.x_total.assign(m_sz, 0.0);
        tot_line_.assign(m_sz, 0.0);
        if (m <= 0 || nb <= 0) return;

        long double n_ld = 0.0L;
        for (int b = 0; b < nb; ++b) n_ld += static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
        const double n = static_cast<double>(n_ld);

        for (int k = 0; k < m; ++k) {
            // tot_ij = weighted.mean(X[k,:], bl) = Σ X*bl / Σ bl
            long double num = 0.0L;
            for (int b = 0; b < nb; ++b) {
                num += static_cast<long double>(x.x_blocks[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)]) *
                       static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
            }
            const double tot_ij = static_cast<double>(num / n_ld);
            // loo[k,b] = (tot_ij - X*rel_b)/(1-rel_b)
            for (int b = 0; b < nb; ++b) {
                const double bl = static_cast<double>(block_sizes[static_cast<std::size_t>(b)]);
                const double rel = bl / n;
                const double xv = x.x_blocks[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)];
                x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)] =
                    (tot_ij - xv * rel) / (1.0 - rel);
            }
            // tot_line[k] = weighted.mean(loo[k,:], 1 - bl/n)
            long double wmean_num = 0.0L, wmean_den = 0.0L;
            for (int b = 0; b < nb; ++b) {
                const double w = 1.0 - static_cast<double>(block_sizes[static_cast<std::size_t>(b)]) / n;
                wmean_num += static_cast<long double>(x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)]) *
                             static_cast<long double>(w);
                wmean_den += static_cast<long double>(w);
            }
            tot_line_[static_cast<std::size_t>(k)] = static_cast<double>(wmean_num / wmean_den);
            // est[k] = mean(tot_line - loo)*nb + weighted.mean(loo, bl)
            long double diffsum = 0.0L;
            for (int b = 0; b < nb; ++b) {
                diffsum += static_cast<long double>(tot_line_[static_cast<std::size_t>(k)]) -
                           static_cast<long double>(x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)]);
            }
            const double term1 = static_cast<double>(diffsum / static_cast<long double>(nb)) *
                                 static_cast<double>(nb);
            long double wmean_bl_num = 0.0L;
            for (int b = 0; b < nb; ++b) {
                wmean_bl_num += static_cast<long double>(x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)]) *
                                static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
            }
            const double term2 = static_cast<double>(wmean_bl_num / n_ld);
            x.x_total[static_cast<std::size_t>(k)] = term1 + term2;
        }
    }

    /// Build the nl×nr COLUMN-MAJOR xmat from the row-major x_total vector
    /// (k = j + nr*i ⇒ xmat(i,j) at i + nl*j).
    [[nodiscard]] static std::vector<double> xmat_from_total(const F4Blocks& x) {
        const int nl = x.nl, nr = x.nr;
        std::vector<double> xmat(static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr), 0.0);
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)] =
                    x.x_total[static_cast<std::size_t>(j) + static_cast<std::size_t>(nr) * static_cast<std::size_t>(i)];
        return xmat;
    }

    /// Build an nl×nr COLUMN-MAJOR xmat from a single LOO block's row-major slice
    /// (the S7 per-block re-fit input; loo[k] for k = j + nr*i at block b).
    [[nodiscard]] static std::vector<double> xmat_from_loo_block(const F4Blocks& x, int b) {
        const int nl = x.nl, nr = x.nr;
        const std::size_t M = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
        std::vector<double> xmat(M, 0.0);
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j) {
                const std::size_t k = static_cast<std::size_t>(j) +
                                      static_cast<std::size_t>(nr) * static_cast<std::size_t>(i);
                xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)] =
                    x.x_loo[k + M * static_cast<std::size_t>(b)];
            }
        return xmat;
    }

    /// SVD seed: B = t(V[:,0:r]) (r×nr), A = xmat · t(B) (nl×r). xmat is nl×nr
    /// column-major; A,B returned column-major. (AT2 qpadm_weights seed.)
    static void seed_AB(const std::vector<double>& xmat, int nl, int nr, int r,
                        std::vector<double>& A, std::vector<double>& B) {
        const core::SvdResult sv = core::jacobi_svd(xmat, nl, nr);
        // B (r×nr): B[p,j] = V[j,p]  (V is nr×k column-major).
        B.assign(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
        for (int p = 0; p < r; ++p)
            for (int j = 0; j < nr; ++j)
                B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)] =
                    sv.V[static_cast<std::size_t>(j) + static_cast<std::size_t>(nr) * static_cast<std::size_t>(p)];
        // A (nl×r) = xmat (nl×nr) · t(B) (nr×r).
        A.assign(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r), 0.0);
        for (int i = 0; i < nl; ++i)
            for (int p = 0; p < r; ++p) {
                double acc = 0.0;
                for (int j = 0; j < nr; ++j)
                    acc += xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)] *
                           B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)];
                A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)] = acc;
            }
    }

    /// Ridge the diagonal of an n×n column-major matrix IN PLACE: tr = Σ diag;
    /// then diag(mat) += fudge·tr. The single source for the AT2 fudge-ridge idiom
    /// (OQ-4) used by jackknife_cov (on Qf, dim m) and als_ridge_solve (on coeffs,
    /// dim t). PARITY: the trace is a plain left-to-right `double` sum over the
    /// diagonal and the per-diagonal add is `mat += fudge*tr` — byte-identical to
    /// the three former inline copies, so the oracle stays bit-stable ([7.4]).
    static void ridge_diagonal(std::vector<double>& mat, int n, double fudge) {
        double tr = 0.0;
        for (int k = 0; k < n; ++k)
            tr += mat[static_cast<std::size_t>(k) + static_cast<std::size_t>(n) * static_cast<std::size_t>(k)];
        for (int k = 0; k < n; ++k)
            mat[static_cast<std::size_t>(k) + static_cast<std::size_t>(n) * static_cast<std::size_t>(k)] += fudge * tr;
    }

    /// xvec[k] = c(t(xmat))[k] = xmat(i,j) for k = i*nr + j (the AT2 row-major
    /// flatten of t(xmat); m = nl·nr). The single source for the byte-identical
    /// xvec build shared by opt_A/opt_B ([7.2]); xmat is nl×nr column-major.
    [[nodiscard]] static std::vector<double> als_xvec(const std::vector<double>& xmat,
                                                      int nl, int nr) {
        const int m = nl * nr;
        std::vector<double> xvec(static_cast<std::size_t>(m));
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xvec[static_cast<std::size_t>(i * nr + j)] =
                    xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)];
        return xvec;
    }

    /// AT2 GLS-ridge least-squares CORE shared by opt_A and opt_B ([7.1]). Given the
    /// vectorized linear operator `linop(k, a)` = the m×t Kronecker matrix L (k the
    /// row-major SNP-pair index 0..m-1, a the vectorized-unknown index 0..t-1),
    /// solves the ridged normal equations
    ///   coeffs = Lᵀ·qinv·L  (t×t) ; rhs = xvecᵀ·qinv·L  (length t) ;
    ///   diag(coeffs) += fudge·tr(coeffs) ;  solved = solve(coeffs, rhs)
    /// and returns `solved` (length t; EMPTY on a singular solve). qinv is m×m
    /// column-major; W = qinv·L (m×t) is formed once and reused for both coeffs and
    /// rhs — the EXACT op order of the two former 58-line copies.
    ///
    /// PARITY (load-bearing, architecture.md §12/§13): the accumulation loops here
    /// are byte-identical to the originals — W's inner sum is over `kc` (the qinv
    /// column), coeffs' over `k`, rhs' over `k`, all plain left-to-right `double`
    /// sums; the only per-caller change is the operator value `linop(k, a)`, pure
    /// index arithmetic yielding the same `double`, so the multiply/add order — and
    /// therefore the FP64 oracle — is bit-identical to opt_A/opt_B pre-extraction.
    template <typename Linop>
    [[nodiscard]] static std::vector<double> als_ridge_solve(
        const Linop& linop, int m, int t, const std::vector<double>& xvec,
        const std::vector<double>& qinv, double fudge) {
        // W = qinv (m×m) · L (m×t) → m×t
        std::vector<double> W(static_cast<std::size_t>(m) * static_cast<std::size_t>(t), 0.0);
        for (int kr = 0; kr < m; ++kr)
            for (int a = 0; a < t; ++a) {
                double acc = 0.0;
                for (int kc = 0; kc < m; ++kc)
                    acc += qinv[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(kc)] *
                           linop(kc, a);
                W[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(a)] = acc;
            }
        // coeffs = Lᵀ (t×m) · W (m×t) → t×t ; rhs[a] = Σ_k xvec[k]·W[k,a]
        std::vector<double> coeffs(static_cast<std::size_t>(t) * static_cast<std::size_t>(t), 0.0);
        std::vector<double> rhs(static_cast<std::size_t>(t), 0.0);
        for (int a = 0; a < t; ++a) {
            for (int c = 0; c < t; ++c) {
                double acc = 0.0;
                for (int k = 0; k < m; ++k) acc += linop(k, a) * W[static_cast<std::size_t>(k) + static_cast<std::size_t>(m) * static_cast<std::size_t>(c)];
                coeffs[static_cast<std::size_t>(a) + static_cast<std::size_t>(t) * static_cast<std::size_t>(c)] = acc;
            }
            double rr = 0.0;
            for (int k = 0; k < m; ++k) rr += xvec[static_cast<std::size_t>(k)] * W[static_cast<std::size_t>(k) + static_cast<std::size_t>(m) * static_cast<std::size_t>(a)];
            rhs[static_cast<std::size_t>(a)] = rr;
        }
        ridge_diagonal(coeffs, t, fudge);
        std::vector<double> solved;
        const core::LinAlgStatus st = core::solve(coeffs, t, rhs, solved);
        if (!st.ok) return {};
        return solved;
    }

    /// AT2 opt_A: A (nl×r) minimizing c(E)'·qinv·c(E), E = xmat - A·B, given B
    /// (r×nr). The operator is B2 = I_{nl} ⊗ B ((nl·r)×m): with a = i*r + p (block i
    /// of B) and k = i'*nr + j, B2[a,k] = (i==i') ? B[p,j] : 0. als_ridge_solve takes
    /// L(k, a) = B2[a, k] (the canonical m×t form, args transposed vs B2), then we
    /// reshape rowwise (A2[a], a = i*r + p ⇒ A(i,p)) into A. Column-major; m = nl·nr.
    static std::vector<double> opt_A(const std::vector<double>& B,
                                     const std::vector<double>& xmat, int nl, int nr, int r,
                                     const std::vector<double>& qinv, double fudge) {
        const int m = nl * nr;
        const int t = nl * r;  // dim of A (vectorized)
        // L(k, a) = B2[a, k] : a = i*r + p, k = i'*nr + j ⇒ (i==i') ? B[p,j] : 0.
        const auto L = [&](int k, int a) -> double {
            const int i = a / r, p = a % r;
            const int ii = k / nr, j = k % nr;
            return (i == ii) ? B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)]
                             : 0.0;
        };
        const std::vector<double> xvec = als_xvec(xmat, nl, nr);
        const std::vector<double> A2 = als_ridge_solve(L, m, t, xvec, qinv, fudge);
        // A = matrix(A2, nl, byrow=TRUE): A2[a], a = i*r + p ⇒ A(i,p). Column-major out.
        std::vector<double> A(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r), 0.0);
        if (!A2.empty())
            for (int i = 0; i < nl; ++i)
                for (int p = 0; p < r; ++p)
                    A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)] =
                        A2[static_cast<std::size_t>(i * r + p)];
        return A;
    }

    /// AT2 opt_B: B (r×nr) minimizing c(E)'·qinv·c(E), E = xmat - A·B, given A
    /// (nl×r). The operator is A2 = A ⊗ I_{nr} (m×(r·nr)): row k = i*nr + j, col
    /// c = p*nr + jc, A2[k,c] = (j==jc) ? A[i,p] : 0 — already the canonical m×t
    /// form, so als_ridge_solve takes L(k, c) = A2[k, c] directly, then we reshape
    /// rowwise (B2[c], c = p*nr + j ⇒ B(p,j)) into B. Column-major; m = nl·nr.
    static std::vector<double> opt_B(const std::vector<double>& A,
                                     const std::vector<double>& xmat, int nl, int nr, int r,
                                     const std::vector<double>& qinv, double fudge) {
        const int m = nl * nr;
        const int t = r * nr;  // dim of B (vectorized)
        // L(k, c) = A2[k, c] : k = i*nr + j, c = p*nr + jc ⇒ (j==jc) ? A[i,p] : 0.
        const auto L = [&](int k, int c) -> double {
            const int i = k / nr, j = k % nr;
            const int p = c / nr, jc = c % nr;
            return (j == jc) ? A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)]
                             : 0.0;
        };
        const std::vector<double> xvec = als_xvec(xmat, nl, nr);
        const std::vector<double> B2 = als_ridge_solve(L, m, t, xvec, qinv, fudge);
        // B = matrix(B2, ncol=nr, byrow=TRUE): B2[c], c = p*nr + j ⇒ B(p,j). Col-major out.
        std::vector<double> B(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
        if (!B2.empty())
            for (int p = 0; p < r; ++p)
                for (int j = 0; j < nr; ++j)
                    B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)] =
                        B2[static_cast<std::size_t>(p * nr + j)];
        return B;
    }

    /// AT2 qpadm_weights full body (svd seed → ALS → constrained weight solve →
    /// normalize). Returns the normalized weights, the refined A,B, and chisq.
    [[nodiscard]] static GlsWeights als_weights(const std::vector<double>& xmat, int nl, int nr,
                                                int r, const std::vector<double>& qinv,
                                                const QpAdmOptions& opts) {
        GlsWeights gw;
        gw.r = r;
        if (r == 0) {  // AT2: rnk==0 ⇒ weights = 1 (single-source trivial)
            gw.w.assign(static_cast<std::size_t>(nl), 1.0);
            gw.A.assign(static_cast<std::size_t>(nl) * 0, 0.0);
            gw.B.assign(0, 0.0);
            gw.chisq = chisq_of(xmat, gw.A, gw.B, nl, nr, r, qinv);
            gw.status = Status::Ok;
            return gw;
        }
        std::vector<double> A, B;
        seed_AB(xmat, nl, nr, r, A, B);
        for (int it = 0; it < opts.als_iterations; ++it) {
            A = opt_A(B, xmat, nl, nr, r, qinv, opts.fudge);
            B = opt_B(A, xmat, nl, nr, r, qinv, opts.fudge);
        }
        gw.A = A;
        gw.B = B;
        // Constrained weight solve (AT2): x = t(cbind(A,1)) → (r+1)×nl ;
        // y = c(rep(0,r),1) ; rhs = crossprod(x) = x·t(x)? No: crossprod(x)=t(x)·x.
        // In R: x is (r+1)×nl ; crossprod(x) = t(x)·x = nl×nl ; lhs = crossprod(x,y)
        // = t(x)·y = nl-vector ; w = solve(rhs, lhs) (length nl). Build directly.
        const int rp = r + 1;
        // x = t(cbind(A, 1)) is (r+1)×nl ⇒ x(p,i) = xm(i,p) where xm = cbind(A,1)
        // (nl×(r+1)): xm(i,p)=A(i,p) for p<r, xm(i,r)=1. AT2 then solves
        //   rhs = crossprod(x) = t(x)·x = nl×nl : rhs(i,i') = Σ_p xm(i,p)·xm(i',p)
        //   lhs = crossprod(x,y) = t(x)·y, y = c(rep(0,r),1) ⇒ lhs(i) = xm(i,r) = 1
        //   w = solve(rhs, lhs) ; weights = w/Σw.
        const auto xm = [&](int i, int p) -> double {
            return (p < r) ? A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)]
                           : 1.0;
        };
        std::vector<double> RHS(static_cast<std::size_t>(nl) * static_cast<std::size_t>(nl), 0.0);
        std::vector<double> LHS(static_cast<std::size_t>(nl), 0.0);
        for (int i = 0; i < nl; ++i) {
            for (int ip = 0; ip < nl; ++ip) {
                double acc = 0.0;
                for (int p = 0; p < rp; ++p) acc += xm(i, p) * xm(ip, p);
                RHS[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(ip)] = acc;
            }
            LHS[static_cast<std::size_t>(i)] = xm(i, r);  // = 1
        }
        std::vector<double> w;
        const core::LinAlgStatus st = core::solve(RHS, nl, LHS, w);
        if (!st.ok) { gw.status = Status::RankDeficient; return gw; }
        double sum = 0.0;
        for (double v : w) sum += v;
        gw.w.assign(static_cast<std::size_t>(nl), 0.0);
        for (int i = 0; i < nl; ++i) gw.w[static_cast<std::size_t>(i)] = w[static_cast<std::size_t>(i)] / sum;
        gw.chisq = chisq_of(xmat, A, B, nl, nr, r, qinv);
        gw.status = Status::Ok;
        return gw;
    }

    /// chisq = vec(E)'·qinv·vec(E), E = xmat - A·B, vectorized ROW-MAJOR
    /// (k = j + nr*i = the AT2 c(t(res)) order; res = t(xmat - A·B) ⇒ c(res) is the
    /// row-major flatten of (xmat - A·B)). A is nl×r, B is r×nr (column-major).
    [[nodiscard]] static double chisq_of(const std::vector<double>& xmat,
                                         const std::vector<double>& A,
                                         const std::vector<double>& B,
                                         int nl, int nr, int r,
                                         const std::vector<double>& qinv) {
        const int m = nl * nr;
        std::vector<double> e(static_cast<std::size_t>(m), 0.0);
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j) {
                double ab = 0.0;
                for (int p = 0; p < r; ++p)
                    ab += A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)] *
                          B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)];
                const double resid =
                    xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)] - ab;
                e[static_cast<std::size_t>(i * nr + j)] = resid;  // row-major k = j + nr*i
            }
        long double acc = 0.0L;
        for (int a = 0; a < m; ++a) {
            long double row = 0.0L;
            for (int b = 0; b < m; ++b)
                row += static_cast<long double>(qinv[static_cast<std::size_t>(a) + static_cast<std::size_t>(m) * static_cast<std::size_t>(b)]) *
                       static_cast<long double>(e[static_cast<std::size_t>(b)]);
            acc += static_cast<long double>(e[static_cast<std::size_t>(a)]) * row;
        }
        return static_cast<double>(acc);
    }
};

}  // namespace

/// Factory for the CPU reference backend (declared in device/backend_factory.hpp,
/// X-9/B8). Returned as a base-class pointer so callers depend only on the CUDA-free
/// `ComputeBackend` interface (the DI seam, architecture.md §8; injected into
/// `Resources`, §9). The GPU factory `make_cuda_backend` mirrors this signature in
/// the same namespace.
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace steppe::device
