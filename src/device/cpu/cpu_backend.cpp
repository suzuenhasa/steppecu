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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/domain/block_partition_rule.hpp" // core::block_ranges, core::BlockRange (the X-3/B3 single-source inverse)
#include "core/internal/decode_af.hpp"     // genotype_code, accumulate_genotype, finalize_af (shared)
#include "core/internal/f2_estimator.hpp"  // het_correction, f2_term, finalize_f2 (shared primitive)
#include "core/internal/small_linalg.hpp"  // core::solve/inverse/jacobi_svd (the qpAdm reference solvers)
#include "core/internal/views.hpp"         // steppe::core::MatView (Q/V/N contract)
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

/// The CPU reference backend: the scalar / long-double correctness oracle.
///
/// One method (`compute_f2`); later milestones add decode / gemm / jackknife /
/// svd here alongside the GPU backend. Move-only ownership via the base class's
/// deleted copy/move (architecture.md §9).
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
                long count = 0;  // pairwise-valid SNP count == Vpair(i, j)

                for (long s = 0; s < M; ++s) {
                    const bool vi = V.element(i, s) != 0.0;
                    const bool vj = V.element(j, s) != 0.0;
                    if (!vi || !vj) continue;  // not jointly valid: contributes nothing

                    const double p_i = Q.element(i, s);
                    const double p_j = Q.element(j, s);

                    // Het bias correction per population, via the SHARED primitive
                    // (q(1-q)/max(N-1, kHetCorrDenomFloor)). N is the non-missing
                    // HAPLOID count from the Q/V/N contract (2 × diploids, or
                    // 1 × pseudo-haploids for ancient DNA). Both entries are valid
                    // here, so `valid = true`.
                    const double hc_i = core::het_correction(p_i, N.element(i, s), true);
                    const double hc_j = core::het_correction(p_j, N.element(j, s), true);

                    // The unbiased per-SNP f2 summand (p_i - p_j)² - hc_i - hc_j,
                    // via the SHARED primitive in its cancellation-free form (it
                    // forms the difference and squares it directly). Accumulate in
                    // LONG DOUBLE: the summand is small, and the cross-SNP sum is
                    // where oracle accuracy is won (see file header).
                    terms[static_cast<std::size_t>(count)] =
                        static_cast<long double>(core::f2_term(p_i, p_j, hc_i, hc_j));
                    ++count;
                }

                // Cancellation-free cross-SNP accumulation, then the shared
                // finalize (numerator / Vpair with the Vpair==0 ⇒ 0 guard). The
                // numerator is held wider than native here (long double) — the CPU
                // analogue of the GPU's native-FP64 numerator/divide step
                // (architecture.md §12).
                const long double numerator =
                    pairwise_sum(terms.data(), static_cast<std::size_t>(count));
                const double vpair = static_cast<double>(count);
                const double f2_ij =
                    core::finalize_f2(static_cast<double>(numerator), vpair);

                // Mirror into both triangles of the symmetric column-major outputs.
                const std::size_t ij =
                    static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(j);
                const std::size_t ji =
                    static_cast<std::size_t>(j) + static_cast<std::size_t>(P) * static_cast<std::size_t>(i);
                out.f2[ij] = f2_ij;
                out.f2[ji] = f2_ij;
                out.vpair[ij] = vpair;
                out.vpair[ji] = vpair;
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
                    long count = 0;
                    for (long s = s0; s < s1; ++s) {
                        const bool vi = V.element(i, s) != 0.0;
                        const bool vj = V.element(j, s) != 0.0;
                        if (!vi || !vj) continue;
                        const double p_i = Q.element(i, s);
                        const double p_j = Q.element(j, s);
                        const double hc_i = core::het_correction(p_i, N.element(i, s), true);
                        const double hc_j = core::het_correction(p_j, N.element(j, s), true);
                        terms[static_cast<std::size_t>(count)] =
                            static_cast<long double>(core::f2_term(p_i, p_j, hc_i, hc_j));
                        ++count;
                    }
                    const long double numerator =
                        pairwise_sum(terms.data(), static_cast<std::size_t>(count));
                    const double vpair = static_cast<double>(count);
                    const double f2_ij =
                        core::finalize_f2(static_cast<double>(numerator), vpair);
                    const std::size_t ij = base +
                        static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(j);
                    const std::size_t ji = base +
                        static_cast<std::size_t>(j) + static_cast<std::size_t>(P) * static_cast<std::size_t>(i);
                    out.f2[ij] = f2_ij;
                    out.f2[ji] = f2_ij;
                    out.vpair[ij] = vpair;
                    out.vpair[ji] = vpair;
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
                const std::size_t byte_in_rec = static_cast<std::size_t>(s) / 4u;
                const int pos_in_byte = static_cast<int>(s & 3);
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

        const std::size_t M = static_cast<std::size_t>(m);
        std::vector<double> xtau(M * static_cast<std::size_t>(nb), 0.0);
        for (int b = 0; b < nb; ++b) {
            const double bl = static_cast<double>(block_sizes[static_cast<std::size_t>(b)]);
            const double h = n / bl;
            const double sh = std::sqrt(h - 1.0);
            for (int k = 0; k < m; ++k) {
                const double loo = x.x_loo[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)];
                const double est = x.x_total[static_cast<std::size_t>(k)];
                const double totline = tot_line_[static_cast<std::size_t>(k)];
                xtau[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)] =
                    (est * h - loo * (h - 1.0) - totline) / sh;
            }
        }
        // Q = xtau · xtauᵀ / numblocks  (m×m, symmetric, column-major).
        out.Q.assign(M * M, 0.0);
        for (int kk = 0; kk < m; ++kk) {
            for (int ll = kk; ll < m; ++ll) {
                long double acc = 0.0L;
                for (int b = 0; b < nb; ++b) {
                    acc += static_cast<long double>(
                               xtau[static_cast<std::size_t>(kk) + M * static_cast<std::size_t>(b)]) *
                           static_cast<long double>(
                               xtau[static_cast<std::size_t>(ll) + M * static_cast<std::size_t>(b)]);
                }
                const double v = static_cast<double>(acc / static_cast<long double>(nb));
                out.Q[static_cast<std::size_t>(kk) + M * static_cast<std::size_t>(ll)] = v;
                out.Q[static_cast<std::size_t>(ll) + M * static_cast<std::size_t>(kk)] = v;
            }
        }
        // Fudge (OQ-4): Qf = Q; diag(Qf) += fudge * tr(Q); Qinv = inverse(Qf).
        double tr = 0.0;
        for (int k = 0; k < m; ++k) tr += out.Q[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(k)];
        std::vector<double> Qf = out.Q;
        for (int k = 0; k < m; ++k)
            Qf[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(k)] += fudge * tr;
        const core::LinAlgStatus st = core::inverse(Qf, m, out.Qinv);
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

    // ---- AT2 ALS shared with S7 (a public-on-the-class entry for the LOO re-fit;
    //      design §3 nested_models reuses gls_weights per LOO block). The S7
    //      driver in core/qpadm calls gls_weights with a one-block F4Blocks, so no
    //      extra virtual is needed.

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
        const std::size_t M = static_cast<std::size_t>(m);
        x.x_loo.assign(M * static_cast<std::size_t>(nb), 0.0);
        x.x_total.assign(M, 0.0);
        tot_line_.assign(M, 0.0);
        if (m <= 0 || nb <= 0) return;

        long double n_ld = 0.0L;
        for (int b = 0; b < nb; ++b) n_ld += static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
        const double n = static_cast<double>(n_ld);

        for (int k = 0; k < m; ++k) {
            // tot_ij = weighted.mean(X[k,:], bl) = Σ X*bl / Σ bl
            long double num = 0.0L;
            for (int b = 0; b < nb; ++b) {
                num += static_cast<long double>(x.x_blocks[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)]) *
                       static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
            }
            const double tot_ij = static_cast<double>(num / n_ld);
            // loo[k,b] = (tot_ij - X*rel_b)/(1-rel_b)
            for (int b = 0; b < nb; ++b) {
                const double bl = static_cast<double>(block_sizes[static_cast<std::size_t>(b)]);
                const double rel = bl / n;
                const double xv = x.x_blocks[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)];
                x.x_loo[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)] =
                    (tot_ij - xv * rel) / (1.0 - rel);
            }
            // tot_line[k] = weighted.mean(loo[k,:], 1 - bl/n)
            long double wln = 0.0L, wld = 0.0L;
            for (int b = 0; b < nb; ++b) {
                const double w = 1.0 - static_cast<double>(block_sizes[static_cast<std::size_t>(b)]) / n;
                wln += static_cast<long double>(x.x_loo[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)]) *
                       static_cast<long double>(w);
                wld += static_cast<long double>(w);
            }
            tot_line_[static_cast<std::size_t>(k)] = static_cast<double>(wln / wld);
            // est[k] = mean(tot_line - loo)*nb + weighted.mean(loo, bl)
            long double diffsum = 0.0L;
            for (int b = 0; b < nb; ++b) {
                diffsum += static_cast<long double>(tot_line_[static_cast<std::size_t>(k)]) -
                           static_cast<long double>(x.x_loo[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)]);
            }
            const double term1 = static_cast<double>(diffsum / static_cast<long double>(nb)) *
                                 static_cast<double>(nb);
            long double wbn = 0.0L;
            for (int b = 0; b < nb; ++b) {
                wbn += static_cast<long double>(x.x_loo[static_cast<std::size_t>(k) + M * static_cast<std::size_t>(b)]) *
                       static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
            }
            const double term2 = static_cast<double>(wbn / n_ld);
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

    /// AT2 opt_A: A (nl×r) minimizing c(E)'·qinv·c(E), E = xmat - A·B, given B
    /// (r×nr). Builds B2 = I_{nl} ⊗ B ((nl·r)×m), coeffs = B2·qinv·t(B2)
    /// ((nl·r)×(nl·r)), rhs = c(t(xmat))·qinv·t(B2), ridge on the diagonal, solve,
    /// reshape rowwise into A. All matrices column-major; m = nl·nr; the qinv index
    /// and c(t(xmat)) use the row-major k = j + nr*i order.
    static std::vector<double> opt_A(const std::vector<double>& B,
                                     const std::vector<double>& xmat, int nl, int nr, int r,
                                     const std::vector<double>& qinv, double fudge) {
        const int m = nl * nr;
        const int t = nl * r;  // dim of A (vectorized)
        // B2 is (nl·r) × m. AT2: B2 = diag(nl) %x% B, with R's c(t(xmat)) row-major
        // (k = j + nr*i) convention. The (a, k) entry: a = i*r + p (block i of B),
        // k = i'*nr + j ; B2[a,k] = (i==i') ? B[p,j] : 0.
        // We compute coeffs = B2·qinv·t(B2) and rhs = xvecᵀ·qinv·t(B2) directly.
        // First W = qinv·t(B2): m × t.  (t(B2) is m × t.)
        const auto B2 = [&](int a, int k) -> double {
            const int i = a / r, p = a % r;
            const int ii = k / nr, j = k % nr;
            return (i == ii) ? B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)]
                             : 0.0;
        };
        // xvec[k] = c(t(xmat))[k] = xmat(i,j) for k = i*nr + j.
        std::vector<double> xvec(static_cast<std::size_t>(m));
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xvec[static_cast<std::size_t>(i * nr + j)] =
                    xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)];
        // W = qinv (m×m) · t(B2) (m×t)  → m×t
        std::vector<double> W(static_cast<std::size_t>(m) * static_cast<std::size_t>(t), 0.0);
        for (int kr = 0; kr < m; ++kr)
            for (int a = 0; a < t; ++a) {
                double acc = 0.0;
                for (int kc = 0; kc < m; ++kc)
                    acc += qinv[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(kc)] *
                           B2(a, kc);
                W[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(a)] = acc;
            }
        // coeffs = B2 (t×m) · W (m×t)  → t×t ; rhs[a] = Σ_k xvec[k]·W[k,a]
        std::vector<double> coeffs(static_cast<std::size_t>(t) * static_cast<std::size_t>(t), 0.0);
        std::vector<double> rhs(static_cast<std::size_t>(t), 0.0);
        for (int a = 0; a < t; ++a) {
            for (int c = 0; c < t; ++c) {
                double acc = 0.0;
                for (int k = 0; k < m; ++k) acc += B2(a, k) * W[static_cast<std::size_t>(k) + static_cast<std::size_t>(m) * static_cast<std::size_t>(c)];
                coeffs[static_cast<std::size_t>(a) + static_cast<std::size_t>(t) * static_cast<std::size_t>(c)] = acc;
            }
            double rr = 0.0;
            for (int k = 0; k < m; ++k) rr += xvec[static_cast<std::size_t>(k)] * W[static_cast<std::size_t>(k) + static_cast<std::size_t>(m) * static_cast<std::size_t>(a)];
            rhs[static_cast<std::size_t>(a)] = rr;
        }
        double tr = 0.0;
        for (int a = 0; a < t; ++a) tr += coeffs[static_cast<std::size_t>(a) + static_cast<std::size_t>(t) * static_cast<std::size_t>(a)];
        for (int a = 0; a < t; ++a) coeffs[static_cast<std::size_t>(a) + static_cast<std::size_t>(t) * static_cast<std::size_t>(a)] += fudge * tr;
        std::vector<double> A2;
        const core::LinAlgStatus st = core::solve(coeffs, t, rhs, A2);
        // A = matrix(A2, nl, byrow=TRUE): A2[a], a = i*r + p ⇒ A(i,p). Column-major out.
        std::vector<double> A(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r), 0.0);
        if (st.ok)
            for (int i = 0; i < nl; ++i)
                for (int p = 0; p < r; ++p)
                    A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)] =
                        A2[static_cast<std::size_t>(i * r + p)];
        return A;
    }

    /// AT2 opt_B: B (r×nr) minimizing c(E)'·qinv·c(E), E = xmat - A·B, given A
    /// (nl×r). Builds A2 = A ⊗ I_{nr} (m×(r·nr)), coeffs = t(A2)·qinv·A2, rhs =
    /// c(t(xmat))·qinv·A2, ridge, solve, reshape into B (byrow over nr columns).
    static std::vector<double> opt_B(const std::vector<double>& A,
                                     const std::vector<double>& xmat, int nl, int nr, int r,
                                     const std::vector<double>& qinv, double fudge) {
        const int m = nl * nr;
        const int t = r * nr;  // dim of B (vectorized)
        // A2 is m × (r·nr). AT2: A2 = A %x% diag(nr). Row k = i*nr + j (row-major),
        // col c = p*nr + jc. A2[k,c] = (j==jc) ? A[i,p] : 0.
        const auto A2 = [&](int k, int c) -> double {
            const int i = k / nr, j = k % nr;
            const int p = c / nr, jc = c % nr;
            return (j == jc) ? A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)]
                             : 0.0;
        };
        std::vector<double> xvec(static_cast<std::size_t>(m));
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xvec[static_cast<std::size_t>(i * nr + j)] =
                    xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)];
        // W = qinv (m×m) · A2 (m×t) → m×t
        std::vector<double> W(static_cast<std::size_t>(m) * static_cast<std::size_t>(t), 0.0);
        for (int kr = 0; kr < m; ++kr)
            for (int c = 0; c < t; ++c) {
                double acc = 0.0;
                for (int kc = 0; kc < m; ++kc)
                    acc += qinv[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(kc)] *
                           A2(kc, c);
                W[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(c)] = acc;
            }
        // coeffs = t(A2) (t×m) · W (m×t) → t×t ; rhs[c] = Σ_k xvec[k]·W[k,c]
        std::vector<double> coeffs(static_cast<std::size_t>(t) * static_cast<std::size_t>(t), 0.0);
        std::vector<double> rhs(static_cast<std::size_t>(t), 0.0);
        for (int a = 0; a < t; ++a) {
            for (int c = 0; c < t; ++c) {
                double acc = 0.0;
                for (int k = 0; k < m; ++k) acc += A2(k, a) * W[static_cast<std::size_t>(k) + static_cast<std::size_t>(m) * static_cast<std::size_t>(c)];
                coeffs[static_cast<std::size_t>(a) + static_cast<std::size_t>(t) * static_cast<std::size_t>(c)] = acc;
            }
            double rr = 0.0;
            for (int k = 0; k < m; ++k) rr += xvec[static_cast<std::size_t>(k)] * W[static_cast<std::size_t>(k) + static_cast<std::size_t>(m) * static_cast<std::size_t>(a)];
            rhs[static_cast<std::size_t>(a)] = rr;
        }
        double tr = 0.0;
        for (int a = 0; a < t; ++a) tr += coeffs[static_cast<std::size_t>(a) + static_cast<std::size_t>(t) * static_cast<std::size_t>(a)];
        for (int a = 0; a < t; ++a) coeffs[static_cast<std::size_t>(a) + static_cast<std::size_t>(t) * static_cast<std::size_t>(a)] += fudge * tr;
        std::vector<double> B2;
        const core::LinAlgStatus st = core::solve(coeffs, t, rhs, B2);
        // B = matrix(B2, ncol=nr, byrow=TRUE): B2[c], c = p*nr + j ⇒ B(p,j). Col-major out.
        std::vector<double> B(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
        if (st.ok)
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
