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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/internal/decode_af.hpp"     // genotype_code, genotype_valid, finalize_af (shared)
#include "core/internal/f2_estimator.hpp"  // het_correction, f2_term, finalize_f2 (shared primitive)
#include "core/internal/views.hpp"         // steppe::core::MatView (Q/V/N contract)
#include "device/backend.hpp"              // steppe::ComputeBackend, steppe::F2Result, DecodeResult
#include "steppe/config.hpp"               // steppe::Precision

namespace steppe::core {

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
    [[nodiscard]] F2Result compute_f2(const MatView& Q,
                                      const MatView& V,
                                      const MatView& N,
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
        // the diagonal and any untouched entry default to 0 (f2(i,i) = 0).
        out.f2.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(P), 0.0);
        out.vpair.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(P), 0.0);

        if (P <= 0 || M <= 0) return out;  // empty block: nothing to accumulate

        // Per-SNP summand scratch, reused across pairs. Sized to the SNP count so
        // we never materialize the [P × P × M] intermediate (architecture.md
        // §11.1): the reference is O(P²·M) time, O(M) extra space.
        std::vector<long double> terms(static_cast<std::size_t>(M));

        // Walk the upper triangle (i < j); f2 is symmetric, so mirror each
        // result to (j, i) and leave the diagonal at its zero default. This is
        // the AT2 pairwise-complete path: a SNP contributes to pair (i, j) only
        // when it is valid in BOTH populations.
        for (int i = 0; i < P; ++i) {
            for (int j = i + 1; j < P; ++j) {
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
                    const double hc_i = het_correction(p_i, N.element(i, s), true);
                    const double hc_j = het_correction(p_j, N.element(j, s), true);

                    // The unbiased per-SNP f2 summand (p_i - p_j)² - hc_i - hc_j,
                    // via the SHARED primitive in its cancellation-free form (it
                    // forms the difference and squares it directly). Accumulate in
                    // LONG DOUBLE: the summand is small, and the cross-SNP sum is
                    // where oracle accuracy is won (see file header).
                    terms[static_cast<std::size_t>(count)] =
                        static_cast<long double>(f2_term(p_i, p_j, hc_i, hc_j));
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
                    finalize_f2(static_cast<double>(numerator), vpair);

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

    /// The CPU REFERENCE genotype decode (architecture.md §5 S0/S1; ROADMAP M1):
    /// the obviously-correct scalar transcription of the oracle math, the
    /// bit-for-bit target the GPU decode is diffed against. For each population p
    /// (segment of gathered individuals) and SNP s, sum the ref-allele codes (AC)
    /// and count non-missing individuals (AN) over the segment, then finalize via
    /// the SHARED primitive (core/internal/decode_af.hpp `genotype_code`,
    /// `genotype_valid`, `finalize_af`) so the GPU kernel and this oracle cannot
    /// diverge on the unpack / missing-handling / divide. Integer accumulators,
    /// single FP64 divide ⇒ Q exact; N, V integer-valued ⇒ exact.
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
                long ac = 0;  // Σ ref-allele copies over non-missing individuals
                long an = 0;  // count of non-missing individuals

                for (std::size_t g = seg_begin; g < seg_end; ++g) {
                    const std::uint8_t byte =
                        tile.packed[g * tile.bytes_per_record + byte_in_rec];
                    const std::uint8_t code = core::genotype_code(byte, pos_in_byte);
                    if (core::genotype_valid(code)) {
                        ac += static_cast<long>(code);
                        ++an;
                    }
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
};

}  // namespace

/// Factory for the CPU reference backend. Returned as a base-class pointer so
/// callers depend only on the CUDA-free `ComputeBackend` interface (the DI seam,
/// architecture.md §8; injected into `Resources`, §9). The GPU factory mirrors
/// this signature in the device layer.
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace steppe::core
