// src/device/cpu/cpu_backend.cpp
//
// CpuBackend — the CPU reference backend and the correctness oracle for the whole
// compute pipeline. It implements the same CUDA-free ComputeBackend interface as the
// GPU backend, but in plain host C++20 with obviously-correct scalar / long-double
// math, so the library runs with no GPU and every GPU result is continuously diffed
// against what this file produces. The GPU and CPU paths share no control structure,
// only the same small per-element primitives, so neither can drift from the other on
// the formula. Every method takes a Precision argument and every method ignores it:
// the precision knob governs only the GPU matmuls, while this oracle always computes
// in native double (wider, in long double, where accumulation is cancellation-
// sensitive). The math reproduces ADMIXTOOLS 2 exactly, or within a tight tolerance
// where floating-point order matters.
//
// Reference: docs/reference/src_device_cpu_cpu_backend.cpp.md

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/internal/dates_fit.hpp"
#include "core/internal/decode_af.hpp"
#include "core/internal/sfs_hist.hpp"
#include "core/internal/f2_estimator.hpp"
#include "core/internal/pchisq.hpp"
#include "core/internal/qpfstats_jackknife.hpp"
#include "core/internal/small_linalg.hpp"
#include "core/internal/wc_fst.hpp"
#include "core/internal/views.hpp"
#include "core/qpadm/qpadm_bounds.hpp"
#include "core/qpadm/qpgraph_model.hpp"
#include "core/qpadm/qpgraph_objective.hpp"
#include "core/qpadm/qpgraph_opt_constants.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "device/device_f2_blocks.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/qpadm.hpp"

namespace steppe::device {

namespace {

// Pairwise cascade summation (cancellation-free accumulator) — reference §3
[[nodiscard]] long double pairwise_sum(const long double* a, std::size_t n) noexcept {
    constexpr std::size_t kPairwiseBaseCase = 128;
    if (n == 0) return 0.0L;
    if (n <= kPairwiseBaseCase) {
        long double s = 0.0L;
        for (std::size_t k = 0; k < n; ++k) s += a[k];
        return s;
    }
    const std::size_t h = n / 2;
    return pairwise_sum(a, h) + pairwise_sum(a + h, n - h);
}

// detect_ploidy_host: per-sample ploidy prepass — reference §4
[[nodiscard]] std::vector<int> detect_ploidy_host(const DecodeTileView& tile) {
    std::vector<int> ploidy(tile.n_individuals, core::kPloidyPseudoHaploid);
    const std::size_t window =
        (static_cast<std::size_t>(core::kPloidyDetectSnps) < tile.n_snp)
            ? static_cast<std::size_t>(core::kPloidyDetectSnps)
            : tile.n_snp;
    if (window == 0 || tile.n_individuals == 0) return ploidy;
    for (std::size_t g = 0; g < tile.n_individuals; ++g) {
        const std::uint8_t* rec = tile.packed + g * tile.bytes_per_record;
        for (std::size_t s = 0; s < window; ++s) {
            const std::size_t byte_in_rec =
                s / static_cast<std::size_t>(core::kCodesPerByte);
            const int pos_in_byte =
                static_cast<int>(s % static_cast<std::size_t>(core::kCodesPerByte));
            if (core::genotype_code(rec[byte_in_rec], pos_in_byte) ==
                core::kHeterozygousGenotypeCode) {
                ploidy[g] = core::kPloidyDiploid;
                break;
            }
        }
    }
    return ploidy;
}

// se_sample_cov_diag: sample-covariance diagonal — reference §10
[[nodiscard]] std::vector<double> se_sample_cov_diag(const std::vector<double>& w,
                                                     int nrows, int ncols) {
    const std::size_t nc = static_cast<std::size_t>(ncols);
    const auto row_major_at = [&w, nc](int i, int c) -> double {
        return w[static_cast<std::size_t>(i) * nc + static_cast<std::size_t>(c)];
    };

    std::vector<double> mean(nc, 0.0);
    for (int i = 0; i < nrows; ++i)
        for (int c = 0; c < ncols; ++c)
            mean[static_cast<std::size_t>(c)] += row_major_at(i, c);
    for (int c = 0; c < ncols; ++c) mean[static_cast<std::size_t>(c)] /= static_cast<double>(nrows);

    std::vector<double> diag(nc, 0.0);
    for (int c = 0; c < ncols; ++c) {
        const double mc = mean[static_cast<std::size_t>(c)];
        long double acc = 0.0L;
        for (int i = 0; i < nrows; ++i) {
            const double d = row_major_at(i, c) - mc;
            acc += static_cast<long double>(d) * static_cast<long double>(d);
        }
        diag[static_cast<std::size_t>(c)] = static_cast<double>(acc / static_cast<long double>(nrows - 1));
    }
    return diag;
}

// survivor_blocks: drop partially-covered blocks — reference §5
[[nodiscard]] std::vector<int> survivor_blocks(const std::vector<double>& vpair,
                                               int P, int nb,
                                               const std::vector<int>& block_sizes) {
    std::vector<int> surv;
    if (nb <= 0) return surv;
    surv.reserve(static_cast<std::size_t>(nb));
    const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
    const bool have_vpair =
        !vpair.empty() && vpair.size() == slab * static_cast<std::size_t>(nb) &&
        block_sizes.size() == static_cast<std::size_t>(nb);
    for (int b = 0; b < nb; ++b) {
        bool keep = true;
        if (have_vpair) {
            const std::size_t base = slab * static_cast<std::size_t>(b);
            bool any_missing = false, any_present = false;
            for (std::size_t e = 0; e < slab; ++e) {
                if (core::pair_block_is_missing(vpair[base + e])) any_missing = true;
                else any_present = true;
                if (any_missing && any_present) break;
            }
            keep = !(any_missing && any_present);
        }
        if (keep) surv.push_back(b);
    }
    return surv;
}

// Per-pair f2 oracle body — reference §3
struct F2Pair { double f2 = 0.0; double vpair = 0.0; };

[[nodiscard]] F2Pair f2_pair_over_range(const core::MatView& Q, const core::MatView& V,
                                        const core::MatView& N, int i, int j,
                                        long s0, long s1, long double* terms) {
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
    const long double numerator = pairwise_sum(terms, static_cast<std::size_t>(count));
    const double vpair = static_cast<double>(count);
    return F2Pair{core::finalize_f2(static_cast<double>(numerator), vpair), vpair};
}

// CpuBackend: CPU reference backend / correctness oracle — reference §1
class CpuBackend final : public ComputeBackend {
public:
    // compute_f2: whole-tensor f2 oracle — reference §3
    [[nodiscard]] F2Result compute_f2(const core::MatView& Q,
                                      const core::MatView& V,
                                      const core::MatView& N,
                                      const Precision& precision) override {
        (void)precision;

        const int P = Q.P;
        const long M = Q.M;

        F2Result out;
        out.P = P;
        out.f2.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(P), 0.0);
        out.vpair.assign(static_cast<std::size_t>(P) * static_cast<std::size_t>(P), 0.0);

        if (P <= 0 || M <= 0) return out;

        std::vector<long double> terms(static_cast<std::size_t>(M));

        for (int i = 0; i < P; ++i) {
            for (int j = i; j < P; ++j) {
                const F2Pair pr = f2_pair_over_range(Q, V, N, i, j, 0, M, terms.data());

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

    // compute_f2_blocks: per-block f2 oracle — reference §3
    [[nodiscard]] F2BlockTensor compute_f2_blocks(const core::MatView& Q, const core::MatView& V,
                                                  const core::MatView& N, const int* block_id,
                                                  int n_block,
                                                  const Precision& precision) override {
        (void)precision;

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

        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                               M, n_block);
        for (int b = 0; b < n_block; ++b) {
            out.block_sizes[static_cast<std::size_t>(b)] =
                static_cast<int>(ranges[static_cast<std::size_t>(b)].size());
        }

        std::vector<long double> terms(static_cast<std::size_t>(M));
        for (int b = 0; b < n_block; ++b) {
            const long s0 = ranges[static_cast<std::size_t>(b)].begin;
            const long s1 = ranges[static_cast<std::size_t>(b)].end;
            const std::size_t base = slab * static_cast<std::size_t>(b);
            for (int i = 0; i < P; ++i) {
                for (int j = i; j < P; ++j) {
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

    // decode_af: genotype decode to allele frequencies — reference §4
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

        std::vector<int> detected;
        const int* eff_ploidy = tile.sample_ploidy;
        if (eff_ploidy == nullptr && tile.detect_ploidy_on_device) {
            detected = detect_ploidy_host(tile);
            eff_ploidy = detected.data();
        }

        for (int i = 0; i < P; ++i) {
            const std::size_t seg_begin = tile.pop_offsets[static_cast<std::size_t>(i)];
            const std::size_t seg_end = tile.pop_offsets[static_cast<std::size_t>(i) + 1];

            for (long s = 0; s < M; ++s) {
                const std::size_t byte_in_rec =
                    static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
                const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);
                double ac = 0.0;
                std::int64_t n = 0;

                for (std::size_t g = seg_begin; g < seg_end; ++g) {
                    const std::uint8_t byte =
                        tile.packed[g * tile.bytes_per_record + byte_in_rec];
                    const std::uint8_t code = core::genotype_code(byte, pos_in_byte);
                    const int pl = (eff_ploidy != nullptr)
                                       ? eff_ploidy[g]
                                       : tile.ploidy;
                    core::accumulate_genotype_ploidy(code, pl, ac, n);
                }

                const core::AfResult r = core::finalize_af_counts(ac, n);
                const std::size_t off =
                    static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
                out.q[off] = r.q;
                out.v[off] = r.v;
                out.n[off] = r.n;
            }
        }
        return out;
    }

    // fst_wc_per_site: the per-site Weir & Cockerham 1984 FST parity oracle. A host loop
    // calling the SAME wc_accumulate / wc_finalize the GPU kernel uses (native FP64), plus
    // a host Σnum/Σden/n_valid over the (valid && summary-included) sites — reference §4
    [[nodiscard]] FstPerSite fst_wc_per_site(
        const DecodeTileView& tile, int popA, int popB,
        std::span<const std::uint8_t> summary_include) override {
        FstPerSite out;
        out.precision_tag = Precision::Kind::Fp64;

        const long M = static_cast<long>(tile.n_snp);
        const int P = tile.n_pop;
        if (M <= 0 || P <= 0 || popA < 0 || popB < 0 || popA >= P || popB >= P) return out;

        const std::size_t Mz = static_cast<std::size_t>(M);
        out.num.assign(Mz, 0.0);
        out.den.assign(Mz, 0.0);
        out.fst.assign(Mz, 0.0);
        out.valid.assign(Mz, std::uint8_t{0});

        const std::size_t segA_begin = tile.pop_offsets[static_cast<std::size_t>(popA)];
        const std::size_t segA_end = tile.pop_offsets[static_cast<std::size_t>(popA) + 1];
        const std::size_t segB_begin = tile.pop_offsets[static_cast<std::size_t>(popB)];
        const std::size_t segB_end = tile.pop_offsets[static_cast<std::size_t>(popB) + 1];

        const bool have_inc = summary_include.size() == Mz;
        long double sum_num = 0.0L, sum_den = 0.0L;
        long n_valid = 0;

        for (long s = 0; s < M; ++s) {
            const std::size_t byte_in_rec =
                static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
            const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);

            core::WcPerPop A;
            for (std::size_t g = segA_begin; g < segA_end; ++g) {
                const std::uint8_t byte = tile.packed[g * tile.bytes_per_record + byte_in_rec];
                core::wc_accumulate(core::genotype_code(byte, pos_in_byte), A);
            }
            core::WcPerPop B;
            for (std::size_t g = segB_begin; g < segB_end; ++g) {
                const std::uint8_t byte = tile.packed[g * tile.bytes_per_record + byte_in_rec];
                core::wc_accumulate(core::genotype_code(byte, pos_in_byte), B);
            }

            const core::WcSite r = core::wc_finalize(A, B);
            const std::size_t ss = static_cast<std::size_t>(s);
            out.num[ss] = r.num;
            out.den[ss] = r.den;
            out.fst[ss] = r.fst;
            out.valid[ss] = r.valid ? std::uint8_t{1} : std::uint8_t{0};

            const bool keep = r.valid && (!have_inc || summary_include[ss] != 0);
            if (keep) {
                sum_num += static_cast<long double>(r.num);
                sum_den += static_cast<long double>(r.den);
                ++n_valid;
            }
        }
        out.sum_num = static_cast<double>(sum_num);
        out.sum_den = static_cast<double>(sum_den);
        out.n_valid = n_valid;
        return out;
    }

    // joint_sfs_2pop: the 2D joint site-frequency spectrum parity oracle. A host loop
    // calling the SAME wc_accumulate per-pop fold + sfs_hist.hpp inlines (complete-site
    // test, per-pop fold, mixed-radix cell index) the GPU kernel uses, accumulating into a
    // std::int64_t grid — the unit-test oracle (test-only; not user-facing).
    [[nodiscard]] SfsJoint joint_sfs_2pop(const DecodeTileView& tile, int popA, int popB,
                                          bool folded) override {
        SfsJoint out;
        out.precision_tag = Precision::Kind::Fp64;
        out.folded = folded;

        const long M = static_cast<long>(tile.n_snp);
        const int P = tile.n_pop;
        if (M <= 0 || P <= 0 || popA < 0 || popB < 0 || popA >= P || popB >= P) return out;
        out.n_total = M;

        const std::size_t segA_begin = tile.pop_offsets[static_cast<std::size_t>(popA)];
        const std::size_t segA_end = tile.pop_offsets[static_cast<std::size_t>(popA) + 1];
        const std::size_t segB_begin = tile.pop_offsets[static_cast<std::size_t>(popB)];
        const std::size_t segB_end = tile.pop_offsets[static_cast<std::size_t>(popB) + 1];

        const long NA = static_cast<long>(segA_end - segA_begin);
        const long NB = static_cast<long>(segB_end - segB_begin);
        out.NA = NA;
        out.NB = NB;
        if (NA <= 0 || NB <= 0) return out;

        const long extA = core::sfs_axis_extent(NA, folded);
        const long extB = core::sfs_axis_extent(NB, folded);
        out.extA = extA;
        out.extB = extB;
        out.grid.assign(static_cast<std::size_t>(extA * extB), 0);

        long n_complete = 0;
        for (long s = 0; s < M; ++s) {
            const std::size_t byte_in_rec =
                static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
            const int pos_in_byte = static_cast<int>(s % core::kCodesPerByte);

            core::WcPerPop A;
            for (std::size_t g = segA_begin; g < segA_end; ++g) {
                const std::uint8_t byte = tile.packed[g * tile.bytes_per_record + byte_in_rec];
                core::wc_accumulate(core::genotype_code(byte, pos_in_byte), A);
            }
            core::WcPerPop B;
            for (std::size_t g = segB_begin; g < segB_end; ++g) {
                const std::uint8_t byte = tile.packed[g * tile.bytes_per_record + byte_in_rec];
                core::wc_accumulate(core::genotype_code(byte, pos_in_byte), B);
            }

            if (!core::sfs_site_complete(A.n, NA, B.n, NB)) continue;
            const long idx[2] = {core::sfs_axis_index(A.ac, NA, folded),
                                 core::sfs_axis_index(B.ac, NB, folded)};
            const long ext[2] = {extA, extB};
            const long lin = core::sfs_linear_index(idx, ext, 2);
            out.grid[static_cast<std::size_t>(lin)] += 1;
            ++n_complete;
        }
        out.n_complete = n_complete;
        out.n_dropped_incomplete = out.n_total - out.n_complete;
        return out;
    }

    // detect_sample_ploidy_device — reference §4
    [[nodiscard]] std::vector<int> detect_sample_ploidy_device(
        const DecodeTileView& tile) override {
        return detect_ploidy_host(tile);
    }

    // dstat_block_reduce: per-SNP D reduction — reference §8
    void dstat_block_reduce(const double* Q, const double* V, int P, long M,
                            const int* block_id, int n_block,
                            std::span<const int> quadruples,
                            double* numsum, double* densum, double* cnt) override {
        const int N = static_cast<int>(quadruples.size() / 4);
        if (P <= 0 || M <= 0 || N <= 0 || n_block <= 0) return;

        const std::vector<core::BlockRange> ranges =
            core::block_ranges(std::span<const int>(block_id, static_cast<std::size_t>(M)),
                               M, n_block);

        for (int k = 0; k < N; ++k) {
            const int p1 = quadruples[static_cast<std::size_t>(4 * k + 0)];
            const int p2 = quadruples[static_cast<std::size_t>(4 * k + 1)];
            const int p3 = quadruples[static_cast<std::size_t>(4 * k + 2)];
            const int p4 = quadruples[static_cast<std::size_t>(4 * k + 3)];
            for (int b = 0; b < n_block; ++b) {
                const core::BlockRange& rng = ranges[static_cast<std::size_t>(b)];
                long double nsum = 0.0L, dsum = 0.0L;
                double c = 0.0;
                for (long s = rng.begin; s < rng.end; ++s) {
                    const std::size_t col = static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
                    if (V[col + static_cast<std::size_t>(p1)] == 0.0 ||
                        V[col + static_cast<std::size_t>(p2)] == 0.0 ||
                        V[col + static_cast<std::size_t>(p3)] == 0.0 ||
                        V[col + static_cast<std::size_t>(p4)] == 0.0) {
                        continue;
                    }
                    const double a = Q[col + static_cast<std::size_t>(p1)];
                    const double bb = Q[col + static_cast<std::size_t>(p2)];
                    const double cc = Q[col + static_cast<std::size_t>(p3)];
                    const double dd = Q[col + static_cast<std::size_t>(p4)];
                    nsum += static_cast<long double>((a - bb) * (cc - dd));
                    dsum += static_cast<long double>((a + bb - 2.0 * a * bb) *
                                                     (cc + dd - 2.0 * cc * dd));
                    c += 1.0;
                }
                const std::size_t out = static_cast<std::size_t>(k) *
                                            static_cast<std::size_t>(n_block) +
                                        static_cast<std::size_t>(b);
                numsum[out] = static_cast<double>(nsum);
                densum[out] = static_cast<double>(dsum);
                cnt[out] = c;
            }
        }
    }

    // dates_curve: weighted-LD autocorrelation curve — reference §11
    [[nodiscard]] DatesMoments dates_curve(
        const double* src1_freq, const double* src2_freq, const double* src_valid,
        const std::uint8_t* packed, std::size_t bytes_per_record, int n_target,
        [[maybe_unused]] const int* target_ploidy, const int* grid_cell, long M,
        const int* chrom_first, const int* chrom_last, int n_chrom,
        int numqbins, int n_bin, int diffmax, double binsize, int qbin,
        const Precision& precision) override {
        (void)precision;
        DatesMoments out;
        out.n_chrom = n_chrom;
        out.n_bin = n_bin;
        if (n_chrom <= 0 || n_bin <= 0 || M <= 0 || n_target <= 0 || numqbins <= 0) {
            out.status = Status::Ok;
            return out;
        }
        const std::size_t cb = static_cast<std::size_t>(n_chrom) * static_cast<std::size_t>(n_bin);
        out.s0.assign(cb, 0.0);  out.s1.assign(cb, 0.0);  out.s2.assign(cb, 0.0);
        out.s11.assign(cb, 0.0); out.s12.assign(cb, 0.0); out.s22.assign(cb, 0.0);

        const std::size_t nq = static_cast<std::size_t>(numqbins);
        const std::size_t dm = static_cast<std::size_t>(diffmax) + 1;
        std::vector<double> dd00(static_cast<std::size_t>(n_chrom) * dm, 0.0);
        std::vector<double> dd11(static_cast<std::size_t>(n_chrom) * dm, 0.0);
        std::vector<double> dd01(static_cast<std::size_t>(n_chrom) * dm, 0.0);
        std::vector<double> dd10(static_cast<std::size_t>(n_chrom) * dm, 0.0);
        std::vector<double> dd02(static_cast<std::size_t>(n_chrom) * dm, 0.0);
        std::vector<double> dd20(static_cast<std::size_t>(n_chrom) * dm, 0.0);

        std::vector<double> z0q(nq), z1q(nq), z2q(nq);
        std::vector<double> w0(static_cast<std::size_t>(M)), wt(static_cast<std::size_t>(M));
        std::vector<long> xindex(static_cast<std::size_t>(M));

        for (int i = 0; i < n_target; ++i) {
            const std::uint8_t* rec = packed + static_cast<std::size_t>(i) * bytes_per_record;
            long numx = 0;
            long double dot12 = 0.0L, dot22 = 0.0L;
            for (long s = 0; s < M; ++s) {
                if (src_valid[s] == 0.0) continue;
                const std::size_t byte_in_rec =
                    static_cast<std::size_t>(s) / static_cast<std::size_t>(core::kCodesPerByte);
                const int pos = static_cast<int>(s % core::kCodesPerByte);
                const std::uint8_t code = core::genotype_code(rec[byte_in_rec], pos);
                if (code == core::kMissingGenotypeCode) continue;
                const double g = static_cast<double>(code);
                const double w1 = src1_freq[s];
                const double w2v = src2_freq[s];
                const double w0v = g / 2.0;
                w0[static_cast<std::size_t>(numx)] = w0v;
                wt[static_cast<std::size_t>(numx)] = w1 - w2v;
                const double a = w0v - w2v;
                const double b = w1 - w2v;
                dot12 += static_cast<long double>(a) * static_cast<long double>(b);
                dot22 += static_cast<long double>(b) * static_cast<long double>(b);
                xindex[static_cast<std::size_t>(numx)] = s;
                ++numx;
            }
            if (numx == 0) continue;
            const double yreg = (dot22 != 0.0L)
                                    ? static_cast<double>(dot12 / dot22)
                                    : 0.0;

            std::fill(z0q.begin(), z0q.end(), 0.0);
            std::fill(z1q.begin(), z1q.end(), 0.0);
            std::fill(z2q.begin(), z2q.end(), 0.0);
            for (long k1 = 0; k1 < numx; ++k1) {
                const long s = xindex[static_cast<std::size_t>(k1)];
                const double w1 = src1_freq[s];
                const double w2v = src2_freq[s];
                const double pred = yreg * w1 + (1.0 - yreg) * w2v;
                const double r = w0[static_cast<std::size_t>(k1)] - pred;
                const double y = r * wt[static_cast<std::size_t>(k1)];
                const int cell = grid_cell[s];
                const std::size_t cc = static_cast<std::size_t>(cell);
                z0q[cc] += 1.0;
                z1q[cc] += y;
                z2q[cc] += y * y;
            }

            for (int kc = 0; kc < n_chrom; ++kc) {
                const int slo = chrom_first[kc];
                const int shi = chrom_last[kc];
                if (slo < 0 || shi < slo) continue;
                const int len = shi - slo + 1;
                if (len <= 1) continue;
                double* a00 = dd00.data() + static_cast<std::size_t>(kc) * dm;
                double* a11 = dd11.data() + static_cast<std::size_t>(kc) * dm;
                double* a01 = dd01.data() + static_cast<std::size_t>(kc) * dm;
                double* a10 = dd10.data() + static_cast<std::size_t>(kc) * dm;
                double* a02 = dd02.data() + static_cast<std::size_t>(kc) * dm;
                double* a20 = dd20.data() + static_cast<std::size_t>(kc) * dm;
                const double* p0 = z0q.data() + static_cast<std::size_t>(slo);
                const double* p1 = z1q.data() + static_cast<std::size_t>(slo);
                const double* p2 = z2q.data() + static_cast<std::size_t>(slo);
                const int lmax = (diffmax < len - 1) ? diffmax : len - 1;
                for (int lag = 0; lag <= lmax; ++lag) {
                    long double c00 = 0.0L, c11 = 0.0L, c01 = 0.0L, c10 = 0.0L,
                                c02 = 0.0L, c20 = 0.0L;
                    const int n_pair = len - lag;
                    for (int g = 0; g < n_pair; ++g) {
                        const double z0a = p0[g], z0b = p0[g + lag];
                        const double z1a = p1[g], z1b = p1[g + lag];
                        const double z2a = p2[g], z2b = p2[g + lag];
                        c00 += static_cast<long double>(z0a) * z0b;
                        c11 += static_cast<long double>(z1a) * z1b;
                        c01 += static_cast<long double>(z0a) * z1b;
                        c10 += static_cast<long double>(z1a) * z0b;
                        c02 += static_cast<long double>(z0a) * z2b;
                        c20 += static_cast<long double>(z2a) * z0b;
                    }
                    const std::size_t li = static_cast<std::size_t>(lag);
                    a00[li] += static_cast<double>(c00);
                    a11[li] += static_cast<double>(c11);
                    a01[li] += static_cast<double>(c01);
                    a10[li] += static_cast<double>(c10);
                    a02[li] += static_cast<double>(c02);
                    a20[li] += static_cast<double>(c20);
                }
            }
        }

        const double dbinsize = binsize / static_cast<double>(qbin);
        for (int kc = 0; kc < n_chrom; ++kc) {
            const double* a00 = dd00.data() + static_cast<std::size_t>(kc) * dm;
            const double* a11 = dd11.data() + static_cast<std::size_t>(kc) * dm;
            const double* a01 = dd01.data() + static_cast<std::size_t>(kc) * dm;
            const double* a10 = dd10.data() + static_cast<std::size_t>(kc) * dm;
            const double* a02 = dd02.data() + static_cast<std::size_t>(kc) * dm;
            const double* a20 = dd20.data() + static_cast<std::size_t>(kc) * dm;
            for (int d = 1; d <= diffmax; ++d) {
                if (a00[static_cast<std::size_t>(d)] < 0.5) continue;
                const double ys = static_cast<double>(d) * dbinsize;
                const int s = static_cast<int>(ys / binsize);
                if (s < 0 || s >= n_bin) continue;
                const std::size_t o = static_cast<std::size_t>(kc) * static_cast<std::size_t>(n_bin) +
                                      static_cast<std::size_t>(s);
                out.s0[o]  += a00[static_cast<std::size_t>(d)];
                out.s1[o]  += a01[static_cast<std::size_t>(d)];
                out.s2[o]  += a10[static_cast<std::size_t>(d)];
                out.s12[o] += a11[static_cast<std::size_t>(d)];
                out.s11[o] += a02[static_cast<std::size_t>(d)];
                out.s22[o] += a20[static_cast<std::size_t>(d)];
            }
        }
        out.status = Status::Ok;
        return out;
    }

    // dates_repack: target-genotype repack — reference §11
    void dates_repack(const std::uint8_t* src, std::size_t src_bpr, const long* kept_src,
                      long M_kept, int n_target, std::size_t dst_bpr, std::uint8_t* dst) override {
        core::dates::dates_repack_host(src, src_bpr, kept_src, M_kept, n_target, dst_bpr, dst);
    }

    // dates_fit: exponential-decay fit — reference §11
    [[nodiscard]] std::vector<DatesExpFit> dates_fit(const double* curves, int win_len,
                                                     int n_curves, double step,
                                                     bool affine) override {
        std::vector<DatesExpFit> out(static_cast<std::size_t>(n_curves > 0 ? n_curves : 0));
        for (int c = 0; c < n_curves; ++c) {
            std::vector<double> y(static_cast<std::size_t>(win_len > 0 ? win_len : 0));
            for (int j = 0; j < win_len; ++j)
                y[static_cast<std::size_t>(j)] =
                    curves[static_cast<std::size_t>(c) * static_cast<std::size_t>(win_len) +
                           static_cast<std::size_t>(j)];
            const core::dates::ExpFitHost f = core::dates::fit_exp_decay(y, step, affine);
            DatesExpFit& o = out[static_cast<std::size_t>(c)];
            o.date_gen = f.date_gen;
            o.error_sd = f.error_sd;
            o.ok = f.ok ? 1 : 0;
        }
        return out;
    }

    // ls_forward_backward: the Li-Stephens copying forward-backward REFERENCE oracle
    // (the `steppe paint` FB core, Phase 0). Exact, per-column RESCALED, scalar,
    // native FP64 — the FB is a product of sub-one probabilities that underflows over
    // M columns, so alpha and beta are renormalized every column (kalis's convention,
    // not log-space; §2e-3). The copying posterior gamma_l(k) = alpha*beta normalized
    // per column is invariant to those per-column scalings, so it matches kalis's exact
    // posterior regardless of rescale bookkeeping. This is the diff oracle the GPU
    // forward-backward (Phase 1) is gated against; it is dev/test-only, never a runtime.
    // Reference: docs/planning/li-stephens-engine-scope.md §2a.
    [[nodiscard]] LsPosterior ls_forward_backward(const std::uint8_t* recipient,
                                                  const std::uint8_t* donors, const double* pi,
                                                  const double* rho, const double* mu, int K,
                                                  long M, const Precision& precision) override {
        (void)precision;  // the reference runs in native FP64 by construction
        LsPosterior out;
        out.K = K;
        out.M = M;
        if (K <= 0 || M <= 0) { out.status = Status::Ok; return out; }

        const std::size_t Ks = static_cast<std::size_t>(K);
        const std::size_t Ms = static_cast<std::size_t>(M);

        // donors are donor-major (K rows of M); recipient is M alleles. An allele byte
        // > 1 is treated as missing -> an uninformative (constant 1) emission column.
        const auto emission = [&](long l, int k) -> double {
            const std::uint8_t r = recipient[static_cast<std::size_t>(l)];
            const std::uint8_t d =
                donors[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
            if (r > 1u || d > 1u) return 1.0;  // missing: no information
            const double m = mu[static_cast<std::size_t>(l)];
            return (r == d) ? (1.0 - m) : m;
        };

        // alpha / beta are stored per-column normalized (K x M, donor-major).
        std::vector<double> alpha(Ks * Ms, 0.0);
        std::vector<double> beta(Ks * Ms, 0.0);

        // --- Forward sweep (left to right), rescaled each column ---------------
        // l = 0: alpha_0(k) = pi[k] * e_0(k).
        {
            long double s = 0.0L;
            for (int k = 0; k < K; ++k) {
                const double a = pi[static_cast<std::size_t>(k)] * emission(0, k);
                alpha[static_cast<std::size_t>(k) * Ms] = a;
                s += static_cast<long double>(a);
            }
            const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
            for (int k = 0; k < K; ++k) alpha[static_cast<std::size_t>(k) * Ms] *= inv;
        }
        for (long l = 1; l < M; ++l) {
            // alpha_{l-1} is already normalized (sum == 1); still form the reduction in
            // native FP64 so the collapsed rank-1 transition is exact.
            long double prev_sum = 0.0L;
            for (int k = 0; k < K; ++k)
                prev_sum += static_cast<long double>(
                    alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l - 1)]);
            const double r = rho[static_cast<std::size_t>(l)];
            long double s = 0.0L;
            for (int k = 0; k < K; ++k) {
                const double aprev =
                    alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l - 1)];
                const double trans = (1.0 - r) * aprev +
                                     r * pi[static_cast<std::size_t>(k)] *
                                         static_cast<double>(prev_sum);
                const double a = emission(l, k) * trans;
                alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] = a;
                s += static_cast<long double>(a);
            }
            const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
            for (int k = 0; k < K; ++k)
                alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] *= inv;
        }

        // --- Backward sweep (right to left), rescaled each column --------------
        // l = M-1: beta = 1.
        for (int k = 0; k < K; ++k)
            beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(M - 1)] = 1.0;
        for (long l = M - 2; l >= 0; --l) {
            const double r = rho[static_cast<std::size_t>(l + 1)];
            // The shared rank-1 term T = sum_k pi[k] * e_{l+1}(k) * beta_{l+1}(k).
            long double T = 0.0L;
            for (int k = 0; k < K; ++k) {
                const double b =
                    beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l + 1)];
                T += static_cast<long double>(pi[static_cast<std::size_t>(k)]) *
                     static_cast<long double>(emission(l + 1, k)) * static_cast<long double>(b);
            }
            long double s = 0.0L;
            for (int k = 0; k < K; ++k) {
                const double bnext =
                    beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l + 1)];
                const double b = (1.0 - r) * emission(l + 1, k) * bnext +
                                 r * static_cast<double>(T);
                beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] = b;
                s += static_cast<long double>(b);
            }
            const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
            for (int k = 0; k < K; ++k)
                beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] *= inv;
        }

        // --- Posterior gamma_l(k) = alpha*beta normalized per column ----------
        out.gamma.assign(Ks * Ms, 0.0);
        for (long l = 0; l < M; ++l) {
            long double denom = 0.0L;
            for (int k = 0; k < K; ++k) {
                const std::size_t o = static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                denom += static_cast<long double>(alpha[o]) * static_cast<long double>(beta[o]);
            }
            const double inv = (denom > 0.0L) ? static_cast<double>(1.0L / denom) : 0.0;
            for (int k = 0; k < K; ++k) {
                const std::size_t o = static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                out.gamma[o] = alpha[o] * beta[o] * inv;
            }
        }
        out.status = Status::Ok;
        return out;
    }

    // ls_paint_coancestry: the Li-Stephens ChromoPainter coancestry REFERENCE oracle
    // (the `steppe paint` FACE, Phase 2) — the diff oracle the fused GPU coancestry sink
    // is gated against. For each recipient it runs the SAME per-column-rescaled native-FP64
    // forward-backward as ls_forward_backward, keeping the normalized forward a_l resident,
    // and folds gamma into the two per-donor accumulators by the §1 formulas:
    //   chunklength_k = sum_l gamma_l(k)*w_l
    //   chunkcount_k  = gamma_0(k) + sum_{l>=1} gamma_l(k)*rho_l*pi_k / ((1-rho_l)*a_{l-1}(k)+rho_l*pi_k)
    // The switch denominator is guarded (contribute 0 when it is <= 0) so the self donor
    // under leave-one-out (pi_k=0 => a=0, gamma=0 => 0/0) and rho_l=0 zero-mass donors do
    // not produce NaN — the identical guard the gamma path uses (cpu:649). The coancestry
    // sums accumulate in NATIVE double (not long double) so they match the GPU sink by
    // construction (§2c reduction carve-out; the FB internals still reduce in long double,
    // exactly as ls_forward_backward). Reference: li-stephens-phase2-paint-face-spec §1,§2.
    [[nodiscard]] LsCoancestry ls_paint_coancestry(
        const std::uint8_t* recipients, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, const double* w, int K, long M, int N,
        const Precision& precision) override {
        (void)precision;  // native FP64 by construction (§2c)
        LsCoancestry out;
        out.K = K;
        out.N = N;
        if (K <= 0 || M <= 0 || N <= 0) { out.status = Status::Ok; return out; }

        const std::size_t Ks = static_cast<std::size_t>(K);
        const std::size_t Ms = static_cast<std::size_t>(M);
        out.chunkcounts.assign(static_cast<std::size_t>(N) * Ks, 0.0);
        out.chunklengths.assign(static_cast<std::size_t>(N) * Ks, 0.0);

        std::vector<double> alpha(Ks * Ms, 0.0);  // normalized forward, K x M donor-major
        std::vector<double> beta(Ks * Ms, 0.0);   // normalized backward

        for (long r = 0; r < N; ++r) {
            const std::uint8_t* recipient = recipients + static_cast<std::size_t>(r) * Ms;
            const double* pir = pi + static_cast<std::size_t>(r) * Ks;

            const auto emission = [&](long l, int k) -> double {
                const std::uint8_t rr = recipient[static_cast<std::size_t>(l)];
                const std::uint8_t d =
                    donors[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
                if (rr > 1u || d > 1u) return 1.0;
                const double m = mu[static_cast<std::size_t>(l)];
                return (rr == d) ? (1.0 - m) : m;
            };

            // --- Forward sweep (identical to ls_forward_backward) ---------------
            {
                long double s = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double a = pir[static_cast<std::size_t>(k)] * emission(0, k);
                    alpha[static_cast<std::size_t>(k) * Ms] = a;
                    s += static_cast<long double>(a);
                }
                const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
                for (int k = 0; k < K; ++k) alpha[static_cast<std::size_t>(k) * Ms] *= inv;
            }
            for (long l = 1; l < M; ++l) {
                long double prev_sum = 0.0L;
                for (int k = 0; k < K; ++k)
                    prev_sum += static_cast<long double>(
                        alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l - 1)]);
                const double rl = rho[static_cast<std::size_t>(l)];
                long double s = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double aprev =
                        alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l - 1)];
                    const double trans = (1.0 - rl) * aprev +
                                         rl * pir[static_cast<std::size_t>(k)] *
                                             static_cast<double>(prev_sum);
                    const double a = emission(l, k) * trans;
                    alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] = a;
                    s += static_cast<long double>(a);
                }
                const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
                for (int k = 0; k < K; ++k)
                    alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] *= inv;
            }

            // --- Backward sweep -------------------------------------------------
            for (int k = 0; k < K; ++k)
                beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(M - 1)] = 1.0;
            for (long l = M - 2; l >= 0; --l) {
                const double rl = rho[static_cast<std::size_t>(l + 1)];
                long double T = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double b =
                        beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l + 1)];
                    T += static_cast<long double>(pir[static_cast<std::size_t>(k)]) *
                         static_cast<long double>(emission(l + 1, k)) * static_cast<long double>(b);
                }
                long double s = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double bnext =
                        beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l + 1)];
                    const double b = (1.0 - rl) * emission(l + 1, k) * bnext +
                                     rl * static_cast<double>(T);
                    beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] = b;
                    s += static_cast<long double>(b);
                }
                const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
                for (int k = 0; k < K; ++k)
                    beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] *= inv;
            }

            // --- Fold gamma into the two coancestry accumulators (native FP64) --
            double* cnt = out.chunkcounts.data() + static_cast<std::size_t>(r) * Ks;
            double* len = out.chunklengths.data() + static_cast<std::size_t>(r) * Ks;
            for (long l = 0; l < M; ++l) {
                long double denom = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const std::size_t o =
                        static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                    denom += static_cast<long double>(alpha[o]) * static_cast<long double>(beta[o]);
                }
                const double ginv = (denom > 0.0L) ? static_cast<double>(1.0L / denom) : 0.0;
                const double rl = rho[static_cast<std::size_t>(l)];
                const double wl = w[static_cast<std::size_t>(l)];
                for (int k = 0; k < K; ++k) {
                    const std::size_t o =
                        static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                    const double g = alpha[o] * beta[o] * ginv;  // gamma_l(k)
                    len[static_cast<std::size_t>(k)] += g * wl;
                    double sw;
                    if (l == 0) {
                        sw = g;  // the initial chunk
                    } else {
                        const double aprev =
                            alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l - 1)];
                        const double pk = pir[static_cast<std::size_t>(k)];
                        const double den = (1.0 - rl) * aprev + rl * pk;
                        sw = (den > 0.0) ? g * rl * pk / den : 0.0;  // guard: pi_k=0/rho=0 -> 0
                    }
                    cnt[static_cast<std::size_t>(k)] += sw;
                }
            }
        }
        out.status = Status::Ok;
        return out;
    }

    // ls_localanc: the Li-Stephens LOCAL-ANCESTRY REFERENCE oracle (the `steppe paint
    // --face localanc` output, Phase 3) — the diff oracle the fused GPU local-ancestry
    // sink is gated against. For each recipient it runs the SAME per-column-rescaled
    // native-FP64 forward-backward as ls_paint_coancestry, then keeps the SNP axis and
    // folds gamma_l(k) into the per-SNP per-label posterior:
    //   post[(r*M + l)*P + g] = sum_{k : donor_group[k]==g} gamma_l(k)
    // The per-column gamma denominator is guarded (ginv=0 when denom underflows) exactly
    // as the gamma path (cpu:649), so a degenerate all-missing column has post_l == 0
    // rather than NaN. The per-label sums accumulate in NATIVE double (matching the GPU
    // reduction; the FB internals still reduce in long double). No switch term and no
    // genetic weight — localanc is the per-position marginal. dev/test only, never a
    // runtime. Reference: li-stephens-phase3-localanc-face-spec §5.
    [[nodiscard]] LsLocalAncestry ls_localanc(
        const std::uint8_t* recipients, const std::uint8_t* donors, const double* pi,
        const double* rho, const double* mu, const int* donor_group, int K, long M, int N,
        int P, const Precision& precision) override {
        (void)precision;  // native FP64 by construction (§2c)
        LsLocalAncestry out;
        out.P = P;
        out.M = M;
        out.N = N;
        if (K <= 0 || M <= 0 || N <= 0 || P <= 0) { out.status = Status::Ok; return out; }

        const std::size_t Ks = static_cast<std::size_t>(K);
        const std::size_t Ms = static_cast<std::size_t>(M);
        const std::size_t Ps = static_cast<std::size_t>(P);
        out.post.assign(static_cast<std::size_t>(N) * Ms * Ps, 0.0);

        std::vector<double> alpha(Ks * Ms, 0.0);  // normalized forward, K x M donor-major
        std::vector<double> beta(Ks * Ms, 0.0);   // normalized backward

        for (long r = 0; r < N; ++r) {
            const std::uint8_t* recipient = recipients + static_cast<std::size_t>(r) * Ms;
            const double* pir = pi + static_cast<std::size_t>(r) * Ks;

            const auto emission = [&](long l, int k) -> double {
                const std::uint8_t rr = recipient[static_cast<std::size_t>(l)];
                const std::uint8_t d =
                    donors[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)];
                if (rr > 1u || d > 1u) return 1.0;
                const double m = mu[static_cast<std::size_t>(l)];
                return (rr == d) ? (1.0 - m) : m;
            };

            // --- Forward sweep (identical to ls_forward_backward) ---------------
            {
                long double s = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double a = pir[static_cast<std::size_t>(k)] * emission(0, k);
                    alpha[static_cast<std::size_t>(k) * Ms] = a;
                    s += static_cast<long double>(a);
                }
                const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
                for (int k = 0; k < K; ++k) alpha[static_cast<std::size_t>(k) * Ms] *= inv;
            }
            for (long l = 1; l < M; ++l) {
                long double prev_sum = 0.0L;
                for (int k = 0; k < K; ++k)
                    prev_sum += static_cast<long double>(
                        alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l - 1)]);
                const double rl = rho[static_cast<std::size_t>(l)];
                long double s = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double aprev =
                        alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l - 1)];
                    const double trans = (1.0 - rl) * aprev +
                                         rl * pir[static_cast<std::size_t>(k)] *
                                             static_cast<double>(prev_sum);
                    const double a = emission(l, k) * trans;
                    alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] = a;
                    s += static_cast<long double>(a);
                }
                const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
                for (int k = 0; k < K; ++k)
                    alpha[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] *= inv;
            }

            // --- Backward sweep -------------------------------------------------
            for (int k = 0; k < K; ++k)
                beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(M - 1)] = 1.0;
            for (long l = M - 2; l >= 0; --l) {
                const double rl = rho[static_cast<std::size_t>(l + 1)];
                long double T = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double b =
                        beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l + 1)];
                    T += static_cast<long double>(pir[static_cast<std::size_t>(k)]) *
                         static_cast<long double>(emission(l + 1, k)) * static_cast<long double>(b);
                }
                long double s = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const double bnext =
                        beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l + 1)];
                    const double b = (1.0 - rl) * emission(l + 1, k) * bnext +
                                     rl * static_cast<double>(T);
                    beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] = b;
                    s += static_cast<long double>(b);
                }
                const double inv = (s > 0.0L) ? static_cast<double>(1.0L / s) : 0.0;
                for (int k = 0; k < K; ++k)
                    beta[static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l)] *= inv;
            }

            // --- Fold gamma into the per-SNP per-label posterior (native FP64) --
            double* post_r = out.post.data() + static_cast<std::size_t>(r) * Ms * Ps;
            for (long l = 0; l < M; ++l) {
                long double denom = 0.0L;
                for (int k = 0; k < K; ++k) {
                    const std::size_t o =
                        static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                    denom += static_cast<long double>(alpha[o]) * static_cast<long double>(beta[o]);
                }
                const double ginv = (denom > 0.0L) ? static_cast<double>(1.0L / denom) : 0.0;
                double* post_l = post_r + static_cast<std::size_t>(l) * Ps;
                for (int k = 0; k < K; ++k) {
                    const std::size_t o =
                        static_cast<std::size_t>(k) * Ms + static_cast<std::size_t>(l);
                    const double g = alpha[o] * beta[o] * ginv;  // gamma_l(k)
                    post_l[static_cast<std::size_t>(donor_group[static_cast<std::size_t>(k)])] += g;
                }
            }
        }
        out.status = Status::Ok;
        return out;
    }

    // qpfstats_smooth: joint-f2 smoothing solve — reference §12
    [[nodiscard]] QpfstatsSmooth qpfstats_smooth(std::span<const double> x,
                                                 std::span<const double> ymat,
                                                 std::span<const double> y,
                                                 int npopcomb, int npairs,
                                                 int n_block, double ridge,
                                                 const Precision& precision) override {
        (void)precision;
        QpfstatsSmooth out;
        out.npairs = npairs;
        out.n_block = n_block;
        if (npopcomb <= 0 || npairs <= 0 || n_block <= 0) { out.status = Status::Ok; return out; }

        const std::size_t nc = static_cast<std::size_t>(npopcomb);
        const std::size_t np = static_cast<std::size_t>(npairs);

        const auto xat = [&](int c, int p) -> double {
            return x[static_cast<std::size_t>(c) + nc * static_cast<std::size_t>(p)];
        };

        std::vector<double> A_shared(np * np, 0.0);
        for (int i = 0; i < npairs; ++i) {
            for (int j = i; j < npairs; ++j) {
                long double s = 0.0L;
                for (int c = 0; c < npopcomb; ++c)
                    s += static_cast<long double>(xat(c, i)) * static_cast<long double>(xat(c, j));
                double v = static_cast<double>(s);
                if (i == j) v += ridge;
                A_shared[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] = v;
                A_shared[static_cast<std::size_t>(j) + np * static_cast<std::size_t>(i)] = v;
            }
        }


        out.b.assign(np * static_cast<std::size_t>(n_block), 0.0);
        std::vector<int> nan_rows;
        std::vector<double> rhs(np, 0.0);
        std::vector<double> A_blk(np * np, 0.0);
        std::vector<double> bcol;
        for (int blk = 0; blk < n_block; ++blk) {
            nan_rows.clear();
            std::fill(rhs.begin(), rhs.end(), 0.0);
            for (int c = 0; c < npopcomb; ++c) {
                const double yv = ymat[static_cast<std::size_t>(c) +
                                       nc * static_cast<std::size_t>(blk)];
                if (!std::isfinite(yv)) { nan_rows.push_back(c); continue; }
                for (int p = 0; p < npairs; ++p)
                    rhs[static_cast<std::size_t>(p)] += xat(c, p) * yv;
            }
            if (static_cast<int>(nan_rows.size()) == npopcomb) continue;

            const double* Aptr = A_shared.data();
            if (!nan_rows.empty()) {
                A_blk = A_shared;
                for (int i = 0; i < npairs; ++i) {
                    for (int j = i; j < npairs; ++j) {
                        long double s = 0.0L;
                        for (int c : nan_rows)
                            s += static_cast<long double>(xat(c, i)) *
                                 static_cast<long double>(xat(c, j));
                        const double d = static_cast<double>(s);
                        A_blk[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] -= d;
                        if (i != j)
                            A_blk[static_cast<std::size_t>(j) + np * static_cast<std::size_t>(i)] -= d;
                    }
                }
                Aptr = A_blk.data();
            }
            const std::vector<double> Acopy(Aptr, Aptr + np * np);
            const core::LinAlgStatus st = core::solve(Acopy, npairs, rhs, bcol);
            if (!st.ok) { out.status = Status::NonSpdCovariance; return out; }
            for (int p = 0; p < npairs; ++p)
                out.b[static_cast<std::size_t>(p) +
                      np * static_cast<std::size_t>(blk)] = bcol[static_cast<std::size_t>(p)];
        }

        out.bglob.assign(np, 0.0);
        {
            nan_rows.clear();
            std::fill(rhs.begin(), rhs.end(), 0.0);
            for (int c = 0; c < npopcomb; ++c) {
                const double yv = y[static_cast<std::size_t>(c)];
                if (!std::isfinite(yv)) { nan_rows.push_back(c); continue; }
                for (int p = 0; p < npairs; ++p)
                    rhs[static_cast<std::size_t>(p)] += xat(c, p) * yv;
            }
            if (static_cast<int>(nan_rows.size()) < npopcomb) {
                std::vector<double> Ay = A_shared;
                if (!nan_rows.empty()) {
                    for (int i = 0; i < npairs; ++i)
                        for (int j = i; j < npairs; ++j) {
                            long double s = 0.0L;
                            for (int c : nan_rows)
                                s += static_cast<long double>(xat(c, i)) *
                                     static_cast<long double>(xat(c, j));
                            const double d = static_cast<double>(s);
                            Ay[static_cast<std::size_t>(i) + np * static_cast<std::size_t>(j)] -= d;
                            if (i != j)
                                Ay[static_cast<std::size_t>(j) + np * static_cast<std::size_t>(i)] -= d;
                        }
                }
                const core::LinAlgStatus st = core::solve(Ay, npairs, rhs, bcol);
                if (!st.ok) { out.status = Status::NonSpdCovariance; return out; }
                out.bglob = bcol;
            }
        }

        out.status = Status::Ok;
        return out;
    }

    // qpfstats_blocks_smooth: fused reduce→smooth — reference §12
    [[nodiscard]] QpfstatsSmooth qpfstats_blocks_smooth(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples, std::span<const double> x, int npopcomb, int npairs,
        std::span<const int> block_sizes, double ridge, const Precision& precision) override {
        QpfstatsSmooth out;
        out.npairs = npairs;
        out.n_block = n_block;
        if (npopcomb <= 0 || npairs <= 0 || n_block <= 0) { out.status = Status::Ok; return out; }

        const std::size_t nc = static_cast<std::size_t>(npopcomb);
        const std::size_t np = static_cast<std::size_t>(npairs);
        const std::size_t nbb = static_cast<std::size_t>(n_block);
        const std::size_t nb_out = nc * nbb;

        std::vector<double> numsum(nb_out, 0.0), densum(nb_out, 0.0), cnt(nb_out, 0.0);
        dstat_block_reduce(Q, V, P, M, block_id, n_block, quadruples,
                           numsum.data(), densum.data(), cnt.data());

        std::vector<double> numer_rm(nb_out, 0.0), ymat(nb_out, 0.0);
        std::vector<double> y(nc, 0.0);
        for (int c = 0; c < npopcomb; ++c) {
            const std::size_t base = static_cast<std::size_t>(c) * nbb;
            for (int b = 0; b < n_block; ++b) {
                const double cn = cnt[base + static_cast<std::size_t>(b)];
                const double mean = (cn > 0.0)
                    ? (numsum[base + static_cast<std::size_t>(b)] / cn) : std::nan("");
                numer_rm[base + static_cast<std::size_t>(b)] = mean;
                ymat[static_cast<std::size_t>(c) + nc * static_cast<std::size_t>(b)] = mean;
            }
            y[static_cast<std::size_t>(c)] =
                core::matrix_jackknife_est_col(numer_rm.data(), cnt.data(), c, n_block);
        }

        QpfstatsSmooth sm = qpfstats_smooth(x, std::span<const double>(ymat),
                                            std::span<const double>(y), npopcomb, npairs,
                                            n_block, ridge, precision);
        if (sm.status != Status::Ok) { out.status = sm.status; return out; }
        out.b = std::move(sm.b);
        out.bglob = std::move(sm.bglob);

        out.recenter_shift.assign(np, 0.0);
        std::vector<int> bl(block_sizes.begin(), block_sizes.end());
        std::vector<double> series(nbb, 0.0);
        for (int p = 0; p < npairs; ++p) {
            for (int b = 0; b < n_block; ++b)
                series[static_cast<std::size_t>(b)] =
                    out.b[static_cast<std::size_t>(p) + np * static_cast<std::size_t>(b)];
            const double est = core::f2blocks_pair_est(series, bl);
            out.recenter_shift[static_cast<std::size_t>(p)] =
                out.bglob[static_cast<std::size_t>(p)] - est;
        }

        out.status = Status::Ok;
        return out;
    }


    // assemble_f4: qpAdm four-slab f4 assembly — reference §6
    [[nodiscard]] F4Blocks assemble_f4(const F2BlockTensor& f2,
                                       std::span<const int> left_idx,
                                       std::span<const int> right_idx,
                                       const Precision& precision) override {
        (void)precision;

        const int nl = static_cast<int>(left_idx.size()) - 1;
        const int nr = static_cast<int>(right_idx.size()) - 1;
        const int nb = f2.n_block;
        const int P = f2.P;
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

        F4Blocks out;
        out.nl = nl;
        out.nr = nr;
        const std::size_t m = static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr);
        if (nl <= 0 || nr <= 0 || nb <= 0) { out.n_block = 0; return out; }

        const std::vector<int> surv = survivor_blocks(f2.vpair, P, nb, f2.block_sizes);
        const int nb_s = static_cast<int>(surv.size());
        out.n_block = nb_s;
        out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
        for (int bs = 0; bs < nb_s; ++bs)
            out.block_sizes[static_cast<std::size_t>(bs)] =
                f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
        out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        if (nb_s <= 0) return out;

        const int L0 = left_idx[0];
        const int R0 = right_idx[0];
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
                for (int bs = 0; bs < nb_s; ++bs) {
                    const int b = surv[static_cast<std::size_t>(bs)];
                    const double x = 0.5 * (f2at(Li, R0, b) + f2at(L0, Rj, b) -
                                            f2at(L0, R0, b) - f2at(Li, Rj, b));
                    out.x_blocks[k + m * static_cast<std::size_t>(bs)] = x;
                }
            }
        }

        compute_loo_and_total(out, out.block_sizes);
        return out;
    }

    // assemble_f4_quartets: batched f4 — reference §6
    [[nodiscard]] F4Blocks assemble_f4_quartets(const F2BlockTensor& f2,
                                                std::span<const int> quartets,
                                                const Precision& precision) override {
        (void)precision;

        const int N = static_cast<int>(quartets.size()) / 4;
        const int nb = f2.n_block;
        const int P = f2.P;
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

        F4Blocks out;
        out.nl = N;
        out.nr = 1;
        const std::size_t m = static_cast<std::size_t>(N);
        if (N <= 0 || nb <= 0) { out.n_block = 0; return out; }

        const std::vector<int> surv = survivor_blocks(f2.vpair, P, nb, f2.block_sizes);
        const int nb_s = static_cast<int>(surv.size());
        out.n_block = nb_s;
        out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
        for (int bs = 0; bs < nb_s; ++bs)
            out.block_sizes[static_cast<std::size_t>(bs)] =
                f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
        out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        if (nb_s <= 0) return out;

        const auto f2at = [&](int i, int j, int b) -> double {
            return f2.f2[static_cast<std::size_t>(i) +
                         static_cast<std::size_t>(P) * static_cast<std::size_t>(j) +
                         slab * static_cast<std::size_t>(b)];
        };
        for (int k = 0; k < N; ++k) {
            const int p1 = quartets[static_cast<std::size_t>(4 * k) + 0];
            const int p2 = quartets[static_cast<std::size_t>(4 * k) + 1];
            const int p3 = quartets[static_cast<std::size_t>(4 * k) + 2];
            const int p4 = quartets[static_cast<std::size_t>(4 * k) + 3];
            for (int bs = 0; bs < nb_s; ++bs) {
                const int b = surv[static_cast<std::size_t>(bs)];
                const double x = 0.5 * (f2at(p2, p3, b) + f2at(p1, p4, b) -
                                        f2at(p1, p3, b) - f2at(p2, p4, b));
                out.x_blocks[static_cast<std::size_t>(k) +
                             m * static_cast<std::size_t>(bs)] = x;
            }
        }

        compute_loo_and_total(out, out.block_sizes);
        return out;
    }

    // assemble_f3_triples: batched f3 — reference §6
    [[nodiscard]] F4Blocks assemble_f3_triples(const F2BlockTensor& f2,
                                               std::span<const int> triples,
                                               const Precision& precision) override {
        (void)precision;

        const int N = static_cast<int>(triples.size()) / 3;
        const int nb = f2.n_block;
        const int P = f2.P;
        const std::size_t slab = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);

        F4Blocks out;
        out.nl = N;
        out.nr = 1;
        const std::size_t m = static_cast<std::size_t>(N);
        if (N <= 0 || nb <= 0) { out.n_block = 0; return out; }

        const std::vector<int> surv = survivor_blocks(f2.vpair, P, nb, f2.block_sizes);
        const int nb_s = static_cast<int>(surv.size());
        out.n_block = nb_s;
        out.block_sizes.assign(static_cast<std::size_t>(nb_s), 0);
        for (int bs = 0; bs < nb_s; ++bs)
            out.block_sizes[static_cast<std::size_t>(bs)] =
                f2.block_sizes[static_cast<std::size_t>(surv[static_cast<std::size_t>(bs)])];
        out.x_blocks.assign(m * static_cast<std::size_t>(nb_s), 0.0);
        if (nb_s <= 0) return out;

        const auto f2at = [&](int i, int j, int b) -> double {
            return f2.f2[static_cast<std::size_t>(i) +
                         static_cast<std::size_t>(P) * static_cast<std::size_t>(j) +
                         slab * static_cast<std::size_t>(b)];
        };
        for (int k = 0; k < N; ++k) {
            const int pc = triples[static_cast<std::size_t>(3 * k) + 0];
            const int pa = triples[static_cast<std::size_t>(3 * k) + 1];
            const int pb = triples[static_cast<std::size_t>(3 * k) + 2];
            for (int bs = 0; bs < nb_s; ++bs) {
                const int b = surv[static_cast<std::size_t>(bs)];
                const double x = 0.5 * (f2at(pc, pa, b) + f2at(pc, pb, b) -
                                        f2at(pa, pb, b));
                out.x_blocks[static_cast<std::size_t>(k) +
                             m * static_cast<std::size_t>(bs)] = x;
            }
        }

        compute_loo_and_total(out, out.block_sizes);
        return out;
    }

    // jackknife_cov: block-jackknife covariance — reference §7
    [[nodiscard]] JackknifeCov jackknife_cov(const F4Blocks& x,
                                             std::span<const int> block_sizes,
                                             double fudge,
                                             const Precision& precision) override {
        (void)precision;

        const int m = x.nl * x.nr;
        const int nb = x.n_block;
        JackknifeCov out;
        out.m = m;
        if (m <= 0 || nb <= 0) { out.status = Status::Ok; return out; }

        long double n_ld = 0.0L;
        for (int b = 0; b < nb; ++b) n_ld += static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
        const double n = static_cast<double>(n_ld);

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
        std::vector<double> Qf = out.Q;
        ridge_diagonal(Qf, m, fudge);
        const core::LinAlgStatus st = core::inverse(Qf, m, out.Qinv);
        if (!st.ok) out.Qinv.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
        out.status = st.ok ? Status::Ok : Status::NonSpdCovariance;
        return out;
    }

    // jackknife_diag: covariance diagonal only — reference §7
    [[nodiscard]] JackknifeDiag jackknife_diag(const F4Blocks& x,
                                               std::span<const int> block_sizes,
                                               const Precision& precision) override {
        (void)precision;

        const int m = x.nl * x.nr;
        const int nb = x.n_block;
        JackknifeDiag out;
        out.m = m;
        if (m <= 0 || nb <= 0) { out.status = Status::Ok; return out; }

        long double n_ld = 0.0L;
        for (int b = 0; b < nb; ++b) n_ld += static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
        const double n = static_cast<double>(n_ld);
        const std::size_t m_sz = static_cast<std::size_t>(m);

        out.var.assign(m_sz, 0.0);
        for (int k = 0; k < m; ++k) {
            long double acc = 0.0L;
            const double est = x.x_total[static_cast<std::size_t>(k)];
            const double totline = tot_line_[static_cast<std::size_t>(k)];
            for (int b = 0; b < nb; ++b) {
                const double bl = static_cast<double>(block_sizes[static_cast<std::size_t>(b)]);
                const double h = n / bl;
                const double sqrt_h_minus_1 = std::sqrt(h - 1.0);
                const double loo = x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)];
                const double xtau = (est * h - loo * (h - 1.0) - totline) / sqrt_h_minus_1;
                acc += static_cast<long double>(xtau) * static_cast<long double>(xtau);
            }
            out.var[static_cast<std::size_t>(k)] = static_cast<double>(acc / static_cast<long double>(nb));
        }
        out.status = Status::Ok;
        return out;
    }

    // ratio_block_jackknife: shared ratio jackknife — reference §8
    [[nodiscard]] RatioBlockJackknife ratio_block_jackknife(
        const RatioJackArray& num, const RatioJackArray& den, const RatioJackArray& weight,
        const RatioJackArray& xblk_num, const RatioJackArray& xblk_den, int N, int n_block,
        int tot_mode, double setmiss_thresh, bool compute_p,
        const Precision& precision) override {
        (void)precision;
        RatioBlockJackknife out;
        out.N = N;
        if (N <= 0 || n_block <= 0) { out.status = Status::Ok; return out; }
        out.est.assign(static_cast<std::size_t>(N), std::nan(""));
        out.se.assign(static_cast<std::size_t>(N), std::nan(""));
        out.z.assign(static_cast<std::size_t>(N), std::nan(""));
        if (compute_p) out.p.assign(static_cast<std::size_t>(N), std::nan(""));

        const auto at = [](const RatioJackArray& a, long k, int b) -> double {
            return a.data[a.base + k * a.item_stride +
                          static_cast<long>(b) * a.block_stride];
        };

        for (int k = 0; k < N; ++k) {
            const long kk = static_cast<long>(k);
            double est = std::nan(""), se = std::nan(""), z = std::nan("");

            if (tot_mode == 0) {
                long double n_ld = 0.0L; int nb_surv = 0;
                long double totnum = 0.0L, totden = 0.0L;
                for (int b = 0; b < n_block; ++b) {
                    const double den_blk = at(xblk_den, kk, b);
                    if (std::fabs(den_blk) < setmiss_thresh) continue;
                    const double bl = at(weight, kk, b);
                    n_ld += static_cast<long double>(bl);
                    ++nb_surv;
                    totnum += static_cast<long double>(at(xblk_num, kk, b)) *
                              static_cast<long double>(bl);
                    totden += static_cast<long double>(den_blk) * static_cast<long double>(bl);
                }
                if (nb_surv <= 1 || n_ld <= 0.0L || totden == 0.0L) {
                    out.est[static_cast<std::size_t>(k)] = std::nan("");
                    out.se[static_cast<std::size_t>(k)] = std::nan("");
                    out.z[static_cast<std::size_t>(k)] = std::nan("");
                    if (compute_p) out.p[static_cast<std::size_t>(k)] = std::nan("");
                    continue;
                }
                const double n = static_cast<double>(n_ld);
                const double nb = static_cast<double>(nb_surv);
                const double tot = static_cast<double>(totnum / totden);
                long double diffsum = 0.0L, wmean_num = 0.0L;
                for (int b = 0; b < n_block; ++b) {
                    const double den_blk = at(xblk_den, kk, b);
                    if (std::fabs(den_blk) < setmiss_thresh) continue;
                    const double bl = at(weight, kk, b);
                    const double Rb = at(num, kk, b) / at(den, kk, b);
                    diffsum += static_cast<long double>(tot) - static_cast<long double>(Rb);
                    wmean_num += static_cast<long double>(Rb) * static_cast<long double>(bl);
                }
                const double term1 =
                    static_cast<double>(diffsum / static_cast<long double>(nb_surv)) * nb;
                const double term2 = static_cast<double>(wmean_num / n_ld);
                est = term1 + term2;
                long double var_acc = 0.0L;
                for (int b = 0; b < n_block; ++b) {
                    const double den_blk = at(xblk_den, kk, b);
                    if (std::fabs(den_blk) < setmiss_thresh) continue;
                    const double bl = at(weight, kk, b);
                    const double Rb = at(num, kk, b) / at(den, kk, b);
                    const double h = n / bl;
                    const double tau = h * tot - (h - 1.0) * Rb;
                    const double xtau = (tau - est) / std::sqrt(h - 1.0);
                    var_acc += static_cast<long double>(xtau) * static_cast<long double>(xtau);
                }
                const double var = static_cast<double>(var_acc / static_cast<long double>(nb_surv));
                se = (var > 0.0) ? std::sqrt(var) : std::nan("");
                z = est / se;

            } else {
                long double sum_cnt = 0.0L; int nb_surv = 0;
                for (int b = 0; b < n_block; ++b) {
                    const double cn = at(weight, kk, b);
                    if (cn > 0.0) { sum_cnt += static_cast<long double>(cn); ++nb_surv; }
                }
                if (nb_surv <= 1 || sum_cnt <= 0.0L) {
                    out.est[static_cast<std::size_t>(k)] = std::nan("");
                    out.se[static_cast<std::size_t>(k)] = std::nan("");
                    out.z[static_cast<std::size_t>(k)] = std::nan("");
                    if (compute_p) out.p[static_cast<std::size_t>(k)] = std::nan("");
                    continue;
                }
                const double nb = static_cast<double>(nb_surv);
                long double tot_num_w = 0.0L, tot_den_w = 0.0L;
                for (int b = 0; b < n_block; ++b) {
                    const double cn = at(weight, kk, b);
                    if (cn <= 0.0) continue;
                    const long double est_num =
                        static_cast<long double>(at(num, kk, b)) / static_cast<long double>(cn);
                    const long double est_den =
                        static_cast<long double>(at(den, kk, b)) / static_cast<long double>(cn);
                    tot_num_w += est_num * static_cast<long double>(cn);
                    tot_den_w += est_den * static_cast<long double>(cn);
                }
                const long double tot_num = tot_num_w / sum_cnt;
                const long double tot_den = tot_den_w / sum_cnt;
                long double tot_w_num = 0.0L, tot_w_den = 0.0L, wmean_R_num = 0.0L;
                std::vector<double> Rb(static_cast<std::size_t>(n_block), std::nan(""));
                for (int b = 0; b < n_block; ++b) {
                    const double cn = at(weight, kk, b);
                    if (cn <= 0.0) continue;
                    const long double est_num =
                        static_cast<long double>(at(num, kk, b)) / static_cast<long double>(cn);
                    const long double est_den =
                        static_cast<long double>(at(den, kk, b)) / static_cast<long double>(cn);
                    const long double rel = static_cast<long double>(cn) / sum_cnt;
                    const long double loo_num = (tot_num - est_num * rel) / (1.0L - rel);
                    const long double loo_den = (tot_den - est_den * rel) / (1.0L - rel);
                    const long double R = loo_num / loo_den;
                    Rb[static_cast<std::size_t>(b)] = static_cast<double>(R);
                    const long double w = 1.0L - rel;
                    tot_w_num += R * w;
                    tot_w_den += w;
                    wmean_R_num += R * static_cast<long double>(cn);
                }
                const long double tot = tot_w_num / tot_w_den;
                long double diffsum = 0.0L;
                for (int b = 0; b < n_block; ++b) {
                    const double cn = at(weight, kk, b);
                    if (cn <= 0.0) continue;
                    diffsum += tot - static_cast<long double>(Rb[static_cast<std::size_t>(b)]);
                }
                const long double est_ld =
                    (diffsum / static_cast<long double>(nb_surv)) * static_cast<long double>(nb) +
                    wmean_R_num / sum_cnt;
                long double var_acc = 0.0L;
                for (int b = 0; b < n_block; ++b) {
                    const double cn = at(weight, kk, b);
                    if (cn <= 0.0) continue;
                    const long double h = sum_cnt / static_cast<long double>(cn);
                    const long double R =
                        static_cast<long double>(Rb[static_cast<std::size_t>(b)]);
                    const long double tau = h * tot - (h - 1.0L) * R - est_ld;
                    var_acc += (tau * tau) / (h - 1.0L);
                }
                const double var = static_cast<double>(var_acc / static_cast<long double>(nb_surv));
                est = static_cast<double>(est_ld);
                se = (var > 0.0) ? std::sqrt(var) : std::nan("");
                z = est / se;
            }

            out.est[static_cast<std::size_t>(k)] = est;
            out.se[static_cast<std::size_t>(k)] = se;
            out.z[static_cast<std::size_t>(k)] = z;
            if (compute_p) {
                static const double kInvSqrt2 = 1.0 / std::sqrt(2.0);
                out.p[static_cast<std::size_t>(k)] = std::erfc(std::fabs(z) * kInvSqrt2);
            }
        }
        out.status = Status::Ok;
        return out;
    }

    // f4ratio_blocks_jackknife — reference §8
    [[nodiscard]] RatioBlockJackknife f4ratio_blocks_jackknife(
        const F2BlockTensor& f2, std::span<const int> flat, int N, double setmiss_thresh,
        const Precision& precision) override {
        RatioBlockJackknife out;
        out.N = N;
        if (N <= 0) { out.status = Status::Ok; return out; }
        const F4Blocks X = assemble_f4_quartets(f2, flat, precision);
        const long m = static_cast<long>(X.nl) * static_cast<long>(X.nr);
        const int nb = X.n_block;
        if (m <= 0 || nb <= 0) {
            out.est.assign(static_cast<std::size_t>(N), std::nan(""));
            out.se.assign(static_cast<std::size_t>(N), std::nan(""));
            out.z.assign(static_cast<std::size_t>(N), std::nan(""));
            out.status = Status::Ok;
            return out;
        }
        std::vector<double> w(static_cast<std::size_t>(nb));
        for (int b = 0; b < nb; ++b)
            w[static_cast<std::size_t>(b)] =
                static_cast<double>(X.block_sizes[static_cast<std::size_t>(b)]);

        RatioJackArray num{X.x_loo.data(), 0, 1, m};
        RatioJackArray den{X.x_loo.data(), static_cast<long>(N), 1, m};
        RatioJackArray weight{w.data(), 0, 0, 1};
        RatioJackArray xbn{X.x_blocks.data(), 0, 1, m};
        RatioJackArray xbd{X.x_blocks.data(), static_cast<long>(N), 1, m};
        return ratio_block_jackknife(num, den, weight, xbn, xbd, N, nb,
                                     /*tot_mode=*/0, setmiss_thresh, /*compute_p=*/false,
                                     precision);
    }

    // dstat_blocks_jackknife — reference §8
    [[nodiscard]] RatioBlockJackknife dstat_blocks_jackknife(
        const double* Q, const double* V, int P, long M, const int* block_id, int n_block,
        std::span<const int> quadruples) override {
        const int N = static_cast<int>(quadruples.size() / 4);
        RatioBlockJackknife out;
        out.N = N;
        if (N <= 0 || n_block <= 0 || P <= 0 || M <= 0) {
            out.est.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            out.se.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            out.z.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            out.p.assign(static_cast<std::size_t>(std::max(N, 0)), std::nan(""));
            out.status = Status::Ok;
            return out;
        }
        const std::size_t nb_out =
            static_cast<std::size_t>(N) * static_cast<std::size_t>(n_block);
        std::vector<double> numsum(nb_out, 0.0), densum(nb_out, 0.0), cnt(nb_out, 0.0);
        dstat_block_reduce(Q, V, P, M, block_id, n_block, quadruples, numsum.data(),
                           densum.data(), cnt.data());
        const long nbl = static_cast<long>(n_block);
        RatioJackArray num{numsum.data(), 0, nbl, 1};
        RatioJackArray den{densum.data(), 0, nbl, 1};
        RatioJackArray weight{cnt.data(), 0, nbl, 1};
        RatioJackArray null{nullptr, 0, 0, 0};
        return ratio_block_jackknife(num, den, weight, null, null, N, n_block,
                                     /*tot_mode=*/1, /*setmiss_thresh=*/0.0, /*compute_p=*/true,
                                     Precision{});
    }

    // provides_rank_sweep: capability query — reference §9
    [[nodiscard]] bool provides_rank_sweep() const override { return true; }

    // rank_sweep: qpAdm rank test — reference §9
    [[nodiscard]] RankSweep rank_sweep(const F4Blocks& x,
                                       const JackknifeCov& cov,
                                       double alpha,
                                       const QpAdmOptions& opts,
                                       const Precision& precision) override {
        (void)precision;
        RankSweep rs;
        const int nl = x.nl, nr = x.nr;
        const int m = nl * nr;
        const int rmax = (nl < nr ? nl : nr) - 1;
        if (rmax < 0) {
            rs.status = Status::RankDeficient;
            rs.svd_path = (nl <= kGesvdjMaxDim && nr <= kGesvdjMaxDim) ? 1 : 2;
            return rs;
        }
        std::vector<double> xmat = xmat_from_total(x);

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

        const std::size_t n = static_cast<std::size_t>(rmax) + 1;
        rs.rd_f4rank.resize(n); rs.rd_dof.resize(n); rs.rd_chisq.resize(n);
        rs.rd_p.resize(n); rs.rd_dofdiff.resize(n);
        rs.rd_chisqdiff.resize(n); rs.rd_p_nested.resize(n);
        for (std::size_t k = 0; k < n; ++k) {
            const int r = rmax - static_cast<int>(k);
            rs.rd_f4rank[k] = r;
            rs.rd_dof[k] = rs.dof[static_cast<std::size_t>(r)];
            rs.rd_chisq[k] = rs.chisq[static_cast<std::size_t>(r)];
            rs.rd_p[k] = rs.p[static_cast<std::size_t>(r)];
            if (r - 1 >= 0) {
                const int dof_diff = rs.dof[static_cast<std::size_t>(r - 1)] -
                                     rs.dof[static_cast<std::size_t>(r)];
                const double chisq_diff = rs.chisq[static_cast<std::size_t>(r - 1)] -
                                          rs.chisq[static_cast<std::size_t>(r)];
                rs.rd_dofdiff[k] = dof_diff;
                rs.rd_chisqdiff[k] = chisq_diff;
                rs.rd_p_nested[k] = core::internal::pchisq_upper(chisq_diff, dof_diff);
            } else {
                rs.rd_dofdiff[k] = INT_MIN;
                rs.rd_chisqdiff[k] = std::numeric_limits<double>::quiet_NaN();
                rs.rd_p_nested[k] = std::numeric_limits<double>::quiet_NaN();
            }
        }

        rs.f4rank = rmax;
        for (int r = 0; r <= rmax; ++r) {
            if (rs.p[static_cast<std::size_t>(r)] > alpha) { rs.f4rank = r; break; }
        }

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

        rs.svd_path = (nl <= kGesvdjMaxDim && nr <= kGesvdjMaxDim) ? 1 : 2;

        rs.status = degenerate ? Status::RankDeficient
                               : (cov.status != Status::Ok ? cov.status : Status::Ok);
        return rs;
    }

    // gls_weights: GLS weights via ALS — reference §9
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

    // gls_weights_loo_batched: leave-one-out re-fits — reference §10
    [[nodiscard]] std::vector<double> gls_weights_loo_batched(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override {
        const int nl = x.nl, nr = x.nr, nb = x.n_block;
        const int m = nl * nr;
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

    // se_from_wmat: leave-one-block-out SE — reference §10
    [[nodiscard]] std::vector<double> se_from_wmat(
        const F4Blocks& x, const JackknifeCov& cov, int r,
        const QpAdmOptions& opts, const Precision& precision) override {
        const int nl = x.nl, nb = x.n_block;
        std::vector<double> se(static_cast<std::size_t>(nl < 0 ? 0 : nl), 0.0);
        constexpr int kMinJackknifeBlocks = 2;
        if (nb < kMinJackknifeBlocks || nl <= 0) return se;
        std::vector<double> wmat = gls_weights_loo_batched(x, cov, r, opts, precision);
        const double scale = static_cast<double>(nb - 1) / std::sqrt(static_cast<double>(nb));
        for (double& v : wmat) v *= scale;
        const std::vector<double> diag = se_sample_cov_diag(wmat, nb, nl);
        for (int i = 0; i < nl; ++i)
            se[static_cast<std::size_t>(i)] = std::sqrt(diag[static_cast<std::size_t>(i)]);
        return se;
    }

private:
    // tot_line_: jackknife centering cache — reference §7
    std::vector<double> tot_line_{};

    // compute_loo_and_total: leave-one-out pass — reference §7
    void compute_loo_and_total(F4Blocks& x, const std::vector<int>& block_sizes) {
        const int nl = x.nl, nr = x.nr, nb = x.n_block;
        const int m = nl * nr;
        const std::size_t m_sz = static_cast<std::size_t>(m);
        x.x_loo.assign(m_sz * static_cast<std::size_t>(nb), 0.0);
        x.x_total.assign(m_sz, 0.0);
        tot_line_.assign(m_sz, 0.0);
        if (m <= 0 || nb <= 0) return;

        long double n_ld = 0.0L;
        for (int b = 0; b < nb; ++b) n_ld += static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
        const double n = static_cast<double>(n_ld);

        for (int k = 0; k < m; ++k) {
            long double num = 0.0L;
            for (int b = 0; b < nb; ++b) {
                num += static_cast<long double>(x.x_blocks[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)]) *
                       static_cast<long double>(block_sizes[static_cast<std::size_t>(b)]);
            }
            const double tot_ij = static_cast<double>(num / n_ld);
            for (int b = 0; b < nb; ++b) {
                const double bl = static_cast<double>(block_sizes[static_cast<std::size_t>(b)]);
                const double rel = bl / n;
                const double xv = x.x_blocks[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)];
                x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)] =
                    (tot_ij - xv * rel) / (1.0 - rel);
            }
            long double wmean_num = 0.0L, wmean_den = 0.0L;
            for (int b = 0; b < nb; ++b) {
                const double w = 1.0 - static_cast<double>(block_sizes[static_cast<std::size_t>(b)]) / n;
                wmean_num += static_cast<long double>(x.x_loo[static_cast<std::size_t>(k) + m_sz * static_cast<std::size_t>(b)]) *
                             static_cast<long double>(w);
                wmean_den += static_cast<long double>(w);
            }
            tot_line_[static_cast<std::size_t>(k)] = static_cast<double>(wmean_num / wmean_den);
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

    // xmat_from_total — reference §9
    [[nodiscard]] static std::vector<double> xmat_from_total(const F4Blocks& x) {
        const int nl = x.nl, nr = x.nr;
        std::vector<double> xmat(static_cast<std::size_t>(nl) * static_cast<std::size_t>(nr), 0.0);
        for (int i = 0; i < nl; ++i)
            for (int j = 0; j < nr; ++j)
                xmat[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(j)] =
                    x.x_total[static_cast<std::size_t>(j) + static_cast<std::size_t>(nr) * static_cast<std::size_t>(i)];
        return xmat;
    }

    // seed_AB: SVD seed — reference §9
    static void seed_AB(const std::vector<double>& xmat, int nl, int nr, int r,
                        std::vector<double>& A, std::vector<double>& B) {
        const core::SvdResult sv = core::jacobi_svd(xmat, nl, nr);
        B.assign(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
        for (int p = 0; p < r; ++p)
            for (int j = 0; j < nr; ++j)
                B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)] =
                    sv.V[static_cast<std::size_t>(j) + static_cast<std::size_t>(nr) * static_cast<std::size_t>(p)];
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

    // ridge_diagonal: fudge-ridge idiom — reference §9
    static void ridge_diagonal(std::vector<double>& mat, int n, double fudge) {
        double tr = 0.0;
        for (int k = 0; k < n; ++k)
            tr += mat[static_cast<std::size_t>(k) + static_cast<std::size_t>(n) * static_cast<std::size_t>(k)];
        for (int k = 0; k < n; ++k)
            mat[static_cast<std::size_t>(k) + static_cast<std::size_t>(n) * static_cast<std::size_t>(k)] += fudge * tr;
    }

    // als_xvec: row-major flatten — reference §9
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

    // als_ridge_solve: shared GLS-ridge core — reference §9
    template <typename Linop>
    [[nodiscard]] static std::vector<double> als_ridge_solve(
        const Linop& linop, int m, int t, const std::vector<double>& xvec,
        const std::vector<double>& qinv, double fudge) {
        std::vector<double> W(static_cast<std::size_t>(m) * static_cast<std::size_t>(t), 0.0);
        for (int kr = 0; kr < m; ++kr)
            for (int a = 0; a < t; ++a) {
                double acc = 0.0;
                for (int kc = 0; kc < m; ++kc)
                    acc += qinv[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(kc)] *
                           linop(kc, a);
                W[static_cast<std::size_t>(kr) + static_cast<std::size_t>(m) * static_cast<std::size_t>(a)] = acc;
            }
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

    // opt_A: ALS half-step for A — reference §9
    static std::vector<double> opt_A(const std::vector<double>& B,
                                     const std::vector<double>& xmat, int nl, int nr, int r,
                                     const std::vector<double>& qinv, double fudge) {
        const int m = nl * nr;
        const int t = nl * r;
        const auto L = [&](int k, int a) -> double {
            const int i = a / r, p = a % r;
            const int ii = k / nr, j = k % nr;
            return (i == ii) ? B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)]
                             : 0.0;
        };
        const std::vector<double> xvec = als_xvec(xmat, nl, nr);
        const std::vector<double> A2 = als_ridge_solve(L, m, t, xvec, qinv, fudge);
        std::vector<double> A(static_cast<std::size_t>(nl) * static_cast<std::size_t>(r), 0.0);
        if (!A2.empty())
            for (int i = 0; i < nl; ++i)
                for (int p = 0; p < r; ++p)
                    A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)] =
                        A2[static_cast<std::size_t>(i * r + p)];
        return A;
    }

    // opt_B: ALS half-step for B — reference §9
    static std::vector<double> opt_B(const std::vector<double>& A,
                                     const std::vector<double>& xmat, int nl, int nr, int r,
                                     const std::vector<double>& qinv, double fudge) {
        const int m = nl * nr;
        const int t = r * nr;
        const auto L = [&](int k, int c) -> double {
            const int i = k / nr, j = k % nr;
            const int p = c / nr, jc = c % nr;
            return (j == jc) ? A[static_cast<std::size_t>(i) + static_cast<std::size_t>(nl) * static_cast<std::size_t>(p)]
                             : 0.0;
        };
        const std::vector<double> xvec = als_xvec(xmat, nl, nr);
        const std::vector<double> B2 = als_ridge_solve(L, m, t, xvec, qinv, fudge);
        std::vector<double> B(static_cast<std::size_t>(r) * static_cast<std::size_t>(nr), 0.0);
        if (!B2.empty())
            for (int p = 0; p < r; ++p)
                for (int j = 0; j < nr; ++j)
                    B[static_cast<std::size_t>(p) + static_cast<std::size_t>(r) * static_cast<std::size_t>(j)] =
                        B2[static_cast<std::size_t>(p * nr + j)];
        return B;
    }

    // als_weights: qpadm_weights body — reference §9
    [[nodiscard]] static GlsWeights als_weights(const std::vector<double>& xmat, int nl, int nr,
                                                int r, const std::vector<double>& qinv,
                                                const QpAdmOptions& opts) {
        GlsWeights gw;
        gw.r = r;
        if (r == 0) {
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
        const int rp = r + 1;
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
            LHS[static_cast<std::size_t>(i)] = xm(i, r);
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

    // chisq_of: residual quadratic form — reference §9
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
                e[static_cast<std::size_t>(i * nr + j)] = resid;
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

    // qpgraph_fit_fleet: graph edge-weight optimizer — reference §13
    [[nodiscard]] QpGraphFleet qpgraph_fit_fleet(const QpGraphTopoArena& topo,
                                                 std::span<const double> f_obs,
                                                 std::span<const double> qinv,
                                                 int numstart, int maxit, double tol,
                                                 const Precision& precision) override {
        (void)precision;
        using core::qpadm::QpGraphModel;
        QpGraphModel m;
        m.npop = topo.npop; m.nedge_norm = topo.nedge_norm; m.nadmix = topo.nadmix;
        m.npair = topo.npair; m.npath = topo.npath; m.base_leaf = topo.base_leaf;
        m.pwts0 = topo.pwts0;
        m.pe_edge = topo.pe_edge; m.pe_leaf = topo.pe_leaf; m.pe_path = topo.pe_path;
        m.pae_path = topo.pae_path; m.pae_admixedge = topo.pae_admixedge;
        m.cmb1 = topo.cmb1; m.cmb2 = topo.cmb2;
        const std::vector<double> fobs(f_obs.begin(), f_obs.end());
        const std::vector<double> ppinv(qinv.begin(), qinv.end());
        const int D = m.nadmix;
        const double fudge = topo.fudge;
        const bool constrained = topo.constrained;

        QpGraphFleet out;
        if (D == 0) {
            std::vector<double> bl, fit(static_cast<std::size_t>(m.npair), 0.0);
            const double sc = core::qpadm::qpgraph_score(m, nullptr, fobs, ppinv, fudge,
                                                         constrained, &bl, &fit);
            out.score = sc; out.restart_spread = 0.0; out.edge_length = bl; out.f3_fit = fit;
            out.status = std::isfinite(sc) ? Status::Ok : Status::NonSpdCovariance;
            return out;
        }

        namespace opt = core::qpadm::qpgraph_opt;
        const auto clamp01 = [](double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
        const auto init_theta = [](unsigned inst, int dim) -> double {
            unsigned long long z = (static_cast<unsigned long long>(inst) * opt::kSplitmixInstMul) +
                                   (static_cast<unsigned long long>(dim) * opt::kSplitmixDimMul) +
                                   opt::kSplitmixSeedInc;
            z = (z ^ (z >> 30)) * opt::kSplitmixMix1;
            z = (z ^ (z >> 27)) * opt::kSplitmixMix2;
            z = z ^ (z >> 31);
            return static_cast<double>(z & opt::kMantissaMask) / opt::kMantissaDiv;
        };
        const auto sco = [&](const std::vector<double>& th) {
            return core::qpadm::qpgraph_score(m, th.data(), fobs, ppinv, fudge, constrained);
        };

        double best_score = std::numeric_limits<double>::infinity();
        std::vector<double> best_theta(static_cast<std::size_t>(D), 0.0);
        double smin = std::numeric_limits<double>::infinity(), smax = -std::numeric_limits<double>::infinity();
        std::vector<double> thmin(static_cast<std::size_t>(D), 1.0), thmax(static_cast<std::size_t>(D), 0.0);

        const double h = opt::kFdStep;
        for (int inst = 0; inst < numstart; ++inst) {
            std::vector<double> th(static_cast<std::size_t>(D));
            for (int d = 0; d < D; ++d) th[static_cast<std::size_t>(d)] = clamp01(init_theta(static_cast<unsigned>(inst), d));
            double s = sco(th);
            for (int it = 0; it < maxit; ++it) {
                double max_dx = 0.0, max_ds = 0.0;
                for (int d = 0; d < D; ++d) {
                    const double w = th[static_cast<std::size_t>(d)];
                    const double wp = std::min(1.0, w + h), wm = std::max(0.0, w - h);
                    std::vector<double> thp = th, thm = th;
                    thp[static_cast<std::size_t>(d)] = wp; thm[static_cast<std::size_t>(d)] = wm;
                    const double sp = sco(thp), sm = sco(thm);
                    const double dwp = wp - w, dwm = w - wm;
                    double g, curv;
                    if (dwp > 0.0 && dwm > 0.0) {
                        g = (sp - sm) / (dwp + dwm);
                        curv = (sp - 2.0 * s + sm) / (opt::kCurvHalf * (dwp + dwm) * (dwp + dwm) + opt::kCurvGuard);
                    } else if (dwp > 0.0) { g = (sp - s) / dwp; curv = 1.0; }
                    else { g = (s - sm) / dwm; curv = 1.0; }
                    double step = (curv > opt::kCurvThresh) ? (g / curv) : (g * opt::kGradStepScale);
                    if (step > opt::kTrustClamp) step = opt::kTrustClamp;
                    if (step < -opt::kTrustClamp) step = -opt::kTrustClamp;
                    double wn = clamp01(w - step);
                    std::vector<double> thn = th; thn[static_cast<std::size_t>(d)] = wn;
                    double sn = sco(thn);
                    int bt = 0;
                    while (sn > s && bt < opt::kMaxBacktrack) {
                        wn = opt::kBacktrackHalf * (wn + w); thn[static_cast<std::size_t>(d)] = wn; sn = sco(thn); ++bt;
                    }
                    const double dx = std::fabs(wn - w), ds = std::fabs(sn - s);
                    if (sn <= s) { th[static_cast<std::size_t>(d)] = wn; s = sn; }
                    if (dx > max_dx) max_dx = dx;
                    if (ds > max_ds) max_ds = ds;
                }
                if (max_dx < tol * opt::kTolDxScale && max_ds < tol * opt::kTolDsScale) break;
            }
            if (s < smin) smin = s;
            if (s > smax) smax = s;
            for (int d = 0; d < D; ++d) {
                if (th[static_cast<std::size_t>(d)] < thmin[static_cast<std::size_t>(d)]) thmin[static_cast<std::size_t>(d)] = th[static_cast<std::size_t>(d)];
                if (th[static_cast<std::size_t>(d)] > thmax[static_cast<std::size_t>(d)]) thmax[static_cast<std::size_t>(d)] = th[static_cast<std::size_t>(d)];
            }
            if (s < best_score) { best_score = s; best_theta = th; }
        }

        out.score = best_score;
        out.restart_spread = (std::isfinite(smax) && std::isfinite(smin)) ? (smax - smin) : 0.0;
        out.theta = best_theta;
        out.theta_lo = thmin; out.theta_hi = thmax;
        out.edge_length.assign(static_cast<std::size_t>(m.nedge_norm), 0.0);
        out.f3_fit.assign(static_cast<std::size_t>(m.npair), 0.0);
        std::vector<double> bl, fit(static_cast<std::size_t>(m.npair), 0.0);
        const double sc_final = core::qpadm::qpgraph_score(m, best_theta.data(), fobs, ppinv,
                                                           fudge, constrained, &bl, &fit);
        if (std::isfinite(sc_final)) { out.edge_length = bl; out.f3_fit = fit; out.status = Status::Ok; }
        else out.status = Status::NonSpdCovariance;
        return out;
    }

    // qpgraph_fit_fleet_batch: per-topology fleet — reference §13
    [[nodiscard]] QpGraphFleetBatch qpgraph_fit_fleet_batch(
        const std::vector<QpGraphTopoArena>& topos, std::span<const double> f_obs,
        std::span<const double> qinv, int numstart, int maxit, double tol,
        const Precision& precision) override {
        QpGraphFleetBatch out;
        out.best_score.reserve(topos.size());
        out.restart_spread.reserve(topos.size());
        for (const QpGraphTopoArena& topo : topos) {
            const QpGraphFleet f =
                qpgraph_fit_fleet(topo, f_obs, qinv, numstart, maxit, tol, precision);
            out.best_score.push_back(f.status == Status::Ok ? f.score
                                                            : std::numeric_limits<double>::infinity());
            out.restart_spread.push_back(f.restart_spread);
        }
        out.status = Status::Ok;
        return out;
    }
};

}  // namespace

// make_cpu_backend: factory (DI seam) — reference §14
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace steppe::device
