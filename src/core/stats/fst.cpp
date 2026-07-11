// src/core/stats/fst.cpp
//
// run_fst: the standalone pairwise per-SNP Weir & Cockerham 1984 FST computed straight
// from genotype files. It reuses the shared genotype decode front-end (read the triple to
// the canonical population-contiguous tile) and then branches into its own per-site WC
// variance-component kernel — a distinct per-site reduction, NOT the f2/AF decode path
// (the AF grid drops the allele count n and the observed het h that WC needs). Native FP64.
//
// Diploid is forced (matching plink2 on a shared .bed): each 2-bit code is a diploid
// dosage 0/1/2. Missing genotypes are excluded per-site per-sample by the WC accumulator.
#include "steppe/fst.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/internal/decode_af.hpp"
#include "core/internal/primary_backend.hpp"
#include "core/stats/apply_snp_filter.hpp"
#include "core/stats/bp_windows.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"
#include "io/filter/snp_filter.hpp"
#include "io/genotype_tile.hpp"
#include "io/snp_reader.hpp"

namespace steppe {

namespace {

// The Yi et al. 2010 branch length T = -ln(1 - Fst), with Fst clamped to [0, 1) so T is finite
// (Fst < 0 -> 0; Fst >= 1 -> just below 1). A NaN pairwise Fst (empty / monomorphic window)
// propagates to a NaN T (matching the hand oracle np.clip(fst, 0, 1-1e-15) then -log(1-.)).
[[nodiscard]] double pbs_branch_T(double fst) noexcept {
    if (std::isnan(fst)) return std::nan("");
    double f = fst;
    if (f < 0.0) f = 0.0;
    const double hi = 1.0 - 1e-15;
    if (f > hi) f = hi;
    return -std::log(1.0 - f);
}

// Shared windowed-fold front-end for run_fst_windowed / run_fst_pbs: read the triple keeping the
// requested distinct pops `want` (via the shared genotype front-end), apply the QC filter, build
// the per-chromosome bp windows (allel-exact), resolve each pop label to its tile index, and run
// the GPU window fold for `pairs_in_want` (pairs expressed as indices into `want`). On success it
// fills `windows`, `kept_ids`, and the backend `fold` result and returns Status::Ok.
[[nodiscard]] Status run_windowed_fold(const std::string& geno, const std::string& snp,
                                       const std::string& ind, std::span<const std::string> want,
                                       const std::vector<std::pair<int, int>>& pairs_in_want,
                                       long win_size, long win_step, device::Resources& resources,
                                       const FilterConfig& filter,
                                       std::vector<core::BpWindow>& windows,
                                       std::vector<std::string>& kept_ids, FstWindowed& fold) {
    ComputeBackend& be = device::primary_backend(resources);

    const bool allow_device =
        !(io::filter::filter_is_active(filter) && core::host_filter_forced());
    core::GenotypeFrontEnd fe =
        core::read_genotype_front_end(geno, snp, ind, want, be, allow_device);
    const core::SnpFilterOutcome flt =
        core::apply_snp_filter(fe.tile, fe.dev_tile, fe.snptab, filter, be);
    kept_ids = flt.kept_ids;
    const io::GenotypeTile& tile = fe.tile;
    const io::SnpTable& snptab = fe.snptab;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P < static_cast<int>(want.size()) || M <= 0) return Status::InvalidConfig;

    // Resolve each requested label to its tile pop index (order-independent; the front-end may
    // reorder the requested pops in the tile).
    std::vector<int> idx(want.size(), -1);
    for (std::size_t p = 0; p < tile.pop_labels.size(); ++p) {
        for (std::size_t w = 0; w < want.size(); ++w) {
            if (tile.pop_labels[p] == want[w]) idx[w] = static_cast<int>(p);
        }
    }
    for (const int v : idx) {
        if (v < 0) return Status::InvalidConfig;
    }

    std::vector<int> pair_a, pair_b;
    pair_a.reserve(pairs_in_want.size());
    pair_b.reserve(pairs_in_want.size());
    for (const auto& pr : pairs_in_want) {
        pair_a.push_back(idx[static_cast<std::size_t>(pr.first)]);
        pair_b.push_back(idx[static_cast<std::size_t>(pr.second)]);
    }

    // Per-chromosome bp windows from the kept SNP coordinates (chromosome-reset, allel-exact).
    windows = core::build_bp_windows(std::span<const int>(snptab.chrom.data(), snptab.chrom.size()),
                                     std::span<const double>(snptab.physpos.data(),
                                                             snptab.physpos.size()),
                                     win_size, win_step);
    std::vector<long> win_lo, win_hi;
    win_lo.reserve(windows.size());
    win_hi.reserve(windows.size());
    for (const core::BpWindow& w : windows) {
        win_lo.push_back(w.lo);
        win_hi.push_back(w.hi);
    }

    // Force diploid (the WC accumulator reads each code as a diploid dosage; the ploidy vector
    // only sizes the view). Then the GPU window fold over the decoded sufficient statistic.
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view =
        fe.dev_tile.valid() ? core::make_decode_tile_view(fe.dev_tile, sample_ploidy, P)
                            : core::make_decode_tile_view(tile, sample_ploidy, P);
    fold = be.fst_wc_windowed(view, std::span<const int>(pair_a), std::span<const int>(pair_b),
                              std::span<const long>(win_lo), std::span<const long>(win_hi));
    return Status::Ok;
}

}  // namespace

FstResult run_fst(const std::string& geno, const std::string& snp, const std::string& ind,
                  const std::string& popA, const std::string& popB,
                  device::Resources& resources, const FilterConfig& filter) {
    FstResult res;
    res.precision_tag = Precision::Kind::Fp64;
    res.popA = popA;
    res.popB = popB;
    res.fst_ratio = std::nan("");
    res.fst_mean = std::nan("");

    if (popA == popB) {
        res.status = Status::InvalidConfig;
        return res;
    }

    ComputeBackend& be = device::primary_backend(resources);

    // Read the triple to the canonical tile, keeping ONLY the two requested populations
    // (population-contiguous, so each pop is one [begin, end) individual range).
    const std::vector<std::string> want{popA, popB};
    const bool allow_device =
        !(io::filter::filter_is_active(filter) && core::host_filter_forced());
    core::GenotypeFrontEnd fe = core::read_genotype_front_end(
        geno, snp, ind, std::span<const std::string>(want), be, allow_device);
    // Per-SNP QC filter: subset the SNP axis (only the SNP set changes; per-site WC math is
    // untouched) so a filtered run is bit-exact vs an externally pre-subset triple.
    const core::SnpFilterOutcome flt =
        core::apply_snp_filter(fe.tile, fe.dev_tile, fe.snptab, filter, be);
    res.kept_snp_ids = flt.kept_ids;
    const io::GenotypeTile& tile = fe.tile;
    const io::SnpTable& snptab = fe.snptab;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P < 2 || M <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Resolve the two population labels to tile indices (order-independent).
    int idxA = -1, idxB = -1;
    for (std::size_t p = 0; p < tile.pop_labels.size(); ++p) {
        if (tile.pop_labels[p] == popA) idxA = static_cast<int>(p);
        if (tile.pop_labels[p] == popB) idxB = static_cast<int>(p);
    }
    if (idxA < 0 || idxB < 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Force diploid: the WC accumulator reads each code as a diploid dosage; the ploidy
    // vector only sizes the view (the kernel ignores it), so a diploid fill is correct.
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view =
        fe.dev_tile.valid() ? core::make_decode_tile_view(fe.dev_tile, sample_ploidy, P)
                            : core::make_decode_tile_view(tile, sample_ploidy, P);

    // Autosome mask for the genome-wide summary (plink2 method=wc summarizes autosomes).
    const std::size_t Mz = static_cast<std::size_t>(M);
    std::vector<std::uint8_t> summary_include(Mz, 0);
    for (std::size_t s = 0; s < Mz && s < snptab.chrom.size(); ++s) {
        const int chr = snptab.chrom[s];
        summary_include[s] =
            (chr >= kAutosomeChromMin && chr <= kAutosomeChromMax) ? std::uint8_t{1}
                                                                   : std::uint8_t{0};
    }

    const FstPerSite fps = be.fst_wc_per_site(
        view, idxA, idxB, std::span<const std::uint8_t>(summary_include));

    res.num = fps.num;
    res.den = fps.den;
    res.fst = fps.fst;
    res.valid = fps.valid;
    res.n_valid = fps.n_valid;
    res.precision_tag = fps.precision_tag;

    // Carry the per-SNP coordinates (kept order == tile SNP order).
    res.chrom.assign(Mz, 0);
    res.physpos.assign(Mz, 0.0);
    res.snp_id.assign(Mz, std::string());
    res.a1.assign(Mz, '?');
    res.a2.assign(Mz, '?');
    for (std::size_t s = 0; s < Mz; ++s) {
        if (s < snptab.chrom.size()) res.chrom[s] = snptab.chrom[s];
        if (s < snptab.physpos.size()) res.physpos[s] = snptab.physpos[s];
        if (s < snptab.id.size()) res.snp_id[s] = snptab.id[s];
        if (s < snptab.ref.size()) res.a1[s] = snptab.ref[s];
        if (s < snptab.alt.size()) res.a2[s] = snptab.alt[s];
    }

    // Genome-wide summary: the ratio-of-averages (plink2 .fst.summary) and the unweighted
    // mean of the per-SNP FST over the same autosomal valid sites.
    res.fst_ratio = (fps.sum_den != 0.0) ? (fps.sum_num / fps.sum_den) : std::nan("");
    if (fps.n_valid > 0) {
        long double acc = 0.0L;
        for (std::size_t s = 0; s < Mz; ++s) {
            if (res.valid[s] && summary_include[s]) acc += static_cast<long double>(res.fst[s]);
        }
        res.fst_mean = static_cast<double>(acc / static_cast<long double>(fps.n_valid));
    }

    res.status = Status::Ok;
    return res;
}

FstMatrixResult run_fst_all_pairs(const std::string& geno, const std::string& snp,
                                  const std::string& ind, const std::vector<std::string>& pops,
                                  int min_n, bool sure, device::Resources& resources,
                                  const FilterConfig& filter) {
    FstMatrixResult res;
    res.precision_tag = Precision::Kind::Fp64;

    ComputeBackend& be = device::primary_backend(resources);

    // Population selection: an explicit set (the emitted matrix order) or ALL pops with
    // N >= min_n. Mirrors run_pca's decode-once front-end (io::PopSelection), no new reader.
    io::PopSelection sel;
    if (!pops.empty()) {
        sel.mode = io::PopSelection::Mode::Explicit;
        sel.labels.assign(pops.begin(), pops.end());
    } else {
        sel.mode = io::PopSelection::Mode::MinN;
        sel.min_n = (min_n >= 1) ? static_cast<std::size_t>(min_n) : std::size_t{1};
    }

    const bool allow_device =
        !(io::filter::filter_is_active(filter) && core::host_filter_forced());
    core::GenotypeFrontEnd fe =
        core::read_genotype_front_end(geno, snp, ind, sel, be, allow_device);
    const core::SnpFilterOutcome flt =
        core::apply_snp_filter(fe.tile, fe.dev_tile, fe.snptab, filter, be);
    res.kept_snp_ids = flt.kept_ids;
    const io::GenotypeTile& tile = fe.tile;
    const io::SnpTable& snptab = fe.snptab;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    res.pops = tile.pop_labels;  // cell (i,j) is labeled by the tile pop order the unrank uses
    if (P < 2 || M <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // Force diploid (the WC accumulator reads each code as a diploid dosage; the ploidy vector
    // only sizes the view). Autosome summary mask, identical block to the single-pair run_fst.
    const std::vector<int> sample_ploidy(tile.n_individuals, core::kPloidyDiploid);
    const DecodeTileView view =
        fe.dev_tile.valid() ? core::make_decode_tile_view(fe.dev_tile, sample_ploidy, P)
                            : core::make_decode_tile_view(tile, sample_ploidy, P);

    const std::size_t Mz = static_cast<std::size_t>(M);
    std::vector<std::uint8_t> summary_include(Mz, 0);
    for (std::size_t s = 0; s < Mz && s < snptab.chrom.size(); ++s) {
        const int chr = snptab.chrom[s];
        summary_include[s] =
            (chr >= kAutosomeChromMin && chr <= kAutosomeChromMax) ? std::uint8_t{1}
                                                                   : std::uint8_t{0};
    }

    const FstMatrix mat =
        be.fst_wc_all_pairs(view, std::span<const std::uint8_t>(summary_include), sure);
    res.enumerated = mat.enumerated;
    res.capped = mat.capped;
    res.precision_tag = mat.precision_tag;
    res.status = mat.status;
    if (mat.status != Status::Ok) return res;

    // Expand the per-pair Σ vectors into the symmetric P x P matrix. The device enumerated
    // pairs by the flat rank r = j*(j-1)/2 + i (i < j) — the exact ordering readv2_unrank_pair
    // inverts — so iterate (i < j) and index pair vectors by that same r. Diagonal = 0; an
    // all-invalid pair (den == 0) gets the NaN sentinel, mirroring run_fst's fst_ratio guard.
    const std::size_t Pz = static_cast<std::size_t>(P);
    res.fst.assign(Pz * Pz, 0.0);
    res.num.assign(Pz * Pz, 0.0);
    res.den.assign(Pz * Pz, 0.0);
    res.n_valid.assign(Pz * Pz, 0L);
    for (int j = 1; j < P; ++j) {
        for (int i = 0; i < j; ++i) {
            const std::size_t r =
                static_cast<std::size_t>(j) * static_cast<std::size_t>(j - 1) / 2u +
                static_cast<std::size_t>(i);
            const double num = mat.pair_num[r];
            const double den = mat.pair_den[r];
            const double fst = (den != 0.0) ? (num / den) : std::nan("");
            const std::size_t ij = static_cast<std::size_t>(i) * Pz + static_cast<std::size_t>(j);
            const std::size_t ji = static_cast<std::size_t>(j) * Pz + static_cast<std::size_t>(i);
            res.fst[ij] = res.fst[ji] = fst;
            res.num[ij] = res.num[ji] = num;
            res.den[ij] = res.den[ji] = den;
            res.n_valid[ij] = res.n_valid[ji] = mat.pair_cnt[r];
        }
    }

    res.status = Status::Ok;
    return res;
}

FstWindowedResult run_fst_windowed(const std::string& geno, const std::string& snp,
                                   const std::string& ind, const std::string& popA,
                                   const std::string& popB, long win_size, long win_step,
                                   device::Resources& resources, const FilterConfig& filter) {
    FstWindowedResult res;
    res.precision_tag = Precision::Kind::Fp64;
    res.popA = popA;
    res.popB = popB;
    if (popA == popB || win_size <= 0 || win_step <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const std::vector<std::string> want{popA, popB};
    const std::vector<std::pair<int, int>> pairs{{0, 1}};  // (A, B)
    std::vector<core::BpWindow> windows;
    FstWindowed fold;
    res.status = run_windowed_fold(geno, snp, ind, std::span<const std::string>(want), pairs,
                                   win_size, win_step, resources, filter, windows,
                                   res.kept_snp_ids, fold);
    if (res.status != Status::Ok) return res;
    res.precision_tag = fold.precision_tag;

    const std::size_t W = windows.size();
    res.chrom.resize(W);
    res.start.resize(W);
    res.end.resize(W);
    res.n_snp.resize(W);
    res.num.resize(W);
    res.den.resize(W);
    res.fst.resize(W);
    for (std::size_t k = 0; k < W; ++k) {
        res.chrom[k] = windows[k].chrom;
        res.start[k] = windows[k].start;
        res.end[k] = windows[k].end;
        const long n = windows[k].hi - windows[k].lo;
        res.n_snp[k] = n;
        const double num = fold.win_num[k];
        const double den = fold.win_den[k];
        res.num[k] = num;
        res.den[k] = den;
        // Ratio of averages; NaN for an empty window (n == 0) or an all-monomorphic window
        // (Σden == 0), mirroring allel's nansum(a)/(nansum(a)+nansum(b)+nansum(c)) = 0/0.
        res.fst[k] = (n > 0 && den != 0.0) ? (num / den) : std::nan("");
    }
    return res;
}

FstPbsResult run_fst_pbs(const std::string& geno, const std::string& snp, const std::string& ind,
                         const std::string& popA, const std::string& popB, const std::string& popC,
                         long win_size, long win_step, device::Resources& resources,
                         const FilterConfig& filter) {
    FstPbsResult res;
    res.precision_tag = Precision::Kind::Fp64;
    res.popA = popA;
    res.popB = popB;
    res.popC = popC;
    if (popA == popB || popA == popC || popB == popC || win_size <= 0 || win_step <= 0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    const std::vector<std::string> want{popA, popB, popC};
    // Pairs (as indices into want): 0=AB, 1=AC, 2=BC.
    const std::vector<std::pair<int, int>> pairs{{0, 1}, {0, 2}, {1, 2}};
    std::vector<core::BpWindow> windows;
    FstWindowed fold;
    res.status = run_windowed_fold(geno, snp, ind, std::span<const std::string>(want), pairs,
                                   win_size, win_step, resources, filter, windows,
                                   res.kept_snp_ids, fold);
    if (res.status != Status::Ok) return res;
    res.precision_tag = fold.precision_tag;

    const std::size_t W = windows.size();
    const long n_win = fold.n_win;
    res.chrom.resize(W);
    res.start.resize(W);
    res.end.resize(W);
    res.n_snp.resize(W);
    res.fst_ab.resize(W);
    res.fst_ac.resize(W);
    res.fst_bc.resize(W);
    res.pbs_a.resize(W);
    res.pbs_b.resize(W);
    res.pbs_c.resize(W);
    for (std::size_t k = 0; k < W; ++k) {
        res.chrom[k] = windows[k].chrom;
        res.start[k] = windows[k].start;
        res.end[k] = windows[k].end;
        const long n = windows[k].hi - windows[k].lo;
        res.n_snp[k] = n;

        const std::size_t kab = 0 * static_cast<std::size_t>(n_win) + k;
        const std::size_t kac = 1 * static_cast<std::size_t>(n_win) + k;
        const std::size_t kbc = 2 * static_cast<std::size_t>(n_win) + k;
        const double f_ab = (n > 0 && fold.win_den[kab] != 0.0)
                                ? (fold.win_num[kab] / fold.win_den[kab])
                                : std::nan("");
        const double f_ac = (n > 0 && fold.win_den[kac] != 0.0)
                                ? (fold.win_num[kac] / fold.win_den[kac])
                                : std::nan("");
        const double f_bc = (n > 0 && fold.win_den[kbc] != 0.0)
                                ? (fold.win_num[kbc] / fold.win_den[kbc])
                                : std::nan("");
        res.fst_ab[k] = f_ab;
        res.fst_ac[k] = f_ac;
        res.fst_bc[k] = f_bc;

        const double t_ab = pbs_branch_T(f_ab);
        const double t_ac = pbs_branch_T(f_ac);
        const double t_bc = pbs_branch_T(f_bc);
        res.pbs_a[k] = 0.5 * (t_ab + t_ac - t_bc);
        res.pbs_b[k] = 0.5 * (t_ab + t_bc - t_ac);
        res.pbs_c[k] = 0.5 * (t_ac + t_bc - t_ab);
    }
    return res;
}

}  // namespace steppe
