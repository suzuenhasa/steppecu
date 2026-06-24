// src/core/stats/dates.cpp — the DATES admixture-dating entry (run_dates).
//
// run_dates is a genotype-reading SIBLING of run_dstat: it REUSES the extract-f2 decode
// FRONT-END (io reader + decode_af + the per-SNP genpos seam) and DIVERGES into the cuFFT
// autocorrelation LD engine (ComputeBackend::dates_curve, the .cu / the CpuBackend FFT-free
// oracle), NEVER touching the f2 cache. Pinned to the DATES reference C source
// (github.com/priyamoorjani/DATES Version 750): dates.c:585-740 (per-sample weight/residual/
// scatter), :1280-1345 (ddadd FFT moments), :1352-1380 (ddcorr lag->bin), fixjcorr (the
// leave-one-chrom CORR subtraction), :896-945 (the corr curve), fitexp.c (the exp+affine
// fit), statsubs.c weightjack (the leave-one-chrom SE). The HOST work is TINY: setqbins +
// fixjcorr (a few thousand grid cells / ~1000 bins) + ~23 exp fits on the ~1000-point curve
// + the weighted jackknife — NEVER a host O(M²) SNP-pair loop (the central trap, designed out
// by the FFT reformulation: cov(lag)=IFFT(|FFT(grid)|²)).
//
// PRECISION: the cuFFT autocorrelation is native double; the weight/residual is the §12
// cancellation carve-out (wt = freq diff). EmulatedFp64 is acknowledged in the seam but the
// math here is native FP64 / long double.

#include "steppe/dates.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // (reused elsewhere; chrom segmentation here)
#include "device/backend.hpp"                     // ComputeBackend, DecodeTileView, DatesMoments
#include "device/resources.hpp"                   // device::Resources

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"  // kCentimorgansPerMorgan

namespace steppe {

namespace {

inline constexpr std::size_t kPrimaryGpu = 0;
/// Forced-diploid ploidy for the source allele-freq decode (DATES uses plain ref/an; the
/// per-sample dosage path uses g/2 directly, ploidy-independent — see dates.c getgtypes).
inline constexpr int kPloidyDiploid = 2;
/// DATES setqbins inter-chromosome gap: +5 Morgans between chromosomes so each chromosome's
/// fine-grid cells are disjoint and the per-chrom FFT segments never overlap (dates.c:1140).
inline constexpr double kInterChromGapMorgans = 5.0;

/// One DATES single-exponential + affine fit over a curve (DATES dates_expfit / fitexp.c).
/// Fits corr[i] ~ co0·v^i + c over the fit window [lo_idx, hi_idx), where i is the bin index
/// from the window start and v = exp(-lambda·step) (step = binsize). The (co0, c) are LINEAR
/// at fixed v (solve a 2×2 normal equation == dates_expfit scorit's regressit), so this is a
/// 1-D search over v + a local refine. Returns the date in generations and the residual sd.
/// step is in MORGANS (== binsize). lambda = log(2)/(halflife·step); halflife = the v-decay
/// half-life in bins; date_gen = lambda (DATES: mean(generations) = log(2)/(hlife·step)).
struct ExpFit {
    double date_gen = std::nan("");
    double error_sd = std::nan("");
    bool ok = false;
};

/// Solve the 2×2 (or 1×1 non-affine) linear least squares for (co0, c) given the basis
/// b[i] = v^i over the window, minimizing Σ (corr[i] - co0·b[i] - c)². Returns the residual
/// sum of squares (NOT divided by n) and writes co0/c. Affine adds the constant column.
double linfit_2x2(const std::vector<double>& y, double v, bool affine, double& co0, double& c) {
    const std::size_t n = y.size();
    // Normal equations for columns {v^i, (1)}: [Sbb Sb1; Sb1 S11] [co0; c] = [Sby; Sy].
    long double Sbb = 0.0L, Sb1 = 0.0L, S11 = 0.0L, Sby = 0.0L, Sy = 0.0L;
    double bi = 1.0;  // v^0
    for (std::size_t i = 0; i < n; ++i) {
        const long double b = static_cast<long double>(bi);
        const long double yi = static_cast<long double>(y[i]);
        Sbb += b * b;
        Sby += b * yi;
        if (affine) { Sb1 += b; S11 += 1.0L; Sy += yi; }
        bi *= v;
    }
    if (affine) {
        const long double det = Sbb * S11 - Sb1 * Sb1;
        if (det == 0.0L) { co0 = std::nan(""); c = std::nan(""); return std::nan(""); }
        co0 = static_cast<double>((Sby * S11 - Sb1 * Sy) / det);
        c = static_cast<double>((Sbb * Sy - Sb1 * Sby) / det);
    } else {
        co0 = (Sbb != 0.0L) ? static_cast<double>(Sby / Sbb) : std::nan("");
        c = 0.0;
    }
    // residual sum of squares
    long double rss = 0.0L;
    double bb = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double pred = co0 * bb + c;
        const long double r = static_cast<long double>(y[i]) - pred;
        rss += r * r;
        bb *= v;
    }
    return static_cast<double>(rss);
}

/// Fit A·exp(-lambda·d)+c over the windowed curve. `y` is the corr values over the fit window
/// (bin start at lo); `step` is the bin width in MORGANS. 1-D coarse-to-fine search over the
/// per-bin decay factor v∈(0,1) (== DATES fitexp multi-start + GSL refine, reduced to the
/// 1-exp case), with (co0,c) the inner 2×2 linear solve. date_gen = log(2)/(halflife·step),
/// halflife = -log(2)/log(v) bins -> lambda = -log(v)/step -> date_gen = lambda (generations).
ExpFit fit_exp_decay(const std::vector<double>& y, double step, bool affine) {
    ExpFit out;
    if (y.size() < 3 || step <= 0.0) return out;
    // Coarse grid over v in (eps, 1-eps), then golden-ish local refine on the best.
    auto score = [&](double v, double& co0, double& c) -> double {
        if (!(v > 0.0) || !(v < 1.0)) return std::nan("");
        return linfit_2x2(y, v, affine, co0, c);
    };
    double best_v = 0.5, best_rss = std::numeric_limits<double>::infinity(), best_co0 = 0.0, best_c = 0.0;
    const int coarse = 4000;
    for (int k = 1; k < coarse; ++k) {
        const double v = static_cast<double>(k) / static_cast<double>(coarse);
        double co0 = 0.0, c = 0.0;
        const double rss = score(v, co0, c);
        // require a DECAYING positive exponential (co0>0): the admixture-LD signal is positive
        // and decays; reject negative-amplitude / growing fits (DATES sorts to a decay).
        if (std::isnan(rss)) continue;
        if (co0 <= 0.0) continue;
        if (rss < best_rss) { best_rss = rss; best_v = v; best_co0 = co0; best_c = c; }
    }
    if (!std::isfinite(best_rss)) {
        // fall back: allow any-sign amplitude (degenerate curve)
        for (int k = 1; k < coarse; ++k) {
            const double v = static_cast<double>(k) / static_cast<double>(coarse);
            double co0 = 0.0, c = 0.0;
            const double rss = score(v, co0, c);
            if (std::isnan(rss)) continue;
            if (rss < best_rss) { best_rss = rss; best_v = v; best_co0 = co0; best_c = c; }
        }
        if (!std::isfinite(best_rss)) return out;
    }
    // Local refine (ternary search) around best_v for a tighter v.
    double lo = std::max(1e-9, best_v - 1.0 / static_cast<double>(coarse));
    double hi = std::min(1.0 - 1e-9, best_v + 1.0 / static_cast<double>(coarse));
    for (int it = 0; it < 200; ++it) {
        const double m1 = lo + (hi - lo) / 3.0;
        const double m2 = hi - (hi - lo) / 3.0;
        double c1a = 0.0, c1b = 0.0, c2a = 0.0, c2b = 0.0;
        const double r1 = score(m1, c1a, c1b);
        const double r2 = score(m2, c2a, c2b);
        const double rr1 = std::isnan(r1) ? std::numeric_limits<double>::infinity() : r1;
        const double rr2 = std::isnan(r2) ? std::numeric_limits<double>::infinity() : r2;
        if (rr1 < rr2) hi = m2; else lo = m1;
    }
    best_v = 0.5 * (lo + hi);
    best_rss = score(best_v, best_co0, best_c);
    if (std::isnan(best_rss) || !(best_v > 0.0) || !(best_v < 1.0)) return out;

    // v = exp(-lambda·step) per bin -> lambda = -log(v)/step (generations since admixture).
    const double lambda = -std::log(best_v) / step;
    out.date_gen = lambda;
    out.error_sd = std::sqrt(best_rss / static_cast<double>(y.size()));
    out.ok = std::isfinite(out.date_gen) && out.date_gen > 0.0;
    return out;
}

/// The DATES weighted block jackknife (statsubs.c weightjack/weightjackx). `mean` is the
/// full-data date; jmean[k] the leave-one-chrom date; jwt[k] the SNP-count weight (total -
/// count[chrom]). Returns est (the jackknife date) + sig (the SE). Mirrors the dstat.cpp
/// jackknife FAMILY but with the standard weightjack algebra (DATES dowtjack -> weightjack).
void weight_jack(const std::vector<double>& jmean, const std::vector<double>& jwt, double mean,
                 double& est, double& sig) {
    est = std::nan(""); sig = std::nan("");
    // drop zero-weight / non-finite blocks (weightjack's jwt<1e-6 filter).
    std::vector<double> m, w;
    for (std::size_t k = 0; k < jmean.size(); ++k) {
        if (jwt[k] < 1e-6) continue;
        if (!std::isfinite(jmean[k])) continue;
        m.push_back(jmean[k]); w.push_back(jwt[k]);
    }
    const int g = static_cast<int>(m.size());
    if (g <= 1) return;
    long double yn = 0.0L;
    for (double x : w) yn += static_cast<long double>(x);
    // jackest = Σ(mean - jmean[k]) + dot(jwt, jmean)/yn  (weightjackx equation 2)
    long double tdiff_sum = 0.0L, wdot = 0.0L;
    for (int k = 0; k < g; ++k) {
        tdiff_sum += static_cast<long double>(mean) - static_cast<long double>(m[static_cast<std::size_t>(k)]);
        wdot += static_cast<long double>(w[static_cast<std::size_t>(k)]) *
                static_cast<long double>(m[static_cast<std::size_t>(k)]);
    }
    const long double jackest = tdiff_sum + wdot / yn;
    // hh[k] = yn/jwt[k]; xtau[k] = hh·mean - (hh-1)·jmean - jackest; yvar = Σ xtau²/(hh-1)/g
    long double yvar = 0.0L;
    for (int k = 0; k < g; ++k) {
        const long double hh = yn / static_cast<long double>(w[static_cast<std::size_t>(k)]);
        const long double tau = hh * static_cast<long double>(mean) -
                                (hh - 1.0L) * static_cast<long double>(m[static_cast<std::size_t>(k)]) -
                                jackest;
        yvar += (tau * tau) / (hh - 1.0L);
    }
    yvar /= static_cast<long double>(g);
    est = static_cast<double>(jackest);
    sig = (yvar > 0.0L) ? static_cast<double>(std::sqrt(static_cast<double>(yvar))) : std::nan("");
}

/// Resolve a population label to its P-axis index in the decoded (sorted-ASC) partition.
int find_pop(const std::vector<std::string>& labels, const std::string& name) {
    for (std::size_t i = 0; i < labels.size(); ++i)
        if (labels[i] == name) return static_cast<int>(i);
    return -1;
}

}  // namespace

DatesResult run_dates(const std::string& geno, const std::string& snp, const std::string& ind,
                      const std::string& target, const std::string& source1,
                      const std::string& source2, const DatesOptions& opts,
                      device::Resources& resources) {
    DatesResult res;
    res.precision_tag = Precision::Kind::Fp64;

    if (opts.binsize_morgans <= 0.0 || opts.qbin <= 0 || opts.maxdis_morgans <= 0.0) {
        res.status = Status::InvalidConfig;
        return res;
    }

    // ---- 1. DECODE FRONT-END (REUSE — mirrors dstat.cpp:195-234) -----------------------
    io::GenoReader reader(geno);
    const std::size_t n_present = reader.records_present();
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::Explicit;
    sel.labels = {target, source1, source2};  // the three pops DATES needs.
    const io::IndPartition part = io::read_ind(ind, sel, n_present);
    const io::SnpTable snptab = io::read_snp(snp, SIZE_MAX);
    const std::size_t M0 = std::min(reader.header().n_snp, snptab.count);
    const io::GenotypeTile tile = reader.read_tile(part, 0, M0);

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) { res.status = Status::InvalidConfig; return res; }

    // Source allele frequencies via decode_af (forced diploid — DATES plain ref/an).
    std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);
    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.data();
    view.ploidy = kPloidyDiploid;
    const DecodeResult dec = be.decode_af(view);

    // Resolve the three pops against the decoded (sorted) partition order.
    const int p_tgt = find_pop(tile.pop_labels, target);
    const int p_s1 = find_pop(tile.pop_labels, source1);
    const int p_s2 = find_pop(tile.pop_labels, source2);
    if (p_tgt < 0 || p_s1 < 0 || p_s2 < 0) { res.status = Status::InvalidConfig; return res; }

    // ---- 2. AUTOSOME keep + per-SNP source freqs + validity, in FILE ORDER -------------
    // DATES processes SNPs in file order (the .snp is sorted chrom-then-genpos). We keep the
    // autosomes in lockstep: source freqs (w1/w2), the both-sources-valid mask, chrom, genpos.
    std::vector<double> s1_freq, s2_freq, s_valid, genpos_kept;
    std::vector<int> chrom_kept;
    s1_freq.reserve(static_cast<std::size_t>(M));
    s2_freq.reserve(static_cast<std::size_t>(M));
    s_valid.reserve(static_cast<std::size_t>(M));
    genpos_kept.reserve(static_cast<std::size_t>(M));
    chrom_kept.reserve(static_cast<std::size_t>(M));
    // Also subset the TARGET packed genotypes to the kept SNP axis: the target segment in the
    // tile holds, per target individual, the FULL SNP record; we re-pack the kept SNPs into a
    // dense per-target-individual record so grid_cell[s] indexes the kept axis.
    const std::size_t tgt_begin = tile.pop_offsets[static_cast<std::size_t>(p_tgt)];
    const std::size_t tgt_end = tile.pop_offsets[static_cast<std::size_t>(p_tgt) + 1];
    const int n_target = static_cast<int>(tgt_end - tgt_begin);
    if (n_target <= 0) { res.status = Status::InvalidConfig; return res; }

    // First pass: which SNPs survive (autosomes 1..22), recording the source freqs/validity.
    std::vector<long> kept_src;  // kept index -> original SNP index (file order)
    kept_src.reserve(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        const int chr = snptab.chrom[static_cast<std::size_t>(s)];
        if (chr < 1 || chr > 22) continue;
        const std::size_t col = static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
        const double v1 = dec.v[col + static_cast<std::size_t>(p_s1)];
        const double v2 = dec.v[col + static_cast<std::size_t>(p_s2)];
        s1_freq.push_back(dec.q[col + static_cast<std::size_t>(p_s1)]);
        s2_freq.push_back(dec.q[col + static_cast<std::size_t>(p_s2)]);
        s_valid.push_back((v1 != 0.0 && v2 != 0.0) ? 1.0 : 0.0);
        chrom_kept.push_back(chr);
        genpos_kept.push_back(snptab.genpos_morgans[static_cast<std::size_t>(s)]);
        kept_src.push_back(s);
    }
    const long M_kept = static_cast<long>(kept_src.size());
    if (M_kept <= 0) { res.status = Status::InvalidConfig; return res; }

    // Re-pack the target genotypes onto the kept SNP axis (dense per-individual record).
    const std::size_t kept_bpr = io::packed_bytes(static_cast<std::size_t>(M_kept));
    std::vector<std::uint8_t> tgt_packed(static_cast<std::size_t>(n_target) * kept_bpr, 0);
    for (int i = 0; i < n_target; ++i) {
        const std::uint8_t* src_rec =
            tile.packed.data() + (tgt_begin + static_cast<std::size_t>(i)) * tile.bytes_per_record;
        std::uint8_t* dst_rec = tgt_packed.data() + static_cast<std::size_t>(i) * kept_bpr;
        for (long ks = 0; ks < M_kept; ++ks) {
            const long s = kept_src[static_cast<std::size_t>(ks)];
            const std::size_t sb = static_cast<std::size_t>(s) / static_cast<std::size_t>(io::kCodesPerByte);
            const int sp = static_cast<int>(s % io::kCodesPerByte);
            const std::uint8_t code = io::code_in_byte(src_rec[sb], sp);
            const std::size_t db = static_cast<std::size_t>(ks) / static_cast<std::size_t>(io::kCodesPerByte);
            const int dp = static_cast<int>(ks % io::kCodesPerByte);
            const int shift = (io::kCodesPerByte - 1 - dp) * io::kBitsPerCode;
            dst_rec[db] = static_cast<std::uint8_t>(dst_rec[db] | (code << shift));
        }
    }

    // ---- 3. setqbins: the fine genetic-map grid cell per kept SNP (dates.c:1128-1160) --
    // Cumulative genpos across SNPs (file order) with a +5 Morgan gap between chromosomes;
    // cell = floor(ydis / qb), qb = binsize/qbin. Per-chrom first/last cell (slo/shi).
    const double qb = opts.binsize_morgans / static_cast<double>(opts.qbin);
    std::vector<int> grid_cell(static_cast<std::size_t>(M_kept), 0);
    double ydis = 0.0, last_genpos = genpos_kept[0];
    int cur_chrom = chrom_kept[0];
    int max_cell = 0;
    // present chromosomes in order of appearance + their cell extents.
    std::vector<int> chrom_present;
    std::vector<int> chrom_first, chrom_last;
    auto touch_chrom = [&](int chr, int cell) {
        if (chrom_present.empty() || chrom_present.back() != chr) {
            chrom_present.push_back(chr);
            chrom_first.push_back(cell);
            chrom_last.push_back(cell);
        } else {
            chrom_last.back() = std::max(chrom_last.back(), cell);
            chrom_first.back() = std::min(chrom_first.back(), cell);
        }
    };
    for (long ks = 0; ks < M_kept; ++ks) {
        const int chr = chrom_kept[static_cast<std::size_t>(ks)];
        const double gp = genpos_kept[static_cast<std::size_t>(ks)];
        if (ks == 0) {
            ydis = 0.0;
        } else if (chr != cur_chrom) {
            ydis += kInterChromGapMorgans;
            cur_chrom = chr;
        } else {
            ydis += (gp - last_genpos);
        }
        last_genpos = gp;
        const int cell = static_cast<int>(std::floor(ydis / qb));
        grid_cell[static_cast<std::size_t>(ks)] = cell;
        max_cell = std::max(max_cell, cell);
        touch_chrom(chr, cell);
    }
    const int numqbins = max_cell + 1;
    const int n_chrom = static_cast<int>(chrom_present.size());

    // ---- 4. The curve dimensions (dates.c: numbins = round(maxdis/binsize)+5; diffmax) -
    const int n_bin = static_cast<int>(std::lround(opts.maxdis_morgans / opts.binsize_morgans));
    const int diffmax = static_cast<int>(std::lround(static_cast<double>(opts.qbin) *
                                                     opts.maxdis_morgans / opts.binsize_morgans));
    if (n_bin <= 0 || diffmax <= 0) { res.status = Status::InvalidConfig; return res; }

    // ---- 5. THE cuFFT AUTOCORRELATION LD ENGINE (the S2 divergence; dates_curve) -------
    // Per-(chrom, bin) CORR sufficient statistics summed over EVERY target sample. The
    // ~10^12 SNP-pair object is NEVER formed (cov(lag) = IFFT(|FFT(grid)|²)). Native FP64.
    std::vector<int> tgt_ploidy(static_cast<std::size_t>(n_target), kPloidyDiploid);
    const Precision precision;  // default policy; the seam runs native FFT + native weight.
    DatesMoments mom = be.dates_curve(
        s1_freq.data(), s2_freq.data(), s_valid.data(), tgt_packed.data(), kept_bpr, n_target,
        tgt_ploidy.data(), grid_cell.data(), M_kept, chrom_first.data(), chrom_last.data(),
        n_chrom, numqbins, n_bin, diffmax, opts.binsize_morgans, opts.qbin, precision);

    // ---- 6. fixjcorr: total + leave-one-chrom CORR by SUBTRACTION (dates.c fixjcorr) ----
    // tot[s] = Σ_k chrom[k][s]; loo[k][s] = tot[s] - chrom[k][s]. We hold the six S-moments.
    const std::size_t nb = static_cast<std::size_t>(n_bin);
    auto at = [&](const std::vector<double>& a, int kc, int s) -> double {
        return a[static_cast<std::size_t>(kc) * nb + static_cast<std::size_t>(s)];
    };
    // The full-data corr curve (datacol 3): corr = (S12/S0)/sqrt((S11/S0)·(S22/S0)).
    auto corr_from = [](double S0, double S11, double S12, double S22) -> double {
        if (S0 < 0.5) return 0.0;
        const double v11 = S11 / S0;
        const double v12 = S12 / S0;
        const double v22 = S22 / S0;
        return v12 / std::sqrt(v11 * v22 + 1.0e-20);
    };

    // Build the total per-bin S-moments, and per-chrom LOO via subtraction.
    auto total_corr_curve = [&](int drop_chrom) -> std::vector<double> {
        std::vector<double> curve(nb, 0.0);
        for (int s = 0; s < n_bin; ++s) {
            double S0 = 0.0, S11 = 0.0, S12 = 0.0, S22 = 0.0;
            for (int kc = 0; kc < n_chrom; ++kc) {
                if (kc == drop_chrom) continue;
                S0 += at(mom.s0, kc, s); S11 += at(mom.s11, kc, s);
                S12 += at(mom.s12, kc, s); S22 += at(mom.s22, kc, s);
            }
            curve[static_cast<std::size_t>(s)] = corr_from(S0, S11, S12, S22);
        }
        return curve;
    };

    const std::vector<double> full_curve = total_corr_curve(/*drop=*/-1);

    // Surface the curve (cM vs corr). DATES allocates numbins = round(maxdis/binsize)+5 and
    // emits numbins-5 == round(maxdis/binsize) bins (dumpit len=numbins-5); our n_bin is
    // ALREADY round(maxdis/binsize) (the emitted count, the +5 pad is not allocated here), so
    // we emit ALL n_bin bins — bin k has center distance (k+1)·binsize, up to maxdis.
    const int n_emit = n_bin;
    res.curve_cm.reserve(static_cast<std::size_t>(n_emit));
    res.curve_corr.reserve(static_cast<std::size_t>(n_emit));
    for (int s = 0; s < n_emit; ++s) {
        const double cm = (static_cast<double>(s) + 1.0) * opts.binsize_morgans *
                          kCentimorgansPerMorgan;
        res.curve_cm.push_back(cm);
        res.curve_corr.push_back(full_curve[static_cast<std::size_t>(s)]);
    }

    // ---- 7. THE EXP FIT over the fit window + per-chrom LOO + the weighted jackknife ----
    // Fit window: bins with center distance >= lovalfit (cM) and < maxdis. DATES bin k center
    // distance = (k+1)·binsize. lo edge in MORGANS:
    const double loval_morgans = opts.lovalfit_cm / kCentimorgansPerMorgan;
    auto windowed = [&](const std::vector<double>& curve) -> std::vector<double> {
        std::vector<double> w;
        for (int s = 0; s < n_emit; ++s) {
            const double d = (static_cast<double>(s) + 1.0) * opts.binsize_morgans;
            if (d < loval_morgans) continue;
            if (d > opts.maxdis_morgans) break;
            w.push_back(curve[static_cast<std::size_t>(s)]);
        }
        return w;
    };

    const ExpFit full_fit = fit_exp_decay(windowed(full_curve), opts.binsize_morgans, opts.affine);
    if (!full_fit.ok) { res.status = Status::RankDeficient; res.date_gen = std::nan(""); return res; }
    res.fit_error_sd = full_fit.error_sd;

    // Per-chrom LOO date + SNP-count weight (DATES dates_wtjack: weight = total - count[chr]).
    // count[chr] = number of kept SNPs on that chromosome.
    std::vector<int> snp_count(static_cast<std::size_t>(n_chrom), 0);
    for (long ks = 0; ks < M_kept; ++ks) {
        const int chr = chrom_kept[static_cast<std::size_t>(ks)];
        for (int kc = 0; kc < n_chrom; ++kc)
            if (chrom_present[static_cast<std::size_t>(kc)] == chr) { ++snp_count[static_cast<std::size_t>(kc)]; break; }
    }
    long total_count = 0;
    for (int c : snp_count) total_count += c;

    std::vector<double> jmean(static_cast<std::size_t>(n_chrom), std::nan(""));
    std::vector<double> jwt(static_cast<std::size_t>(n_chrom), 0.0);
    for (int kc = 0; kc < n_chrom; ++kc) {
        const ExpFit f = fit_exp_decay(windowed(total_corr_curve(kc)), opts.binsize_morgans,
                                       opts.affine);
        jmean[static_cast<std::size_t>(kc)] = f.ok ? f.date_gen : std::nan("");
        jwt[static_cast<std::size_t>(kc)] =
            static_cast<double>(total_count - snp_count[static_cast<std::size_t>(kc)]);
    }
    res.loo_date_gen = jmean;
    res.loo_weight = jwt;

    double est = std::nan(""), sig = std::nan("");
    weight_jack(jmean, jwt, full_fit.date_gen, est, sig);
    res.date_gen = std::isfinite(est) ? est : full_fit.date_gen;
    res.se = sig;
    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
