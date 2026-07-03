// src/core/stats/dates.cpp — run_dates, the admixture-dating tool.
//
// Estimates how many generations ago an admixed target formed from two source
// populations, plus a standard error and the raw decay curve. Reuses the shared
// genotype decode front-end, then diverges into an FFT autocorrelation LD engine
// that never touches the f2 cache: the covariance over ~10^12 position pairs is
// obtained as an FFT autocorrelation, so the host does only tiny work — no O(M^2)
// SNP-pair loop. The FFT engine, weight/residual, and jackknife run in native FP64.
//
// Reference: docs/reference/src_core_stats_dates.cpp.md

#include "steppe/dates.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/internal/host_device.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "device/backend.hpp"
#include "device/resources.hpp"

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"

namespace steppe {

namespace {

// Named constants — reference §2
inline constexpr std::size_t kPrimaryGpu = 0;
inline constexpr int kPloidyDiploid = 2;
inline constexpr double kInterChromGapMorgans = 5.0;
inline constexpr double kMinJackWeight = 1e-6;
inline constexpr double kCorrDenomFloor = 1.0e-20;

// The weighted block jackknife — reference §3
void weight_jack(const std::vector<double>& jmean, const std::vector<double>& jwt, double mean,
                 double& est, double& sig) {
    est = std::nan(""); sig = std::nan("");
    std::vector<double> m, w;
    for (std::size_t k = 0; k < jmean.size(); ++k) {
        if (jwt[k] < kMinJackWeight) continue;
        if (!std::isfinite(jmean[k])) continue;
        m.push_back(jmean[k]); w.push_back(jwt[k]);
    }
    const int g = static_cast<int>(m.size());
    if (g <= 1) return;
    long double yn = 0.0L;
    for (double x : w) yn += static_cast<long double>(x);
    long double tdiff_sum = 0.0L, wdot = 0.0L;
    for (int k = 0; k < g; ++k) {
        tdiff_sum += static_cast<long double>(mean) - static_cast<long double>(m[static_cast<std::size_t>(k)]);
        wdot += static_cast<long double>(w[static_cast<std::size_t>(k)]) *
                static_cast<long double>(m[static_cast<std::size_t>(k)]);
    }
    const long double jackest = tdiff_sum + wdot / yn;
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

// Resolve a population label to its axis index — reference §3
int find_pop(const std::vector<std::string>& labels, const std::string& name) {
    for (std::size_t i = 0; i < labels.size(); ++i)
        if (labels[i] == name) return static_cast<int>(i);
    return -1;
}

}  // namespace

// run_dates: the pipeline — reference §4
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

    const std::vector<std::string> dates_pops{target, source1, source2};
    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
    const core::GenotypeFrontEnd fe =
        core::read_genotype_front_end(geno, snp, ind, dates_pops, be);
    const io::SnpTable& snptab = fe.snptab;
    const io::GenotypeTile& tile = fe.tile;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) { res.status = Status::InvalidConfig; return res; }

    std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);
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

    const int p_tgt = find_pop(tile.pop_labels, target);
    const int p_s1 = find_pop(tile.pop_labels, source1);
    const int p_s2 = find_pop(tile.pop_labels, source2);
    if (p_tgt < 0 || p_s1 < 0 || p_s2 < 0) { res.status = Status::InvalidConfig; return res; }

    std::vector<double> s1_freq, s2_freq, s_valid, genpos_kept;
    std::vector<int> chrom_kept;
    s1_freq.reserve(static_cast<std::size_t>(M));
    s2_freq.reserve(static_cast<std::size_t>(M));
    s_valid.reserve(static_cast<std::size_t>(M));
    genpos_kept.reserve(static_cast<std::size_t>(M));
    chrom_kept.reserve(static_cast<std::size_t>(M));
    const std::size_t tgt_begin = tile.pop_offsets[static_cast<std::size_t>(p_tgt)];
    const std::size_t tgt_end = tile.pop_offsets[static_cast<std::size_t>(p_tgt) + 1];
    const int n_target = static_cast<int>(tgt_end - tgt_begin);
    if (n_target <= 0) { res.status = Status::InvalidConfig; return res; }

    std::vector<long> kept_src;
    kept_src.reserve(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        const int chr = snptab.chrom[static_cast<std::size_t>(s)];
        if (chr < kAutosomeChromMin || chr > kAutosomeChromMax) continue;
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

    const std::size_t kept_bpr = io::packed_bytes(static_cast<std::size_t>(M_kept));
    std::vector<std::uint8_t> tgt_packed(static_cast<std::size_t>(n_target) * kept_bpr, 0);
    const std::uint8_t* tgt_src =
        tile.packed.data() + tgt_begin * tile.bytes_per_record;
    be.dates_repack(tgt_src, tile.bytes_per_record, kept_src.data(), M_kept, n_target,
                    kept_bpr, tgt_packed.data());

    const double qb = opts.binsize_morgans / static_cast<double>(opts.qbin);
    std::vector<int> grid_cell(static_cast<std::size_t>(M_kept), 0);
    double ydis = 0.0, last_genpos = genpos_kept[0];
    int cur_chrom = chrom_kept[0];
    int max_cell = 0;
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

    const long n_bin_l = std::lround(opts.maxdis_morgans / opts.binsize_morgans);
    const long diffmax_l = std::lround(static_cast<double>(opts.qbin) *
                                       opts.maxdis_morgans / opts.binsize_morgans);
    STEPPE_ASSERT(n_bin_l <= static_cast<long>(std::numeric_limits<int>::max()),
                  "DATES n_bin grid dimension overflows int");
    STEPPE_ASSERT(diffmax_l <= static_cast<long>(std::numeric_limits<int>::max()),
                  "DATES diffmax grid dimension overflows int");
    const int n_bin = static_cast<int>(n_bin_l);
    const int diffmax = static_cast<int>(diffmax_l);
    if (n_bin <= 0 || diffmax <= 0) { res.status = Status::InvalidConfig; return res; }

    std::vector<int> tgt_ploidy(static_cast<std::size_t>(n_target), kPloidyDiploid);
    const Precision precision;
    DatesMoments mom = be.dates_curve(
        s1_freq.data(), s2_freq.data(), s_valid.data(), tgt_packed.data(), kept_bpr, n_target,
        tgt_ploidy.data(), grid_cell.data(), M_kept, chrom_first.data(), chrom_last.data(),
        n_chrom, numqbins, n_bin, diffmax, opts.binsize_morgans, opts.qbin, precision);

    const std::size_t nb = static_cast<std::size_t>(n_bin);
    auto at = [&](const std::vector<double>& a, int kc, int s) -> double {
        return a[static_cast<std::size_t>(kc) * nb + static_cast<std::size_t>(s)];
    };
    auto corr_from = [](double S0, double S11, double S12, double S22) -> double {
        if (S0 < 0.5) return 0.0;
        const double v11 = S11 / S0;
        const double v12 = S12 / S0;
        const double v22 = S22 / S0;
        return v12 / std::sqrt(v11 * v22 + kCorrDenomFloor);
    };

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

    const int n_emit = n_bin;
    auto bin_center_morgans = [&](int s) -> double {
        return (static_cast<double>(s) + 1.0) * opts.binsize_morgans;
    };
    res.curve_cm.reserve(static_cast<std::size_t>(n_emit));
    res.curve_corr.reserve(static_cast<std::size_t>(n_emit));
    for (int s = 0; s < n_emit; ++s) {
        const double cm = bin_center_morgans(s) * kCentimorgansPerMorgan;
        res.curve_cm.push_back(cm);
        res.curve_corr.push_back(full_curve[static_cast<std::size_t>(s)]);
    }

    const double loval_morgans = opts.lovalfit_cm / kCentimorgansPerMorgan;
    auto windowed = [&](const std::vector<double>& curve) -> std::vector<double> {
        std::vector<double> w;
        for (int s = 0; s < n_emit; ++s) {
            const double d = bin_center_morgans(s);
            if (d < loval_morgans) continue;
            if (d > opts.maxdis_morgans) break;
            w.push_back(curve[static_cast<std::size_t>(s)]);
        }
        return w;
    };

    const std::vector<double> win_full = windowed(full_curve);
    const int win_len = static_cast<int>(win_full.size());
    const int n_curves = n_chrom + 1;
    std::vector<double> curves(static_cast<std::size_t>(n_curves) *
                               static_cast<std::size_t>(win_len), 0.0);
    auto copy_curve = [&](int c, const std::vector<double>& w) {
        for (int j = 0; j < win_len && j < static_cast<int>(w.size()); ++j)
            curves[static_cast<std::size_t>(c) * static_cast<std::size_t>(win_len) +
                   static_cast<std::size_t>(j)] = w[static_cast<std::size_t>(j)];
    };
    copy_curve(0, win_full);
    for (int kc = 0; kc < n_chrom; ++kc) copy_curve(kc + 1, windowed(total_corr_curve(kc)));

    const std::vector<DatesExpFit> fits =
        be.dates_fit(curves.data(), win_len, n_curves, opts.binsize_morgans, opts.affine);

    const DatesExpFit& full_fit = fits.at(0);
    if (!full_fit.ok) { res.status = Status::RankDeficient; res.date_gen = std::nan(""); return res; }
    res.fit_error_sd = full_fit.error_sd;

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
        const DatesExpFit& f = fits.at(static_cast<std::size_t>(kc + 1));
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
