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
#include <limits>     // std::numeric_limits<int> (grid-dimension fits-int bound check)
#include <string>
#include <vector>

#include "core/internal/host_device.hpp"           // STEPPE_ASSERT (debug-only fail-fast)
#include "core/stats/genotype_front_end.hpp"      // C1 shared genotype decode front-end
#include "device/backend.hpp"                     // ComputeBackend, DecodeTileView, DatesMoments
#include "device/resources.hpp"                   // device::Resources

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
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
/// weightjack zero-weight filter: blocks whose SNP-count weight is below this are dropped
/// before the jackknife (statsubs.c weightjack jwt<1e-6 guard; jwt is a SNP count so any
/// positive block clears it — this only excludes the empty/degenerate leave-one-chrom blocks).
inline constexpr double kMinJackWeight = 1e-6;
/// corr_from denominator floor: a divide-by-zero guard added to sqrt(v11·v22) so a bin with
/// zero variance (v11·v22 == 0) returns corr 0 instead of NaN. Negligible vs any real variance
/// (DATES corr_from; matches the dates.c ddcorr 0-variance handling).
inline constexpr double kCorrDenomFloor = 1.0e-20;

// M5/M6 host primitives (ExpFit, linfit_2x2, fit_exp_decay, dates_repack_host) moved to the
// shared CUDA-FREE header core/internal/dates_fit.hpp so the CpuBackend reference oracle (in
// steppe_device) reuses the SAME definitions WITHOUT steppe_device depending on steppe_core
// (the one-way dependency: core -> device via ComputeBackend). The exp fit + the repack now
// run through the be.dates_fit / be.dates_repack backend seams (device on the CUDA path).

/// The DATES weighted block jackknife (statsubs.c weightjack/weightjackx). `mean` is the
/// full-data date; jmean[k] the leave-one-chrom date; jwt[k] the SNP-count weight (total -
/// count[chrom]). Returns est (the jackknife date) + sig (the SE). Mirrors the dstat.cpp
/// jackknife FAMILY but with the standard weightjack algebra (DATES dowtjack -> weightjack).
void weight_jack(const std::vector<double>& jmean, const std::vector<double>& jwt, double mean,
                 double& est, double& sig) {
    est = std::nan(""); sig = std::nan("");
    // drop zero-weight / non-finite blocks (weightjack's jwt<kMinJackWeight filter).
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

    // ---- 1. DECODE FRONT-END (C1 shared helper; mirrors dstat/qpfstats) ----------------
    // One core helper opens the GenoReader, reads the Explicit{target,source1,source2}
    // IndPartition + the SnpTable, and reads the canonical tile (M-FR-2 format dispatch).
    // `be` is the primary-GPU backend (forwarded to the non-TGENO transpose, reused below for
    // the decode). DATES diverges below into the per-SNP source-freq + autosome keep.
    const std::vector<std::string> dates_pops{target, source1, source2};  // the 3 pops DATES needs.
    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
    const core::GenotypeFrontEnd fe =
        core::read_genotype_front_end(geno, snp, ind, dates_pops, be);
    const io::SnpTable& snptab = fe.snptab;
    const io::GenotypeTile& tile = fe.tile;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) { res.status = Status::InvalidConfig; return res; }

    // Source allele frequencies via decode_af (forced diploid — DATES plain ref/an).
    std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);
    // `be` was bound above (the tile read dispatch); reused here for the decode.
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
        if (chr < kAutosomeChromMin || chr > kAutosomeChromMax) continue;  // AT2 auto_only: autosomes 1..22.
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
    // M5 — ON-DEVICE GATHER (host-compute audit): the dates.cpp host bit-shuffle hot loop
    // (O(n_target × M_kept)) is gone — the bound backend runs the repack (the CUDA backend
    // a device gather; the CpuBackend the bit-exact host oracle). INTEGER/BIT-EXACT, so
    // tgt_packed is bit-identical to the prior host repack ⇒ dates_curve moments unchanged.
    const std::size_t kept_bpr = io::packed_bytes(static_cast<std::size_t>(M_kept));
    std::vector<std::uint8_t> tgt_packed(static_cast<std::size_t>(n_target) * kept_bpr, 0);
    const std::uint8_t* tgt_src =
        tile.packed.data() + tgt_begin * tile.bytes_per_record;
    be.dates_repack(tgt_src, tile.bytes_per_record, kept_src.data(), M_kept, n_target,
                    kept_bpr, tgt_packed.data());

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
    // Round in `long` (std::lround returns long int) and bound-check each grid dimension
    // FITS int BEFORE the narrowing cast — a pathological maxdis/binsize/qbin could otherwise
    // overflow the int these dims index. Behavior-neutral: opts are validated at the entry
    // (the binsize/qbin/maxdis > 0 guard) and the assert compiles out under NDEBUG, so the
    // narrowed values are identical to the prior single-cast form.
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
        return v12 / std::sqrt(v11 * v22 + kCorrDenomFloor);
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
    // Single-source the DATES bin-center distance convention (dates.c dumpit/fitexp): bin s has
    // center distance (s+1)·binsize in MORGANS. The curve emit multiplies by kCentimorgansPerMorgan
    // for cM; the fit-window edge test stays in Morgans. Identical FP expression at both sites, so
    // the curve axis and the fit window cannot drift apart.
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

    // ---- 7. THE EXP FIT over the fit window + per-chrom LOO + the weighted jackknife ----
    // Fit window: bins with center distance >= lovalfit (cM) and < maxdis. DATES bin k center
    // distance = (k+1)·binsize. lo edge in MORGANS:
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

    // M6 — BATCHED EXP FIT (host-compute audit): the n_chrom+1 single-exponential fits (the
    // full-data fit + the per-chrom leave-one-out fits) run as ONE batched dates_fit call —
    // the CUDA backend runs one thread per curve (the 4000-grid + ternary refine + 2×2 FP64
    // normal solve on device), the CpuBackend the bit-exact host oracle. The host no longer
    // runs the 4000×win_len×(n_chrom+1) fit arithmetic. DATES date golden is the loose 2%
    // tier; the inner normal-eq accumulators are the FP64 cancellation carve-out on device.
    // All curves share ONE window (same n_emit / loval / maxdis), so the batch is dense.
    const std::vector<double> win_full = windowed(full_curve);
    const int win_len = static_cast<int>(win_full.size());
    const int n_curves = n_chrom + 1;  // [0] = full, [1..n_chrom] = LOO drop chrom kc.
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
