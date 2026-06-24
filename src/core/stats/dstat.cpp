// src/core/stats/dstat.cpp — the genotype-path NORMALIZED-D entry (run_dstat; qpDstat Part B).
//
// run_dstat is the genotype-reading SIBLING of run_f4 / run_f4ratio: it does NOT read the f2
// cache. It REUSES the extract-f2 decode FRONT-END (the io reader + decode_af [per-SNP Q/V/N]
// + assign_blocks [from genpos]) and DIVERGES at S2 into the per-SNP D kernel
// (ComputeBackend::dstat_block_reduce, the .cu) + the num/den block-jackknife (host-pure
// here, the f4ratio.cpp ratio-jackknife FAMILY). Pinned to the AT2 qpdstat_geno golden
// (allsnps=TRUE, f4mode=FALSE, blgsize=0.05) — docs/research/dates-genotype-stat-seam.md (i).
//
// THE THREE PARITY PINS (proven on box5090):
//  (1) ALLELE FREQ — AT2 uses PLAIN ref/an/2 (NO pseudo-haploid adjustment). decode_af with
//      sample_ploidy FORCED to all-diploid (=2) gives Q = Σcode/(2*an) == AT2 gmat_to_aftable
//      EXACTLY. We FORCE diploid here (NOT the extract-f2 Auto per-sample detect, which flips
//      the sign of near-zero D — this set has mixed PH/diploid samples).
//  (2) BLOCKS — assign_blocks (SNP-anchored walk, per-chrom reset, Morgans) == AT2
//      get_block_lengths byte-identical, over the AUTOSOME-filtered set (AT2 auto_only).
//  (3) SNP MASK — allsnps=TRUE per-(block,quadruple) finiteness (V==1 for all 4 pops at that
//      SNP). NO maxmiss, NO MAF, NO drop-mono. autosomes_only ON.
//
// THE NUM/DEN BLOCK-JACKKNIFE (AT2 est_to_loo_dat + jack_dat_stats; the f4ratio.cpp shape):
// per quadruple k, over survivor blocks (cnt[k,b]>0): est_num/est_den per block; per-type
// est_to_loo -> loo_num_b/loo_den_b; R_b = loo_num_b/loo_den_b; tot = weighted.mean(R_b,
// 1-cnt_b/Σcnt); est = mean(tot-R_b)*nb + weighted.mean(R_b,cnt); h_b = Σcnt/cnt_b;
// xtau_b = (h_b*tot-(h_b-1)*R_b-est)^2/(h_b-1); var = mean(xtau_b). D = est, se = sqrt(var),
// z = D/se, p = f4_two_sided_p(z) (== AT2 ztop). REUSES the f4-ratio jackknife shape but with
// per-(block,quadruple) cnt weights (NOT a shared block_size) and NO setmiss (pure cnt>0
// survivor mask). All accumulation in long double (the §12 num/den cancellation).
//
// CUDA-FREE: the per-SNP D reduction routes through the ComputeBackend seam (CpuBackend
// oracle / CudaBackend kernel). Domain outcomes (a degenerate quadruple) are a per-row NaN
// sentinel, never an exception (architecture.md §10).

#include "steppe/dstat.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // assign_blocks, BlockPartition
#include "device/backend.hpp"                     // ComputeBackend, DecodeTileView, DecodeResult
#include "device/resources.hpp"                   // device::Resources (the injected backend bundle)
#include "steppe/f4.hpp"                          // f4_two_sided_p (REUSED — AT2 ztop == erfc(|z|/sqrt2))

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe {

namespace {

/// Forced-diploid ploidy (the AT2 plain ref/an/2 pin; PARITY PIN (1)). NOT the extract-f2
/// Auto per-sample detection — the genotype D path diverges from the f2 path's
/// adjust_pseudohaploid convention and forces diploid.
inline constexpr int kPloidyDiploid = 2;

/// The single-entry primary GPU index (mirrors f4.cpp/f4ratio.cpp's kPrimaryGpu).
inline constexpr std::size_t kPrimaryGpu = 0;

/// One quadruple's num/den block-jackknife (AT2 est_to_loo_dat + jack_dat_stats; the
/// f4ratio.cpp ratio-jackknife FAMILY). `numsum`/`densum`/`cnt` are the row-major [N × nb]
/// per-(quadruple, block) reductions; `k` is this quadruple's row. Writes D (=est), se, z.
/// Survivor blocks are cnt[k,b] > 0 (NO setmiss). All accumulation in long double.
void dstat_jackknife(const double* numsum, const double* densum, const double* cnt,
                     int k, int n_block, double& D, double& se, double& z) {
    // ---- survivor pass: per-block est_num/est_den + Σcnt + survivor count ----
    long double sum_cnt = 0.0L;
    int nb_surv = 0;
    for (int b = 0; b < n_block; ++b) {
        const std::size_t o = static_cast<std::size_t>(k) * static_cast<std::size_t>(n_block) +
                              static_cast<std::size_t>(b);
        if (cnt[o] > 0.0) { sum_cnt += static_cast<long double>(cnt[o]); ++nb_surv; }
    }
    if (nb_surv <= 1 || sum_cnt <= 0.0L) {
        D = std::nan(""); se = std::nan(""); z = std::nan("");
        return;
    }
    const double nb = static_cast<double>(nb_surv);

    // ---- est_to_loo per type (AT2 est_to_loo_dat): tot_t = weighted.mean(est_t, cnt) ----
    // tot_num/tot_den are the cnt-weighted means of the per-block est; the LOO replicate of
    // block b is (tot_t - est_t_b*rel_bl)/(1-rel_bl), rel_bl = cnt_b/Σcnt.
    long double tot_num_w = 0.0L, tot_den_w = 0.0L;  // Σ est_t_b * cnt_b
    for (int b = 0; b < n_block; ++b) {
        const std::size_t o = static_cast<std::size_t>(k) * static_cast<std::size_t>(n_block) +
                              static_cast<std::size_t>(b);
        if (cnt[o] <= 0.0) continue;
        const long double est_num = static_cast<long double>(numsum[o]) /
                                    static_cast<long double>(cnt[o]);
        const long double est_den = static_cast<long double>(densum[o]) /
                                    static_cast<long double>(cnt[o]);
        tot_num_w += est_num * static_cast<long double>(cnt[o]);
        tot_den_w += est_den * static_cast<long double>(cnt[o]);
    }
    const long double tot_num = tot_num_w / sum_cnt;  // weighted.mean(est_num, cnt)
    const long double tot_den = tot_den_w / sum_cnt;  // weighted.mean(est_den, cnt)

    // ---- per-block R_b = loo_num_b/loo_den_b + the jack_dat_stats accumulators ----
    // tot   = weighted.mean(R_b, 1 - cnt_b/Σcnt)
    // est   = mean(tot - R_b)*nb + weighted.mean(R_b, cnt)
    long double tot_w_num = 0.0L, tot_w_den = 0.0L;  // Σ R_b*(1-rel), Σ(1-rel)  -> tot
    long double diffsum = 0.0L;                       // Σ (placeholder) — needs tot first; two-pass
    long double wmean_R_num = 0.0L;                   // Σ R_b * cnt_b  -> weighted.mean(R_b, cnt)
    std::vector<double> Rb(static_cast<std::size_t>(n_block), std::nan(""));
    for (int b = 0; b < n_block; ++b) {
        const std::size_t o = static_cast<std::size_t>(k) * static_cast<std::size_t>(n_block) +
                              static_cast<std::size_t>(b);
        if (cnt[o] <= 0.0) continue;
        const long double est_num = static_cast<long double>(numsum[o]) /
                                    static_cast<long double>(cnt[o]);
        const long double est_den = static_cast<long double>(densum[o]) /
                                    static_cast<long double>(cnt[o]);
        const long double rel = static_cast<long double>(cnt[o]) / sum_cnt;  // rel_bl
        const long double loo_num = (tot_num - est_num * rel) / (1.0L - rel);
        const long double loo_den = (tot_den - est_den * rel) / (1.0L - rel);
        const long double R = loo_num / loo_den;
        Rb[static_cast<std::size_t>(b)] = static_cast<double>(R);
        const long double w = 1.0L - rel;                 // weight for tot
        tot_w_num += R * w;
        tot_w_den += w;
        wmean_R_num += R * static_cast<long double>(cnt[o]);
    }
    const long double tot = tot_w_num / tot_w_den;        // weighted.mean(R_b, 1-cnt/Σcnt)
    for (int b = 0; b < n_block; ++b) {
        const std::size_t o = static_cast<std::size_t>(k) * static_cast<std::size_t>(n_block) +
                              static_cast<std::size_t>(b);
        if (cnt[o] <= 0.0) continue;
        diffsum += tot - static_cast<long double>(Rb[static_cast<std::size_t>(b)]);
    }
    const long double est = (diffsum / static_cast<long double>(nb_surv)) *
                                static_cast<long double>(nb) +
                            wmean_R_num / sum_cnt;          // mean(tot-R)*nb + wmean(R,cnt)

    // ---- xtau variance (AT2 jack_dat_stats): h_b = Σcnt/cnt_b;
    //      xtau_b = (h_b*tot - (h_b-1)*R_b - est)^2 / (h_b-1); var = mean(xtau_b). ----
    long double var_acc = 0.0L;
    for (int b = 0; b < n_block; ++b) {
        const std::size_t o = static_cast<std::size_t>(k) * static_cast<std::size_t>(n_block) +
                              static_cast<std::size_t>(b);
        if (cnt[o] <= 0.0) continue;
        const long double h = sum_cnt / static_cast<long double>(cnt[o]);
        const long double R = static_cast<long double>(Rb[static_cast<std::size_t>(b)]);
        const long double tau = h * tot - (h - 1.0L) * R - est;
        var_acc += (tau * tau) / (h - 1.0L);
    }
    const double var = static_cast<double>(var_acc / static_cast<long double>(nb_surv));

    D = static_cast<double>(est);
    se = (var > 0.0) ? std::sqrt(var) : std::nan("");
    z = D / se;
}

}  // namespace

// ---- Public entry (include/steppe/dstat.hpp) ----------------------------------------
DstatResult run_dstat(const std::string& geno, const std::string& snp, const std::string& ind,
                      std::span<const std::string> pop_union,
                      std::span<const std::array<int, 4>> quadruples, double blgsize_morgans,
                      device::Resources& resources) {
    DstatResult res;
    res.precision_tag = Precision::Kind::Fp64;  // long-double host jackknife; native FP64.

    const int N = static_cast<int>(quadruples.size());
    if (N <= 0) { res.status = Status::Ok; return res; }

    // Echo the quadruple P-axis indices (the emitter/binding label rows) + the flat 4*N table.
    res.p1.reserve(static_cast<std::size_t>(N));
    res.p2.reserve(static_cast<std::size_t>(N));
    res.p3.reserve(static_cast<std::size_t>(N));
    res.p4.reserve(static_cast<std::size_t>(N));
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(N) * 4);
    for (const std::array<int, 4>& q : quadruples) {
        res.p1.push_back(q[0]); res.p2.push_back(q[1]);
        res.p3.push_back(q[2]); res.p4.push_back(q[3]);
        flat.push_back(q[0]); flat.push_back(q[1]); flat.push_back(q[2]); flat.push_back(q[3]);
    }

    // ---- 1. DECODE FRONT-END (REUSE — mirrors cmd_extract_f2.cpp:157-356) -----------
    // THE P-AXIS CONTRACT (dstat.hpp): run_dstat reads ONLY the `pop_union` populations (AT2
    // indvec — read_ind(Explicit{pop_union}), NOT the whole 27594-ind prefix), so a 4-pop D
    // over a giant prefix decodes a tiny P. The P axis is that Explicit partition (sorted ASC
    // by label); the caller resolved the quadruple indices against THIS same order (the app
    // builds a PopResolver over read_ind(Explicit{pop_union}) — identical ordering), so every
    // index in `quadruples` is a valid row of the decoded Q/V. An io fault (missing/unreadable
    // file) PROPAGATES as an exception; the app try/catch maps it to a nonzero IoError exit
    // (mirroring cmd_f4's device-error catch). Domain outcomes (no survivor blocks) ride on
    // the per-row NaN sentinel below, never an exception.
    io::GenoReader reader(geno);
    const std::size_t n_present = reader.records_present();
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::Explicit;
    sel.labels.assign(pop_union.begin(), pop_union.end());  // the AT2 indvec (only these pops).
    const io::IndPartition part = io::read_ind(ind, sel, n_present);
    const io::SnpTable snptab = io::read_snp(snp, SIZE_MAX);
    const std::size_t M0 = std::min(reader.header().n_snp, snptab.count);
    const io::GenotypeTile tile = reader.read_tile(part, 0, M0);

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) {
        res.est.assign(static_cast<std::size_t>(N), std::nan(""));
        res.se.assign(static_cast<std::size_t>(N), std::nan(""));
        res.z.assign(static_cast<std::size_t>(N), std::nan(""));
        res.p.assign(static_cast<std::size_t>(N), std::nan(""));
        res.status = Status::Ok;
        return res;
    }

    // PARITY PIN (1): FORCED DIPLOID ploidy (AT2 plain ref/an/2; NOT extract-f2 Auto).
    std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);

    // PARITY PIN (3) — AUTOSOMES ONLY (AT2 auto_only default; chr 1..22). allsnps=TRUE
    // applies NO maxmiss/MAF/drop-mono, so the ONLY SNP filter is the autosome keep + the
    // per-(block,quadruple) finiteness mask (applied inside the D kernel via V). We subset
    // the SNP axis (Q/V/chrom/genpos) to the autosomes in LOCKSTEP after the decode.
    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;

    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.data();  // forced diploid (the AT2 plain /2 pin).
    view.ploidy = kPloidyDiploid;
    // ---- 1. Decode + the AUTOSOME keep + the lockstep Q/V subset (PARITY PIN (3); chr 1..22).
    // DEVICE-RESIDENT seam (host-compute audit C1/C2/M3/M4 cure): on the CUDA backend the decode
    // → autosome keep-mask → CUB/scan-gather Q/V compaction all run ON-DEVICE; the resident
    // compacted Q/V + the small kept chrom/genpos escape in `ddr` (NO ~1.1GB Q/V/N D2H, NO host
    // filter loop, NO Q/V H2D re-upload). The CpuBackend (device_count==0, the parity oracle)
    // keeps the host path. BOTH produce the IDENTICAL kept SET, kept ORDER, and chrom_kept/
    // genpos_kept → identical assign_blocks/golden. (DATES seam: per-SNP genpos retained, unused by D.)
    const bool resident = (be.capabilities().device_count > 0);
    steppe::device::DeviceDecodeResult ddr;
    std::vector<double> Qk, Vk;
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    if (resident) {
        ddr = be.decode_af_compact_autosome(
            view, std::span<const int>(snptab.chrom.data(), static_cast<std::size_t>(M)),
            std::span<const double>(snptab.genpos_morgans.data(), static_cast<std::size_t>(M)),
            kAutosomeChromMin, kAutosomeChromMax);
        chrom_kept = ddr.chrom_kept;
        genpos_kept = ddr.genpos_kept;
    } else {
        const DecodeResult dec = be.decode_af(view);
        Qk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M));
        Vk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M));
        chrom_kept.reserve(static_cast<std::size_t>(M));
        genpos_kept.reserve(static_cast<std::size_t>(M));
        for (long s = 0; s < M; ++s) {
            const int chr = snptab.chrom[static_cast<std::size_t>(s)];
            if (chr < 1 || chr > 22) continue;  // AT2 auto_only: autosomes 1..22.
            const std::size_t src = static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
            for (int p = 0; p < P; ++p) {
                Qk.push_back(dec.q[src + static_cast<std::size_t>(p)]);
                Vk.push_back(dec.v[src + static_cast<std::size_t>(p)]);
            }
            chrom_kept.push_back(chr);
            genpos_kept.push_back(snptab.genpos_morgans[static_cast<std::size_t>(s)]);
        }
    }
    const long M_kept = static_cast<long>(chrom_kept.size());
    if (M_kept <= 0) {
        res.est.assign(static_cast<std::size_t>(N), std::nan(""));
        res.se.assign(static_cast<std::size_t>(N), std::nan(""));
        res.z.assign(static_cast<std::size_t>(N), std::nan(""));
        res.p.assign(static_cast<std::size_t>(N), std::nan(""));
        res.status = Status::Ok;
        return res;
    }

    // ---- 2. assign_blocks over the KEPT (autosome) SNP axis (PARITY PIN (2)) ---------
    const core::BlockPartition partition = core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans);
    const int n_block = partition.n_block;
    if (n_block <= 0) {
        res.est.assign(static_cast<std::size_t>(N), std::nan(""));
        res.se.assign(static_cast<std::size_t>(N), std::nan(""));
        res.z.assign(static_cast<std::size_t>(N), std::nan(""));
        res.p.assign(static_cast<std::size_t>(N), std::nan(""));
        res.status = Status::Ok;
        return res;
    }

    // ---- 3. The per-SNP D kernel (the S2 divergence) — per (quadruple, block) num/den/cnt
    // ON THE GPU (device-resident, batched over N), the f2-cache NEVER touched. Outputs are
    // tiny [N × n_block] row-major.
    const std::size_t nb_out = static_cast<std::size_t>(N) * static_cast<std::size_t>(n_block);
    std::vector<double> numsum(nb_out, 0.0), densum(nb_out, 0.0), cnt(nb_out, 0.0);
    if (resident)
        be.dstat_block_reduce(ddr, partition.block_id.data(), n_block,
                              std::span<const int>(flat), numsum.data(), densum.data(),
                              cnt.data());
    else
        be.dstat_block_reduce(Qk.data(), Vk.data(), P, M_kept, partition.block_id.data(),
                              n_block, std::span<const int>(flat), numsum.data(),
                              densum.data(), cnt.data());

    // ---- 4. The num/den block-jackknife per quadruple (host-pure; the f4ratio FAMILY) --
    res.est.assign(static_cast<std::size_t>(N), 0.0);
    res.se.assign(static_cast<std::size_t>(N), 0.0);
    res.z.assign(static_cast<std::size_t>(N), 0.0);
    res.p.assign(static_cast<std::size_t>(N), 0.0);
    for (int k = 0; k < N; ++k) {
        double D = 0.0, se = 0.0, z = 0.0;
        dstat_jackknife(numsum.data(), densum.data(), cnt.data(), k, n_block, D, se, z);
        res.est[static_cast<std::size_t>(k)] = D;
        res.se[static_cast<std::size_t>(k)] = se;
        res.z[static_cast<std::size_t>(k)] = z;
        res.p[static_cast<std::size_t>(k)] = f4_two_sided_p(z);  // == AT2 ztop.
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
