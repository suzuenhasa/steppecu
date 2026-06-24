// src/core/stats/qpfstats.cpp — the genotype-path JOINT f2 SMOOTHER (run_qpfstats).
//
// qpfstats reproduces admixtools::qpfstats() (the R-ridge path; pinned to the AT2 R golden
// golden_qpfstats_geno). It is a GENOTYPE-PATH tool that REUSES the qpDstat-B genotype
// machinery verbatim (no new genotype/finiteness code):
//   FRONT-END (mirrors stats/dstat.cpp run_dstat): read_ind(Explicit{pops}) + read_snp +
//     read_tile + decode_af (FORCED diploid) + autosome keep + assign_blocks.
//   THE GENOTYPE-f4 NUMERATOR ENGINE: ComputeBackend::dstat_block_reduce
//     (src/device/cuda/dstat_kernel.cu, the allsnps=TRUE per-(item,block) finiteness
//     reduction) DRIVEN over the FULL f2∪f3∪f4 popcomb set (666 combos for n=9), batched
//     over BOTH the comb axis AND the n_block axis ON THE GPU. f4mode=TRUE ⇒ numerator-only
//     (the D denominator densum is computed by the kernel but IGNORED here).
//
// THE SMOOTHING REGRESSION (ComputeBackend::qpfstats_smooth, the on-device shared-factor
// batched least-squares; AT2 qpfstats_regression REFORMULATED — NO host per-block loop):
//   numer[comb,block] = numsum/cnt (the AT2 rowMeans, NaN where cnt==0).
//   x[comb,pair] = construct_fstat_matrix (±1/2 f4-identity coefficients; *2 pure-f2 row).
//   ymat = numer (col-major [npopcomb × nblock]); y = matrix_jackknife_est(numer,cnt).
//   A_shared = x'x + ridge·I (ridge=1e-5); b[:,blk] = solve(A_blk, x'·ymat[:,blk]);
//   bglob = solve(A_y, x'·y). The NaN-comb-row downdate + all-NaN→0 handled in the seam.
//
// THE OUTPUT (AT2 qpfstats scatter+recenter): scatter b → f2blocks[npop,npop,nblock]
// (off-diagonal pairs, symmetric); recenter f2blocks2 = f2blocks - f2(f2blocks)$est + bglob
// (the per-pair block-jackknife est of the smoothed tensor, block_lengths weights). The
// diagonal stays 0. THIS is the smoothed F2BlockTensor the downstream tools consume.
//
// THE THREE PARITY PINS (inherited from qpDstat-B, proven on box5090): forced diploid (AT2
// plain ref/an/2), assign_blocks == AT2 get_block_lengths, allsnps=TRUE per-(comb,block)
// finiteness (no maxmiss/MAF/drop-mono; autosomes_only ON).
//
// CUDA-FREE: the per-SNP numerator reduction + the smoothing solve route through the
// ComputeBackend seam (CpuBackend oracle / CudaBackend kernels). All host work is O(small):
// the popcomb/design build (data-independent), the recenter jackknife (per-pair over blocks),
// the scatter. NO host per-SNP / per-jackknife-block solve loop (the CPU-bound trap).

#include "steppe/qpfstats.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // assign_blocks, BlockPartition
#include "device/backend.hpp"                     // ComputeBackend, DecodeTileView, QpfstatsSmooth
#include "device/resources.hpp"                   // device::Resources

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe {

namespace {

/// Forced-diploid ploidy (the AT2 plain ref/an/2 pin; inherited from qpDstat-B). NOT the
/// extract-f2 Auto per-sample detection — qpfstats uses the genotype-f4 numerator engine.
inline constexpr int kPloidyDiploid = 2;

/// The single-entry primary GPU index (mirrors dstat.cpp's kPrimaryGpu).
inline constexpr std::size_t kPrimaryGpu = 0;

/// The AT2 internal ridge constant (qpfstats_regression default `ridge = 1e-5`).
inline constexpr double kRidge = 1e-5;

/// The basis pair index for an off-diagonal pop pair (i,j), i<j: the row-major
/// upper-triangle enumeration over the n(n-1)/2 = C(npop,2) pairs — EXACTLY AT2
/// construct_fstat_matrix's symmetric `indmat` order (verified numerically on box5090:
/// indmat[i,j] for i<j == this row-major upper-tri index, 0-based). Symmetric in (i,j).
[[nodiscard]] inline int pair_index(int i, int j, int npop) {
    if (i > j) std::swap(i, j);
    // sum_{r=0}^{i-1} (npop-1-r) + (j-i-1) = i*npop - i*(i+1)/2 + (j-i-1)
    return i * npop - (i * (i + 1)) / 2 + (j - i - 1);
}

/// One population combination (p1,p2,p3,p4) over the P-axis indices. The f-stat numerator
/// is (a-b)(c-d) with a=Q[p1],b=Q[p2],c=Q[p3],d=Q[p4] — the SAME formula for f2/f3/f4
/// (index aliasing distinguishes them: f2=(A,B,A,B), f3=(A,B,A,D), f4=(A,B,C,D)).
struct PopComb { int p1, p2, p3, p4; };

/// Build the AT2 qpfstats popcomb set f2 ∪ f3 ∪ f4 over the npop SORTED pops, and the
/// matching construct_fstat_matrix design `x` (COLUMN-MAJOR [npopcomb × npairs]).
///
/// VERIFIED vs AT2 source (admixtools 2.0.10, box5090): for npop=9 ⇒ 36 + 252 + 378 = 666.
///   f2: (p1,p2,p1,p2) for p1<p2                       — C(npop,2) combos.
///   f3: fstat_get_popcombs(fnum=3) %>% transmute(pop1,pop2,pop4=pop3,pop3=pop1) — the
///       3-rotation expansion of C(npop,3) triples, then mapped to (pop1,pop2,pop1,pop4)
///       [pop3:=pop1, pop4:=the old pop3]. C(npop,3)*3 combos.
///   f4: fstat_get_popcombs(fnum=4) — the 3-rotation expansion of C(npop,4) quads
///       [(1,2,3,4),(1,3,2,4),(1,4,2,3)]. C(npop,4)*3 combos.
/// The design row (construct_fstat_matrix): coeffs +1 on pair(p1,p4), +1 on pair(p2,p3),
/// -1 on pair(p1,p3), -1 on pair(p2,p4); ×2 if it is a pure-f2 row (p1==p3 && p2==p4);
/// then /2 ALL. Pure-f2 (A,B,A,B): pair(A,B) gets (1+1)*2/2 = 2 ... wait it collapses to
/// (1+1-1·[pair(A,A) invalid]) — handled by accumulating coefficients into the pair index,
/// since pair(p1,p3)=pair(A,A) and pair(p2,p4)=pair(B,B) are diagonal (skipped: AT2's indmat
/// has NA on the diagonal, but for a pure-f2 row p1==p3 so those two -1 terms hit pairs
/// (A,A)/(B,B) which DO NOT EXIST in the basis; the ×2/÷2 makes the surviving +1+1 = 2/2·... ).
/// We replicate AT2 EXACTLY by accumulating ONLY the off-diagonal pairs (i!=j) — the
/// diagonal coefficient writes AT2 does into indmat[x,x] are NA and never read (the design
/// has no diagonal column), so skipping them is byte-identical (confirmed: f2 row a,b → the
/// single nonzero coeff is +1 at pair(a,b)).
void build_popcomb_and_design(int npop, std::vector<PopComb>& combs,
                              std::vector<double>& x, int& npopcomb, int& npairs) {
    combs.clear();
    npairs = npop * (npop - 1) / 2;

    // --- f2: (i,j,i,j) for i<j ---
    for (int i = 0; i < npop; ++i)
        for (int j = i + 1; j < npop; ++j)
            combs.push_back({i, j, i, j});

    // --- f3: C(npop,3) triples, the 3 rotations, mapped (pop1,pop2,pop1,pop4) ---
    // AT2 fstat_get_popcombs(fnum=3) over sorted combn(pop,3): rows {(a,b,c),(b,c,a),(c,a,b)}
    // per combn, in combn order then rotation order (rbind of the three rotation blocks).
    // Then transmute(pop1, pop2, pop4=pop3, pop3=pop1): (P1,P2,P3) -> (P1, P2, P1, P3).
    {
        // combn(npop,3) in column-major (R combn) order: i<j<k ascending.
        std::vector<std::array<int, 3>> tri;
        for (int i = 0; i < npop; ++i)
            for (int j = i + 1; j < npop; ++j)
                for (int k = j + 1; k < npop; ++k)
                    tri.push_back({i, j, k});
        // rbind(outmat[,1:3], outmat[,c(2,3,1)], outmat[,c(3,1,2)]) — three rotation blocks,
        // each in the SAME combn row order. (the row order within each block is combn order.)
        const int rot[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
        for (int r = 0; r < 3; ++r) {
            for (const std::array<int, 3>& t : tri) {
                const int P1 = t[rot[r][0]], P2 = t[rot[r][1]], P3 = t[rot[r][2]];
                // transmute: pop1=P1, pop2=P2, pop3=P1, pop4=P3
                combs.push_back({P1, P2, P1, P3});
            }
        }
    }

    // --- f4: C(npop,4) quads, the 3 rotations, interleaved per-quad (slice(rep..)) ---
    // AT2 fstat_get_popcombs(fnum=4): rbind(outmat[,1:4], outmat[,c(1,3,2,4)],
    // outmat[,c(1,4,2,3)]) then slice(rep(1:nr, each=3) + (0:2)*nr) — the per-quad
    // INTERLEAVE: quad0 rot0, quad0 rot1, quad0 rot2, quad1 rot0, ... (NOT three blocks).
    {
        std::vector<std::array<int, 4>> quad;
        for (int i = 0; i < npop; ++i)
            for (int j = i + 1; j < npop; ++j)
                for (int k = j + 1; k < npop; ++k)
                    for (int l = k + 1; l < npop; ++l)
                        quad.push_back({i, j, k, l});
        const int rot[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
        for (const std::array<int, 4>& q : quad)
            for (int r = 0; r < 3; ++r)
                combs.push_back({q[rot[r][0]], q[rot[r][1]], q[rot[r][2]], q[rot[r][3]]});
    }

    npopcomb = static_cast<int>(combs.size());

    // --- the design x [npopcomb × npairs], COLUMN-MAJOR (x[c + npopcomb*p]) ---
    // EXACT AT2 construct_fstat_matrix semantics: the four writes are ASSIGNMENTS
    // (out[i, idx] = ±1), NOT accumulations — so when two of the four pairs collide on the
    // SAME basis column (the pure-f2 row, where pair(p1,p4)==pair(p2,p3)==pair(a,b)), the
    // later assignment OVERWRITES (the value stays 1, not 1+1=2). A diagonal pair (i==j,
    // AT2 indmat[x,x]=NA) is a no-op (R's `out[i, NA] = v` drops). Then the pure-f2 row is
    // ×2 (recovering the f2 normalization vs the f4 identity) and ALL entries /2.
    x.assign(static_cast<std::size_t>(npopcomb) * static_cast<std::size_t>(npairs), 0.0);
    const auto set = [&](int c, int i, int j, double v) {
        if (i == j) return;  // diagonal pair: NA in AT2 indmat, the write is dropped
        const int p = pair_index(i, j, npop);
        x[static_cast<std::size_t>(c) + static_cast<std::size_t>(npopcomb) *
                                            static_cast<std::size_t>(p)] = v;  // ASSIGN (AT2)
    };
    for (int c = 0; c < npopcomb; ++c) {
        const PopComb& pc = combs[static_cast<std::size_t>(c)];
        // construct_fstat_matrix order: +pair(p1,p4), +pair(p2,p3), -pair(p1,p3), -pair(p2,p4)
        // (sequential assignment; a later collision overwrites the earlier — the AT2 idiom).
        set(c, pc.p1, pc.p4, 1.0);
        set(c, pc.p2, pc.p3, 1.0);
        set(c, pc.p1, pc.p3, -1.0);
        set(c, pc.p2, pc.p4, -1.0);
        if (pc.p1 == pc.p3 && pc.p2 == pc.p4) {  // pure-f2 row: ×2 the whole row
            for (int p = 0; p < npairs; ++p)
                x[static_cast<std::size_t>(c) + static_cast<std::size_t>(npopcomb) *
                                                    static_cast<std::size_t>(p)] *= 2.0;
        }
    }
    // /2 ALL (out/2 in construct_fstat_matrix).
    for (double& v : x) v *= 0.5;
}

/// matrix_jackknife_est_full per popcomb column (AT2 R/io.R; the GLOBAL per-comb estimate
/// `y` used for bglob). For column-c per-block means `numer[c,b]` (= numsum/cnt, NaN where
/// invalid) + counts `cnt[c,b]`:
///   valid = is.finite(numer); rel_bl = cnt/Σcnt;
///   tot = Σ(numer·cnt over valid)/Σ(cnt over valid);
///   loo = (tot - numer·rel_bl)/(1-rel_bl), masked to NA where !valid OR !finite;
///   tot2 = Σ(loo·(1-rel_bl))/Σ(1-rel_bl) over finite loo;
///   weighted_loo_mean = Σ(loo·cnt)/Σ(cnt) over finite loo;
///   y = n_finite·tot2 - Σloo + weighted_loo_mean.
/// `numer`/`cnt` are ROW-MAJOR [npopcomb × n_block] (numer[c*nb+b], the dstat output shape).
[[nodiscard]] double matrix_jackknife_est_col(const double* numer, const double* cnt,
                                              int c, int n_block) {
    const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(n_block);
    long double sum_n_all = 0.0L;  // Σ cnt (all blocks)
    for (int b = 0; b < n_block; ++b) sum_n_all += static_cast<long double>(cnt[base + b]);
    if (sum_n_all <= 0.0L) return std::nan("");

    // tot = Σ(numer·cnt over valid) / Σ(cnt over valid).
    long double tot_num = 0.0L, tot_w = 0.0L;
    for (int b = 0; b < n_block; ++b) {
        const double nu = numer[base + b];
        const double cn = cnt[base + b];
        if (!std::isfinite(nu) || cn <= 0.0) continue;
        tot_num += static_cast<long double>(nu) * static_cast<long double>(cn);
        tot_w += static_cast<long double>(cn);
    }
    if (tot_w <= 0.0L) return std::nan("");
    const long double tot = tot_num / tot_w;

    long double sum_loo = 0.0L, sum_omrb = 0.0L, sum_loo_omrb = 0.0L;
    long double sum_loo_cnt = 0.0L, sum_cnt_finite = 0.0L;
    int n_finite = 0;
    for (int b = 0; b < n_block; ++b) {
        const double nu = numer[base + b];
        const double cn = cnt[base + b];
        if (!std::isfinite(nu) || cn <= 0.0) continue;
        const long double rel = static_cast<long double>(cn) / sum_n_all;  // rel_bl
        if (rel >= 1.0L) continue;  // 1-rel==0 ⇒ loo not finite
        const long double loo = (tot - static_cast<long double>(nu) * rel) / (1.0L - rel);
        if (!std::isfinite(static_cast<double>(loo))) continue;
        const long double omrb = 1.0L - rel;
        sum_loo += loo;
        sum_omrb += omrb;
        sum_loo_omrb += loo * omrb;
        sum_loo_cnt += loo * static_cast<long double>(cn);
        sum_cnt_finite += static_cast<long double>(cn);
        ++n_finite;
    }
    if (n_finite == 0 || sum_omrb <= 0.0L || sum_cnt_finite <= 0.0L) return std::nan("");
    const long double tot2 = sum_loo_omrb / sum_omrb;
    const long double weighted_loo_mean = sum_loo_cnt / sum_cnt_finite;
    return static_cast<double>(static_cast<long double>(n_finite) * tot2 - sum_loo +
                               weighted_loo_mean);
}

/// f2(f2blocks)$est for one pair series (the recentering target). AT2 f2(array): est_to_loo
/// (block_lengths weights) then jack_vec_stats2 = weighted.mean(loo, 1-1/h), h=n/bl. For the
/// per-block estimate vector `arr[b]` (length n_block) + block_lengths `bl[b]`:
///   tot = weighted.mean(arr, bl); rel_bl = bl/Σbl; loo = (tot - arr·rel_bl)/(1-rel_bl);
///   h = Σbl/bl; est = weighted.mean(loo, 1 - 1/h).
/// NaN-safe (na.rm=TRUE: skip non-finite blocks throughout). All-zero/empty → 0.
[[nodiscard]] double f2blocks_pair_est(const std::vector<double>& arr,
                                       const std::vector<int>& bl) {
    const int nb = static_cast<int>(arr.size());
    long double sum_bl = 0.0L;
    for (int b = 0; b < nb; ++b)
        if (std::isfinite(arr[static_cast<std::size_t>(b)]))
            sum_bl += static_cast<long double>(bl[static_cast<std::size_t>(b)]);
    if (sum_bl <= 0.0L) return 0.0;

    long double tot_num = 0.0L;
    for (int b = 0; b < nb; ++b) {
        const double a = arr[static_cast<std::size_t>(b)];
        if (!std::isfinite(a)) continue;
        tot_num += static_cast<long double>(a) *
                   static_cast<long double>(bl[static_cast<std::size_t>(b)]);
    }
    const long double tot = tot_num / sum_bl;  // weighted.mean(arr, bl)

    // est = weighted.mean(loo, 1 - 1/h), h = Σbl/bl, loo = (tot - arr·rel_bl)/(1-rel_bl).
    long double num = 0.0L, den = 0.0L;
    for (int b = 0; b < nb; ++b) {
        const double a = arr[static_cast<std::size_t>(b)];
        if (!std::isfinite(a)) continue;
        const long double blb = static_cast<long double>(bl[static_cast<std::size_t>(b)]);
        const long double rel = blb / sum_bl;       // rel_bl
        if (rel >= 1.0L) continue;
        const long double loo = (tot - static_cast<long double>(a) * rel) / (1.0L - rel);
        const long double h = sum_bl / blb;          // h
        const long double w = 1.0L - 1.0L / h;       // 1 - 1/h
        num += loo * w;
        den += w;
    }
    if (den <= 0.0L) return 0.0;
    return static_cast<double>(num / den);
}

}  // namespace

// ---- Public entry (include/steppe/qpfstats.hpp) -------------------------------------
QpfstatsResult run_qpfstats(const std::string& geno, const std::string& snp,
                            const std::string& ind, std::span<const std::string> pops,
                            double blgsize_morgans, const Precision& precision,
                            device::Resources& resources) {
    QpfstatsResult res;
    res.precision_tag = precision.kind;

    // ---- 0. The SORTED pop set (AT2 `sp = sort(pops)`; the dimnames order) ----------
    std::vector<std::string> sp(pops.begin(), pops.end());
    std::sort(sp.begin(), sp.end());
    sp.erase(std::unique(sp.begin(), sp.end()), sp.end());
    const int npop = static_cast<int>(sp.size());
    res.pop_labels = sp;
    if (npop < 4) {
        // qpfstats needs at least the f4 basis (n>=4 for a non-degenerate full basis).
        res.status = Status::InvalidConfig;
        return res;
    }

    // ---- 1. DECODE FRONT-END (REUSE qpDstat-B; mirrors stats/dstat.cpp:185-264) -------
    // read_ind(Explicit{sp}) decodes ONLY these pops (the AT2 indvec), sorted ASC by label
    // (== sp). FORCED diploid (the AT2 plain ref/an/2 pin). An io fault PROPAGATES.
    io::GenoReader reader(geno);
    const std::size_t n_present = reader.records_present();
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::Explicit;
    sel.labels = sp;
    const io::IndPartition part = io::read_ind(ind, sel, n_present);
    const io::SnpTable snptab = io::read_snp(snp, SIZE_MAX);
    const std::size_t M0 = std::min(reader.header().n_snp, snptab.count);
    const io::GenotypeTile tile = reader.read_tile(part, 0, M0);

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P != npop) {
        // read_ind returns the partition in label-sorted order; if a pop is missing the
        // partition is shorter than requested — a config fault (not a domain outcome).
        res.status = Status::InvalidConfig;
        return res;
    }
    if (P <= 0 || M <= 0) { res.status = Status::Ok; return res; }

    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;

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

    // Autosome keep-mask + lockstep subset of Q/V + chrom/genpos (AT2 auto_only; chr<=22).
    std::vector<double> Qk, Vk;
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    Qk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M));
    Vk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M));
    chrom_kept.reserve(static_cast<std::size_t>(M));
    genpos_kept.reserve(static_cast<std::size_t>(M));
    for (long s = 0; s < M; ++s) {
        const int chr = snptab.chrom[static_cast<std::size_t>(s)];
        if (chr < 1 || chr > 22) continue;
        const std::size_t src = static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
        for (int p = 0; p < P; ++p) {
            Qk.push_back(dec.q[src + static_cast<std::size_t>(p)]);
            Vk.push_back(dec.v[src + static_cast<std::size_t>(p)]);
        }
        chrom_kept.push_back(chr);
        genpos_kept.push_back(snptab.genpos_morgans[static_cast<std::size_t>(s)]);
    }
    const long M_kept = static_cast<long>(chrom_kept.size());
    if (M_kept <= 0) { res.status = Status::Ok; return res; }

    // ---- 2. assign_blocks over the KEPT (autosome) SNP axis (AT2 get_block_lengths) ---
    const core::BlockPartition partition = core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans);
    const int n_block = partition.n_block;
    if (n_block <= 0) { res.status = Status::Ok; return res; }

    // Per-block SNP count (AT2 block_lengths) — the recenter jackknife weight + the f2-dir
    // block_sizes. The SINGLE-SOURCE inverse of assign_blocks (the same primitive the f2 /
    // dstat paths use): each block's contiguous SNP slice [begin, end).
    const std::vector<core::BlockRange> ranges = core::block_ranges(
        std::span<const int>(partition.block_id), M_kept, n_block);
    std::vector<int> block_lengths(static_cast<std::size_t>(n_block), 0);
    for (int b = 0; b < n_block; ++b)
        block_lengths[static_cast<std::size_t>(b)] =
            static_cast<int>(ranges[static_cast<std::size_t>(b)].size());

    // ---- 3. Build the popcomb set + the design x (data-independent; tiny host) --------
    std::vector<PopComb> combs;
    std::vector<double> x;  // COLUMN-MAJOR [npopcomb × npairs]
    int npopcomb = 0, npairs = 0;
    build_popcomb_and_design(npop, combs, x, npopcomb, npairs);

    // Flat 4*npopcomb quad table for the dstat numerator engine (the SAME flat layout the
    // dstat kernel reads: quad k at [4k..4k+3] = {p1,p2,p3,p4}).
    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(npopcomb) * 4);
    for (const PopComb& pc : combs) {
        flat.push_back(pc.p1); flat.push_back(pc.p2); flat.push_back(pc.p3); flat.push_back(pc.p4);
    }

    // ---- 4. The genotype-f4 NUMERATOR ENGINE (REUSE dstat_block_reduce) ON THE GPU ----
    // Per (comb, block): numsum = Σ(a-b)(c-d), cnt = #SNPs valid in all 4 pops (allsnps).
    // densum (the D denominator) is emitted by the kernel but IGNORED (f4mode=TRUE). Output
    // ROW-MAJOR [npopcomb × n_block].
    const std::size_t nb_out =
        static_cast<std::size_t>(npopcomb) * static_cast<std::size_t>(n_block);
    std::vector<double> numsum(nb_out, 0.0), densum(nb_out, 0.0), cnt(nb_out, 0.0);
    be.dstat_block_reduce(Qk.data(), Vk.data(), P, M_kept, partition.block_id.data(), n_block,
                          std::span<const int>(flat), numsum.data(), densum.data(), cnt.data());

    // ---- 5. numer = numsum/cnt (the AT2 rowMeans; NaN where cnt==0); ymat = t(numer) ---
    // ymat is COLUMN-MAJOR [npopcomb × n_block] (ymat[c + npopcomb*b]); numer/cnt are
    // ROW-MAJOR [npopcomb × n_block]. y = matrix_jackknife_est over each comb column.
    std::vector<double> ymat(nb_out, 0.0);
    std::vector<double> y(static_cast<std::size_t>(npopcomb), 0.0);
    std::vector<double> numer_rm(nb_out, 0.0);  // row-major numer for matrix_jackknife_est
    for (int c = 0; c < npopcomb; ++c) {
        const std::size_t base =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_block);
        for (int b = 0; b < n_block; ++b) {
            const double cn = cnt[base + b];
            const double mean = (cn > 0.0) ? (numsum[base + b] / cn) : std::nan("");
            numer_rm[base + b] = mean;
            ymat[static_cast<std::size_t>(c) +
                 static_cast<std::size_t>(npopcomb) * static_cast<std::size_t>(b)] = mean;
        }
        y[static_cast<std::size_t>(c)] =
            matrix_jackknife_est_col(numer_rm.data(), cnt.data(), c, n_block);
    }

    // ---- 6. THE SMOOTHING SOLVE (ON THE GPU; shared-factor batched; NO host block loop) -
    // A_shared = x'x + ridge·I; b[:,blk] = solve(A_blk, x'·ymat[:,blk]); bglob = solve(A_y,
    // x'·y). The CUDA backend: ONE syrk + ONE potrf + ONE gemm + a Dtrsm pair (no-NaN
    // blocks) + the CUB-grouped NaN downdate. The CpuBackend is the native small_linalg
    // oracle. The matmul sub-steps run `precision` (EmulatedFp64{40} default); the
    // Cholesky/solve native FP64 (the §12 carve-out).
    const QpfstatsSmooth sm = be.qpfstats_smooth(
        std::span<const double>(x), std::span<const double>(ymat), std::span<const double>(y),
        npopcomb, npairs, n_block, kRidge, precision);
    if (sm.status != Status::Ok) { res.status = sm.status; return res; }

    // ---- 7. Scatter b → f2blocks[npop,npop,n_block] (off-diagonal pairs, symmetric) ----
    // f2blocks[i + P·j + P·P·b] = b[pair(i,j), b] for i!=j (lower+upper); diagonal stays 0.
    F2BlockTensor& T = res.f2;
    T.P = npop;
    T.n_block = n_block;
    const std::size_t slab = static_cast<std::size_t>(npop) * static_cast<std::size_t>(npop);
    T.f2.assign(slab * static_cast<std::size_t>(n_block), 0.0);
    T.vpair.assign(slab * static_cast<std::size_t>(n_block), 0.0);
    T.block_sizes.assign(static_cast<std::size_t>(n_block), 0);
    const std::size_t np = static_cast<std::size_t>(npairs);
    for (int b = 0; b < n_block; ++b) {
        T.block_sizes[static_cast<std::size_t>(b)] =
            block_lengths[static_cast<std::size_t>(b)];
        const std::size_t boff = slab * static_cast<std::size_t>(b);
        for (int i = 0; i < npop; ++i) {
            for (int j = i + 1; j < npop; ++j) {
                const int p = pair_index(i, j, npop);
                const double v = sm.b[static_cast<std::size_t>(p) +
                                      np * static_cast<std::size_t>(b)];
                T.f2[boff + static_cast<std::size_t>(i) +
                     static_cast<std::size_t>(npop) * static_cast<std::size_t>(j)] = v;
                T.f2[boff + static_cast<std::size_t>(j) +
                     static_cast<std::size_t>(npop) * static_cast<std::size_t>(i)] = v;
            }
        }
    }

    // ---- 8. RECENTER: f2blocks2 = f2blocks - f2(f2blocks)$est + bglob (AT2 qpfstats) ---
    // Per pair (i,j): the constant shift c(i,j) = bglob[pair] - f2blocks_pair_est[pair] is
    // added to EVERY block's (i,j). The recenter target uses block_lengths weights (the AT2
    // f2() jackknife est over the smoothed tensor). The diagonal stays 0.
    std::vector<double> series(static_cast<std::size_t>(n_block), 0.0);
    for (int i = 0; i < npop; ++i) {
        for (int j = i + 1; j < npop; ++j) {
            const int p = pair_index(i, j, npop);
            for (int b = 0; b < n_block; ++b)
                series[static_cast<std::size_t>(b)] =
                    sm.b[static_cast<std::size_t>(p) + np * static_cast<std::size_t>(b)];
            const double est = f2blocks_pair_est(series, T.block_sizes);
            const double shift = sm.bglob[static_cast<std::size_t>(p)] - est;
            for (int b = 0; b < n_block; ++b) {
                const std::size_t boff = slab * static_cast<std::size_t>(b);
                T.f2[boff + static_cast<std::size_t>(i) +
                     static_cast<std::size_t>(npop) * static_cast<std::size_t>(j)] += shift;
                T.f2[boff + static_cast<std::size_t>(j) +
                     static_cast<std::size_t>(npop) * static_cast<std::size_t>(i)] += shift;
            }
        }
    }

    // ---- 9. vpair: per-block kept-SNP count replicated across pairs (so write_f2_dir's
    // missing-block detect sees nonzero where a block contributed). The smoothed f2 has no
    // per-pair valid count (the regression imputes), so we record the block SNP count on
    // every off-diagonal pair (a block that contributed to ANY comb has count>0). ----
    for (int b = 0; b < n_block; ++b) {
        const double bs = static_cast<double>(T.block_sizes[static_cast<std::size_t>(b)]);
        const std::size_t boff = slab * static_cast<std::size_t>(b);
        for (int i = 0; i < npop; ++i)
            for (int j = 0; j < npop; ++j)
                if (i != j)
                    T.vpair[boff + static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(npop) * static_cast<std::size_t>(j)] = bs;
    }

    res.status = Status::Ok;
    return res;
}

}  // namespace steppe
