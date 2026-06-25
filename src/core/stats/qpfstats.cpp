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
// THE FUSED reduce→jackknife→smooth→recenter (ComputeBackend::qpfstats_blocks_smooth, ONE
// device residency; AT2 qpfstats_regression REFORMULATED — NO host per-block / per-comb loop):
//   numer[comb,block] = numsum/cnt (the AT2 rowMeans, NaN where cnt==0).
//   x[comb,pair] = construct_fstat_matrix (±1/2 f4-identity coefficients; *2 pure-f2 row).
//   ymat = numer (col-major [npopcomb × nblock]); y = matrix_jackknife_est(numer,cnt).
//   A_shared = x'x + ridge·I (ridge=1e-5); b[:,blk] = solve(A_blk, x'·ymat[:,blk]);
//   bglob = solve(A_y, x'·y). The NaN-comb-row downdate + all-NaN→0 handled in the seam.
//   recenter_shift[pair] = bglob[pair] - f2blocks_pair_est(b[pair,:], block_sizes).
// The numer/ymat materialize, the per-comb jackknife (matrix_jackknife_est_col), AND the
// per-pair recenter jackknife (f2blocks_pair_est) ALL run ON-DEVICE inside the seam — the
// ~1.7GB-each numsum/cnt D2H and the ~305k×711 host long-double jackknife loop (the GPU-idle
// "0" half of the 100/0 alternation) are ELIMINATED. Only b/bglob/recenter_shift cross back.
//
// THE OUTPUT (AT2 qpfstats scatter+recenter): scatter b → f2blocks[npop,npop,nblock]
// (off-diagonal pairs, symmetric); recenter f2blocks2 = f2blocks + recenter_shift[pair] (the
// per-pair block-jackknife est of the smoothed tensor, computed in the seam). Diagonal stays 0.
// THIS is the smoothed F2BlockTensor the downstream tools consume.
//
// THE THREE PARITY PINS (inherited from qpDstat-B, proven on box5090): forced diploid (AT2
// plain ref/an/2), assign_blocks == AT2 get_block_lengths, allsnps=TRUE per-(comb,block)
// finiteness (no maxmiss/MAF/drop-mono; autosomes_only ON).
//
// CUDA-FREE: the WHOLE numerator reduce + jackknife + smoothing solve + recenter route through
// the ONE fused ComputeBackend seam (CpuBackend oracle / CudaBackend kernels). All host work is
// O(small): the popcomb/design build (data-independent) and the tensor scatter. NO host
// per-SNP / per-jackknife-block / per-comb-jackknife loop (the CPU-bound trap is GONE).

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
#include "core/stats/read_canonical_tile.hpp"     // M-FR-2 TGENO/GENO format dispatch
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

// The per-comb matrix_jackknife_est_col + the per-pair f2blocks_pair_est long-double REFERENCE
// jackknives moved to core/internal/qpfstats_jackknife.hpp (SINGLE-SOURCE, shared by the
// CpuBackend fused oracle). On the production path they run ON-DEVICE inside the fused seam
// (ComputeBackend::qpfstats_blocks_smooth); the host no longer loops over combs/pairs.

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
    // M-FR-2 FORMAT DISPATCH: TGENO -> read_tile (unchanged); GENO (SNP-major PA) ->
    // the io-leaf SNP-major gather + the on-device transpose_to_canonical. `tile` is
    // the canonical individual-major packing the decode front-end expects either way.
    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
    const io::GenotypeTile tile = core::read_canonical_tile(reader, part, be, 0, M0);

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P != npop) {
        // read_ind returns the partition in label-sorted order; if a pop is missing the
        // partition is shorter than requested — a config fault (not a domain outcome).
        res.status = Status::InvalidConfig;
        return res;
    }
    if (P <= 0 || M <= 0) { res.status = Status::Ok; return res; }

    // `be` was bound above (the tile read dispatch); reused here for the decode.
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

    // ---- 1. Decode + the AUTOSOME keep + the lockstep Q/V subset (AT2 auto_only; chr 1..22).
    // DEVICE-RESIDENT seam (host-compute audit C1/C2/M3/M4 cure): on the CUDA backend the
    // decode → autosome keep-mask → CUB/scan-gather Q/V compaction all run ON-DEVICE and the
    // resident compacted Q/V + the small kept chrom/genpos escape in `ddr` (NO ~1.1GB Q/V/N D2H,
    // NO host filter loop, NO H2D re-upload). The CpuBackend (device_count==0, the parity oracle)
    // keeps the host path: decode_af → the host autosome loop → host Q/V subset. BOTH produce the
    // IDENTICAL kept SET, kept ORDER, and chrom_kept/genpos_kept → identical assign_blocks/golden.
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
            if (chr < 1 || chr > 22) continue;
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

    // ---- 4-8. FUSED reduce→jackknife→smooth→recenter ON THE GPU (ONE device residency) -
    // The genotype-f4 NUMERATOR reduce (dstat_block_reduce), the per-comb block-JACKKNIFE
    // (matrix_jackknife_est_col → y, and numer=numsum/cnt → ymat), the SMOOTHING SOLVE
    // (A_shared = x'x + ridge·I; b[:,blk]=solve(A_blk, x'·ymat[:,blk]); bglob=solve(A_y, x'·y)),
    // AND the per-pair RECENTER jackknife (f2blocks_pair_est → recenter_shift) — ALL on-device,
    // numsum/cnt/ymat/y/b RESIDENT in VRAM. The host no longer materializes numer/ymat, runs
    // the ~305k×711 long-double jackknife, or loops the recenter (the 100/0 GPU-idle half +
    // the ~1.7GB-each D2H are GONE). The matmul sub-steps run `precision` (EmulatedFp64{40}
    // default); the jackknives + the Cholesky/solve are native FP64 (the §12 carve-out). The
    // recenter weights are block_lengths (the AT2 f2() block_sizes). Only b/bglob/recenter_shift
    // cross back. The CpuBackend composes its existing oracles (unchanged reference math).
    const QpfstatsSmooth sm = resident
        ? be.qpfstats_blocks_smooth(
              ddr, partition.block_id.data(), n_block,
              std::span<const int>(flat), std::span<const double>(x), npopcomb, npairs,
              std::span<const int>(block_lengths), kRidge, precision)
        : be.qpfstats_blocks_smooth(
              Qk.data(), Vk.data(), P, M_kept, partition.block_id.data(), n_block,
              std::span<const int>(flat), std::span<const double>(x), npopcomb, npairs,
              std::span<const int>(block_lengths), kRidge, precision);
    if (sm.status != Status::Ok) { res.status = sm.status; return res; }

    // ---- 7-8. Scatter b → f2blocks[npop,npop,n_block] (off-diagonal, symmetric) + the
    // per-pair RECENTER shift (computed on-device in the seam). f2blocks2[i + P·j + P·P·b] =
    // b[pair(i,j), b] + recenter_shift[pair] for i!=j (lower+upper); the diagonal stays 0.
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
                                      np * static_cast<std::size_t>(b)] +
                                 sm.recenter_shift[static_cast<std::size_t>(p)];
                T.f2[boff + static_cast<std::size_t>(i) +
                     static_cast<std::size_t>(npop) * static_cast<std::size_t>(j)] = v;
                T.f2[boff + static_cast<std::size_t>(j) +
                     static_cast<std::size_t>(npop) * static_cast<std::size_t>(i)] = v;
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
