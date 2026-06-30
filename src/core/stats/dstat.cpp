// src/core/stats/dstat.cpp — the genotype-path NORMALIZED-D entry (run_dstat; qpDstat Part B).
//
// run_dstat is the genotype-reading SIBLING of run_f4 / run_f4ratio: it does NOT read the f2
// cache. It REUSES the extract-f2 decode FRONT-END (the io reader + decode_af [per-SNP Q/V/N]
// + assign_blocks [from genpos]) and DIVERGES at S2 into the per-SNP D kernel
// (ComputeBackend::dstat_block_reduce, the .cu) + the num/den block-jackknife — the SHARED
// on-device ratio_block_jackknife backend virtual (NOT a host loop; ONE engine with f4-ratio,
// reached via dstat_blocks_jackknife; see the per-virtual note below). Pinned to the AT2
// qpdstat_geno golden (allsnps=TRUE, f4mode=FALSE, blgsize=0.05) —
// docs/research/dates-genotype-stat-seam.md (i).
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
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // assign_blocks, BlockPartition
#include "core/stats/genotype_front_end.hpp"      // C1 shared genotype decode front-end
#include "device/backend.hpp"                     // ComputeBackend, DecodeTileView, DecodeResult
#include "device/resources.hpp"                   // device::Resources (the injected backend bundle)

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
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

/// The num/den block-jackknife (AT2 est_to_loo_dat + jack_dat_stats) is NO LONGER a host loop
/// here: it is the SHARED on-device ratio_block_jackknife backend virtual (backend.hpp), reached
/// via dstat_blocks_jackknife — ONE engine with f4-ratio. On the CUDA path the per-(quadruple,
/// block) numsum/densum/cnt stay RESIDENT (dNum/dDen/dCnt) and feed the kernel directly (the
/// [N·n_block] D2H is DROPPED); on the CpuBackend the SAME entry reduces then delegates to the
/// long-double oracle (tot_mode=1, cnt>0 survivor mask, weight=cnt, p=f4_two_sided_p; the
/// reference math UNCHANGED). Native FP64 (the §12 num/den cancellation carve-out).

/// Fill a DEGENERATE D table: write the per-row NaN sentinel into the four stat columns
/// (est/se/z/p, `row_count` rows each) and keep status Ok — a domain outcome, never an
/// exception (architecture.md §10). Writes ONLY the four columns + status, so the p1..p4
/// label rows and precision_tag the caller already populated are PRESERVED (in-place fill,
/// not a fresh result). De-duplicates the verbatim early-return fill that rode on the three
/// degenerate guards (empty pop/SNP axis, no kept autosome SNP, no block).
void fill_nan(DstatResult& res, int row_count) {
    const auto count = static_cast<std::size_t>(row_count);
    res.est.assign(count, std::nan(""));
    res.se.assign(count, std::nan(""));
    res.z.assign(count, std::nan(""));
    res.p.assign(count, std::nan(""));
    res.status = Status::Ok;
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
    // The shared genotype DECODE FRONT-END (C1): the core helper opens the GenoReader, reads
    // the Explicit{pop_union} IndPartition + the SnpTable, and reads the canonical individual-
    // major tile (M-FR-2 format dispatch). `be` is the primary-GPU backend (forwarded to the
    // non-TGENO transpose, reused below for the decode). The decode diverges below (forced
    // diploid + autosome keep).
    ComputeBackend& be = *resources.gpus.at(kPrimaryGpu).backend;
    const core::GenotypeFrontEnd fe =
        core::read_genotype_front_end(geno, snp, ind, pop_union, be);
    const io::SnpTable& snptab = fe.snptab;
    const io::GenotypeTile& tile = fe.tile;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) {
        fill_nan(res, N);
        return res;
    }

    // PARITY PIN (1): FORCED DIPLOID ploidy (AT2 plain ref/an/2; NOT extract-f2 Auto).
    std::vector<int> sample_ploidy(tile.n_individuals, kPloidyDiploid);

    // PARITY PIN (3) — AUTOSOMES ONLY (AT2 auto_only default; chr 1..22). allsnps=TRUE
    // applies NO maxmiss/MAF/drop-mono, so the ONLY SNP filter is the autosome keep + the
    // per-(block,quadruple) finiteness mask (applied inside the D kernel via V). We subset
    // the SNP axis (Q/V/chrom/genpos) to the autosomes in LOCKSTEP after the decode.
    // `be` was bound above (the tile read dispatch); reused here for the decode.

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
            if (chr < kAutosomeChromMin || chr > kAutosomeChromMax) continue;  // AT2 auto_only: autosomes 1..22.
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
        fill_nan(res, N);
        return res;
    }

    // ---- 2. assign_blocks over the KEPT (autosome) SNP axis (PARITY PIN (2)) ---------
    const core::BlockPartition partition = core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans);
    const int n_block = partition.n_block;
    if (n_block <= 0) {
        fill_nan(res, N);
        return res;
    }

    // ---- 3 + 4 FUSED — the per-SNP D reduce + the SHARED on-device ratio-block-jackknife in
    // ONE backend call (host-compute-audit M2 cure). On the CUDA path the per-(quadruple,block)
    // numsum/densum/cnt stay RESIDENT (dNum/dDen/dCnt) and feed the ratio_block_jackknife kernel
    // directly — DROPPING the former [N·n_block] numsum/densum/cnt D2H + the host per-quadruple
    // dstat_jackknife loop. On the CpuBackend the SAME entry reduces then delegates to the
    // long-double dstat_jackknife oracle (tot_mode=1, cnt>0 mask, weight=cnt, the reference math
    // UNCHANGED). p = f4_two_sided_p(z) (== AT2 ztop) is computed in-kernel / in-oracle. The
    // num/den block-jackknife is the SAME ratio block-jackknife as f4-ratio (one shared virtual).
    const RatioBlockJackknife jk =
        resident
            ? be.dstat_blocks_jackknife(ddr, partition.block_id.data(), n_block,
                                        std::span<const int>(flat))
            : be.dstat_blocks_jackknife(Qk.data(), Vk.data(), P, M_kept,
                                        partition.block_id.data(), n_block,
                                        std::span<const int>(flat));
    res.est = jk.est;
    res.se = jk.se;
    res.z = jk.z;
    res.p = jk.p;
    res.status = jk.status;
    return res;
}

}  // namespace steppe
