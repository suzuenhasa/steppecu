// src/app/extract_f2_core.cpp
//
// The LIBRARY genotype->f2 extract entry (steppe::run_extract_f2; include/steppe/extract.hpp)
// — M(py-2). The decode->filter->assign_blocks->compute_f2_blocks_multigpu_tiered->to_host
// chain, lifted VERBATIM from cmd_extract_f2.cpp:157-498, with the CLI's std::fprintf(stderr)
// + kExit* exit codes REPLACED by std::runtime_error / std::invalid_argument throws and a
// value result (F2ExtractResult). The MATH is byte-identical to the CLI — the goldens are
// untouched; the CLI command run_extract_f2_command now calls THIS (a thin wrapper) so there
// is ONE copy of the chain (DRY; the §4 layering / arch-grep gate — NO CUDA header here, the
// GPU is reached only through resources.hpp build path + the CUDA-free seams).
//
// THE PARITY PINS (cmd_extract_f2.cpp): (1) the P axis is read_ind(sel) sorted ASC by label;
// (2) per-SAMPLE ploidy auto-detect (AT2 adjust_pseudohaploid) unless forced; (3) the AT2
// maxmiss is the POPULATION-axis coverage test, applied separately while the sample-axis
// geno_max_missing is forced to the no-op (1.0); (4) autosomes_only is the FilterConfig flag.
#include "steppe/extract.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"  // assign_blocks, block_size_cm_to_morgans
#include "core/fstats/f2_blocks_multigpu.hpp"     // compute_f2_blocks_multigpu_tiered (CUDA-FREE)
#include "core/stats/genotype_front_end.hpp"      // C1 shared genotype decode front-end
#include "core/internal/views.hpp"                // steppe::core::MatView
#include "device/backend.hpp"                     // DecodeTileView, DecodeResult (CUDA-FREE)
#include "device/decode_budget.hpp"               // decode_tile_snps (regime-B SNP-tile width, CUDA-FREE)
#include "device/f2_blocks_out.hpp"               // F2BlocksOut (CUDA-FREE)
#include "device/resources.hpp"                   // Resources (CUDA-FREE)
#include "steppe/config.hpp"                      // Precision, FilterConfig

#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"  // read_snp_table / read_ind_partition (format-aware .snp|.bim, .ind|.fam)
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"
#include "io/filter/snp_filter.hpp"

namespace steppe {

namespace {

using steppe::core::MatView;
namespace flt = steppe::io::filter;

// AT2 ploidy values (mirror cmd_extract_f2.cpp's kPloidy{PseudoHaploid,Diploid}).
constexpr int kPloidyPseudoHaploid = 1;
constexpr int kPloidyDiploid = 2;

// Validate that every requested Explicit pop label is present in the resolved partition
// (cli-bindings.md §4.2). read_ind silently drops an unknown label (only throwing on a
// FULLY empty selection), so verify the request against the resolved groups here — the
// SAME validate_explicit_pops cmd_extract_f2.cpp uses, but THROWING (the library contract)
// instead of returning a CLI exit code.
void validate_explicit_pops(const io::PopSelection& sel, const io::IndPartition& part) {
    if (sel.mode != io::PopSelection::Mode::Explicit) return;
    for (const std::string& want : sel.labels) {
        const bool found = std::any_of(
            part.groups.begin(), part.groups.end(),
            [&want](const io::PopGroup& g) { return g.label == want; });
        if (!found) {
            throw std::invalid_argument(
                "extract_f2: pops contains an unknown population '" + want +
                "' (not present in the .ind)");
        }
    }
}

}  // namespace

F2ExtractResult run_extract_f2(const std::string& geno,
                               const std::string& snp,
                               const std::string& ind,
                               const io::PopSelection& sel,
                               const FilterConfig& filter,
                               const Precision& precision,
                               double blgsize_morgans,
                               ExtractPloidy ploidy,
                               device::Resources& resources) {
    if (geno.empty() || snp.empty() || ind.empty()) {
        throw std::invalid_argument(
            "extract_f2: the genotype triple (.geno/.snp/.ind) is required");
    }
    if (resources.gpus.empty()) {
        throw std::runtime_error(
            "no CUDA device available (steppe is a GPU product; a CUDA-capable GPU is "
            "required)");
    }

    // ---- 1. Open the .geno/.bed + read .ind/.fam (selection) + .snp/.bim ---------------
    // The GenoReader ctor pins the on-disk format (TGENO/GENO/EIGENSTRAT/PLINK); the
    // SnpTable + IndPartition reads dispatch the parser on that format (read_snp/read_ind
    // for the EIGENSTRAT family; read_bim/read_fam for PLINK) — M-FR PLINK. The geno/snp/
    // ind paths already carry the correct extensions (config resolve_genotype_triple).
    // The shared genotype DECODE FRONT-END (C1): one core helper opens the GenoReader, reads
    // the IndPartition for `sel` + the SnpTable, and reads the canonical individual-major tile
    // (M-FR-2 format dispatch). `backend` is bound here (forwarded to the non-TGENO transpose,
    // reused below for the decode). validate_explicit_pops stays in THIS caller (the library
    // contract — it needs the resolved partition); folding both reads into the helper only
    // reorders which fault fires on a malformed-pops input (an error path, never a golden).
    steppe::ComputeBackend& backend = *resources.gpus.front().backend;
    const steppe::core::GenotypeFrontEnd fe =
        steppe::core::read_genotype_front_end(geno, snp, ind, sel, backend);
    validate_explicit_pops(sel, fe.part);
    const io::IndPartition& part = fe.part;
    const io::SnpTable& snptab = fe.snptab;
    const io::GenotypeTile& tile = fe.tile;

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) {
        throw std::invalid_argument(
            "extract_f2: empty selection (P=" + std::to_string(P) + ") or no SNPs (M=" +
            std::to_string(M) + ")");
    }
    if (snptab.count < static_cast<std::size_t>(M)) {
        throw std::runtime_error(
            "extract_f2: .snp has " + std::to_string(snptab.count) +
            " rows but the tile spans " + std::to_string(M) +
            " SNPs (the .snp/.geno SNP axes must agree)");
    }

    // The P labels in P-axis index order (= pops.txt) — the name<->index map.
    std::vector<std::string> pop_labels;
    pop_labels.reserve(static_cast<std::size_t>(P));
    for (const io::PopGroup& g : part.groups) pop_labels.push_back(g.label);

    // ---- 2. Decode + REGIME-B FILTER + lockstep Q/V/N compaction ----------------------
    // `backend` was bound above (the tile read dispatch); reused here for the decode.

    // ---- PER-SAMPLE PLOIDY (the f2 pseudo-haploid fix; AT2 adjust_pseudohaploid) -------
    // M-FR-0 (the L2 host-compute fix): for AUTO, the per-sample ploidy detection moves
    // ON-DEVICE — set DecodeTileView.detect_ploidy_on_device so the decode derives the
    // ploidy ITSELF from the uploaded d_packed (the GPU prepass; the CpuBackend oracle
    // runs the host scan), instead of the eager host io::detect_sample_ploidy here. The
    // explicit PseudoHaploid/Diploid modes still fill a uniform vector (no detection).
    std::vector<int> sample_ploidy;          // explicit modes only; EMPTY for Auto.
    bool detect_on_device = false;
    switch (ploidy) {
        case ExtractPloidy::Auto:
            detect_on_device = true;         // the on-device prepass drives the decode.
            break;
        case ExtractPloidy::PseudoHaploid:
            sample_ploidy.assign(tile.n_individuals, kPloidyPseudoHaploid);
            break;
        case ExtractPloidy::Diploid:
            sample_ploidy.assign(tile.n_individuals, kPloidyDiploid);
            break;
    }

    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.empty() ? nullptr : sample_ploidy.data();
    view.detect_ploidy_on_device = detect_on_device;  // M-FR-0: Auto ⇒ on-device prepass.
    view.ploidy = kPloidyDiploid;  // uniform fallback (unused when a vector/flag is set).

    // The pseudo-haploid/diploid REPORT counts (observability). For the explicit modes
    // the vector is in hand; for AUTO the prepass runs inside the decode, so derive the
    // counts from the backend's on-device prepass (the SAME launch_detect_ploidy; one
    // cheap [n_individuals] pass) — the GPU-first source, never the eager host scan.
    // This is a SEPARATE local so it never disturbs view.sample_ploidy (which stays
    // nullptr for Auto — the decode's detect_ploidy_on_device flag is what drives it).
    std::size_t n_ph = 0, n_dip = 0;
    const std::vector<int> ploidy_for_counts =
        detect_on_device ? backend.detect_sample_ploidy_device(view) : sample_ploidy;
    for (int pl : ploidy_for_counts) (pl == kPloidyPseudoHaploid ? n_ph : n_dip)++;

    // THE maxmiss SEMANTIC (cmd_extract_f2.cpp:358-416): the AT2 extract_f2 maxmiss is the
    // POPULATION-axis coverage test, applied SEPARATELY; the sample-axis geno_max_missing is
    // forced to the no-op (1.0) so the sample-axis predicate does not double-filter.
    std::vector<std::size_t> pop_individuals(static_cast<std::size_t>(P));
    for (int p = 0; p < P; ++p) {
        pop_individuals[static_cast<std::size_t>(p)] =
            tile.pop_offsets[static_cast<std::size_t>(p) + 1] -
            tile.pop_offsets[static_cast<std::size_t>(p)];
    }
    const double maxmiss = filter.geno_max_missing;  // the AT2 pop-coverage maxmiss.

    // The COMPACTED kept tensor + the kept axis. The CUDA path computes the keep-set
    // + the lockstep Q/V/N compaction ON-DEVICE (the regime-B keep-mask kernel: the
    // pooled-MAF Σ_pop Q·N [FFMA-immune] + the shared keep_decision_pooled + the
    // SEPARATE pop-coverage maxmiss) — the host per-SNP filter loop + the full-tile
    // D2H are GONE; only the small compacted Q/V/N cross to host. The CpuBackend (the
    // parity oracle) keeps the host path: decode_af + build_snp_keep_mask + the host
    // maxmiss loop + the host lockstep subset (UNCHANGED). BOTH produce the IDENTICAL
    // kept SET, kept ORDER, chrom_kept/genpos_kept, and compacted Q/V/N (the keep-set
    // is bit-exact-gated: same SNPs, same order ⇒ identical assign_blocks/golden).
    const bool on_device = (backend.capabilities().device_count > 0);
    std::vector<double> Qk, Vk, Nk;
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    std::vector<double> physpos_kept;  // the AT2 bp block-fallback axis (all-zero-map case).
    long M_kept = 0;

    if (on_device) {
        // HOIST PLOIDY TO EXPLICIT before the SNP-tile loop. The full-view detection
        // (ploidy_for_counts, computed above over the fixed first-min(kPloidyDetectSnps,
        // M) SNP prefix) becomes the SINGLE explicit per-sample vector EVERY tile uses.
        // This is REQUIRED for tiled parity: a per-tile on-device re-detect
        // (detect_ploidy_on_device) would scan only THAT tile's leading SNPs, giving a
        // DIFFERENT ploidy on tiles past the first (and possibly on a first tile whose
        // width < kPloidyDetectSnps) — wrong Q/N. For Auto this is bit-identical to the
        // single-shot on-device detect (the SAME launch_detect_ploidy over the SAME
        // full-M prefix); for the explicit modes ploidy_for_counts == the uniform vector
        // that was already in view.sample_ploidy, so nothing moves.
        view.sample_ploidy = ploidy_for_counts.data();
        view.detect_ploidy_on_device = false;

        // SNP-TILE the decode so peak VRAM is O(P × tile_M), not O(P × M): the decode
        // allocates ~6 [P × M] FP64 buffers (dQ/dV/dN + the compacted q/v/n) BEFORE the
        // tier runs, an OOM at large P × M. tile_M is a multiple of 4 (every s_lo stays
        // 2-bit-packing-aligned); the last tile may be shorter. Byte-identical to the
        // single-shot path: per-SNP keep independence + the monotone per-tile scan +
        // the in-file-order host append ⇒ the SAME kept SET, the SAME file ORDER.
        const long tile_M = steppe::device::decode_tile_snps(
            backend.capabilities().free_vram_bytes, P, tile.n_individuals, M);

        for (long s_lo = 0; s_lo < M; s_lo += tile_M) {
            const long tw = std::min(tile_M, M - s_lo);
            const std::size_t tws = static_cast<std::size_t>(tw);
            const std::size_t so = static_cast<std::size_t>(s_lo);

            DecodeTileView tview = view;   // full packed/bytes_per_record/ploidy…
            tview.n_snp = tws;             // …only the SNP-column count is the tile width.

            steppe::device::DeviceDecodeResult ddr = backend.decode_af_compact_filter(
                tview,
                std::span<const char>(snptab.ref.data() + so, tws),
                std::span<const char>(snptab.alt.data() + so, tws),
                std::span<const int>(snptab.chrom.data() + so, tws),
                std::span<const double>(snptab.genpos_morgans.data() + so, tws),
                std::span<const double>(snptab.physpos.data() + so, tws),
                filter, std::span<const std::size_t>(pop_individuals.data(),
                                                     pop_individuals.size()),
                kPloidyDiploid, maxmiss, s_lo);

            const long mk = ddr.M_kept;
            if (mk <= 0) continue;  // no SNP kept in this tile — nothing to append.

            // D2H this tile's compacted Q/V/N and FREE the resident tile buffers at the
            // end of the iteration (never accumulate a resident [P × M_kept] tensor —
            // that would be O(P × M) again). The host layout is column-major [P × mk]
            // (element (pop i, kept SNP d) at i + P·d), so consecutive kept SNPs are
            // P-strided; APPENDING each tile's arrays in file (tile) order reproduces
            // the single-shot [P × M_kept] contiguous layout EXACTLY, bit-for-bit.
            std::vector<double> Qt, Vt, Nt;
            ddr.to_host_qvn(Qt, Vt, Nt);
            Qk.insert(Qk.end(), Qt.begin(), Qt.end());
            Vk.insert(Vk.end(), Vt.begin(), Vt.end());
            Nk.insert(Nk.end(), Nt.begin(), Nt.end());
            chrom_kept.insert(chrom_kept.end(), ddr.chrom_kept.begin(),
                              ddr.chrom_kept.end());
            genpos_kept.insert(genpos_kept.end(), ddr.genpos_kept.begin(),
                               ddr.genpos_kept.end());
            physpos_kept.insert(physpos_kept.end(), ddr.physpos_kept.begin(),
                                ddr.physpos_kept.end());
            M_kept += mk;
        }

        if (M_kept <= 0) {
            throw std::invalid_argument(
                "extract_f2: every SNP was filtered out (0 of " + std::to_string(M) +
                " kept) — relax the filters");
        }
    } else {
        // CPU PARITY ORACLE: the verbatim host regime-B path (UNCHANGED).
        const DecodeResult dec = backend.decode_af(view);
        flt::DecodedTileSummaryInput fin;
        fin.q = dec.q.data();
        fin.v = dec.v.data();
        fin.n = dec.n.data();
        fin.P = P;
        fin.M = M;
        fin.pop_individuals = pop_individuals;
        fin.ploidy = kPloidyDiploid;  // used only by the DISABLED sample-axis predicate.

        FilterConfig class_filter = filter;
        class_filter.geno_max_missing = 1.0;  // pop-coverage maxmiss is applied below.
        flt::SnpMembership mem(class_filter);
        std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, class_filter, mem);

        // Pop-coverage maxmiss (AT2 population-axis): drop SNP s if the fraction of
        // selected pops with N(pop,s)==0 (no data) exceeds maxmiss. maxmiss>=1 keeps
        // every SNP; maxmiss==0 is the global intersection (the golden_fit0 SNP set).
        if (maxmiss < 1.0) {
            for (long s = 0; s < M; ++s) {
                if (!keep[static_cast<std::size_t>(s)]) continue;
                int n_missing_pops = 0;
                for (int p = 0; p < P; ++p) {
                    const std::size_t off =
                        static_cast<std::size_t>(p) + static_cast<std::size_t>(P) * static_cast<std::size_t>(s);
                    if (dec.n[off] <= 0.0) ++n_missing_pops;
                }
                const double frac_missing_pops =
                    static_cast<double>(n_missing_pops) / static_cast<double>(P);
                if (frac_missing_pops > maxmiss) keep[static_cast<std::size_t>(s)] = false;
            }
        }

        std::size_t n_kept = 0;
        for (bool k : keep) n_kept += k ? 1u : 0u;
        M_kept = static_cast<long>(n_kept);
        if (M_kept <= 0) {
            throw std::invalid_argument(
                "extract_f2: every SNP was filtered out (0 of " + std::to_string(M) +
                " kept) — relax the filters");
        }

        // Subset Q/V/N AND the parallel chrom/genpos in LOCKSTEP (the SNP axis must
        // stay aligned for assign_blocks; cmd_extract_f2.cpp:427-449).
        Qk.assign(static_cast<std::size_t>(P) * n_kept, 0.0);
        Vk.assign(static_cast<std::size_t>(P) * n_kept, 0.0);
        Nk.assign(static_cast<std::size_t>(P) * n_kept, 0.0);
        chrom_kept.reserve(n_kept);
        genpos_kept.reserve(n_kept);
        physpos_kept.reserve(n_kept);
        std::size_t d = 0;
        for (std::size_t s = 0; s < keep.size(); ++s) {
            if (!keep[s]) continue;
            for (int i = 0; i < P; ++i) {
                const std::size_t src =
                    static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * s;
                const std::size_t dst =
                    static_cast<std::size_t>(i) + static_cast<std::size_t>(P) * d;
                Qk[dst] = dec.q[src];
                Vk[dst] = dec.v[src];
                Nk[dst] = dec.n[src];
            }
            chrom_kept.push_back(snptab.chrom[s]);
            genpos_kept.push_back(snptab.genpos_morgans[s]);
            physpos_kept.push_back(snptab.physpos[s]);
            ++d;
        }
    }

    // ---- 3. assign_blocks over the KEPT SNP axis --------------------------------------
    // physpos_kept drives the AT2 bp block-fallback ONLY when genpos_kept is all zero
    // (a dataset with no genetic map — e.g. VCF/PLINK-derived); a real map (the AADR)
    // ignores it, so the partition — and every golden — is bit-identical (the pass gate).
    const steppe::core::BlockPartition partition = steppe::core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans, std::span<const double>(physpos_kept));
    if (partition.n_block <= 0) {
        throw std::invalid_argument(
            "extract_f2: assign_blocks produced 0 blocks (check blgsize and the .snp "
            "genetic positions)");
    }

    // ---- 4. compute_f2_blocks_multigpu_tiered (GPU; the UNIFIED adaptive entry) -------
    const MatView Q{Qk.data(), P, M_kept};
    const MatView V{Vk.data(), P, M_kept};
    const MatView N{Nk.data(), P, M_kept};
    device::F2BlocksOut dev_f2 = steppe::core::compute_f2_blocks_multigpu_tiered(
        resources, Q, V, N, partition, precision);

    // ---- 5. Materialize -> host tensor with REAL f2 + vpair (tier-agnostic) -----------
    F2ExtractResult out;
    out.f2 = dev_f2.to_host();
    out.pop_labels = std::move(pop_labels);
    out.n_snp_total = M;
    out.n_snp_kept = M_kept;
    out.n_pseudo_haploid = n_ph;
    out.n_diploid = n_dip;
    out.precision_tag = precision.kind;
    // Map the internal device tier -> the CUDA-free public ExtractTier mirror (observability;
    // the CLI summary / meta echo it so a --tier / STEPPE_FORCE_TIER override is visibly honored).
    switch (dev_f2.tier) {
        case device::OutputTier::Resident: out.tier = ExtractTier::Resident; break;
        case device::OutputTier::HostRam:  out.tier = ExtractTier::HostRam;  break;
        case device::OutputTier::Disk:     out.tier = ExtractTier::Disk;     break;
    }
    out.status = Status::Ok;
    return out;
}

}  // namespace steppe
