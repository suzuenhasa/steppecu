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
#include "core/internal/views.hpp"                // steppe::core::MatView
#include "device/backend.hpp"                     // DecodeTileView, DecodeResult (CUDA-FREE)
#include "device/f2_blocks_out.hpp"               // F2BlocksOut (CUDA-FREE)
#include "device/resources.hpp"                   // Resources (CUDA-FREE)
#include "steppe/config.hpp"                      // Precision, FilterConfig

#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/ploidy_detect.hpp"   // detect_sample_ploidy (AT2 adjust_pseudohaploid)
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

    // ---- 1. Open the .geno + read .ind (selection) + .snp -----------------------------
    io::GenoReader reader(geno);
    const std::size_t n_present = reader.records_present();
    const io::IndPartition part = io::read_ind(ind, sel, n_present);
    validate_explicit_pops(sel, part);

    const io::SnpTable snptab = io::read_snp(snp, SIZE_MAX);
    const std::size_t M0 = std::min(reader.header().n_snp, snptab.count);
    const io::GenotypeTile tile = reader.read_tile(part, 0, M0);

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

    // ---- PER-SAMPLE PLOIDY (the f2 pseudo-haploid fix; AT2 adjust_pseudohaploid) -------
    std::vector<int> sample_ploidy;
    std::size_t n_ph = 0, n_dip = 0;
    switch (ploidy) {
        case ExtractPloidy::Auto:
            sample_ploidy = io::detect_sample_ploidy(tile);
            break;
        case ExtractPloidy::PseudoHaploid:
            sample_ploidy.assign(tile.n_individuals, kPloidyPseudoHaploid);
            break;
        case ExtractPloidy::Diploid:
            sample_ploidy.assign(tile.n_individuals, kPloidyDiploid);
            break;
    }
    for (int pl : sample_ploidy) (pl == kPloidyPseudoHaploid ? n_ph : n_dip)++;

    // ---- 5. Decode + REGIME-B FILTER + lockstep Q/V/N compaction ----------------------
    steppe::ComputeBackend& backend = *resources.gpus.front().backend;

    DecodeTileView view;
    view.packed = tile.packed.data();
    view.bytes_per_record = tile.bytes_per_record;
    view.n_snp = tile.n_snp;
    view.n_individuals = tile.n_individuals;
    view.pop_offsets = tile.pop_offsets.data();
    view.n_pop = P;
    view.sample_ploidy = sample_ploidy.data();
    view.ploidy = kPloidyDiploid;  // unused (sample_ploidy non-null); the safe default.

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
    const bool resident = (backend.capabilities().device_count > 0);
    std::vector<double> Qk, Vk, Nk;
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    long M_kept = 0;

    if (resident) {
        const std::size_t Mu = static_cast<std::size_t>(M);
        steppe::device::DeviceDecodeResult ddr = backend.decode_af_compact_filter(
            view,
            std::span<const char>(snptab.ref.data(), Mu),
            std::span<const char>(snptab.alt.data(), Mu),
            std::span<const int>(snptab.chrom.data(), Mu),
            std::span<const double>(snptab.genpos_morgans.data(), Mu),
            filter, std::span<const std::size_t>(pop_individuals.data(),
                                                 pop_individuals.size()),
            kPloidyDiploid, maxmiss);
        M_kept = ddr.M_kept;
        if (M_kept <= 0) {
            throw std::invalid_argument(
                "extract_f2: every SNP was filtered out (0 of " + std::to_string(M) +
                " kept) — relax the filters");
        }
        // The small compacted Q/V/N D2H (the regime-B read-back; the only host copy,
        // analogous to the regime-A chrom/genpos D2H — just the compacted arrays).
        ddr.to_host_qvn(Qk, Vk, Nk);
        chrom_kept = std::move(ddr.chrom_kept);
        genpos_kept = std::move(ddr.genpos_kept);
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
            ++d;
        }
    }

    // ---- 7. assign_blocks over the KEPT SNP axis --------------------------------------
    const steppe::core::BlockPartition partition = steppe::core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans);
    if (partition.n_block <= 0) {
        throw std::invalid_argument(
            "extract_f2: assign_blocks produced 0 blocks (check blgsize and the .snp "
            "genetic positions)");
    }

    // ---- 8. compute_f2_blocks_multigpu_tiered (GPU; the UNIFIED adaptive entry) -------
    const MatView Q{Qk.data(), P, M_kept};
    const MatView V{Vk.data(), P, M_kept};
    const MatView N{Nk.data(), P, M_kept};
    device::F2BlocksOut dev_f2 = steppe::core::compute_f2_blocks_multigpu_tiered(
        resources, Q, V, N, partition, precision);

    // ---- 8b. Materialize -> host tensor with REAL f2 + vpair (tier-agnostic) ----------
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
