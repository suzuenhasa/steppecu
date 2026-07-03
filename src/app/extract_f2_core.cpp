// src/app/extract_f2_core.cpp
//
// The one real implementation of steppe::run_extract_f2 — the library
// genotype->f2 extract. It throws instead of exiting and holds no CUDA code, so
// the CLI and the Python bindings share this single copy of the chain.
//
// Reference: docs/reference/src_app_extract_f2_core.cpp.md

#include "steppe/extract.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/domain/block_partition_rule.hpp"
#include "core/fstats/f2_blocks_multigpu.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "core/internal/views.hpp"
#include "device/backend.hpp"
#include "device/decode_budget.hpp"
#include "device/device_decode_result.hpp"
#include "device/f2_blocks_out.hpp"
#include "device/resources.hpp"
#include "device/stream_f2_blocks.hpp"
#include "device/tier_select.hpp"
#include "steppe/config.hpp"

#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"
#include "io/filter/snp_filter.hpp"

namespace steppe {

namespace {

using steppe::core::MatView;
namespace flt = steppe::io::filter;

// AT2 ploidy values — reference §5
constexpr int kPloidyPseudoHaploid = 1;
constexpr int kPloidyDiploid = 2;

// decode_one_tile: the single regime-B decode call shared by every extract-f2 pass.
// Phase A (metadata), the Resident dense re-decode, and the Phase-B block source all
// route through this one function, so all passes decode byte-identically — the kept
// set and its file order cannot drift between passes. Reference §2
[[nodiscard]] steppe::device::DeviceDecodeResult decode_one_tile(
    steppe::ComputeBackend& backend, const DecodeTileView& base_view,
    const io::SnpTable& snptab, const FilterConfig& filter,
    const std::vector<std::size_t>& pop_individuals, double maxmiss,
    long s_lo, long tw) {
    const std::size_t tws = static_cast<std::size_t>(tw);
    const std::size_t so = static_cast<std::size_t>(s_lo);
    DecodeTileView tview = base_view;
    tview.n_snp = tws;
    return backend.decode_af_compact_filter(
        tview,
        std::span<const char>(snptab.ref.data() + so, tws),
        std::span<const char>(snptab.alt.data() + so, tws),
        std::span<const int>(snptab.chrom.data() + so, tws),
        std::span<const double>(snptab.genpos_morgans.data() + so, tws),
        std::span<const double>(snptab.physpos.data() + so, tws),
        filter,
        std::span<const std::size_t>(pop_individuals.data(), pop_individuals.size()),
        kPloidyDiploid, maxmiss, s_lo);
}

// Explicit-pops validation — reference §4
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

// run_extract_f2: the genotype->f2 pipeline — reference §2
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

    std::vector<std::string> pop_labels;
    pop_labels.reserve(static_cast<std::size_t>(P));
    for (const io::PopGroup& g : part.groups) pop_labels.push_back(g.label);

    std::vector<int> sample_ploidy;
    bool detect_on_device = false;
    switch (ploidy) {
        case ExtractPloidy::Auto:
            detect_on_device = true;
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
    view.detect_ploidy_on_device = detect_on_device;
    view.ploidy = kPloidyDiploid;

    std::size_t n_ph = 0, n_dip = 0;
    const std::vector<int> ploidy_for_counts =
        detect_on_device ? backend.detect_sample_ploidy_device(view) : sample_ploidy;
    for (int pl : ploidy_for_counts) (pl == kPloidyPseudoHaploid ? n_ph : n_dip)++;

    std::vector<std::size_t> pop_individuals(static_cast<std::size_t>(P));
    for (int p = 0; p < P; ++p) {
        pop_individuals[static_cast<std::size_t>(p)] =
            tile.pop_offsets[static_cast<std::size_t>(p) + 1] -
            tile.pop_offsets[static_cast<std::size_t>(p)];
    }
    const double maxmiss = filter.geno_max_missing;

    const bool on_device = (backend.capabilities().device_count > 0);
    std::vector<double> Qk, Vk, Nk;
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    std::vector<double> physpos_kept;
    long M_kept = 0;
    long device_tile_M = 0;

    if (on_device) {
        view.sample_ploidy = ploidy_for_counts.data();
        view.detect_ploidy_on_device = false;

        // ---- Phase A: metadata pass. Decode every file-order tile to determine the
        // kept set + its chrom/genpos/physpos, but DISCARD Q/V/N (each tile's device
        // decode is dropped at the end of the loop body). Host RAM stays O(M_kept)
        // metadata instead of the old O(P × M_kept) dense Q/V/N wall; Phase B below
        // re-decodes to feed f2. Reference §2a
        device_tile_M = steppe::device::decode_tile_snps(
            backend.capabilities().free_vram_bytes, P, tile.n_individuals, M);

        for (long s_lo = 0; s_lo < M; s_lo += device_tile_M) {
            const long tw = std::min(device_tile_M, M - s_lo);
            steppe::device::DeviceDecodeResult ddr = decode_one_tile(
                backend, view, snptab, filter, pop_individuals, maxmiss, s_lo, tw);

            const long mk = ddr.M_kept;
            if (mk <= 0) continue;

            chrom_kept.insert(chrom_kept.end(), ddr.chrom_kept.begin(),
                              ddr.chrom_kept.end());
            genpos_kept.insert(genpos_kept.end(), ddr.genpos_kept.begin(),
                               ddr.genpos_kept.end());
            physpos_kept.insert(physpos_kept.end(), ddr.physpos_kept.begin(),
                                ddr.physpos_kept.end());
            M_kept += mk;
            // ddr (and its resident VRAM Q/V/N) is released here — Q/V/N discarded.
        }

        if (M_kept <= 0) {
            throw std::invalid_argument(
                "extract_f2: every SNP was filtered out (0 of " + std::to_string(M) +
                " kept) — relax the filters");
        }
    } else {
        const DecodeResult dec = backend.decode_af(view);
        flt::DecodedTileSummaryInput fin;
        fin.q = dec.q.data();
        fin.v = dec.v.data();
        fin.n = dec.n.data();
        fin.P = P;
        fin.M = M;
        fin.pop_individuals = pop_individuals;
        fin.ploidy = kPloidyDiploid;

        FilterConfig class_filter = filter;
        class_filter.geno_max_missing = 1.0;
        flt::SnpMembership mem(class_filter);
        std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, class_filter, mem);

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

    const steppe::core::BlockPartition partition = steppe::core::assign_blocks(
        std::span<const int>(chrom_kept), std::span<const double>(genpos_kept),
        blgsize_morgans, std::span<const double>(physpos_kept));
    if (partition.n_block <= 0) {
        throw std::invalid_argument(
            "extract_f2: assign_blocks produced 0 blocks (check blgsize and the .snp "
            "genetic positions)");
    }

    device::F2BlocksOut dev_f2;
    if (on_device) {
        // Tier decision, now input-aware: the Resident output engine needs the full
        // dense P×M_kept host Q/V/N; select_output_tier's host-input clamp keeps large
        // models off that wall and routes them to the bounded block-source path.
        const std::size_t free_vram = backend.capabilities().free_vram_bytes;
        const std::size_t free_host = steppe::device::free_host_ram_bytes();
        const device::OutputTier tier = device::resolve_output_tier(
            resources.config.force_tier, std::getenv("STEPPE_FORCE_TIER"), P, M_kept,
            partition.n_block, free_vram, free_host);

        if (tier == device::OutputTier::Resident) {
            // ---- Phase B (Resident): re-decode into the dense host buffer, exactly as
            // Phase A once did, then run the UNCHANGED resident engine — byte-identical
            // to today for small/mid models. The re-decode is deterministic, so this
            // rebuilds the same Q/V/N the single-pass path produced. Reference §2a
            Qk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M_kept));
            Vk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M_kept));
            Nk.reserve(static_cast<std::size_t>(P) * static_cast<std::size_t>(M_kept));
            std::vector<double> Qt, Vt, Nt;
            for (long s_lo = 0; s_lo < M; s_lo += device_tile_M) {
                const long tw = std::min(device_tile_M, M - s_lo);
                steppe::device::DeviceDecodeResult ddr = decode_one_tile(
                    backend, view, snptab, filter, pop_individuals, maxmiss, s_lo, tw);
                if (ddr.M_kept <= 0) continue;
                ddr.to_host_qvn(Qt, Vt, Nt);
                Qk.insert(Qk.end(), Qt.begin(), Qt.end());
                Vk.insert(Vk.end(), Vt.begin(), Vt.end());
                Nk.insert(Nk.end(), Nt.begin(), Nt.end());
            }
            const MatView Q{Qk.data(), P, M_kept};
            const MatView V{Vk.data(), P, M_kept};
            const MatView N{Nk.data(), P, M_kept};
            dev_f2 = steppe::core::compute_f2_blocks_multigpu_tiered(
                resources, Q, V, N, partition, precision);
        } else {
            // ---- Phase B (HostRam/Disk): reuse the UNCHANGED streamed f2 engine, but
            // feed each of its per-chunk tiles by RE-DECODING those kept columns on-device
            // instead of copying them from a dense host Q/V/N (never materialized here).
            // Host RAM stays O(M_kept metadata); the streamed engine's bytes — and thus
            // the f2 cache — are bit-identical to the resident path. Reference §2a
            //
            // kept_to_raw[k] = raw .snp row of kept column k. A single forward two-pointer
            // scan over the file-order raw rows and the Phase-A kept metadata: the kept set
            // is a monotone subset of the raw rows, so chrom+physpos match in order.
            std::vector<long> kept_to_raw(static_cast<std::size_t>(M_kept));
            {
                long k = 0;
                for (long r = 0; r < M && k < M_kept; ++r) {
                    const std::size_t rz = static_cast<std::size_t>(r);
                    const std::size_t kz = static_cast<std::size_t>(k);
                    if (snptab.chrom[rz] == chrom_kept[kz] &&
                        snptab.physpos[rz] == physpos_kept[kz]) {
                        kept_to_raw[kz] = r;
                        ++k;
                    }
                }
                if (k != M_kept)
                    throw std::runtime_error(
                        "extract_f2: kept-to-raw scan matched " + std::to_string(k) +
                        " of " + std::to_string(M_kept) + " kept columns (the kept set is "
                        "not a monotone subset of the raw .snp rows)");
            }

            steppe::device::RedecodeSource redecode;
            redecode.base_view = &view;
            redecode.ref = snptab.ref.data();
            redecode.alt = snptab.alt.data();
            redecode.chrom = snptab.chrom.data();
            redecode.genpos = snptab.genpos_morgans.data();
            redecode.physpos = snptab.physpos.data();
            redecode.filter = &filter;
            redecode.pop_individuals = pop_individuals.data();
            redecode.n_pop = static_cast<std::size_t>(P);
            redecode.maxmiss = maxmiss;
            redecode.kept_to_raw = kept_to_raw.data();
            redecode.M_kept = M_kept;

            dev_f2 = steppe::core::compute_f2_blocks_multigpu_tiered(
                resources, MatView{nullptr, P, M_kept}, MatView{nullptr, P, M_kept},
                MatView{nullptr, P, M_kept}, partition, precision, &redecode);
        }
    } else {
        const MatView Q{Qk.data(), P, M_kept};
        const MatView V{Vk.data(), P, M_kept};
        const MatView N{Nk.data(), P, M_kept};
        dev_f2 = steppe::core::compute_f2_blocks_multigpu_tiered(
            resources, Q, V, N, partition, precision);
    }

    F2ExtractResult out;
    // For HostRam, the streamed result already lives in dev_f2.host — MOVE it out
    // rather than to_host()-copying it. to_host() allocates a second full P²·nb·2
    // (f2+vpair) host tensor, which for large models doubles host RAM and OOMs (a
    // top-2000 2M model is ~46 GB per copy). Resident/Disk still materialize via
    // to_host() (from VRAM / the on-disk cache respectively).
    if (dev_f2.tier == device::OutputTier::HostRam) {
        out.f2 = std::move(dev_f2.host);
    } else {
        out.f2 = dev_f2.to_host();
    }
    out.pop_labels = std::move(pop_labels);
    out.n_snp_total = M;
    out.n_snp_kept = M_kept;
    out.n_pseudo_haploid = n_ph;
    out.n_diploid = n_dip;
    out.precision_tag = precision.kind;
    switch (dev_f2.tier) {
        case device::OutputTier::Resident: out.tier = ExtractTier::Resident; break;
        case device::OutputTier::HostRam:  out.tier = ExtractTier::HostRam;  break;
        case device::OutputTier::Disk:     out.tier = ExtractTier::Disk;     break;
    }
    out.status = Status::Ok;
    return out;
}

}  // namespace steppe
