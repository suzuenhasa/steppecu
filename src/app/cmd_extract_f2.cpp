// src/app/cmd_extract_f2.cpp
//
// The `steppe extract-f2` command (M(cli-4); cli-bindings.md §4.1, §2.6 chain, §4.3
// dir). PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate): the
// GPU is reached ONLY through the CUDA-FREE seams. main() owns stdout/stderr
// (architecture.md §10 — the library never prints).
//
// THE CHAIN (mirrors tests/reference/test_decode_equivalence.cu + test_filter_oracle.cu
// + test_f2_blocks_equivalence.cu, all CUDA-free seams):
//   1. GenoReader(geno)              -> records_present()
//   2. read_ind(ind, PopSelection, n_present) -> IndPartition (groups sorted ASC by
//      label = the P-axis order = pops.txt order). Explicit names validated here.
//   3. read_snp(snp, SIZE_MAX)       -> SnpTable (chrom, genpos_morgans, ref, alt)
//   4. reader.read_tile(part, 0, M)  -> GenotypeTile (packed bytes, pop-contiguous)
//   5. backend->decode_af(view)      -> DecodeResult Q/V/N [P x M] (the GPU decodes)
//   6. filters (when non-default)    -> per-SNP keep mask, subset Q/V/N + snptab axis
//   7. assign_blocks(chrom,genpos,blgsize) -> BlockPartition
//   8. compute_f2_blocks_multigpu_device(resources, Q,V,N, partition, precision)
//      -> DeviceF2Blocks (resident) -> to_host() -> F2BlockTensor (REAL f2 + vpair)
//   9. write_f2_dir(out, tensor, labels, meta)  -> f2.bin (REAL vpair) + pops.txt + meta.json
#include "app/cmd_extract_f2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "app/f2_dir_writer.hpp"
#include "core/config/exit_code.hpp"
#include "core/domain/block_partition_rule.hpp"  // assign_blocks, block_size_cm_to_morgans
#include "core/fstats/f2_blocks_multigpu.hpp"     // compute_f2_blocks_multigpu_device (CUDA-FREE)
#include "core/internal/views.hpp"                // steppe::core::MatView
#include "device/backend.hpp"                     // DecodeTileView, DecodeResult, BackendCapabilities (CUDA-FREE)
#include "device/device_f2_blocks.hpp"            // DeviceF2Blocks (CUDA-FREE)
#include "device/resources.hpp"                   // Resources, build_resources (CUDA-FREE)
#include "device/tier_select.hpp"                 // select_output_tier, free_host_ram_bytes (CUDA-FREE, --dry-run)
#include "steppe/config.hpp"                      // Precision, FilterConfig, kCentimorgansPerMorgan
#include "steppe/error.hpp"                       // steppe::Status

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/ploidy_detect.hpp"   // detect_sample_ploidy (AT2 adjust_pseudohaploid)
#include "io/snp_reader.hpp"
#include "io/filter/include_exclude.hpp"
#include "io/filter/snp_filter.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;
using steppe::core::MatView;
namespace flt = steppe::io::filter;

// AT2 ploidy values (mirror core::kPloidy{PseudoHaploid,Diploid}; the app names them
// here so the resolved per-sample vector / forced-uniform fallback read clearly).
constexpr int kPloidyPseudoHaploid = 1;
constexpr int kPloidyDiploid = 2;

// A human label for the resolved precision (recorded in meta.json; cli-bindings §4.3
// — the ENGAGED tag). Mirrors result_emit's emu/fp64/tf32 vocabulary.
[[nodiscard]] const char* precision_label(const Precision& p) {
    switch (p.kind) {
        case Precision::Kind::EmulatedFp64: return "emu";
        case Precision::Kind::Tf32:         return "tf32";
        case Precision::Kind::Fp64:         return "fp64";
    }
    return "fp64";
}

// A human echo of the pop selection request (the resolved labels live in pops.txt;
// this records which MODE/threshold produced them — cli-bindings §4.3).
[[nodiscard]] std::string pop_selection_str(const io::PopSelection& sel) {
    switch (sel.mode) {
        case io::PopSelection::Mode::Explicit: {
            std::string s = "explicit:";
            for (std::size_t i = 0; i < sel.labels.size(); ++i) {
                if (i) s += ",";
                s += sel.labels[i];
            }
            return s;
        }
        case io::PopSelection::Mode::AutoTopK: return "auto-top:" + std::to_string(sel.k);
        case io::PopSelection::Mode::MinN:     return "min-n:" + std::to_string(sel.min_n);
    }
    return "unknown";
}

// Validate that every requested Explicit pop label is present in the resolved
// partition (cli-bindings.md §4.2: an unknown pop name is InvalidConfig fail-fast
// naming the offending label). read_ind already drops unknown labels silently (and
// only throws on a FULLY empty selection), so we resolve against the groups labels
// here — the prompt's contract, mirroring pop_resolver's unknown-name pattern. Sets
// `offending` and returns false on the first missing label.
[[nodiscard]] bool validate_explicit_pops(const io::PopSelection& sel,
                                          const io::IndPartition& part,
                                          std::string& offending) {
    if (sel.mode != io::PopSelection::Mode::Explicit) return true;
    for (const std::string& want : sel.labels) {
        const bool found = std::any_of(
            part.groups.begin(), part.groups.end(),
            [&want](const io::PopGroup& g) { return g.label == want; });
        if (!found) { offending = want; return false; }
    }
    return true;
}

}  // namespace

int run_extract_f2_command(const cfg::RunConfig& config) {
    // ---- 0. Required inputs -------------------------------------------------------
    if (config.geno().empty() || config.snp().empty() || config.ind().empty()) {
        std::fprintf(stderr,
                     "steppe extract-f2: the genotype triple is required "
                     "(use --prefix P or all of --geno/--snp/--ind)\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.out_dir().empty() && !config.dry_run()) {
        std::fprintf(stderr, "steppe extract-f2: --out DIR is required\n");
        return cfg::kExitInvalidConfig;
    }
    const io::PopSelection& sel = config.pop_selection();
    const bool no_selection =
        (sel.mode == io::PopSelection::Mode::AutoTopK && sel.k == 0);
    if (no_selection) {
        std::fprintf(stderr,
                     "steppe extract-f2: a population selection is required "
                     "(--pops a,b,.. | --auto-top-k K | --min-n N)\n");
        return cfg::kExitInvalidConfig;
    }

    // ---- 1. Open the .geno + read .ind (selection) + .snp -------------------------
    io::IndPartition part;
    io::SnpTable snptab;
    io::GenotypeTile tile;
    std::size_t n_present = 0;
    try {
        io::GenoReader reader(config.geno());
        n_present = reader.records_present();
        part = io::read_ind(config.ind(), sel, n_present);

        // Explicit-name validation (cli-bindings.md §4.2). read_ind silently drops an
        // unknown label, so verify the request against the resolved groups here.
        std::string offending;
        if (!validate_explicit_pops(sel, part, offending)) {
            std::fprintf(stderr,
                         "steppe extract-f2: --pops contains an unknown population "
                         "'%s' (not present in %s)\n",
                         offending.c_str(), config.ind().c_str());
            return cfg::kExitInvalidConfig;
        }

        snptab = io::read_snp(config.snp(), SIZE_MAX);
        const std::size_t M0 = std::min(reader.header().n_snp, snptab.count);
        tile = reader.read_tile(part, 0, M0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe extract-f2: input error: %s\n", e.what());
        return cfg::kExitIoError;
    }

    const int P = static_cast<int>(tile.n_pop());
    const long M = static_cast<long>(tile.n_snp);
    if (P <= 0 || M <= 0) {
        std::fprintf(stderr,
                     "steppe extract-f2: empty selection (P=%d) or no SNPs (M=%ld)\n", P, M);
        return cfg::kExitInvalidConfig;
    }
    if (snptab.count < static_cast<std::size_t>(M)) {
        std::fprintf(stderr,
                     "steppe extract-f2: .snp has %zu rows but the tile spans %ld SNPs "
                     "(the .snp/.geno SNP axes must agree)\n", snptab.count, M);
        return cfg::kExitIoError;
    }

    // The P labels in P-axis index order (= pops.txt) — the name<->index map the
    // engine lacks (cli-bindings.md §1).
    std::vector<std::string> pop_labels;
    pop_labels.reserve(static_cast<std::size_t>(P));
    for (const io::PopGroup& g : part.groups) pop_labels.push_back(g.label);

    // ---- PER-SAMPLE PLOIDY (the f2 pseudo-haploid fix; AT2 adjust_pseudohaploid) ---
    // Default --ploidy auto: detect each gathered sample's ploidy from the tile (a het
    // call in the leading SNPs ⇒ diploid, none ⇒ pseudo-haploid) — AT2-parity per
    // SAMPLE, so mixed-ploidy pops (Turkey_N/Serbia/Yamnaya/Karitiana) decode correctly.
    // --ploidy 1/2 force a uniform vector (pseudo-haploid / diploid) for every sample.
    // The vector is parallel to the gathered sample axis and fed to decode_af via
    // DecodeTileView::sample_ploidy (per-sample weighting AC += code/(3-ploidy),
    // N += ploidy). n_ph/n_dip are echoed below for observability.
    std::vector<int> sample_ploidy;
    std::size_t n_ph = 0, n_dip = 0;
    switch (config.ploidy()) {
        case cfg::PloidyMode::Auto:
            sample_ploidy = io::detect_sample_ploidy(tile);
            break;
        case cfg::PloidyMode::PseudoHaploid:
            sample_ploidy.assign(tile.n_individuals, kPloidyPseudoHaploid);
            break;
        case cfg::PloidyMode::Diploid:
            sample_ploidy.assign(tile.n_individuals, kPloidyDiploid);
            break;
    }
    for (int pl : sample_ploidy) (pl == kPloidyPseudoHaploid ? n_ph : n_dip)++;

    // The engaged precision (the requested DeviceConfig.precision; the resident path
    // honors it or downgrades to native — recorded as the tag in meta.json).
    const Precision precision = config.device().precision;
    const FilterConfig& filter = config.filter();

    // ---- DRY RUN: report sizes / tier / precision, NO compute ---------------------
    // The tier estimate uses the CUDA-free vram-budget seam (select_output_tier); the
    // free-VRAM probe routes through build_resources' BackendCapabilities (no direct
    // CUDA call). On a no-GPU box this still reports the static shape and skips the
    // tier (the §4.5 dry-run is a planning aid).
    if (config.dry_run()) {
        std::printf("steppe extract-f2 --dry-run:\n");
        std::printf("  geno:       %s\n", config.geno().c_str());
        std::printf("  snp:        %s (%ld SNPs read)\n", config.snp().c_str(), M);
        std::printf("  ind:        %s (%zu records present)\n", config.ind().c_str(), n_present);
        std::printf("  selection:  %s -> P = %d populations\n", pop_selection_str(sel).c_str(), P);
        std::printf("  precision:  %s (mantissa_bits=%d)\n", precision_label(precision), precision.mantissa_bits);
        std::printf("  blgsize:    %.4g Morgans (%.4g cM)\n",
                    config.blgsize_cm() / kCentimorgansPerMorgan, config.blgsize_cm());
        std::printf("  filters:    maf>=%.4g maxmiss<=%.4g autosomes_only=%d drop_mono=%d transversions_only=%d\n",
                    filter.maf_min, filter.geno_max_missing,
                    filter.autosomes_only ? 1 : 0, filter.drop_monomorphic ? 1 : 0,
                    filter.transversions_only ? 1 : 0);
        // Tier estimate at the FULL M (an upper bound; the kept-M after filtering is
        // <= M, so the resident tier verdict here is conservative).
        try {
            device::Resources resources = device::build_resources(config.device());
            if (!resources.gpus.empty()) {
                const auto& caps = resources.gpus.front().caps;
                // n_block upper bound is unknown without assign_blocks; report the
                // resident result bytes at the full M's worst-case (P*P*n_block*16);
                // we approximate n_block via assign_blocks below the dry-run only if
                // cheap — here we report VRAM + the per-block slab cost so the user
                // sees the envelope.
                const std::size_t slab_bytes =
                    static_cast<std::size_t>(P) * static_cast<std::size_t>(P) * 2u * sizeof(double);
                std::printf("  device:     GPU %d, free VRAM = %zu MiB, total = %zu MiB\n",
                            resources.gpus.front().device_id,
                            caps.free_vram_bytes >> 20, caps.total_vram_bytes >> 20);
                std::printf("  f2 slab:    %zu bytes / block (f2 + vpair, FP64); resident if "
                            "P*P*n_block*16 fits ~70%% of free VRAM\n", slab_bytes);
            } else {
                std::printf("  device:     (no CUDA device visible)\n");
            }
        } catch (const std::exception& e) {
            std::printf("  device:     probe failed: %s\n", e.what());
        }
        return cfg::kExitOk;
    }

    // ---- SOURCE-PROVENANCE HASH (overlapped, --hash opt-in; default OFF) ----------
    // The whole-source-.geno SHA-256 is a ~tens-of-seconds whole-file read+compress that
    // historically dominated extract-f2 (~37s of ~41s on the 6.7 GB 1240K .geno) yet
    // only produces a provenance value (it once caught a corrupt golden via a sha
    // mismatch), so it is SKIPPED by default. With --hash we compute it, and since it
    // depends ONLY on the .geno path (independent of the decode/f2 GPU pipeline) we run
    // it on a BACKGROUND THREAD started HERE, before the GPU work, and JOIN it just
    // before meta.json is written — the hash overlaps the decode+filter+f2+D2H wall time
    // instead of adding to it. The small .snp/.ind shas are left to the writer (cheap).
    const bool hash_source = config.hash_source();
    std::string geno_sha;            // filled by the worker iff hash_source
    std::thread geno_hash_thread;    // joined before write_f2_dir (never detached)
    if (hash_source) {
        const std::string geno_path = config.geno();
        geno_hash_thread = std::thread([geno_path, &geno_sha]() {
            geno_sha = sha256_file(std::filesystem::path(geno_path));
        });
    }
    // RAII safety: if any early return / exception fires before the explicit join below,
    // make sure the worker is joined so the thread is never destroyed un-joined.
    struct ThreadJoiner {
        std::thread& t;
        ~ThreadJoiner() { if (t.joinable()) t.join(); }
    } geno_hash_joiner{geno_hash_thread};

    // ---- 5. Decode the tile on the GPU backend -> Q/V/N [P x M] -------------------
    device::DeviceF2Blocks dev_f2;
    std::vector<double> Qk, Vk, Nk;  // the (possibly filtered) Q/V/N (own storage)
    std::vector<int> chrom_kept;
    std::vector<double> genpos_kept;
    long M_kept = 0;
    std::size_t n_kept = 0;
    Precision engaged = precision;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe extract-f2: no CUDA device available (steppe is a GPU "
                         "product; a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        // The decode runs on the GPU backend bound to the first device.
        steppe::ComputeBackend& backend = *resources.gpus.front().backend;

        DecodeTileView view;
        view.packed = tile.packed.data();
        view.bytes_per_record = tile.bytes_per_record;
        view.n_snp = tile.n_snp;
        view.n_individuals = tile.n_individuals;
        view.pop_offsets = tile.pop_offsets.data();
        view.n_pop = P;
        // PER-SAMPLE ploidy (AT2 adjust_pseudohaploid; the f2 pseudo-haploid fix):
        // the decode folds it as AC += code/(3-ploidy), N += ploidy per sample, so
        // mixed-ploidy pops are correct. sample_ploidy is always populated (auto /
        // forced) above; the scalar fallback (kept diploid) is never reached here.
        view.sample_ploidy = sample_ploidy.data();
        view.ploidy = kPloidyDiploid;  // unused (sample_ploidy non-null), kept as the safe default
        const DecodeResult dec = backend.decode_af(view);

        // ---- 6. Filters: build the per-SNP keep mask, subset the SNP axis ---------
        // The keep-mask path is parity-NEUTRAL by construction (test_filter_oracle's
        // drop-equals-mask invariant). It handles the allele-class drops, MAF, the
        // flag-gated drops, and autosomes_only (the extract-f2 default = AT2 extract_f2's
        // default auto_only=TRUE = chr 1-22).
        //
        // THE maxmiss SEMANTIC (load-bearing; ROADMAP.md M2 line "AT2 maxmiss is
        // POPULATION-axis"). AT2 extract_f2 keeps SNP s iff (fraction of POPULATIONS
        // with NO genotyped individual at s) <= maxmiss — NOT the sample-axis missing
        // fraction the FilterConfig::geno_max_missing predicate computes. maxmiss=0 is
        // the GLOBAL POP INTERSECTION (every selected pop has data at s), which is the
        // SNP set AT2's golden_fit0 f2 uses. So we apply the pop-coverage maxmiss as a
        // SEPARATE per-SNP test here and force build_snp_keep_mask's geno_max_missing to
        // the no-op (1.0) so the sample-axis predicate does not double-filter.
        std::vector<std::size_t> pop_individuals(static_cast<std::size_t>(P));
        for (int p = 0; p < P; ++p) {
            pop_individuals[static_cast<std::size_t>(p)] =
                tile.pop_offsets[static_cast<std::size_t>(p) + 1] -
                tile.pop_offsets[static_cast<std::size_t>(p)];
        }
        flt::DecodedTileSummaryInput fin;
        fin.q = dec.q.data();
        fin.v = dec.v.data();
        fin.n = dec.n.data();
        fin.P = P;
        fin.M = M;
        fin.pop_individuals = pop_individuals;
        // fin.ploidy is used ONLY by the SAMPLE-axis missing predicate
        // (nonmissing_indiv = Σ N/ploidy), which the extract-f2 path DISABLES below
        // (class_filter.geno_max_missing = 1.0 ⇒ no-op); the pooled MAF it also feeds
        // is convention-invariant (Q·N/N). The AT2 pop-coverage maxmiss (applied
        // separately on N>0) and the f2 het-correction N (now per-sample, via decode)
        // do NOT read this. Kept diploid (a valid {1,2} value) for the disabled path;
        // a future per-sample sample-axis maxmiss would need the per-sample vector here.
        fin.ploidy = kPloidyDiploid;

        FilterConfig class_filter = filter;
        class_filter.geno_max_missing = 1.0;  // pop-coverage maxmiss is applied below
        flt::SnpMembership mem(class_filter);
        std::vector<bool> keep = flt::build_snp_keep_mask(fin, snptab, class_filter, mem);

        // Pop-coverage maxmiss (AT2 population-axis): drop SNP s if the fraction of
        // selected pops with N(pop,s)==0 (no data) exceeds maxmiss. maxmiss>=1 keeps
        // every SNP (the pairwise-complete path); maxmiss==0 is the global intersection.
        const double maxmiss = filter.geno_max_missing;
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

        for (bool k : keep) n_kept += k ? 1u : 0u;
        M_kept = static_cast<long>(n_kept);
        if (M_kept <= 0) {
            std::fprintf(stderr,
                         "steppe extract-f2: every SNP was filtered out "
                         "(0 of %ld kept) — relax the filters\n", M);
            return cfg::kExitInvalidConfig;
        }

        // Subset Q/V/N AND the parallel chrom/genpos in LOCKSTEP (the SNP axis must
        // stay aligned for assign_blocks; mirrors test_filter_oracle::drop_columns).
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

        // ---- 7. assign_blocks over the KEPT SNP axis ------------------------------
        const double bs_morgans =
            steppe::core::block_size_cm_to_morgans(config.blgsize_cm());
        const steppe::core::BlockPartition partition = steppe::core::assign_blocks(
            std::span<const int>(chrom_kept), std::span<const double>(genpos_kept), bs_morgans);
        if (partition.n_block <= 0) {
            std::fprintf(stderr,
                         "steppe extract-f2: assign_blocks produced 0 blocks "
                         "(check --blgsize and the .snp genetic positions)\n");
            return cfg::kExitInvalidConfig;
        }

        // ---- 8. compute_f2_blocks_multigpu_device (GPU, resident) -----------------
        const MatView Q{Qk.data(), P, M_kept};
        const MatView V{Vk.data(), P, M_kept};
        const MatView N{Nk.data(), P, M_kept};
        dev_f2 = steppe::core::compute_f2_blocks_multigpu_device(
            resources, Q, V, N, partition, precision);
        // The engaged precision reflects what the backend HONORED (the resident path
        // downgrades emu -> native if emulation is not honorable on this build/device).
        engaged = precision;
        (void)engaged;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe extract-f2: device error: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    // ---- 8b. Materialize (the ONE D2H) -> host tensor with REAL f2 + vpair --------
    F2BlockTensor host_f2;
    try {
        host_f2 = dev_f2.to_host();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe extract-f2: materialize (D2H) failed: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    // ---- 9. Write the f2_blocks dir (f2.bin REAL vpair + pops.txt + meta.json) -----
    F2DirMeta meta;
#ifdef STEPPE_VERSION
    meta.steppe_version = STEPPE_VERSION;
#else
    meta.steppe_version = "0.0.0";
#endif
    meta.precision_tag = precision_label(engaged);
    meta.precision_mantissa_bits = engaged.mantissa_bits;
    meta.blgsize_cm = config.blgsize_cm();
    meta.n_block = host_f2.n_block;
    meta.P = host_f2.P;
    meta.n_snp_total = M;
    meta.n_snp_kept = M_kept;
    meta.maf_min = filter.maf_min;
    meta.geno_max_missing = filter.geno_max_missing;
    meta.mind_max_missing = filter.mind_max_missing;
    meta.autosomes_only = filter.autosomes_only;
    meta.drop_monomorphic = filter.drop_monomorphic;
    meta.transversions_only = filter.transversions_only;
    meta.geno_path = config.geno();
    meta.snp_path = config.snp();
    meta.ind_path = config.ind();
    meta.pop_selection = pop_selection_str(sel);
    // Source-provenance shas (the --hash opt-in; default OFF). When requested, JOIN the
    // background geno-hash worker HERE (its wall time overlapped the GPU pipeline above)
    // and PRE-fill meta.geno_sha256 so the writer skips re-hashing the big .geno; the
    // writer fills the small .snp/.ind shas. When OFF, hash_source_files stays false and
    // every *_sha256 stays empty — meta.json records source_hash_computed:false so the
    // absence is recognizably DELIBERATE (see f2_dir_writer.cpp / cli-bindings.md §4.3).
    meta.hash_source_files = hash_source;
    if (hash_source) {
        if (geno_hash_thread.joinable()) geno_hash_thread.join();
        meta.geno_sha256 = geno_sha;  // "" if the .geno could not be opened
    }

    const F2DirWriteResult wr =
        write_f2_dir(config.out_dir(), host_f2, pop_labels, meta);
    if (!wr.ok) {
        std::fprintf(stderr, "steppe extract-f2: %s\n", wr.error.c_str());
        return cfg::kExitIoError;
    }

    std::printf("steppe extract-f2: wrote %s\n", config.out_dir().c_str());
    std::printf("  P = %d populations, %d blocks, %ld of %ld SNPs kept\n",
                host_f2.P, host_f2.n_block, M_kept, M);
    std::printf("  precision = %s, blgsize = %.4g Morgans (%.4g cM)\n",
                precision_label(engaged),
                config.blgsize_cm() / kCentimorgansPerMorgan, config.blgsize_cm());
    const char* ploidy_mode = (config.ploidy() == cfg::PloidyMode::Auto) ? "auto"
                            : (config.ploidy() == cfg::PloidyMode::PseudoHaploid) ? "1 (forced pseudo-haploid)"
                            : "2 (forced diploid)";
    std::printf("  ploidy = %s: %zu pseudo-haploid + %zu diploid samples\n",
                ploidy_mode, n_ph, n_dip);
    if (!wr.f2_cache_id.empty()) {
        std::printf("  f2_cache_id = %s\n", wr.f2_cache_id.c_str());
    }
    return cfg::kExitOk;
}

}  // namespace steppe::app
