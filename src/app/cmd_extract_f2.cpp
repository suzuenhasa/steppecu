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
//   8. compute_f2_blocks_multigpu_tiered(resources, Q,V,N, partition, precision)
//      -> F2BlocksOut (the UNIFIED adaptive entry: Resident is the device-resident path
//      UNCHANGED — byte-identical to the small goldens — while HostRam/Disk stream the
//      SNP-tile input so high-P full-autosome runs that OOMd the resident feeder
//      complete; the tier is auto-selected from runtime free VRAM/RAM, or pinned by
//      --tier / config.force_tier / STEPPE_FORCE_TIER) -> to_host() -> F2BlockTensor
//   9. write_f2_dir(out, tensor, labels, meta)  -> f2.bin (REAL vpair) + pops.txt + meta.json
#include "app/cmd_extract_f2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>   // std::getenv (STEPPE_FORCE_TIER, the --dry-run tier estimate)
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "app/f2_dir_writer.hpp"
#include "core/config/exit_code.hpp"
#include "core/domain/block_partition_rule.hpp"  // assign_blocks, block_size_cm_to_morgans (--dry-run sizing)
#include "device/resources.hpp"                   // Resources, build_resources (CUDA-FREE)
#include "device/tier_select.hpp"                 // resolve_output_tier, OutputTier, free_host_ram_bytes (CUDA-FREE, --dry-run)
#include "steppe/config.hpp"                      // Precision, FilterConfig, kCentimorgansPerMorgan
#include "steppe/error.hpp"                       // steppe::Status
#include "steppe/extract.hpp"                     // run_extract_f2 + F2ExtractResult (the library entry; CUDA-FREE)

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

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

// A human label for an OutputTier (echoed in the --dry-run plan and the post-run
// summary so the operator sees WHICH tier was selected — Resident is the existing
// device-resident path; HostRam/Disk are the M5 SNP-tile input-streaming tiers that
// keep the GPU peak independent of M; cli-bindings.md §4.3, §4.5).
[[nodiscard]] const char* tier_label(device::OutputTier t) {
    switch (t) {
        case device::OutputTier::Resident: return "resident";
        case device::OutputTier::HostRam:  return "host";
        case device::OutputTier::Disk:     return "disk";
    }
    return "resident";
}

// The same label over the CUDA-free ExtractTier mirror the library entry returns (the
// post-run summary echoes the tier the tiered compute actually used).
[[nodiscard]] const char* tier_label(steppe::ExtractTier t) {
    switch (t) {
        case steppe::ExtractTier::Resident: return "resident";
        case steppe::ExtractTier::HostRam:  return "host";
        case steppe::ExtractTier::Disk:     return "disk";
    }
    return "resident";
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
    std::size_t n_present = 0;
    long M = 0;
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
        // SIZING + VALIDATION ONLY (the dry-run + .snp/.geno-axis checks). The P/M the
        // tile would report are derivable from the partition + header WITHOUT reading
        // (and, for GENO, WITHOUT transposing) a tile: P == #selected pops, M == the
        // common SNP prefix. The REAL canonical-tile read (TGENO direct or GENO gather
        // + on-device transpose, M-FR-2) happens inside steppe::run_extract_f2 below.
        // Deriving P/M here keeps this format-agnostic — a GENO prefix no longer throws
        // at an up-front individual-major read_tile (which targets TGENO only).
        M = static_cast<long>(std::min(reader.header().n_snp, snptab.count));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe extract-f2: input error: %s\n", e.what());
        return cfg::kExitIoError;
    }

    const int P = static_cast<int>(part.groups.size());
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

    // PER-SAMPLE PLOIDY (AT2 adjust_pseudohaploid) is detected INSIDE the library entry
    // (steppe::run_extract_f2) now — the per-sample n_ph/n_dip observability counts come
    // back on the F2ExtractResult for the post-run echo. The CLI no longer re-detects here
    // (the up-front read is sizing/validation only, deriving P/M from the partition +
    // header — no tile is read here, so a GENO prefix does not throw before the real run).
    std::size_t n_ph = 0, n_dip = 0;

    // The engaged precision (the requested DeviceConfig.precision; the resident path
    // honors it or downgrades to native — recorded as the tag in meta.json).
    const Precision precision = config.device().precision;
    const FilterConfig& filter = config.filter();

    // ---- DRY RUN: report sizes / tier / precision, NO GPU compute -----------------
    // The tier estimate routes through the SAME resolve_output_tier the real run uses
    // (f2_blocks_multigpu_tiered), with the SAME precedence (config.force_tier from
    // --tier > STEPPE_FORCE_TIER env > auto select_output_tier), so a --tier X verdict
    // here is EXACT and the auto verdict uses the identical policy. The block count
    // n_block is computed from assign_blocks over the FULL .snp axis (CUDA-free, cheap),
    // and the SNP count uses the FULL M — both UPPER BOUNDS (filtering only drops SNPs,
    // so M_kept <= M and the real-run n_block <= this), which keeps the auto verdict
    // CONSERVATIVE toward Resident: if it says streaming here it will stream for real,
    // and a Resident verdict at full M still fits at the (smaller) kept M. The free-VRAM
    // probe routes through build_resources' BackendCapabilities (no direct CUDA call);
    // on a no-GPU box the tier estimate is skipped (the §4.5 dry-run is a planning aid).
    if (config.dry_run()) {
        // n_block over the full .snp axis (the same assign_blocks rule the real run uses).
        const double bs_morgans_dry =
            steppe::core::block_size_cm_to_morgans(config.blgsize_cm());
        const steppe::core::BlockPartition dry_partition = steppe::core::assign_blocks(
            std::span<const int>(snptab.chrom.data(), static_cast<std::size_t>(M)),
            std::span<const double>(snptab.genpos_morgans.data(), static_cast<std::size_t>(M)),
            bs_morgans_dry);
        const int n_block_dry = dry_partition.n_block;

        std::printf("steppe extract-f2 --dry-run:\n");
        std::printf("  geno:       %s\n", config.geno().c_str());
        std::printf("  snp:        %s (%ld SNPs read)\n", config.snp().c_str(), M);
        std::printf("  ind:        %s (%zu records present)\n", config.ind().c_str(), n_present);
        std::printf("  selection:  %s -> P = %d populations\n", pop_selection_str(sel).c_str(), P);
        std::printf("  precision:  %s (mantissa_bits=%d)\n", precision_label(precision), precision.mantissa_bits);
        std::printf("  blgsize:    %.4g Morgans (%.4g cM) -> %d blocks (over the full .snp axis)\n",
                    config.blgsize_cm() / kCentimorgansPerMorgan, config.blgsize_cm(), n_block_dry);
        std::printf("  filters:    maf>=%.4g maxmiss<=%.4g autosomes_only=%d drop_mono=%d transversions_only=%d\n",
                    filter.maf_min, filter.geno_max_missing,
                    filter.autosomes_only ? 1 : 0, filter.drop_monomorphic ? 1 : 0,
                    filter.transversions_only ? 1 : 0);
        try {
            device::Resources resources = device::build_resources(config.device());
            if (!resources.gpus.empty()) {
                const auto& caps = resources.gpus.front().caps;
                const std::size_t free_host_bytes = device::free_host_ram_bytes();
                // The SAME resolver the tiered entry calls — same precedence, same shape
                // inputs (full-M upper bound). --tier X is exact here; auto matches the run.
                const device::OutputTier dry_tier = device::resolve_output_tier(
                    config.device().force_tier, std::getenv("STEPPE_FORCE_TIER"),
                    P, M, n_block_dry, caps.free_vram_bytes, free_host_bytes);
                const std::size_t slab_bytes =
                    static_cast<std::size_t>(P) * static_cast<std::size_t>(P) * 2u * sizeof(double);
                std::printf("  device:     GPU %d, free VRAM = %zu MiB, total = %zu MiB; free host RAM = %zu MiB\n",
                            resources.gpus.front().device_id,
                            caps.free_vram_bytes >> 20, caps.total_vram_bytes >> 20,
                            free_host_bytes >> 20);
                std::printf("  f2 slab:    %zu bytes / block (f2 + vpair, FP64); result = %zu MiB at %d blocks\n",
                            slab_bytes, (slab_bytes * static_cast<std::size_t>(n_block_dry < 0 ? 0 : n_block_dry)) >> 20,
                            n_block_dry);
                std::printf("  tier:       %s (the result tier the run will use; auto verdict at the "
                            "full-M upper bound is conservative toward resident)\n", tier_label(dry_tier));
            } else {
                std::printf("  device:     (no CUDA device visible — tier estimate skipped)\n");
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

    // ---- 5-8b. THE GPU EXTRACT CHAIN -> host tensor (the LIBRARY entry; DRY) -------
    // The decode -> filter (pop-axis maxmiss) -> assign_blocks ->
    // compute_f2_blocks_multigpu_tiered -> to_host chain now lives ONCE in
    // steppe::run_extract_f2 (include/steppe/extract.hpp / extract_f2_core.cpp); the CLI
    // calls it so there is no duplicated math (the goldens are byte-identical — the lib
    // body was lifted VERBATIM from this command). The CLI keeps stdout/stderr ownership:
    // it MAPS the library's exceptions back to the CLI exit codes here (architecture.md
    // §10 — the library throws, main() prints + returns a code). The decode/compute reads
    // the SAME .geno/.snp/.ind triple again inside the lib (the io reads are tiny vs. the
    // GPU work); the dry-run above used its own up-front read.
    const ExtractPloidy lib_ploidy =
        (config.ploidy() == cfg::PloidyMode::PseudoHaploid) ? ExtractPloidy::PseudoHaploid
      : (config.ploidy() == cfg::PloidyMode::Diploid)       ? ExtractPloidy::Diploid
                                                            : ExtractPloidy::Auto;
    F2ExtractResult extracted;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe extract-f2: no CUDA device available (steppe is a GPU "
                         "product; a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        extracted = steppe::run_extract_f2(
            config.geno(), config.snp(), config.ind(), sel, filter, precision,
            steppe::core::block_size_cm_to_morgans(config.blgsize_cm()), lib_ploidy,
            resources);
    } catch (const std::invalid_argument& e) {
        // A config-level fault (unknown pop, empty selection, every SNP filtered).
        std::fprintf(stderr, "steppe extract-f2: %s\n", e.what());
        return cfg::kExitInvalidConfig;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe extract-f2: device error: %s\n", e.what());
        return cfg::kExitRuntimeError;
    }

    const F2BlockTensor& host_f2 = extracted.f2;
    const long M_kept = extracted.n_snp_kept;
    pop_labels = extracted.pop_labels;       // the lib's P-axis labels (= the up-front read's).
    n_ph = extracted.n_pseudo_haploid;       // observability echo (the lib detected these).
    n_dip = extracted.n_diploid;
    const Precision engaged = precision;     // recorded tag (the lib honors/downgrades internally).

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
    std::printf("  tier = %s (the f2_blocks result tier; resident = device-resident, "
                "host/disk = SNP-tile input-streaming)\n", tier_label(extracted.tier));
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
