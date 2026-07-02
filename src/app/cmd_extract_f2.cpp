// src/app/cmd_extract_f2.cpp
//
// The `steppe extract-f2` command. Plain C++20, app-only, no CUDA header: the GPU is
// reached only through CUDA-free seams, and main() owns stdout/stderr — the library
// never prints.
//
// This command does the up-front sizing/validation reads and the final directory write.
// The GPU extract chain — decode -> filter -> assign_blocks -> compute f2 blocks ->
// to_host -> host tensor — lives in the library entry steppe::run_extract_f2
// (extract_f2_core.cpp), which this command delegates to, so the numerics live in one
// place and are not duplicated here. The tiered f2-blocks compute keeps the
// device-resident path byte-identical while the HostRam/Disk tiers stream the SNP-tile
// input, so high-P full-autosome runs that would OOM the resident feeder still complete.
#include "app/cmd_extract_f2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>   // std::getenv for the STEPPE_FORCE_TIER override
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "app/exit_code_for_caught.hpp"           // exit_code_for_caught (maps a device OOM to exit 3)
#include "app/f2_dir_writer.hpp"
#include "app/precision_label.hpp"                // precision_label (shared host-app helper)
#include "core/config/exit_code.hpp"
#include "core/domain/block_partition_rule.hpp"  // assign_blocks, block_size_cm_to_morgans (--dry-run sizing)
#include "device/resources.hpp"                   // Resources, build_resources (CUDA-FREE)
#include "device/tier_select.hpp"                 // resolve_output_tier, OutputTier, free_host_ram_bytes (CUDA-FREE, --dry-run)
#include "steppe/config.hpp"                      // Precision, FilterConfig, kCentimorgansPerMorgan
#include "steppe/error.hpp"                       // steppe::Status
#include "steppe/extract.hpp"                     // run_extract_f2 + F2ExtractResult (the library entry; CUDA-FREE)

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"  // read_snp_table / read_ind_partition (format-aware .snp|.bim, .ind|.fam)
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Human label for an OutputTier, echoed in the --dry-run plan and the post-run summary.
// Resident is the device-resident path; HostRam/Disk stream the SNP-tile input so the
// GPU peak stays independent of M.
[[nodiscard]] const char* tier_label(device::OutputTier t) {
    switch (t) {
        case device::OutputTier::Resident: return "resident";
        case device::OutputTier::HostRam:  return "host";
        case device::OutputTier::Disk:     return "disk";
    }
    return "resident";
}

// Same label over the CUDA-free ExtractTier the library entry returns — the post-run
// summary echoes the tier the compute actually used.
[[nodiscard]] const char* tier_label(steppe::ExtractTier t) {
    switch (t) {
        case steppe::ExtractTier::Resident: return "resident";
        case steppe::ExtractTier::HostRam:  return "host";
        case steppe::ExtractTier::Disk:     return "disk";
    }
    return "resident";
}

// Human echo of the pop-selection request: which mode/threshold was asked for. The
// resolved labels themselves land in pops.txt.
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
// partition: an unknown pop name must fail fast, naming the offending label. read_ind
// silently drops unknown labels (and only throws on a fully empty selection), so the
// check has to happen here against the resolved groups. Sets `offending` and returns
// false on the first missing label.
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
    // ---- Required inputs ----------------------------------------------------------
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

    // ---- Open the .geno + read .ind (selection) + .snp ----------------------------
    io::IndPartition part;
    io::SnpTable snptab;
    std::size_t n_present = 0;
    long M = 0;
    try {
        io::GenoReader reader(config.geno());
        const io::GenoFormat fmt = reader.header().format;  // picks the .snp|.bim, .ind|.fam parser
        n_present = reader.records_present();
        part = io::read_ind_partition(fmt, config.ind(), sel, n_present);

        // Explicit-name validation: read_ind silently drops an unknown label, so verify
        // the request against the resolved groups here.
        std::string offending;
        if (!validate_explicit_pops(sel, part, offending)) {
            std::fprintf(stderr,
                         "steppe extract-f2: --pops contains an unknown population "
                         "'%s' (not present in %s)\n",
                         offending.c_str(), config.ind().c_str());
            return cfg::kExitInvalidConfig;
        }

        snptab = io::read_snp_table(fmt, config.snp(), SIZE_MAX);
        // Sizing + validation only (dry-run sizes and the .snp/.geno axis check). P and M
        // are derivable from the partition + header without reading — and, for GENO,
        // without transposing — a tile: P == #selected pops, M == the common SNP prefix.
        // The real canonical-tile read (TGENO direct, or GENO gather + on-device
        // transpose) happens inside run_extract_f2 below. Deriving P/M here keeps this
        // format-agnostic: a GENO prefix must not throw at an up-front individual-major
        // read that only TGENO supports.
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

    // The P labels in P-axis index order (= pops.txt) — the name<->index map the engine
    // itself does not carry.
    std::vector<std::string> pop_labels;
    pop_labels.reserve(static_cast<std::size_t>(P));
    for (const io::PopGroup& g : part.groups) pop_labels.push_back(g.label);

    // Per-sample ploidy (AT2's adjust_pseudohaploid) is detected inside run_extract_f2;
    // the n_ph/n_dip counts come back on F2ExtractResult for the post-run echo. The CLI
    // does not re-detect here — the up-front read is sizing/validation only and reads no
    // tile.
    std::size_t n_ph = 0, n_dip = 0;

    // The requested precision, passed to the library entry. The tag actually recorded in
    // meta.json is read back from the result (extracted.precision_tag — the kind the run
    // honored or downgraded to); see the `engaged` construction below.
    const Precision precision = config.device().precision;
    const FilterConfig& filter = config.filter();

    // ---- DRY RUN: report sizes / tier / precision, NO GPU compute -----------------
    // The tier estimate routes through the same resolve_output_tier the real run uses,
    // with the same precedence (--tier > STEPPE_FORCE_TIER env > auto), so a --tier X
    // verdict here is exact and the auto verdict uses the identical policy. n_block and
    // the SNP count are taken over the FULL .snp axis — both upper bounds, since filtering
    // only drops SNPs (M_kept <= M, real n_block <= this). That keeps the auto verdict
    // conservative toward Resident: if it says stream here it will stream for real, and a
    // Resident verdict at full M still fits at the smaller kept M. The free-VRAM probe
    // goes through build_resources' capabilities, not a direct CUDA call; with no GPU
    // visible the estimate is skipped.
    if (config.dry_run()) {
        // n_block over the full .snp axis (the same assign_blocks rule the real run uses).
        const double bs_morgans_dry =
            steppe::core::block_size_cm_to_morgans(config.blgsize_cm());
        const steppe::core::BlockPartition dry_partition = steppe::core::assign_blocks(
            std::span<const int>(snptab.chrom.data(), static_cast<std::size_t>(M)),
            std::span<const double>(snptab.genpos_morgans.data(), static_cast<std::size_t>(M)),
            bs_morgans_dry,
            std::span<const double>(snptab.physpos.data(), static_cast<std::size_t>(M)));
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
                // The same resolver the tiered entry calls — same precedence, same full-M
                // upper-bound inputs. --tier X is exact here; auto matches the run.
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

    // ---- SOURCE-PROVENANCE HASH (--hash opt-in; default OFF) ----------------------
    // The whole-.geno SHA-256 is a tens-of-seconds whole-file read that once dominated
    // extract-f2 wall time (~37s of ~41s on a 6.7 GB .geno) yet only yields a provenance
    // value, so it is skipped by default. Under --hash it depends only on the .geno path
    // (independent of the GPU pipeline), so we run it on a background thread started here,
    // before the GPU work, and join it just before meta.json is written — overlapping the
    // hash with the GPU work instead of adding to it. The small .snp/.ind hashes are left
    // to the writer.
    const bool hash_source = config.hash_source();
    std::string geno_sha;                 // filled by the worker iff hash_source
    std::exception_ptr geno_hash_err;     // captured iff the worker throws; degraded at the join
    std::thread geno_hash_thread;         // joined before write_f2_dir (never detached)
    if (hash_source) {
        const std::string geno_path = config.geno();
        geno_hash_thread = std::thread([geno_path, &geno_sha, &geno_hash_err]() {
            // Catch everything: an exception escaping a std::thread's top-level callable
            // calls std::terminate. sha256_file uses an exception-disabled ifstream and
            // returns "" on open failure, so the only realistic throw is bad_alloc on its
            // read chunk — capture it and degrade at the join site (the provenance hash is
            // non-essential) instead.
            try {
                geno_sha = sha256_file(std::filesystem::path(geno_path));
            } catch (...) {
                geno_hash_err = std::current_exception();
            }
        });
    }
    // RAII safety: if any early return / exception fires before the explicit join below,
    // this guarantees the worker is joined so the thread is never destroyed un-joined. The
    // worker lambda catches everything, so neither this join nor the explicit one can
    // observe an escaping exception — std::terminate is impossible on either path.
    struct ThreadJoiner {
        std::thread& t;
        ~ThreadJoiner() { if (t.joinable()) t.join(); }
    } geno_hash_joiner{geno_hash_thread};

    // ---- THE GPU EXTRACT CHAIN -> host tensor (via the library entry) --------------
    // The decode -> filter (pop-axis maxmiss) -> assign_blocks -> compute f2 blocks ->
    // to_host chain lives in steppe::run_extract_f2 (extract_f2_core.cpp); the CLI calls
    // it so there is no duplicated math. The CLI keeps stdout/stderr ownership and maps the
    // library's exceptions back to exit codes here — the library throws, main() prints and
    // returns a code. The library re-reads the same .geno/.snp/.ind triple internally
    // (those io reads are tiny next to the GPU work); the dry-run above used its own
    // up-front read.
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
        return exit_code_for_caught(e);
    }

    const F2BlockTensor& host_f2 = extracted.f2;
    const long M_kept = extracted.n_snp_kept;
    pop_labels = extracted.pop_labels;       // the lib's P-axis labels (= the up-front read's).
    n_ph = extracted.n_pseudo_haploid;       // observability echo (the lib detected these).
    n_dip = extracted.n_diploid;
    // The engaged precision: take the KIND from the library result (precision_tag is the
    // single source of truth for the precision the library honored or downgraded to), but
    // keep mantissa_bits from the requested config — the result carries only the kind, not
    // the mantissa cap, which is meaningful only for EmulatedFp64 and unchanged by a
    // downgrade.
    Precision engaged = precision;
    engaged.kind = extracted.precision_tag;

    // ---- Write the f2_blocks dir (f2.bin vpair + pops.txt + meta.json) -------------
    F2DirMeta meta;
#ifdef STEPPE_VERSION
    meta.steppe_version = STEPPE_VERSION;  // = ${PROJECT_VERSION}, injected by CMake
#else
    meta.steppe_version = "0.0.0+unknown";  // non-release sentinel (standalone compile only)
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
    // Source-provenance hashes (--hash opt-in; default OFF). When requested, join the
    // background geno-hash worker here (its wall time overlapped the GPU pipeline above)
    // and pre-fill meta.geno_sha256 so the writer skips re-hashing the big .geno; the
    // writer fills the small .snp/.ind hashes. When OFF, every *_sha256 stays empty and
    // meta.json records source_hash_computed:false, so the absence reads as deliberate
    // (see f2_dir_writer.cpp).
    meta.hash_source_files = hash_source;
    if (hash_source) {
        if (geno_hash_thread.joinable()) geno_hash_thread.join();
        if (geno_hash_err) {
            // Degrade: a throw escaped the background hash worker (realistically a bad_alloc
            // on sha256_file's read chunk). The provenance hash is non-essential, so fall
            // back to the empty-sha + source_hash_computed:false state rather than aborting
            // the whole extract. Clearing hash_source_files also stops the writer re-hashing
            // on the main thread (geno_sha256 is empty here) — which would redo the
            // backgrounded work and could re-throw the same bad_alloc inside write_f2_dir.
            std::fprintf(stderr,
                         "steppe extract-f2: warning: source .geno hash failed; writing "
                         "meta.json with source_hash_computed:false\n");
            meta.hash_source_files = false;
            meta.geno_sha256.clear();
        } else {
            meta.geno_sha256 = geno_sha;  // "" if the .geno could not be opened
        }
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
