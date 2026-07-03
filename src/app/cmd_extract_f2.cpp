// src/app/cmd_extract_f2.cpp
//
// The `steppe extract-f2` command: turns a genotype triple into an on-disk
// f2_blocks directory. Plain C++20 with no CUDA header; it owns stdout/stderr and
// does only the cheap up-front sizing/validation and the final directory write,
// delegating the heavy genotype->f2 GPU chain to steppe::run_extract_f2.
//
// Reference: docs/reference/src_app_cmd_extract_f2.cpp.md
#include "app/cmd_extract_f2.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_writer.hpp"
#include "app/precision_label.hpp"
#include "core/config/exit_code.hpp"
#include "core/domain/block_partition_rule.hpp"
#include "device/resources.hpp"
#include "device/tier_select.hpp"
#include "steppe/config.hpp"
#include "steppe/error.hpp"
#include "steppe/extract.hpp"

#include "io/eigenstrat_format.hpp"
#include "io/geno_reader.hpp"
#include "io/genotype_source.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// Local helper functions — reference §3
[[nodiscard]] const char* tier_label(device::OutputTier t) {
    switch (t) {
        case device::OutputTier::Resident: return "resident";
        case device::OutputTier::HostRam:  return "host";
        case device::OutputTier::Disk:     return "disk";
    }
    return "resident";
}

[[nodiscard]] const char* tier_label(steppe::ExtractTier t) {
    switch (t) {
        case steppe::ExtractTier::Resident: return "resident";
        case steppe::ExtractTier::HostRam:  return "host";
        case steppe::ExtractTier::Disk:     return "disk";
    }
    return "resident";
}

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

// run_extract_f2_command — reference §4
int run_extract_f2_command(const cfg::RunConfig& config) {
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

    io::IndPartition part;
    io::SnpTable snptab;
    std::size_t n_present = 0;
    long M = 0;
    try {
        io::GenoReader reader(config.geno());
        const io::GenoFormat fmt = reader.header().format;
        n_present = reader.records_present();
        part = io::read_ind_partition(fmt, config.ind(), sel, n_present);

        std::string offending;
        if (!validate_explicit_pops(sel, part, offending)) {
            std::fprintf(stderr,
                         "steppe extract-f2: --pops contains an unknown population "
                         "'%s' (not present in %s)\n",
                         offending.c_str(), config.ind().c_str());
            return cfg::kExitInvalidConfig;
        }

        snptab = io::read_snp_table(fmt, config.snp(), SIZE_MAX);
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

    std::vector<std::string> pop_labels;
    pop_labels.reserve(static_cast<std::size_t>(P));
    for (const io::PopGroup& g : part.groups) pop_labels.push_back(g.label);

    std::size_t n_ph = 0, n_dip = 0;

    const Precision precision = config.device().precision;
    const FilterConfig& filter = config.filter();

    if (config.dry_run()) {
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

    const bool hash_source = config.hash_source();
    std::string geno_sha;
    std::exception_ptr geno_hash_err;
    std::thread geno_hash_thread;
    if (hash_source) {
        const std::string geno_path = config.geno();
        geno_hash_thread = std::thread([geno_path, &geno_sha, &geno_hash_err]() {
            try {
                geno_sha = sha256_file(std::filesystem::path(geno_path));
            } catch (...) {
                geno_hash_err = std::current_exception();
            }
        });
    }
    struct ThreadJoiner {
        std::thread& t;
        ~ThreadJoiner() { if (t.joinable()) t.join(); }
    } geno_hash_joiner{geno_hash_thread};

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
        std::fprintf(stderr, "steppe extract-f2: %s\n", e.what());
        return cfg::kExitInvalidConfig;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe extract-f2: device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    const F2BlockTensor& host_f2 = extracted.f2;
    const long M_kept = extracted.n_snp_kept;
    pop_labels = extracted.pop_labels;
    n_ph = extracted.n_pseudo_haploid;
    n_dip = extracted.n_diploid;
    Precision engaged = precision;
    engaged.kind = extracted.precision_tag;

    F2DirMeta meta;
#ifdef STEPPE_VERSION
    meta.steppe_version = STEPPE_VERSION;
#else
    meta.steppe_version = "0.0.0+unknown";
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
    meta.hash_source_files = hash_source;
    if (hash_source) {
        if (geno_hash_thread.joinable()) geno_hash_thread.join();
        if (geno_hash_err) {
            std::fprintf(stderr,
                         "steppe extract-f2: warning: source .geno hash failed; writing "
                         "meta.json with source_hash_computed:false\n");
            meta.hash_source_files = false;
            meta.geno_sha256.clear();
        } else {
            meta.geno_sha256 = geno_sha;
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
