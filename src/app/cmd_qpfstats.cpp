// src/app/cmd_qpfstats.cpp — the `steppe qpfstats` command (genotype-path joint f2 smoother).
//
// Validates inputs, runs the CUDA-free build_resources -> run_qpfstats seam, and writes the
// smoothed f2 as a read_f2-able AT2-shaped f2 dir (f2.bin + pops.txt + meta.json).
#include "app/cmd_qpfstats.hpp"

#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <vector>

#include "app/exit_code_for_caught.hpp"
#include "app/f2_dir_writer.hpp"
#include "app/precision_label.hpp"
#include "core/config/exit_code.hpp"
#include "steppe/config.hpp"
#include "device/resources.hpp"
#include "io/genotype_source.hpp"
#include "steppe/qpfstats.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

}  // namespace

int run_qpfstats_command(const cfg::RunConfig& config) {
    const std::string& prefix = config.qpdstat_prefix();
    if (prefix.empty()) {
        std::fprintf(stderr, "steppe qpfstats: --prefix (the genotype triple prefix) is required\n");
        return cfg::kExitInvalidConfig;
    }
    const std::vector<std::string>& pops = config.pops();
    if (pops.size() < 4) {
        std::fprintf(stderr,
                     "steppe qpfstats: --pops needs at least 4 populations (the f4 basis); got %zu\n",
                     pops.size());
        return cfg::kExitInvalidConfig;
    }
    if (config.out_dir().empty()) {
        std::fprintf(stderr, "steppe qpfstats: --out-dir (the smoothed f2 dir destination) is required\n");
        return cfg::kExitInvalidConfig;
    }

    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);
    const std::string& geno = triple.geno;
    const std::string& snp = triple.snp;
    const std::string& ind = triple.ind;

    const double blgsize_morgans = config.blgsize_cm() / kCentimorgansPerMorgan;
    const Precision precision = config.device().precision;
    QpfstatsResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (resources.gpus.empty()) {
            std::fprintf(stderr,
                         "steppe qpfstats: no CUDA device available (steppe is a GPU product; "
                         "a CUDA-capable GPU is required)\n");
            return cfg::kExitRuntimeError;
        }
        result = run_qpfstats(geno, snp, ind, std::span<const std::string>(pops),
                              blgsize_morgans, precision, resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe qpfstats: device/input error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (result.status != Status::Ok) {
        std::fprintf(stderr,
                     "steppe qpfstats: could not build the smoothed f2 (status=%d; check "
                     "--pops are all present in the prefix)\n", static_cast<int>(result.status));
        return cfg::kExitInvalidConfig;
    }

    F2DirMeta meta;
    meta.precision_tag = precision_label(precision);
    meta.precision_mantissa_bits = precision.mantissa_bits;
    meta.blgsize_cm = config.blgsize_cm();
    meta.n_block = result.f2.n_block;
    meta.P = result.f2.P;
    meta.autosomes_only = true;
    meta.geno_path = geno;
    meta.snp_path = snp;
    meta.ind_path = ind;
    meta.pop_selection = "qpfstats-smoothed";

    const F2DirWriteResult wr = write_f2_dir(config.out_dir(), result.f2, result.pop_labels, meta);
    if (!wr.ok) {
        std::fprintf(stderr, "steppe qpfstats: %s\n", wr.error.c_str());
        return cfg::kExitInvalidConfig;
    }

    std::printf("steppe qpfstats: wrote smoothed f2 dir %s\n", config.out_dir().c_str());
    std::printf("  P = %d, n_block = %d, precision = %s, blgsize = %.4g cM\n",
                result.f2.P, result.f2.n_block, precision_label(precision), config.blgsize_cm());
    std::printf("  f2_cache_id = %s\n", wr.f2_cache_id.c_str());
    return cfg::kExitOk;
}

}  // namespace steppe::app
