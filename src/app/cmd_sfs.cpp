// src/app/cmd_sfs.cpp
//
// The `steppe sfs` command — the standalone 2D joint site-frequency spectrum over a
// population pair, accumulated on the GPU straight from a genotype triple. App-only and
// CUDA-free: the GPU is reached only through the run_sfs seam. Emits the (extA x extB)
// integer joint matrix (TSV/CSV grid + a dims/provenance header, or a nested JSON array).
//
// v1 SCOPE: 2D (two-pop), folded (per-pop minor) or unfolded (A1-copy count), over sites
// with COMPLETE data in both pops. Gated BIT-EXACT vs scikit-allel joint_sfs /
// joint_sfs_folded, NOT ADMIXTOOLS2. A 3D extension + hypergeometric projection are
// documented follow-ups.
#include "app/cmd_sfs.hpp"

#include <cstdio>
#include <exception>
#include <ostream>
#include <string>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/cmd_emit.hpp"
#include "app/exit_code_for_caught.hpp"
#include "app/result_emit.hpp"
#include "core/config/exit_code.hpp"
#include "device/resources.hpp"
#include "io/genotype_source.hpp"
#include "steppe/error.hpp"
#include "steppe/sfs.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

const char* status_text(steppe::Status s) {
    switch (s) {
        case steppe::Status::Ok: return "ok";
        case steppe::Status::RankDeficient: return "rank_deficient";
        case steppe::Status::NonSpdCovariance: return "non_spd";
        case steppe::Status::ChisqUndefined: return "chisq_undefined";
        case steppe::Status::DeviceOom: return "device_oom";
        case steppe::Status::InvalidConfig: return "invalid_config";
    }
    return "unknown";
}

// The joint SFS matrix: a header comment block (TSV/CSV) or a nested row-major array
// (JSON). Cells are exact std::int64_t integers (no fmt_double).
void emit_sfs_matrix(std::ostream& os, OutputFormat fmt, const steppe::SfsResult& r) {
    const long extA = r.extA;
    const long extB = r.extB;
    if (fmt == OutputFormat::Json) {
        os << "{\n";
        os << "  \"popA\": " << json_quote(r.popA) << ",\n";
        os << "  \"popB\": " << json_quote(r.popB) << ",\n";
        os << "  \"folded\": " << (r.folded ? "true" : "false") << ",\n";
        os << "  \"nIndA\": " << r.NA << ",\n";
        os << "  \"nIndB\": " << r.NB << ",\n";
        os << "  \"extA\": " << extA << ",\n";
        os << "  \"extB\": " << extB << ",\n";
        os << "  \"sites_total\": " << r.n_total << ",\n";
        os << "  \"complete\": " << r.n_complete << ",\n";
        os << "  \"dropped_incomplete\": " << r.n_dropped_incomplete << ",\n";
        os << "  \"sfs\": [";
        for (long i = 0; i < extA; ++i) {
            os << (i == 0 ? "\n    [" : ",\n    [");
            for (long j = 0; j < extB; ++j) {
                const std::size_t k = static_cast<std::size_t>(i) * static_cast<std::size_t>(extB) +
                                      static_cast<std::size_t>(j);
                os << (j == 0 ? "" : ", ") << (k < r.grid.size() ? r.grid[k] : 0);
            }
            os << "]";
        }
        os << "\n  ]\n";
        os << "}\n";
        return;
    }
    const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
    os << "# steppe sfs\n";
    os << "# popA=" << r.popA << " popB=" << r.popB << " folded="
       << (r.folded ? "true" : "false") << "\n";
    os << "# nIndA=" << r.NA << " nIndB=" << r.NB << " dims=" << extA << "x" << extB << "\n";
    os << "# sites_total=" << r.n_total << " complete=" << r.n_complete
       << " dropped_incomplete=" << r.n_dropped_incomplete << "\n";
    for (long i = 0; i < extA; ++i) {
        for (long j = 0; j < extB; ++j) {
            const std::size_t k = static_cast<std::size_t>(i) * static_cast<std::size_t>(extB) +
                                  static_cast<std::size_t>(j);
            os << (j == 0 ? "" : std::string(1, sep)) << (k < r.grid.size() ? r.grid[k] : 0);
        }
        os << "\n";
    }
}

}  // namespace

int run_sfs_command(const cfg::RunConfig& config) {
    if (config.qpdstat_prefix().empty()) {
        std::fprintf(stderr, "steppe sfs: --prefix PREFIX.{geno,snp,ind} is required\n");
        return cfg::kExitInvalidConfig;
    }
    const std::vector<std::string>& pops = config.pops();
    if (pops.size() != 2) {
        std::fprintf(stderr,
                     "steppe sfs: --pops must name EXACTLY two populations A,B (v1 is the 2D "
                     "joint SFS; a 3D extension is a follow-up); got %zu\n",
                     pops.size());
        return cfg::kExitInvalidConfig;
    }
    if (pops[0] == pops[1]) {
        std::fprintf(stderr, "steppe sfs: --pops A,B must name two DIFFERENT populations\n");
        return cfg::kExitInvalidConfig;
    }

    const std::string& prefix = config.qpdstat_prefix();
    const io::GenotypeTriple triple = io::resolve_genotype_triple(prefix);

    steppe::SfsResult result;
    try {
        device::Resources resources = device::build_resources(config.device());
        if (!require_first_gpu(resources, "sfs")) return cfg::kExitRuntimeError;
        result = run_sfs(triple.geno, triple.snp, triple.ind, pops[0], pops[1], config.sfs_fold(),
                         resources);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe sfs: input/device error: %s\n", e.what());
        return exit_code_for_caught(e);
    }

    if (result.status != Status::Ok) {
        std::fprintf(stderr, "steppe sfs: %s (check the two --pops labels exist in the .ind)\n",
                     status_text(result.status));
        return cfg::exit_code_for(result.status);
    }

    // Human diagnostic to stderr (out of the parsed matrix stream).
    std::fprintf(stderr,
                 "steppe sfs: %s vs %s — %ldx%ld joint SFS (%s) over %ld complete sites "
                 "(%ld dropped incomplete)\n",
                 result.popA.c_str(), result.popB.c_str(), result.extA, result.extB,
                 result.folded ? "folded" : "unfolded", result.n_complete,
                 result.n_dropped_incomplete);

    if (const auto rc = emit_to_destination(
            config, "sfs", [&](std::ostream& os, OutputFormat fmt) {
                emit_sfs_matrix(os, fmt, result);
            })) {
        return *rc;
    }
    return cfg::exit_code_for(result.status);
}

}  // namespace steppe::app
