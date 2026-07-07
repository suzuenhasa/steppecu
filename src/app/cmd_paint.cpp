// src/app/cmd_paint.cpp
//
// The `steppe paint` command (Li-Stephens haplotype copying). Phase 0: resolve +
// validate the request host-only and report the run plan; no GPU kernel launch.
//
// The genotype triples are read through a host CpuBackend used ONLY as the io /
// transpose / ploidy-detect oracle (the front-end needs a backend to canonicalize a
// tile) — the paint statistic itself is NOT computed here (that is the Phase-1 GPU
// forward-backward). Reference: docs/planning/li-stephens-engine-scope.md §1, §3.
#include "app/cmd_paint.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "app/cmd_emit.hpp"
#include "core/config/exit_code.hpp"
#include "core/internal/decode_af.hpp"
#include "core/stats/decode_keep_autosomes.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "core/stats/li_stephens.hpp"
#include "core/stats/li_stephens_validate.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "io/genotype_source.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"

namespace steppe::app {

namespace {

namespace cfg = steppe::config;

// A PopSelection that keeps every individual (each haploid column is a haplotype):
// MinN with a floor of 1 retains all populations with at least one member.
[[nodiscard]] io::PopSelection all_individuals() {
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::MinN;
    sel.min_n = 1;
    return sel;
}

// Count samples in a canonical tile whose auto-detected ploidy is diploid (a
// heterozygous code was seen in the detection window) — the phased/haploid gate.
[[nodiscard]] long count_diploid(ComputeBackend& be, const io::GenotypeTile& tile) {
    // detect_sample_ploidy_device recomputes ploidy from the packed codes; the input
    // ploidy vector only sizes the view, so a pseudo-haploid placeholder is fine.
    const std::vector<int> placeholder(tile.n_individuals, core::kPloidyPseudoHaploid);
    const DecodeTileView view =
        core::make_decode_tile_view(tile, placeholder, static_cast<int>(tile.n_pop()));
    const std::vector<int> ploidy = be.detect_sample_ploidy_device(view);
    long n = 0;
    for (int p : ploidy)
        if (p == core::kPloidyDiploid) ++n;
    return n;
}

}  // namespace

int run_paint_command(const cfg::RunConfig& config) {
    // --- Required inputs -----------------------------------------------------
    if (config.qpdstat_prefix().empty()) {
        std::fprintf(stderr,
                     "steppe paint: --prefix PREFIX.{geno,snp,ind} (the phased RECIPIENT "
                     "haplotypes) is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (config.donors_prefix().empty()) {
        std::fprintf(stderr,
                     "steppe paint: --donors PREFIX.{geno,snp,ind} (the phased DONOR panel) "
                     "is required\n");
        return cfg::kExitInvalidConfig;
    }

    const std::string& recip_prefix = config.qpdstat_prefix();
    const std::string& donor_prefix = config.donors_prefix();
    const io::GenotypeTriple recip = io::resolve_genotype_triple(recip_prefix);
    const io::GenotypeTriple donor = io::resolve_genotype_triple(donor_prefix);

    io::SnpTable snptab;
    core::PaintRequest req;
    try {
        // The io / transpose / ploidy-detect oracle — host-only, no device. NOT a
        // compute path: the paint forward-backward is the Phase-1 GPU engine.
        std::unique_ptr<ComputeBackend> be = device::make_cpu_backend();

        const core::GenotypeFrontEnd fe_r = core::read_genotype_front_end(
            recip.geno, recip.snp, recip.ind, all_individuals(), *be);
        const core::GenotypeFrontEnd fe_d = core::read_genotype_front_end(
            donor.geno, donor.snp, donor.ind, all_individuals(), *be);

        snptab = fe_r.snptab;

        req.Ne = config.ls_ne();
        req.theta_auto = std::isnan(config.ls_theta());
        req.theta = req.theta_auto ? 0.0 : config.ls_theta();
        req.self_copy = config.ls_self_copy();
        req.recip_batch = config.ls_recip_batch();
        req.allow_bp_fallback = config.ls_bp_fallback();
        req.n_recipients = static_cast<long>(fe_r.tile.n_individuals);
        req.n_donors = static_cast<long>(fe_d.tile.n_individuals);
        req.n_diploid_samples =
            count_diploid(*be, fe_r.tile) + count_diploid(*be, fe_d.tile);
        // Phase-0 heuristic: the same triple for donors and recipients IS the
        // panel-vs-self (all-vs-all) painting case; full per-haplotype ID-overlap
        // detection lands with the Phase-1/2 engine.
        req.donors_superset_recipients = (recip_prefix == donor_prefix);
        req.sure = config.sweep_sure();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe paint: input error: %s\n", e.what());
        return cfg::kExitIoError;
    }

    std::string err;
    const Status vs = core::validate_paint_request(req, snptab, err);
    if (vs != Status::Ok) {
        std::fprintf(stderr, "steppe paint: %s\n", err.c_str());
        return cfg::kExitInvalidConfig;
    }

    // Validated: report the run plan. The GPU forward-backward + coancestry face are
    // Phase 1/2; Phase 0 stops at a validated plan (no kernel launch).
    const double theta_val =
        req.theta_auto ? core::watterson_emission_rate(static_cast<int>(req.n_donors)) : req.theta;
    if (const auto rc = emit_to_destination(
            config, "paint", [&](std::ostream& os, OutputFormat fmt) {
                const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
                if (fmt == OutputFormat::Json) {
                    os << "{\n";
                    os << "  \"phase\": \"validated-plan (no compute; GPU FB is Phase 1)\",\n";
                    os << "  \"recipients\": " << req.n_recipients << ",\n";
                    os << "  \"donors\": " << req.n_donors << ",\n";
                    os << "  \"snps\": " << snptab.count << ",\n";
                    os << "  \"Ne\": " << req.Ne << ",\n";
                    os << "  \"theta\": " << theta_val
                       << (req.theta_auto ? " ,\n  \"theta_auto\": true,\n"
                                          : " ,\n  \"theta_auto\": false,\n");
                    os << "  \"self_copy\": " << (req.self_copy ? "true" : "false") << ",\n";
                    os << "  \"leave_one_out\": "
                       << (req.donors_superset_recipients && !req.self_copy ? "true" : "false")
                       << ",\n";
                    os << "  \"recip_batch\": " << req.recip_batch << "\n";
                    os << "}\n";
                    return;
                }
                os << "field" << sep << "value\n";
                os << "phase" << sep << "validated-plan\n";
                os << "recipients" << sep << req.n_recipients << "\n";
                os << "donors" << sep << req.n_donors << "\n";
                os << "snps" << sep << snptab.count << "\n";
                os << "Ne" << sep << req.Ne << "\n";
                os << "theta" << sep << theta_val << (req.theta_auto ? " (auto)" : "") << "\n";
                os << "self_copy" << sep << (req.self_copy ? "on" : "off") << "\n";
                os << "leave_one_out" << sep
                   << (req.donors_superset_recipients && !req.self_copy ? "on" : "off") << "\n";
                os << "recip_batch" << sep << req.recip_batch << "\n";
            })) {
        return *rc;
    }
    std::fprintf(stderr,
                 "steppe paint: request validated (%ld recipients x %ld donors x %zu SNPs). "
                 "The GPU forward-backward + coancestry face are Phase 1/2; no compute run.\n",
                 req.n_recipients, req.n_donors, snptab.count);
    return cfg::kExitOk;
}

}  // namespace steppe::app
