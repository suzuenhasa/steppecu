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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
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
#include "steppe/config.hpp"

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

// Decode a canonical individual-major tile to a flat haplotype-major allele-byte
// buffer allele[g*M + l] in {0,1, kLsMissingAllele}. Each individual (haploid column)
// is one donor/recipient haplotype; the layout is exactly what the FB expects (donor-
// major for donors, recipient-major for recipients).
[[nodiscard]] std::vector<std::uint8_t> decode_haplotypes(const io::GenotypeTile& tile, long M) {
    const std::size_t Ms = static_cast<std::size_t>(M);
    std::vector<std::uint8_t> out(tile.n_individuals * Ms, core::kLsMissingAllele);
    for (std::size_t g = 0; g < tile.n_individuals; ++g) {
        const std::uint8_t* rec = tile.packed.data() + g * tile.bytes_per_record;
        for (long l = 0; l < M; ++l) {
            const std::size_t byte = static_cast<std::size_t>(l) / core::kCodesPerByte;
            const int pos = static_cast<int>(static_cast<std::size_t>(l) % core::kCodesPerByte);
            out[g * Ms + static_cast<std::size_t>(l)] =
                core::haploid_allele_from_code(core::genotype_code(rec[byte], pos));
        }
    }
    return out;
}

// Per-individual ancestry/pop label, in tile (population-contiguous) order: individual
// k belongs to pop p where pop_offsets[p] <= k < pop_offsets[p+1], labelled pop_labels[p].
[[nodiscard]] std::vector<std::string> per_individual_labels(const io::GenotypeTile& tile) {
    std::vector<std::string> lab(tile.n_individuals, std::string("PANEL"));
    const std::size_t P = tile.n_pop();
    if (P > 0 && tile.pop_offsets.size() == P + 1) {
        for (std::size_t p = 0; p < P; ++p)
            for (std::size_t k = tile.pop_offsets[p]; k < tile.pop_offsets[p + 1]; ++k)
                if (k < lab.size()) lab[k] = tile.pop_labels[p];
    }
    return lab;
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
    core::GenotypeFrontEnd fe_r;
    core::GenotypeFrontEnd fe_d;
    try {
        // The io / transpose / ploidy-detect oracle — host-only, no device (the compute
        // path below drives the GPU/CPU forward-backward through its own backend).
        std::unique_ptr<ComputeBackend> be = device::make_cpu_backend();

        fe_r = core::read_genotype_front_end(recip.geno, recip.snp, recip.ind,
                                             all_individuals(), *be);
        fe_d = core::read_genotype_front_end(donor.geno, donor.snp, donor.ind,
                                             all_individuals(), *be);

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
        // The same triple for donors and recipients IS the panel-vs-self (all-vs-all)
        // painting case: the two reads use the identical selection so individual order
        // matches, and recipient r's self donor is donor r (leave-one-out via pi_r=0).
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

    // ---- Common marker check: recipients and donors must share the .snp set --------
    const long M = static_cast<long>(fe_r.tile.n_snp);
    if (static_cast<long>(fe_d.tile.n_snp) != M ||
        fe_r.snptab.count < static_cast<std::size_t>(M) ||
        fe_d.snptab.count < static_cast<std::size_t>(M)) {
        std::fprintf(stderr,
                     "steppe paint: recipients and donors must share the same .snp marker "
                     "set (recipient M=%ld, donor M=%zu)\n",
                     M, fe_d.tile.n_snp);
        return cfg::kExitInvalidConfig;
    }
    for (long l = 0; l < M; ++l) {
        const std::size_t ls = static_cast<std::size_t>(l);
        const bool chrom_ok =
            (ls >= fe_r.snptab.chrom.size() || ls >= fe_d.snptab.chrom.size()) ||
            (fe_r.snptab.chrom[ls] == fe_d.snptab.chrom[ls]);
        const bool pos_ok = fe_r.snptab.genpos_morgans[ls] == fe_d.snptab.genpos_morgans[ls];
        if (!chrom_ok || !pos_ok) {
            std::fprintf(stderr,
                         "steppe paint: recipients and donors disagree on the .snp marker "
                         "at index %ld (chrom/genpos mismatch)\n",
                         l);
            return cfg::kExitInvalidConfig;
        }
    }

    // ---- Decode both panels to haploid allele bytes + build the FB model inputs -----
    const int K = static_cast<int>(req.n_donors);
    const long Nrec = req.n_recipients;
    const double theta_val =
        req.theta_auto ? core::watterson_emission_rate(K) : req.theta;

    const std::vector<std::uint8_t> donors = decode_haplotypes(fe_d.tile, M);
    const std::vector<std::uint8_t> recips = decode_haplotypes(fe_r.tile, M);

    std::vector<int> chrom(fe_r.snptab.chrom.begin(),
                           fe_r.snptab.chrom.begin() + std::min<std::size_t>(
                               fe_r.snptab.chrom.size(), static_cast<std::size_t>(M)));
    std::vector<double> gpos(fe_r.snptab.genpos_morgans.begin(),
                             fe_r.snptab.genpos_morgans.begin() + static_cast<std::size_t>(M));
    const std::vector<double> rho = core::build_recomb_probs(chrom, gpos, req.Ne, K);
    const std::vector<double> w = core::build_genetic_weights(chrom, gpos);
    const std::vector<double> mu(static_cast<std::size_t>(M), theta_val);

    // Per-recipient copying prior: leave-one-out (self donor zeroed) when the donor
    // panel is the recipient panel and self-copy is off; else uniform.
    const bool loo = req.donors_superset_recipients && !req.self_copy;
    std::vector<double> pi_all(static_cast<std::size_t>(Nrec) * static_cast<std::size_t>(K), 0.0);
    for (long r = 0; r < Nrec; ++r) {
        const int self_r = loo ? static_cast<int>(r) : -1;
        const std::vector<double> pir = core::build_uniform_pi(K, self_r);
        std::copy(pir.begin(), pir.end(),
                  pi_all.begin() + static_cast<std::size_t>(r) * static_cast<std::size_t>(K));
    }

    // ---- Backend: GPU when a device is visible, else the CPU reference oracle -------
    std::unique_ptr<ComputeBackend> be;
    try {
        int dev = 0;
        if (!config.device().devices.empty()) dev = config.device().devices.front();
        be = (device::visible_device_count() > 0) ? device::make_cuda_backend(dev)
                                                  : device::make_cpu_backend();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe paint: device init failed: %s\n", e.what());
        return cfg::kExitIoError;
    }

    // ---- Batched coancestry run over recipient waves; the K*M posterior never leaves
    //      the device (only the small N*K accumulators return). ------------------------
    std::vector<double> counts(static_cast<std::size_t>(Nrec) * static_cast<std::size_t>(K), 0.0);
    std::vector<double> lengths(static_cast<std::size_t>(Nrec) * static_cast<std::size_t>(K), 0.0);
    const long wave = std::max<long>(1, req.recip_batch);
    const steppe::Precision prec = steppe::Precision::fp64();
    try {
        for (long r0 = 0; r0 < Nrec; r0 += wave) {
            const long nb = std::min<long>(wave, Nrec - r0);
            const steppe::LsCoancestry co = be->ls_paint_coancestry(
                recips.data() + static_cast<std::size_t>(r0) * static_cast<std::size_t>(M),
                donors.data(),
                pi_all.data() + static_cast<std::size_t>(r0) * static_cast<std::size_t>(K),
                rho.data(), mu.data(), w.data(), K, M, static_cast<int>(nb), prec);
            if (co.status != Status::Ok) {
                std::fprintf(stderr, "steppe paint: coancestry run returned a non-Ok status\n");
                return cfg::kExitInvalidConfig;
            }
            const std::size_t off = static_cast<std::size_t>(r0) * static_cast<std::size_t>(K);
            std::copy(co.chunkcounts.begin(), co.chunkcounts.end(), counts.begin() + off);
            std::copy(co.chunklengths.begin(), co.chunklengths.end(), lengths.begin() + off);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe paint: coancestry run failed: %s\n", e.what());
        return cfg::kExitIoError;
    }

    // ---- Resolve donor labels (the coancestry columns) ------------------------------
    std::vector<std::string> donor_label = per_individual_labels(fe_d.tile);
    if (!config.labels_file().empty()) {
        std::ifstream lf(config.labels_file());
        if (!lf) {
            std::fprintf(stderr, "steppe paint: cannot open --labels file: %s\n",
                         config.labels_file().c_str());
            return cfg::kExitIoError;
        }
        std::vector<std::string> file_labels;
        std::string line;
        while (std::getline(lf, line)) {
            std::istringstream ss(line);
            std::string first, second;
            if (!(ss >> first)) continue;  // skip blank lines
            file_labels.push_back((ss >> second) ? second : first);
        }
        if (static_cast<int>(file_labels.size()) != K) {
            std::fprintf(stderr,
                         "steppe paint: --labels file has %zu labels but the donor panel has "
                         "%d haplotypes (one label per donor, in .ind order)\n",
                         file_labels.size(), K);
            return cfg::kExitInvalidConfig;
        }
        donor_label = std::move(file_labels);
    }
    std::vector<std::string> recip_label = per_individual_labels(fe_r.tile);

    // Distinct donor labels in first-appearance order (the coancestry columns).
    std::vector<std::string> group_labels;
    std::unordered_map<std::string, int> group_index;
    for (int k = 0; k < K; ++k) {
        const std::string& g = donor_label[static_cast<std::size_t>(k)];
        if (group_index.emplace(g, static_cast<int>(group_labels.size())).second)
            group_labels.push_back(g);
    }
    const int P = static_cast<int>(group_labels.size());

    // Aggregate the small N*K accumulators to N*P_label (never touches the K*M posterior).
    const bool full = config.paint_full();
    std::vector<double> g_cnt, g_len;
    if (!full) {
        g_cnt.assign(static_cast<std::size_t>(Nrec) * static_cast<std::size_t>(P), 0.0);
        g_len.assign(static_cast<std::size_t>(Nrec) * static_cast<std::size_t>(P), 0.0);
        for (long r = 0; r < Nrec; ++r) {
            for (int k = 0; k < K; ++k) {
                const int gi = group_index[donor_label[static_cast<std::size_t>(k)]];
                const std::size_t src = static_cast<std::size_t>(r) * static_cast<std::size_t>(K) +
                                        static_cast<std::size_t>(k);
                const std::size_t dst = static_cast<std::size_t>(r) * static_cast<std::size_t>(P) +
                                        static_cast<std::size_t>(gi);
                g_cnt[dst] += counts[src];
                g_len[dst] += lengths[src];
            }
        }
    }

    // ---- Emit the coancestry (long format), cM lengths ------------------------------
    const double kCm = steppe::kCentimorgansPerMorgan;
    if (const auto rc = emit_to_destination(
            config, "paint", [&](std::ostream& os, OutputFormat fmt) {
                const char sep = (fmt == OutputFormat::Tsv) ? '\t' : ',';
                const char* col = full ? "donor" : "donor_label";
                if (fmt == OutputFormat::Json) {
                    os << "[\n";
                    bool first_rec = true;
                    for (long r = 0; r < Nrec; ++r) {
                        if (!first_rec) os << ",\n";
                        first_rec = false;
                        const std::string rn =
                            recip_label[static_cast<std::size_t>(r)] + ":" + std::to_string(r);
                        os << "  {\"recipient\": " << json_quote(rn) << ", \"coancestry\": [";
                        const int cols = full ? K : P;
                        for (int c = 0; c < cols; ++c) {
                            const std::size_t idx =
                                static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                                static_cast<std::size_t>(c);
                            const double cc = full ? counts[idx] : g_cnt[idx];
                            const double ll = full ? lengths[idx] : g_len[idx];
                            const std::string cl =
                                full ? (donor_label[static_cast<std::size_t>(c)] + ":" +
                                        std::to_string(c))
                                     : group_labels[static_cast<std::size_t>(c)];
                            if (c) os << ", ";
                            os << "{\"" << col << "\": " << json_quote(cl)
                               << ", \"chunks\": " << json_double(cc)
                               << ", \"length_cM\": " << json_double(ll * kCm) << "}";
                        }
                        os << "]}";
                    }
                    os << "\n]\n";
                    return;
                }
                os << "recipient" << sep << col << sep << "expected_chunks" << sep
                   << "expected_length_cM\n";
                for (long r = 0; r < Nrec; ++r) {
                    const std::string rn =
                        recip_label[static_cast<std::size_t>(r)] + ":" + std::to_string(r);
                    const int cols = full ? K : P;
                    for (int c = 0; c < cols; ++c) {
                        const std::size_t idx =
                            static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                            static_cast<std::size_t>(c);
                        const double cc = full ? counts[idx] : g_cnt[idx];
                        const double ll = full ? lengths[idx] : g_len[idx];
                        const std::string cl =
                            full ? (donor_label[static_cast<std::size_t>(c)] + ":" +
                                    std::to_string(c))
                                 : group_labels[static_cast<std::size_t>(c)];
                        os << csv_field(rn, sep) << sep << csv_field(cl, sep) << sep
                           << fmt_double(cc) << sep << fmt_double(ll * kCm) << "\n";
                    }
                }
            })) {
        return *rc;
    }
    std::fprintf(stderr,
                 "steppe paint: coancestry computed (%ld recipients x %d donors -> %d %s over "
                 "%ld SNPs)\n",
                 Nrec, K, full ? K : P, full ? "donors" : "labels", M);
    return cfg::kExitOk;
}

}  // namespace steppe::app
