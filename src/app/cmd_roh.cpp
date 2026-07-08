// src/app/cmd_roh.cpp
//
// The `steppe roh` handler (hapROH runs-of-homozygosity detection). Reads a target
// ancient genotype triple + a phased reference-haplotype panel triple through the shipped
// EIGENSTRAT front-end (exactly as cmd_paint), intersects them onto a common site axis
// (with the strand/polarity contract), decodes the panel to donor-major haplotype bytes +
// the target to pseudo-haploid allele bytes, builds the panel allele-frequency + the per-
// SNP ROH transition, then per chromosome runs the block-per-target (K+1)-state forward-
// backward (GPU when visible, else the CPU reference oracle), calls ROH segments, and
// writes the per-segment table + the per-individual ROH summary — mirroring hapROH.
//
// Reference: docs/planning/haproh-face-spec.md
#include "app/cmd_roh.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "app/exit_code_for_caught.hpp"
#include "core/config/exit_code.hpp"
#include "core/internal/decode_af.hpp"
#include "core/stats/genotype_front_end.hpp"
#include "core/stats/li_stephens.hpp"
#include "core/stats/roh_model.hpp"
#include "core/stats/roh_segments.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "io/genotype_source.hpp"
#include "io/ind_reader.hpp"
#include "io/snp_reader.hpp"
#include "steppe/config.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

// Keep every individual (each haploid column is a haplotype) — MinN with floor 1.
[[nodiscard]] io::PopSelection all_individuals() {
    io::PopSelection sel;
    sel.mode = io::PopSelection::Mode::MinN;
    sel.min_n = 1;
    return sel;
}

// Decode a canonical individual-major tile to a flat haplotype-major allele-byte buffer
// allele[g*M + l] in {0,1,kLsMissingAllele} (exactly cmd_paint::decode_haplotypes). Each
// individual (haploid column) is one target/donor haplotype.
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

// Per-individual pop label, in tile (population-contiguous) order.
[[nodiscard]] std::vector<std::string> per_individual_labels(const io::GenotypeTile& tile) {
    std::vector<std::string> lab(tile.n_individuals, std::string("SAMPLE"));
    const std::size_t P = tile.n_pop();
    if (P > 0 && tile.pop_offsets.size() == P + 1) {
        for (std::size_t p = 0; p < P; ++p)
            for (std::size_t k = tile.pop_offsets[p]; k < tile.pop_offsets[p + 1]; ++k)
                if (k < lab.size()) lab[k] = tile.pop_labels[p];
    }
    return lab;
}

[[nodiscard]] bool parse_device(const std::string& raw, steppe::DeviceConfig& dc, std::string& err) {
    std::string s;
    for (char c : raw)
        if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty() || s == "auto") return true;
    try {
        dc.devices.push_back(std::stoi(s));
    } catch (...) {
        err = "--device ordinal '" + raw + "' is not an integer";
        return false;
    }
    return true;
}

[[nodiscard]] std::vector<double> parse_bins(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try {
            out.push_back(std::stod(tok));
        } catch (...) {
        }
    }
    return out;
}

[[nodiscard]] std::vector<std::string> read_list(const std::string& path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string tok;
        if (ss >> tok) out.push_back(tok);
    }
    return out;
}

}  // namespace

int run_roh(const RohArgs& args) {
    // --- flag validation -----------------------------------------------------
    if (args.prefix.empty()) {
        std::fprintf(stderr,
                     "steppe roh: --prefix PREFIX.{geno,snp,ind} (the TARGET ancient sample) is "
                     "required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.ref_panel.empty()) {
        std::fprintf(stderr,
                     "steppe roh: --ref-panel PREFIX.{geno,snp,ind} (the phased reference panel) "
                     "is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.format != "tsv" && args.format != "csv") {
        std::fprintf(stderr, "steppe roh: --format must be tsv | csv (got '%s')\n",
                     args.format.c_str());
        return cfg::kExitInvalidConfig;
    }
    const char sep = (args.format == "csv") ? ',' : '\t';

    const std::vector<double> bins = parse_bins(args.roh_bins);
    if (bins.empty()) {
        std::fprintf(stderr, "steppe roh: --roh-bins must be a comma list of cM values\n");
        return cfg::kExitInvalidConfig;
    }

    core::RohParams pr;
    pr.e_rate = args.e_rate;
    pr.roh_in = args.roh_in;
    pr.roh_out = args.roh_out;
    pr.roh_jump = args.roh_jump;
    pr.in_val = args.in_val;
    pr.cutoff_post = args.cutoff_post;

    // Fail-fast: with --n-ref set, K = 2*n_ref reference haplotypes and the column-0
    // forward prior alpha_0(0) = 1 - K*in_val (roh_fb.hpp) turns NEGATIVE once
    // 2*n_ref*in_val >= 1 (n_ref >= 1/(2*in_val), ~5000 at the 1e-4 default). Reject up
    // front rather than emit hapsburg's silently-broken negative prior. The 1000G default
    // n_ref=2504 (K=5008, product 0.5008) is safely inside. Checked again below against the
    // ACTUAL selected K for the --n-ref 0 (all-panel) path.
    if (args.n_ref > 0 && !core::roh_prior_valid(2 * args.n_ref, pr.in_val)) {
        std::fprintf(stderr,
                     "steppe roh: --n-ref %d with --in-val %g makes the column-0 ROH prior "
                     "1 - 2*n_ref*in_val = %g negative (need 2*n_ref*in_val < 1, i.e. n_ref < "
                     "%g). Lower --n-ref or --in-val.\n",
                     args.n_ref, pr.in_val, 1.0 - 2.0 * static_cast<double>(args.n_ref) * pr.in_val,
                     1.0 / (2.0 * pr.in_val));
        return cfg::kExitInvalidConfig;
    }

    // --- read target + panel front-ends (host CpuBackend as io/transpose oracle) ---
    core::GenotypeFrontEnd fe_t, fe_p;
    try {
        const io::GenotypeTriple tt = io::resolve_genotype_triple(args.prefix);
        const io::GenotypeTriple pp = io::resolve_genotype_triple(args.ref_panel);
        std::unique_ptr<ComputeBackend> be_io = device::make_cpu_backend();
        fe_t = core::read_genotype_front_end(tt.geno, tt.snp, tt.ind, all_individuals(), *be_io);
        fe_p = core::read_genotype_front_end(pp.geno, pp.snp, pp.ind, all_individuals(), *be_io);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe roh: input error: %s\n", e.what());
        return cfg::kExitIoError;
    }

    const long Mt = static_cast<long>(fe_t.tile.n_snp);
    const long Mp = static_cast<long>(fe_p.tile.n_snp);
    if (Mt <= 0 || Mp <= 0) {
        std::fprintf(stderr, "steppe roh: empty target or panel (target M=%ld, panel M=%ld)\n", Mt,
                     Mp);
        return cfg::kExitInvalidConfig;
    }

    // --- select the panel haplotype columns (exclude-pops + n-ref cap) -------------
    const std::vector<std::string> panel_lab = per_individual_labels(fe_p.tile);
    std::unordered_set<std::string> excl;
    if (!args.exclude_pops.empty()) {
        std::stringstream ss(args.exclude_pops);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim
            std::size_t a = tok.find_first_not_of(" \t");
            std::size_t b = tok.find_last_not_of(" \t");
            if (a != std::string::npos) excl.insert(tok.substr(a, b - a + 1));
        }
    }
    const int Kpanel_all = static_cast<int>(fe_p.tile.n_individuals);
    // K = 2*n_ref haplotype columns cap (each phased individual = 2 pseudo-haploid columns
    // in the staged panel). 0 = keep all.
    const int k_cap = (args.n_ref > 0) ? 2 * args.n_ref : Kpanel_all;
    std::vector<int> panel_cols;
    panel_cols.reserve(static_cast<std::size_t>(Kpanel_all));
    for (int k = 0; k < Kpanel_all && static_cast<int>(panel_cols.size()) < k_cap; ++k) {
        if (excl.count(panel_lab[static_cast<std::size_t>(k)])) continue;
        panel_cols.push_back(k);
    }
    const int K = static_cast<int>(panel_cols.size());
    if (K < 2) {
        std::fprintf(stderr,
                     "steppe roh: need >= 2 reference haplotype columns after exclude/n-ref (got "
                     "%d of %d)\n",
                     K, Kpanel_all);
        return cfg::kExitInvalidConfig;
    }
    // Same column-0 prior guard against the ACTUAL selected K (catches an oversized all-panel
    // run where --n-ref was left at 0 but the panel itself has >= 1/in_val haplotype columns).
    if (!core::roh_prior_valid(K, pr.in_val)) {
        std::fprintf(stderr,
                     "steppe roh: %d selected reference haplotypes with --in-val %g make the "
                     "column-0 ROH prior 1 - K*in_val = %g negative (need K*in_val < 1). Cap the "
                     "panel with --n-ref or raise --in-val.\n",
                     K, pr.in_val, 1.0 - static_cast<double>(K) * pr.in_val);
        return cfg::kExitInvalidConfig;
    }

    // --- select the target individuals (--samples pop-label subset, else all) ------
    const std::vector<std::string> tgt_lab = per_individual_labels(fe_t.tile);
    const int Ntgt_all = static_cast<int>(fe_t.tile.n_individuals);
    std::vector<int> tgt_cols;
    std::vector<std::string> tgt_iid;
    if (!args.samples.empty()) {
        std::unordered_set<std::string> want;
        for (const std::string& s : read_list(args.samples)) want.insert(s);
        for (int g = 0; g < Ntgt_all; ++g)
            if (want.count(tgt_lab[static_cast<std::size_t>(g)])) tgt_cols.push_back(g);
        if (tgt_cols.empty()) {
            std::fprintf(stderr, "steppe roh: no target individual matched --samples\n");
            return cfg::kExitInvalidConfig;
        }
    } else {
        for (int g = 0; g < Ntgt_all; ++g) tgt_cols.push_back(g);
    }
    for (int g : tgt_cols)
        tgt_iid.push_back(tgt_lab[static_cast<std::size_t>(g)] + ":" + std::to_string(g));
    const int Ntgt = static_cast<int>(tgt_cols.size());

    // --- decode both panels to haploid allele bytes --------------------------------
    const std::vector<std::uint8_t> panel_hap = decode_haplotypes(fe_p.tile, Mp);
    const std::vector<std::uint8_t> tgt_hap = decode_haplotypes(fe_t.tile, Mt);

    // --- intersect target ∩ panel by (chrom, physpos); enforce the polarity contract,
    //     grouped by chromosome ------------------------------------------------------
    // Panel site index by (chrom, pos).
    std::unordered_map<long long, std::size_t> panel_by_key;
    panel_by_key.reserve(static_cast<std::size_t>(Mp) * 2);
    auto site_key = [](int ch, long long pos) -> long long {
        return static_cast<long long>(ch) * 1000000000LL + pos;
    };
    for (long l = 0; l < Mp; ++l) {
        const int ch = (static_cast<std::size_t>(l) < fe_p.snptab.chrom.size())
                           ? fe_p.snptab.chrom[static_cast<std::size_t>(l)]
                           : 0;
        const long long pos = static_cast<long long>(
            (static_cast<std::size_t>(l) < fe_p.snptab.physpos.size())
                ? fe_p.snptab.physpos[static_cast<std::size_t>(l)]
                : 0.0);
        panel_by_key.emplace(site_key(ch, pos), static_cast<std::size_t>(l));
    }

    struct KeptSite {
        int ch;
        double morgan;
        long long bp;
        double p_freq;
        std::size_t panel_l;  // panel site index
        std::size_t tgt_l;    // target site index
        bool flip;            // flip the target allele bit (consistent strand flip)
    };
    std::unordered_map<int, std::vector<KeptSite>> by_chrom;
    long long n_drop_nomatch = 0, n_drop_polarity = 0;
    const std::size_t Mps = static_cast<std::size_t>(Mp);
    const std::size_t Mts = static_cast<std::size_t>(Mt);

    for (long lt = 0; lt < Mt; ++lt) {
        const std::size_t lts = static_cast<std::size_t>(lt);
        const int ch = (lts < fe_t.snptab.chrom.size()) ? fe_t.snptab.chrom[lts] : 0;
        const long long pos = static_cast<long long>(
            (lts < fe_t.snptab.physpos.size()) ? fe_t.snptab.physpos[lts] : 0.0);
        const auto it = panel_by_key.find(site_key(ch, pos));
        if (it == panel_by_key.end()) {
            ++n_drop_nomatch;
            continue;
        }
        const std::size_t lp = it->second;
        // Polarity: REF/ALT must match or be a consistent flip; else drop.
        bool flip = false;
        const bool have_ta = lts < fe_t.snptab.ref.size() && lts < fe_t.snptab.alt.size();
        const bool have_pa = lp < fe_p.snptab.ref.size() && lp < fe_p.snptab.alt.size();
        if (have_ta && have_pa) {
            const char tR = fe_t.snptab.ref[lts], tA = fe_t.snptab.alt[lts];
            const char pR = fe_p.snptab.ref[lp], pA = fe_p.snptab.alt[lp];
            if (tR == pR && tA == pA) {
                flip = false;
            } else if (tR == pA && tA == pR) {
                flip = true;
            } else {
                ++n_drop_polarity;
                continue;
            }
        }
        // Panel allele frequency over the SELECTED K haplotype columns (non-missing).
        double sum = 0.0;
        int cnt = 0;
        for (int kk = 0; kk < K; ++kk) {
            const std::uint8_t b = panel_hap[static_cast<std::size_t>(panel_cols[static_cast<std::size_t>(kk)]) * Mps + lp];
            if (b <= 1u) {
                sum += static_cast<double>(b);
                ++cnt;
            }
        }
        const double p_freq = (cnt > 0) ? sum / static_cast<double>(cnt) : 0.5;
        const double morgan = (lp < fe_p.snptab.genpos_morgans.size())
                                  ? fe_p.snptab.genpos_morgans[lp]
                                  : 0.0;
        by_chrom[ch].push_back(KeptSite{ch, morgan, pos, p_freq, lp, lts, flip});
    }

    long long n_kept = 0;
    for (auto& kv : by_chrom) n_kept += static_cast<long long>(kv.second.size());
    if (n_kept == 0) {
        std::fprintf(stderr,
                     "steppe roh: no target site intersected the panel (dropped: no_match=%lld "
                     "polarity=%lld)\n",
                     n_drop_nomatch, n_drop_polarity);
        return cfg::kExitInvalidConfig;
    }

    // --- backend (GPU when visible, else the CPU reference oracle) ------------------
    std::unique_ptr<ComputeBackend> be;
    try {
        steppe::DeviceConfig dc;
        std::string derr;
        if (!parse_device(args.device, dc, derr)) {
            std::fprintf(stderr, "steppe roh: %s\n", derr.c_str());
            return cfg::kExitInvalidConfig;
        }
        int dev = dc.devices.empty() ? 0 : dc.devices.front();
        be = (device::visible_device_count() > 0) ? device::make_cuda_backend(dev)
                                                   : device::make_cpu_backend();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe roh: device init failed: %s\n", e.what());
        return cfg::kExitIoError;
    }
    const steppe::Precision prec = steppe::Precision::fp64();

    // --- per-chromosome forward-backward + segment calling -------------------------
    std::vector<int> chroms;
    for (const auto& kv : by_chrom) chroms.push_back(kv.first);
    std::sort(chroms.begin(), chroms.end());

    std::vector<core::RohSegment> all_segments;
    const std::size_t Ks = static_cast<std::size_t>(K);

    // Pre-sort each chromosome's intersected sites by genetic position (shared over targets).
    for (int ch : chroms)
        std::sort(by_chrom[ch].begin(), by_chrom[ch].end(),
                  [](const KeptSite& a, const KeptSite& b) { return a.morgan < b.morgan; });

    // Exact-hapROH parity path (only_calls=True, docs §3b option 1): run ONE target per FB
    // launch on the target's OWN covered (non-missing) sites, rebuilding the map/allele-freq/
    // transition on that subset — so a target-missing site is DROPPED, not emitted uniform.
    // The kernel is batch-ready over targets (n_target>1, the shared-axis scale path); the CLI
    // uses per-target compaction because the concordance gate wants site-for-site parity.
    for (int t = 0; t < Ntgt; ++t) {
        const std::size_t tcol = static_cast<std::size_t>(tgt_cols[static_cast<std::size_t>(t)]);
        for (int ch : chroms) {
            const std::vector<KeptSite>& sr = by_chrom[ch];
            // Gather this target's non-missing covered sites on this chromosome.
            std::vector<const KeptSite*> cov;
            cov.reserve(sr.size());
            for (const KeptSite& s : sr) {
                const std::uint8_t a = tgt_hap[tcol * Mts + s.tgt_l];
                if (a <= 1u) cov.push_back(&s);
            }
            const long M = static_cast<long>(cov.size());
            if (M <= 0) continue;
            const std::size_t Ms = static_cast<std::size_t>(M);

            std::vector<std::uint8_t> ob(Ms, core::kLsMissingAllele);
            std::vector<std::uint8_t> refhaps(Ks * Ms, core::kLsMissingAllele);
            std::vector<double> pvec(Ms, 0.5);
            std::vector<double> gpos(Ms, 0.0);
            std::vector<long long> bp(Ms, 0);
            for (long l = 0; l < M; ++l) {
                const KeptSite& s = *cov[static_cast<std::size_t>(l)];
                const std::size_t ls = static_cast<std::size_t>(l);
                pvec[ls] = s.p_freq;
                gpos[ls] = s.morgan;
                bp[ls] = s.bp;
                std::uint8_t a = tgt_hap[tcol * Mts + s.tgt_l];
                if (a <= 1u && s.flip) a = static_cast<std::uint8_t>(1u - a);
                ob[ls] = a;
                for (int kk = 0; kk < K; ++kk)
                    refhaps[static_cast<std::size_t>(kk) * Ms + ls] =
                        panel_hap[static_cast<std::size_t>(panel_cols[static_cast<std::size_t>(kk)]) *
                                      Mps + s.panel_l];
            }
            const std::vector<double> T = core::roh_build_transition(gpos, pr, K);

            RohPosterior post;
            try {
                post = be->roh_fb(ob.data(), refhaps.data(), pvec.data(), T.data(), K, M,
                                  /*n_target=*/1, pr.e_rate, pr.in_val, prec);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "steppe roh: FB failed on chromosome %d: %s\n", ch, e.what());
                return exit_code_for_caught(e);
            }
            if (post.status != Status::Ok) {
                std::fprintf(stderr,
                             "steppe roh: FB returned a non-Ok status on chromosome %d\n", ch);
                return cfg::kExitRuntimeError;
            }

            std::vector<core::RohSegment> segs = core::roh_call_segments(
                gpos, bp, post.p_roh, ch, tgt_iid[static_cast<std::size_t>(t)], pr);
            for (core::RohSegment& sg : segs) all_segments.push_back(std::move(sg));
        }
    }

    // Pass the full ordered target list so every target gets a summary row — a zero-ROH
    // individual (no called segments) emits an all-zeros row instead of being dropped.
    const std::vector<core::RohSummary> summ =
        core::roh_summarize(all_segments, bins, /*snp_cm=*/50.0, /*merge_gap_cm=*/0.5,
                            /*merge_min_len1_cm=*/2.0, /*merge_min_len2_cm=*/4.0, tgt_iid);

    // --- emit the per-segment table ------------------------------------------------
    auto write_segments = [&](std::ostream& os) {
        os << "iid" << sep << "ch" << sep << "Start" << sep << "End" << sep << "StartM" << sep
           << "EndM" << sep << "length" << sep << "lengthM" << sep << "lengthCM" << sep
           << "StartBP" << sep << "EndBP" << "\n";
        for (const core::RohSegment& s : all_segments) {
            os << s.iid << sep << s.ch << sep << s.start << sep << s.end << sep << s.startM << sep
               << s.endM << sep << s.length << sep << s.lengthM << sep << (s.lengthM * 100.0) << sep
               << s.startBP << sep << s.endBP << "\n";
        }
    };
    if (args.out.empty()) {
        write_segments(std::cout);
        std::cout.flush();
    } else {
        std::ofstream os(args.out, std::ios::trunc);
        if (!os) {
            std::fprintf(stderr, "steppe roh: cannot open --out: %s\n", args.out.c_str());
            return cfg::kExitIoError;
        }
        write_segments(os);
        if (!os) {
            std::fprintf(stderr, "steppe roh: write failed: %s\n", args.out.c_str());
            return cfg::kExitIoError;
        }
    }

    // --- emit the per-individual summary -------------------------------------------
    auto write_summary = [&](std::ostream& os) {
        os << "iid" << sep << "max_roh";
        for (double c : bins)
            os << sep << "sum_roh>" << static_cast<int>(c) << sep << "n_roh>" << static_cast<int>(c);
        os << "\n";
        for (const core::RohSummary& s : summ) {
            os << s.iid << sep << s.max_roh;
            for (std::size_t j = 0; j < bins.size(); ++j)
                os << sep << s.sum_cm[j] << sep << s.n_seg[j];
            os << "\n";
        }
    };
    std::string summary_path = args.summary;
    if (summary_path.empty() && !args.out.empty()) summary_path = args.out + ".summary";
    if (summary_path.empty()) {
        std::fprintf(stderr, "steppe roh: per-individual summary (%zu targets):\n", summ.size());
        write_summary(std::cerr);
    } else {
        std::ofstream os(summary_path, std::ios::trunc);
        if (!os) {
            std::fprintf(stderr, "steppe roh: cannot open summary: %s\n", summary_path.c_str());
            return cfg::kExitIoError;
        }
        write_summary(os);
    }

    std::fprintf(stderr,
                 "steppe roh: %d targets, K=%d ref haplotypes, %zu chromosomes | sites kept=%lld, "
                 "dropped: no_match=%lld polarity=%lld | segments=%zu\n",
                 Ntgt, K, chroms.size(), n_kept, n_drop_nomatch, n_drop_polarity,
                 all_segments.size());
    return cfg::kExitOk;
}

}  // namespace steppe::app
