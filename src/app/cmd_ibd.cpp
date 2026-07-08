// src/app/cmd_ibd.cpp
//
// The `steppe ibd` handler (ancIBD IBD-segment detection). Builds the 1240K target
// set, reads the imputed VCF's phased GT + GP through the shipped likelihood reader
// (ancIBD-native path), joins the genetic map + derived-allele frequency, resolves
// the sample/pair set, then per chromosome runs the block-per-pair 5-state forward-
// backward (GPU when visible, else the CPU reference oracle), calls IBD segments, and
// writes the per-segment table + the per-pair relatedness summary — mirroring
// ancIBD's output columns.
//
// Reference: docs/planning/ancibd-face-spec.md
#include "app/cmd_ibd.hpp"

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
#include <vector>

#include "app/exit_code_for_caught.hpp"
#include "core/config/exit_code.hpp"
#include "core/stats/ancibd_model.hpp"
#include "core/stats/ancibd_segments.hpp"
#include "device/backend.hpp"
#include "device/backend_factory.hpp"
#include "io/faidx_reader.hpp"
#include "io/target_build.hpp"
#include "io/target_sites.hpp"
#include "io/vcf_reader.hpp"
#include "steppe/config.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

// Read a whitespace two-column "key value" table (rsID -> double), tolerant of a
// leading header row (non-numeric second column) and blank lines.
[[nodiscard]] bool read_key_value(const std::string& path, std::unordered_map<std::string, double>& m,
                                  std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open file: " + path; return false; }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string key, val;
        if (!(ss >> key >> val)) continue;
        try {
            const double v = std::stod(val);
            m[key] = v;
        } catch (...) {
            continue;  // header / unparseable second column -> skip
        }
    }
    return true;
}

// Read a one-column list (one token per line).
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

[[nodiscard]] bool parse_device(const std::string& raw, steppe::DeviceConfig& dc, std::string& err) {
    std::string s;
    for (char c : raw) if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty() || s == "auto") return true;
    try { dc.devices.push_back(std::stoi(s)); }
    catch (...) { err = "--device ordinal '" + raw + "' is not an integer"; return false; }
    return true;
}

[[nodiscard]] bool parse_assembly_flag(const std::string& s, io::Assembly& out) {
    std::string t;
    for (char c : s) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "grch37" || t == "hg19" || t == "b37" || t == "37") { out = io::Assembly::GRCh37; return true; }
    if (t == "grch38" || t == "hg38" || t == "b38" || t == "38") { out = io::Assembly::GRCh38; return true; }
    return false;
}

// Build the target-site set: either a pre-built --targets table or the native
// --panel + --fasta (+ --lift) build (a condensed mirror of `ingest`'s native branch).
[[nodiscard]] bool build_targets(const IbdArgs& a, io::TargetSites& targets, std::string& err) {
    if (!a.targets.empty()) {
        targets = io::read_target_sites(a.targets);
        return true;
    }
    io::AssemblyDetection det = io::detect_vcf_assembly(a.vcf);
    io::Assembly build = det.assembly;
    if (!a.assembly.empty() && !parse_assembly_flag(a.assembly, build)) {
        err = "--assembly must be GRCh37 or GRCh38 (got '" + a.assembly + "')";
        return false;
    }
    if (build == io::Assembly::Unknown) {
        err = "could not detect the VCF assembly; pass --assembly GRCh37|GRCh38";
        return false;
    }
    io::TargetBuildOptions bopts;
    if (build == io::Assembly::GRCh37) {
        bopts.identity_lift = true;
    } else if (a.lift.empty()) {
        err = "a cross-build VCF needs --lift (rsID->pos map)";
        return false;
    }
    if (a.fasta.empty()) { err = "native --panel build needs --fasta"; return false; }
    io::FaidxReader fa(a.fasta);
    io::TargetBuildCounts tc;
    targets = io::build_target_sites(a.panel, a.lift, fa, bopts, tc);
    return true;
}

}  // namespace

int run_ibd(const IbdArgs& args) {
    // --- flag validation -----------------------------------------------------
    if (args.vcf.empty()) {
        std::fprintf(stderr, "steppe ibd: --gp-vcf PATH (the imputed VCF with phased GT + GP) is required\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.targets.empty() == args.panel.empty()) {
        std::fprintf(stderr, "steppe ibd: choose ONE target source — --targets <table> OR --panel <.snp> (+ --fasta)\n");
        return cfg::kExitInvalidConfig;
    }
    if (args.map.empty()) {
        std::fprintf(stderr, "steppe ibd: --map <file> (per-site genetic position) is required\n");
        return cfg::kExitInvalidConfig;
    }
    const bool af_panel = (args.af_mode == "panel");
    const bool af_sample = (args.af_mode == "sample");
    const bool af_half = (args.af_mode == "half");
    if (!af_panel && !af_sample && !af_half) {
        std::fprintf(stderr, "steppe ibd: --af-mode must be panel | sample | half (got '%s')\n", args.af_mode.c_str());
        return cfg::kExitInvalidConfig;
    }
    if (af_panel && args.af.empty()) {
        std::fprintf(stderr, "steppe ibd: --af-mode panel needs --af <file> (derived AF per site)\n");
        return cfg::kExitInvalidConfig;
    }
    const bool map_morgan = (args.map_unit == "morgan");
    if (!map_morgan && args.map_unit != "cm") {
        std::fprintf(stderr, "steppe ibd: --map-unit must be cm | morgan (got '%s')\n", args.map_unit.c_str());
        return cfg::kExitInvalidConfig;
    }
    const char sep = (args.format == "csv") ? ',' : '\t';

    core::AncibdParams pr;
    pr.ibd_in = args.ibd_in;
    pr.ibd_out = args.ibd_out;
    pr.ibd_jump = args.ibd_jump;
    pr.in_val = args.in_val;
    pr.min_error = args.min_error;
    pr.p_min = args.p_min;
    pr.cutoff_post = args.post_cutoff;
    pr.max_gap_merge = args.max_gap_cm / 100.0;  // cM -> Morgan
    pr.min_cm = args.min_cm;

    // --- build targets + read GP + phased GT ---------------------------------
    io::TargetSites targets;
    io::VcfReader::LikelihoodResult glr;
    std::unordered_map<std::string, double> map_pos, af_pos;
    try {
        std::string err;
        if (!build_targets(args, targets, err)) {
            std::fprintf(stderr, "steppe ibd: %s\n", err.c_str());
            return cfg::kExitInvalidConfig;
        }
        io::VcfReader::Options opts;
        io::VcfReader reader(args.vcf, targets, /*sample_id=*/"", opts);
        glr = reader.genotype_likelihoods(io::GlField::GP, nullptr, /*ancibd_native=*/true);

        if (!read_key_value(args.map, map_pos, err)) {
            std::fprintf(stderr, "steppe ibd: --map: %s\n", err.c_str());
            return cfg::kExitIoError;
        }
        if (af_panel && !read_key_value(args.af, af_pos, err)) {
            std::fprintf(stderr, "steppe ibd: --af: %s\n", err.c_str());
            return cfg::kExitIoError;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe ibd: input error: %s\n", e.what());
        return cfg::kExitIoError;
    }

    const io::LikelihoodTile& tile = glr.tile;
    const int n_all = static_cast<int>(tile.n_sample);
    if (n_all < 2) {
        std::fprintf(stderr, "steppe ibd: need >= 2 samples in the VCF (got %d)\n", n_all);
        return cfg::kExitInvalidConfig;
    }

    // --- resolve the selected sample subset ----------------------------------
    std::unordered_map<std::string, int> id_to_col;
    for (int c = 0; c < n_all; ++c) id_to_col[tile.sample_ids[static_cast<std::size_t>(c)]] = c;
    std::vector<int> sel_col;   // tile column for each selected sample
    std::vector<std::string> sel_iid;
    if (!args.samples.empty()) {
        for (const std::string& iid : read_list(args.samples)) {
            const auto it = id_to_col.find(iid);
            if (it == id_to_col.end()) {
                std::fprintf(stderr, "steppe ibd: --samples IID '%s' not found in the VCF\n", iid.c_str());
                return cfg::kExitInvalidConfig;
            }
            sel_col.push_back(it->second);
            sel_iid.push_back(iid);
        }
    } else {
        for (int c = 0; c < n_all; ++c) { sel_col.push_back(c); sel_iid.push_back(tile.sample_ids[static_cast<std::size_t>(c)]); }
    }
    const int ns = static_cast<int>(sel_col.size());
    std::unordered_map<std::string, int> iid_to_sel;
    for (int s = 0; s < ns; ++s) iid_to_sel[sel_iid[static_cast<std::size_t>(s)]] = s;

    // --- resolve the pair set (sel-index space) ------------------------------
    std::vector<std::pair<int, int>> pairs;  // (sel index a, sel index b)
    if (!args.pairs.empty()) {
        std::ifstream pf(args.pairs);
        std::string line;
        while (std::getline(pf, line)) {
            std::istringstream ss(line);
            std::string a, b;
            if (!(ss >> a >> b)) continue;
            const auto ia = iid_to_sel.find(a), ib = iid_to_sel.find(b);
            if (ia == iid_to_sel.end() || ib == iid_to_sel.end()) {
                std::fprintf(stderr, "steppe ibd: --pairs IID '%s'/'%s' not in the selected sample set\n", a.c_str(), b.c_str());
                return cfg::kExitInvalidConfig;
            }
            pairs.emplace_back(ia->second, ib->second);
        }
    } else {
        for (int i = 0; i < ns; ++i)
            for (int j = i + 1; j < ns; ++j) pairs.emplace_back(i, j);
    }
    const int n_pair = static_cast<int>(pairs.size());
    if (n_pair == 0) {
        std::fprintf(stderr, "steppe ibd: no pairs to run\n");
        return cfg::kExitInvalidConfig;
    }

    // --- gather kept sites (present for ALL selected samples + map available),
    //     grouped by chromosome ------------------------------------------------
    struct SiteRef { std::size_t si; double morgan; long long bp; double p_panel; };
    std::unordered_map<int, std::vector<SiteRef>> by_chrom;
    long long n_drop_missing = 0, n_drop_map = 0, n_drop_af = 0;
    const double map_to_morgan = map_morgan ? 1.0 : 0.01;
    for (std::size_t si = 0; si < tile.n_site; ++si) {
        const io::LikelihoodSite& meta = tile.sites[si];
        bool all_present = true;
        for (int s = 0; s < ns; ++s)
            if (tile.present[tile.mask_index(si, static_cast<std::size_t>(sel_col[static_cast<std::size_t>(s)]))] == 0) {
                all_present = false;
                break;
            }
        if (!all_present) { ++n_drop_missing; continue; }
        const auto mit = map_pos.find(meta.rsid);
        if (mit == map_pos.end()) { ++n_drop_map; continue; }
        double p_panel = 0.5;
        if (af_panel) {
            const auto ait = af_pos.find(meta.rsid);
            if (ait == af_pos.end()) { ++n_drop_af; continue; }
            p_panel = ait->second;
        }
        by_chrom[meta.chrom].push_back({si, mit->second * map_to_morgan, meta.pos38, p_panel});
    }

    // --- backend (GPU when visible, else the CPU reference oracle) ------------
    std::unique_ptr<ComputeBackend> be;
    try {
        steppe::DeviceConfig dc;
        std::string derr;
        if (!parse_device(args.device, dc, derr)) {
            std::fprintf(stderr, "steppe ibd: %s\n", derr.c_str());
            return cfg::kExitInvalidConfig;
        }
        int dev = dc.devices.empty() ? 0 : dc.devices.front();
        be = (device::visible_device_count() > 0) ? device::make_cuda_backend(dev)
                                                   : device::make_cpu_backend();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe ibd: device init failed: %s\n", e.what());
        return cfg::kExitIoError;
    }
    const steppe::Precision prec = steppe::Precision::fp64();

    // --- per-chromosome forward-backward + segment calling -------------------
    std::vector<int> chroms;
    for (const auto& kv : by_chrom) chroms.push_back(kv.first);
    std::sort(chroms.begin(), chroms.end());

    std::vector<core::IbdSegment> all_segments;
    const std::size_t nsamp = static_cast<std::size_t>(ns);
    bool map_unit_warned = false;
    for (int ch : chroms) {
        std::vector<SiteRef>& sr = by_chrom[ch];
        std::sort(sr.begin(), sr.end(), [](const SiteRef& x, const SiteRef& y) { return x.morgan < y.morgan; });
        const long M = static_cast<long>(sr.size());
        if (M <= 0) continue;

        // Map-unit sanity guard: a whole chromosome spans ~50-280 cM (0.5-2.8 Morgan).
        // If --map-unit disagrees with the file, the values are off by 100x and IBD
        // calling silently yields empty (cm on a Morgan file) or nonsense (morgan on a
        // cM file). Warn once so a unit mismatch is a visible diagnostic, not a silent
        // 0-segment run (see round-1 gate footgun).
        if (!map_unit_warned) {
            const double span_cm = (sr.back().morgan - sr.front().morgan) * 100.0;
            if (span_cm < 5.0) {
                std::fprintf(stderr,
                    "steppe ibd: WARNING chrom %d spans only %.4g cM after applying "
                    "--map-unit %s; a whole chromosome is ~50-280 cM. The map file is "
                    "likely in Morgan (try --map-unit morgan). Expect few/no segments.\n",
                    ch, span_cm, args.map_unit.c_str());
                map_unit_warned = true;
            } else if (span_cm > 1000.0) {
                std::fprintf(stderr,
                    "steppe ibd: WARNING chrom %d spans %.4g cM after applying "
                    "--map-unit %s; a whole chromosome is ~50-280 cM. The map file is "
                    "likely in cM (try --map-unit cm). Expect inflated segment lengths.\n",
                    ch, span_cm, args.map_unit.c_str());
                map_unit_warned = true;
            }
        }
        const std::size_t Ms = static_cast<std::size_t>(M);

        std::vector<double> gp3(Ms * nsamp * 3, 0.0);
        std::vector<std::uint8_t> phased2(Ms * nsamp * 2, 0u);
        std::vector<double> pvec(Ms, 0.5);
        std::vector<double> rmap(Ms, 0.0);
        std::vector<long long> bp(Ms, 0);
        for (long l = 0; l < M; ++l) {
            const SiteRef& s = sr[static_cast<std::size_t>(l)];
            rmap[static_cast<std::size_t>(l)] = s.morgan;
            bp[static_cast<std::size_t>(l)] = s.bp;
            for (int c = 0; c < ns; ++c) {
                const int col = sel_col[static_cast<std::size_t>(c)];
                const std::size_t src = tile.base(s.si, static_cast<std::size_t>(col));
                const std::size_t dst = (static_cast<std::size_t>(l) * nsamp + static_cast<std::size_t>(c)) * 3;
                gp3[dst + 0] = tile.l[src + 0];
                gp3[dst + 1] = tile.l[src + 1];
                gp3[dst + 2] = tile.l[src + 2];
                const std::size_t psrc = tile.phase_base(s.si, static_cast<std::size_t>(col));
                const std::size_t pdst = (static_cast<std::size_t>(l) * nsamp + static_cast<std::size_t>(c)) * 2;
                phased2[pdst + 0] = tile.phased_gt[psrc + 0];
                phased2[pdst + 1] = tile.phased_gt[psrc + 1];
            }
            // derived AF for this site
            if (af_panel) {
                pvec[static_cast<std::size_t>(l)] = s.p_panel;
            } else if (af_half) {
                pvec[static_cast<std::size_t>(l)] = 0.5;
            } else {  // sample: p_der = 1 - mean over selected haplotypes of P(ancestral)
                double sum = 0.0;
                int cnt = 0;
                for (int c = 0; c < ns; ++c) {
                    const std::size_t pd = (static_cast<std::size_t>(l) * nsamp + static_cast<std::size_t>(c));
                    double hA, hB;
                    core::ancibd_haplo_prob(gp3[pd * 3 + 0], gp3[pd * 3 + 1], phased2[pd * 2 + 0],
                                            phased2[pd * 2 + 1], pr.min_error, hA, hB);
                    sum += hA + hB;
                    cnt += 2;
                }
                pvec[static_cast<std::size_t>(l)] = (cnt > 0) ? 1.0 - sum / cnt : 0.5;
            }
        }
        const std::vector<double> T = core::ancibd_build_transition(rmap, pr);

        std::vector<int> pair_idx(static_cast<std::size_t>(n_pair) * 2);
        for (int q = 0; q < n_pair; ++q) {
            pair_idx[static_cast<std::size_t>(2 * q + 0)] = pairs[static_cast<std::size_t>(q)].first;
            pair_idx[static_cast<std::size_t>(2 * q + 1)] = pairs[static_cast<std::size_t>(q)].second;
        }

        AncibdPosterior post;
        try {
            post = be->ancibd_fb(gp3.data(), phased2.data(), pvec.data(), T.data(), pair_idx.data(),
                                 ns, M, n_pair, pr.in_val, pr.p_min, pr.min_error, prec);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe ibd: FB failed on chromosome %d: %s\n", ch, e.what());
            return exit_code_for_caught(e);
        }
        if (post.status != Status::Ok) {
            std::fprintf(stderr, "steppe ibd: FB returned a non-Ok status on chromosome %d\n", ch);
            return cfg::kExitRuntimeError;
        }

        for (int q = 0; q < n_pair; ++q) {
            std::vector<double> p_ibd(post.p_ibd.begin() + static_cast<long>(static_cast<std::size_t>(q) * Ms),
                                      post.p_ibd.begin() + static_cast<long>((static_cast<std::size_t>(q) + 1) * Ms));
            const std::string& id1 = sel_iid[static_cast<std::size_t>(pairs[static_cast<std::size_t>(q)].first)];
            const std::string& id2 = sel_iid[static_cast<std::size_t>(pairs[static_cast<std::size_t>(q)].second)];
            std::vector<core::IbdSegment> segs = core::ancibd_call_segments(rmap, bp, p_ibd, ch, id1, id2, pr);
            for (core::IbdSegment& sg : segs) all_segments.push_back(std::move(sg));
        }
    }

    const std::vector<core::IbdSummary> summ = core::ancibd_summarize(all_segments);

    // --- emit the per-segment table ------------------------------------------
    auto write_segments = [&](std::ostream& os) {
        os << "iid1" << sep << "iid2" << sep << "ch" << sep << "Start" << sep << "End" << sep
           << "StartM" << sep << "EndM" << sep << "length" << sep << "lengthM" << sep << "lengthCM"
           << sep << "StartBP" << sep << "EndBP" << "\n";
        for (const core::IbdSegment& s : all_segments) {
            os << s.iid1 << sep << s.iid2 << sep << s.ch << sep << s.start << sep << s.end << sep
               << s.startM << sep << s.endM << sep << s.length << sep << s.lengthM << sep
               << (s.lengthM * 100.0) << sep << s.startBP << sep << s.endBP << "\n";
        }
    };
    if (args.out.empty()) {
        write_segments(std::cout);
        std::cout.flush();
    } else {
        std::ofstream os(args.out, std::ios::trunc);
        if (!os) { std::fprintf(stderr, "steppe ibd: cannot open --out: %s\n", args.out.c_str()); return cfg::kExitIoError; }
        write_segments(os);
        if (!os) { std::fprintf(stderr, "steppe ibd: write failed: %s\n", args.out.c_str()); return cfg::kExitIoError; }
    }

    // --- emit the per-pair summary -------------------------------------------
    auto write_summary = [&](std::ostream& os) {
        os << "iid1" << sep << "iid2" << sep << "max_IBD";
        for (double c : core::kIbdSummaryCutoffsCm) {
            os << sep << "sum_IBD>" << static_cast<int>(c) << sep << "n_IBD>" << static_cast<int>(c);
        }
        os << "\n";
        for (const core::IbdSummary& s : summ) {
            os << s.iid1 << sep << s.iid2 << sep << s.max_IBD;
            for (int j = 0; j < 4; ++j) os << sep << s.sum_cm[j] << sep << s.n_seg[j];
            os << "\n";
        }
    };
    std::string summary_path = args.summary;
    if (summary_path.empty() && !args.out.empty()) summary_path = args.out + ".summary";
    if (summary_path.empty()) {
        std::fprintf(stderr, "steppe ibd: per-pair summary (%zu pairs):\n", summ.size());
        write_summary(std::cerr);
    } else {
        std::ofstream os(summary_path, std::ios::trunc);
        if (!os) { std::fprintf(stderr, "steppe ibd: cannot open summary: %s\n", summary_path.c_str()); return cfg::kExitIoError; }
        write_summary(os);
    }

    std::fprintf(stderr,
                 "steppe ibd: %d samples, %d pairs, %zu chromosomes | sites kept per-chr, dropped: "
                 "missing=%lld no_map=%lld no_af=%lld | segments=%zu\n",
                 ns, n_pair, chroms.size(), n_drop_missing, n_drop_map, n_drop_af, all_segments.size());
    return cfg::kExitOk;
}

}  // namespace steppe::app
