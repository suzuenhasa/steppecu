// src/app/cmd_ingest.cpp
//
// Reference: docs/reference/src_app_cmd_ingest.cpp.md
//
// The `steppe ingest` handler: read the target-site table, run the native
// VcfReader over the .vcf.gz, write the per-site report TSV, and (only when
// --emit-tile is given) route the SnpMajorTile through the shared device
// transpose to write the raw canonical 2-bit tile bytes. The report path is
// pure host (no GPU) — the Stage-1 block-correctness gate never needs a device.
#include "app/cmd_ingest.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "app/cmd_common.hpp"
#include "app/exit_code_for_caught.hpp"
#include "core/config/exit_code.hpp"
#include "core/stats/read_canonical_tile.hpp"
#include "device/resources.hpp"
#include "device/vcf_gpu_ingest.hpp"
#include "io/consumer_raw_reader.hpp"
#include "io/eigenstrat_format.hpp"
#include "io/faidx_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/likelihood_tensor_writer.hpp"
#include "io/likelihood_tile.hpp"
#include "io/panel_merge_writer.hpp"
#include "io/target_build.hpp"
#include "io/target_sites.hpp"
#include "io/vcf_panel_reader.hpp"
#include "io/vcf_reader.hpp"
#include "steppe/config.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

[[nodiscard]] const char* raw_layout_name(io::RawLayout l) {
    switch (l) {
        case io::RawLayout::TwentyThreeAndMe: return "23andMe";
        case io::RawLayout::AncestryDNA: return "AncestryDNA";
        case io::RawLayout::MyHeritage: return "MyHeritage";
        case io::RawLayout::Unknown: return "unknown";
    }
    return "unknown";
}

[[nodiscard]] const char* assembly_name(io::Assembly a) {
    switch (a) {
        case io::Assembly::GRCh37: return "GRCh37";
        case io::Assembly::GRCh38: return "GRCh38";
        case io::Assembly::Unknown: return "Unknown";
    }
    return "Unknown";
}

// Parse the --assembly override (case-insensitive; common aliases). GRCh37/GRCh38
// only — other builds (T2T…) go through the cross-build path with user --lift.
[[nodiscard]] bool parse_assembly_flag(const std::string& s, io::Assembly& out) {
    std::string t;
    for (char c : s) t += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (t == "grch37" || t == "hg19" || t == "b37" || t == "37") { out = io::Assembly::GRCh37; return true; }
    if (t == "grch38" || t == "hg38" || t == "b38" || t == "38") { out = io::Assembly::GRCh38; return true; }
    return false;
}

// OPTIONAL convenience (mission item 2): resolve an on-box GRCh37 fasta from the
// STEPPE_GRCH37_FASTA env var. Returns "" when unset / missing — the caller then
// falls back to the normal "needs --fasta" error. No downloads, no hardcoded
// absolute paths (that would be brittle); the gate supplies --fasta explicitly.
[[nodiscard]] std::string resolve_default_grch37_fasta() {
    if (const char* p = std::getenv("STEPPE_GRCH37_FASTA")) {
        std::error_code ec;
        if (p[0] != '\0' && std::filesystem::exists(p, ec)) return std::string(p);
    }
    return "";
}

[[nodiscard]] const char* call_str(io::VcfCall c) {
    switch (c) {
        case io::VcfCall::Homref: return "homref";
        case io::VcfCall::Het: return "het";
        case io::VcfCall::Homalt: return "homalt";
        case io::VcfCall::Missing: return "missing";
        case io::VcfCall::Dropped: return "dropped";
    }
    return "dropped";
}

// Write the per-site report in the Stage-0 oracle schema (spec §9).
[[nodiscard]] bool write_report(const std::string& path, const io::VcfIngestResult& r,
                                std::string& err) {
    std::ofstream o(path, std::ios::trunc);
    if (!o) { err = "cannot open --report file: " + path; return false; }
    o << "rsID\tchrom\tpos37\tpos38\tA1\tA2\tcall\tdosage\tsource\tflip\tdrop_reason\n";
    for (const io::VcfSiteCall& c : r.calls) {
        o << c.rsid << '\t' << c.chrom << '\t' << c.pos37 << '\t' << c.pos38 << '\t' << c.a1
          << '\t' << c.a2 << '\t' << call_str(c.call) << '\t';
        if (c.dosage < 0) o << "NA"; else o << c.dosage;
        o << '\t' << c.source << '\t' << c.flip << '\t' << c.drop_reason << '\n';
    }
    return static_cast<bool>(o);
}

// Build nikki's full-panel 2-bit code vector, aligned to the merge target panel's
// .snp row order (one code per SNP record in the .geno). `calls` is the in-scope
// target subset (panel order); every OTHER panel row (non-rsID / non-autosomal /
// no-lift / dup, or a target row that resolved to missing/dropped) becomes
// kMissingCode. A call is written only when BOTH its rsID AND its pos37 match the
// panel row (the critic's dup/mismap polarity guard) and it is a covered call
// with dosage in {0,1,2}. Streams every non-blank .snp line as one panel row so
// the vector length == the .geno record count.
[[nodiscard]] bool build_panel_codes(const std::string& panel_snp,
                                     const std::vector<io::VcfSiteCall>& calls,
                                     std::vector<std::uint8_t>& out,
                                     long long& n_dup_panel_rsid, std::string& err) {
    // rsID -> call (target rsIDs are unique post de-dup; keep first, flag any dup).
    std::unordered_map<std::string, const io::VcfSiteCall*> by_rsid;
    by_rsid.reserve(calls.size() * 2);
    for (const io::VcfSiteCall& c : calls) by_rsid.emplace(c.rsid, &c);

    std::ifstream in(panel_snp);
    if (!in) { err = "cannot open merge panel .snp: " + panel_snp; return false; }

    out.clear();
    n_dup_panel_rsid = 0;
    std::unordered_map<std::string, int> panel_rsid_seen;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;
            err = "interior blank line in merge panel .snp (desyncs the SNP axis): " + panel_snp;
            return false;
        }
        std::istringstream ls(line);
        std::string rsid, chrom_tok, genpos_tok, physpos_tok;
        ls >> rsid >> chrom_tok >> genpos_tok >> physpos_tok;

        std::uint8_t code = io::kMissingCode;
        const auto it = rsid.empty() ? by_rsid.end() : by_rsid.find(rsid);
        if (it != by_rsid.end()) {
            if (++panel_rsid_seen[rsid] == 2) ++n_dup_panel_rsid;  // panel rsID uniqueness audit
            long long physpos = -1;
            if (!physpos_tok.empty()) {
                const char* b = physpos_tok.data();
                std::from_chars(b, b + physpos_tok.size(), physpos);
            }
            const io::VcfSiteCall& c = *it->second;
            const bool covered = c.call == io::VcfCall::Homref || c.call == io::VcfCall::Het ||
                                 c.call == io::VcfCall::Homalt;
            // pos37 guard: only write the call onto the panel row it was built from.
            if (covered && c.pos37 == physpos && c.dosage >= 0 && c.dosage <= 2) {
                code = static_cast<std::uint8_t>(c.dosage);
            }
        }
        out.push_back(code);
    }
    return true;
}

// Parse a "--device" ordinal list into a DeviceConfig (empty/"auto" -> default).
[[nodiscard]] bool parse_device(const std::string& raw, steppe::DeviceConfig& dc, std::string& err) {
    std::string s;
    for (char c : raw) if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty() || s == "auto") return true;
    std::vector<int> ords;
    std::size_t start = 0;
    while (start <= s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) {
            try { ords.push_back(std::stoi(tok)); }
            catch (...) { err = "--device ordinal '" + tok + "' is not an integer"; return false; }
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    dc.devices = std::move(ords);
    return true;
}

// Parse --gl-field (case-insensitive) into the tile field enum.
[[nodiscard]] bool parse_gl_field(const std::string& s, io::GlField& out) {
    std::string t;
    for (char c : s) t += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (t == "PL") { out = io::GlField::PL; return true; }
    if (t == "GL") { out = io::GlField::GL; return true; }
    if (t == "GP") { out = io::GlField::GP; return true; }
    return false;
}

[[nodiscard]] const char* gl_field_name(io::GlField f) {
    switch (f) {
        case io::GlField::PL: return "PL";
        case io::GlField::GL: return "GL";
        case io::GlField::GP: return "GP";
    }
    return "PL";
}

// Parse a "CHROM:START-END" region (inclusive) into a VcfPanelRegion. The chrom is
// 'chr'-stripped to match the reader's stripped CHROM comparison. Returns false on a
// malformed spec.
[[nodiscard]] bool parse_region(const std::string& s, io::VcfPanelRegion& out, std::string& err) {
    const std::size_t colon = s.find(':');
    if (colon == std::string::npos) {
        err = "--region must be CHROM:START-END (missing ':') — got '" + s + "'";
        return false;
    }
    std::string chrom = s.substr(0, colon);
    if (chrom.size() > 3 && (chrom[0] == 'c' || chrom[0] == 'C') &&
        (chrom[1] == 'h' || chrom[1] == 'H') && (chrom[2] == 'r' || chrom[2] == 'R')) {
        chrom = chrom.substr(3);
    }
    const std::string rng = s.substr(colon + 1);
    const std::size_t dash = rng.find('-');
    if (dash == std::string::npos) {
        err = "--region must be CHROM:START-END (missing '-') — got '" + s + "'";
        return false;
    }
    try {
        out.start = std::stoll(rng.substr(0, dash));
        out.end = std::stoll(rng.substr(dash + 1));
    } catch (...) {
        err = "--region START/END are not integers — got '" + s + "'";
        return false;
    }
    if (out.end < out.start) {
        err = "--region END < START — got '" + s + "'";
        return false;
    }
    out.chrom = std::move(chrom);
    out.active = true;
    return true;
}

// The dedicated phased-VCF -> canonical haplotype-panel path (independent of the
// target-site genotyper). Streams the VCF once, emits the host-only sites x haps
// {0,2,3} matrix for the bit-exact gate, and reports the pass counters + the
// unphased-drop guard on stderr.
[[nodiscard]] int run_phased_vcf_panel(const IngestArgs& args) {
    if (args.emit_hap_codes.empty() && args.emit_tile.empty()) {
        std::fprintf(stderr,
                     "steppe ingest: --phased-vcf needs an output — pass --emit-hap-codes FILE "
                     "(the host-only sites x haps {0,2,3} text matrix) and/or --emit-tile FILE "
                     "(the packed SNP-major binary panel, the decode->pack product)\n");
        return cfg::kExitInvalidConfig;
    }
    io::VcfPanelOptions opts;
    opts.map_path = args.map;
    opts.unphased_max = args.unphased_max;
    if (!args.region.empty()) {
        std::string err;
        if (!parse_region(args.region, opts.region, err)) {
            std::fprintf(stderr, "steppe ingest: %s\n", err.c_str());
            return cfg::kExitInvalidConfig;
        }
    }

    // GPU-native ingest path selector: --gpu OR STEPPE_VCF_GPU=1 routes decode
    // through the nvcomp GPU-DEFLATE + GPU GT-parse + device-resident pack, which
    // returns a BYTE-IDENTICAL VcfPanelResult (the invariance gate). The CPU reader
    // stays the default and the bit-exact reference.
    bool want_gpu = args.gpu;
    if (!want_gpu) {
        const char* e = std::getenv("STEPPE_VCF_GPU");
        if (e != nullptr && e[0] != '\0' && e[0] != '0') want_gpu = true;
    }

    io::VcfPanelResult panel;
    try {
        if (want_gpu) {
            if (!device::vcf_gpu_available()) {
                std::fprintf(stderr,
                             "steppe ingest: --gpu / STEPPE_VCF_GPU requested but steppe was built "
                             "without nvcomp (GPU phased-VCF path unavailable)\n");
                return cfg::kExitInvalidConfig;
            }
            steppe::DeviceConfig dc;
            std::string derr;
            if (!parse_device(args.device, dc, derr)) {
                std::fprintf(stderr, "steppe ingest: %s\n", derr.c_str());
                return cfg::kExitInvalidConfig;
            }
            const int device_id = dc.devices.empty() ? 0 : dc.devices.front();
            panel = device::read_vcf_panel_gpu(args.phased_vcf, opts, device_id);
        } else {
            panel = io::read_vcf_panel(args.phased_vcf, opts);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe ingest: %s\n", e.what());
        return cfg::kExitIoError;
    }

    if (!args.emit_hap_codes.empty()) {
        std::ofstream o(args.emit_hap_codes, std::ios::trunc);
        if (!o) {
            std::fprintf(stderr, "steppe ingest: cannot open --emit-hap-codes file: %s\n",
                         args.emit_hap_codes.c_str());
            return cfg::kExitIoError;
        }
        io::dump_hap_codes(panel.tile, panel.snptab, o);
        if (!o) {
            std::fprintf(stderr, "steppe ingest: failed writing --emit-hap-codes file\n");
            return cfg::kExitIoError;
        }
    }

    // OPTIONAL binary panel: the packed SNP-major 2-bit tile — the reader's real
    // decode->pack product, with NO text formatting. Header = 3 uint64 LE
    // (n_snp, n_individuals, src_bytes_per_record), then the raw snp_major bytes.
    if (!args.emit_tile.empty()) {
        std::ofstream tf(args.emit_tile, std::ios::binary | std::ios::trunc);
        if (!tf) {
            std::fprintf(stderr, "steppe ingest: cannot open --emit-tile file: %s\n",
                         args.emit_tile.c_str());
            return cfg::kExitIoError;
        }
        const std::uint64_t hdr[3] = {
            static_cast<std::uint64_t>(panel.tile.n_snp),
            static_cast<std::uint64_t>(panel.tile.n_individuals),
            static_cast<std::uint64_t>(panel.tile.src_bytes_per_record)};
        tf.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        tf.write(reinterpret_cast<const char*>(panel.tile.snp_major.data()),
                 static_cast<std::streamsize>(panel.tile.snp_major.size()));
        if (!tf) {
            std::fprintf(stderr, "steppe ingest: failed writing --emit-tile file\n");
            return cfg::kExitIoError;
        }
    }

    // OPTIONAL EIGENSTRAT .snp dump: id chrom genpos_Morgans physpos ref alt. genpos is
    // written at full round-trip precision (17 sig digits) so a panel built from this
    // .snp carries genpos BIT-IDENTICAL to the VCF reader's — paint's exact-== marker
    // check then treats a VCF-read panel and a .snp-read panel as the same markers.
    if (!args.emit_snp.empty()) {
        std::ofstream so(args.emit_snp, std::ios::trunc);
        if (!so) {
            std::fprintf(stderr, "steppe ingest: cannot open --emit-snp file: %s\n",
                         args.emit_snp.c_str());
            return cfg::kExitIoError;
        }
        so << std::setprecision(17);
        const io::SnpTable& st = panel.snptab;
        for (std::size_t s = 0; s < st.count; ++s) {
            so << st.id[s] << '\t' << st.chrom[s] << '\t' << st.genpos_morgans[s] << '\t'
               << static_cast<long long>(st.physpos[s]) << '\t' << st.ref[s] << '\t'
               << st.alt[s] << '\n';
        }
        if (!so) {
            std::fprintf(stderr, "steppe ingest: failed writing --emit-snp file\n");
            return cfg::kExitIoError;
        }
    }

    const io::VcfPanelCounts& c = panel.counts;
    const double frac = c.diploid_calls > 0
                            ? static_cast<double>(c.unphased_het_dropped) /
                                  static_cast<double>(c.diploid_calls)
                            : 0.0;
    std::fprintf(stderr,
                 "steppe ingest: phased-VCF panel | samples=%zu haps=%zu sites=%lld | "
                 "skipped multiallelic=%lld non_snp=%lld dup_pos=%lld out_of_region=%lld | "
                 "unphased_het_dropped=%lld/%lld (frac=%.6g) half_missing_haps=%lld "
                 "missing_haps=%lld\n",
                 panel.n_sample, panel.tile.n_individuals, c.emitted_sites,
                 c.skipped_multiallelic, c.skipped_non_snp, c.skipped_dup_pos,
                 c.skipped_out_of_region, c.unphased_het_dropped, c.diploid_calls, frac,
                 c.half_missing_haps, c.missing_haps);
    return cfg::kExitOk;
}

// Write the DEBUG raw-triplet dump (VCF-native order, self-keyed) for the bit-exact
// bcftools gate. TSV: rsID chrom pos38 sample v0 v1 v2.
[[nodiscard]] bool write_raw_gl(const std::string& path, const std::vector<io::RawGlRow>& rows,
                                std::string& err) {
    std::ofstream o(path, std::ios::trunc);
    if (!o) { err = "cannot open --emit-pl-raw file: " + path; return false; }
    o << "rsID\tchrom\tpos38\tsample\tv0\tv1\tv2\n";
    for (const io::RawGlRow& r : rows) {
        o << r.rsid << '\t' << r.chrom << '\t' << r.pos38 << '\t' << r.sample_id << '\t' << r.v0
          << '\t' << r.v1 << '\t' << r.v2 << '\n';
    }
    return static_cast<bool>(o);
}

}  // namespace

int run_ingest(const IngestArgs& args) {
    // --- dedicated phased-VCF -> haplotype-panel path -------------------------
    // A self-contained forward-only streaming reader (no target-site table, no
    // --panel/--fasta/--lift): reads SNPs + individuals INLINE from the phased VCF.
    // Handled first and returned, so none of the target-site validation applies.
    if (!args.phased_vcf.empty()) {
        return run_phased_vcf_panel(args);
    }

    // --- target-site source selection (mutually exclusive) --------------------
    // Native mode is anchored on --panel (the AADR .snp). --fasta/--lift/--assembly/
    // --emit-targets are native-only knobs; --fasta is required (or auto-resolved for
    // GRCh37) and --lift is required ONLY for a cross-build (GRCh38/other) VCF —
    // both are validated after the VCF build is detected (below).
    const bool native = !args.panel.empty();
    const bool legacy = !args.targets.empty();
    const bool native_knobs = !args.fasta.empty() || !args.lift.empty() ||
                              !args.assembly.empty() || !args.emit_targets.empty();

    // --- consumer-raw (23andMe / AncestryDNA / MyHeritage) mode ---------------
    // --raw replaces --vcf as the genotype input: a GRCh37 hardcall export joined
    // to the native --panel by identity (no --fasta / --lift; ref38 unused). It is
    // mutually exclusive with --vcf, requires the native --panel source, and cannot
    // supply GL/PL/GP soft info (those need a VCF).
    const bool raw_mode = !args.raw.empty();
    if (raw_mode) {
        if (!args.vcf.empty()) {
            std::fprintf(stderr, "steppe ingest: choose ONE genotype input — --vcf OR --raw, not both\n");
            return cfg::kExitInvalidConfig;
        }
        if (legacy) {
            std::fprintf(stderr, "steppe ingest: --raw is a native-panel path; anchor it on --panel "
                                 "<.snp>, not the --targets table\n");
            return cfg::kExitInvalidConfig;
        }
        if (!native) {
            std::fprintf(stderr, "steppe ingest: --raw needs --panel <.snp> (the GRCh37 AADR panel to "
                                 "harmonize the consumer file against)\n");
            return cfg::kExitInvalidConfig;
        }
        if (args.likelihoods) {
            std::fprintf(stderr, "steppe ingest: --likelihoods requires --vcf; a consumer raw file "
                                 "carries no GL/PL/GP soft info\n");
            return cfg::kExitInvalidConfig;
        }
    }
    if (native && legacy) {
        std::fprintf(stderr,
                     "steppe ingest: choose ONE target source — either --targets <table> "
                     "(Stage-1) OR --panel (+ --fasta; --lift for a cross-build VCF), not both\n");
        return cfg::kExitInvalidConfig;
    }
    if (legacy && native_knobs) {
        std::fprintf(stderr, "steppe ingest: --fasta/--lift/--assembly/--emit-targets belong to the "
                             "native --panel build, not the --targets table path\n");
        return cfg::kExitInvalidConfig;
    }
    if (!native && !legacy) {
        if (native_knobs) {
            std::fprintf(stderr, "steppe ingest: --fasta/--lift/--assembly/--emit-targets given "
                                 "without --panel; the native build is anchored on --panel <.snp>\n");
        } else {
            std::fprintf(stderr,
                         "steppe ingest: no target source — pass --targets <table> OR "
                         "--panel .snp (+ --fasta ref.fa; --lift only for a cross-build VCF)\n");
        }
        return cfg::kExitInvalidConfig;
    }
    if (args.min_dp < 0 || args.min_gq < 0) {
        std::fprintf(stderr, "steppe ingest: --min-dp/--min-gq must be >= 0\n");
        return cfg::kExitInvalidConfig;
    }

    // --- GL/PL/GP likelihood-mode flag validation ----------------------------
    const bool want_gl = args.likelihoods;
    if (!want_gl && (!args.emit_likelihoods.empty() || !args.emit_pl_raw.empty())) {
        std::fprintf(stderr, "steppe ingest: --emit-likelihoods/--emit-pl-raw require --likelihoods\n");
        return cfg::kExitInvalidConfig;
    }
    io::GlField gl_field = io::GlField::PL;
    if (want_gl) {
        if (!parse_gl_field(args.gl_field, gl_field)) {
            std::fprintf(stderr, "steppe ingest: --gl-field must be PL, GL or GP (got '%s')\n",
                         args.gl_field.c_str());
            return cfg::kExitInvalidConfig;
        }
        if (args.vcf.empty()) {
            std::fprintf(stderr, "steppe ingest: --likelihoods requires --vcf\n");
            return cfg::kExitInvalidConfig;
        }
        if (args.emit_likelihoods.empty() && args.emit_pl_raw.empty()) {
            std::fprintf(stderr, "steppe ingest: --likelihoods needs an output — pass "
                                 "--emit-likelihoods and/or --emit-pl-raw\n");
            return cfg::kExitInvalidConfig;
        }
    }

    // Genotyping is requested by --report/--emit-tile; a bare native --emit-targets
    // builds the table only (no VCF needed — the gate-1 table-reproduction path).
    const bool want_genotype = !args.report.empty() || !args.emit_tile.empty();
    if (want_genotype && args.vcf.empty() && !raw_mode) {
        std::fprintf(stderr, "steppe ingest: --vcf (or --raw) is required for --report/--emit-tile\n");
        return cfg::kExitInvalidConfig;
    }
    if (!want_genotype && !want_gl && args.emit_targets.empty() && args.emit_merged.empty()) {
        std::fprintf(stderr,
                     "steppe ingest: nothing to do — pass --report and/or --emit-tile "
                     "(or --likelihoods; in native mode, optionally --emit-targets/--emit-merged)\n");
        return cfg::kExitInvalidConfig;
    }

    // --- Stage-3 merge flag validation ---------------------------------------
    const bool want_merge = !args.emit_merged.empty();
    if (want_merge) {
        if (args.merge_into.empty()) {
            std::fprintf(stderr, "steppe ingest: --emit-merged requires --merge-into <panel prefix>\n");
            return cfg::kExitInvalidConfig;
        }
        if (args.vcf.empty() && !raw_mode) {
            std::fprintf(stderr, "steppe ingest: --emit-merged needs --vcf or --raw (the sample to merge)\n");
            return cfg::kExitInvalidConfig;
        }
        // The column-alignment guard keys nikki's call onto the panel row by BOTH
        // rsID AND pos37, so the target source (native --panel/--fasta/--lift OR a
        // 7-col --targets table) MUST carry a real pos37. A 6-col table (pos37==0)
        // would fail the guard on every row (all-missing) — a low called-count is
        // the visible signature of that mistake.
    } else if (!args.merge_into.empty()) {
        std::fprintf(stderr, "steppe ingest: --merge-into is only meaningful with --emit-merged\n");
        return cfg::kExitInvalidConfig;
    }
    // The merge needs the genotype calls -> force the genotype pass on even if
    // neither --report nor --emit-tile was requested.
    const bool need_genotype = want_genotype || want_merge;

    // --- build (native) or read (legacy) the target-site set ------------------
    io::TargetSites targets;
    try {
        if (native && raw_mode) {
            // --- consumer-raw native build: GRCh37 identity, no fasta/lift -----
            // The panel is fixed GRCh37 and consumer files are GRCh37 too, so the
            // join is direct (pos38 := pos37) and ref38 is unused (the reader
            // reconciles the two observed alleles against the panel A1/A2). Warn
            // that GRCh37 is assumed (consumer files carry no header the detector
            // reads) and note any ignored native-VCF knobs.
            std::fprintf(stderr, "steppe ingest: --raw mode assumes GRCh37 (consumer files carry no "
                                 "detectable ##reference); identity join, ref38 unused\n");
            if (!args.fasta.empty() || !args.lift.empty() || !args.assembly.empty()) {
                std::fprintf(stderr, "steppe ingest: NOTE --raw ignores --fasta/--lift/--assembly "
                                     "(no reference base or liftover is needed)\n");
            }
            io::TargetBuildOptions bopts;
            io::TargetBuildCounts tc;
            targets = io::build_target_sites_noref(args.panel, bopts, tc);
            std::fprintf(stderr,
                         "steppe ingest: consumer-raw target build | panel_total=%lld autosomal=%lld "
                         "non_rsid=%lld palindromic=%lld dup_rsids=%lld | dropped_dup=%lld | "
                         "emitted=%lld\n",
                         tc.panel_total, tc.panel_autosomal, tc.panel_non_rsid, tc.panel_palindromic,
                         tc.panel_dup_rsids, tc.lift_dropped_dup, tc.emitted);
            if (!args.emit_targets.empty()) io::write_target_table(args.emit_targets, targets);
        } else if (native) {
            // --- resolve the VCF assembly (mission item 1) --------------------
            io::AssemblyDetection det;
            bool have_det = false;
            if (!args.vcf.empty()) {
                det = io::detect_vcf_assembly(args.vcf);
                have_det = true;
                std::fprintf(stderr, "steppe ingest: detected assembly %s (%s)\n",
                             assembly_name(det.assembly), det.evidence.c_str());
            }
            io::Assembly build = io::Assembly::Unknown;
            if (!args.assembly.empty()) {
                if (!parse_assembly_flag(args.assembly, build)) {
                    std::fprintf(stderr,
                                 "steppe ingest: --assembly must be GRCh37 or GRCh38 (got '%s')\n",
                                 args.assembly.c_str());
                    return cfg::kExitInvalidConfig;
                }
                if (have_det && det.assembly != io::Assembly::Unknown && det.assembly != build) {
                    std::fprintf(stderr,
                                 "steppe ingest: WARNING --assembly %s overrides the header-detected "
                                 "%s; proceeding with the override\n",
                                 assembly_name(build), assembly_name(det.assembly));
                }
            } else if (have_det) {
                build = det.assembly;
                if (build == io::Assembly::Unknown) {
                    std::fprintf(stderr,
                                 "steppe ingest: could not detect the VCF assembly from the header "
                                 "(##reference / ##contig autosome length); pass --assembly "
                                 "GRCh37|GRCh38\n");
                    return cfg::kExitInvalidConfig;  // FAIL clearly, never mis-genotype
                }
            } else {
                // Native table-only build (no VCF, e.g. bare --emit-targets): there is
                // no header to read, so --assembly picks the lift path explicitly.
                std::fprintf(stderr, "steppe ingest: a native build without --vcf needs --assembly "
                                     "GRCh37|GRCh38 to select the lift path\n");
                return cfg::kExitInvalidConfig;
            }

            // --- branch on the VCF build vs the fixed GRCh37 panel ------------
            io::TargetBuildOptions bopts;
            if (build == io::Assembly::GRCh37) {
                bopts.identity_lift = true;  // same-build direct join, no lift needed
                if (!args.lift.empty()) {
                    std::fprintf(stderr, "steppe ingest: NOTE a same-build GRCh37 VCF is lift-free; "
                                         "ignoring --lift (identity join pos38:=pos37)\n");
                }
                // fix #3: the identity path applied to a MISLABELED/lifted VCF silently
                // mis-genotypes; the only structural guard (H3 rsID cross-check) covers
                // rsID-bearing records only. Warn loudly when GRCh37 rests on a spoofable
                // ##reference token ALONE (no corroborating contig length) and no override.
                if (args.assembly.empty() && have_det && det.from_reference && !det.from_contig) {
                    std::fprintf(stderr,
                                 "steppe ingest: WARNING GRCh37 chosen from ##reference alone (no "
                                 "##contig length to corroborate); a mislabeled/lifted VCF would be "
                                 "mis-genotyped on the identity path — pass --assembly to confirm\n");
                }
            } else {  // GRCh38 (or an --assembly override for another cross build)
                if (args.lift.empty()) {
                    std::fprintf(stderr,
                                 "steppe ingest: a %s VCF against the GRCh37 panel needs --lift "
                                 "(rsID->pos map); only a same-build GRCh37 VCF is lift-free\n",
                                 assembly_name(build));
                    return cfg::kExitInvalidConfig;
                }
            }

            // --- resolve the build-matched fasta (GRCh37 may auto-resolve) ----
            std::string fasta_path = args.fasta;
            if (fasta_path.empty() && build == io::Assembly::GRCh37) {
                fasta_path = resolve_default_grch37_fasta();
                if (!fasta_path.empty()) {
                    std::fprintf(stderr, "steppe ingest: using STEPPE_GRCH37_FASTA=%s\n",
                                 fasta_path.c_str());
                }
            }
            if (fasta_path.empty()) {
                std::fprintf(stderr,
                             "steppe ingest: native build needs --fasta (a %s reference .fa with a "
                             "sibling .fai)\n",
                             assembly_name(build));
                return cfg::kExitInvalidConfig;
            }

            io::FaidxReader fa(fasta_path);
            io::TargetBuildCounts tc;
            targets = io::build_target_sites(args.panel, args.lift, fa, bopts, tc);
            std::fprintf(stderr,
                         "steppe ingest: native target build | panel_total=%lld autosomal=%lld "
                         "non_rsid=%lld palindromic=%lld dup_rsids=%lld | lift_ok=%lld "
                         "no_lift=%lld dropped_dup=%lld | emitted=%lld\n",
                         tc.panel_total, tc.panel_autosomal, tc.panel_non_rsid, tc.panel_palindromic,
                         tc.panel_dup_rsids, tc.lift_ok, tc.lift_no_lift, tc.lift_dropped_dup,
                         tc.emitted);
            if (!args.emit_targets.empty()) io::write_target_table(args.emit_targets, targets);
        } else {
            targets = io::read_target_sites(args.targets);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe ingest: %s\n", e.what());
        return cfg::kExitIoError;
    }

    if (!need_genotype && !want_gl) return cfg::kExitOk;  // native table-only build (gate 1)

    // --- GL/PL/GP likelihood path (independent of the hard-call genotype pass) -
    if (want_gl) {
        io::VcfReader::LikelihoodResult glr;
        std::vector<io::RawGlRow> raw;
        try {
            io::VcfReader::Options opts;
            opts.min_dp = args.min_dp;  // carried but NOT applied to the GL tensor (soft info)
            opts.min_gq = args.min_gq;
            io::VcfReader reader(args.vcf, targets, args.sample, opts);
            glr = reader.genotype_likelihoods(gl_field, args.emit_pl_raw.empty() ? nullptr : &raw);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe ingest: %s\n", e.what());
            return cfg::kExitIoError;
        }

        if (!args.emit_pl_raw.empty()) {
            std::string err;
            if (!write_raw_gl(args.emit_pl_raw, raw, err)) {
                std::fprintf(stderr, "steppe ingest: %s\n", err.c_str());
                return cfg::kExitIoError;
            }
        }

        // --emit-likelihoods: upload the tensor device-resident, prove residency
        // via the device checksum vs the host sum, then write the STPGL1 artifact.
        if (!args.emit_likelihoods.empty()) {
            try {
                steppe::DeviceConfig dc;
                std::string derr;
                if (!parse_device(args.device, dc, derr)) {
                    std::fprintf(stderr, "steppe ingest: %s\n", derr.c_str());
                    return cfg::kExitInvalidConfig;
                }
                device::Resources resources = device::build_resources(dc);
                if (!require_first_gpu(resources, "ingest")) return cfg::kExitRuntimeError;
                ComputeBackend& backend = *resources.gpus.front().backend;
                const io::LikelihoodTile& tile = glr.tile;
                device::LikelihoodTensor tensor = backend.upload_likelihood_tensor(
                    tile.l.data(), tile.present.data(), static_cast<long>(tile.n_site),
                    static_cast<int>(tile.n_sample));
                const double dev_sum = backend.likelihood_tensor_checksum(tensor);
                double host_sum = 0.0;
                for (const double v : tile.l) host_sum += v;
                const double tol = 1e-6 * (1.0 + std::abs(host_sum));
                if (std::abs(dev_sum - host_sum) > tol) {
                    std::fprintf(stderr,
                                 "steppe ingest: GL device residency check FAILED "
                                 "(device_sum=%.10g host_sum=%.10g)\n",
                                 dev_sum, host_sum);
                    return cfg::kExitRuntimeError;
                }
                std::fprintf(stderr,
                             "steppe ingest: GL tensor device-resident on GPU %d (residency "
                             "checksum OK: device_sum=%.6f == host_sum=%.6f)\n",
                             tensor.device_id, dev_sum, host_sum);
                io::write_likelihood_tensor(args.emit_likelihoods, tile);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "steppe ingest: GL device/write error: %s\n", e.what());
                return exit_code_for_caught(e);
            }
        }

        const io::VcfCounts& gc = glr.counts;
        std::fprintf(stderr,
                     "steppe ingest: GL field=%s sites=%zu samples=%zu | present=%lld missing=%lld "
                     "| multiallelic_skipped=%lld non_panel=%lld field_absent=%lld rsid_mismap=%lld "
                     "| variant_at_target=%lld records=%lld\n",
                     gl_field_name(gl_field), glr.tile.n_site, glr.tile.n_sample, gc.gl_present,
                     gc.gl_missing, gc.gl_multiallelic_skipped, gc.gl_non_panel, gc.gl_field_absent,
                     gc.gl_rsid_mismap, gc.variant_at_target, gc.records_seen);

        if (!need_genotype) return cfg::kExitOk;  // GL-only invocation done
    }

    io::VcfIngestResult result;
    try {
        if (raw_mode) {
            io::ConsumerRawReader reader(args.raw, targets, args.sample);
            result = reader.genotype();
            std::fprintf(stderr, "steppe ingest: consumer raw layout detected = %s\n",
                         raw_layout_name(reader.layout()));
        } else {
            io::VcfReader::Options opts;
            opts.min_dp = args.min_dp;
            opts.min_gq = args.min_gq;
            io::VcfReader reader(args.vcf, targets, args.sample, opts);
            result = reader.genotype();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "steppe ingest: %s\n", e.what());
        return cfg::kExitIoError;
    }

    if (!args.report.empty()) {
        std::string err;
        if (!write_report(args.report, result, err)) {
            std::fprintf(stderr, "steppe ingest: %s\n", err.c_str());
            return cfg::kExitIoError;
        }
    }

    // --- Stage-3 merge: append nikki as a size-1 population into the panel ----
    if (want_merge) {
        std::vector<std::uint8_t> codes;
        long long n_dup_panel_rsid = 0;
        std::string err;
        if (!build_panel_codes(args.merge_into + ".snp", result.calls, codes, n_dup_panel_rsid,
                               err)) {
            std::fprintf(stderr, "steppe ingest: %s\n", err.c_str());
            return cfg::kExitIoError;
        }
        try {
            const io::MergeCounts mc = io::write_merged_panel(args.merge_into, args.emit_merged,
                                                             codes, result.sample_id);
            const char* fmt = mc.format == io::GenoFormat::Tgeno ? "TGENO"
                              : mc.format == io::GenoFormat::Geno ? "GENO/PA"
                              : "EIGENSTRAT";
            std::fprintf(stderr,
                         "steppe ingest: merged %s panel -> %s | n_snp=%lld n_ind %lld->%lld | "
                         "nikki called=%lld missing=%lld | panel_dup_rsid_hits=%lld\n",
                         fmt, args.emit_merged.c_str(), mc.n_snp, mc.n_ind_src, mc.n_ind_out,
                         mc.n_called, mc.n_missing, n_dup_panel_rsid);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe ingest: merge failed: %s\n", e.what());
            return cfg::kExitIoError;
        }
    }

    // The canonical-tile path needs a device for the shared transpose.
    if (!args.emit_tile.empty()) {
        try {
            steppe::DeviceConfig dc;
            std::string derr;
            if (!parse_device(args.device, dc, derr)) {
                std::fprintf(stderr, "steppe ingest: %s\n", derr.c_str());
                return cfg::kExitInvalidConfig;
            }
            device::Resources resources = device::build_resources(dc);
            if (!require_first_gpu(resources, "ingest")) return cfg::kExitRuntimeError;
            ComputeBackend& backend = *resources.gpus.front().backend;
            const io::GenotypeTile canon = core::transpose_snp_major(result.tile, backend);
            std::ofstream tf(args.emit_tile, std::ios::binary | std::ios::trunc);
            if (!tf) {
                std::fprintf(stderr, "steppe ingest: cannot open --emit-tile file: %s\n",
                             args.emit_tile.c_str());
                return cfg::kExitIoError;
            }
            tf.write(reinterpret_cast<const char*>(canon.packed.data()),
                     static_cast<std::streamsize>(canon.packed.size()));
            if (!tf) {
                std::fprintf(stderr, "steppe ingest: failed writing --emit-tile file\n");
                return cfg::kExitIoError;
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "steppe ingest: device/transpose error: %s\n", e.what());
            return exit_code_for_caught(e);
        }
    }

    const io::VcfCounts& c = result.counts;
    std::fprintf(stderr,
                 "steppe ingest: sample=%s sites=%zu | called_variant=%lld called_refblock=%lld "
                 "(homref=%lld het=%lld homalt=%lld) | missing=%lld dropped=%lld | records=%lld\n",
                 result.sample_id.c_str(), result.calls.size(), c.called_variant, c.called_refblock,
                 c.homref, c.het, c.homalt, c.missing_total, c.dropped_total, c.records_seen);
    return cfg::kExitOk;
}

}  // namespace steppe::app
