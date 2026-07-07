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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
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
#include "io/eigenstrat_format.hpp"
#include "io/faidx_reader.hpp"
#include "io/genotype_tile.hpp"
#include "io/panel_merge_writer.hpp"
#include "io/target_build.hpp"
#include "io/target_sites.hpp"
#include "io/vcf_reader.hpp"
#include "steppe/config.hpp"

namespace steppe::app {

namespace cfg = steppe::config;

namespace {

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

}  // namespace

int run_ingest(const IngestArgs& args) {
    // --- target-site source selection (mutually exclusive) --------------------
    // Native mode is anchored on --panel (the AADR .snp). --fasta/--lift/--assembly/
    // --emit-targets are native-only knobs; --fasta is required (or auto-resolved for
    // GRCh37) and --lift is required ONLY for a cross-build (GRCh38/other) VCF —
    // both are validated after the VCF build is detected (below).
    const bool native = !args.panel.empty();
    const bool legacy = !args.targets.empty();
    const bool native_knobs = !args.fasta.empty() || !args.lift.empty() ||
                              !args.assembly.empty() || !args.emit_targets.empty();
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

    // Genotyping is requested by --report/--emit-tile; a bare native --emit-targets
    // builds the table only (no VCF needed — the gate-1 table-reproduction path).
    const bool want_genotype = !args.report.empty() || !args.emit_tile.empty();
    if (want_genotype && args.vcf.empty()) {
        std::fprintf(stderr, "steppe ingest: --vcf PATH is required for --report/--emit-tile\n");
        return cfg::kExitInvalidConfig;
    }
    if (!want_genotype && args.emit_targets.empty() && args.emit_merged.empty()) {
        std::fprintf(stderr,
                     "steppe ingest: nothing to do — pass --report and/or --emit-tile "
                     "(and, in native mode, optionally --emit-targets/--emit-merged)\n");
        return cfg::kExitInvalidConfig;
    }

    // --- Stage-3 merge flag validation ---------------------------------------
    const bool want_merge = !args.emit_merged.empty();
    if (want_merge) {
        if (args.merge_into.empty()) {
            std::fprintf(stderr, "steppe ingest: --emit-merged requires --merge-into <panel prefix>\n");
            return cfg::kExitInvalidConfig;
        }
        if (args.vcf.empty()) {
            std::fprintf(stderr, "steppe ingest: --emit-merged needs --vcf (the sample to merge)\n");
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
        if (native) {
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

    if (!need_genotype) return cfg::kExitOk;  // native table-only build (gate 1)

    io::VcfIngestResult result;
    try {
        io::VcfReader::Options opts;
        opts.min_dp = args.min_dp;
        opts.min_gq = args.min_gq;
        io::VcfReader reader(args.vcf, targets, args.sample, opts);
        result = reader.genotype();
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
