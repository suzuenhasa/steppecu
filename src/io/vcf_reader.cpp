// src/io/vcf_reader.cpp
//
// Reference: docs/reference/src_io_vcf_reader.cpp.md
//
// VcfReader implementation — the gVCF interval-join state machine, a faithful
// C++ port of the Stage-0 oracle (experiments/nikki-stage0/oracle.py). Every
// branch is tagged with the oracle line / critique fix it mirrors.
#include "io/vcf_reader.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "io/allele_reconcile.hpp"
#include "io/eigenstrat_format.hpp"
#include "io/gl_normalize.hpp"
#include "io/gzip_line_reader.hpp"
#include "io/vcf_record.hpp"

namespace steppe::io {

namespace {

namespace vd = steppe::io::vcfdetail;

// up()/complement()/reconcile()/Recon are shared with consumer_raw_reader via
// io/allele_reconcile.hpp (hoisted out of this file; behaviour unchanged).

// One tie-broken variant record kept per target position.
struct VariantRec {
    std::string id;
    std::string ref;
    std::string alt;
    bool filt_pass = false;
    std::string gt;
    bool has_dp = false;
    long long dp = 0;
    bool has_gq = false;
    long long gq = 0;
};

// Tie-break: prefer FILTER==PASS, then higher GQ (oracle.py _better(), L1).
[[nodiscard]] bool better(const VariantRec& a, const VariantRec& b) {
    if (a.filt_pass != b.filt_pass) return a.filt_pass;
    // GQ==0 is falsy in the oracle's `(a["gq"] or -1)` — treat it as -1 so a
    // GQ==0 record does NOT out-rank a record carrying no GQ field (fix b).
    const long long ag = (a.has_gq && a.gq != 0) ? a.gq : -1;
    const long long bg = (b.has_gq && b.gq != 0) ? b.gq : -1;
    return ag > bg;
}

// The variant-path genotype resolution (oracle.py _resolve_variant()).
struct VarOut {
    VcfCall call = VcfCall::Missing;
    int dosage = -1;
    int flip = 0;
    std::string reason;
};
[[nodiscard]] VarOut resolve_variant(const VariantRec& rec, char a1, char a2,
                                     const VcfReader::Options& opts) {
    // Floor first (H4): PASS, DP>=min_dp, GQ>=min_gq. A floor of 0 means "field
    // not required" — this lets a phased GT-only hardcall VCF (1000G phase3: no
    // DP/GQ) genotype on the GT alone (--min-dp 0 --min-gq 0). nikki runs the
    // frozen defaults (min_dp=8, min_gq=20, both > 0) so has_dp/has_gq stay
    // REQUIRED and its GRCh38 output is byte-identical.
    if (!rec.filt_pass) return {VcfCall::Missing, -1, 0, "not_pass"};
    if (opts.min_dp > 0 && (!rec.has_dp || rec.dp < opts.min_dp))
        return {VcfCall::Missing, -1, 0, "below_floor"};
    if (opts.min_gq > 0 && (!rec.has_gq || rec.gq < opts.min_gq))
        return {VcfCall::Missing, -1, 0, "below_floor"};

    // Parse the diploid GT.
    const char sep = (rec.gt.find('/') != std::string::npos) ? '/' : '|';
    const std::vector<std::string_view> parts = vd::split(rec.gt, sep);
    if (parts.size() != 2 || parts[0] == "." || parts[1] == ".") {
        return {VcfCall::Missing, -1, 0, "half_or_missing_gt"};  // M4
    }

    const std::vector<std::string_view> alts = vd::split(rec.alt, ',');
    const std::string_view ref = rec.ref;

    const auto allele_base = [&](std::string_view idx_tok, bool& ok) -> std::string_view {
        const auto v = vd::parse_int(idx_tok);
        if (!v) { ok = false; return {}; }
        const long long idx = *v;
        ok = true;
        if (idx == 0) return ref;
        if (idx - 1 < static_cast<long long>(alts.size())) return alts[idx - 1];
        ok = false;  // allele index out of range
        return {};
    };

    int a1_copies = 0;
    bool flip_any = false;
    for (const std::string_view a : parts) {
        bool ok = false;
        const std::string_view b = allele_base(a, ok);
        // spanning-del '*' / indel / bad-index / non-single-base -> MISSING (M4)
        if (!ok || b.size() != 1 || b[0] == '*') {
            return {VcfCall::Missing, -1, 0, "non_panel_allele"};
        }
        const Recon r = reconcile(b[0], a1, a2);
        if (r.which < 0) return {VcfCall::Missing, -1, 0, "non_panel_allele"};  // M4
        flip_any = flip_any || r.flip;
        if (r.which == 1) ++a1_copies;
    }
    // call named from the A1 perspective: 2 A1 copies == homref-A1.
    const VcfCall call =
        (a1_copies == 2) ? VcfCall::Homref : (a1_copies == 1 ? VcfCall::Het : VcfCall::Homalt);
    return {call, a1_copies, flip_any ? 1 : 0, ""};
}

// --- build detection helpers -------------------------------------------------

// Per-autosome (chr1..) reference length, GRCh37 vs GRCh38. The whole panel is
// GRCh37, so only these two builds are auto-detected. Several autosomes are
// listed (not just chr1) so a SINGLE-chromosome VCF — e.g. the 1000G phase3
// per-chromosome files, chr22 only — is still detectable from its own contig
// length (critic fix #2).
struct AutosomeLen {
    int chrom;
    long long g37;
    long long g38;
};
constexpr std::array<AutosomeLen, 6> kAutosomeLen = {{
    {1, 249'250'621, 248'956'422},
    {2, 243'199'373, 242'193'529},
    {3, 198'022'430, 198'295'559},
    {20, 63'025'520, 64'444'167},
    {21, 48'129'895, 46'709'983},
    {22, 51'304'566, 50'818'468},
}};

// Extract the value of `key` from a "##contig=<...>" angle-bracket attr list.
// Anchors the key to a token boundary ('<' or ',') and bounds the value at the
// next ',' or '>' so "ID"/"length" never mis-hit a substring inside another
// value (e.g. assembly=, md5=) (critic fix #6).
[[nodiscard]] std::optional<std::string_view> contig_attr(std::string_view line, std::string_view key) {
    const std::size_t lt = line.find('<');
    if (lt == std::string_view::npos) return std::nullopt;
    std::string_view inner = line.substr(lt + 1);
    const std::size_t gt = inner.rfind('>');
    if (gt != std::string_view::npos) inner = inner.substr(0, gt);
    std::size_t start = 0;
    while (start <= inner.size()) {
        const std::size_t comma = inner.find(',', start);
        const std::string_view tok =
            (comma == std::string_view::npos) ? inner.substr(start) : inner.substr(start, comma - start);
        const std::size_t eq = tok.find('=');
        if (eq != std::string_view::npos && tok.substr(0, eq) == key) return tok.substr(eq + 1);
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return std::nullopt;
}

[[nodiscard]] std::string to_lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

[[nodiscard]] bool has_token(const std::string& hay, std::string_view needle) {
    return hay.find(needle) != std::string::npos;
}

// Resolve a ##reference VALUE (already lower-cased) to a build, or Unknown when
// it names neither / both (critic fix #3: a lone ##reference is spoofable — the
// caller warns when the identity path rests on it alone).
[[nodiscard]] Assembly reference_assembly(const std::string& val) {
    const bool r37 =
        has_token(val, "grch37") || has_token(val, "hg19") || has_token(val, "b37") ||
        has_token(val, "human_g1k_v37");
    const bool r38 = has_token(val, "grch38") || has_token(val, "hg38") || has_token(val, "b38");
    if (r37 && !r38) return Assembly::GRCh37;
    if (r38 && !r37) return Assembly::GRCh38;
    return Assembly::Unknown;  // absent or contradictory
}

}  // namespace

AssemblyDetection detect_vcf_assembly(const std::string& vcf_path) {
    GzipLineReader reader(vcf_path);

    Assembly contig_asm = Assembly::Unknown;
    std::string contig_evidence;
    bool contig_conflict = false;

    Assembly ref_asm = Assembly::Unknown;
    std::string ref_evidence;

    std::string line;
    while (reader.next_line(line)) {
        if (line.rfind("##", 0) != 0) break;  // reached #CHROM / a body line -> stop

        if (line.rfind("##contig", 0) == 0) {
            const auto id = contig_attr(line, "ID");
            const auto len = contig_attr(line, "length");
            if (!id || !len) continue;
            std::string_view idv = *id;
            if (idv.size() > 3 && (idv[0] == 'c' || idv[0] == 'C') && idv[1] == 'h' && idv[2] == 'r') {
                idv.remove_prefix(3);
            }
            long long chrom_ll = 0;
            if (std::from_chars(idv.data(), idv.data() + idv.size(), chrom_ll).ec != std::errc{}) continue;
            long long length = 0;
            if (std::from_chars(len->data(), len->data() + len->size(), length).ec != std::errc{}) continue;
            for (const AutosomeLen& a : kAutosomeLen) {
                if (a.chrom != static_cast<int>(chrom_ll)) continue;
                Assembly hit = Assembly::Unknown;
                if (length == a.g37) hit = Assembly::GRCh37;
                else if (length == a.g38) hit = Assembly::GRCh38;
                if (hit == Assembly::Unknown) break;
                std::string ev = "##contig chr" + std::to_string(a.chrom) + " length=" +
                                 std::to_string(length) + " -> " +
                                 (hit == Assembly::GRCh37 ? "GRCh37" : "GRCh38");
                if (contig_asm == Assembly::Unknown) {
                    contig_asm = hit;
                    contig_evidence = std::move(ev);
                } else if (contig_asm != hit) {
                    contig_conflict = true;
                    contig_evidence += " ; " + ev;
                }
                break;
            }
        } else if (line.rfind("##reference", 0) == 0) {
            const std::size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            const std::string val = to_lower(std::string_view(line).substr(eq + 1));
            const Assembly a = reference_assembly(val);
            if (a != Assembly::Unknown) {
                ref_asm = a;
                ref_evidence = "##reference '" + line.substr(eq + 1) + "' -> " +
                               (a == Assembly::GRCh37 ? "GRCh37" : "GRCh38");
            }
        }
    }

    AssemblyDetection out;
    if (contig_conflict) {
        out.from_contig = true;
        out.evidence = "##contig autosome lengths disagree (" + contig_evidence + ")";
        return out;  // Unknown
    }
    if (contig_asm != Assembly::Unknown && ref_asm != Assembly::Unknown && contig_asm != ref_asm) {
        out.from_contig = true;
        out.from_reference = true;
        out.evidence = "conflict: " + contig_evidence + " vs " + ref_evidence;
        return out;  // Unknown (fail-clear)
    }
    if (contig_asm != Assembly::Unknown) {
        out.assembly = contig_asm;
        out.from_contig = true;
        out.evidence = contig_evidence;
        if (ref_asm == contig_asm) {
            out.from_reference = true;
            out.evidence += " (corroborated by " + ref_evidence + ")";
        }
        return out;
    }
    if (ref_asm != Assembly::Unknown) {
        out.assembly = ref_asm;
        out.from_reference = true;
        out.evidence = ref_evidence;
        return out;
    }
    out.evidence = "no ##contig autosome length or recognizable ##reference token in the header";
    return out;  // Unknown
}

VcfReader::VcfReader(std::string vcf_path, const TargetSites& targets, std::string sample_id,
                     Options opts)
    : vcf_path_(std::move(vcf_path)),
      targets_(targets),
      sample_id_(std::move(sample_id)),
      opts_(opts) {}

VcfIngestResult VcfReader::genotype() {
    GzipLineReader reader(vcf_path_);

    // --- header: locate the sample column ------------------------------------
    int sample_col = -1;
    std::string resolved_sample;
    std::string line;
    bool saw_chrom_header = false;
    while (reader.next_line(line)) {
        if (line.rfind("##", 0) == 0) continue;
        if (line.rfind("#CHROM", 0) == 0 || (!line.empty() && line[0] == '#')) {
            const std::vector<std::string_view> h = vd::split(line, '\t');
            if (h.size() < 10) {
                throw std::runtime_error(
                    "io::VcfReader: #CHROM header has no sample column in " + vcf_path_);
            }
            if (sample_id_.empty()) {
                if (h.size() != 10) {
                    throw std::runtime_error(
                        "io::VcfReader: multiple samples present — pass --sample to select one");
                }
                sample_col = 9;
                resolved_sample = std::string(h[9]);
            } else {
                for (std::size_t c = 9; c < h.size(); ++c) {
                    if (h[c] == sample_id_) { sample_col = static_cast<int>(c); break; }
                }
                if (sample_col < 0) {
                    throw std::runtime_error("io::VcfReader: sample '" + sample_id_ +
                                             "' not found in #CHROM header of " + vcf_path_);
                }
                resolved_sample = sample_id_;
            }
            saw_chrom_header = true;
            break;
        }
    }
    if (!saw_chrom_header) {
        throw std::runtime_error("io::VcfReader: no #CHROM header found in " + vcf_path_);
    }

    // --- gVCF coverage bitmaps + variant map ---------------------------------
    std::unordered_map<int, std::vector<std::uint8_t>> pass_cov, fail_cov;
    for (const auto& [chrom, ci] : targets_.by_chrom) {
        pass_cov[chrom].assign(ci.pos.size(), 0);
        fail_cov[chrom].assign(ci.pos.size(), 0);
    }
    std::unordered_map<int, std::unordered_map<long long, VariantRec>> variant;

    VcfCounts counts;

    while (reader.next_line(line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string_view> f = vd::split(line, '\t');
        if (static_cast<int>(f.size()) <= sample_col) continue;  // truncated record
        ++counts.records_seen;

        // CHROM -> autosome int (strip optional 'chr'); skip non-target chroms.
        std::string_view chrom_sv = f[0];
        if (chrom_sv.size() > 3 && (chrom_sv[0] == 'c' || chrom_sv[0] == 'C') &&
            (chrom_sv[1] == 'h') && (chrom_sv[2] == 'r')) {
            chrom_sv.remove_prefix(3);
        }
        const auto chrom_val = vd::parse_int(chrom_sv);
        if (!chrom_val) continue;  // X/Y/MT/other -> out of scope (M1)
        const int chrom = static_cast<int>(*chrom_val);
        const auto ci_it = targets_.by_chrom.find(chrom);
        if (ci_it == targets_.by_chrom.end()) continue;
        const ChromIndex& ci = ci_it->second;

        const auto pos_opt = vd::parse_int(f[1]);
        if (!pos_opt) continue;
        const long long pos = *pos_opt;
        const std::string_view id = f[2];
        const std::string_view ref = f[3];
        const std::string_view alt = f[4];
        const std::string_view filt = f[6];
        const std::string_view info = f[7];
        const std::string_view format = f[8];
        const std::string_view sample = f[static_cast<std::size_t>(sample_col)];
        const bool filt_pass = (filt == "PASS");

        if (alt == ".") {
            // ---- ref-confidence block (H4 floor; L1 inclusive interval) ------
            const auto end_sv = vd::info_field(info, "END");
            const long long e = end_sv ? vd::parse_int(*end_sv).value_or(pos) : pos;

            // depth = MinDP (INFO) if present, else FORMAT DP.
            std::optional<long long> depth;
            if (const auto mindp = vd::info_field(info, "MinDP")) depth = vd::parse_int(*mindp);
            if (!depth) {
                const int di = vd::format_index(format, "DP");
                depth = vd::parse_int(vd::subfield(sample, di));
            }
            // Same "floor of 0 = field not required" relaxation as the variant
            // path (fix #4, kept symmetric): a GT-only gVCF ref block carrying no
            // depth still counts as passing when --min-dp 0. At the frozen default
            // (min_dp=8 > 0) depth is REQUIRED, so nikki is byte-identical.
            const bool passing =
                filt_pass && (opts_.min_dp <= 0 || (depth.has_value() && *depth >= opts_.min_dp));

            const auto lo = std::lower_bound(ci.pos.begin(), ci.pos.end(), pos);
            const auto hi = std::upper_bound(ci.pos.begin(), ci.pos.end(), e);  // inclusive END
            if (lo >= hi) continue;
            std::vector<std::uint8_t>& flags = passing ? pass_cov[chrom] : fail_cov[chrom];
            for (auto it = lo; it != hi; ++it) {
                flags[static_cast<std::size_t>(it - ci.pos.begin())] = 1;
            }
        } else {
            // ---- explicit variant record ------------------------------------
            if (ci.slot.find(pos) == ci.slot.end()) continue;  // not a target position
            ++counts.variant_at_target;
            VariantRec rec;
            rec.id.assign(id);
            rec.ref.assign(ref);
            rec.alt.assign(alt);
            rec.filt_pass = filt_pass;
            const int gi = vd::format_index(format, "GT");
            rec.gt.assign(vd::subfield(sample, gi));
            const int di = vd::format_index(format, "DP");
            if (const auto d = vd::parse_int(vd::subfield(sample, di))) { rec.has_dp = true; rec.dp = *d; }
            const int qi = vd::format_index(format, "GQ");
            if (const auto q = vd::parse_int(vd::subfield(sample, qi))) { rec.has_gq = true; rec.gq = *q; }

            auto& slot_map = variant[chrom];
            const auto prev = slot_map.find(pos);
            if (prev == slot_map.end() || better(rec, prev->second)) {
                slot_map[pos] = std::move(rec);
            }
        }
    }

    // --- resolve every target site in panel order ----------------------------
    VcfIngestResult out;
    out.sample_id = resolved_sample;
    out.calls.reserve(targets_.sites.size());
    SnpMajorTile& tile = out.tile;
    tile.snp_major.reserve(targets_.sites.size());

    for (const TargetSite& s : targets_.sites) {
        VcfSiteCall c;
        c.rsid = s.rsid;
        c.chrom = s.chrom;
        c.pos37 = s.pos37;
        c.pos38 = s.pos38;
        c.a1 = s.a1;
        c.a2 = s.a2;
        c.call = VcfCall::Dropped;
        c.dosage = -1;
        c.source = "none";
        c.flip = 0;
        c.drop_reason.clear();

        // step 1: palindrome drop (source stays "none", dosage NA).
        if (s.palindrome) {
            c.drop_reason = "palindrome";
            ++counts.drop_palindrome;
            ++counts.dropped_total;
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(kMissingCode);
            continue;
        }

        const ChromIndex& ci = targets_.by_chrom.at(s.chrom);
        const std::size_t i = ci.slot.at(s.pos38);

        // step 2: explicit variant record (absolute precedence over any block).
        const auto vit = variant.find(s.chrom);
        const VariantRec* vrec = nullptr;
        if (vit != variant.end()) {
            const auto r = vit->second.find(s.pos38);
            if (r != vit->second.end()) vrec = &r->second;
        }
        if (vrec != nullptr) {
            // H3 rsID cross-check: emitted BEFORE source="variant" (oracle lazy
            // assignment) so a mismap carries source="none", call="dropped".
            if (vrec->id.rfind("rs", 0) == 0 && vrec->id != s.rsid) {
                c.call = VcfCall::Dropped;
                c.drop_reason = "rsid_mismap";
                ++counts.drop_rsid_mismap;
                ++counts.dropped_total;
                out.calls.push_back(std::move(c));
                tile.snp_major.push_back(kMissingCode);
                continue;
            }
            c.source = "variant";
            const VarOut vo = resolve_variant(*vrec, s.a1, s.a2, opts_);
            c.call = vo.call;
            c.dosage = vo.dosage;
            c.flip = vo.flip;
            c.drop_reason = vo.reason;
            std::uint8_t code = kMissingCode;
            if (vo.call == VcfCall::Missing) {
                ++counts.missing_total;
                if (vo.reason == "not_pass") ++counts.missing_not_pass;
                else if (vo.reason == "below_floor") ++counts.missing_below_floor;
                else if (vo.reason == "half_or_missing_gt") ++counts.missing_half_or_missing_gt;
                else if (vo.reason == "non_panel_allele") ++counts.missing_non_panel;
            } else {
                ++counts.called_variant;
                if (vo.call == VcfCall::Homref) ++counts.homref;
                else if (vo.call == VcfCall::Het) ++counts.het;
                else ++counts.homalt;
                code = static_cast<std::uint8_t>(vo.dosage);
            }
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(code);
            continue;
        }

        // step 3: hom-ref inside a passing ref block (H1 reconciliation).
        if (pass_cov[s.chrom][i]) {
            // Base UNAVAILABLE ('.') -> no_refbase (missing). A real 'N' in the
            // assembly is NOT no_refbase: it flows into reconcile() below, matches
            // neither allele, and drops as ref_change — mirroring the oracle, whose
            // no_refbase fires only when the FASTA fetch returned nothing (fix a).
            if (s.ref38 == '.') {
                c.call = VcfCall::Missing;      // source stays "none" (oracle)
                c.drop_reason = "no_refbase";
                ++counts.missing_total;
                ++counts.missing_no_refbase;
                out.calls.push_back(std::move(c));
                tile.snp_major.push_back(kMissingCode);
                continue;
            }
            const Recon r = reconcile(s.ref38, s.a1, s.a2);
            if (r.which < 0) {
                c.call = VcfCall::Dropped;
                c.source = "refblock";
                c.flip = r.flip ? 1 : 0;
                c.drop_reason = "ref_change";
                ++counts.dropped_total;
                ++counts.drop_ref_change;
                out.calls.push_back(std::move(c));
                tile.snp_major.push_back(kMissingCode);
                continue;
            }
            c.source = "refblock";
            c.flip = r.flip ? 1 : 0;
            c.call = VcfCall::Homref;
            c.dosage = (r.which == 1) ? 2 : 0;  // H1: 2 if REF==A1, 0 if REF==A2
            ++counts.called_refblock;
            ++counts.homref;
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(static_cast<std::uint8_t>(c.dosage));
            continue;
        }

        // step 4: covered only by a below-floor / non-PASS block -> MISSING (H4).
        if (fail_cov[s.chrom][i]) {
            c.call = VcfCall::Missing;
            c.source = "refblock";
            c.drop_reason = "below_floor";
            ++counts.missing_total;
            ++counts.missing_below_floor;
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(kMissingCode);
            continue;
        }

        // step 5: no record covers the site -> MISSING.
        c.call = VcfCall::Missing;
        c.drop_reason = "no_coverage";
        ++counts.missing_total;
        ++counts.missing_no_coverage;
        out.calls.push_back(std::move(c));
        tile.snp_major.push_back(kMissingCode);
    }

    // --- pack the canonical individual-major source tile (1 sample) ----------
    // Each SNP record is one source byte carrying the code in slot 0; the shared
    // device transpose repacks to canonical MSB-first individual-major layout.
    for (std::uint8_t& byte : tile.snp_major) {
        byte = pack_code_into_byte(0, 0, byte);
    }
    tile.src_bytes_per_record = packed_bytes(1);  // == 1
    tile.n_snp = targets_.sites.size();
    tile.sel_rows = {0};
    tile.n_individuals = 1;
    tile.pop_offsets = {0, 1};
    tile.pop_labels = {resolved_sample};

    out.counts = counts;
    return out;
}

namespace {

// The FORMAT key string for each field.
[[nodiscard]] const char* gl_field_key(GlField f) {
    switch (f) {
        case GlField::PL: return "PL";
        case GlField::GL: return "GL";
        case GlField::GP: return "GP";
    }
    return "PL";
}

// One kept variant record per target position for the GL pass — the FORMAT column
// plus the resolved samples' raw sample columns (NOT floored: GL is soft info). No
// GT/DP/GQ parse: the triplet is self-contained and the floor is deliberately not
// applied to the GL tensor.
struct GlVariantRec {
    std::string id;
    std::string ref;
    std::string alt;
    std::string format;
    bool filt_pass = false;
    std::vector<std::string> samples;  // one per resolved sample column, in tile order
};

}  // namespace

VcfReader::LikelihoodResult VcfReader::genotype_likelihoods(GlField field,
                                                            std::vector<RawGlRow>* raw_out,
                                                            bool ancibd_native) {
    GzipLineReader reader(vcf_path_);
    const char* key = gl_field_key(field);

    // --- header: resolve the sample column(s) --------------------------------
    // MULTI-SAMPLE (critic fix #3): unlike genotype() (single-sample-only), the GL
    // tensor is genuinely [n_site x n_sample x 3] — with no --sample we resolve
    // EVERY sample column (>=9), which PCAngsd fundamentally needs. --sample still
    // selects exactly one.
    std::vector<int> sample_cols;
    std::vector<std::string> sample_ids;
    std::string line;
    bool saw_chrom_header = false;
    while (reader.next_line(line)) {
        if (line.rfind("##", 0) == 0) continue;
        if (line.rfind("#CHROM", 0) == 0 || (!line.empty() && line[0] == '#')) {
            const std::vector<std::string_view> h = vd::split(line, '\t');
            if (h.size() < 10) {
                throw std::runtime_error(
                    "io::VcfReader: #CHROM header has no sample column in " + vcf_path_);
            }
            if (sample_id_.empty()) {
                for (std::size_t c = 9; c < h.size(); ++c) {
                    sample_cols.push_back(static_cast<int>(c));
                    sample_ids.emplace_back(h[c]);
                }
            } else {
                int sc = -1;
                for (std::size_t c = 9; c < h.size(); ++c) {
                    if (h[c] == sample_id_) { sc = static_cast<int>(c); break; }
                }
                if (sc < 0) {
                    throw std::runtime_error("io::VcfReader: sample '" + sample_id_ +
                                             "' not found in #CHROM header of " + vcf_path_);
                }
                sample_cols.push_back(sc);
                sample_ids.push_back(sample_id_);
            }
            saw_chrom_header = true;
            break;
        }
    }
    if (!saw_chrom_header) {
        throw std::runtime_error("io::VcfReader: no #CHROM header found in " + vcf_path_);
    }
    const int n_sample = static_cast<int>(sample_cols.size());
    const int max_col = sample_cols.empty() ? 8 : sample_cols.back();

    // --- single streaming pass: keep one variant record per target position ---
    // Only explicit variant records (ALT != '.') at target positions carry a
    // likelihood; ref-confidence blocks (ALT == '.') carry no PL/GL/GP and are
    // skipped (those sites stay missing). Tie-break: prefer FILTER==PASS, else keep
    // first-seen (deterministic; per-sample GQ can't order a multi-sample record).
    std::unordered_map<int, std::unordered_map<long long, GlVariantRec>> variant;
    VcfCounts counts;
    while (reader.next_line(line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string_view> f = vd::split(line, '\t');
        if (static_cast<int>(f.size()) <= max_col) continue;  // truncated record
        ++counts.records_seen;

        std::string_view chrom_sv = f[0];
        if (chrom_sv.size() > 3 && (chrom_sv[0] == 'c' || chrom_sv[0] == 'C') &&
            chrom_sv[1] == 'h' && chrom_sv[2] == 'r') {
            chrom_sv.remove_prefix(3);
        }
        const auto chrom_val = vd::parse_int(chrom_sv);
        if (!chrom_val) continue;
        const int chrom = static_cast<int>(*chrom_val);
        const auto ci_it = targets_.by_chrom.find(chrom);
        if (ci_it == targets_.by_chrom.end()) continue;
        const ChromIndex& ci = ci_it->second;

        const auto pos_opt = vd::parse_int(f[1]);
        if (!pos_opt) continue;
        const long long pos = *pos_opt;
        if (f[4] == ".") continue;                     // ref-confidence block: no likelihood
        if (ci.slot.find(pos) == ci.slot.end()) continue;  // not a target position
        ++counts.variant_at_target;

        const bool filt_pass = (f[6] == "PASS");
        GlVariantRec rec;
        rec.id.assign(f[2]);
        rec.ref.assign(f[3]);
        rec.alt.assign(f[4]);
        rec.format.assign(f[8]);
        rec.filt_pass = filt_pass;
        rec.samples.reserve(sample_cols.size());
        for (const int c : sample_cols) rec.samples.emplace_back(f[static_cast<std::size_t>(c)]);

        auto& slot_map = variant[chrom];
        const auto prev = slot_map.find(pos);
        if (prev == slot_map.end() || (rec.filt_pass && !prev->second.filt_pass)) {
            slot_map[pos] = std::move(rec);
        }
    }

    // --- resolve every target site in panel order into the tile ---------------
    LikelihoodResult out;
    out.sample_id = sample_id_;
    LikelihoodTile& tile = out.tile;
    tile.n_site = targets_.sites.size();
    tile.n_sample = static_cast<std::size_t>(n_sample);
    tile.field = field;
    tile.sample_ids = sample_ids;
    tile.sites.reserve(targets_.sites.size());
    const std::size_t cells = tile.n_site * tile.n_sample;
    tile.l.assign(cells * 3, 0.0);
    tile.present.assign(cells, 0);
    // ancIBD path: keep the phased haplotype bits (native REF/ALT) alongside the tensor.
    if (ancibd_native) {
        tile.native_order = true;
        tile.phased_gt.assign(cells * 2, 0xFFu);  // 0xFF = missing/unset
    }

    // Default every cell to the uninformative triplet (present=0); a resolved,
    // valid triplet overwrites it below.
    for (std::size_t i = 0; i < cells; ++i) {
        tile.l[i * 3 + 0] = glnorm::kUninformative[0];
        tile.l[i * 3 + 1] = glnorm::kUninformative[1];
        tile.l[i * 3 + 2] = glnorm::kUninformative[2];
    }

    for (std::size_t si = 0; si < targets_.sites.size(); ++si) {
        const TargetSite& s = targets_.sites[si];
        LikelihoodSite meta;
        meta.rsid = s.rsid;
        meta.chrom = s.chrom;
        meta.pos37 = s.pos37;
        meta.pos38 = s.pos38;
        meta.a1 = s.a1;
        meta.a2 = s.a2;
        tile.sites.push_back(std::move(meta));

        // Every (site, sample) starts missing; the guards below leave it that way.
        counts.gl_missing += n_sample;
        // Palindrome (A/T, C/G) drop on the hard-call path (strand-ambiguity guard).
        // The ancIBD-native GP path KEEPS palindromes: the imputed VCF is already on
        // the panel's strand (GP is never reconciled against a panel), so the ambiguity
        // does not apply, and reference ancIBD retains these sites. Dropping them ran
        // steppe on ~1.5% fewer markers and pushed a genuinely-related pair below the
        // summary snp_cm density floor. See vcf_reader.hpp `ancibd_native`.
        if (s.palindrome && !ancibd_native) continue;

        const auto vit = variant.find(s.chrom);
        if (vit == variant.end()) continue;
        const auto rit = vit->second.find(s.pos38);
        if (rit == vit->second.end()) continue;  // ref-block / no variant -> missing
        const GlVariantRec& rec = rit->second;

        // H3 rsID cross-check (mirror the hard-call drop).
        if (rec.id.rfind("rs", 0) == 0 && rec.id != s.rsid) {
            ++counts.gl_rsid_mismap;
            continue;
        }
        // Biallelic-only v1: >1 ALT -> skip the whole site.
        if (rec.alt.find(',') != std::string::npos) {
            ++counts.gl_multiallelic_skipped;
            continue;
        }
        // Panel A1/A2 polarity, reusing reconcile() exactly (the single silent-
        // corruption risk). Require REF and ALT to be the two panel alleles; the
        // tensor's g axis is copies-of-A1 == the hard-call dosage axis.
        if (rec.ref.size() != 1 || rec.alt.size() != 1 ||
            rec.ref[0] == '*' || rec.alt[0] == '*') {
            ++counts.gl_non_panel;
            continue;
        }
        const Recon rr = reconcile(rec.ref[0], s.a1, s.a2);
        const Recon ra = reconcile(rec.alt[0], s.a1, s.a2);
        if (rr.which < 0 || ra.which < 0 || rr.which == ra.which) {
            ++counts.gl_non_panel;
            continue;
        }
        // g == j (ALT copies) when ALT==A1 (no swap); g == 2-j (swap RR<->AA) when
        // REF==A1. RA (j==1) is invariant.
        const bool swap = (rr.which == 1);

        const int fi = vd::format_index(rec.format, key);
        if (fi < 0) {
            ++counts.gl_field_absent;
            continue;  // FORMAT lacked the field at this site -> all samples missing
        }
        // ancIBD path: locate the GT subfield once per record for the phased-bit read.
        const int gt_fi = ancibd_native ? vd::format_index(rec.format, "GT") : -1;

        for (int k = 0; k < n_sample; ++k) {
            const std::string_view sub = vd::subfield(rec.samples[static_cast<std::size_t>(k)], fi);
            const std::vector<std::string_view> toks = vd::split(sub, ',');
            if (toks.size() != 3 || sub.empty() || sub == ".") continue;  // missing/wrong arity

            std::array<double, 3> lin;  // VCF-native (RR, RA, AA) order
            bool ok = true;
            if (field == GlField::PL) {
                const auto p0 = vd::parse_int(toks[0]);
                const auto p1 = vd::parse_int(toks[1]);
                const auto p2 = vd::parse_int(toks[2]);
                if (!p0 || !p1 || !p2) ok = false;
                else lin = glnorm::normalize_pl(*p0, *p1, *p2);
            } else if (field == GlField::GL) {
                const auto g0 = vd::parse_double(toks[0]);
                const auto g1 = vd::parse_double(toks[1]);
                const auto g2 = vd::parse_double(toks[2]);
                if (!g0 || !g1 || !g2) ok = false;
                else lin = glnorm::normalize_gl(*g0, *g1, *g2);
            } else {  // GP
                const auto g0 = vd::parse_double(toks[0]);
                const auto g1 = vd::parse_double(toks[1]);
                const auto g2 = vd::parse_double(toks[2]);
                if (!g0 || !g1 || !g2) ok = false;
                else lin = glnorm::normalize_gp(*g0, *g1, *g2);
            }
            if (!ok) continue;  // unparseable / non-finite -> stays missing

            // Place into g == copies-of-A1 order — UNLESS the ancIBD-native path asked
            // for VCF-native (RR, RA, AA) order (so g0=P(hom-REF), g1=P(het) feed
            // LoadH5Multi2.get_haplo_prob on the REF/ALT axis, matching ancIBD exactly).
            const std::size_t b = tile.base(si, static_cast<std::size_t>(k));
            if (swap && !ancibd_native) {
                tile.l[b + 0] = lin[2];
                tile.l[b + 1] = lin[1];
                tile.l[b + 2] = lin[0];
            } else {
                tile.l[b + 0] = lin[0];
                tile.l[b + 1] = lin[1];
                tile.l[b + 2] = lin[2];
            }
            tile.present[tile.mask_index(si, static_cast<std::size_t>(k))] = 1;
            ++counts.gl_present;
            --counts.gl_missing;

            // ancIBD path: parse the two phased haplotype allele bits (native REF/ALT).
            // The GT value is "a|b" (phased; GLIMPSE output) — map '0'->0 (REF/ancestral),
            // '1'->1 (ALT/derived), anything else -> missing. Phase is load-bearing for
            // ancIBD's haplotype-sharing states, so an UNPHASED ('/') heterozygote has no
            // defined hap order and its bits are left missing (0xFF) rather than assigned an
            // arbitrary order; a homozygous '/' call is phase-invariant and is kept.
            if (ancibd_native && gt_fi >= 0) {
                const std::string_view gtv =
                    vd::subfield(rec.samples[static_cast<std::size_t>(k)], gt_fi);
                const bool phased = gtv.find('|') != std::string_view::npos;
                const char asep = phased ? '|' : '/';
                const std::vector<std::string_view> ga = vd::split(gtv, asep);
                if (ga.size() == 2 && ga[0].size() == 1 && ga[1].size() == 1) {
                    const bool unphased_het = !phased && ga[0] != ga[1];
                    if (!unphased_het) {
                        const auto bit = [](char c) -> std::uint8_t {
                            return (c == '0') ? 0u : (c == '1') ? 1u : 0xFFu;
                        };
                        const std::size_t pb = tile.phase_base(si, static_cast<std::size_t>(k));
                        tile.phased_gt[pb + 0] = bit(ga[0][0]);
                        tile.phased_gt[pb + 1] = bit(ga[1][0]);
                    }
                }
            }

            if (raw_out != nullptr) {
                RawGlRow row;
                row.rsid = s.rsid;
                row.chrom = s.chrom;
                row.pos38 = s.pos38;
                row.sample_id = sample_ids[static_cast<std::size_t>(k)];
                row.v0.assign(toks[0]);  // VCF-native order, verbatim (pre-swap)
                row.v1.assign(toks[1]);
                row.v2.assign(toks[2]);
                raw_out->push_back(std::move(row));
            }
        }
    }

    out.counts = counts;
    return out;
}

}  // namespace steppe::io
