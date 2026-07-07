// src/io/vcf_reader.cpp
//
// VcfReader implementation — the gVCF interval-join state machine, a faithful
// C++ port of the Stage-0 oracle (experiments/nikki-stage0/oracle.py). Every
// branch is tagged with the oracle line / critique fix it mirrors.
#include "io/vcf_reader.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "io/eigenstrat_format.hpp"
#include "io/gzip_line_reader.hpp"
#include "io/vcf_record.hpp"

namespace steppe::io {

namespace {

namespace vd = steppe::io::vcfdetail;

[[nodiscard]] char up(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

[[nodiscard]] char complement(char b) {
    switch (up(b)) {
        case 'A': return 'T';
        case 'T': return 'A';
        case 'C': return 'G';
        case 'G': return 'C';
        default: return 'N';
    }
}

// reconcile(base, A1, A2) -> which in {+1=A1, 0=A2, -1=none}, flip flag.
// Same strand first, then complement (oracle.py reconcile(), spec §5c).
struct Recon {
    int which = -1;  // +1 == A1, 0 == A2, -1 == neither
    bool flip = false;
};
[[nodiscard]] Recon reconcile(char base, char a1, char a2) {
    const char b = up(base);
    if (b == a1) return {1, false};
    if (b == a2) return {0, false};
    const char cb = complement(b);
    if (cb == a1) return {1, true};
    if (cb == a2) return {0, true};
    return {-1, false};
}

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
    const long long ag = a.has_gq ? a.gq : -1;
    const long long bg = b.has_gq ? b.gq : -1;
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
    // Floor first (H4): PASS, DP>=min_dp, GQ>=min_gq.
    if (!rec.filt_pass) return {VcfCall::Missing, -1, 0, "not_pass"};
    if (!rec.has_dp || rec.dp < opts.min_dp) return {VcfCall::Missing, -1, 0, "below_floor"};
    if (!rec.has_gq || rec.gq < opts.min_gq) return {VcfCall::Missing, -1, 0, "below_floor"};

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

}  // namespace

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
            const bool passing = filt_pass && depth.has_value() && *depth >= opts_.min_dp;

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
            if (s.ref38 == '.' || s.ref38 == 'N') {
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

}  // namespace steppe::io
