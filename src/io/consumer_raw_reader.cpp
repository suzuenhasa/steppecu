// src/io/consumer_raw_reader.cpp
//
// ConsumerRawReader implementation — see the header. Two conceptual passes over
// the raw export: (1) auto-detect the layout from the header and stream every
// record into a by-rsID + by-(chrom,pos37) join index; (2) resolve each panel
// target site in panel order, reconciling the two observed alleles to copies-of-A1
// exactly as the VCF hardcall path, and pack the canonical individual-major tile.
#include "io/consumer_raw_reader.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "io/allele_reconcile.hpp"
#include "io/eigenstrat_format.hpp"  // pack_code_into_byte / packed_bytes / kMissingCode
#include "io/gzip_line_reader.hpp"
#include "io/vcf_record.hpp"  // vcfdetail::split

namespace steppe::io {

namespace {

namespace vd = steppe::io::vcfdetail;

// One observed diploid call: the two raw nucleotide letters (upper-cased). A
// non-ACGT letter ('-', '0', 'D', 'I', '.', ...) marks a no-call at that allele.
struct RawObs {
    char b1 = '0';
    char b2 = '0';
};

[[nodiscard]] bool is_acgt(char c) {
    return c == 'A' || c == 'C' || c == 'G' || c == 'T';
}

// Strip one layer of surrounding double quotes and edge whitespace/CR from a CSV
// cell (MyHeritage quotes every field: "rs123","1","72526","AA").
[[nodiscard]] std::string_view unquote(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
        s.remove_suffix(1);
    }
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        s.remove_prefix(1);
        s.remove_suffix(1);
    }
    return s;
}

// Strict non-negative integer parse of a whole token; false on any non-digit.
[[nodiscard]] bool parse_ll(std::string_view s, long long& out) {
    if (s.empty()) return false;
    long long v = 0;
    for (const char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    out = v;
    return true;
}

// Auto-detect the layout from the header (or first data) line.
[[nodiscard]] RawLayout detect_layout(std::string_view line) {
    if (line.find(',') != std::string_view::npos) return RawLayout::MyHeritage;  // quoted CSV
    const std::vector<std::string_view> f = vd::split(line, '\t');
    if (f.size() >= 5) return RawLayout::AncestryDNA;   // rsid chrom pos allele1 allele2
    if (f.size() >= 4) return RawLayout::TwentyThreeAndMe;  // rsid chrom pos genotype
    return RawLayout::Unknown;
}

// A header row is a non-data line whose first cell names the rsID column.
[[nodiscard]] bool is_header_row(std::string_view line, RawLayout layout) {
    const char delim = (layout == RawLayout::MyHeritage) ? ',' : '\t';
    const std::vector<std::string_view> f = vd::split(line, delim);
    if (f.empty()) return false;
    const std::string_view first = unquote(f[0]);
    return first.size() == 4 && (first[0] == 'r' || first[0] == 'R') &&
           (first[1] == 's' || first[1] == 'S') && (first[2] == 'i' || first[2] == 'I') &&
           (first[3] == 'd' || first[3] == 'D');
}

// Parse one data line into (rsid, chrom, pos37, observed alleles). Returns false
// when the line is malformed for the layout (too few fields / non-numeric pos).
[[nodiscard]] bool parse_record(std::string_view line, RawLayout layout, std::string& rsid,
                                int& chrom, long long& pos37, RawObs& obs) {
    const char delim = (layout == RawLayout::MyHeritage) ? ',' : '\t';
    const std::vector<std::string_view> f = vd::split(line, delim);

    std::size_t min_fields = 4;
    if (layout == RawLayout::AncestryDNA) min_fields = 5;
    if (f.size() < min_fields) return false;

    rsid.assign(unquote(f[0]));
    const std::string_view chrom_sv = unquote(f[1]);
    const std::string_view pos_sv = unquote(f[2]);
    if (!parse_ll(pos_sv, pos37)) return false;
    long long chrom_ll = 0;
    chrom = parse_ll(chrom_sv, chrom_ll) ? static_cast<int>(chrom_ll) : 0;  // X/Y/MT -> 0 (unmatched)

    if (layout == RawLayout::AncestryDNA) {
        const std::string_view a1 = unquote(f[3]);
        const std::string_view a2 = unquote(f[4]);
        obs.b1 = a1.empty() ? '0' : up(a1[0]);
        obs.b2 = a2.empty() ? '0' : up(a2[0]);
    } else {
        // 23andMe genotype / MyHeritage RESULT: two letters (or "--" no-call).
        const std::string_view g = unquote(f[3]);
        if (g.size() >= 2) {
            obs.b1 = up(g[0]);
            obs.b2 = up(g[1]);
        } else if (g.size() == 1) {
            obs.b1 = up(g[0]);  // haploid (X/Y/MT); autosomes always carry 2 letters
            obs.b2 = obs.b1;
        } else {
            obs.b1 = '0';
            obs.b2 = '0';
        }
    }
    return true;
}

}  // namespace

ConsumerRawReader::ConsumerRawReader(std::string raw_path, const TargetSites& targets,
                                     std::string sample_id)
    : raw_path_(std::move(raw_path)), targets_(targets), sample_id_(std::move(sample_id)) {}

VcfIngestResult ConsumerRawReader::genotype() {
    GzipLineReader reader(raw_path_);

    // --- pass 1: detect layout + stream every record into the join index ------
    std::unordered_map<std::string, RawObs> by_rsid;
    std::unordered_map<int, std::unordered_map<long long, RawObs>> by_pos;
    VcfCounts counts;

    std::string line;
    layout_ = RawLayout::Unknown;
    const auto store = [&](std::string_view l) {
        std::string rsid;
        int chrom = 0;
        long long pos37 = 0;
        RawObs obs;
        if (!parse_record(l, layout_, rsid, chrom, pos37, obs)) return;
        ++counts.records_seen;
        if (!rsid.empty()) by_rsid[rsid] = obs;      // last-wins on a duplicate rsID
        by_pos[chrom][pos37] = obs;                  // fallback join key
    };

    // Skip #/## comment lines; the first remaining line detects the layout.
    while (reader.next_line(line)) {
        if (line.empty() || line[0] == '#') continue;
        layout_ = detect_layout(line);
        if (layout_ == RawLayout::Unknown) {
            throw std::runtime_error(
                "io::ConsumerRawReader: could not detect a 23andMe / AncestryDNA / MyHeritage "
                "layout from the header of " + raw_path_);
        }
        if (!is_header_row(line, layout_)) store(line);  // no header row -> it was data
        break;
    }
    if (layout_ == RawLayout::Unknown) {
        throw std::runtime_error("io::ConsumerRawReader: no data lines in " + raw_path_);
    }
    while (reader.next_line(line)) {
        if (line.empty() || line[0] == '#') continue;
        store(line);
    }

    // --- resolve the sample label --------------------------------------------
    std::string sample = sample_id_;
    if (sample.empty()) {
        sample = std::filesystem::path(raw_path_).stem().string();
        if (sample.empty()) sample = "sample";
    }

    // --- pass 2: resolve every target site in panel order --------------------
    VcfIngestResult out;
    out.sample_id = sample;
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

        // step 1: palindrome drop (identical to the VCF path).
        if (s.palindrome) {
            c.drop_reason = "palindrome";
            ++counts.drop_palindrome;
            ++counts.dropped_total;
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(kMissingCode);
            continue;
        }

        // step 2: join by rsID, falling back to chrom+pos37.
        const RawObs* obs = nullptr;
        const auto rit = by_rsid.find(s.rsid);
        if (rit != by_rsid.end()) {
            obs = &rit->second;
        } else {
            const auto cit = by_pos.find(s.chrom);
            if (cit != by_pos.end()) {
                const auto pit = cit->second.find(s.pos37);
                if (pit != cit->second.end()) obs = &pit->second;
            }
        }
        if (obs == nullptr) {  // site not assayed by this chip -> MISSING
            c.call = VcfCall::Missing;
            c.drop_reason = "no_coverage";
            ++counts.missing_total;
            ++counts.missing_no_coverage;
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(kMissingCode);
            continue;
        }
        ++counts.variant_at_target;
        c.source = "raw";

        // step 3: no-call (--, 00, DD/II) or any non-ACGT letter -> MISSING.
        const char b1 = up(obs->b1);
        const char b2 = up(obs->b2);
        if (!is_acgt(b1) || !is_acgt(b2)) {
            c.call = VcfCall::Missing;
            c.drop_reason = "no_call";
            ++counts.missing_total;
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(kMissingCode);
            continue;
        }

        // step 4: reconcile both alleles to copies-of-A1 (A1-counted polarity —
        // 2 A1 copies == homref, matching the VCF hardcall path exactly).
        const Recon r1 = reconcile(b1, s.a1, s.a2);
        const Recon r2 = reconcile(b2, s.a1, s.a2);
        if (r1.which < 0 || r2.which < 0) {  // an allele matches neither panel base
            c.call = VcfCall::Missing;
            c.drop_reason = "non_panel_allele";
            ++counts.missing_total;
            ++counts.missing_non_panel;
            out.calls.push_back(std::move(c));
            tile.snp_major.push_back(kMissingCode);
            continue;
        }
        const int a1_copies = (r1.which == 1 ? 1 : 0) + (r2.which == 1 ? 1 : 0);
        c.flip = (r1.flip || r2.flip) ? 1 : 0;
        c.call = (a1_copies == 2) ? VcfCall::Homref
                                  : (a1_copies == 1 ? VcfCall::Het : VcfCall::Homalt);
        c.dosage = a1_copies;
        ++counts.called_variant;
        if (c.call == VcfCall::Homref) ++counts.homref;
        else if (c.call == VcfCall::Het) ++counts.het;
        else ++counts.homalt;
        out.calls.push_back(std::move(c));
        tile.snp_major.push_back(static_cast<std::uint8_t>(a1_copies));
    }

    // --- pack the canonical individual-major source tile (1 sample) ----------
    // Identical layout to VcfReader::genotype(): one source byte per SNP record
    // carrying the 2-bit code in slot 0; the shared device transpose repacks it.
    for (std::uint8_t& byte : tile.snp_major) {
        byte = pack_code_into_byte(0, 0, byte);
    }
    tile.src_bytes_per_record = packed_bytes(1);  // == 1
    tile.n_snp = targets_.sites.size();
    tile.sel_rows = {0};
    tile.n_individuals = 1;
    tile.pop_offsets = {0, 1};
    tile.pop_labels = {sample};

    out.counts = counts;
    return out;
}

}  // namespace steppe::io
