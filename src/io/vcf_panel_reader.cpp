// src/io/vcf_panel_reader.cpp
//
// The multi-sample phased-VCF -> canonical haplotype-panel reader (Phase-1 core).
// Streams the .vcf.gz once, decodes every kept biallelic-SNP record's phased GT
// into two haploid columns per sample, and packs them SNP-major into an
// io::SnpMajorTile with codes {0,2,3}. See vcf_panel_reader.hpp for the contract.
//
// Reuse (SHAPE, cited): the tab/format/pipe tokenizers vcfdetail::split /
// format_index / subfield / parse_int (io/vcf_record.hpp); the multi-sample
// #CHROM all-columns scan (vcf_reader.cpp:608-643); the phased bit-split
// (vcf_reader.cpp:836-852), here hoisted to a top-level UNCONDITIONAL per-sample
// loop, made per-hap independent, and remapped to the panel's 2-bit code (0/2/3)
// instead of the GL path's 0xFF sentinel; the 2-bit packers pack_code_into_byte /
// code_in_byte / packed_bytes (io/eigenstrat_format.hpp); the SNP-major fresh-pack
// layout (geno_reader.cpp read_eigenstrat_snp_major_tile: src_bpr=packed_bytes(all
// source rows), byte_in_rec = i/4, pack_code_into_byte(byte, i, code)).
#include "io/vcf_panel_reader.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io/eigenstrat_format.hpp"
#include "io/gzip_line_reader.hpp"
#include "io/vcf_record.hpp"

namespace steppe::io {

namespace vd = vcfdetail;

namespace {

// Strip a leading 'chr'/'CHR' from a CHROM token (phase-3 uses bare "22", but be
// permissive). Returns a view over the original buffer.
[[nodiscard]] std::string_view strip_chr(std::string_view s) {
    if (s.size() > 3 && (s[0] == 'c' || s[0] == 'C') && (s[1] == 'h' || s[1] == 'H') &&
        (s[2] == 'r' || s[2] == 'R')) {
        s.remove_prefix(3);
    }
    return s;
}

// Map a 'chr'-stripped CHROM token to the .snp integer convention (numeric, or
// X/Y/MT -> 23/24/90, else -1). Only used for SnpTable.chrom; region filtering
// compares the string form.
[[nodiscard]] int chrom_to_int(std::string_view stripped) {
    if (stripped == "X" || stripped == "x") return kChromCodeX;
    if (stripped == "Y" || stripped == "y") return kChromCodeY;
    if (stripped == "MT" || stripped == "mt" || stripped == "M" || stripped == "m")
        return kChromCodeMt;
    const auto v = vd::parse_int(stripped);
    return v ? static_cast<int>(*v) : kFirstOtherChromCode;
}

// A single-base SNP allele: exactly one A/C/G/T/N nucleotide (rejects indels,
// '*' spanning deletions, '<SYMBOLIC>', and '.'). Mirrors bcftools `-v snps`.
[[nodiscard]] bool is_snp_allele(std::string_view a) {
    if (a.size() != 1) return false;
    switch (a[0]) {
        case 'A': case 'C': case 'G': case 'T': case 'N':
        case 'a': case 'c': case 'g': case 't': case 'n':
            return true;
        default:
            return false;
    }
}

// The per-haplotype allele-token -> 2-bit code map (the load-bearing {0,2,3}
// contract): '0'->0, '1'->2, everything else (".", multi-digit, empty) -> 3
// (missing). NEVER emits code 1.
[[nodiscard]] std::uint8_t hap_code(std::string_view allele) {
    if (allele.size() == 1) {
        if (allele[0] == '0') return 0u;
        if (allele[0] == '1') return 2u;
    }
    return kMissingCode;  // 3
}

// A per-chromosome genetic map: (bp, cM) sorted ascending by bp, for linear
// interpolation to Morgans.
struct ChromMap {
    std::vector<long long> bp;
    std::vector<double> cm;
};

// Parse a plink-format .map (whitespace: chrom id cM bp) into per-chrom (bp,cM)
// tables. Non-conforming / header lines (a bp or cM token that will not parse)
// are skipped. Kept sorted by bp per chromosome.
[[nodiscard]] std::unordered_map<int, ChromMap> read_genetic_map(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_vcf_panel: cannot open --map genetic map: " + path);
    }
    std::unordered_map<int, ChromMap> maps;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string chrom_tok, id_tok, cm_tok, bp_tok;
        if (!(ls >> chrom_tok >> id_tok >> cm_tok >> bp_tok)) continue;  // need 4 columns
        char* endp = nullptr;
        const double cm = std::strtod(cm_tok.c_str(), &endp);
        if (endp != cm_tok.c_str() + cm_tok.size()) continue;  // header / non-numeric cM
        const auto bp = vd::parse_int(bp_tok);
        if (!bp) continue;
        const int chrom = chrom_to_int(strip_chr(chrom_tok));
        ChromMap& m = maps[chrom];
        m.bp.push_back(*bp);
        m.cm.push_back(cm);
    }
    for (auto& [chrom, m] : maps) {
        (void)chrom;
        std::vector<std::size_t> order(m.bp.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) { return m.bp[a] < m.bp[b]; });
        ChromMap sorted;
        sorted.bp.reserve(m.bp.size());
        sorted.cm.reserve(m.cm.size());
        for (std::size_t i : order) {
            sorted.bp.push_back(m.bp[i]);
            sorted.cm.push_back(m.cm[i]);
        }
        m = std::move(sorted);
    }
    return maps;
}

// Linear-interpolate cM at bp within a per-chrom map, clamping to the endpoints
// (keeps genpos monotonic non-decreasing, which paint requires). Returns Morgans.
[[nodiscard]] double interp_morgans(const ChromMap& m, long long bp) {
    if (m.bp.empty()) return 0.0;
    if (bp <= m.bp.front()) return m.cm.front() / 100.0;
    if (bp >= m.bp.back()) return m.cm.back() / 100.0;
    // upper_bound: first entry with bp_i > bp
    const auto it = std::upper_bound(m.bp.begin(), m.bp.end(), bp);
    const std::size_t hi = static_cast<std::size_t>(it - m.bp.begin());
    const std::size_t lo = hi - 1;
    const long long b0 = m.bp[lo];
    const long long b1 = m.bp[hi];
    const double c0 = m.cm[lo];
    const double c1 = m.cm[hi];
    const double frac = (b1 == b0) ? 0.0
                                   : static_cast<double>(bp - b0) / static_cast<double>(b1 - b0);
    return (c0 + (c1 - c0) * frac) / 100.0;
}

}  // namespace

VcfPanelResult read_vcf_panel(const std::string& vcf_path, const VcfPanelOptions& opts) {
    GzipLineReader reader(vcf_path);

    // --- optional genetic map ------------------------------------------------
    std::unordered_map<int, ChromMap> maps;
    if (!opts.map_path.empty()) maps = read_genetic_map(opts.map_path);

    // --- header: resolve EVERY sample column (>=9) --- reuse vcf_reader.cpp:608-643
    std::vector<int> sample_cols;
    std::vector<std::string> sample_ids;
    std::string line;
    bool saw_chrom_header = false;
    while (reader.next_line(line)) {
        if (line.rfind("##", 0) == 0) continue;
        if (!line.empty() && line[0] == '#') {
            const std::vector<std::string_view> h = vd::split(line, '\t');
            if (h.size() < 10) {
                throw std::runtime_error(
                    "io::read_vcf_panel: #CHROM header has no sample column in " + vcf_path);
            }
            for (std::size_t c = 9; c < h.size(); ++c) {
                sample_cols.push_back(static_cast<int>(c));
                sample_ids.emplace_back(h[c]);
            }
            saw_chrom_header = true;
            break;
        }
    }
    if (!saw_chrom_header) {
        throw std::runtime_error("io::read_vcf_panel: no #CHROM header found in " + vcf_path);
    }

    VcfPanelResult out;
    out.sample_ids = std::move(sample_ids);
    out.n_sample = out.sample_ids.size();
    const std::size_t n_sample = out.n_sample;
    const std::size_t n_hap = n_sample * 2;                     // two haploid columns / sample
    const std::size_t src_bpr = packed_bytes(n_hap);            // SNP-major record width
    const int max_col = sample_cols.empty() ? 8 : sample_cols.back();

    VcfPanelCounts counts;
    std::vector<std::uint8_t>& packed = out.tile.snp_major;     // append src_bpr bytes / kept SNP
    SnpTable& snptab = out.snptab;

    long long last_emitted_pos = -1;
    std::string last_emitted_chrom;

    // --- single streaming pass: decode+pack each kept record inline ----------
    while (reader.next_line(line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string_view> f = vd::split(line, '\t');
        if (static_cast<int>(f.size()) <= max_col) continue;   // truncated record
        ++counts.records_seen;

        const std::string_view chrom_sv = strip_chr(f[0]);
        const auto pos_opt = vd::parse_int(f[1]);
        if (!pos_opt) continue;
        const long long pos = *pos_opt;

        // In-stream POS-range filter (no tabix; stop once past the range end on the
        // matching chromosome — VCF is position-sorted within a chromosome).
        if (opts.region.active) {
            if (chrom_sv == opts.region.chrom) {
                if (pos > opts.region.end) break;               // past the range: done
                if (pos < opts.region.start) { ++counts.skipped_out_of_region; continue; }
            } else {
                ++counts.skipped_out_of_region;
                continue;
            }
        }

        const std::string_view ref = f[3];
        const std::string_view alt = f[4];
        // Biallelic filter, matching `-m2 -M2 -v snps`.
        if (alt.find(',') != std::string_view::npos) { ++counts.skipped_multiallelic; continue; }
        if (!is_snp_allele(ref) || !is_snp_allele(alt)) { ++counts.skipped_non_snp; continue; }

        // `norm -d all`: collapse a duplicate POS (keep the first-seen). VCF is
        // position-sorted so a duplicate is adjacent to the last emitted site.
        if (pos == last_emitted_pos && chrom_sv == last_emitted_chrom) {
            ++counts.skipped_dup_pos;
            continue;
        }

        // Locate GT once per record (reuse vcfdetail::format_index).
        const int gt_fi = vd::format_index(f[8], "GT");

        // Append one zero-initialized SNP-major record and fill its haplotype codes.
        const std::size_t rec_off = packed.size();
        packed.resize(rec_off + src_bpr, std::uint8_t{0});
        std::uint8_t* rec = packed.data() + rec_off;

        for (std::size_t k = 0; k < n_sample; ++k) {
            const std::string_view sample = f[static_cast<std::size_t>(sample_cols[k])];
            const std::string_view gtv = (gt_fi >= 0) ? vd::subfield(sample, gt_fi)
                                                      : std::string_view{};

            std::uint8_t h0 = kMissingCode;  // 3
            std::uint8_t h1 = kMissingCode;
            // Phased bit-split (reuse vcf_reader.cpp:836-852 SHAPE): split on '|'
            // (phased) or '/' (unphased), hap1 = allele before, hap2 = after.
            const bool phased = gtv.find('|') != std::string_view::npos;
            const char sep = phased ? '|' : '/';
            const std::vector<std::string_view> ga = vd::split(gtv, sep);
            if (ga.size() == 2) {
                const bool unphased_het = !phased && ga[0] != ga[1];
                if (unphased_het) {
                    // No defined hap order -> both haplotypes missing. Paint's own
                    // n_diploid gate does NOT catch this; the reader counts it.
                    ++counts.unphased_het_dropped;
                    // h0, h1 stay kMissingCode
                } else {
                    h0 = hap_code(ga[0]);
                    h1 = hap_code(ga[1]);
                    const bool m0 = (h0 == kMissingCode);
                    const bool m1 = (h1 == kMissingCode);
                    if (m0 != m1) ++counts.half_missing_haps;   // one hap present, one '.'
                }
            }
            // else: malformed / empty GT -> both haplotypes missing (code 3).

            if (h0 == kMissingCode) ++counts.missing_haps;
            if (h1 == kMissingCode) ++counts.missing_haps;

            // Pack the two haplotype columns 2k and 2k+1 (SNP-major fresh-pack shape,
            // geno_reader.cpp:561-573): byte_in_rec = i/4, pack_code_into_byte(byte, i, code).
            const std::size_t i0 = 2u * k;
            const std::size_t i1 = i0 + 1u;
            rec[i0 / static_cast<std::size_t>(kCodesPerByte)] = pack_code_into_byte(
                rec[i0 / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(i0), h0);
            rec[i1 / static_cast<std::size_t>(kCodesPerByte)] = pack_code_into_byte(
                rec[i1 / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(i1), h1);
        }

        // SNP metadata (inline .snp): id, chrom, genpos (Morgans), physpos, ref/alt.
        const int chrom_int = chrom_to_int(chrom_sv);
        double genpos = 0.0;
        if (!maps.empty()) {
            const auto mit = maps.find(chrom_int);
            if (mit != maps.end()) genpos = interp_morgans(mit->second, pos);
        }
        snptab.id.emplace_back(f[2]);
        snptab.chrom.push_back(chrom_int);
        snptab.genpos_morgans.push_back(genpos);
        snptab.physpos.push_back(static_cast<double>(pos));
        snptab.ref.push_back(ref[0]);
        snptab.alt.push_back(alt[0]);

        ++counts.emitted_sites;
        last_emitted_pos = pos;
        last_emitted_chrom.assign(chrom_sv);
    }

    // --- finalize the tile ---------------------------------------------------
    snptab.count = static_cast<std::size_t>(counts.emitted_sites);
    counts.diploid_calls = counts.emitted_sites * static_cast<long long>(n_sample);

    out.tile.src_bytes_per_record = src_bpr;
    out.tile.n_snp = static_cast<std::size_t>(counts.emitted_sites);
    out.tile.n_individuals = n_hap;
    out.tile.sel_rows.resize(n_hap);
    for (std::size_t i = 0; i < n_hap; ++i) out.tile.sel_rows[i] = i;   // identity: all haps
    out.tile.pop_offsets = {0, n_hap};                                  // one "PANEL" pop
    out.tile.pop_labels = {std::string("PANEL")};

    // --- unphased-drop guard: fail loud above the threshold ------------------
    if (counts.diploid_calls > 0) {
        const double frac =
            static_cast<double>(counts.unphased_het_dropped) /
            static_cast<double>(counts.diploid_calls);
        if (frac > opts.unphased_max) {
            std::ostringstream oss;
            oss << "io::read_vcf_panel: unphased-het fraction " << frac << " ("
                << counts.unphased_het_dropped << " of " << counts.diploid_calls
                << " diploid GT calls) exceeds --unphased-max " << opts.unphased_max
                << " — the panel is not sufficiently phased for haplotype painting (unphased "
                   "GTs were dropped to missing and would silently pass paint's n_diploid gate)";
            throw std::runtime_error(oss.str());
        }
    }

    out.counts = counts;
    return out;
}

void dump_hap_codes(const SnpMajorTile& tile, const SnpTable& snptab, std::ostream& os) {
    const std::size_t n_snp = tile.n_snp;
    const std::size_t n_ind = tile.n_individuals;
    const std::size_t bpr = tile.src_bytes_per_record;
    for (std::size_t s = 0; s < n_snp; ++s) {
        const std::uint8_t* rec = tile.snp_major.data() + s * bpr;
        // CHROM<TAB>POS prefix (matches the bcftools oracle's %CHROM %POS).
        const int chrom = (s < snptab.chrom.size()) ? snptab.chrom[s] : kFirstOtherChromCode;
        const long long pos =
            (s < snptab.physpos.size()) ? static_cast<long long>(snptab.physpos[s]) : 0;
        os << chrom << '\t' << pos;
        for (std::size_t g = 0; g < n_ind; ++g) {
            const std::size_t src_row = tile.sel_rows.empty() ? g : tile.sel_rows[g];
            const std::uint8_t code = code_in_byte(
                rec[src_row / static_cast<std::size_t>(kCodesPerByte)], static_cast<int>(src_row));
            os << '\t' << static_cast<int>(code);
        }
        os << '\n';
    }
}

}  // namespace steppe::io
