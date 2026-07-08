// src/io/beagle_reader.cpp
//
// Implementation of the thin beagle-GL reader. Streams lines via GzipLineReader
// (transparent gzip/plain), tokenizes on whitespace, and appends site-major
// triplets into a LikelihoodTile with the g = copies-of-A1 reversal documented in
// beagle_reader.hpp.
#include "io/beagle_reader.hpp"

#include <array>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "io/gl_normalize.hpp"
#include "io/gzip_line_reader.hpp"

namespace steppe::io {

namespace {

// Split a line into whitespace-delimited views (no allocation of substrings).
void split_ws(const std::string& line, std::vector<std::string_view>& out) {
    out.clear();
    const char* p = line.data();
    const char* end = p + line.size();
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;
        if (p >= end) break;
        const char* start = p;
        while (p < end && *p != ' ' && *p != '\t') ++p;
        out.emplace_back(start, static_cast<std::size_t>(p - start));
    }
}

// Parse a whitespace field to double; throws on an unparseable token.
[[nodiscard]] double to_double(std::string_view tok, long row, std::size_t col) {
    // std::from_chars<double> is not universally available on the toolchain used
    // here, so fall back to strtod over a small null-terminated copy.
    char buf[64];
    const std::size_t n = tok.size() < sizeof(buf) - 1 ? tok.size() : sizeof(buf) - 1;
    for (std::size_t i = 0; i < n; ++i) buf[i] = tok[i];
    buf[n] = '\0';
    char* endp = nullptr;
    const double v = std::strtod(buf, &endp);
    if (endp == buf) {
        throw std::runtime_error("beagle_reader: unparseable GL value '" + std::string(tok) +
                                 "' at data row " + std::to_string(row) + ", field " +
                                 std::to_string(col));
    }
    return v;
}

// A beagle marker is conventionally "chrom_pos" (e.g. "1_752566"); split it so the
// tile site carries chrom/pos38 when present, else leave them 0 (PCAngsd is
// self-contained on the markers — chrom/pos are not load-bearing).
void parse_marker(std::string_view marker, LikelihoodSite& site) {
    site.rsid.assign(marker);
    const std::size_t us = marker.rfind('_');
    if (us == std::string_view::npos || us == 0 || us + 1 >= marker.size()) return;
    const std::string_view chrom_sv = marker.substr(0, us);
    const std::string_view pos_sv = marker.substr(us + 1);
    int chrom = 0;
    long long pos = 0;
    const auto rc = std::from_chars(chrom_sv.data(), chrom_sv.data() + chrom_sv.size(), chrom);
    const auto rp = std::from_chars(pos_sv.data(), pos_sv.data() + pos_sv.size(), pos);
    if (rc.ec == std::errc() && rc.ptr == chrom_sv.data() + chrom_sv.size() &&
        rp.ec == std::errc() && rp.ptr == pos_sv.data() + pos_sv.size()) {
        site.chrom = chrom;
        site.pos38 = pos;
        site.pos37 = pos;
    }
}

}  // namespace

BeagleReadResult read_beagle_gl(const std::string& path) {
    GzipLineReader reader(path);
    BeagleReadResult res;
    LikelihoodTile& tile = res.tile;
    tile.field = GlField::GP;      // beagle triplets are linear posteriors (renormalized)
    tile.native_order = false;     // stored in the A1-copy g-axis (reversed), phased_gt empty

    std::string line;
    std::vector<std::string_view> toks;

    // --- header ---------------------------------------------------------------
    if (!reader.next_line(line)) {
        throw std::runtime_error("beagle_reader: empty file (no header): " + path);
    }
    split_ws(line, toks);
    const std::size_t ncol = toks.size();
    if (ncol < 6 || (ncol - 3) % 3 != 0) {
        throw std::runtime_error(
            "beagle_reader: header has " + std::to_string(ncol) +
            " columns; expected 'marker allele1 allele2' + 3 GL columns per individual "
            "(cols-3 must be a positive multiple of 3): " + path);
    }
    const int n_sample = static_cast<int>((ncol - 3) / 3);
    tile.n_sample = static_cast<std::size_t>(n_sample);
    tile.sample_ids.resize(static_cast<std::size_t>(n_sample));
    for (int s = 0; s < n_sample; ++s) {
        // The first of each repeated triple is the id; synthesize Ind{s} if empty.
        std::string_view id = toks[static_cast<std::size_t>(3 + s * 3)];
        tile.sample_ids[static_cast<std::size_t>(s)] =
            id.empty() ? ("Ind" + std::to_string(s)) : std::string(id);
    }

    // --- data rows ------------------------------------------------------------
    const std::size_t fields_per_row = 3 + static_cast<std::size_t>(n_sample) * 3;
    long row = 0;
    while (reader.next_line(line)) {
        if (line.empty()) continue;
        split_ws(line, toks);
        if (toks.empty()) continue;
        if (toks.size() != fields_per_row) {
            throw std::runtime_error(
                "beagle_reader: data row " + std::to_string(row) + " has " +
                std::to_string(toks.size()) + " fields; expected " +
                std::to_string(fields_per_row) + " (3 + 3*" + std::to_string(n_sample) +
                " individuals): " + path);
        }

        LikelihoodSite site;
        parse_marker(toks[0], site);
        const std::string_view a1 = toks[1];
        const std::string_view a2 = toks[2];
        site.a1 = a1.empty() ? 'N' : a1[0];
        site.a2 = a2.empty() ? 'N' : a2[0];
        tile.sites.push_back(std::move(site));

        const std::size_t base = tile.l.size();
        tile.l.resize(base + static_cast<std::size_t>(n_sample) * 3);
        tile.present.resize(tile.present.size() + static_cast<std::size_t>(n_sample));
        const std::size_t mask_base = tile.present.size() - static_cast<std::size_t>(n_sample);

        for (int s = 0; s < n_sample; ++s) {
            const std::size_t c = 3 + static_cast<std::size_t>(s) * 3;
            const double g_a1a1 = to_double(toks[c + 0], row, c + 0);
            const double g_het = to_double(toks[c + 1], row, c + 1);
            const double g_a2a2 = to_double(toks[c + 2], row, c + 2);
            const std::array<double, 3> norm = glnorm::normalize_gp(g_a1a1, g_het, g_a2a2);
            const bool degenerate =
                !(g_a1a1 + g_het + g_a2a2 > 0.0);  // matches normalize_gp's uninformative guard
            // Reverse into the g = copies-of-A1 axis: g=0 -> A2A2, g=1 -> het, g=2 -> A1A1.
            const std::size_t b = base + static_cast<std::size_t>(s) * 3;
            tile.l[b + 0] = norm[2];
            tile.l[b + 1] = norm[1];
            tile.l[b + 2] = norm[0];
            tile.present[mask_base + static_cast<std::size_t>(s)] = degenerate ? 0u : 1u;
        }
        ++row;
    }

    tile.n_site = static_cast<std::size_t>(row);
    res.n_site = row;
    res.n_sample = n_sample;
    return res;
}

}  // namespace steppe::io
