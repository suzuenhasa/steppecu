// src/io/target_sites.cpp
//
// Reference: docs/reference/src_io_target_sites.cpp.md
//
// TargetSites parser — see the header. Builds the per-chrom sorted pos38 index
// (non-palindromic sites, duplicates kept, last-wins slot map) to mirror the
// Stage-0 oracle exactly.
#include "io/target_sites.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace steppe::io {

namespace {

[[nodiscard]] char up(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

}  // namespace

bool is_palindrome(char a1, char a2) {
    const char x = up(a1);
    const char y = up(a2);
    return (x == 'A' && y == 'T') || (x == 'T' && y == 'A') ||
           (x == 'C' && y == 'G') || (x == 'G' && y == 'C');
}

void build_chrom_index(TargetSites& ts) {
    ts.by_chrom.clear();
    for (const TargetSite& s : ts.sites) {
        if (s.palindrome) continue;
        ts.by_chrom[s.chrom].pos.push_back(s.pos38);
    }
    for (auto& [chrom, ci] : ts.by_chrom) {
        (void)chrom;
        std::sort(ci.pos.begin(), ci.pos.end());
        ci.slot.reserve(ci.pos.size() * 2);
        for (std::size_t i = 0; i < ci.pos.size(); ++i) {
            ci.slot[ci.pos[i]] = i;  // duplicate pos38 -> highest index (last-wins)
        }
    }
}

TargetSites read_target_sites(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::read_target_sites: cannot open target table: " + path);
    }

    TargetSites ts;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::vector<std::string> f;
        std::string tok;
        while (ls >> tok) f.push_back(tok);
        if (f.empty()) continue;
        // Skip a header line (first column literally "rsID" / "rsid").
        if (f[0] == "rsID" || f[0] == "rsid") continue;

        TargetSite s;
        try {
            if (f.size() >= 7) {
                // rsID chrom pos37 pos38 A1 A2 ref38
                s.rsid = f[0];
                s.chrom = std::stoi(f[1]);
                s.pos37 = std::stoll(f[2]);
                s.pos38 = std::stoll(f[3]);
                s.a1 = up(f[4][0]);
                s.a2 = up(f[5][0]);
                s.ref38 = f[6].empty() ? '.' : up(f[6][0]);
            } else if (f.size() == 6) {
                // rsID chrom pos38 A1 A2 ref38
                s.rsid = f[0];
                s.chrom = std::stoi(f[1]);
                s.pos38 = std::stoll(f[2]);
                s.a1 = up(f[3][0]);
                s.a2 = up(f[4][0]);
                s.ref38 = f[5].empty() ? '.' : up(f[5][0]);
            } else {
                throw std::runtime_error(
                    "io::read_target_sites: malformed record (expected 6 or 7 fields) at line " +
                    std::to_string(line_no));
            }
        } catch (const std::invalid_argument&) {
            throw std::runtime_error(
                "io::read_target_sites: non-numeric chrom/pos at line " + std::to_string(line_no));
        }
        s.palindrome = is_palindrome(s.a1, s.a2);
        ts.sites.push_back(std::move(s));
    }

    // Build the per-chrom sorted pos38 index over the NON-palindromic sites, then
    // the last-wins slot map (matching oracle sorted() + enumerate()).
    build_chrom_index(ts);
    return ts;
}

}  // namespace steppe::io
