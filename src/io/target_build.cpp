// Reference: docs/reference/src_io_target_build.cpp.md
// src/io/target_build.cpp
//
// target_build implementation — see the header. Two passes over the panel .snp:
//   pass 1  parse + filter (autosome, rsID) + palindrome flag + per-rsID count
//   pass 2  drop dups / no-lifts, fetch ref38 natively, assemble TargetSites
// then the shared build_chrom_index() (identical to read_target_sites' index).
// Every counter/branch is tagged to the oracle.py line it reproduces.
#include "io/target_build.hpp"

#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace steppe::io {

namespace {

[[nodiscard]] char up(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

[[nodiscard]] std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream ls(line);
    std::string tok;
    while (ls >> tok) out.push_back(tok);
    return out;
}

// Strict integer parse of a whole token; false on any non-digit / overflow.
[[nodiscard]] bool parse_ll(const std::string& tok, long long& out) {
    if (tok.empty()) return false;
    const char* b = tok.data();
    const char* e = b + tok.size();
    const auto [ptr, ec] = std::from_chars(b, e, out);
    return ec == std::errc{} && ptr == e;
}

// One surviving autosomal-rs panel row, before the lift/dedup drop.
struct PanelRow {
    std::string rsid;
    int chrom = 0;
    long long pos37 = 0;
    char a1 = 'N';
    char a2 = 'N';
    bool pal = false;
};

// Parse the orchestrated rsID<TAB>pos38 lift map. A leading header row (first
// token "rsID"/"rsid", or a non-numeric pos field) is skipped (critic fix #5:
// `cut -f1,4 nikki_target_sites.tsv` carries the header line).
[[nodiscard]] std::unordered_map<std::string, long long> read_lift_map(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("io::build_target_sites: cannot open lift map: " + path);
    }
    std::unordered_map<std::string, long long> m;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> f = split_ws(line);
        if (f.size() < 2) continue;
        long long pos = 0;
        if (!parse_ll(f[1], pos)) continue;  // header / garbage pos -> skip
        m[f[0]] = pos;
    }
    return m;
}

}  // namespace

TargetSites build_target_sites(const std::string& panel_snp, const std::string& lift_map,
                               FaidxReader& fasta, const TargetBuildOptions& opts,
                               TargetBuildCounts& counts) {
    counts = TargetBuildCounts{};

    // --- pass 1: parse + filter the panel, count per-rsID multiplicity --------
    std::ifstream in(panel_snp);
    if (!in) {
        throw std::runtime_error("io::build_target_sites: cannot open panel .snp: " + panel_snp);
    }
    std::vector<PanelRow> rows;
    std::unordered_map<std::string, int> rs_count;
    std::string line;
    while (std::getline(in, line)) {
        const std::vector<std::string> f = split_ws(line);
        if (f.size() < 6) continue;  // oracle load_panel:47-49 (short rows NOT counted)
        ++counts.panel_total;

        // M1: autosomes only (chrom 1..22). A non-numeric / out-of-range chrom
        // (X/Y/MT/...) is dropped and does NOT count as autosomal (oracle:51).
        long long chrom_ll = 0;
        if (!opts.autosomes_only) {
            throw std::runtime_error("io::build_target_sites: only autosomes_only=true is supported");
        }
        if (!parse_ll(f[1], chrom_ll) || chrom_ll < 1 || chrom_ll > 22) continue;
        ++counts.panel_autosomal;

        const std::string& rsid = f[0];
        if (rsid.rfind("rs", 0) != 0) {  // Fork #1: rsID join only (oracle:54)
            ++counts.panel_non_rsid;
            continue;
        }
        ++rs_count[rsid];  // rs_count over autosomal-rs rows ONLY (critic fix #1)

        long long pos37 = 0;
        if (!parse_ll(f[3], pos37)) {
            throw std::runtime_error("io::build_target_sites: non-numeric physpos '" + f[3] +
                                     "' for " + rsid);
        }
        PanelRow r;
        r.rsid = rsid;
        r.chrom = static_cast<int>(chrom_ll);
        r.pos37 = pos37;
        r.a1 = up(f[4][0]);
        r.a2 = up(f[5][0]);
        r.pal = is_palindrome(r.a1, r.a2);
        if (r.pal) ++counts.panel_palindromic;  // PRE-dedup (oracle:58, critic fix #7.1)
        rows.push_back(std::move(r));
    }

    // Strict 1:1 de-dup set: distinct rsIDs with multiplicity > 1 (oracle:63).
    std::unordered_set<std::string> dup;
    for (const auto& [rsid, n] : rs_count) {
        if (n > 1) dup.insert(rsid);
    }
    counts.panel_dup_rsids = static_cast<long long>(dup.size());  // len(dup) (critic fix #3)

    // --- pass 2: lift join + native ref38 fetch + assemble --------------------
    const std::unordered_map<std::string, long long> lift = read_lift_map(lift_map);

    TargetSites ts;
    ts.sites.reserve(rows.size());
    for (const PanelRow& r : rows) {
        if (dup.count(r.rsid) != 0) {  // H3 dup drop (oracle lift_panel:79-82)
            ++counts.lift_dropped_dup;  // ROWS dropped (n_dropdup); per-occurrence
            continue;
        }
        const auto lit = lift.find(r.rsid);
        if (lit == lift.end()) {  // no lift -> dropped (oracle:86-90)
            ++counts.lift_no_lift;
            continue;
        }
        ++counts.lift_ok;
        const long long pos38 = lit->second;

        TargetSite s;
        s.rsid = r.rsid;
        s.chrom = r.chrom;
        s.pos37 = r.pos37;
        s.pos38 = pos38;
        s.a1 = r.a1;
        s.a2 = r.a2;
        // Native ref38 for EVERY lifted site, palindrome or not (critic fix #2).
        s.ref38 = fasta.base_at(std::to_string(r.chrom), pos38);
        s.palindrome = r.pal;
        ts.sites.push_back(std::move(s));
    }
    counts.emitted = static_cast<long long>(ts.sites.size());

    // Shared per-chrom index (identical to read_target_sites' construction).
    build_chrom_index(ts);
    return ts;
}

void write_target_table(const std::string& path, const TargetSites& ts) {
    std::ofstream o(path, std::ios::trunc);
    if (!o) {
        throw std::runtime_error("io::write_target_table: cannot open output: " + path);
    }
    o << "rsID\tchrom\tpos37\tpos38\tA1\tA2\tref38\n";
    for (const TargetSite& s : ts.sites) {
        o << s.rsid << '\t' << s.chrom << '\t' << s.pos37 << '\t' << s.pos38 << '\t' << s.a1 << '\t'
          << s.a2 << '\t' << s.ref38 << '\n';
    }
    if (!o) {
        throw std::runtime_error("io::write_target_table: failed writing: " + path);
    }
}

}  // namespace steppe::io
