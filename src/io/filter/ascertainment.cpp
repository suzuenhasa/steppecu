// src/io/filter/ascertainment.cpp
//
// Implements the same-ascertainment guard: id-namespace classification and the
// target-vs-external containment decision. Host-pure io-leaf.
#include "io/filter/ascertainment.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>

namespace steppe::io::filter {

namespace {

[[nodiscard]] bool all_digits(const std::string& s, std::size_t from) noexcept {
    if (from >= s.size()) return false;
    for (std::size_t i = from; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

[[nodiscard]] bool starts_with_ci(const std::string& s, const char* pfx) noexcept {
    std::size_t i = 0;
    for (; pfx[i] != '\0'; ++i) {
        if (i >= s.size()) return false;
        const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(pfx[i])));
        if (a != b) return false;
    }
    return true;
}

}  // namespace

IdNamespace classify_snp_id(const std::string& id) noexcept {
    if (id.empty()) return IdNamespace::Other;
    // rsNNNN (a bare dbSNP rsID).
    if (starts_with_ci(id, "rs") && all_digits(id, 2)) return IdNamespace::RsId;
    // 23andMe internal probe ids: 'i' followed by digits.
    if ((id[0] == 'i' || id[0] == 'I') && all_digits(id, 1)) return IdNamespace::Probe;
    // Vendor/array probe namespaces.
    if (starts_with_ci(id, "gsa-") || starts_with_ci(id, "ax-") ||
        starts_with_ci(id, "affx") || starts_with_ci(id, "exm") ||
        starts_with_ci(id, "seq-") || starts_with_ci(id, "kgp")) {
        return IdNamespace::Probe;
    }
    // chr:pos style (a WGS/novel-site id convention): contains a ':' separator.
    if (id.find(':') != std::string::npos) return IdNamespace::ChrPos;
    return IdNamespace::Other;
}

AscertainmentTag classify_ascertainment(const std::vector<std::string>& ids) {
    AscertainmentTag tag;
    tag.n = ids.size();
    if (ids.empty()) return tag;
    std::size_t rs = 0, cp = 0, pr = 0;
    for (const std::string& id : ids) {
        switch (classify_snp_id(id)) {
            case IdNamespace::RsId:   ++rs; break;
            case IdNamespace::ChrPos: ++cp; break;
            case IdNamespace::Probe:  ++pr; break;
            case IdNamespace::Other:  break;
        }
    }
    const double nd = static_cast<double>(ids.size());
    tag.rs_frac = static_cast<double>(rs) / nd;
    tag.chrpos_frac = static_cast<double>(cp) / nd;
    tag.probe_frac = static_cast<double>(pr) / nd;
    // Plurality; ties resolve rs > chrpos > probe > other (a stable, documented order).
    const std::size_t other = ids.size() - rs - cp - pr;
    std::size_t best = rs;
    tag.dominant = IdNamespace::RsId;
    if (cp > best) { best = cp; tag.dominant = IdNamespace::ChrPos; }
    if (pr > best) { best = pr; tag.dominant = IdNamespace::Probe; }
    if (other > best) { tag.dominant = IdNamespace::Other; }
    return tag;
}

namespace {

const char* ns_name(IdNamespace ns) noexcept {
    switch (ns) {
        case IdNamespace::RsId:   return "rsID";
        case IdNamespace::ChrPos: return "chr:pos";
        case IdNamespace::Probe:  return "array-probe";
        case IdNamespace::Other:  return "other";
    }
    return "other";
}

}  // namespace

AscertainmentVerdict check_same_ascertainment(const std::vector<std::string>& target_ids,
                                              const std::vector<std::string>& external_ids) {
    AscertainmentVerdict v;
    v.target_tag = classify_ascertainment(target_ids);
    v.external_tag = classify_ascertainment(external_ids);

    // Distinct external ids present in the target .snp.
    std::unordered_set<std::string> target_set(target_ids.begin(), target_ids.end());
    std::unordered_set<std::string> ext_distinct(external_ids.begin(), external_ids.end());
    v.external_total = ext_distinct.size();
    std::size_t present = 0;
    for (const std::string& id : ext_distinct) {
        if (target_set.find(id) != target_set.end()) ++present;
    }
    v.external_present = present;
    v.present_frac = (v.external_total == 0)
                         ? 1.0
                         : static_cast<double>(present) / static_cast<double>(v.external_total);

    if (v.external_total > 0 && v.present_frac < kAscertainmentContainmentFloor) {
        v.mixed = true;
        v.reason =
            "the external SNP list (" + std::to_string(v.external_total) + " ids, dominant " +
            ns_name(v.external_tag.dominant) + ") shares only " + std::to_string(present) +
            " ids with the target .snp (" + std::to_string(target_ids.size()) + " sites, dominant " +
            ns_name(v.target_tag.dominant) + ") — the list looks drawn from a DIFFERENT panel "
            "than the target, so the intersection is ascertainment-skewed (the consumer-DNA "
            "f4-bias failure). Pass --allow-mixed-ascertainment to proceed anyway.";
    }
    return v;
}

}  // namespace steppe::io::filter
