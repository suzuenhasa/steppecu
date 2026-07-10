// src/io/filter/ascertainment.hpp
//
// The same-ascertainment guard. A SNP-keyed external input (a --keep-snps / prune.in list, or
// explicit include ids) combined with the target only stays statistically sound when the two
// were ascertained the same way; silently intersecting a WGS/GSA target with a 1240K panel
// (or vice-versa) skews the site set and biases f4/PCA/admixture — the consumer-DNA failure.
// This leaf classifies a SNP-id set by its id namespace and decides whether the external list
// is drawn from the same panel as the target (by containment): a genuine restrict-to-subset
// list is ~fully present in the target; a cross-panel list shares few ids. Host-only io-leaf:
// C++20, no CUDA, no core/device dependency.
#ifndef STEPPE_IO_FILTER_ASCERTAINMENT_HPP
#define STEPPE_IO_FILTER_ASCERTAINMENT_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace steppe::io::filter {

// The dominant id convention of a panel — a coarse ascertainment fingerprint.
enum class IdNamespace { RsId, ChrPos, Probe, Other };

// Per-set namespace fingerprint (dominant convention + the raw fractions + count).
struct AscertainmentTag {
    std::size_t n = 0;
    IdNamespace dominant = IdNamespace::Other;
    double rs_frac = 0.0;
    double chrpos_frac = 0.0;
    double probe_frac = 0.0;
};

// Classify one SNP id: rsNNN -> RsId, chr:pos style (contains ':') -> ChrPos, array/probe ids
// (23andMe "iNNN", "GSA-*", "AX-*", "affx*", "exm*", "seq-*", "kgp*") -> Probe, else Other.
[[nodiscard]] IdNamespace classify_snp_id(const std::string& id) noexcept;

// Plurality namespace over an id list, plus the per-class fractions.
[[nodiscard]] AscertainmentTag classify_ascertainment(const std::vector<std::string>& ids);

// If fewer than this fraction of the external list's ids exist in the target, the list is
// drawn from a different panel (an ascertainment mix) — refuse unless overridden. A legitimate
// restrict-to-subset list (an LD-prune output on this target, a curated panel subset) is ~1.0.
inline constexpr double kAscertainmentContainmentFloor = 0.5;

// The guard verdict: whether the external list mixes ascertainment with the target, the
// containment evidence, both fingerprints, and a human explanation when mixed.
struct AscertainmentVerdict {
    bool mixed = false;
    std::size_t external_present = 0;   // external ids found in the target .snp
    std::size_t external_total = 0;     // distinct external ids supplied
    double present_frac = 1.0;          // external_present / external_total (1.0 if none)
    AscertainmentTag target_tag;
    AscertainmentTag external_tag;
    std::string reason;                 // populated when mixed == true
};

// Compare the external SNP-keyed list against the target .snp id set. mixed == true when the
// list is largely absent from the target (present_frac < kAscertainmentContainmentFloor).
[[nodiscard]] AscertainmentVerdict check_same_ascertainment(
    const std::vector<std::string>& target_ids, const std::vector<std::string>& external_ids);

}  // namespace steppe::io::filter

#endif  // STEPPE_IO_FILTER_ASCERTAINMENT_HPP
