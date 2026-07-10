// tests/unit/test_filters.cpp
//
// Host-only unit test of the SHARED M2 filter predicates + the include/exclude
// membership + the per-sample --mind resolution (architecture.md §13 "Unit tests";
// ROADMAP M2, §5). Pure C++ TU, NO GPU, NO real data: it exercises the host-pure
// predicate LOGIC in src/io/filter/filter_decision.hpp (and SnpMembership /
// run_mind_prepass) on hand-built inputs — control flow / boundary logic, not a
// precision or accuracy claim, so synthetic inputs are allowed (ROADMAP §0: the
// no-synthetic rule is for precision claims; feeding is_strand_ambiguous the pair
// 'G','T' is logic). The numerical-accuracy gate is the real-AADR oracle test
// (tests/reference/test_filter_oracle.cu).
//
// B19 verdict gate (cleanup include_exclude 1.1): the prune.in file branch of
// read_snp_id_list / SnpMembership had ZERO test coverage. These add a temp-file
// round-trip (first-token-per-line, blank-skip, CRLF '\r' stripped, trailing
// columns tolerated, append semantics), a throw-on-missing-file check, and — the
// fix itself — a throw-on-DIRECTORY check: a directory opens but read-fails on
// POSIX, which pre-fix yielded a silently-empty keep-set (fail-open) and now
// fail-fasts (architecture.md §2). The membership test pins prune.in ∪ include
// with exclude overriding a prune-supplied id, and that a directory prune_in_path
// propagates the throw out of the SnpMembership constructor.
//
// Dual harness (identical to tests/unit/test_f2.cpp): with -DSTEPPE_TEST_WITH_GTEST
// it uses GoogleTest; otherwise a self-checking main() returning non-zero on the
// first failure — all CTest needs to gate. No CUDA here.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "io/filter/ascertainment.hpp"     // classify_snp_id / check_same_ascertainment
#include "io/filter/filter_decision.hpp"   // the shared predicates (single source)
#include "io/filter/snp_filter.hpp"        // derive_per_snp_summary / build_snp_keep_mask / snp_keep_decision / filter_is_active
#include "io/filter/include_exclude.hpp"   // SnpMembership
#include "io/filter/mind_prepass.hpp"      // run_mind_prepass
#include "io/snp_reader.hpp"               // SnpTable
#include "io/eigenstrat_format.hpp"        // code_in_byte (to hand-pack a --mind input)
#include "steppe/config.hpp"               // FilterConfig, kAutosomeChromMin/Max

namespace {

using namespace steppe::io::filter;
using steppe::FilterConfig;
using steppe::io::SnpTable;

constexpr double kEps = 1e-12;
[[nodiscard]] bool close(double a, double b, double eps = kEps) {
    return std::fabs(a - b) <= eps * std::fmax(std::fabs(b), 1.0);
}

// ---- temp-file helpers for the prune.in file-branch tests --------------------
// Mirror the RAII temp-file convention in tests/unit/test_snp_reader.cpp: a
// unique path per fixture (so parallel ctest invocations do not collide) and a
// scope-bound cleanup. Used only by the read_snp_id_list / prune.in tests below.
[[nodiscard]] std::filesystem::path temp_prune(const char* tag) {
    static int counter = 0;
    auto p = std::filesystem::temp_directory_path();
    p /= ("steppe_prune_in_test_" + std::string(tag) + "_" +
          std::to_string(++counter) + ".in");
    return p;
}

struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const char* tag) : path(temp_prune(tag)) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

void write_text(const std::filesystem::path& path, const std::string& body) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(body.data(), static_cast<std::streamsize>(body.size()));
    out.close();
}

// Whether `fn` throws std::runtime_error (the documented prune.in failure mode).
[[nodiscard]] bool throws_runtime_error(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;  // a different exception type is NOT the documented contract
    }
    return false;
}

// Whether `fn` throws std::invalid_argument (the snp_filter B20 fail-fast contract:
// bad ploidy / short pop_individuals / null q-n / short SnpTable). A DIFFERENT
// exception type is NOT the contract.
[[nodiscard]] bool throws_invalid_argument(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// ---- MAF boundary: PINNED >= side --------------------------------------------
// snp_passes_maf keeps iff folded MAF >= maf_min (inclusive). At maf_min == 0
// everything passes (no-op default). At the boundary, MAF exactly == threshold is
// KEPT; just below is dropped.
[[nodiscard]] bool test_maf_boundary() {
    // folded_maf folds the pooled ref freq.
    if (!close(folded_maf(0.3), 0.3)) return false;     // min(0.3, 0.7)
    if (!close(folded_maf(0.8), 0.2)) return false;     // min(0.8, 0.2)
    if (!close(folded_maf(0.5), 0.5)) return false;     // symmetric peak
    // No-op at maf_min == 0: every MAF (incl. 0) passes.
    if (!snp_passes_maf(0.0, 0.0)) return false;
    // Inclusive >= boundary: exactly-at-threshold is KEPT, just-below is dropped.
    if (!snp_passes_maf(0.05, 0.05)) return false;      // == threshold -> keep
    if (snp_passes_maf(0.0499999, 0.05)) return false;  // < threshold -> drop
    if (!snp_passes_maf(0.06, 0.05)) return false;      // > threshold -> keep
    return true;
}

// ---- geno boundary: PINNED <= side -------------------------------------------
// snp_passes_geno keeps iff per-SNP missing frac <= geno_max_missing (inclusive).
// geno_max_missing == 1 keeps everything (no-op default).
[[nodiscard]] bool test_geno_boundary() {
    if (!snp_passes_geno(1.0, 1.0)) return false;       // no-op default: all kept
    if (!snp_passes_geno(0.1, 0.1)) return false;       // == threshold -> keep
    if (snp_passes_geno(0.1000001, 0.1)) return false;  // > threshold -> drop
    if (!snp_passes_geno(0.05, 0.1)) return false;      // < threshold -> keep
    return true;
}

// ---- mind boundary: PINNED <= side -------------------------------------------
[[nodiscard]] bool test_mind_boundary() {
    if (!sample_passes_mind(1.0, 1.0)) return false;        // no-op default
    if (!sample_passes_mind(0.2, 0.2)) return false;        // == threshold -> keep
    if (sample_passes_mind(0.2000001, 0.2)) return false;   // > threshold -> drop
    if (!sample_passes_mind(0.0, 0.2)) return false;        // perfect sample kept
    return true;
}

// ---- monomorphic --------------------------------------------------------------
[[nodiscard]] bool test_monomorphic() {
    if (!is_monomorphic(0.0)) return false;   // folded MAF 0 -> monomorphic
    if (!is_monomorphic(1.0)) return false;   // fixed for the other allele -> MAF 0
    if (is_monomorphic(0.5)) return false;    // maximally polymorphic
    if (is_monomorphic(0.01)) return false;   // a single minor copy is NOT monomorphic
    return true;
}

// ---- strand-ambiguous: GENETICS-CORRECT palindrome class (A/T, C/G) ----------
// The genetics definition of strand-ambiguous = self-complementary pairs A/T and
// C/G. We PIN that: AT/TA/CG/GC are ambiguous; the transition/transversion pairs
// (AG/GA/CT/TC and GT/TG/AC/CA) are NOT self-complementary. (The brief's "GT/AC
// dropped, GA/CT kept" example describes the ts/tv split — see filter_decision.hpp
// CONVENTION FLAG and the is_transversion test below.)
[[nodiscard]] bool test_strand_ambiguous() {
    // The four self-complementary (ambiguous) ordered pairs.
    if (!is_strand_ambiguous('A', 'T')) return false;
    if (!is_strand_ambiguous('T', 'A')) return false;
    if (!is_strand_ambiguous('C', 'G')) return false;
    if (!is_strand_ambiguous('G', 'C')) return false;
    // The brief's example pairs GT and AC are NOT self-complementary (they are
    // transversions) -> not ambiguous by the genetics-correct predicate.
    if (is_strand_ambiguous('G', 'T')) return false;
    if (is_strand_ambiguous('A', 'C')) return false;
    // Transitions GA/CT are not ambiguous either.
    if (is_strand_ambiguous('G', 'A')) return false;
    if (is_strand_ambiguous('C', 'T')) return false;
    // Case-insensitive; non-ACGT -> not ambiguous.
    if (!is_strand_ambiguous('a', 't')) return false;
    if (is_strand_ambiguous('A', 'N')) return false;
    return true;
}

// ---- multiallelic / non-clean ------------------------------------------------
[[nodiscard]] bool test_multiallelic() {
    if (!is_multiallelic('A', 'N')) return false;   // non-ACGT
    if (!is_multiallelic('0', 'C')) return false;   // non-ACGT code
    if (!is_multiallelic('A', 'A')) return false;   // ref == alt
    if (is_multiallelic('A', 'G')) return false;    // clean biallelic
    if (is_multiallelic('C', 'T')) return false;
    return true;
}

// ---- transition / transversion (the brief's GT/AC vs GA/CT split) -------------
// In the real AADR HO panel the only allele pairs are transitions (GA/AG/CT/TC)
// and transversions (GT/TG/CA/AC). PIN both classifications.
[[nodiscard]] bool test_transition_transversion() {
    // Transitions: purine<->purine (A<->G) or pyrimidine<->pyrimidine (C<->T).
    if (!is_transition('G', 'A') || !is_transition('A', 'G')) return false;
    if (!is_transition('C', 'T') || !is_transition('T', 'C')) return false;
    if (is_transversion('G', 'A') || is_transversion('C', 'T')) return false;
    // Transversions: purine<->pyrimidine (the brief's GT/AC class).
    if (!is_transversion('G', 'T') || !is_transversion('T', 'G')) return false;
    if (!is_transversion('A', 'C') || !is_transversion('C', 'A')) return false;
    if (is_transition('G', 'T') || is_transition('A', 'C')) return false;
    // ts and tv are mutually exclusive over clean biallelic ACGT pairs.
    if (is_transition('A', 'C') && is_transversion('A', 'C')) return false;
    // Brief's literal expectation: transversions_only keeps GT/AC, drops GA/CT.
    if (!is_transversion('G', 'T')) return false;  // GT kept under transversions_only
    if (is_transversion('G', 'A')) return false;   // GA dropped under transversions_only
    return true;
}

// ---- is_autosome (chr 1..22, AT2 auto_only) ----------------------------------
[[nodiscard]] bool test_is_autosome() {
    if (!is_autosome(1) || !is_autosome(22)) return false;     // boundaries kept
    if (!is_autosome(steppe::kAutosomeChromMin)) return false;
    if (!is_autosome(steppe::kAutosomeChromMax)) return false;
    if (is_autosome(23)) return false;   // X (EIGENSTRAT code) dropped
    if (is_autosome(24)) return false;   // Y dropped
    if (is_autosome(90)) return false;   // MT dropped
    if (is_autosome(0) || is_autosome(-1)) return false;
    return true;
}

// ---- include/exclude membership ----------------------------------------------
[[nodiscard]] bool test_membership() {
    // No-op when empty.
    {
        FilterConfig cfg;
        SnpMembership m(cfg);
        if (!m.is_noop()) return false;
        if (!m.passes("rs1") || !m.passes("anything")) return false;
    }
    // Include set: only listed ids pass.
    {
        FilterConfig cfg;
        cfg.include_snp_ids = {"rs1", "rs3"};
        SnpMembership m(cfg);
        if (m.is_noop()) return false;
        if (!m.passes("rs1") || !m.passes("rs3")) return false;
        if (m.passes("rs2")) return false;
    }
    // Exclude set: listed ids fail, others pass (no include constraint).
    {
        FilterConfig cfg;
        cfg.exclude_snp_ids = {"rs2"};
        SnpMembership m(cfg);
        if (m.passes("rs2")) return false;
        if (!m.passes("rs1")) return false;
    }
    // Exclude overrides include: an id in both is dropped.
    {
        FilterConfig cfg;
        cfg.include_snp_ids = {"rs1", "rs2"};
        cfg.exclude_snp_ids = {"rs2"};
        SnpMembership m(cfg);
        if (!m.passes("rs1")) return false;
        if (m.passes("rs2")) return false;          // exclude wins
        if (m.passes("rs9")) return false;          // not in include set
    }
    return true;
}

// ---- read_snp_id_list: valid-file branch (the previously-zero file coverage) --
// Pins the documented parse contract on a real temp file: first whitespace token
// of each non-blank line is the id, trailing columns are tolerated, blank lines
// are skipped, and a CRLF (Windows-origin) line has its trailing '\r' delimited
// off the token (the token-extraction reader is CRLF-robust by construction).
[[nodiscard]] bool test_read_snp_id_list_valid() {
    TempFile f("valid");
    // Line 2 carries a trailing column (tolerated); line 3 is blank (skipped);
    // line 4 is whitespace-only (skipped); line 5 is CRLF-terminated.
    write_text(f.path,
               "rs100\n"
               "rs200 extra_column_ignored\n"
               "\n"
               "   \n"
               "rs300\r\n"
               "rs400\n");
    std::vector<std::string> ids;
    read_snp_id_list(f.path.string(), ids);
    // Exactly the four real ids, in file order, with NO trailing '\r' on rs300.
    const std::vector<std::string> want = {"rs100", "rs200", "rs300", "rs400"};
    if (ids != want) return false;

    // read_snp_id_list APPENDS (documented out-param semantics): a second read
    // accumulates rather than replacing.
    read_snp_id_list(f.path.string(), ids);
    if (ids.size() != 8) return false;
    return true;
}

// ---- read_snp_id_list: missing file throws (open-failure contract) -----------
[[nodiscard]] bool test_read_snp_id_list_missing_throws() {
    auto missing = temp_prune("missing");  // path NOT created
    std::error_code ec;
    if (std::filesystem::exists(missing, ec)) return false;  // sanity: truly absent
    std::vector<std::string> ids;
    return throws_runtime_error(
        [&] { read_snp_id_list(missing.string(), ids); });
}

// ---- read_snp_id_list: directory throws (the B19 fix — opens but read-fails) --
// A directory path constructs an ifstream successfully on POSIX/libstdc++ (the
// `if (!in)` open guard passes), but the first read sets badbit. Pre-B19 this was
// silently swallowed (empty list, fail-open); the post-loop bad()/fail()-without-
// eof() guard now fail-fasts. This case FAILED before the fix.
[[nodiscard]] bool test_read_snp_id_list_directory_throws() {
    auto dir = temp_prune("dir");
    std::error_code ec;
    std::filesystem::create_directory(dir, ec);
    if (ec) return false;  // could not stage the fixture
    std::vector<std::string> ids;
    const bool threw = throws_runtime_error(
        [&] { read_snp_id_list(dir.string(), ids); });
    // The fail-fast must NOT leave a partially-populated list masquerading as data.
    const bool clean = ids.empty();
    std::filesystem::remove(dir, ec);
    return threw && clean;
}

// ---- SnpMembership: prune.in unions with include, exclude overrides ----------
// The constructor's prune.in file branch (cfg.prune_in_path) had zero coverage.
// Pin: (1) prune ids contribute to the keep-set, (2) include ids union with them,
// (3) exclude overrides a prune-supplied id (exclude wins), and (4) a directory
// prune_in_path makes the CONSTRUCTOR fail-fast (the B19 contract end-to-end).
[[nodiscard]] bool test_membership_prune_in() {
    {
        TempFile f("membership");
        write_text(f.path, "rsP1\nrsP2\nrsBoth\n");  // rsBoth also excluded below
        FilterConfig cfg;
        cfg.prune_in_path = f.path.string();
        cfg.include_snp_ids = {"rsI1"};      // unions with the prune ids
        cfg.exclude_snp_ids = {"rsBoth"};    // overrides the prune-supplied id
        SnpMembership m(cfg);

        if (m.is_noop()) return false;                 // a real constraint exists
        if (m.keep_count() != 4) return false;         // rsP1, rsP2, rsBoth, rsI1
        if (m.drop_count() != 1) return false;         // rsBoth
        if (!m.passes("rsP1") || !m.passes("rsP2")) return false;  // prune ids kept
        if (!m.passes("rsI1")) return false;                       // include id kept
        if (m.passes("rsBoth")) return false;          // exclude wins over prune
        if (m.passes("rsOther")) return false;         // not in the keep-set union
    }
    // A directory prune_in_path must propagate the read-fail throw out of the ctor.
    {
        auto dir = temp_prune("ctor_dir");
        std::error_code ec;
        std::filesystem::create_directory(dir, ec);
        if (ec) return false;
        FilterConfig cfg;
        cfg.prune_in_path = dir.string();
        const bool threw = throws_runtime_error([&] { SnpMembership m(cfg); });
        std::filesystem::remove(dir, ec);
        if (!threw) return false;
    }
    return true;
}

// ---- --mind per-sample predicate over a hand-packed tile ----------------------
// Pack 3 individuals × 4 SNPs (1 byte/record). Sample 0: all non-missing; sample 1:
// 2 of 4 missing (frac 0.5); sample 2: all missing. Pin the kept set at a 0.4 cap.
[[nodiscard]] bool test_mind_prepass() {
    // Build one byte per individual (4 codes, MSB-first via code_in_byte order).
    // code 0/1/2 non-missing, code 3 (kMissingCode) missing.
    auto pack = [](std::uint8_t c0, std::uint8_t c1, std::uint8_t c2, std::uint8_t c3) {
        return static_cast<std::uint8_t>((c0 << 6) | (c1 << 4) | (c2 << 2) | c3);
    };
    std::vector<std::uint8_t> packed = {
        pack(0, 1, 2, 0),  // sample 0: 4 non-missing -> frac 0.00
        pack(0, 3, 1, 3),  // sample 1: 2 non-missing -> frac 0.50
        pack(3, 3, 3, 3),  // sample 2: 0 non-missing -> frac 1.00
    };
    // Sanity: confirm the bit order matches io::code_in_byte (the decode path).
    if (steppe::io::code_in_byte(packed[1], 0) != 0 ||
        steppe::io::code_in_byte(packed[1], 1) != 3 ||
        steppe::io::code_in_byte(packed[1], 2) != 1 ||
        steppe::io::code_in_byte(packed[1], 3) != 3) {
        return false;
    }

    MindPrepassInput in;
    in.packed = packed.data();
    in.bytes_per_record = 1;
    in.n_snp = 4;
    in.n_individuals = 3;

    // No-op: mind_max_missing == 1.0 keeps all 3.
    {
        FilterConfig cfg;  // mind_max_missing default 1.0
        const MindSummary s = run_mind_prepass(in, cfg);
        if (s.kept.size() != 3) return false;
        if (!close(s.missing_frac[0], 0.0) || !close(s.missing_frac[1], 0.5) ||
            !close(s.missing_frac[2], 1.0)) return false;
    }
    // Active cap 0.4: keep only sample 0 (frac 0 <= 0.4); drop samples 1 (0.5) & 2 (1.0).
    {
        FilterConfig cfg;
        cfg.mind_max_missing = 0.4;
        const MindSummary s = run_mind_prepass(in, cfg);
        if (s.kept.size() != 1 || s.kept[0] != 0) return false;
    }
    // Active cap exactly 0.5: keep samples 0 and 1 (the <= boundary side).
    {
        FilterConfig cfg;
        cfg.mind_max_missing = 0.5;
        const MindSummary s = run_mind_prepass(in, cfg);
        if (s.kept.size() != 2 || s.kept[0] != 0 || s.kept[1] != 1) return false;
    }
    return true;
}

// ---- --mind no-data fail-safe: n_snp == 0 under an ACTIVE filter (B21) --------
// The header/.cpp contradiction B21 fixed: the no-data convention is KEEP-ALL, even
// under an active mind filter. With zero SNPs the per-sample missing fraction is
// undefined, so every sample reports missing_frac == 0 and is kept (frac 0 <= any
// threshold) — the OPPOSITE of snp_filter's empty-denominator (frac 1.0 ⇒ drop). We
// pin this on BOTH the no-op default and a maximally-strict active cap so a future
// edit that "drops on no SNPs" (the wording the old header wrongly implied) fails
// here. Also covers packed == nullptr (the other arm of the no-data branch).
[[nodiscard]] bool test_mind_prepass_no_snps() {
    // n_snp == 0 with real samples; packed present but spanning zero SNPs.
    std::vector<std::uint8_t> packed = {0x00, 0x00, 0x00};  // bytes irrelevant: n_snp==0
    MindPrepassInput in;
    in.packed = packed.data();
    in.bytes_per_record = 1;
    in.n_snp = 0;
    in.n_individuals = 3;

    // No-op default (mind_max_missing == 1.0): all kept, frac 0 (undefined ⇒ 0).
    {
        FilterConfig cfg;  // default 1.0
        const MindSummary s = run_mind_prepass(in, cfg);
        if (s.kept.size() != 3) return false;
        for (std::size_t g = 0; g < 3; ++g) {
            if (s.nonmissing[g] != 0) return false;        // no SNPs counted
            if (!close(s.missing_frac[g], 0.0)) return false;  // undefined ⇒ 0
        }
        if (s.kept[0] != 0 || s.kept[1] != 1 || s.kept[2] != 2) return false;
    }
    // ACTIVE filter, maximally strict (cap 0.0): STILL keeps all 3 — the keep-all
    // fail-safe. This is the load-bearing B21 assertion (the old header implied
    // "drop all when active"; the code keeps all, and now the doc agrees).
    {
        FilterConfig cfg;
        cfg.mind_max_missing = 0.0;  // would drop any sample with frac > 0
        const MindSummary s = run_mind_prepass(in, cfg);
        if (s.kept.size() != 3) return false;
        if (s.kept[0] != 0 || s.kept[1] != 1 || s.kept[2] != 2) return false;
        for (std::size_t g = 0; g < 3; ++g) {
            if (!close(s.missing_frac[g], 0.0)) return false;
        }
    }
    // packed == nullptr with n_snp > 0 hits the SAME no-data branch (the other arm
    // of `packed != nullptr && n_snp > 0`): active filter still keeps all.
    {
        MindPrepassInput nul = in;
        nul.packed = nullptr;
        nul.n_snp = 8;  // claims SNPs but no buffer ⇒ no-data branch, keep-all
        FilterConfig cfg;
        cfg.mind_max_missing = 0.0;
        const MindSummary s = run_mind_prepass(nul, cfg);
        if (s.kept.size() != 3) return false;
    }
    // n_individuals == 0 with n_snp == 0: empty summary, no throw, empty kept set.
    {
        MindPrepassInput empt;
        empt.packed = nullptr;
        empt.n_snp = 0;
        empt.n_individuals = 0;
        FilterConfig cfg;
        cfg.mind_max_missing = 0.0;
        const MindSummary s = run_mind_prepass(empt, cfg);
        if (!s.kept.empty() || !s.nonmissing.empty() || !s.missing_frac.empty()) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// snp_filter HOST tests (cleanup B20). Previously the ONLY coverage of
// derive_per_snp_summary / build_snp_keep_mask was the GPU/real-AADR .cu oracle
// (a §13 violation — these are pure host functions). These hand-built cases pin:
// (a) the per-SNP summary arithmetic (pooled ref-af, folded MAF, missing frac,
// pooled allele count) on a tiny 2-pop × 3-SNP tile incl. a monomorphic and an
// all-missing SNP; (b) the extracted snp_keep_decision primitive (drop-not-flip
// order, no-op default, each gated filter, membership); (c) the new fail-fast
// guards (ploidy ∉ {1,2}, short pop_individuals, null q/n, short SnpTable); and
// (d) the is_monomorphic UNFOLDED-ref-af contract end-to-end via drop_monomorphic.
// ============================================================================

// A tiny well-formed tile: P=2 pops, M=3 SNPs, column-major (i + P*s), ploidy 2,
// pop_individuals {3, 2} (total 5). SNP0 polymorphic (ref-af 0.5, 0 missing);
// SNP1 monomorphic (ref-af 0, 0 missing); SNP2 all-missing (N==0 ⇒ missing 1.0).
[[nodiscard]] DecodedTileSummaryInput make_tile(const std::vector<double>& q,
                                                const std::vector<double>& n) {
    DecodedTileSummaryInput in;
    in.q = q.data();
    in.n = n.data();
    in.P = 2;
    in.M = 3;
    in.pop_individuals = {3, 2};
    in.ploidy = 2;
    return in;
}

[[nodiscard]] bool test_derive_per_snp_summary() {
    // index = p + 2*s : s0[p0,p1], s1[p0,p1], s2[p0,p1].
    const std::vector<double> q = {0.5, 0.5, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> n = {6.0, 4.0, 6.0, 4.0, 0.0, 0.0};
    const DecodedTileSummaryInput in = make_tile(q, n);
    const std::vector<PerSnpSummary> s = derive_per_snp_summary(in);
    if (s.size() != 3) return false;
    // SNP0: pooled_ref_count = 0.5*6 + 0.5*4 = 5; allele = 10; ref-af 0.5; MAF 0.5;
    //       nonmissing = 6/2 + 4/2 = 5; missing = 1 - 5/5 = 0.
    if (!close(s[0].pooled_ref_af, 0.5) || !close(s[0].pooled_minor_af, 0.5)) return false;
    if (!close(s[0].missing_frac, 0.0) || !close(s[0].pooled_allele_count, 10.0)) return false;
    // SNP1: monomorphic (all Q==0) ⇒ ref-af 0, MAF 0; 0 missing.
    if (!close(s[1].pooled_ref_af, 0.0) || !close(s[1].pooled_minor_af, 0.0)) return false;
    if (!close(s[1].missing_frac, 0.0) || !close(s[1].pooled_allele_count, 10.0)) return false;
    // SNP2: all-missing (N==0) ⇒ pooled_allele_count 0, ref-af 0 (denom guard),
    //       missing_frac 1.0.
    if (!close(s[2].pooled_allele_count, 0.0) || !close(s[2].pooled_ref_af, 0.0)) return false;
    if (!close(s[2].missing_frac, 1.0)) return false;
    // ploidy=1 path: nonmissing_indiv = Σ N (no /2). Make total match so missing=0.
    {
        DecodedTileSummaryInput h = in;
        h.ploidy = 1;
        h.pop_individuals = {6, 4};  // = N at SNP0, so missing_frac 0 there
        const std::vector<PerSnpSummary> sh = derive_per_snp_summary(h);
        if (!close(sh[0].missing_frac, 0.0)) return false;  // nonmissing 10 / total 10
    }
    return true;
}

// The extracted shared per-SNP decision primitive (B20/F28). Build summaries via
// the public derivation, then exercise snp_keep_decision directly.
[[nodiscard]] bool test_snp_keep_decision() {
    const std::vector<double> q = {0.5, 0.5, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> n = {6.0, 4.0, 6.0, 4.0, 0.0, 0.0};
    const std::vector<PerSnpSummary> s = derive_per_snp_summary(make_tile(q, n));
    const PerSnpSummary& poly = s[0];   // ref-af 0.5, MAF 0.5, missing 0
    const PerSnpSummary& mono = s[1];   // ref-af 0.0, MAF 0.0, missing 0
    const PerSnpSummary& miss = s[2];   // all-missing, missing 1.0

    // No-op default: a clean biallelic ACGT transversion SNP is KEPT.
    {
        FilterConfig cfg;  // maf_min 0, geno 1, all flags off
        if (!snp_keep_decision(poly, 'A', 'C', 1, cfg, true)) return false;
        // membership bit false ⇒ dropped even on the no-op path.
        if (snp_keep_decision(poly, 'A', 'C', 1, cfg, false)) return false;
    }
    // Unconditional DROP-NOT-FLIP: multiallelic / strand-ambiguous dropped REGARDLESS
    // of every threshold (membership true, default thresholds).
    {
        FilterConfig cfg;
        if (snp_keep_decision(poly, 'A', 'A', 1, cfg, true)) return false;  // ref==alt
        if (snp_keep_decision(poly, 'A', 'N', 1, cfg, true)) return false;  // non-ACGT
        if (snp_keep_decision(poly, 'A', 'T', 1, cfg, true)) return false;  // palindrome
    }
    // MAF filter (>= inclusive): a monomorphic SNP (MAF 0) drops at maf_min>0, the
    // polymorphic one (MAF 0.5) survives.
    {
        FilterConfig cfg; cfg.maf_min = 0.01;
        if (snp_keep_decision(mono, 'A', 'C', 1, cfg, true)) return false;
        if (!snp_keep_decision(poly, 'A', 'C', 1, cfg, true)) return false;
    }
    // geno filter (<= inclusive): the all-missing SNP (missing 1.0) drops at
    // geno_max_missing<1, survives at the no-op 1.0.
    {
        FilterConfig cfg; cfg.geno_max_missing = 0.5;
        if (snp_keep_decision(miss, 'A', 'C', 1, cfg, true)) return false;
        FilterConfig dflt;
        if (!snp_keep_decision(miss, 'A', 'C', 1, dflt, true)) return false;  // missing<=1
    }
    // drop_monomorphic: takes the UNFOLDED pooled_ref_af inside the primitive
    // (B20/F21). The monomorphic SNP (ref-af 0) drops; the polymorphic one does not.
    {
        FilterConfig cfg; cfg.drop_monomorphic = true;
        if (snp_keep_decision(mono, 'A', 'C', 1, cfg, true)) return false;
        if (!snp_keep_decision(poly, 'A', 'C', 1, cfg, true)) return false;
    }
    // transversions_only: keep a transversion (A/C), drop a transition (A/G).
    {
        FilterConfig cfg; cfg.transversions_only = true;
        if (!snp_keep_decision(poly, 'A', 'C', 1, cfg, true)) return false;
        if (snp_keep_decision(poly, 'A', 'G', 1, cfg, true)) return false;
    }
    // autosomes_only: keep chr 1..22, drop chr 23 (X).
    {
        FilterConfig cfg; cfg.autosomes_only = true;
        if (!snp_keep_decision(poly, 'A', 'C', 22, cfg, true)) return false;
        if (snp_keep_decision(poly, 'A', 'C', 23, cfg, true)) return false;
    }
    return true;
}

// build_snp_keep_mask end-to-end on the tiny tile + a matching SnpTable.
[[nodiscard]] bool test_build_snp_keep_mask() {
    const std::vector<double> q = {0.5, 0.5, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> n = {6.0, 4.0, 6.0, 4.0, 0.0, 0.0};
    const DecodedTileSummaryInput in = make_tile(q, n);

    SnpTable snps;
    snps.id = {"rs0", "rs1", "rs2"};
    snps.chrom = {1, 1, 23};                 // SNP2 on X
    snps.ref = {'A', 'A', 'A'};
    snps.alt = {'C', 'C', 'C'};              // all clean transversions
    snps.count = 3;

    // No-op default keeps every clean biallelic ACGT SNP (incl. the monomorphic and
    // all-missing ones — the documented no-op-when-default property).
    {
        FilterConfig cfg;
        SnpMembership mem(cfg);
        const std::vector<bool> keep = build_snp_keep_mask(in, snps, cfg, mem);
        if (keep.size() != 3) return false;
        if (!keep[0] || !keep[1] || !keep[2]) return false;
    }
    // maf_min>0 drops the monomorphic (SNP1) and the all-missing (SNP2, MAF 0).
    {
        FilterConfig cfg; cfg.maf_min = 0.01;
        SnpMembership mem(cfg);
        const std::vector<bool> keep = build_snp_keep_mask(in, snps, cfg, mem);
        if (!keep[0] || keep[1] || keep[2]) return false;
    }
    // autosomes_only drops the X-chromosome SNP2 only.
    {
        FilterConfig cfg; cfg.autosomes_only = true;
        SnpMembership mem(cfg);
        const std::vector<bool> keep = build_snp_keep_mask(in, snps, cfg, mem);
        if (!keep[0] || !keep[1] || keep[2]) return false;
    }
    // include-set membership keeps only the listed id (rs0).
    {
        FilterConfig cfg; cfg.include_snp_ids = {"rs0"};
        SnpMembership mem(cfg);
        const std::vector<bool> keep = build_snp_keep_mask(in, snps, cfg, mem);
        if (!keep[0] || keep[1] || keep[2]) return false;
    }
    return true;
}

// The B20 fail-fast guards: ploidy ∉ {1,2}, short pop_individuals, null q/n, short
// SnpTable — each a std::invalid_argument, not a silent wrong answer / segfault.
[[nodiscard]] bool test_snp_filter_fail_fast() {
    const std::vector<double> q = {0.5, 0.5, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> n = {6.0, 4.0, 6.0, 4.0, 0.0, 0.0};

    // ploidy ∉ {1,2} (B10/B20).
    if (!throws_invalid_argument([&] {
            DecodedTileSummaryInput in = make_tile(q, n); in.ploidy = 0;
            (void)derive_per_snp_summary(in);
        })) return false;
    if (!throws_invalid_argument([&] {
            DecodedTileSummaryInput in = make_tile(q, n); in.ploidy = 3;
            (void)derive_per_snp_summary(in);
        })) return false;

    // Short pop_individuals (B20/F1) — the spurious-negative-missing-frac path.
    if (!throws_invalid_argument([&] {
            DecodedTileSummaryInput in = make_tile(q, n);
            in.pop_individuals = {3};  // size 1 != P==2
            (void)derive_per_snp_summary(in);
        })) return false;
    // Over-long pop_individuals is ALSO rejected (size must EQUAL P, not just >=).
    if (!throws_invalid_argument([&] {
            DecodedTileSummaryInput in = make_tile(q, n);
            in.pop_individuals = {3, 2, 1};
            (void)derive_per_snp_summary(in);
        })) return false;

    // Null q / null n with P>0 && M>0 (B20/F6).
    if (!throws_invalid_argument([&] {
            DecodedTileSummaryInput in = make_tile(q, n); in.q = nullptr;
            (void)derive_per_snp_summary(in);
        })) return false;
    if (!throws_invalid_argument([&] {
            DecodedTileSummaryInput in = make_tile(q, n); in.n = nullptr;
            (void)derive_per_snp_summary(in);
        })) return false;

    // Short SnpTable ref/alt (B20/F3) via build_snp_keep_mask.
    if (!throws_invalid_argument([&] {
            const DecodedTileSummaryInput in = make_tile(q, n);
            SnpTable snps;
            snps.ref = {'A', 'C'};   // size 2 < M==3
            snps.alt = {'A', 'C'};
            FilterConfig cfg; SnpMembership mem(cfg);
            (void)build_snp_keep_mask(in, snps, cfg, mem);
        })) return false;
    // Short chrom only matters when autosomes_only is active.
    if (!throws_invalid_argument([&] {
            const DecodedTileSummaryInput in = make_tile(q, n);
            SnpTable snps;
            snps.ref = {'A', 'C', 'A'};
            snps.alt = {'C', 'A', 'C'};
            snps.chrom = {1, 1};     // size 2 < M==3
            FilterConfig cfg; cfg.autosomes_only = true;
            SnpMembership mem(cfg);
            (void)build_snp_keep_mask(in, snps, cfg, mem);
        })) return false;
    // Short id only matters when membership is active.
    if (!throws_invalid_argument([&] {
            const DecodedTileSummaryInput in = make_tile(q, n);
            SnpTable snps;
            snps.ref = {'A', 'C', 'A'};
            snps.alt = {'C', 'A', 'C'};
            snps.id = {"rs0", "rs1"};  // size 2 < M==3
            FilterConfig cfg; cfg.include_snp_ids = {"rs0"};
            SnpMembership mem(cfg);
            (void)build_snp_keep_mask(in, snps, cfg, mem);
        })) return false;

    // The degenerate empty cases must NOT throw (M<=0 / P<=0 short-circuit BEFORE
    // the guards — no data to validate). Empty pop_individuals is fine when P<=0.
    {
        DecodedTileSummaryInput in;
        in.P = 0; in.M = 0;  // q/n null, pop_individuals empty
        if (derive_per_snp_summary(in).size() != 0) return false;  // no throw
        SnpTable snps; FilterConfig cfg; SnpMembership mem(cfg);
        if (build_snp_keep_mask(in, snps, cfg, mem).size() != 0) return false;  // no throw
    }
    return true;
}

// filter_is_active: strand-mode alone is NOT a trigger (so a default FilterConfig is inactive
// and the front-end stays byte-identical); every genuine subset knob IS.
bool test_filter_is_active() {
    if (filter_is_active(FilterConfig{})) return false;               // all defaults -> inactive
    { FilterConfig c; c.strand_mode = steppe::StrandMode::Drop; if (filter_is_active(c)) return false; }
    { FilterConfig c; c.strand_mode = steppe::StrandMode::Keep; if (filter_is_active(c)) return false; }
    { FilterConfig c; c.maf_min = 0.05; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.geno_max_missing = 0.9; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.autosomes_only = true; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.drop_monomorphic = true; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.transversions_only = true; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.include_snp_ids = {"rs1"}; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.exclude_snp_ids = {"rs1"}; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.prune_in_path = "x.in"; if (!filter_is_active(c)) return false; }
    { FilterConfig c; c.maf_min = 0.0; c.geno_max_missing = 1.0; if (filter_is_active(c)) return false; }
    return true;
}

// classify_snp_id: rsID / chr:pos / array-probe / other namespaces.
bool test_classify_snp_id() {
    if (classify_snp_id("rs3094315") != IdNamespace::RsId) return false;
    if (classify_snp_id("RS12") != IdNamespace::RsId) return false;
    if (classify_snp_id("1:752566") != IdNamespace::ChrPos) return false;
    if (classify_snp_id("chr1:752566:G:A") != IdNamespace::ChrPos) return false;
    if (classify_snp_id("i5000940") != IdNamespace::Probe) return false;      // 23andMe internal
    if (classify_snp_id("GSA-rs123") != IdNamespace::Probe) return false;
    if (classify_snp_id("AX-11086525") != IdNamespace::Probe) return false;
    if (classify_snp_id("exm12345") != IdNamespace::Probe) return false;
    if (classify_snp_id("affx-1234") != IdNamespace::Probe) return false;
    if (classify_snp_id("rsBAD") != IdNamespace::Other) return false;          // rs + non-digits
    if (classify_snp_id("SNP_A_1") != IdNamespace::Other) return false;
    // Plurality on a set.
    if (classify_ascertainment({"rs1", "rs2", "rs3", "1:5"}).dominant != IdNamespace::RsId)
        return false;
    return true;
}

// check_same_ascertainment: a subset list (⊆ target) is NOT a mix; a cross-panel list (mostly
// absent from the target) IS, with a populated reason. Empty external list -> not mixed.
bool test_same_ascertainment_guard() {
    const std::vector<std::string> target = {"rs1", "rs2", "rs3", "rs4", "rs5",
                                             "rs6", "rs7", "rs8", "rs9", "rs10"};
    // A genuine restrict-to-subset list: fully contained -> not mixed.
    {
        const AscertainmentVerdict v =
            check_same_ascertainment(target, {"rs2", "rs4", "rs6"});
        if (v.mixed) return false;
        if (!(v.present_frac > 0.999)) return false;
    }
    // A cross-panel list: mostly foreign to the target -> mixed, reason populated.
    {
        const AscertainmentVerdict v = check_same_ascertainment(
            target, {"rsX1", "rsX2", "rsX3", "rsX4", "rs1"});  // 1 of 5 present = 0.2 < 0.5
        if (!v.mixed) return false;
        if (v.reason.empty()) return false;
        if (v.external_present != 1 || v.external_total != 5) return false;
    }
    // No external list: never a mix.
    {
        const AscertainmentVerdict v = check_same_ascertainment(target, {});
        if (v.mixed) return false;
    }
    // Exactly at the floor (0.5) is NOT mixed (< is strict).
    {
        const AscertainmentVerdict v =
            check_same_ascertainment(target, {"rs1", "rs2", "zzz1", "zzz2"});  // 2/4 = 0.5
        if (v.mixed) return false;
    }
    return true;
}

struct Case { const char* name; bool (*fn)(); };

constexpr Case kCases[] = {
    {"filter_is_active (strand alone is not a trigger; subset knobs are)", test_filter_is_active},
    {"classify_snp_id (rsID / chr:pos / probe / other)", test_classify_snp_id},
    {"same-ascertainment guard (subset ok; cross-panel refused)", test_same_ascertainment_guard},
    {"MAF boundary (>= inclusive; no-op at 0)", test_maf_boundary},
    {"geno boundary (<= inclusive; no-op at 1)", test_geno_boundary},
    {"mind boundary (<= inclusive; no-op at 1)", test_mind_boundary},
    {"monomorphic (folded MAF == 0)", test_monomorphic},
    {"strand-ambiguous (A/T,C/G palindrome class)", test_strand_ambiguous},
    {"multiallelic / non-clean-ACGT", test_multiallelic},
    {"transition/transversion split (GT/AC vs GA/CT)", test_transition_transversion},
    {"is_autosome (chr 1..22, AT2 auto_only)", test_is_autosome},
    {"include/exclude membership (exclude wins)", test_membership},
    {"read_snp_id_list valid file (first token; blank/CRLF; append)",
     test_read_snp_id_list_valid},
    {"read_snp_id_list missing file throws", test_read_snp_id_list_missing_throws},
    {"read_snp_id_list directory throws (B19 fail-fast)",
     test_read_snp_id_list_directory_throws},
    {"SnpMembership prune.in ∪ include; exclude overrides; ctor dir throws",
     test_membership_prune_in},
    {"--mind per-sample predicate + no-op", test_mind_prepass},
    {"--mind no-data fail-safe (n_snp==0 + active keeps all; B21)",
     test_mind_prepass_no_snps},
    {"derive_per_snp_summary (pooled ref-af/MAF/missing; mono + all-missing)",
     test_derive_per_snp_summary},
    {"snp_keep_decision primitive (drop-not-flip; gated filters; membership)",
     test_snp_keep_decision},
    {"build_snp_keep_mask end-to-end (no-op default; maf; autosomes; include)",
     test_build_snp_keep_mask},
    {"snp_filter B20 fail-fast (ploidy; short pop_individuals; null q/n; short SnpTable)",
     test_snp_filter_fail_fast},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(Filters, AllPredicates) {
    for (const auto& c : kCases) {
        EXPECT_TRUE(c.fn()) << "failed: " << c.name;
    }
}
#else
int main() {
    int failures = 0;
    for (const auto& c : kCases) {
        const bool ok = c.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", c.name);
        if (!ok) ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "test_filters: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_filters: all %zu predicate checks PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif
