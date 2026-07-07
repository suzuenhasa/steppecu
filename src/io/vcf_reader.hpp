// src/io/vcf_reader.hpp
//
// Reference: docs/reference/src_io_vcf_reader.hpp.md
//
// VcfReader — steppe's native gVCF-block-aware hardcall VCF reader (the sixth
// reader arm). Host-parsed (zlib gunzip + text parse), it streams a .vcf.gz once
// to build the gVCF reference-block coverage bitmaps + the per-position variant
// map, then resolves every GRCh38 target site by the interval join
//   variant record  >  passing ref block  >  failing ref block  >  no coverage
// exactly as the Stage-0 oracle (oracle.py resolve()), including the H1 hom-ref
// -> panel-A1 reconciliation, the H4 DP/GQ/PASS floor, the M4 multiallelic
// normalization + non-panel/half-call -> MISSING, the H3 per-record rsID-mismap
// drop, palindrome drop, and strand-flip reconciliation. It emits (a) the
// canonical individual-major SnpMajorTile for the one sample (fed to the shared
// device transpose) and (b) the per-site report in the oracle's exact schema.
//
// The report `call`/`source` labels reproduce the oracle's LAZY field assignment
// (oracle.py:243-326): `source` is "none" until the resolver commits to a path,
// so an rsID-mismap drop carries source="none" and a hom-ref block with REF==A2
// is call="homref" with dosage 0 (NEVER derived from dosage). See §5.
//
// Pure host C++20 io-leaf (zlib only); failures surface as std::runtime_error.
#ifndef STEPPE_IO_VCF_READER_HPP
#define STEPPE_IO_VCF_READER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "io/snp_major_tile.hpp"
#include "io/target_sites.hpp"

namespace steppe::io {

// --- VCF reference-build auto-detection --------------------------------------
// The assembly a VCF's coordinates are on. The AADR panel is fixed GRCh37, so a
// GRCh37 VCF joins the panel directly (identity, no lift) while a GRCh38 VCF must
// be lifted. Unknown -> the caller fails clearly rather than mis-genotyping.
enum class Assembly { GRCh37, GRCh38, Unknown };

struct AssemblyDetection {
    Assembly assembly = Assembly::Unknown;
    std::string evidence;         // human string for the stderr report, e.g.
                                  // "##contig chr22 length=51304566 -> GRCh37"
    bool from_reference = false;  // a ##reference token resolved the build
    bool from_contig = false;     // a ##contig autosome length resolved the build
};

// Detect the assembly from the VCF header ALONE (## and #CHROM lines; never a
// body record). The contig length is the decisive, unspoofable signal; a
// ##reference token corroborates it or resolves an absent/ambiguous contig. A
// contig-vs-reference disagreement yields Unknown (fail-clear). Cheap (a few KB
// of inflate); independent of VcfReader::genotype(). Throws on file-open failure.
[[nodiscard]] AssemblyDetection detect_vcf_assembly(const std::string& vcf_path);

// The resolved call class (report label — never recomputed from dosage).
enum class VcfCall { Homref, Het, Homalt, Missing, Dropped };

// One per-site report row, oracle schema (schema §9 of the stage-0 spec).
struct VcfSiteCall {
    std::string rsid;
    int chrom = 0;
    long long pos37 = 0;
    long long pos38 = 0;
    char a1 = 'N';
    char a2 = 'N';
    VcfCall call = VcfCall::Dropped;
    int dosage = -1;              // copies of panel A1 in {0,1,2}; -1 == NA
    std::string source = "none";  // refblock | variant | none
    int flip = 0;                 // 1 == strand-flip applied
    std::string drop_reason;      // "" | palindrome | rsid_mismap | ref_change | ...
};

// Aggregate counters (for the stderr summary + the recovered/dropped split).
struct VcfCounts {
    long long called_variant = 0;
    long long called_refblock = 0;
    long long homref = 0, het = 0, homalt = 0;
    long long missing_total = 0;
    long long missing_below_floor = 0, missing_not_pass = 0;
    long long missing_non_panel = 0, missing_no_coverage = 0;
    long long missing_half_or_missing_gt = 0, missing_no_refbase = 0;
    long long dropped_total = 0;
    long long drop_palindrome = 0, drop_rsid_mismap = 0, drop_ref_change = 0;
    long long records_seen = 0;
    long long variant_at_target = 0;
};

struct VcfIngestResult {
    SnpMajorTile tile;                  // canonical individual-major source tile (1 sample)
    std::vector<VcfSiteCall> calls;     // one row per target site, panel order
    VcfCounts counts;
    std::string sample_id;
};

class VcfReader {
public:
    struct Options {
        int min_dp = 8;         // ref-block MinDP / variant DP floor (H4)
        int min_gq = 20;        // variant GQ floor (H4)
        bool autosomes_only = true;  // targets are 1..22, so non-1..22 CHROM is skipped
    };

    // sample_id == "" selects the sole sample (fails if the file carries >1).
    VcfReader(std::string vcf_path, const TargetSites& targets, std::string sample_id,
              Options opts);

    [[nodiscard]] VcfIngestResult genotype();

private:
    std::string vcf_path_;
    const TargetSites& targets_;
    std::string sample_id_;
    Options opts_;
};

}  // namespace steppe::io

#endif  // STEPPE_IO_VCF_READER_HPP
