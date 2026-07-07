// tests/unit/test_vcf_build_detect.cpp
//
// Host-only by-construction unit test of io::detect_vcf_assembly (the VCF
// reference-build auto-detector). Pure C++ TU, NO GPU: it authors tiny
// header-only .vcf fixtures and asserts the detected Assembly + evidence flags.
// Pins the detector's rules and the critic fixes:
//   - chr1 contig length -> GRCh37 / GRCh38 (the canonical signal)
//   - a SINGLE-chromosome (chr22-only) VCF is still detectable from its own
//     contig length (fix #2 — the 1000G phase3 per-chromosome shape the gate uses)
//   - the 'chr' prefix is stripped
//   - ##reference-only resolution (from_reference), incl. human_g1k_v37 / GRCh38
//   - absent/ambiguous -> Unknown
//   - contig-vs-reference conflict -> Unknown (fail-clear)
//   - contig + corroborating ##reference -> both flags set
// Self-checking main(); reports NO science number.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "io/vcf_reader.hpp"

namespace {

int g_failures = 0;

void write_file(const std::filesystem::path& p, const std::string& s) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

const char* name(steppe::io::Assembly a) {
    using A = steppe::io::Assembly;
    return a == A::GRCh37 ? "GRCh37" : a == A::GRCh38 ? "GRCh38" : "Unknown";
}

// A minimal header body appended after the caller's ## lines, so the reader stops
// at #CHROM without ever seeing a body record.
const char* kChromLine = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n";

struct Case {
    std::string name;
    std::string header;                 // the ## lines (kChromLine appended)
    steppe::io::Assembly want;
    int want_contig;                    // -1 == don't care, else 0/1
    int want_reference;                 // -1 == don't care, else 0/1
};

void run_case(const std::filesystem::path& tmp, const Case& tc, int idx) {
    const std::filesystem::path p = tmp / ("case_" + std::to_string(idx) + ".vcf");
    write_file(p, "##fileformat=VCFv4.2\n" + tc.header + kChromLine);
    const steppe::io::AssemblyDetection d = steppe::io::detect_vcf_assembly(p.string());
    bool ok = (d.assembly == tc.want);
    if (tc.want_contig >= 0 && d.from_contig != (tc.want_contig != 0)) ok = false;
    if (tc.want_reference >= 0 && d.from_reference != (tc.want_reference != 0)) ok = false;
    if (!ok) {
        std::printf("  [FAIL] %-28s got{asm=%s,contig=%d,ref=%d ev='%s'} want{asm=%s}\n",
                    tc.name.c_str(), name(d.assembly), d.from_contig ? 1 : 0,
                    d.from_reference ? 1 : 0, d.evidence.c_str(), name(tc.want));
        ++g_failures;
    } else {
        std::printf("  [ ok ] %-28s -> %s (%s)\n", tc.name.c_str(), name(d.assembly),
                    d.evidence.c_str());
    }
}

}  // namespace

int main() {
    using A = steppe::io::Assembly;
    std::printf("=== detect_vcf_assembly by-construction unit test (no GPU) ===\n");

    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_build_detect_test_" +
         std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const Case cases[] = {
        // chr1 contig length — the canonical signal.
        {"chr1 contig GRCh37", "##contig=<ID=1,length=249250621>\n", A::GRCh37, 1, 0},
        {"chr1 contig GRCh38", "##contig=<ID=1,length=248956422>\n", A::GRCh38, 1, 0},
        // 'chr'-prefixed ID is normalized.
        {"chr-prefix GRCh38", "##contig=<ID=chr1,length=248956422>\n", A::GRCh38, 1, 0},
        // fix #2: a single-chromosome chr22-only VCF (the 1000G phase3 gate shape)
        // is detectable from its own contig length — no chr1 line present.
        {"chr22-only GRCh37",
         "##contig=<ID=22,assembly=b37,length=51304566>\n", A::GRCh37, 1, -1},
        {"chr22-only GRCh38", "##contig=<ID=22,length=50818468>\n", A::GRCh38, 1, -1},
        // ##reference only (no contig).
        {"reference human_g1k_v37",
         "##reference=file:///refs/human_g1k_v37.fasta\n", A::GRCh37, 0, 1},
        {"reference GRCh38", "##reference=GRCh38\n", A::GRCh38, 0, 1},
        {"reference hg19", "##reference=hg19\n", A::GRCh37, 0, 1},
        // absent / ambiguous -> Unknown.
        {"no signal", "##source=myCaller\n##contig=<ID=1,length=123>\n", A::Unknown, -1, -1},
        // conflict: contig GRCh38 vs ##reference GRCh37 -> Unknown (fail-clear).
        {"contig/ref conflict",
         "##contig=<ID=1,length=248956422>\n##reference=human_g1k_v37\n", A::Unknown, -1, -1},
        // corroborating: contig GRCh37 + ##reference hg19 -> both flags.
        {"contig+reference agree",
         "##contig=<ID=1,length=249250621>\n##reference=hg19.fa\n", A::GRCh37, 1, 1},
        // fix #6: an ID= substring inside another value must not mis-hit; length
        // bounded at the delimiter (md5 hex here would break a naive scan).
        {"attr-order robustness",
         "##contig=<ID=1,assembly=GRCh37,md5=0aabIDlength,length=249250621>\n", A::GRCh37, 1, 0},
    };

    int idx = 0;
    for (const Case& c : cases) run_case(tmp, c, idx++);

    std::filesystem::remove_all(tmp, ec);

    if (g_failures == 0) {
        std::printf("\nRESULT: PASS (build auto-detection: contig length, chr22-only, "
                    "##reference, conflict, corroboration)\n");
        return 0;
    }
    std::printf("\nRESULT: FAIL (%d check(s) failed)\n", g_failures);
    return 1;
}
