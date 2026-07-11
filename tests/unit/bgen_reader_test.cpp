// tests/unit/bgen_reader_test.cpp
//
// Host unit gate for the BGEN v1.2 dosage reader (`steppe pca --bgen` ingest arm). It
// authors tiny hand-crafted BGEN v1.2 files byte-by-byte and checks the reader against an
// INDEPENDENT hand-decode of the same raw probability integers — exercising the FRACTIONAL
// dosage arithmetic that the real fully-called 1000G gate (integer dosages) cannot. Covers:
//   - header parse (offset/Lh/M/N/magic/flags: compression + layout + sample-id bits),
//   - the B-bit LSB-first bit-unpack at B=8 and B=16,
//   - unphased dosage 2 - 2*P(AA) - P(AB) and phased dosage 2 - P(REF|h0) - P(REF|h1),
//   - the missing-call NaN sentinel (ploidy byte bit7),
//   - compression 0 (none) AND 1 (zlib) yielding identical dosages,
//   - out-of-scope shapes (multiallelic K!=2, zstd flag, non-diploid ploidy) throwing clearly.
// Device-free; links steppe::io + zlib (to author the compressed fixture).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

#include "io/bgen_reader.hpp"

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}
bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// Little-endian byte writer for authoring the fixture.
struct BW {
    std::vector<std::uint8_t> b;
    void u8(std::uint32_t v) { b.push_back(static_cast<std::uint8_t>(v & 0xFF)); }
    void u16(std::uint32_t v) {
        u8(v & 0xFF);
        u8((v >> 8) & 0xFF);
    }
    void u32(std::uint32_t v) {
        u8(v & 0xFF);
        u8((v >> 8) & 0xFF);
        u8((v >> 16) & 0xFF);
        u8((v >> 24) & 0xFF);
    }
    void str(const std::string& s) {
        for (char c : s) b.push_back(static_cast<std::uint8_t>(c));
    }
    void bytes(const std::vector<std::uint8_t>& v) {
        b.insert(b.end(), v.begin(), v.end());
    }
};

// One variant's identifying block (rsid "rs{idx}", chrom "1", biallelic A/G).
std::vector<std::uint8_t> variant_id_block(int idx, std::uint32_t pos) {
    BW w;
    w.u16(0);                              // Lid = 0 (empty id)
    const std::string rsid = "rs" + std::to_string(idx);
    w.u16(static_cast<std::uint32_t>(rsid.size()));
    w.str(rsid);
    w.u16(1);
    w.str("1");                            // chrom
    w.u32(pos);
    w.u16(2);                              // K = 2 (biallelic)
    w.u32(1);
    w.str("A");                            // allele0 = REF
    w.u32(1);
    w.str("G");                            // allele1 = ALT (dosage-counted)
    return w.b;
}

// The uncompressed layout-2 genotype payload. `probs` holds 2 raw B-bit values per sample
// (v0,v1 for each of N samples, in sample order); `missing[i]` sets that sample's bit7.
std::vector<std::uint8_t> geno_payload(int N, int B, int phased,
                                       const std::vector<std::uint32_t>& probs,
                                       const std::vector<bool>& missing) {
    BW w;
    w.u32(static_cast<std::uint32_t>(N));
    w.u16(2);       // K
    w.u8(2);        // Pmin
    w.u8(2);        // Pmax
    for (int i = 0; i < N; ++i) w.u8(missing[static_cast<std::size_t>(i)] ? 0x82u : 0x02u);
    w.u8(static_cast<std::uint32_t>(phased));
    w.u8(static_cast<std::uint32_t>(B));
    // Bit-pack the probability values LSB-first (each value B bits).
    std::vector<std::uint8_t> bits;
    std::size_t bitpos = 0;
    auto put = [&](std::uint32_t val) {
        for (int i = 0; i < B; ++i) {
            const std::size_t byte = bitpos >> 3;
            if (byte >= bits.size()) bits.push_back(0);
            if ((val >> i) & 1u)
                bits[byte] = static_cast<std::uint8_t>(bits[byte] | (1u << (bitpos & 7u)));
            ++bitpos;
        }
    };
    for (std::uint32_t v : probs) put(v);
    w.bytes(bits);
    return w.b;
}

// Wrap a payload as a genotype block: compression 0 -> [u32 C][C bytes]; compression 1 ->
// [u32 C][u32 D][zlib(payload)], C = 4 + compressed_len.
std::vector<std::uint8_t> geno_block(const std::vector<std::uint8_t>& payload, int compression) {
    BW w;
    if (compression == 0) {
        w.u32(static_cast<std::uint32_t>(payload.size()));
        w.bytes(payload);
    } else {
        uLongf clen = compressBound(static_cast<uLong>(payload.size()));
        std::vector<std::uint8_t> comp(clen);
        const int rc = compress(comp.data(), &clen, payload.data(),
                                static_cast<uLong>(payload.size()));
        if (rc != Z_OK) throw std::runtime_error("test: zlib compress failed");
        comp.resize(clen);
        w.u32(static_cast<std::uint32_t>(4 + comp.size()));  // C
        w.u32(static_cast<std::uint32_t>(payload.size()));   // D
        w.bytes(comp);
    }
    return w.b;
}

// Full 3-variant / 2-sample fixture (no sample-id block). `compression` = 0 or 1.
std::vector<std::uint8_t> build_fixture(int compression) {
    const int N = 2;
    const int M = 3;
    // Variant genotype blocks.
    // V0: B=8 unphased. s0 v0=51,v1=128 ; s1 v0=0,v1=255.
    auto p0 = geno_payload(N, 8, 0, {51, 128, 0, 255}, {false, false});
    // V1: B=16 unphased. s0 v0=0,v1=0 (hom-ALT) ; s1 MISSING (probs zero).
    auto p1 = geno_payload(N, 16, 0, {0, 0, 0, 0}, {false, true});
    // V2: B=8 phased. s0 v0=255,v1=255 (hom-REF) ; s1 v0=51,v1=204.
    auto p2 = geno_payload(N, 8, 1, {255, 255, 51, 204}, {false, false});

    BW body;
    body.bytes(variant_id_block(0, 1000));
    body.bytes(geno_block(p0, compression));
    body.bytes(variant_id_block(1, 2000));
    body.bytes(geno_block(p1, compression));
    body.bytes(variant_id_block(2, 3000));
    body.bytes(geno_block(p2, compression));

    BW f;
    f.u32(20);  // offset (from byte 4) -> first variant at byte 24 == 4 + Lh (no sample block)
    f.u32(20);  // Lh
    f.u32(static_cast<std::uint32_t>(M));
    f.u32(static_cast<std::uint32_t>(N));
    f.str("bgen");
    f.u32(0x00000008u | static_cast<std::uint32_t>(compression));  // layout 2, no sample ids
    f.bytes(body.b);
    return f.b;
}

void write_file(const std::filesystem::path& p, const std::vector<std::uint8_t>& b) {
    std::ofstream o(p, std::ios::binary | std::ios::trunc);
    o.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

// The independent hand-decode of the fixture's raw integers.
double exp_unphased(std::uint32_t r0, std::uint32_t r1, double denom) {
    return 2.0 - 2.0 * (static_cast<double>(r0) / denom) - (static_cast<double>(r1) / denom);
}
double exp_phased(std::uint32_t r0, std::uint32_t r1, double denom) {
    return 2.0 - (static_cast<double>(r0) / denom) - (static_cast<double>(r1) / denom);
}

}  // namespace

int main() {
    std::error_code ec;
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path(ec) /
        ("steppe_bgen_" + std::to_string(static_cast<long long>(
             std::filesystem::file_time_type::clock::now().time_since_epoch().count())));
    std::filesystem::create_directories(tmp, ec);

    const double d8 = 255.0;      // 2^8 - 1
    const double d16 = 65535.0;   // 2^16 - 1
    const double tol = 1e-5;

    // Compression 0 (none) AND 1 (zlib) must both decode identically.
    for (int comp : {0, 1}) {
        const std::filesystem::path bp = tmp / ("fix_c" + std::to_string(comp) + ".bgen");
        write_file(bp, build_fixture(comp));
        const steppe::io::DosageTile t = steppe::io::read_bgen_dosages(bp.string());
        const std::string tag = "comp" + std::to_string(comp) + ": ";

        check(t.n_snp == 3 && t.n_individuals == 2, (tag + "geometry 3x2").c_str());
        check(t.sample_ids.size() == 2 && t.sample_ids[0] == "sample_0" &&
                  t.sample_ids[1] == "sample_1",
              (tag + "synthesized sample ids").c_str());
        check(t.snps.size() == 3 && t.snps[0].rsid == "rs0" && t.snps[0].chrom == "1" &&
                  t.snps[0].pos == 1000 && t.snps[0].allele0 == "A" && t.snps[0].allele1 == "G",
              (tag + "variant0 metadata").c_str());

        // dosage is individual-major: dosage[i*n_snp + s].
        const std::size_t M = t.n_snp;
        auto dose = [&](int i, int s) { return t.dosage[static_cast<std::size_t>(i) * M + s]; };

        // V0 (B=8, unphased).
        check(close(dose(0, 0), exp_unphased(51, 128, d8), tol),
              (tag + "V0 s0 fractional unphased").c_str());
        check(close(dose(1, 0), exp_unphased(0, 255, d8), tol), (tag + "V0 s1 het").c_str());
        // V1 (B=16, unphased): s0 hom-ALT = 2.0, s1 MISSING = NaN.
        check(close(dose(0, 1), exp_unphased(0, 0, d16), tol), (tag + "V1 s0 hom-ALT==2").c_str());
        check(close(dose(0, 1), 2.0, tol), (tag + "V1 s0 dosage 2.0").c_str());
        check(std::isnan(dose(1, 1)), (tag + "V1 s1 missing -> NaN").c_str());
        // V2 (B=8, phased): s0 hom-REF = 0.0, s1 = 1.0.
        check(close(dose(0, 2), exp_phased(255, 255, d8), tol), (tag + "V2 s0 phased hom-REF").c_str());
        check(close(dose(0, 2), 0.0, tol), (tag + "V2 s0 dosage 0.0").c_str());
        check(close(dose(1, 2), exp_phased(51, 204, d8), tol), (tag + "V2 s1 phased frac").c_str());
        check(close(dose(1, 2), 1.0, tol), (tag + "V2 s1 dosage 1.0").c_str());
    }

    // ---- Scope guards: out-of-scope shapes must throw clearly ----
    auto throws = [](const std::vector<std::uint8_t>& bytes, const std::filesystem::path& p) {
        write_file(p, bytes);
        try {
            (void)steppe::io::read_bgen_dosages(p.string());
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };

    // zstd (compression flag 2) out of scope.
    {
        BW f;
        f.u32(20);
        f.u32(20);
        f.u32(1);
        f.u32(2);
        f.str("bgen");
        f.u32(0x00000008u | 2u);  // layout 2, compression 2 (zstd)
        check(throws(f.b, tmp / "zstd.bgen"), "zstd compression throws");
    }
    // Multiallelic (K=3) out of scope.
    {
        BW f;
        f.u32(20);
        f.u32(20);
        f.u32(1);
        f.u32(1);
        f.str("bgen");
        f.u32(0x00000008u);  // layout 2, comp 0
        // variant with K=3
        f.u16(0);
        f.u16(2);
        f.str("rs");
        f.u16(1);
        f.str("1");
        f.u32(100);
        f.u16(3);  // K = 3 -> out of scope
        f.u32(1);
        f.str("A");
        f.u32(1);
        f.str("C");
        f.u32(1);
        f.str("G");
        check(throws(f.b, tmp / "multi.bgen"), "multiallelic K!=2 throws");
    }
    // Non-diploid ploidy (payload ploidy byte == 1) out of scope.
    {
        auto pay = geno_payload(1, 8, 0, {51, 128}, {false});
        pay[8] = 0x01;  // ploidy byte (after u32 N + u16 K + Pmin + Pmax = offset 8) -> ploidy 1
        // NOTE: also trips Pmin/Pmax==2 guard first; either way it must throw.
        BW f;
        f.u32(20);
        f.u32(20);
        f.u32(1);
        f.u32(1);
        f.str("bgen");
        f.u32(0x00000008u);
        f.bytes(variant_id_block(0, 100));
        f.bytes(geno_block(pay, 0));
        check(throws(f.b, tmp / "haploid.bgen"), "non-diploid ploidy throws");
    }

    std::filesystem::remove_all(tmp, ec);
    if (g_fail == 0) {
        std::printf("RESULT: PASS (bgen reader: fractional dosage, phased/unphased, missing, "
                    "comp0/zlib, scope guards)\n");
        return 0;
    }
    std::printf("RESULT: FAIL (%d check(s) failed)\n", g_fail);
    return 1;
}
