// src/io/bgen_reader.cpp
//
// BGEN v1.2 biallelic-diploid dosage reader. See bgen_reader.hpp for the format and
// scope contract. The whole file is read into memory and parsed with bounds-checked
// cursors (the region-scale inputs are a few MB; a streaming reader is a biobank-
// scale follow-on). Genotype blocks are zlib-inflated (compression 1) or taken raw
// (compression 0); the B-bit LSB-first probability bitstream is unpacked on the CPU.
#include "io/bgen_reader.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

namespace steppe::io {

namespace {

// A bounds-checked little-endian cursor over the file bytes. Every read validates it
// stays inside [0, size) and throws on a short/truncated file — no OOB access.
class ByteCursor {
public:
    ByteCursor(const std::uint8_t* data, std::size_t size, const char* what)
        : data_(data), size_(size), what_(what) {}

    [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
    void seek(std::size_t p) {
        if (p > size_) fail("seek past end");
        pos_ = p;
    }

    [[nodiscard]] std::uint8_t u8() {
        need(1);
        return data_[pos_++];
    }
    [[nodiscard]] std::uint16_t u16() {
        need(2);
        const std::uint16_t v = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(data_[pos_]) |
            (static_cast<std::uint16_t>(data_[pos_ + 1]) << 8));
        pos_ += 2;
        return v;
    }
    [[nodiscard]] std::uint32_t u32() {
        need(4);
        const std::uint32_t v = static_cast<std::uint32_t>(data_[pos_]) |
                                (static_cast<std::uint32_t>(data_[pos_ + 1]) << 8) |
                                (static_cast<std::uint32_t>(data_[pos_ + 2]) << 16) |
                                (static_cast<std::uint32_t>(data_[pos_ + 3]) << 24);
        pos_ += 4;
        return v;
    }
    [[nodiscard]] std::string str(std::size_t len) {
        need(len);
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }
    // Return a pointer to `len` bytes at the cursor and advance past them.
    [[nodiscard]] const std::uint8_t* take(std::size_t len) {
        need(len);
        const std::uint8_t* p = data_ + pos_;
        pos_ += len;
        return p;
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("io::read_bgen_dosages: " + std::string(what_) + ": " + msg);
    }

private:
    void need(std::size_t n) {
        if (pos_ + n > size_) fail("unexpected end of file (truncated)");
    }
    const std::uint8_t* data_;
    std::size_t size_;
    const char* what_;
    std::size_t pos_ = 0;
};

// LSB-first bit reader over a byte span: successive read(B) calls return the B-bit
// unsigned values in the order the BGEN v1.2 probability bitstream packs them (the
// least-significant bit of the first value in the least-significant bit of byte 0).
class BitReader {
public:
    BitReader(const std::uint8_t* data, std::size_t nbytes) : data_(data), nbits_(nbytes * 8) {}

    // Read B bits (1..32) as an unsigned value; out-of-range bits read as 0 (the
    // caller has already validated the payload length covers all samples).
    [[nodiscard]] std::uint32_t read(int B) {
        std::uint32_t v = 0;
        for (int i = 0; i < B; ++i) {
            if (bitpos_ < nbits_) {
                const std::size_t byte = bitpos_ >> 3;
                const int bit = static_cast<int>(bitpos_ & 7u);
                const std::uint32_t b = (static_cast<std::uint32_t>(data_[byte]) >> bit) & 1u;
                v |= (b << i);
            }
            ++bitpos_;
        }
        return v;
    }

private:
    const std::uint8_t* data_;
    std::size_t nbits_;
    std::size_t bitpos_ = 0;
};

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr)
        throw std::runtime_error("io::read_bgen_dosages: cannot open BGEN file: " + path);
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        throw std::runtime_error("io::read_bgen_dosages: fseek failed: " + path);
    }
    const long len = std::ftell(f);
    if (len < 0) {
        std::fclose(f);
        throw std::runtime_error("io::read_bgen_dosages: ftell failed: " + path);
    }
    std::rewind(f);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(len));
    if (len > 0) {
        const std::size_t got = std::fread(buf.data(), 1, static_cast<std::size_t>(len), f);
        if (got != static_cast<std::size_t>(len)) {
            std::fclose(f);
            throw std::runtime_error("io::read_bgen_dosages: short read: " + path);
        }
    }
    std::fclose(f);
    return buf;
}

// zlib-inflate `src` (a zlib-wrapped stream) into exactly `dst_len` bytes.
std::vector<std::uint8_t> zlib_inflate(const std::uint8_t* src, std::size_t src_len,
                                       std::size_t dst_len) {
    std::vector<std::uint8_t> out(dst_len);
    if (dst_len == 0) return out;
    uLongf out_len = static_cast<uLongf>(dst_len);
    const int rc = uncompress(out.data(), &out_len, src, static_cast<uLong>(src_len));
    if (rc != Z_OK)
        throw std::runtime_error("io::read_bgen_dosages: zlib inflate failed (code " +
                                 std::to_string(rc) + ")");
    if (out_len != static_cast<uLongf>(dst_len))
        throw std::runtime_error(
            "io::read_bgen_dosages: inflated length mismatch (header D disagrees with stream)");
    return out;
}

// Parse the optional sample-identifier block (present when flags bit31 is set). The
// block spans [4+Lh, 4+offset): uint32 block_length, uint32 N, then N*(uint16 Lid,id).
std::vector<std::string> parse_sample_ids(const std::uint8_t* data, std::size_t size,
                                          std::size_t block_start, std::uint32_t n_samples) {
    ByteCursor c(data, size, "sample-id block");
    c.seek(block_start);
    (void)c.u32();  // block_length (redundant; we bound by N)
    const std::uint32_t n2 = c.u32();
    if (n2 != n_samples)
        c.fail("sample-id count disagrees with header N");
    std::vector<std::string> ids;
    ids.reserve(n_samples);
    for (std::uint32_t i = 0; i < n_samples; ++i) {
        const std::uint16_t lid = c.u16();
        ids.push_back(c.str(lid));
    }
    return ids;
}

}  // namespace

DosageTile read_bgen_dosages(const std::string& path) {
    const std::vector<std::uint8_t> file = read_file(path);
    const std::uint8_t* data = file.data();
    const std::size_t size = file.size();

    ByteCursor hdr(data, size, "header");
    const std::uint32_t offset = hdr.u32();
    const std::uint32_t Lh = hdr.u32();
    const std::uint32_t M = hdr.u32();
    const std::uint32_t N = hdr.u32();
    const std::string magic = hdr.str(4);
    if (magic != "bgen")
        hdr.fail("bad magic (not a BGEN file)");

    // flags live at the last 4 bytes of the header block [4, 4+Lh).
    if (Lh < 20) hdr.fail("header length Lh < 20");
    ByteCursor fc(data, size, "flags");
    fc.seek(static_cast<std::size_t>(4) + Lh - 4);
    const std::uint32_t flags = fc.u32();
    const int compression = static_cast<int>(flags & 0x3u);
    const int layout = static_cast<int>((flags >> 2) & 0xFu);
    const bool sample_ids_present = ((flags >> 31) & 0x1u) != 0;

    if (layout != 2)
        hdr.fail("layout " + std::to_string(layout) + " out of scope (v1 supports layout 2 only)");
    if (compression == 2)
        hdr.fail("zstd compression (flag 2) out of scope (v1 supports none/zlib; re-export with "
                 "plink2 --export bgen-1.2 which emits zlib)");
    if (compression != 0 && compression != 1)
        hdr.fail("unknown compression flag " + std::to_string(compression));

    DosageTile tile;
    tile.n_snp = M;
    tile.n_individuals = N;

    // Sample IDs (parsed if present; otherwise synthesized sample_0.. as a fallback).
    if (sample_ids_present) {
        tile.sample_ids = parse_sample_ids(data, size, static_cast<std::size_t>(4) + Lh, N);
    } else {
        tile.sample_ids.reserve(N);
        for (std::uint32_t i = 0; i < N; ++i)
            tile.sample_ids.push_back("sample_" + std::to_string(i));
    }

    tile.snps.reserve(M);
    const float kMissing = std::numeric_limits<float>::quiet_NaN();
    tile.dosage.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(M), kMissing);

    ByteCursor c(data, size, "variant");
    c.seek(static_cast<std::size_t>(4) + offset);

    for (std::uint32_t v = 0; v < M; ++v) {
        // --- Variant identifying block (layout 2) ---
        BgenSnpMeta meta;
        const std::uint16_t lid = c.u16();
        (void)c.str(lid);  // variant id (not retained; rsid is the join key)
        const std::uint16_t lrsid = c.u16();
        meta.rsid = c.str(lrsid);
        const std::uint16_t lchr = c.u16();
        meta.chrom = c.str(lchr);
        meta.pos = c.u32();
        const std::uint16_t K = c.u16();
        if (K != 2)
            c.fail("variant " + std::to_string(v) + " has " + std::to_string(K) +
                   " alleles (biallelic K==2 only in v1)");
        const std::uint32_t la0 = c.u32();
        meta.allele0 = c.str(la0);  // REF
        const std::uint32_t la1 = c.u32();
        meta.allele1 = c.str(la1);  // ALT (dosage-counted)

        // --- Genotype data block ---
        const std::uint32_t C = c.u32();  // total length of the block that follows
        std::vector<std::uint8_t> payload;
        const std::uint8_t* pl = nullptr;
        std::size_t pl_len = 0;
        if (compression == 0) {
            pl = c.take(C);
            pl_len = C;
        } else {
            if (C < 4) c.fail("compressed block length C < 4");
            const std::uint32_t D = c.u32();  // uncompressed length
            const std::uint8_t* comp = c.take(C - 4);
            payload = zlib_inflate(comp, C - 4, D);
            pl = payload.data();
            pl_len = payload.size();
        }

        // --- Uncompressed layout-2 payload ---
        ByteCursor g(pl, pl_len, "genotype payload");
        const std::uint32_t Ng = g.u32();
        if (Ng != N) g.fail("payload N disagrees with header N");
        const std::uint16_t Kg = g.u16();
        if (Kg != 2) g.fail("payload K != 2");
        const std::uint8_t pmin = g.u8();
        const std::uint8_t pmax = g.u8();
        if (pmin != 2 || pmax != 2)
            g.fail("min/max ploidy != 2 (diploid only in v1)");
        const std::uint8_t* ploidy_miss = g.take(N);  // one byte/sample
        const std::uint8_t phased = g.u8();
        const std::uint8_t B = g.u8();
        if (B < 1 || B > 32) g.fail("bits-per-probability B out of range");
        const double denom = static_cast<double>((std::uint64_t{1} << B) - 1);

        // 2 probabilities per sample (biallelic diploid, phased or unphased). The bitstream
        // occupies the rest of the payload.
        const std::size_t bits_off = g.pos();
        const std::size_t bits_len = pl_len - bits_off;
        const std::uint8_t* bits = g.take(bits_len);
        BitReader br(bits, bits_len);

        for (std::uint32_t i = 0; i < N; ++i) {
            const std::uint8_t pm = ploidy_miss[i];
            const bool missing = (pm & 0x80u) != 0;
            const int ploidy = static_cast<int>(pm & 0x3Fu);
            if (ploidy != 2)
                g.fail("sample " + std::to_string(i) + " ploidy " + std::to_string(ploidy) +
                       " != 2 (diploid only in v1)");
            // Always consume this sample's 2 probability fields (missing samples still
            // occupy their bit-slots, set to 0).
            const std::uint32_t r0 = br.read(B);
            const std::uint32_t r1 = br.read(B);
            const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(M) +
                                    static_cast<std::size_t>(v);
            if (missing) {
                tile.dosage[idx] = kMissing;
                continue;
            }
            const double v0 = static_cast<double>(r0) / denom;
            const double v1 = static_cast<double>(r1) / denom;
            // ALT dosage: unphased v0=P(AA),v1=P(AB) -> 2 - 2*v0 - v1; phased v0,v1=P(REF)
            // on the two haps -> 2 - v0 - v1.
            const double dose = (phased != 0) ? (2.0 - v0 - v1) : (2.0 - 2.0 * v0 - v1);
            tile.dosage[idx] = static_cast<float>(dose);
        }

        tile.snps.push_back(std::move(meta));
    }

    return tile;
}

}  // namespace steppe::io
