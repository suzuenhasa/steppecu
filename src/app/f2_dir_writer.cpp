// src/app/f2_dir_writer.cpp
//
// Writer for the f2_blocks directory format (STPF2BK1) — the inverse of read_f2_dir.
// Plain C++20, no CUDA: it pulls in the CUDA-free f2_disk_format.hpp only for the shared
// on-disk header struct and magic/version/dtype stamps, so writer and reader cannot drift.
// A self-contained SHA-256 (below) content-addresses the output without a crypto dependency.
#include "app/f2_dir_writer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

// x86 SHA-NI: hardware SHA-256 rounds via the x86 SHA-extension intrinsics (a compiler
// intrinsic header, not a new dependency). Runtime-dispatched against
// __builtin_cpu_supports("sha") with the portable scalar transform as the fallback, so
// this compiles and runs correctly on any CPU. Only the SHA-NI function is built for the
// sha/sse4.1 ISA (via a target attribute); the rest of the file stays at the default ISA.
// Non-x86 arches use the scalar path only.
#if defined(__x86_64__) || defined(__i386__)
#  define STEPPE_SHA_HAS_X86 1
#  include <immintrin.h>
#else
#  define STEPPE_SHA_HAS_X86 0
#endif

#include "device/f2_disk_format.hpp"  // F2DiskHeader, kF2DiskMagic/Version/DtypeFp64, kF2DiskHeaderSize (CUDA-FREE)

namespace steppe::app {

namespace {

// ---------------------------------------------------------------------------
// A small, self-contained SHA-256 (FIPS 180-4) — no external crypto dependency.
// Computes the f2_cache_id (sha256 of f2.bin) and the dataset content shas recorded in
// meta.json, so the output is content-addressed and its shas match the source metadata.
// `update` streams, so a multi-GB .geno is hashed in chunks without a full-file buffer.
//
// The streaming update compresses whole 64-byte blocks directly from the caller's buffer,
// staging only the unaligned head/tail remainder — a per-byte copy into a staging buffer
// was the hashing bottleneck. The digest is standard (FIPS 180-4 / sha256sum-compatible).
//
// On a CPU with the SHA extension the per-block round is a hardware instruction, several
// times faster than the scalar loop; sha256_blocks() dispatches to it, else the scalar path.
// ---------------------------------------------------------------------------

// The SHA-256 round constants (FIPS 180-4) — shared by the scalar and SHA-NI block
// functions (SHA-NI loads them into vector lanes, the scalar path indexes them directly).
// Single home so the two implementations cannot drift.
alignas(64) inline constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

inline std::uint32_t sha_rotr(std::uint32_t x, std::uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

// Scalar SHA-256 compression of `n` consecutive 64-byte blocks into state[8] (the
// portable fallback). state is updated in place (Davies-Meyer feed-forward).
inline void sha256_blocks_scalar(std::uint32_t state[8], const std::uint8_t* p, std::size_t n) {
    for (std::size_t blk = 0; blk < n; ++blk, p += 64) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(p[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(p[i * 4 + 2]) << 8) |
                   (static_cast<std::uint32_t>(p[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = sha_rotr(w[i - 15], 7) ^ sha_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = sha_rotr(w[i - 2], 17) ^ sha_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        std::uint32_t e = state[4], f = state[5], g = state[6], hh = state[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + S1 + ch + kSha256K[i] + w[i];
            const std::uint32_t S0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += hh;
    }
}

#if STEPPE_SHA_HAS_X86
// Hardware SHA-256 compression via the x86 SHA-extension intrinsics (the fast path).
// The target attribute scopes the SHA-NI instructions to this function alone; call it
// ONLY after __builtin_cpu_supports("sha") confirms the CPU has them. state is the same
// big-endian-loaded {a..h} the scalar path maintains, so the two are bit-identical.
__attribute__((target("sha,sse4.1,ssse3")))
inline void sha256_blocks_shani(std::uint32_t state[8], const std::uint8_t* data, std::size_t n) {
    const __m128i shuf = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    // Load state into the SHA-NI register layout (ABEF / CDGH), once.
    __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));  // DCBA
    __m128i st1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));  // HGFE
    tmp = _mm_shuffle_epi32(tmp, 0xB1);          // CDAB
    st1 = _mm_shuffle_epi32(st1, 0x1B);          // EFGH
    __m128i abef = _mm_alignr_epi8(tmp, st1, 8); // ABEF
    __m128i cdgh = _mm_blend_epi16(st1, tmp, 0xF0); // CDGH

    for (std::size_t blk = 0; blk < n; ++blk, data += 64) {
        const __m128i abef0 = abef;
        const __m128i cdgh0 = cdgh;
        __m128i m0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 0)),  shuf);
        __m128i m1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16)), shuf);
        __m128i m2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 32)), shuf);
        __m128i m3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 48)), shuf);
        __m128i msg;

        // Rounds 0-3
        msg = _mm_add_epi32(m0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[0])));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
        msg  = _mm_shuffle_epi32(msg, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);
        // Rounds 4-7
        msg = _mm_add_epi32(m1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[4])));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
        msg  = _mm_shuffle_epi32(msg, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);
        m0 = _mm_sha256msg1_epu32(m0, m1);
        // Rounds 8-11
        msg = _mm_add_epi32(m2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[8])));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
        msg  = _mm_shuffle_epi32(msg, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);
        m1 = _mm_sha256msg1_epu32(m1, m2);
        // Rounds 12-15
        msg = _mm_add_epi32(m3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[12])));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
        tmp  = _mm_alignr_epi8(m3, m2, 4);
        m0   = _mm_add_epi32(m0, tmp);
        m0   = _mm_sha256msg2_epu32(m0, m3);
        msg  = _mm_shuffle_epi32(msg, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);
        m2   = _mm_sha256msg1_epu32(m2, m3);

        // Rounds 16-59: 11 schedule+round groups (m0..m3 rotate). A local macro, NOT a
        // lambda — a lambda's call operator does not inherit this function's target("sha")
        // attribute, so the SHA intrinsics inside it would fail to compile/inline.
#define STEPPE_SHA_RND_GROUP(ma, mb, md, kidx)                                              \
        do {                                                                                \
            msg = _mm_add_epi32((ma), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[kidx]))); \
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);                                   \
            tmp  = _mm_alignr_epi8((ma), (md), 4);                                           \
            (mb) = _mm_add_epi32((mb), tmp);                                                 \
            (mb) = _mm_sha256msg2_epu32((mb), (ma));                                         \
            msg  = _mm_shuffle_epi32(msg, 0x0E);                                             \
            abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);                                   \
            (md) = _mm_sha256msg1_epu32((md), (ma));                                         \
        } while (0)
        STEPPE_SHA_RND_GROUP(m0, m1, m3, 16);  // 16-19
        STEPPE_SHA_RND_GROUP(m1, m2, m0, 20);  // 20-23
        STEPPE_SHA_RND_GROUP(m2, m3, m1, 24);  // 24-27
        STEPPE_SHA_RND_GROUP(m3, m0, m2, 28);  // 28-31
        STEPPE_SHA_RND_GROUP(m0, m1, m3, 32);  // 32-35
        STEPPE_SHA_RND_GROUP(m1, m2, m0, 36);  // 36-39
        STEPPE_SHA_RND_GROUP(m2, m3, m1, 40);  // 40-43
        STEPPE_SHA_RND_GROUP(m3, m0, m2, 44);  // 44-47
        STEPPE_SHA_RND_GROUP(m0, m1, m3, 48);  // 48-51
#undef STEPPE_SHA_RND_GROUP

        // Rounds 52-55 (msg2 on m1, no msg1 for the tail)
        msg = _mm_add_epi32(m1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[52])));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
        tmp  = _mm_alignr_epi8(m1, m0, 4);
        m2   = _mm_add_epi32(m2, tmp);
        m2   = _mm_sha256msg2_epu32(m2, m1);
        msg  = _mm_shuffle_epi32(msg, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);
        // Rounds 56-59
        msg = _mm_add_epi32(m2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[56])));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
        tmp  = _mm_alignr_epi8(m2, m1, 4);
        m3   = _mm_add_epi32(m3, tmp);
        m3   = _mm_sha256msg2_epu32(m3, m2);
        msg  = _mm_shuffle_epi32(msg, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);
        // Rounds 60-63
        msg = _mm_add_epi32(m3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(&kSha256K[60])));
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, msg);
        msg  = _mm_shuffle_epi32(msg, 0x0E);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, msg);

        // Davies-Meyer feed-forward.
        abef = _mm_add_epi32(abef, abef0);
        cdgh = _mm_add_epi32(cdgh, cdgh0);
    }

    // Store ABEF/CDGH back to the plain {a..h} layout.
    tmp = _mm_shuffle_epi32(abef, 0x1B);          // FEBA
    st1 = _mm_shuffle_epi32(cdgh, 0xB1);          // DCHG
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), _mm_blend_epi16(tmp, st1, 0xF0)); // DCBA
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), _mm_alignr_epi8(st1, tmp, 8));     // HGFE
}
#endif  // STEPPE_SHA_HAS_X86

// Dispatch block compression to SHA-NI when the CPU supports it, else scalar. The
// capability check is a cheap cpuid, cached across calls.
inline void sha256_blocks(std::uint32_t state[8], const std::uint8_t* p, std::size_t n) {
#if STEPPE_SHA_HAS_X86
    static const bool has_sha = (__builtin_cpu_supports("sha") != 0);
    if (has_sha) { sha256_blocks_shani(state, p, n); return; }
#endif
    sha256_blocks_scalar(state, p, n);
}

class Sha256 {
public:
    Sha256() { reset(); }

    void update(const std::uint8_t* data, std::size_t len) {
        bit_len_ += static_cast<std::uint64_t>(len) * 8u;
        // 1. Top up a partially-filled staging buffer to a full 64-byte block.
        if (buf_len_ != 0) {
            const std::size_t need = 64 - buf_len_;
            const std::size_t take = (len < need) ? len : need;
            std::memcpy(buf_.data() + buf_len_, data, take);
            buf_len_ += take;
            data += take;
            len -= take;
            if (buf_len_ == 64) { sha256_blocks(h_.data(), buf_.data(), 1); buf_len_ = 0; }
        }
        // 2. Compress all whole 64-byte blocks straight from the caller's buffer in one
        //    call — no per-byte copy into buf_, and the block loop keeps its state in
        //    registers across the whole run.
        const std::size_t nblk = len / 64;
        if (nblk != 0) {
            sha256_blocks(h_.data(), data, nblk);
            data += nblk * 64;
            len -= nblk * 64;
        }
        // 3. Stage the unaligned tail remainder (< 64 bytes) for the next update/final.
        if (len != 0) {
            std::memcpy(buf_.data(), data, len);
            buf_len_ = len;
        }
    }
    void update(const char* data, std::size_t len) {
        update(reinterpret_cast<const std::uint8_t*>(data), len);
    }

    /// Finalize and return the lowercase hex digest (64 chars).
    [[nodiscard]] std::string hex() {
        std::array<std::uint8_t, 32> digest{};
        final_digest(digest.data());
        static const char* kHex = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (std::uint8_t b : digest) {
            out.push_back(kHex[b >> 4]);
            out.push_back(kHex[b & 0x0f]);
        }
        return out;
    }

private:
    std::array<std::uint8_t, 64> buf_{};
    std::size_t buf_len_ = 0;
    std::uint64_t bit_len_ = 0;
    std::array<std::uint32_t, 8> h_{};

    void reset() {
        buf_len_ = 0;
        bit_len_ = 0;
        h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    }

    void final_digest(std::uint8_t* out) {
        // bit_len_ already holds the total message bit length; capture it before staging the
        // padding, which is fed through sha256_blocks() directly and so does not update it.
        const std::uint64_t total_bits = bit_len_;
        // SHA-256 padding staged in buf_: append 0x80, zero-pad to 56 mod 64, then the
        // 64-bit big-endian length, flushing a full block whenever buf_ fills.
        buf_[buf_len_++] = 0x80;
        if (buf_len_ > 56) {
            while (buf_len_ < 64) buf_[buf_len_++] = 0x00;
            sha256_blocks(h_.data(), buf_.data(), 1);
            buf_len_ = 0;
        }
        while (buf_len_ < 56) buf_[buf_len_++] = 0x00;
        for (int i = 0; i < 8; ++i) {
            buf_[56 + i] = static_cast<std::uint8_t>((total_bits >> (56 - 8 * i)) & 0xff);
        }
        sha256_blocks(h_.data(), buf_.data(), 1);
        buf_len_ = 0;
        for (int i = 0; i < 8; ++i) {
            out[i * 4] = static_cast<std::uint8_t>((h_[i] >> 24) & 0xff);
            out[i * 4 + 1] = static_cast<std::uint8_t>((h_[i] >> 16) & 0xff);
            out[i * 4 + 2] = static_cast<std::uint8_t>((h_[i] >> 8) & 0xff);
            out[i * 4 + 3] = static_cast<std::uint8_t>(h_[i] & 0xff);
        }
    }
};

[[nodiscard]] F2DirWriteResult fail(Status status, std::string reason) {
    F2DirWriteResult r;
    r.ok = false;
    r.status = status;
    r.error = std::move(reason);
    return r;
}

// Minimal JSON string escape for the meta.json values (paths/labels are tame, but a
// label could contain a backslash/quote; keep the JSON well-formed regardless).
[[nodiscard]] std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;       break;
        }
    }
    return out;
}

[[nodiscard]] std::string json_str(const std::string& s) { return "\"" + json_escape(s) + "\""; }

[[nodiscard]] std::string bool_str(bool b) { return b ? "true" : "false"; }

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    Sha256 sha;
    // An 8 MiB read window: large enough to amortize the syscall/iostream overhead so
    // the throughput is bounded by the (now bulk) compression, not the read loop. The
    // bytes are compressed straight from this buffer (the bulk update() path) — no
    // second per-byte copy — and the file is never fully buffered (the .geno is ~6.7 GB).
    std::vector<char> chunk(8u << 20);
    while (f) {
        f.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = f.gcount();
        if (got > 0) sha.update(chunk.data(), static_cast<std::size_t>(got));
    }
    return sha.hex();
}

F2DirWriteResult write_f2_dir(const std::filesystem::path& dir,
                              const F2BlockTensor& f2,
                              const std::vector<std::string>& pop_labels,
                              const F2DirMeta& meta) {
    namespace fs = std::filesystem;

    // ---- Shape validation (fail-fast on a wiring/data bug, not a wrong file) ------
    if (f2.P <= 0 || f2.n_block <= 0) {
        return fail(Status::InvalidConfig,
                    "write_f2_dir: degenerate f2 tensor (P=" + std::to_string(f2.P) +
                        " n_block=" + std::to_string(f2.n_block) + ")");
    }
    if (static_cast<int>(pop_labels.size()) != f2.P) {
        return fail(Status::InvalidConfig,
                    "write_f2_dir: pop_labels has " + std::to_string(pop_labels.size()) +
                        " entries but f2 tensor P=" + std::to_string(f2.P) +
                        " (the name<->index map must cover the whole P axis)");
    }
    const std::size_t slab_elems = f2.size();  // P*P*n_block
    if (f2.f2.size() != slab_elems || f2.vpair.size() != slab_elems) {
        return fail(Status::InvalidConfig,
                    "write_f2_dir: f2/vpair length mismatch (f2=" + std::to_string(f2.f2.size()) +
                        " vpair=" + std::to_string(f2.vpair.size()) + " expected=" +
                        std::to_string(slab_elems) + ")");
    }
    if (static_cast<int>(f2.block_sizes.size()) != f2.n_block) {
        return fail(Status::InvalidConfig,
                    "write_f2_dir: block_sizes has " + std::to_string(f2.block_sizes.size()) +
                        " entries but n_block=" + std::to_string(f2.n_block));
    }

    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        return fail(Status::InvalidConfig,
                    "write_f2_dir: cannot create --out dir '" + dir.string() + "': " + ec.message());
    }

    // ---- f2.bin (STPF2BK1): header | f2 | REAL vpair | block_sizes int32 ----------
    const std::uint64_t P = static_cast<std::uint64_t>(f2.P);
    const std::uint64_t nb = static_cast<std::uint64_t>(f2.n_block);
    const std::uint64_t slab_bytes = P * P * nb * sizeof(double);

    device::F2DiskHeader hdr{};
    std::memcpy(hdr.magic, device::kF2DiskMagic, sizeof(hdr.magic));
    hdr.version = device::kF2DiskVersion;
    hdr.dtype = device::kF2DiskDtypeFp64;
    hdr.P = f2.P;
    hdr.n_block = f2.n_block;
    hdr.f2_offset = device::kF2DiskHeaderSize;            // == 64
    hdr.vpair_offset = hdr.f2_offset + slab_bytes;
    hdr.block_sizes_offset = hdr.vpair_offset + slab_bytes;

    const fs::path bin_path = dir / "f2.bin";
    {
        std::ofstream o(bin_path, std::ios::binary | std::ios::trunc);
        if (!o) return fail(Status::InvalidConfig, "write_f2_dir: cannot open f2.bin for write: " + bin_path.string());
        o.write(reinterpret_cast<const char*>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
        o.write(reinterpret_cast<const char*>(f2.f2.data()), static_cast<std::streamsize>(slab_bytes));
        // Real vpair: the per-block pairwise-valid SNP counts, which the downstream NA path
        // reads as `vpair == 0` to detect a dropped pair block — must be true counts, not zeros.
        o.write(reinterpret_cast<const char*>(f2.vpair.data()), static_cast<std::streamsize>(slab_bytes));
        // block_sizes int32 trailer.
        std::vector<std::int32_t> sizes32(f2.block_sizes.begin(), f2.block_sizes.end());
        o.write(reinterpret_cast<const char*>(sizes32.data()),
                static_cast<std::streamsize>(sizeof(std::int32_t) * nb));
        if (!o) return fail(Status::InvalidConfig, "write_f2_dir: f2.bin write failed (disk full?): " + bin_path.string());
    }

    // f2_cache_id = sha256 of the complete f2.bin (content-addressed). Always computed:
    // f2.bin is small (P*P*nb*16 bytes), not the multi-GiB source .geno, so its hash is cheap.
    const std::string cache_hex = sha256_file(bin_path);
    const std::string cache_id = cache_hex.empty() ? std::string{} : ("sha256:" + cache_hex);

    // Dataset content shas, so the directory is reproducible and its shas match the source
    // metadata. A source file is hashed only when requested (hash_source_files) and the
    // caller did not pre-fill the sha. The default skips the source-.geno sha: it is a
    // multi-second whole-file read+compress that produces only a provenance value. When
    // requested, the large geno sha is normally pre-filled by the caller on a background
    // thread (overlapping the GPU pipeline), leaving only the small snp/ind shas here.
    // meta is const, so hold the three shas as locals seeded from any pre-filled value.
    std::string geno_sha = meta.geno_sha256;
    std::string snp_sha  = meta.snp_sha256;
    std::string ind_sha  = meta.ind_sha256;
    if (meta.hash_source_files) {
        if (geno_sha.empty() && !meta.geno_path.empty()) geno_sha = sha256_file(meta.geno_path);
        if (snp_sha.empty()  && !meta.snp_path.empty())  snp_sha  = sha256_file(meta.snp_path);
        if (ind_sha.empty()  && !meta.ind_path.empty())  ind_sha  = sha256_file(meta.ind_path);
    }

    // ---- pops.txt: the P labels, one per line, in P-axis index order --------------
    {
        std::ofstream pf(dir / "pops.txt", std::ios::trunc);
        if (!pf) return fail(Status::InvalidConfig, "write_f2_dir: cannot write pops.txt");
        for (const std::string& s : pop_labels) pf << s << "\n";
        if (!pf) return fail(Status::InvalidConfig, "write_f2_dir: pops.txt write failed");
    }

    // pops_sha256 = sha256 of the exact pops.txt bytes just written (the name<->index map).
    // A swapped or corrupted pops.txt silently reassigns every name to a different P-axis
    // index and changes every downstream result undetectably; stamping its hash lets a
    // reader re-hash the file to catch that. pops.txt is tiny, so this hash is cheap.
    const std::string pops_hex = sha256_file(dir / "pops.txt");
    const std::string pops_sha = pops_hex.empty() ? std::string{} : ("sha256:" + pops_hex);

    // ---- meta.json: provenance / reproducibility sidecar --------------------------
    {
        std::ostringstream js;
        js << "{\n";
        js << "  \"format\": \"STPF2BK1\",\n";
        // meta_schema_version = the meta.json sidecar schema version, distinct from f2.bin's
        // binary version (kF2DiskVersion in F2DiskHeader): a consumer keys off this for the
        // provenance field set, not the numeric payload bytes.
        js << "  \"meta_schema_version\": " << kF2MetaSchemaVersion << ",\n";
        js << "  \"steppe_version\": " << json_str(meta.steppe_version) << ",\n";
        js << "  \"P\": " << f2.P << ",\n";
        js << "  \"n_block\": " << f2.n_block << ",\n";
        js << "  \"precision_tag\": " << json_str(meta.precision_tag) << ",\n";
        js << "  \"precision_mantissa_bits\": " << meta.precision_mantissa_bits << ",\n";
        // blgsize_cm serializes at the default ostringstream precision (~6 sig-figs). It is
        // a coarse binning knob, not a parity-load-bearing value, so no setprecision is applied.
        js << "  \"blgsize_cm\": " << meta.blgsize_cm << ",\n";
        js << "  \"n_snp_total\": " << meta.n_snp_total << ",\n";
        js << "  \"n_snp_kept\": " << meta.n_snp_kept << ",\n";
        js << "  \"filters\": {\n";
        js << "    \"maf_min\": " << meta.maf_min << ",\n";
        js << "    \"maxmiss\": " << meta.geno_max_missing << ",\n";
        js << "    \"mind_max_missing\": " << meta.mind_max_missing << ",\n";
        js << "    \"autosomes_only\": " << bool_str(meta.autosomes_only) << ",\n";
        js << "    \"drop_monomorphic\": " << bool_str(meta.drop_monomorphic) << ",\n";
        js << "    \"transversions_only\": " << bool_str(meta.transversions_only) << "\n";
        js << "  },\n";
        js << "  \"pop_selection\": " << json_str(meta.pop_selection) << ",\n";
        js << "  \"source\": {\n";
        js << "    \"geno\": " << json_str(meta.geno_path) << ",\n";
        js << "    \"snp\": " << json_str(meta.snp_path) << ",\n";
        js << "    \"ind\": " << json_str(meta.ind_path) << ",\n";
        // source_hash_computed marks whether the source-dataset shas were computed or
        // deliberately skipped (the default). When false the *_sha256 fields are "" by
        // design, so a consumer knows the absence is intentional, not a failed hash.
        js << "    \"source_hash_computed\": " << bool_str(meta.hash_source_files) << ",\n";
        js << "    \"geno_sha256\": " << json_str(geno_sha) << ",\n";
        js << "    \"snp_sha256\": " << json_str(snp_sha) << ",\n";
        js << "    \"ind_sha256\": " << json_str(ind_sha) << "\n";
        js << "  },\n";
        // pops_sha256: sha256 of pops.txt, "sha256:<hex>" or "" if the just-written file
        // could not be re-read. Adjacent to f2_cache_id — both content-address a sidecar
        // the writer controls, letting a reader detect a swap/corruption by re-hashing.
        js << "  \"pops_sha256\": " << json_str(pops_sha) << ",\n";
        js << "  \"f2_cache_id\": " << json_str(cache_id) << "\n";
        js << "}\n";

        std::ofstream mf(dir / "meta.json", std::ios::trunc);
        if (!mf) return fail(Status::InvalidConfig, "write_f2_dir: cannot write meta.json");
        mf << js.str();
        if (!mf) return fail(Status::InvalidConfig, "write_f2_dir: meta.json write failed");
    }

    F2DirWriteResult ret;
    ret.ok = true;
    ret.status = Status::Ok;
    ret.f2_cache_id = cache_id;
    return ret;
}

}  // namespace steppe::app
