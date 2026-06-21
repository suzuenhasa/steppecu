// src/app/f2_dir_writer.cpp
//
// The f2_blocks DIRECTORY writer (cli-bindings.md §4.3) — the inverse of read_f2_dir.
// PLAIN C++20, app-only, NO CUDA header (the §4 layering / arch-grep gate); it reaches
// the CUDA-FREE f2_disk_format.hpp only for the on-disk F2DiskHeader struct + the
// magic/version/dtype stamps (the single home so the writer cannot drift from the
// reader). A self-contained SHA-256 (below) computes the f2_cache_id and the dataset
// content shas without pulling a crypto dependency into the app subtree.
#include "app/f2_dir_writer.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include "device/f2_disk_format.hpp"  // F2DiskHeader, kF2DiskMagic/Version/DtypeFp64, kF2DiskHeaderSize (CUDA-FREE)

namespace steppe::app {

namespace {

// ---------------------------------------------------------------------------
// A small, self-contained SHA-256 (FIPS 180-4) — no external crypto dependency
// in the app subtree. Used for the f2_cache_id (sha256 of f2.bin) and the dataset
// content shas recorded in meta.json (so a steppe run is content-addressed and the
// shas match the golden metadata's geno/snp/ind sha256). Streaming `update` so the
// 4 GB .geno is hashed in chunks without a full-file buffer.
// ---------------------------------------------------------------------------
class Sha256 {
public:
    Sha256() { reset(); }

    void update(const std::uint8_t* data, std::size_t len) {
        for (std::size_t i = 0; i < len; ++i) {
            buf_[buf_len_++] = data[i];
            if (buf_len_ == 64) { transform(buf_.data()); bit_len_ += 512; buf_len_ = 0; }
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

    static std::uint32_t rotr(std::uint32_t x, std::uint32_t n) {
        return (x >> n) | (x << (32u - n));
    }

    void transform(const std::uint8_t* p) {
        static const std::uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(p[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(p[i * 4 + 2]) << 8) |
                   (static_cast<std::uint32_t>(p[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        std::uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
    }

    void final_digest(std::uint8_t* out) {
        const std::uint64_t total_bits = bit_len_ + static_cast<std::uint64_t>(buf_len_) * 8u;
        // append 0x80, then zero-pad to 56 mod 64, then the 64-bit big-endian length.
        std::uint8_t pad = 0x80;
        update(&pad, 1);
        std::uint8_t zero = 0x00;
        while (buf_len_ != 56) update(&zero, 1);
        std::uint8_t lenbe[8];
        for (int i = 0; i < 8; ++i) lenbe[i] = static_cast<std::uint8_t>((total_bits >> (56 - 8 * i)) & 0xff);
        update(lenbe, 8);
        for (int i = 0; i < 8; ++i) {
            out[i * 4] = static_cast<std::uint8_t>((h_[i] >> 24) & 0xff);
            out[i * 4 + 1] = static_cast<std::uint8_t>((h_[i] >> 16) & 0xff);
            out[i * 4 + 2] = static_cast<std::uint8_t>((h_[i] >> 8) & 0xff);
            out[i * 4 + 3] = static_cast<std::uint8_t>(h_[i] & 0xff);
        }
    }
};

[[nodiscard]] std::string sha256_of_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    Sha256 sha;
    std::vector<char> chunk(1u << 20);  // 1 MiB streaming window (the .geno is ~4 GB)
    while (f) {
        f.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = f.gcount();
        if (got > 0) sha.update(chunk.data(), static_cast<std::size_t>(got));
    }
    return sha.hex();
}

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
        // REAL vpair (cli-bindings.md §4.3): the per-block pairwise-valid SNP counts the
        // F1 NA path reads as `vpair == 0` to detect a dropped pair block — NOT zeros.
        o.write(reinterpret_cast<const char*>(f2.vpair.data()), static_cast<std::streamsize>(slab_bytes));
        // block_sizes int32 trailer.
        std::vector<std::int32_t> sizes32(f2.block_sizes.begin(), f2.block_sizes.end());
        o.write(reinterpret_cast<const char*>(sizes32.data()),
                static_cast<std::streamsize>(sizeof(std::int32_t) * nb));
        if (!o) return fail(Status::InvalidConfig, "write_f2_dir: f2.bin write failed (disk full?): " + bin_path.string());
    }

    // f2_cache_id = sha256 of the COMPLETE f2.bin (content-addressed; cli-bindings §4.3).
    const std::string cache_hex = sha256_of_file(bin_path);
    const std::string cache_id = cache_hex.empty() ? std::string{} : ("sha256:" + cache_hex);

    // Dataset content shas (so the dir is reproducible + the shas match the golden
    // metadata's geno/snp/ind sha256). The writer is the single SHA home; it hashes
    // the source files when a path is given and the caller did not pre-fill the sha.
    F2DirMeta m = meta;  // local copy so we can fill the dataset shas (meta is const).
    if (m.geno_sha256.empty() && !m.geno_path.empty()) m.geno_sha256 = sha256_of_file(m.geno_path);
    if (m.snp_sha256.empty()  && !m.snp_path.empty())  m.snp_sha256  = sha256_of_file(m.snp_path);
    if (m.ind_sha256.empty()  && !m.ind_path.empty())  m.ind_sha256  = sha256_of_file(m.ind_path);

    // ---- pops.txt: the P labels, one per line, in P-axis index order --------------
    {
        std::ofstream pf(dir / "pops.txt", std::ios::trunc);
        if (!pf) return fail(Status::InvalidConfig, "write_f2_dir: cannot write pops.txt");
        for (const std::string& s : pop_labels) pf << s << "\n";
        if (!pf) return fail(Status::InvalidConfig, "write_f2_dir: pops.txt write failed");
    }

    // ---- meta.json: provenance (architecture.md §12 reproducibility block) --------
    {
        std::ostringstream js;
        js << "{\n";
        js << "  \"format\": \"STPF2BK1\",\n";
        js << "  \"steppe_version\": " << json_str(m.steppe_version) << ",\n";
        js << "  \"P\": " << f2.P << ",\n";
        js << "  \"n_block\": " << f2.n_block << ",\n";
        js << "  \"precision_tag\": " << json_str(m.precision_tag) << ",\n";
        js << "  \"precision_mantissa_bits\": " << m.precision_mantissa_bits << ",\n";
        js << "  \"blgsize_cm\": " << m.blgsize_cm << ",\n";
        js << "  \"n_snp_total\": " << m.n_snp_total << ",\n";
        js << "  \"n_snp_kept\": " << m.n_snp_kept << ",\n";
        js << "  \"filters\": {\n";
        js << "    \"maf_min\": " << m.maf_min << ",\n";
        js << "    \"maxmiss\": " << m.geno_max_missing << ",\n";
        js << "    \"mind_max_missing\": " << m.mind_max_missing << ",\n";
        js << "    \"autosomes_only\": " << bool_str(m.autosomes_only) << ",\n";
        js << "    \"drop_monomorphic\": " << bool_str(m.drop_monomorphic) << ",\n";
        js << "    \"transversions_only\": " << bool_str(m.transversions_only) << "\n";
        js << "  },\n";
        js << "  \"pop_selection\": " << json_str(m.pop_selection) << ",\n";
        js << "  \"source\": {\n";
        js << "    \"geno\": " << json_str(m.geno_path) << ",\n";
        js << "    \"snp\": " << json_str(m.snp_path) << ",\n";
        js << "    \"ind\": " << json_str(m.ind_path) << ",\n";
        js << "    \"geno_sha256\": " << json_str(m.geno_sha256) << ",\n";
        js << "    \"snp_sha256\": " << json_str(m.snp_sha256) << ",\n";
        js << "    \"ind_sha256\": " << json_str(m.ind_sha256) << "\n";
        js << "  },\n";
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
