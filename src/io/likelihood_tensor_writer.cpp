// src/io/likelihood_tensor_writer.cpp
//
// STPGL1 GL-tensor artifact writer. Assembles the four sections in memory to fix
// their byte offsets, then writes a 64-byte header carrying those offsets followed
// by the sections — so the file is self-describing and every section is seekable.
// Little-endian (steppe targets LE; matches the other binary writers).
#include "io/likelihood_tensor_writer.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace steppe::io {

namespace {

// Little-endian append helpers (portable regardless of host endianness).
void put_u32(std::vector<char>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<char>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void put_i32(std::vector<char>& b, std::int32_t v) { put_u32(b, static_cast<std::uint32_t>(v)); }
void put_i64(std::vector<char>& b, std::int64_t v) { put_u64(b, static_cast<std::uint64_t>(v)); }
void put_str(std::vector<char>& b, const std::string& s) {
    put_u32(b, static_cast<std::uint32_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
}
void put_f64(std::vector<char>& b, double v) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));  // bit-cast the IEEE-754 double
    put_u64(b, bits);
}

}  // namespace

void write_likelihood_tensor(const std::string& path, const LikelihoodTile& tile) {
    const std::uint64_t n_site = tile.n_site;
    const std::uint64_t n_sample = tile.n_sample;
    const std::uint64_t cells = n_site * n_sample;

    if (tile.sample_ids.size() != n_sample)
        throw std::runtime_error("write_likelihood_tensor: sample_ids size != n_sample");
    if (tile.sites.size() != n_site)
        throw std::runtime_error("write_likelihood_tensor: sites size != n_site");
    if (tile.l.size() != cells * 3)
        throw std::runtime_error("write_likelihood_tensor: payload size != n_site*n_sample*3");
    if (tile.present.size() != cells)
        throw std::runtime_error("write_likelihood_tensor: present size != n_site*n_sample");

    // --- assemble the four sections ------------------------------------------
    std::vector<char> samples;
    for (const std::string& s : tile.sample_ids) put_str(samples, s);

    std::vector<char> sites;
    for (const LikelihoodSite& s : tile.sites) {
        put_str(sites, s.rsid);
        put_i32(sites, s.chrom);
        put_i64(sites, s.pos37);
        put_i64(sites, s.pos38);
        sites.push_back(s.a1);
        sites.push_back(s.a2);
    }

    std::vector<char> payload;
    payload.reserve(tile.l.size() * 8);
    for (const double v : tile.l) put_f64(payload, v);

    std::vector<char> present;
    present.reserve(tile.present.size());
    for (const std::uint8_t v : tile.present) present.push_back(static_cast<char>(v));

    // --- fix section offsets --------------------------------------------------
    constexpr std::uint64_t kHeader = 64;
    const std::uint64_t off_samples = kHeader;
    const std::uint64_t off_sites = off_samples + samples.size();
    const std::uint64_t off_payload = off_sites + sites.size();
    const std::uint64_t off_present = off_payload + payload.size();

    // --- header ---------------------------------------------------------------
    std::vector<char> header;
    header.reserve(kHeader);
    const char magic[8] = {'S', 'T', 'P', 'G', 'L', '1', '\0', '\0'};
    header.insert(header.end(), magic, magic + 8);
    put_u32(header, 1);  // version
    header.push_back(0);                                          // layout = site-major
    header.push_back(static_cast<char>(static_cast<std::uint8_t>(tile.field)));  // field
    header.push_back(0);                                          // dtype = FP64
    header.push_back(0);                                          // reserved
    put_u64(header, n_site);
    put_u64(header, n_sample);
    put_u64(header, off_samples);
    put_u64(header, off_sites);
    put_u64(header, off_payload);
    put_u64(header, off_present);
    if (header.size() != kHeader)
        throw std::runtime_error("write_likelihood_tensor: header size mismatch (internal)");

    // --- write ----------------------------------------------------------------
    std::ofstream o(path, std::ios::binary | std::ios::trunc);
    if (!o) throw std::runtime_error("write_likelihood_tensor: cannot open output: " + path);
    o.write(header.data(), static_cast<std::streamsize>(header.size()));
    o.write(samples.data(), static_cast<std::streamsize>(samples.size()));
    o.write(sites.data(), static_cast<std::streamsize>(sites.size()));
    o.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    o.write(present.data(), static_cast<std::streamsize>(present.size()));
    if (!o) throw std::runtime_error("write_likelihood_tensor: write failed: " + path);
}

}  // namespace steppe::io
