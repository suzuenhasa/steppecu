// src/app/cmd_cache.cpp
//
// Native host-only f2-cache inspection (steppe cache ls/show/verify). Reads the
// 64-byte STPF2BK1 header directly (never the multi-GB payload), re-uses the
// project's sha256_file for integrity, and pulls the stored content-address out
// of meta.json with a small regex rather than a JSON parser. No GPU, no config.
//
// Reference: docs/reference/src_app_cmd_cache.cpp.md
#include "app/cmd_cache.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <string>

#include "app/f2_dir_writer.hpp"       // sha256_file
#include "core/config/exit_code.hpp"   // config::kExit*
#include "device/f2_disk_format.hpp"   // F2DiskHeader, kF2DiskMagic, kF2DiskHeaderSize

namespace steppe::app {

namespace {

namespace fs = std::filesystem;
namespace cfg = steppe::config;
using steppe::device::F2DiskHeader;
using steppe::device::kF2DiskHeaderSize;

[[nodiscard]] bool read_header(const fs::path& dir, F2DiskHeader& h) {
    std::ifstream f(dir / "f2.bin", std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&h), static_cast<std::streamsize>(kF2DiskHeaderSize));
    if (f.gcount() != static_cast<std::streamsize>(kF2DiskHeaderSize)) return false;
    // Magic + dtype, mirroring the Python _read_header_only guard. dtype gates the
    // 8-byte-element size arithmetic below; a non-FP64 STPF2BK1 is not one we read.
    // (version is intentionally NOT checked — the Python reader doesn't either.)
    return std::memcmp(h.magic, steppe::device::kF2DiskMagic, 8) == 0 &&
           h.dtype == steppe::device::kF2DiskDtypeFp64;
}

[[nodiscard]] std::uint64_t expected_f2_bytes(int P, int n_block) {
    const std::uint64_t p = static_cast<std::uint64_t>(P);
    const std::uint64_t nb = static_cast<std::uint64_t>(n_block);
    return kF2DiskHeaderSize + 2 * p * p * nb * 8 + 4 * nb;
}

[[nodiscard]] std::uint64_t dir_bytes(const fs::path& dir) {
    std::uint64_t total = 0;
    for (const char* name : {"f2.bin", "pops.txt", "meta.json"}) {
        std::error_code ec;
        const auto sz = fs::file_size(dir / name, ec);
        if (!ec) total += sz;
    }
    return total;
}

[[nodiscard]] std::string human_size(std::uint64_t n) {
    static const char* units[] = {"B", "K", "M", "G", "T"};
    double s = static_cast<double>(n);
    int u = 0;
    while (s >= 1024.0 && u < 4) { s /= 1024.0; ++u; }
    char buf[32];
    if (u == 0) std::snprintf(buf, sizeof(buf), "%.0fB", s);
    else std::snprintf(buf, sizeof(buf), "%.1f%s", s, units[u]);
    return buf;
}

[[nodiscard]] std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// Pull the raw quoted value of "<key>" out of meta.json without a JSON parser.
// We capture ANY quoted string (not just a well-formed sha256:...), so a present
// but malformed/tampered id compares unequal and FAILS verify — matching Python's
// meta.get(key): a truthy-but-wrong value is a mismatch, not "no stored id".
[[nodiscard]] std::string extract_id(const std::string& meta, const std::string& key) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch m;
    if (std::regex_search(meta, m, re)) return m[1].str();
    return {};
}

// Pull an integer "<key>": <n> out of meta.json (mirrors meta.get(field) for the
// header P / n_block cross-check). Absent -> nullopt (the field is optional).
[[nodiscard]] std::optional<long long> extract_int(const std::string& meta, const std::string& key) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
    std::smatch m;
    if (std::regex_search(meta, m, re)) return std::stoll(m[1].str());
    return std::nullopt;
}

// sha256_file returns bare hex; the stored id is "sha256:"-prefixed (f2_dir_writer.cpp).
[[nodiscard]] std::string content_id(const fs::path& p) {
    return "sha256:" + sha256_file(p);
}

// The default scan root honors $STEPPE_F2_DIR (the env the fit commands read) so
// `ls` finds the cache the rest of steppe points at — else the current directory.
[[nodiscard]] fs::path default_root() {
    if (const char* env = std::getenv("STEPPE_F2_DIR"); env && *env) return fs::path(env);
    return fs::current_path();
}

}  // namespace

int run_cache_ls(const std::string& root_in) {
    std::error_code ec;
    const fs::path root = root_in.empty() ? default_root() : fs::path(root_in);
    // A missing root is not an error: the ec-constructed iterator is the end
    // iterator, the scan yields nothing, and we report "none" and exit 0 — the
    // same os.walk-yields-nothing behavior as the Python tool.
    bool any = false;
    for (auto it = fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        std::error_code dec;
        if (!it->is_directory(dec) || dec) continue;  // skip an un-stattable entry, never throw
        const fs::path d = it->path();
        if (!fs::exists(d / "f2.bin", ec)) continue;
        F2DiskHeader h{};
        if (!read_header(d, h)) continue;
        if (!any) std::printf("%-32s%6s%10s%10s\n", "PATH", "P", "N_BLOCK", "SIZE");
        any = true;
        std::error_code rec;
        const fs::path rel = fs::relative(d, root, rec);
        std::printf("%-32s%6d%10d%10s\n", (rec ? d : rel).string().c_str(),
                    h.P, h.n_block, human_size(dir_bytes(d)).c_str());
    }
    if (!any) std::fprintf(stderr, "steppe cache: no STPF2BK1 caches under %s\n",
                           root.string().c_str());
    return cfg::kExitOk;
}

int run_cache_show(const std::string& dir_in) {
    const fs::path dir(dir_in);
    F2DiskHeader h{};
    if (!read_header(dir, h)) {
        std::fprintf(stderr, "steppe cache: %s: not an STPF2BK1 cache\n", dir.string().c_str());
        return cfg::kExitInvalidConfig;
    }
    std::error_code ec;
    const std::uint64_t actual = fs::file_size(dir / "f2.bin", ec);
    const std::uint64_t expect = expected_f2_bytes(h.P, h.n_block);
    std::printf("Cache: %s\n", dir.string().c_str());
    std::printf("  P (populations):  %d   (from f2.bin header)\n", h.P);
    std::printf("  n_block:          %d   (from f2.bin header)\n", h.n_block);
    std::printf("  f2.bin size:      %llu bytes (expected %llu)  [%s]\n",
                static_cast<unsigned long long>(actual),
                static_cast<unsigned long long>(expect), actual == expect ? "OK" : "MISMATCH");
    const std::string meta = read_file(dir / "meta.json");
    if (meta.empty()) std::printf("  meta.json:        (none)\n");
    else std::printf("  --- meta.json ---\n%s\n", meta.c_str());
    return cfg::kExitOk;
}

int run_cache_verify(const std::string& dir_in, bool check_sources) {
    (void)check_sources;  // native path leaves source-file hashing to the Python tool
    const fs::path dir(dir_in);
    F2DiskHeader h{};
    if (!read_header(dir, h)) {
        std::fprintf(stderr, "steppe cache: %s: not an STPF2BK1 cache\n", dir.string().c_str());
        return cfg::kExitInvalidConfig;
    }
    std::error_code ec;
    const std::uint64_t actual = fs::file_size(dir / "f2.bin", ec);
    const std::uint64_t expect = expected_f2_bytes(h.P, h.n_block);
    const std::string meta = read_file(dir / "meta.json");
    int failures = 0;

    std::printf("verify %s:\n", dir.string().c_str());
    if (actual == expect) {
        std::printf("  [OK  ] f2.bin size    %llu bytes\n", static_cast<unsigned long long>(actual));
    } else {
        std::printf("  [FAIL] f2.bin size    %llu != expected %llu\n",
                    static_cast<unsigned long long>(actual), static_cast<unsigned long long>(expect));
        ++failures;
    }

    // meta.json P / n_block vs the authoritative header (Python _verify_checks
    // parity: a redundant meta field that contradicts the header is a FAIL).
    if (const auto mp = extract_int(meta, "P")) {
        if (*mp == h.P) std::printf("  [OK  ] header P       %d\n", h.P);
        else {
            std::printf("  [FAIL] header P       header %d != meta %lld\n", h.P, *mp);
            ++failures;
        }
    }
    if (const auto mn = extract_int(meta, "n_block")) {
        if (*mn == h.n_block) std::printf("  [OK  ] header n_block %d\n", h.n_block);
        else {
            std::printf("  [FAIL] header n_block header %d != meta %lld\n", h.n_block, *mn);
            ++failures;
        }
    }

    const std::string computed_f2 = content_id(dir / "f2.bin");
    const std::string stored_f2 = extract_id(meta, "f2_cache_id");
    if (stored_f2.empty())
        std::printf("  [--  ] f2_cache_id    no stored id; computed %s\n", computed_f2.c_str());
    else if (computed_f2 == stored_f2)
        std::printf("  [OK  ] f2_cache_id    %s\n", computed_f2.c_str());
    else {
        std::printf("  [FAIL] f2_cache_id    %s != stored %s\n", computed_f2.c_str(), stored_f2.c_str());
        ++failures;
    }

    if (fs::exists(dir / "pops.txt", ec)) {
        const std::string computed_pops = content_id(dir / "pops.txt");
        const std::string stored_pops = extract_id(meta, "pops_sha256");
        if (stored_pops.empty())
            std::printf("  [--  ] pops_sha256    no stored id; computed %s\n", computed_pops.c_str());
        else if (computed_pops == stored_pops)
            std::printf("  [OK  ] pops_sha256    %s\n", computed_pops.c_str());
        else {
            std::printf("  [FAIL] pops_sha256    %s != stored %s\n", computed_pops.c_str(), stored_pops.c_str());
            ++failures;
        }
    }

    std::printf("  (verify attests the f2.bin payload + pops.txt labels; the meta.json record itself is not hashable.)\n");
    if (failures) {
        std::fprintf(stderr, "steppe cache: verify FAILED (%d check(s))\n", failures);
        return cfg::kExitInvalidConfig;
    }
    return cfg::kExitOk;
}

}  // namespace steppe::app
