// tests/unit/test_geno_reader.cpp
//
// Host-only unit test of io::GenoReader (architecture.md §13 "Unit tests";
// ROADMAP M1, §5). Pure C++ TU, NO GPU, NO real AADR data: it writes tiny
// SYNTHETIC TGENO .geno files to a temp path and exercises the reader's
// happy-path gather plus its FAIL-FAST boundary checks — control flow / boundary
// logic, not a precision claim (ROADMAP §0: the no-synthetic rule is for
// precision claims; a hand-packed 4-individual TGENO record is a layout fixture).
// The numerical-accuracy gate stays the real-AADR equivalence test
// (tests/reference/test_decode_equivalence.cu).
//
// PRIMARY PURPOSE (cleanup geno_reader B17 verdict gate): assert that a
// HOSTILE / OVERSIZED partition THROWS rather than silently corrupting the heap.
// Before the B17 fix, `read_tile` computed `n_ind * bytes_per_rec` as an
// unguarded std::size_t product (can WRAP) and never checked `row <
// records_present_`, so a stale/hostile IndPartition produced a too-small buffer
// and the gather wrote PAST the allocation (silent heap-buffer-overflow). The fix
// (a) rejects an empty partition, (b) validates every row < records_present_, and
// (c) adds a checked-multiply guard before resize. This test pins all three plus
// a normal partition that decodes correctly.
//
// Dual harness (identical to tests/unit/test_filters.cpp): with
// -DSTEPPE_TEST_WITH_GTEST it uses GoogleTest; otherwise a self-checking main()
// returning non-zero on the first failure — all CTest needs to gate. No CUDA.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "io/eigenstrat_format.hpp"  // kGenoHeaderBytes, packed_bytes
#include "io/geno_reader.hpp"        // GenoReader
#include "io/ind_reader.hpp"         // IndPartition, PopGroup

namespace {

using steppe::io::GenoReader;
using steppe::io::GenotypeTile;
using steppe::io::IndPartition;
using steppe::io::PopGroup;
using steppe::io::kGenoHeaderBytes;
using steppe::io::packed_bytes;

// ---- synthetic-TGENO writer --------------------------------------------------
// Write a TGENO .geno: a kGenoHeaderBytes-byte "TGENO <n_ind> <n_snp> ..." header
// (NUL-padded to the fixed width), followed by `records_on_disk` records each of
// packed_bytes(n_snp) bytes. Each record's bytes are filled with a deterministic,
// per-individual marker so the gather can be checked exactly: individual `i`'s
// byte `b` is (uint8_t)(i * 31 + b + 1) (nonzero, distinct per (i,b) for the
// small fixtures used here). `records_on_disk` lets a test write a SHORT file
// (fewer records than the header claims) to drive records_present_ < n_records.
[[nodiscard]] std::vector<std::uint8_t> record_bytes(std::size_t ind,
                                                     std::size_t bytes_per_rec) {
    std::vector<std::uint8_t> rec(bytes_per_rec);
    for (std::size_t b = 0; b < bytes_per_rec; ++b) {
        rec[b] = static_cast<std::uint8_t>((ind * 31u + b + 1u) & 0xFFu);
    }
    return rec;
}

void write_tgeno(const std::filesystem::path& path, std::size_t n_ind,
                 std::size_t n_snp, std::size_t records_on_disk) {
    std::array<char, kGenoHeaderBytes> head{};  // zero-initialized = NUL padding
    const std::string text =
        "TGENO " + std::to_string(n_ind) + " " + std::to_string(n_snp) + " hash1 hash2";
    // text fits the 48-byte header for the small fixtures used here.
    for (std::size_t i = 0; i < text.size() && i < kGenoHeaderBytes; ++i) {
        head[i] = text[i];
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(head.data(), static_cast<std::streamsize>(kGenoHeaderBytes));
    const std::size_t bpr = packed_bytes(n_snp);
    for (std::size_t i = 0; i < records_on_disk; ++i) {
        const std::vector<std::uint8_t> rec = record_bytes(i, bpr);
        out.write(reinterpret_cast<const char*>(rec.data()),
                  static_cast<std::streamsize>(bpr));
    }
    out.close();
}

// A unique temp path per fixture so parallel ctest invocations do not collide.
[[nodiscard]] std::filesystem::path temp_geno(const char* tag) {
    static int counter = 0;
    auto p = std::filesystem::temp_directory_path();
    p /= ("steppe_geno_reader_test_" + std::string(tag) + "_" +
          std::to_string(++counter) + ".geno");
    return p;
}

struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const char* tag) : path(temp_geno(tag)) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

[[nodiscard]] PopGroup group(std::string label, std::vector<std::size_t> rows) {
    PopGroup g;
    g.label = std::move(label);
    g.rows = std::move(rows);
    return g;
}

[[nodiscard]] bool threw_runtime_error(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;  // wrong exception type fails the contract
    }
    return false;  // no throw fails the test
}

// ---- (1) NORMAL partition decodes correctly ----------------------------------
// 6 individuals × 10 SNPs (packed_bytes(10)==3 bytes/record). Select two
// populations out of order to confirm pop-contiguous gather + pop_offsets/labels.
[[nodiscard]] bool test_normal_decode() {
    const std::size_t n_ind = 6, n_snp = 10;
    const std::size_t bpr = packed_bytes(n_snp);  // 3
    TempFile f("normal");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/n_ind);

    GenoReader reader(f.path.string());
    if (reader.records_present() != n_ind) return false;
    if (reader.header().bytes_per_record != bpr) return false;

    IndPartition part;
    part.groups.push_back(group("popB", {4, 5}));  // 2 individuals (rows 4,5)
    part.groups.push_back(group("popA", {0, 2}));  // 2 individuals (rows 0,2)
    part.n_individuals_total = n_ind;

    GenotypeTile tile = reader.read_tile(part, 0, n_snp);

    if (tile.n_individuals != 4) return false;
    if (tile.n_snp != n_snp) return false;
    if (tile.bytes_per_record != bpr) return false;
    if (tile.packed.size() != 4 * bpr) return false;
    if (tile.n_pop() != 2) return false;
    if (tile.pop_labels.size() != 2 || tile.pop_labels[0] != "popB" ||
        tile.pop_labels[1] != "popA") {
        return false;
    }
    // pop_offsets: {0, 2, 4} (2 individuals each, in groups order).
    if (tile.pop_offsets.size() != 3 || tile.pop_offsets[0] != 0 ||
        tile.pop_offsets[1] != 2 || tile.pop_offsets[2] != 4) {
        return false;
    }

    // Gathered order is pop-contiguous in groups order: rows 4,5,0,2.
    const std::size_t gathered_rows[] = {4, 5, 0, 2};
    for (std::size_t g = 0; g < 4; ++g) {
        const std::vector<std::uint8_t> expect = record_bytes(gathered_rows[g], bpr);
        for (std::size_t b = 0; b < bpr; ++b) {
            if (tile.packed[g * bpr + b] != expect[b]) return false;
        }
    }
    return true;
}

// ---- (2) HOSTILE / OVERSIZED partition THROWS (the B17 heap-corruption gate) --
// A partition with a row >= records_present_ is exactly the stale/wrong-dataset
// partition that, pre-fix, seeked into trailing junk and (with enough such rows)
// wrapped the size multiply ⇒ too-small buffer ⇒ gather writes PAST it. The fix
// rejects it at the boundary with a thrown std::runtime_error BEFORE any resize
// or write — so no heap corruption can occur. We assert the THROW.
[[nodiscard]] bool test_hostile_oversized_partition_throws() {
    const std::size_t n_ind = 4, n_snp = 8;
    TempFile f("hostile");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/n_ind);

    GenoReader reader(f.path.string());
    if (reader.records_present() != n_ind) return false;

    // Row n_ind is one past the last present record; a far-out row models a
    // partition built against a stale/larger dataset.
    IndPartition bad1;
    bad1.groups.push_back(group("pop", {0, n_ind}));  // n_ind == 4 is out of range
    if (!threw_runtime_error([&] { (void)reader.read_tile(bad1, 0, n_snp); })) return false;

    IndPartition bad2;
    // A hugely oversized row — the kind that, combined with duplicates, drove the
    // size-multiply wrap. The row-bound guard rejects it before any allocation.
    bad2.groups.push_back(group("pop", {static_cast<std::size_t>(1) << 40}));
    if (!threw_runtime_error([&] { (void)reader.read_tile(bad2, 0, n_snp); })) return false;

    // SANITY: after the throws, a VALID partition on the SAME reader still decodes
    // (the reader was not left in a broken state, and nothing was corrupted).
    IndPartition good;
    good.groups.push_back(group("pop", {0, 1, 2, 3}));
    GenotypeTile tile = reader.read_tile(good, 0, n_snp);
    if (tile.n_individuals != 4) return false;
    const std::size_t bpr = packed_bytes(n_snp);
    for (std::size_t g = 0; g < 4; ++g) {
        const std::vector<std::uint8_t> expect = record_bytes(g, bpr);
        for (std::size_t b = 0; b < bpr; ++b) {
            if (tile.packed[g * bpr + b] != expect[b]) return false;
        }
    }
    return true;
}

// ---- (3) EMPTY partition THROWS ----------------------------------------------
// read_ind throws on an empty selection, but read_tile accepts an arbitrary
// partition; a zero-population partition must fail-fast, not yield a 0-pop tile.
[[nodiscard]] bool test_empty_partition_throws() {
    const std::size_t n_ind = 3, n_snp = 5;
    TempFile f("empty");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/n_ind);
    GenoReader reader(f.path.string());

    IndPartition empty;  // groups is empty
    return threw_runtime_error([&] { (void)reader.read_tile(empty, 0, n_snp); });
}

// ---- (4) SNP-range and partial-file edge cases (boundary correctness) ---------
[[nodiscard]] bool test_range_and_partial_edges() {
    const std::size_t n_ind = 5, n_snp = 12;
    // Header claims 5 records; write only 3 on disk -> records_present_ == 3.
    TempFile f("partial");
    write_tgeno(f.path, n_ind, n_snp, /*records_on_disk=*/3);
    GenoReader reader(f.path.string());
    if (reader.records_present() != 3) return false;  // partial-file cap (clamped down)

    IndPartition part;
    part.groups.push_back(group("pop", {0, 1, 2}));  // all in range

    // snp_begin != 0 rejected (M1 byte-aligned prefix only).
    if (!threw_runtime_error([&] { (void)reader.read_tile(part, 1, n_snp); })) return false;
    // snp_end > n_snp rejected.
    if (!threw_runtime_error([&] { (void)reader.read_tile(part, 0, n_snp + 1); })) return false;
    // snp_end == 0 (degenerate range) rejected.
    if (!threw_runtime_error([&] { (void)reader.read_tile(part, 0, 0); })) return false;

    // A row that is valid against the HEADER (row 3 < n_ind==5) but NOT present on
    // disk (>= records_present_==3) must throw — the silent wrong-individual /
    // heap-overflow path the B17 fix closes.
    IndPartition stale;
    stale.groups.push_back(group("pop", {0, 3}));  // row 3 >= records_present_(3)
    if (!threw_runtime_error([&] { (void)reader.read_tile(stale, 0, n_snp); })) return false;

    // A partial SNP prefix decodes (read fewer SNPs than the record holds).
    GenotypeTile tile = reader.read_tile(part, 0, 4);  // 1 byte/record prefix
    if (tile.bytes_per_record != packed_bytes(4)) return false;
    if (tile.n_individuals != 3) return false;
    if (tile.packed.size() != 3 * packed_bytes(4)) return false;
    return true;
}

// ---- (5) Constructor fail-fast on a degenerate / bad file --------------------
[[nodiscard]] bool test_ctor_failures() {
    // Missing file.
    if (!threw_runtime_error([] { GenoReader r("/nonexistent/steppe/no_such.geno"); })) {
        return false;
    }
    // File shorter than the 48-byte header.
    {
        TempFile f("shorthdr");
        std::ofstream out(f.path, std::ios::binary | std::ios::trunc);
        const char junk[8] = {'T', 'G', 'E', 'N', 'O', ' ', '1', ' '};
        out.write(junk, sizeof(junk));
        out.close();
        if (!threw_runtime_error([&] { GenoReader r(f.path.string()); })) return false;
    }
    // Valid header but no complete records on disk (header-only, 48 bytes exactly).
    {
        TempFile f("hdronly");
        write_tgeno(f.path, /*n_ind=*/4, /*n_snp=*/8, /*records_on_disk=*/0);
        if (!threw_runtime_error([&] { GenoReader r(f.path.string()); })) return false;
    }
    return true;
}

struct Case { const char* name; bool (*fn)(); };

constexpr Case kCases[] = {
    {"normal partition decodes correctly", test_normal_decode},
    {"hostile/oversized partition THROWS (B17 heap-corruption gate)",
     test_hostile_oversized_partition_throws},
    {"empty partition THROWS", test_empty_partition_throws},
    {"SNP-range / partial-file edges", test_range_and_partial_edges},
    {"constructor fail-fast (missing/short/degenerate)", test_ctor_failures},
};

}  // namespace

#ifdef STEPPE_TEST_WITH_GTEST
#include <gtest/gtest.h>

TEST(GenoReader, AllCases) {
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
        std::fprintf(stderr, "test_geno_reader: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_geno_reader: all %zu cases PASS\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
#endif
