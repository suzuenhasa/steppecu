// src/io/geno_reader.cpp
//
// GenoReader: opens a genotype file, auto-detects its on-disk format, validates
// it, and gathers the requested individuals and SNPs into a packed in-memory
// tile. Reads and packs bytes only — never decodes (decode happens later on the
// GPU or CPU reference path). Pure host C++20, no CUDA, no core/device
// dependency; every failure surfaces as std::runtime_error.
//
// Reference: docs/reference/src_io_geno_reader.cpp.md
#include "io/geno_reader.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "io/eigenstrat_format.hpp"

namespace steppe::io {

namespace {

// Strip a trailing CR (CRLF line ending) — reference §8
void strip_cr(std::string& line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
}

// Scan the next ' '/'\t'-delimited token at or after cur; on success sets
// [start, len), advances cur past the token, and returns true (the shared scan
// primitive for the ANCESTRYMAP tokenizers) — reference §8
bool next_ws_token(std::string_view sv, std::size_t& cur,
                   std::size_t& start, std::size_t& len) {
    while (cur < sv.size() && (sv[cur] == ' ' || sv[cur] == '\t')) ++cur;
    if (cur >= sv.size()) return false;
    start = cur;
    while (cur < sv.size() && sv[cur] != ' ' && sv[cur] != '\t') ++cur;
    len = cur - start;
    return true;
}

// Sibling text-file line counter — reference §8
std::size_t count_text_records(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return 0;
    std::size_t n = 0;
    std::string line;
    while (std::getline(in, line)) {
        strip_cr(line);
        if (line.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;
            continue;
        }
        ++n;
    }
    return n;
}

// PLINK sibling-path derivation — reference §8
std::string plink_sibling(const std::string& bed_path, const char* ext) {
    constexpr std::string_view kBed = ".bed";
    if (bed_path.size() >= kBed.size() &&
        bed_path.compare(bed_path.size() - kBed.size(), kBed.size(), kBed) == 0) {
        return bed_path.substr(0, bed_path.size() - kBed.size()) + ext;
    }
    return bed_path + ext;
}

}  // namespace

// 64-bit std::streamoff assertion for record-offset math — reference §4
static_assert(sizeof(std::streamoff) >= 8,
              "GenoReader assumes a >=64-bit std::streamoff for record-offset math");

namespace {

// EIGENSTRAT geometry scan — reference §3
GenoHeader parse_eigenstrat_geometry(const std::string& path) {
    GenoHeader h;
    std::ifstream in(path, std::ios::binary);
    if (!in) return h;

    std::string line;
    std::size_t n_ind = 0;
    std::size_t n_snp = 0;
    bool first = true;
    while (std::getline(in, line)) {
        strip_cr(line);
        if (line.empty()) {
            if (in.peek() == std::char_traits<char>::eof()) break;
            return GenoHeader{};
        }
        if (first) {
            n_ind = line.size();
            first = false;
            if (n_ind == 0) return GenoHeader{};
        } else if (line.size() != n_ind) {
            return GenoHeader{};
        }
        for (char c : line) {
            std::uint8_t code = 0;
            if (!eigenstrat_char_to_code(c, code)) return GenoHeader{};
        }
        ++n_snp;
    }
    if (n_snp == 0 || n_ind == 0) return GenoHeader{};

    h.format = GenoFormat::Eigenstrat;
    h.n_ind = n_ind;
    h.n_snp = n_snp;
    h.n_records = n_snp;
    h.bytes_per_record = packed_bytes(n_ind);
    h.header_bytes = 0;
    return h;
}

// ANCESTRYMAP sibling-path derivation — reference §8
std::string geno_sibling(const std::string& geno_path, const char* ext) {
    constexpr std::string_view kGeno = ".geno";
    if (geno_path.size() >= kGeno.size() &&
        geno_path.compare(geno_path.size() - kGeno.size(), kGeno.size(), kGeno) == 0) {
        return geno_path.substr(0, geno_path.size() - kGeno.size()) + ext;
    }
    return geno_path + ext;
}

// ANCESTRYMAP probe + geometry-from-siblings — reference §3
GenoHeader parse_ancestrymap_geometry(const std::string& geno_path) {
    GenoHeader h;
    std::ifstream in(geno_path, std::ios::binary);
    if (!in) return h;

    std::string line;
    bool got_line = false;
    while (std::getline(in, line)) {
        strip_cr(line);
        bool only_ws = true;
        for (char c : line) {
            if (c != ' ' && c != '\t') { only_ws = false; break; }
        }
        if (only_ws) continue;
        got_line = true;
        break;
    }
    if (!got_line) return GenoHeader{};

    std::vector<std::string_view> toks;
    std::string_view sv(line);
    std::size_t cur = 0, start = 0, len = 0;
    while (next_ws_token(sv, cur, start, len)) {
        toks.emplace_back(sv.data() + start, len);
    }
    if (toks.size() != kAncestrymapFields) return GenoHeader{};
    std::uint8_t probe_code = 0;
    if (!ancestrymap_token_to_code(toks[kAncestrymapFields - 1], probe_code)) {
        return GenoHeader{};
    }

    const std::string snp_path = geno_sibling(geno_path, ".snp");
    const std::string ind_path = geno_sibling(geno_path, ".ind");
    const std::size_t n_snp = count_text_records(snp_path);
    const std::size_t n_ind = count_text_records(ind_path);
    if (n_snp == 0 || n_ind == 0) return GenoHeader{};

    h.format = GenoFormat::Ancestrymap;
    h.n_ind = n_ind;
    h.n_snp = n_snp;
    h.n_records = n_snp;
    h.bytes_per_record = packed_bytes(n_ind);
    h.header_bytes = 0;
    return h;
}

// GenoFormat to human-readable name + reader — reference §8
const char* geno_format_name(GenoFormat format) {
    switch (format) {
        case GenoFormat::Tgeno:       return "TGENO (read via read_tile)";
        case GenoFormat::Geno:        return "GENO (read via read_snp_major_tile)";
        case GenoFormat::Eigenstrat:  return "EIGENSTRAT (read via read_eigenstrat_snp_major_tile)";
        case GenoFormat::Plink:       return "PLINK (read via read_plink_snp_major_tile)";
        case GenoFormat::Ancestrymap: return "ANCESTRYMAP (read via read_ancestrymap_snp_major_tile)";
        case GenoFormat::Unknown:     return "an unrecognized format";
    }
    return "an unrecognized format";
}

}  // namespace

// GenoReader ctor: format auto-detection probe chain — reference §2
GenoReader::GenoReader(const std::string& geno_path) : path_(geno_path) {
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("io::GenoReader: cannot open .geno file: " + path_);
    }

    std::array<char, kGenoHeaderBytes> head{};
    const auto header_size = static_cast<std::streamsize>(kGenoHeaderBytes);
    in.read(head.data(), header_size);
    if (in.gcount() == header_size) {
        header_ = parse_geno_header(head);
    }

    if (header_.format == GenoFormat::Unknown) {
        header_ = parse_eigenstrat_geometry(path_);
    }
    if (header_.format == GenoFormat::Unknown) {
        const auto gc = static_cast<std::size_t>(in.gcount());
        const auto* magic = reinterpret_cast<const unsigned char*>(head.data());
        if (gc >= kBedMagicBytes && magic[0] == kBedMagic0 && magic[1] == kBedMagic1) {
            if (magic[2] != kBedModeSnpMajor) {
                throw std::runtime_error(
                    "io::GenoReader: PLINK .bed is individual-major (mode byte 0x00); only "
                    "SNP-major (0x01) is supported — re-export SNP-major: " + path_);
            }
            const std::string bim = plink_sibling(path_, ".bim");
            const std::string fam = plink_sibling(path_, ".fam");
            const std::size_t n_snp = count_text_records(bim);
            const std::size_t n_ind = count_text_records(fam);
            if (n_snp == 0 || n_ind == 0) {
                throw std::runtime_error(
                    "io::GenoReader: PLINK .bed found (magic 0x6c 0x1b 0x01) but its .bim/.fam "
                    "siblings are missing or empty (.bim=" + bim + " n_snp=" +
                    std::to_string(n_snp) + ", .fam=" + fam + " n_ind=" +
                    std::to_string(n_ind) + ") for " + path_);
            }
            header_.format = GenoFormat::Plink;
            header_.n_ind = n_ind;
            header_.n_snp = n_snp;
            header_.n_records = n_snp;
            header_.bytes_per_record = packed_bytes(n_ind);
            header_.header_bytes = kBedMagicBytes;
        }
    }
    if (header_.format == GenoFormat::Unknown) {
        header_ = parse_ancestrymap_geometry(path_);
    }
    if (header_.format == GenoFormat::Unknown) {
        throw std::runtime_error(
            "io::GenoReader: unrecognized .geno format (expected packed TGENO/GENO "
            "magic, an ASCII EIGENSTRAT .geno of 0/1/2/9 genotype lines, an ANCESTRYMAP "
            ".geno of <snp_id> <sample_id> <genotype> triple lines with .snp/.ind "
            "siblings, or a PLINK .bed of magic 0x6c 0x1b 0x01 with .bim/.fam siblings): " +
            path_);
    }
    if (header_.format == GenoFormat::Eigenstrat ||
        header_.format == GenoFormat::Ancestrymap) {
        records_present_ = header_.n_records;
        return;
    }
    if (header_.bytes_per_record == 0 || header_.n_records == 0) {
        throw std::runtime_error("io::GenoReader: degenerate header (zero records/stride): " + path_);
    }

    in.seekg(0, std::ios::end);
    const std::streamoff fsize = in.tellg();
    if (fsize < static_cast<std::streamoff>(header_.header_bytes)) {
        throw std::runtime_error("io::GenoReader: file smaller than header: " + path_);
    }
    const std::size_t data_bytes =
        static_cast<std::size_t>(fsize) - header_.header_bytes;
    records_present_ = data_bytes / header_.bytes_per_record;
    if (records_present_ == 0) {
        throw std::runtime_error("io::GenoReader: no complete records on disk: " + path_);
    }
    if (records_present_ > header_.n_records) records_present_ = header_.n_records;
}

// read_tile: individual-major (TGENO) gather — reference §5
GenotypeTile GenoReader::read_tile(const IndPartition& part,
                                   std::size_t snp_begin,
                                   std::size_t snp_end) {
    if (header_.format != GenoFormat::Tgeno) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: this is the TGENO (individual-major) path; "
            "this file is GENO (SNP-major PACKEDANCESTRYMAP) — read it via "
            "read_snp_major_tile + the on-device transpose_to_canonical.");
    }
    if (snp_begin != 0) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: M1 requires snp_begin == 0 (byte-aligned "
            "SNP prefix); nonzero begin is the M5 tile loop.");
    }
    if (snp_end <= snp_begin || snp_end > header_.n_snp) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: SNP range [" + std::to_string(snp_begin) + ", " +
            std::to_string(snp_end) + ") out of bounds (n_snp=" +
            std::to_string(header_.n_snp) + ")");
    }

    if (part.groups.empty()) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: empty partition (no selected populations) for " +
            path_);
    }

    const std::size_t tile_snps = snp_end - snp_begin;
    const std::size_t bytes_per_record = packed_bytes(tile_snps);

    std::size_t n_individuals = 0;
    for (const auto& g : part.groups) {
        for (std::size_t row : g.rows) {
            if (row >= records_present_) {
                throw std::runtime_error(
                    "io::GenoReader::read_tile: individual row " + std::to_string(row) +
                    " out of range (records_present=" + std::to_string(records_present_) +
                    ") in " + path_);
            }
        }
        n_individuals += g.rows.size();
    }

    const std::string size_operands =
        std::to_string(n_individuals) + " individuals * " +
        std::to_string(bytes_per_record) + " bytes/record";
    if (n_individuals > std::numeric_limits<std::size_t>::max() / bytes_per_record) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: tile size overflow (" + size_operands +
            " exceeds size_t) for " + path_);
    }

    GenotypeTile tile;
    tile.bytes_per_record = bytes_per_record;
    tile.n_snp = tile_snps;
    tile.n_individuals = n_individuals;
    try {
        tile.packed.resize(n_individuals * bytes_per_record);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: out of memory allocating tile (" +
            size_operands + ") for " + path_);
    } catch (const std::length_error&) {
        throw std::runtime_error(
            "io::GenoReader::read_tile: tile too large for the allocator (" +
            size_operands + " exceeds vector::max_size()) for " + path_);
    }
    tile.pop_offsets.reserve(part.groups.size() + 1);
    tile.pop_labels.reserve(part.groups.size());

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("io::GenoReader::read_tile: cannot reopen .geno: " + path_);
    }

    std::size_t out_ind = 0;
    tile.pop_offsets.push_back(0);
    const std::streamsize rec_bytes = static_cast<std::streamsize>(bytes_per_record);
    for (const auto& g : part.groups) {
        tile.pop_labels.push_back(g.label);
        for (std::size_t row : g.rows) {
            const std::streamoff off =
                static_cast<std::streamoff>(header_.header_bytes) +
                static_cast<std::streamoff>(row) *
                    static_cast<std::streamoff>(header_.bytes_per_record);
            in.seekg(off, std::ios::beg);
            char* dst = reinterpret_cast<char*>(tile.packed.data() + out_ind * bytes_per_record);
            in.read(dst, rec_bytes);
            if (in.gcount() != rec_bytes) {
                throw std::runtime_error(
                    "io::GenoReader::read_tile: short read for individual row " +
                    std::to_string(row) + " in " + path_);
            }
            ++out_ind;
        }
        tile.pop_offsets.push_back(out_ind);
    }
    return tile;
}

// build_selection: pop-contiguous selected-row gather list — reference §7
void GenoReader::build_selection(const IndPartition& part,
                                 std::size_t src_bpr,
                                 std::size_t tile_snps,
                                 const char* reader_name,
                                 SnpMajorTile& tile) const {
    tile.src_bytes_per_record = src_bpr;
    tile.n_snp = tile_snps;
    tile.pop_offsets.reserve(part.groups.size() + 1);
    tile.pop_labels.reserve(part.groups.size());
    tile.pop_offsets.push_back(0);
    std::size_t out_ind = 0;
    for (const auto& g : part.groups) {
        tile.pop_labels.push_back(g.label);
        for (std::size_t row : g.rows) {
            if (row >= header_.n_ind) {
                throw std::runtime_error(
                    std::string("io::GenoReader::") + reader_name + ": individual row " +
                    std::to_string(row) + " out of range (n_ind=" +
                    std::to_string(header_.n_ind) + ") in " + path_);
            }
            tile.sel_rows.push_back(row);
            ++out_ind;
        }
        tile.pop_offsets.push_back(out_ind);
    }
    tile.n_individuals = out_ind;
}

// checked_alloc_snp_major: overflow-checked SNP-major allocation — reference §7
void GenoReader::checked_alloc_snp_major(std::size_t tile_snps,
                                         std::size_t src_bpr,
                                         const char* reader_name,
                                         bool zero_init,
                                         SnpMajorTile& tile) const {
    const std::string size_operands =
        std::to_string(tile_snps) + " snp-records * " +
        std::to_string(src_bpr) + " bytes/record";
    if (tile_snps > std::numeric_limits<std::size_t>::max() / src_bpr) {
        throw std::runtime_error(
            std::string("io::GenoReader::") + reader_name + ": source size overflow (" +
            size_operands + " exceeds size_t) for " + path_);
    }
    try {
        if (zero_init) {
            tile.snp_major.assign(tile_snps * src_bpr, std::uint8_t{0});
        } else {
            tile.snp_major.resize(tile_snps * src_bpr);
        }
    } catch (const std::bad_alloc&) {
        throw std::runtime_error(
            std::string("io::GenoReader::") + reader_name +
            ": out of memory allocating SNP-major source (" + size_operands +
            ") for " + path_);
    } catch (const std::length_error&) {
        throw std::runtime_error(
            std::string("io::GenoReader::") + reader_name +
            ": SNP-major source too large for the allocator (" + size_operands +
            " exceeds vector::max_size()) for " + path_);
    }
}

// check_snp_major_range: shared snp_begin/snp_end/empty-partition guard preamble
// for the four SNP-major readers (Guards 2-4). Keeps per-reader message bytes
// identical via who (method tag) + begin_tag (the P0/P2/'' milestone prefix).
void GenoReader::check_snp_major_range(const IndPartition& part,
                                       std::size_t snp_begin,
                                       std::size_t snp_end,
                                       const char* who,
                                       const char* begin_tag) const {
    const std::string prefix = std::string("io::GenoReader::") + who + ": ";
    if (snp_begin != 0) {
        throw std::runtime_error(
            prefix + begin_tag + "requires snp_begin == 0 "
            "(byte-aligned SNP prefix); nonzero begin is the M5 tile loop.");
    }
    if (snp_end <= snp_begin || snp_end > header_.n_snp) {
        throw std::runtime_error(
            prefix + "SNP range [" +
            std::to_string(snp_begin) + ", " + std::to_string(snp_end) +
            ") out of bounds (n_snp=" + std::to_string(header_.n_snp) + ")");
    }
    if (part.groups.empty()) {
        throw std::runtime_error(
            prefix + "empty partition (no selected "
            "populations) for " + path_);
    }
}

// read_snp_major_tile: GENO SNP-major gather — reference §6
SnpMajorTile GenoReader::read_snp_major_tile(const IndPartition& part,
                                             std::size_t snp_begin,
                                             std::size_t snp_end) {
    if (header_.format != GenoFormat::Geno) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: this is the GENO (SNP-major "
            "PACKEDANCESTRYMAP) path; this file is TGENO (individual-major) — read it "
            "via read_tile.");
    }
    check_snp_major_range(part, snp_begin, snp_end, "read_snp_major_tile", "P0 ");

    const std::size_t tile_snps = snp_end - snp_begin;
    const std::size_t src_bpr = header_.bytes_per_record;
    if (records_present_ < tile_snps) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: requested SNP prefix [0, " +
            std::to_string(tile_snps) + ") exceeds SNP records on disk (" +
            std::to_string(records_present_) + ") in " + path_);
    }

    SnpMajorTile tile;
    build_selection(part, src_bpr, tile_snps, "read_snp_major_tile", tile);
    checked_alloc_snp_major(tile_snps, src_bpr, "read_snp_major_tile",
                            /*zero_init=*/false, tile);

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "io::GenoReader::read_snp_major_tile: cannot reopen .geno: " + path_);
    }
    const std::streamsize rec_bytes = static_cast<std::streamsize>(src_bpr);
    for (std::size_t s = 0; s < tile_snps; ++s) {
        const std::streamoff off =
            static_cast<std::streamoff>(header_.header_bytes) +
            static_cast<std::streamoff>(s) * static_cast<std::streamoff>(src_bpr);
        in.seekg(off, std::ios::beg);
        char* dst = reinterpret_cast<char*>(tile.snp_major.data() + s * src_bpr);
        in.read(dst, rec_bytes);
        if (in.gcount() != rec_bytes) {
            throw std::runtime_error(
                "io::GenoReader::read_snp_major_tile: short read for SNP record " +
                std::to_string(s) + " in " + path_);
        }
    }
    return tile;
}

// read_eigenstrat_snp_major_tile: EIGENSTRAT ASCII SNP-major gather — reference §6
SnpMajorTile GenoReader::read_eigenstrat_snp_major_tile(const IndPartition& part,
                                                        std::size_t snp_begin,
                                                        std::size_t snp_end) {
    if (header_.format != GenoFormat::Eigenstrat) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: this is the EIGENSTRAT "
            "(ASCII SNP-major) path; this file is " +
            std::string(geno_format_name(header_.format)) +
            " — wrong reader for " + path_);
    }
    check_snp_major_range(part, snp_begin, snp_end,
                          "read_eigenstrat_snp_major_tile", "P0 ");

    const std::size_t tile_snps = snp_end - snp_begin;
    const std::size_t src_bpr = header_.bytes_per_record;
    if (tile_snps > records_present_) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: requested SNP prefix [0, " +
            std::to_string(tile_snps) + ") exceeds SNP lines on disk (" +
            std::to_string(records_present_) + ") in " + path_);
    }

    SnpMajorTile tile;
    build_selection(part, src_bpr, tile_snps, "read_eigenstrat_snp_major_tile", tile);
    checked_alloc_snp_major(tile_snps, src_bpr, "read_eigenstrat_snp_major_tile",
                            /*zero_init=*/true, tile);

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "io::GenoReader::read_eigenstrat_snp_major_tile: cannot reopen .geno: " + path_);
    }
    const std::size_t cpb = static_cast<std::size_t>(kCodesPerByte);
    std::string line;
    for (std::size_t s = 0; s < tile_snps; ++s) {
        if (!std::getline(in, line)) {
            throw std::runtime_error(
                "io::GenoReader::read_eigenstrat_snp_major_tile: unexpected EOF at SNP line " +
                std::to_string(s + 1) + " (expected " + std::to_string(tile_snps) +
                " lines) in " + path_);
        }
        strip_cr(line);
        if (line.size() != header_.n_ind) {
            throw std::runtime_error(
                "io::GenoReader::read_eigenstrat_snp_major_tile: SNP line " +
                std::to_string(s + 1) + " has " + std::to_string(line.size()) +
                " genotype chars, expected n_ind=" + std::to_string(header_.n_ind) +
                " (ragged .geno desyncs the SNP axis) in " + path_);
        }
        std::uint8_t* rec = tile.snp_major.data() + s * src_bpr;
        for (std::size_t i = 0; i < header_.n_ind; ++i) {
            std::uint8_t code = 0;
            if (!eigenstrat_char_to_code(line[i], code)) {
                throw std::runtime_error(
                    "io::GenoReader::read_eigenstrat_snp_major_tile: illegal genotype char '" +
                    std::string(1, line[i]) + "' at SNP line " + std::to_string(s + 1) +
                    ", individual column " + std::to_string(i + 1) +
                    " (expected 0/1/2/9) in " + path_);
            }
            const std::size_t byte_in_rec = i / cpb;
            rec[byte_in_rec] =
                pack_code_into_byte(rec[byte_in_rec], static_cast<int>(i), code);
        }
    }
    return tile;
}

// read_plink_snp_major_tile: PLINK .bed SNP-major gather — reference §6
SnpMajorTile GenoReader::read_plink_snp_major_tile(const IndPartition& part,
                                                   std::size_t snp_begin,
                                                   std::size_t snp_end) {
    if (header_.format != GenoFormat::Plink) {
        throw std::runtime_error(
            "io::GenoReader::read_plink_snp_major_tile: this is the PLINK (.bed SNP-major) "
            "path; this file is " +
            std::string(geno_format_name(header_.format)) +
            " — wrong reader for " + path_);
    }
    check_snp_major_range(part, snp_begin, snp_end,
                          "read_plink_snp_major_tile", "P2 ");

    const std::size_t tile_snps = snp_end - snp_begin;
    const std::size_t src_bpr = header_.bytes_per_record;
    if (tile_snps > records_present_) {
        throw std::runtime_error(
            "io::GenoReader::read_plink_snp_major_tile: requested SNP prefix [0, " +
            std::to_string(tile_snps) + ") exceeds SNP records on disk (" +
            std::to_string(records_present_) + ") in " + path_);
    }

    SnpMajorTile tile;
    build_selection(part, src_bpr, tile_snps, "read_plink_snp_major_tile", tile);
    checked_alloc_snp_major(tile_snps, src_bpr, "read_plink_snp_major_tile",
                            /*zero_init=*/true, tile);

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "io::GenoReader::read_plink_snp_major_tile: cannot reopen .bed: " + path_);
    }
    const std::size_t cpb = static_cast<std::size_t>(kCodesPerByte);
    const std::streamsize rec_bytes = static_cast<std::streamsize>(src_bpr);
    std::vector<std::uint8_t> bed_rec(src_bpr);
    for (std::size_t s = 0; s < tile_snps; ++s) {
        const std::streamoff off =
            static_cast<std::streamoff>(header_.header_bytes) +
            static_cast<std::streamoff>(s) * static_cast<std::streamoff>(src_bpr);
        in.seekg(off, std::ios::beg);
        char* dst = reinterpret_cast<char*>(bed_rec.data());
        in.read(dst, rec_bytes);
        if (in.gcount() != rec_bytes) {
            throw std::runtime_error(
                "io::GenoReader::read_plink_snp_major_tile: short read for SNP record " +
                std::to_string(s) + " in " + path_);
        }
        std::uint8_t* canon_rec = tile.snp_major.data() + s * src_bpr;
        for (std::size_t i = 0; i < header_.n_ind; ++i) {
            const std::uint8_t bed_code =
                bed_code_in_byte(bed_rec[i / cpb], static_cast<int>(i % cpb));
            const std::uint8_t canon = kBedToCanon[bed_code];
            canon_rec[i / cpb] =
                pack_code_into_byte(canon_rec[i / cpb], static_cast<int>(i), canon);
        }
    }
    return tile;
}

// read_ancestrymap_snp_major_tile: ANCESTRYMAP text-triple SNP-major gather — reference §6
SnpMajorTile GenoReader::read_ancestrymap_snp_major_tile(const IndPartition& part,
                                                         std::size_t snp_begin,
                                                         std::size_t snp_end) {
    if (header_.format != GenoFormat::Ancestrymap) {
        throw std::runtime_error(
            "io::GenoReader::read_ancestrymap_snp_major_tile: this is the ANCESTRYMAP "
            "(text-triple SNP-major) path; this file is " +
            std::string(geno_format_name(header_.format)) +
            " — wrong reader for " + path_);
    }
    check_snp_major_range(part, snp_begin, snp_end,
                          "read_ancestrymap_snp_major_tile", "");

    const std::size_t tile_snps = snp_end - snp_begin;
    const std::size_t src_bpr = header_.bytes_per_record;
    if (tile_snps > records_present_) {
        throw std::runtime_error(
            "io::GenoReader::read_ancestrymap_snp_major_tile: requested SNP prefix [0, " +
            std::to_string(tile_snps) + ") exceeds SNP blocks on disk (" +
            std::to_string(records_present_) + ") in " + path_);
    }

    SnpMajorTile tile;
    build_selection(part, src_bpr, tile_snps, "read_ancestrymap_snp_major_tile", tile);
    checked_alloc_snp_major(tile_snps, src_bpr, "read_ancestrymap_snp_major_tile",
                            /*zero_init=*/true, tile);

    std::ifstream in(path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "io::GenoReader::read_ancestrymap_snp_major_tile: cannot reopen .geno: " + path_);
    }
    const std::size_t cpb = static_cast<std::size_t>(kCodesPerByte);
    const std::size_t n_ind = header_.n_ind;
    const std::string line_count_operands =
        std::to_string(tile_snps) + " snp-records * " +
        std::to_string(n_ind) + " individuals";
    if (n_ind && tile_snps > std::numeric_limits<std::size_t>::max() / n_ind) {
        throw std::runtime_error(
            "io::GenoReader::read_ancestrymap_snp_major_tile: line count overflow (" +
            line_count_operands + " exceeds size_t) for " + path_);
    }
    const std::size_t n_lines = tile_snps * n_ind;
    std::string line;
    for (std::size_t L = 0; L < n_lines; ++L) {
        if (!std::getline(in, line)) {
            const std::size_t s = L / n_ind, ii = L % n_ind;
            throw std::runtime_error(
                "io::GenoReader::read_ancestrymap_snp_major_tile: unexpected EOF at line " +
                std::to_string(L + 1) + " (SNP " + std::to_string(s) + ", individual " +
                std::to_string(ii) + "; expected " + std::to_string(n_lines) +
                " triple lines for the prefix) in " + path_);
        }
        strip_cr(line);
        std::string_view sv(line);
        std::array<std::string_view, kAncestrymapFields> toks{};
        std::size_t ntok = 0;
        std::size_t cur = 0, start = 0, len = 0;
        bool too_many = false;
        while (next_ws_token(sv, cur, start, len)) {
            if (ntok < kAncestrymapFields) {
                toks[ntok] = sv.substr(start, len);
            } else {
                too_many = true;
                break;
            }
            ++ntok;
        }
        if (too_many || ntok != kAncestrymapFields) {
            const std::size_t s = L / n_ind, ii = L % n_ind;
            throw std::runtime_error(
                "io::GenoReader::read_ancestrymap_snp_major_tile: malformed line " +
                std::to_string(L + 1) + " (SNP " + std::to_string(s) + ", individual " +
                std::to_string(ii) + "): expected " + std::to_string(kAncestrymapFields) +
                " whitespace tokens <snp_id> <sample_id> <genotype>, got " +
                std::to_string(too_many ? ntok + 1 : ntok) + " in " + path_);
        }
        std::uint8_t code = 0;
        if (!ancestrymap_token_to_code(toks[kAncestrymapFields - 1], code)) {
            const std::size_t s = L / n_ind, ii = L % n_ind;
            throw std::runtime_error(
                "io::GenoReader::read_ancestrymap_snp_major_tile: illegal genotype token \"" +
                std::string(toks[kAncestrymapFields - 1]) + "\" at line " +
                std::to_string(L + 1) + " (SNP " + std::to_string(s) + ", individual " +
                std::to_string(ii) + "; expected 0/1/2/-1) in " + path_);
        }
        const std::size_t s = L / n_ind;
        const std::size_t i = L % n_ind;
        std::uint8_t* rec = tile.snp_major.data() + s * src_bpr;
        const std::size_t byte_in_rec = i / cpb;
        rec[byte_in_rec] =
            pack_code_into_byte(rec[byte_in_rec], static_cast<int>(i), code);
    }
    return tile;
}

}  // namespace steppe::io
