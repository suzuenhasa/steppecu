// src/io/gzip_line_reader.hpp
//
// Reference: docs/reference/src_io_gzip_line_reader.hpp.md
//
// GzipLineReader — a streaming line reader over a .vcf.gz (BGZF or plain gzip)
// or a plain-text .vcf, backed by zlib. Stage-1 VCF ingestion needs only
// sequential streaming (no tabix / random access), so a plain zlib inflate over
// the concatenated gzip members reads a BGZF file transparently: every BGZF
// block is a valid gzip member, so on Z_STREAM_END with input remaining the
// stream is inflateReset() and continues. The partial-line carry lives OUTSIDE
// the zlib reset so a VCF line spanning two BGZF blocks is never split, a final
// line with no trailing newline is flushed at EOF, and a zero-output member (the
// 28-byte BGZF EOF marker) is a clean end, not an error.
//
// Pure host C++20 io-leaf (links zlib only); failures surface as std::runtime_error.
#ifndef STEPPE_IO_GZIP_LINE_READER_HPP
#define STEPPE_IO_GZIP_LINE_READER_HPP

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

// zlib's z_stream is an opaque handle here; the .cpp owns the concrete type via a
// heap-allocated shim so this header stays free of <zlib.h>.
namespace steppe::io {

class GzipLineReader {
public:
    // Open + sniff the gzip magic (0x1f 0x8b). A non-gzip file is read as plain
    // text (needed for the by-construction unit fixture). Throws on open failure.
    explicit GzipLineReader(const std::string& path);
    ~GzipLineReader();

    GzipLineReader(const GzipLineReader&) = delete;
    GzipLineReader& operator=(const GzipLineReader&) = delete;

    // Fetch the next line (without its trailing '\n'/'\r'). Returns false at
    // end-of-stream. The final non-newline-terminated line IS returned once.
    [[nodiscard]] bool next_line(std::string& out);

private:
    // Pull more decoded bytes into buf_; returns false only when the source is
    // fully exhausted and nothing more can be produced.
    bool decode_more();

    std::string path_;
    std::ifstream file_;
    bool gzip_ = false;
    bool src_eof_ = false;

    // Decoded-but-unconsumed bytes (the partial-line carry lives here, so it
    // survives every inflateReset at a member boundary).
    std::string buf_;
    std::size_t scan_ = 0;  // next unsearched offset into buf_

    std::vector<unsigned char> in_;
    std::vector<char> out_;

    // Opaque zlib z_stream (heap-owned in the .cpp to keep <zlib.h> out of here).
    void* zstrm_ = nullptr;
};

}  // namespace steppe::io

#endif  // STEPPE_IO_GZIP_LINE_READER_HPP
