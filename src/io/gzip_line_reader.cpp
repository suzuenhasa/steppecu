// src/io/gzip_line_reader.cpp
//
// GzipLineReader implementation — see the header for the concatenated-member
// (BGZF-as-gzip) rationale. zlib is named only in this TU.
#include "io/gzip_line_reader.hpp"

#include <cstring>
#include <stdexcept>

#include <zlib.h>

namespace steppe::io {

namespace {
constexpr std::size_t kInCap = 256 * 1024;
constexpr std::size_t kOutCap = 1024 * 1024;
}  // namespace

GzipLineReader::GzipLineReader(const std::string& path)
    : path_(path), file_(path, std::ios::binary), in_(kInCap), out_(kOutCap) {
    if (!file_) {
        throw std::runtime_error("io::GzipLineReader: cannot open VCF file: " + path);
    }
    // Sniff the two-byte gzip magic, then rewind so the decode path sees the
    // whole stream (header included).
    unsigned char magic[2] = {0, 0};
    file_.read(reinterpret_cast<char*>(magic), 2);
    const std::streamsize got = file_.gcount();
    file_.clear();
    file_.seekg(0);
    gzip_ = (got == 2 && magic[0] == 0x1f && magic[1] == 0x8b);

    if (gzip_) {
        auto* strm = new z_stream;
        std::memset(strm, 0, sizeof(z_stream));
        // 15 + 32: max window, and +32 enables automatic gzip header detection.
        if (inflateInit2(strm, 15 + 32) != Z_OK) {
            delete strm;
            throw std::runtime_error("io::GzipLineReader: inflateInit2 failed for " + path);
        }
        zstrm_ = strm;
    }
}

GzipLineReader::~GzipLineReader() {
    if (zstrm_ != nullptr) {
        auto* strm = static_cast<z_stream*>(zstrm_);
        inflateEnd(strm);
        delete strm;
        zstrm_ = nullptr;
    }
}

bool GzipLineReader::decode_more() {
    if (src_eof_) return false;

    if (!gzip_) {
        // Plain-text path: append a raw chunk.
        file_.read(out_.data(), static_cast<std::streamsize>(out_.size()));
        const std::streamsize n = file_.gcount();
        if (n > 0) {
            buf_.append(out_.data(), static_cast<std::size_t>(n));
            return true;
        }
        src_eof_ = true;
        return false;
    }

    auto* strm = static_cast<z_stream*>(zstrm_);
    // Loop until we have appended at least one decoded byte or the source is
    // truly exhausted. A zero-output member (BGZF EOF block) resets and retries.
    for (;;) {
        if (strm->avail_in == 0) {
            file_.read(reinterpret_cast<char*>(in_.data()),
                       static_cast<std::streamsize>(in_.size()));
            const std::streamsize n = file_.gcount();
            if (n <= 0) {
                src_eof_ = true;
                return false;  // no more input, nothing left to inflate
            }
            strm->next_in = in_.data();
            strm->avail_in = static_cast<uInt>(n);
        }

        strm->next_out = reinterpret_cast<Bytef*>(out_.data());
        strm->avail_out = static_cast<uInt>(out_.size());
        const int ret = inflate(strm, Z_NO_FLUSH);
        const std::size_t produced = out_.size() - strm->avail_out;
        if (produced > 0) buf_.append(out_.data(), produced);

        if (ret == Z_STREAM_END) {
            // End of one gzip/BGZF member; more members may follow. Reset and
            // continue — the buf_ carry is untouched by the reset.
            inflateReset(strm);
            if (produced > 0) return true;
            continue;  // zero-output member (e.g. BGZF EOF) -> read the next
        }
        if (ret == Z_OK || ret == Z_BUF_ERROR) {
            if (produced > 0) return true;
            // No progress: need more input on the next loop iteration. If the
            // file is also at EOF and avail_in is drained, we are done.
            if (strm->avail_in == 0 && file_.eof()) {
                src_eof_ = true;
                return false;
            }
            continue;
        }
        // Z_DATA_ERROR / Z_MEM_ERROR / etc.
        throw std::runtime_error("io::GzipLineReader: zlib inflate error (code " +
                                 std::to_string(ret) + ") reading " + path_);
    }
}

bool GzipLineReader::next_line(std::string& out) {
    for (;;) {
        // Search the unscanned tail of buf_ for a newline.
        const std::size_t nl = buf_.find('\n', scan_);
        if (nl != std::string::npos) {
            std::size_t end = nl;
            if (end > 0 && buf_[end - 1] == '\r') --end;  // tolerate CRLF
            out.assign(buf_, 0, end);
            buf_.erase(0, nl + 1);
            scan_ = 0;
            return true;
        }
        // No complete line buffered; try to decode more.
        scan_ = buf_.size();
        if (!decode_more()) {
            // Source exhausted: flush a final non-newline-terminated line once.
            if (!buf_.empty()) {
                std::size_t end = buf_.size();
                if (end > 0 && buf_[end - 1] == '\r') --end;
                out.assign(buf_, 0, end);
                buf_.clear();
                scan_ = 0;
                return true;
            }
            return false;
        }
    }
}

}  // namespace steppe::io
