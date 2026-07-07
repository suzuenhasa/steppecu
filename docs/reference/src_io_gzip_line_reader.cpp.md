# `gzip_line_reader.cpp` reference

## 1. Purpose

`src/io/gzip_line_reader.cpp` implements `GzipLineReader`, a streaming line
reader over a VCF file. It is the front door of Stage-1 VCF ingestion: the VCF
reader (`vcf_reader.cpp`) opens one of these, then just calls `next_line` in a
loop to walk the header and the body a line at a time. The class hides one thing
entirely from that caller — whether the bytes on disk are compressed.

It reads three shapes of file through a single line-yielding interface:

- a plain-text `.vcf`,
- a classic gzip `.vcf.gz` (one gzip member),
- a bgzip'd `.vcf.gz` (BGZF — the format `bgzip`/tabix write, which is a *stream
  of many* small gzip members concatenated end to end).

The caller never has to know which of the three it got. It opens the reader,
pulls lines until `next_line` returns false, and gets the same clean, newline-
stripped lines in every case.

This is a pure host C++20 io-leaf. It links zlib and nothing else — zlib is named
only in this translation unit, and the header keeps `<zlib.h>` out by holding the
`z_stream` behind an opaque `void*`. There is no CUDA here, no core or device
dependency. Failures surface as `std::runtime_error`.

---

## 2. The one insight: BGZF is just concatenated gzip

The whole design turns on a single fact about the BGZF format. A BGZF file is not
one gzip stream — it is many. `bgzip` compresses the data in ~64 KB chunks and
writes each chunk as its own complete, self-contained gzip member, then glues all
those members together into one file (with a final tiny 28-byte empty member as an
end-of-file marker). Tabix relies on this so it can seek to any block.

Steppe's VCF ingestion only ever reads a VCF top to bottom — no random access, no
tabix index — so it does **not** need to understand BGZF's block structure at all.
It just needs to keep inflating. Because every BGZF block is itself a valid gzip
member, a plain zlib inflate that, on reaching the end of one member, resets and
keeps going will decode a BGZF file transparently, member after member, as if it
were one long stream. That is exactly what this reader does (section 4).

So there is no separate BGZF code path. Plain single-member gzip is just the case
where the "keep going after a member ends" loop happens to find no more members.
The 15+32 window setting handed to `inflateInit2` (section 3) also asks zlib to
auto-detect the gzip header on each member, so both plain-gzip and BGZF work
through the identical inflate call.

---

## 3. Construction: sniff, then decide the path

The constructor opens the file in binary mode and throws
`io::GzipLineReader: cannot open VCF file: <path>` if that fails.

Then it **sniffs the two-byte gzip magic**. It reads the first two bytes, checks
them against `0x1f 0x8b`, and — crucially — clears any EOF/fail state and seeks
back to byte 0 so the decode path later sees the *whole* stream including that
header. The sniff is non-destructive; it only decides which path to take:

- **gzip magic present** → the compressed path. It heap-allocates a zeroed
  `z_stream` and calls `inflateInit2(strm, 15 + 32)`. The `15` is the maximum
  window size; the `+32` tells zlib to auto-detect and skip the gzip/zlib header
  on each member. A failed init throws
  `io::GzipLineReader: inflateInit2 failed for <path>` (and frees the stream
  first, so there is no leak on that error).
- **magic absent** → the plain-text path. No zlib stream is created; the file is
  read as raw bytes.

The plain-text path is not just defensive — it is what lets a by-construction
unit fixture (an ordinary uncompressed `.vcf`) exercise the reader without having
to gzip it first.

The destructor tears the zlib stream down symmetrically: if one was created it
calls `inflateEnd`, frees the heap `z_stream`, and nulls the handle. The class is
non-copyable (copy ctor and assignment are deleted), because it owns that raw
zlib resource and an `ifstream`.

Two fixed buffer capacities are set at construction: a 256 KB compressed-input
buffer (`in_`) and a 1 MB decode-output buffer (`out_`). The output buffer
doubles as the raw read chunk on the plain-text path.

---

## 4. `decode_more` — refilling the decoded byte buffer

`decode_more` is the private engine. Its job: append at least one more decoded
byte to the internal `buf_`, or report that the source is truly exhausted. It
returns `true` when it added bytes, `false` when there is nothing left ever. Once
it has set the `src_eof_` latch, every later call short-circuits to `false`.

**Plain-text path.** It reads one chunk (up to 1 MB) straight from the file into
`buf_`. Any bytes read → `true`. A zero-byte read means end of file → set
`src_eof_`, return `false`.

**Compressed path.** This is a loop, because a single `inflate` call might make no
visible progress (it consumed a header, or hit the empty BGZF EOF member), and the
contract is "return only when I actually produced output, or the file is genuinely
done." Each turn of the loop:

1. **Refill input if drained.** If zlib has no input bytes left (`avail_in == 0`),
   read up to 256 KB of compressed bytes from the file. A zero-byte read here
   means the file is spent → set `src_eof_`, return `false`.
2. **Inflate.** Point zlib at the full 1 MB output buffer and call
   `inflate(..., Z_NO_FLUSH)`. Whatever it produced is appended to `buf_`
   immediately, *before* interpreting the return code — so decoded bytes are never
   dropped on the floor even at a member boundary.
3. **Interpret the return code:**
   - `Z_STREAM_END` — one gzip/BGZF member finished. Call `inflateReset` to arm
     the stream for the *next* member and keep looping. If that member produced
     output, return `true` now; if it produced nothing (the classic case: the
     28-byte empty BGZF EOF marker), just `continue` to read whatever follows.
     The reset touches only zlib's internal state — the `buf_` carry is untouched,
     which is what lets a VCF line straddle a block boundary safely (section 5).
   - `Z_OK` / `Z_BUF_ERROR` — normal in-progress or "needs more buffer" states. If
     output was produced, return `true`. If not, it needs more input: loop again,
     unless the file is also at EOF with input drained, in which case the stream is
     done → set `src_eof_`, return `false`. (`Z_BUF_ERROR` is explicitly *not*
     treated as an error here; for a streaming inflate it just means "give me more
     to chew on.")
   - anything else (`Z_DATA_ERROR`, `Z_MEM_ERROR`, …) — a real corruption/failure.
     Throw `io::GzipLineReader: zlib inflate error (code <ret>) reading <path>`.

The key subtlety is that `inflateReset`-and-continue is what makes many concatenated
members read as one stream, and appending output *before* checking the code is what
keeps that seamless.

---

## 5. `next_line` — turning bytes into lines

`next_line(out)` is the only thing the caller uses. It hands back the next line,
stripped of its terminator, and returns `false` at end-of-stream.

It loops:

1. **Look for a newline** in the not-yet-searched tail of `buf_`, starting at the
   saved scan offset (`scan_`) so it never re-scans bytes it already looked at.
2. **Found one** → the line is everything up to that `\n`. If the byte just before
   it is `\r`, back up one so CRLF line endings are tolerated (Windows-authored
   VCFs work). Assign that slice into `out`, erase the consumed prefix (line +
   `\n`) from `buf_`, reset `scan_` to 0, and return `true`.
3. **No complete line yet** → remember the current buffer end as the new `scan_`
   (so the next search resumes there, not from the front), and call `decode_more`
   to pull in more bytes. If that succeeds, loop and search again.
4. **`decode_more` returned false (source exhausted)** → flush the tail. If `buf_`
   still holds bytes with no trailing newline, that is a legitimate final line:
   strip a trailing `\r` if present, return it once, and clear `buf_`. On the next
   call `buf_` is empty and it returns `false`.

The partial-line carry living in `buf_` — outside zlib and untouched by every
`inflateReset` — is what guarantees a VCF line that spans two BGZF blocks (or two
1 MB decode chunks) is reassembled whole rather than split.

---

## 6. Contracts and invariants

- **Lines are terminator-free.** The returned string never includes the trailing
  `\n`, and never a trailing `\r` (CRLF is normalized to just the content).
- **The final unterminated line is returned exactly once.** A VCF whose last line
  has no trailing newline still yields that line — a common real-world case — and
  yields it a single time; the following call returns `false`.
- **`false` is terminal.** Once `next_line` returns `false` (equivalently, once
  `src_eof_` is latched and `buf_` is empty), it keeps returning `false`. There is
  no reset/rewind; the reader is single-pass, forward-only.
- **The same interface for all three file shapes.** Plain text, single-member
  gzip, and multi-member BGZF are indistinguishable to the caller. Compression is
  decided once, at construction, by the magic sniff.
- **Errors are `std::runtime_error`.** Open failure, `inflateInit2` failure, and
  any hard zlib inflate error all throw with the path in the message. The caller
  is expected to let that propagate, not to poll a status flag.
- **Owns its resources, non-copyable.** The `z_stream` is heap-owned and released
  in the destructor; the reader cannot be copied.

---

## 7. Edge cases the code handles on purpose

- **The empty BGZF EOF member.** The 28-byte terminal block every `bgzip` file
  ends with decodes to zero bytes. That is a clean end, not an error: the
  `Z_STREAM_END`-with-zero-output branch just resets and reads on, and the outer
  loop reaches genuine EOF naturally.
- **A line spanning two blocks/chunks.** Handled by the out-of-zlib `buf_` carry
  (sections 4–5); the reset at a member boundary never disturbs buffered content.
- **CRLF endings.** Stripped in both the mid-stream and final-line paths.
- **A file that is not gzip at all.** Read verbatim as text — no failure, no
  attempt to inflate.
- **A truncated / corrupt gzip stream.** Surfaces as a thrown
  `std::runtime_error` carrying the zlib return code and the path, rather than a
  silent short read. (`Z_BUF_ERROR`, which is *not* corruption, is excluded from
  this and treated as "need more input.")
- **An empty file.** Open succeeds, the sniff reads fewer than two bytes (so the
  plain-text path is chosen), the first `decode_more` reads nothing, and
  `next_line` returns `false` immediately — zero lines, no error.
