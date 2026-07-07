# `gzip_line_reader.hpp` reference

## 1. Purpose

`src/io/gzip_line_reader.hpp` declares `GzipLineReader`, a small streaming
line-at-a-time reader that sits under Stage-1 VCF ingestion. You point it at a
path and pull lines out one by one with `next_line`; it hands back each line with
the trailing newline stripped, and returns `false` when the stream is done.

It reads three kinds of file the same way, transparently:

- a plain-text `.vcf`,
- a plain-gzip `.vcf.gz`,
- a **BGZF** `.vcf.gz` — the block-gzipped format `bgzip` writes, which is what
  most real VCFs ship as.

The caller never has to know which one it got. `GzipLineReader` sniffs the file and
does the right thing.

Two properties follow from where this file sits in the layering:

- **Pure host C++20 io-leaf.** It is a plain host file that links only zlib for the
  decompression. There is no CUDA here, no `RunConfig`, nothing from the core or
  device layers — it is the kind of cheap-to-include, cheap-to-test leaf the `io`
  layer is full of. The header itself doesn't even pull in `<zlib.h>`: the zlib
  `z_stream` is held as an opaque `void*` and the concrete type lives entirely in
  the `.cpp` (section 6).
- **Failures surface as `std::runtime_error`.** Every way a read can go wrong — a
  file that won't open, a zlib init failure, a corrupt compressed stream — comes
  back to the caller as a thrown `std::runtime_error` carrying the path, matching
  the rest of the `io` readers' exception contract.

The reader owns an open file stream and a zlib decode handle, so it is **move-free
and non-copyable**: the copy constructor and copy assignment are deleted. You open
one, stream it to exhaustion, and let it go.

---

## 2. The one job: sequential streaming, no random access

Stage-1 VCF ingestion reads a VCF start to finish, once. It never needs to jump to
a genomic region. That single fact is what makes this reader simple and is worth
stating plainly, because BGZF exists precisely to *enable* random access (via a
`.tbi` tabix index), and this reader deliberately does **not** use any of that
machinery.

There is no tabix index, no virtual-offset seeking, no per-block bookkeeping.
Because we only ever read forward, a BGZF file can be treated as nothing more than
a plain gzip stream and inflated straight through. Section 3 explains why that works
and the one subtlety it introduces.

---

## 3. Why a plain inflate reads BGZF transparently (the key design decision)

A BGZF file is a **concatenation of independent gzip members**: `bgzip` chops the
data into ~64 KB blocks and gzip-compresses each block on its own, then writes them
back-to-back. Critically, every one of those blocks is itself a completely valid,
standalone gzip member.

The gzip spec says a decompressor should treat concatenated members as one logical
stream — inflate member 1, then member 2, and so on, gluing their decompressed
output together. So a plain zlib inflate over the whole file already produces the
right bytes for a BGZF file; we just have to drive the loop across the member
boundaries ourselves. The mechanics (section 6):

- When zlib returns `Z_STREAM_END`, that's the end of *one* member, not
  necessarily the end of the file. If input remains, we call `inflateReset` and
  keep going into the next member.
- The output we've decoded so far lives in a buffer that is **completely untouched
  by the reset**. This is the load-bearing detail. A VCF line can straddle a BGZF
  block boundary — the line's first half decodes out of block N and its second half
  out of block N+1 — and because the partial-line carry lives outside the zlib state
  we reset, the two halves are simply concatenated in the buffer and the line is
  never split.

For a plain gzip file the same loop runs and just happens to hit `Z_STREAM_END`
exactly once, at the true end. For a plain-text file we skip zlib entirely and read
raw chunks. All three paths feed the same line-splitting buffer, so the line-level
behavior is identical regardless of the source encoding.

zlib is initialized with a window-bits argument of `15 + 32`: `15` is the maximum
window, and the `+32` turns on zlib's automatic gzip-header detection, so it parses
each member's gzip header for us.

---

## 4. What `next_line` guarantees

```cpp
[[nodiscard]] bool next_line(std::string& out);
```

Fetch the next line into `out` and return `true`; return `false` at end of stream.
The contract, precisely:

- **The trailing newline is stripped.** The `\n` is removed, and a preceding `\r`
  is removed too, so a Windows CRLF (`\r\n`) VCF is handled the same as a Unix one.
  `out` never contains the line terminator.
- **The final line is returned even with no trailing newline.** Many VCFs (and the
  hand-built unit fixtures) don't end with a newline. The last, un-terminated line
  is flushed and returned exactly once, then the following call returns `false`.
- **Empty lines are real lines.** A blank line comes back as an empty `out` with a
  `true` return; it is not skipped and not confused with end-of-stream. End-of-stream
  is signalled *only* by the `false` return.
- **`[[nodiscard]]`.** The return value is the loop condition — ignoring it would
  mean you never learn the stream ended — so the compiler flags a dropped result.

The intended use is the obvious `while` loop:

```cpp
GzipLineReader reader(path);
std::string line;
while (reader.next_line(line)) {
    // handle line
}
```

This is exactly how `vcf_reader.cpp` drives it across two passes (the header scan
and the record scan).

---

## 5. Construction, sniffing, and the plain-text fallback

```cpp
explicit GzipLineReader(const std::string& path);
```

Opening a reader:

1. Opens the file in binary mode. If it won't open, throws `std::runtime_error`
   naming the path — this is the first and cheapest failure.
2. **Sniffs the two-byte gzip magic** (`0x1f 0x8b`) from the front, then rewinds to
   offset 0 so the decode path later sees the whole stream, header and all. If the
   magic is present the reader takes the zlib path; if not, it takes the plain-text
   path.
3. On the gzip path only, allocates and initializes the zlib decode handle. An
   `inflateInit2` failure throws (after cleaning up the half-built handle).

The plain-text fallback is deliberate, not an accident of leniency. The unit
fixtures for VCF ingestion are hand-written uncompressed `.vcf` text, and being able
to read those by construction — through the exact same `next_line` line-splitting
logic — is what lets the tests exercise the line handling without a compression
dependency. A file that merely lacks the gzip magic is read as text rather than
rejected.

The destructor tears down the zlib handle (`inflateEnd` plus the heap delete) when
one was allocated, and is a no-op on the plain-text path.

---

## 6. Inside the decode loop (`decode_more`) and the edge cases it handles

`next_line` keeps a running buffer of decoded-but-not-yet-returned bytes and
searches it for the next `\n`. When there's no complete line buffered, it calls the
private `decode_more` to pull in more bytes, and only gives up (flushing any final
partial line, then returning `false`) when `decode_more` reports the source is truly
exhausted. The decode loop is where the section-3 member handling and several sharp
edge cases are actually resolved:

- **A line spanning a block boundary** — handled by the buffer surviving
  `inflateReset` (section 3). The carry is never dropped.
- **The BGZF EOF marker.** Every well-formed BGZF file ends with a special 28-byte
  empty block — a valid gzip member that decompresses to *zero* bytes. Decoding it
  yields `Z_STREAM_END` with no output. The loop treats a zero-output member as a
  clean "read the next member" step, not as end-of-data and not as an error, so the
  EOF block is consumed harmlessly and the stream ends only when the input is
  actually gone.
- **`Z_BUF_ERROR` is not fatal.** zlib returns `Z_BUF_ERROR` to mean "I made no
  progress; feed me more input," which is normal mid-stream. The loop treats it like
  `Z_OK`: if no bytes were produced and the input is drained and the file is at EOF,
  the source is done; otherwise it loops to pull the next input chunk.
- **A genuinely corrupt stream** — a `Z_DATA_ERROR`, `Z_MEM_ERROR`, or any other
  hard zlib code — throws `std::runtime_error` with the zlib code and the path. A
  malformed `.gz` fails loudly rather than returning truncated data.

The loop is written to always make forward progress or definitively stop: it returns
`true` only after appending at least one decoded byte, and sets its end-of-source
flag exactly when input and file are both exhausted, so `next_line` can never spin
forever on a dead stream.

---

## 7. Contracts and invariants at a glance

- **Move/copy.** Non-copyable (copy ctor and copy assignment deleted). It owns a
  file stream and a zlib handle; open one, stream it, drop it.
- **Line terminators.** `\n` and a preceding `\r` are stripped; `out` never holds a
  terminator. CRLF and LF are treated identically.
- **End of stream.** Signalled only by `next_line` returning `false`. The final
  un-terminated line and every empty line are returned as normal `true` lines first.
- **Encoding transparency.** Plain text, plain gzip, and multi-member BGZF all
  produce identical line-level output; the caller cannot tell them apart.
- **No random access.** Forward streaming only — no tabix, no seeking. This is a
  deliberate scope choice, not a limitation to be worked around (section 2).
- **Errors.** Open failure, zlib init failure, and stream corruption all surface as
  `std::runtime_error` carrying the path; nothing else leaks out.
