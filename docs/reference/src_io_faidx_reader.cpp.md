# `faidx_reader.cpp` reference

## 1. Purpose

`src/io/faidx_reader.cpp` is a small, native random-access reader for an indexed
FASTA file. Given a contig name and a 1-based position, it hands back the single
reference base that sits there — for us, the GRCh38 base at a lifted-over coordinate.

It exists to replace a shell-out. The original ingestion pipeline had a Stage-0
oracle that fetched reference bases by orchestrating `samtools faidx`. That worked,
but it meant spawning an external process (and depending on samtools being on the
box) just to read one byte per site. `FaidxReader` does the same job in-process, in
plain C++: no zlib, no CUDA, no subprocess. It is a pure host io-leaf.

Its one caller today is the Stage-2 native panel harmonizer (`target_build`), which
needs the GRCh38 reference base (`ref38`) for every lifted site — including the
block-interior sites the input VCF carries no base for — so it can reconcile alleles
and flag ref-changes. `cmd_ingest.cpp` constructs the reader from the `--fasta`
argument and threads it through.

---

## 2. What a `.fai` index is, and why we lean on it

A FASTA file stores sequence as text wrapped to a fixed line width, so the byte
offset of an arbitrary base is not obvious — you can't just add the position to a
base offset, because every wrapped line inserts newline bytes. The `.fai` sidecar
(the index `samtools faidx` writes next to the FASTA) records exactly the four
numbers you need to do that arithmetic, one line per contig:

```
NAME   LENGTH   OFFSET   LINEBASES   LINEWIDTH
```

- **`LENGTH`** — how many bases the contig has (used for range checks).
- **`OFFSET`** — the byte offset of the contig's very first base.
- **`LINEBASES`** — bases per wrapped line.
- **`LINEWIDTH`** — *bytes* per line, newline(s) included. So a Unix `\n` file has
  `LINEWIDTH == LINEBASES + 1`, and a `\r\n` file has `LINEWIDTH - LINEBASES == 2`.

With those four numbers, finding any base is O(1) seek-and-read arithmetic (section
4) — no scanning. That is the whole reason to require the index: it turns a
gigabyte-scale FASTA into random-access storage.

The reader does **not** build the index. If the `.fai` is missing, the constructor
throws with a message that tells the user exactly how to make one
(`run `samtools faidx` on <fasta> first`). Indexing is a one-time offline step; this
class only consumes the result.

---

## 3. Construction: parse the index, open the FASTA

`FaidxReader(fasta_path)` does two things and then it's ready to serve reads:

1. **Parse `<fasta_path>.fai` into an in-memory map.** It reads the sidecar line by
   line into an `unordered_map<name, FaiEntry>`, where `FaiEntry` is just the four
   `long long` fields above. Blank lines are skipped. Each non-blank line must yield
   all five whitespace-separated tokens (name + four numbers) or it's a malformed
   record and the constructor throws, naming the offending line number and file.
2. **Open the FASTA itself in binary mode** and keep the stream open for the lifetime
   of the reader, so every subsequent `base_at` is a seek on an already-open handle
   rather than an open/close per lookup.

The whole index lives in memory but it's tiny — a few dozen contigs, four numbers
each — while the FASTA payload is never slurped; it stays on disk and is touched one
byte at a time on demand (section 4).

### The sanity gate on each record

Beyond "did five tokens parse", the constructor rejects a record whose geometry is
nonsensical: `LINEBASES <= 0`, or `LINEWIDTH < LINEBASES`. Both would make the
section-4 arithmetic produce garbage (a zero or negative `LINEBASES` divides wrong; a
`LINEWIDTH` shorter than `LINEBASES` describes an impossible line). Catching it at
parse time means a bad index fails loudly at construction, not with a silently-wrong
base ten thousand lookups later. Note it does **not** reject `LINEWIDTH == LINEBASES`
(a FASTA with no line terminator on its lines) — that's unusual but arithmetically
coherent, so it's allowed through.

---

## 4. The offset arithmetic (`base_at`)

This is the heart of the file. To find the byte holding the base at 1-based `pos1`
on a contig with index entry `e`:

```
z    = pos1 - 1                                            // convert to 0-based
byte = e.offset + (z / e.linebases) * e.linewidth          // full lines before it, in bytes
                + (z % e.linebases)                         // bases into the current line
```

The first term is where the contig starts. The middle term counts how many complete
wrapped lines precede the target base (`z / linebases`) and multiplies by the *byte*
width of a line (`linewidth`, newlines and all) — this is what steps over the line
terminators correctly regardless of `\n` vs `\r\n`. The last term is the base's
column within its own line, added as raw bytes because within a line one base is one
byte.

Then it's a single `seekg(byte)` + `get()`. Before seeking it calls `fasta_.clear()`
to wipe any sticky EOF/fail bits from a prior read, so one lookup can never poison the
next.

The returned byte is upper-cased before it leaves (section 6).

---

## 5. Contig-name tolerance (`resolve`)

Reference assemblies disagree about whether chromosomes carry a `chr` prefix: an
Ensembl FASTA names them `1`, `2`, …, while a UCSC one names them `chr1`, `chr2`. Our
inputs (the nikki panel, the lift map, the Ensembl FASTA) are all *unprefixed* per
the spec, but we want to tolerate a prefixed FASTA rather than fail on a naming
mismatch.

So `resolve(contig)` — the private lookup behind both `has_contig` and `base_at` —
tries the name as-is first, and if that misses, toggles the `chr` prefix and tries
once more:

- a query starting with `chr` is retried with the prefix stripped (`chr1` → `1`);
- a query without it is retried with the prefix added (`1` → `chr1`).

That single toggle covers both directions. It returns a pointer to the `FaiEntry` on
a hit, or `nullptr` on a miss — `has_contig` just reports whether that pointer is
non-null, and `base_at` turns a null into a thrown "unknown contig" error.

---

## 6. Base normalization: upper-case yes, `N` passes through

Every returned base is upper-cased (`up()`). FASTA files use lowercase to soft-mask
repetitive regions, but for our allele reconciliation a soft-masked `a` and a hard
`A` are the same base — so the mask is folded away on the way out and the caller only
ever sees upper-case letters.

Crucially, a literal `N` in the assembly is **not** special-cased — it's a letter, it
upper-cases to `N`, and it's returned as `N`. That's deliberate: downstream,
`ref38 == 'N'` flows into the reconcile step and becomes a `ref_change`, exactly
matching the behavior of the samtools-based oracle it replaced. The reader's job is to
report the assembly faithfully, not to editorialize about ambiguous bases.

---

## 7. Range checks: a past-end lift is corrupt input, and it's surfaced

`base_at` bounds-checks `pos1` against the contig's `LENGTH` before doing any
arithmetic: `pos1 < 1` or `pos1 > e.length` throws, with a message naming the
position, the valid `[1, length]` range, the contig, and the file.

This is a design stance worth calling out. A liftover that lands past the end of a
contig is corrupt input — a real signal that something upstream is wrong. The reader
refuses to paper over it by, say, returning `N`. It surfaces it as an error so it
gets noticed and fixed, rather than silently contaminating the panel with a made-up
base. The same philosophy covers the low-level read failure: if the seek lands
somewhere the `get()` can't read a byte (`c < 0`), that too throws, naming the byte
offset, contig, and position — a truncated or corrupt FASTA is reported, never
guessed around.

---

## 8. Contracts and invariants

- **Index required.** The `.fai` must exist and parse, or construction throws. The
  reader never builds or repairs the index.
- **1-based positions.** `pos1` is 1-based (matching VCF/FASTA convention and the
  liftover coordinates), and must lie in `[1, LENGTH]`.
- **Upper-case out, `N` preserved.** Every returned base is upper-case; `N` is a
  legitimate returned value.
- **Errors, not sentinels.** Unknown contig, out-of-range position, and read failure
  are all thrown `std::runtime_error`s carrying enough context (file, contig,
  position, byte) to diagnose. Nothing is silently masked.
- **NOT thread-safe.** `base_at` is a `const` method, but it seeks and reads a
  `mutable` `ifstream` — so two threads sharing one `FaidxReader` would race the
  stream's position. A single reader is single-thread-only. If you need concurrent
  reads, give each thread its own reader (each opens its own file handle over the same
  FASTA). The `const` signature advertises "doesn't change the logical index", not
  "safe to share across threads".

---

## 9. Why it's built this way

The reader is deliberately minimal. It holds one small map and one open file handle,
does one seek-and-read per lookup, and links nothing heavy. That keeps it a true
io-leaf: no zlib to decompress a `.gz` (we require a plain, indexed FASTA), no CUDA,
no process spawning. All the cost that matters — reading the reference — is a single
`pread`-shaped seek that the OS page cache handles well when many nearby sites are
queried in sequence, which is exactly the access pattern `target_build` produces as
it walks a chromosome's sites in order.
