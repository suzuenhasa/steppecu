# `faidx_reader.hpp` reference

## 1. Purpose

`src/io/faidx_reader.hpp` declares `FaidxReader`, a small host-only reader that
pulls a single reference base out of an indexed FASTA at a coordinate you name.
Give it a `.fasta` path with a sibling `.fai` index and it will answer "what base
sits at position 12,345 on chromosome 1?" with one disk seek and a one-byte read.

It exists to replace an external tool. The VCF-ingestion pipeline used to shell out
to `samtools faidx` at Stage 0 (the `oracle.fetch_ref_bases` step) just to learn the
GRCh38 reference base for each locus. `FaidxReader` does that same job in-process
C++, so the Stage-2 native panel harmonizer (`target_build`) can fill the per-locus
`ref38` column itself without orchestrating an external binary. For a block-interior
site the VCF carries no base for, `target_build` calls `base_at(chrom, pos38)` and
writes the result straight into the assembled `TargetSites`.

This is a pure host io-leaf. No zlib, no CUDA, no GPU header — just `<fstream>`,
`<string>`, and `<unordered_map>`. It reads plain (uncompressed) FASTA.

The implementation lives in `src/io/faidx_reader.cpp`; this one doc covers both the
header's contract and the `.cpp`'s mechanics, since the `.cpp` is a straight
realization of what the header declares.

---

## 2. What a `.fai` index is, and the byte math

A FASTA index (`.fai`) is a tiny sidecar text file with one line per contig and five
whitespace-separated columns:

```
NAME  LENGTH  OFFSET  LINEBASES  LINEWIDTH
```

| Column | Meaning |
|---|---|
| `NAME` | the contig name (e.g. `1` or `chr1`) |
| `LENGTH` | number of bases in the contig |
| `OFFSET` | byte offset of the contig's first base within the FASTA |
| `LINEBASES` | bases per wrapped line |
| `LINEWIDTH` | bytes per line *including* the newline(s) — so a `\r\n` file has `LINEWIDTH - LINEBASES == 2` |

That last distinction is the whole reason the index exists: because FASTA wraps
sequence across fixed-width lines, the byte position of a base is not simply
`OFFSET + pos`. You have to skip over the line-ending bytes for every full line you
cross. For a 1-based position `pos1`, with `z = pos1 - 1` the 0-based offset:

```
byte = OFFSET + (z / LINEBASES) * LINEWIDTH + (z % LINEBASES)
```

The `z / LINEBASES` full lines each cost `LINEWIDTH` bytes (payload + newline),
and the remainder `z % LINEBASES` lands you inside the current line. `base_at` seeks
to that byte and reads one character. This is the standard faidx arithmetic — the
same formula `samtools` uses — which is what keeps `FaidxReader`'s answers identical
to the tool it replaces.

---

## 3. Construction: parse the index, open the FASTA

`FaidxReader(fasta_path)` does all its parsing up front:

- It opens `fasta_path + ".fai"`. If that sidecar is missing, construction throws a
  `std::runtime_error` whose message tells you to run `samtools faidx` on the FASTA
  first — the one setup step this reader can't do for you.
- It reads the `.fai` line by line into an `unordered_map<string, FaiEntry>`, keyed
  by contig name. Blank lines are skipped. A line that doesn't yield all five fields
  is a malformed record and throws, naming the line number.
- It sanity-checks the geometry of each record: `LINEBASES` must be positive and
  `LINEWIDTH` must be at least `LINEBASES`. A line-width smaller than the line-bases
  is nonsensical (it would mean a line holds more bases than it has bytes) and throws
  rather than silently producing garbage offsets later.
- Finally it opens the FASTA itself in binary mode and keeps that stream open for the
  reader's lifetime. Binary mode matters: on the byte arithmetic above, no newline
  translation is allowed to shift the offsets.

After the constructor returns, the object is ready and every `base_at` is a cheap
seek — no re-parsing, no re-opening.

---

## 4. Contig-name tolerance (the `chr` toggle)

Real-world inputs disagree about whether chromosomes carry a `chr` prefix. The nikki
inputs, the panel, and the Ensembl FASTA are all *unprefixed* (spec §1), but a user
might point the reader at a prefixed assembly. So `resolve(contig)` tries the name
two ways:

1. the name exactly as queried, then
2. the name with its leading `chr` toggled — strip `chr` if present, add `chr` if
   not.

So a query for `"1"` resolves a contig stored as `"chr1"`, and a query for `"chr1"`
resolves one stored as `"1"`. Only these two forms are tried; there's no fuzzy
matching beyond the single prefix toggle. `has_contig(contig)` is just "does
`resolve` find something", and it applies the same toggle, so it agrees with what
`base_at` would accept.

---

## 5. What `base_at` returns, and how it normalizes

`base_at(contig, pos1)` returns a single `char`: the reference base, **upper-cased**.

Two normalization rules matter for parity with the `samtools faidx` oracle it
replaces:

- **Soft-mask is folded away.** FASTA assemblies lower-case soft-masked (repeat)
  regions. `base_at` upper-cases every returned base, so a soft-masked `a` and a hard
  `A` are indistinguishable to the caller — exactly what the downstream reconcile
  logic wants.
- **A literal `N` is returned as `N`.** Where the assembly genuinely has no called
  base, the reader hands back `'N'` (upper-cased like everything else). It does *not*
  invent a base or skip the site. That `N` flows downstream into reconcile, where it
  becomes a `ref_change`, matching the oracle's behavior.

The base is fetched by the section-2 byte math: seek to the computed byte, `get()`
one character, upper-case it, return it.

---

## 6. Error handling: bad input is surfaced, never masked

`base_at` throws `std::runtime_error` in three situations, each naming the contig,
position, and FASTA path so the failure is diagnosable:

1. **Unknown contig** — `resolve` finds no match even after the `chr` toggle.
2. **Position out of range** — `pos1 < 1` or `pos1 > LENGTH` for that contig. This is
   deliberate and load-bearing: a lift that lands past the end of a contig is corrupt
   input, and the reader refuses it loudly rather than quietly returning `'N'`. The
   `N`-as-`N` rule of section 5 is only for bases the *assembly* marks unknown, never
   for coordinates that fall outside the sequence.
3. **Read failure** — the seek+`get()` came back with EOF/error (a negative
   character) despite an in-range position, which would indicate a truncated FASTA or
   an index that disagrees with the file.

Before each seek, `base_at` calls `fasta_.clear()` to reset any sticky stream state
(a previous EOF, say) so one failed read can't poison later ones.

---

## 7. Thread-safety: single-thread only

`base_at` is declared `const`, but it is **not** thread-safe. The FASTA stream member
is `mutable`, and `base_at` seeks and reads it — so two threads calling `base_at` on
the same `FaidxReader` would race the shared stream position and hand each other the
wrong bytes.

The rule is one `FaidxReader` per thread. The `const`-but-mutating design is a
convenience for the single-threaded harmonizer that owns one reader and pulls bases
in a loop; it is not a promise of concurrent access. If a parallel ingest ever needs
this, give each worker its own `FaidxReader` (each opens its own `ifstream`) rather
than sharing one.

---

## 8. Contracts and invariants at a glance

- Construction throws if the `.fai` is missing/malformed or the FASTA can't be
  opened; a successfully constructed reader has a fully parsed index and an open
  binary FASTA stream.
- Coordinates are 1-based on input. Valid range is `[1, LENGTH]` for the resolved
  contig; anything outside throws.
- Returned bases are always upper-case; `N` passes through as `N`; soft-mask is
  folded to upper.
- Contig lookup tolerates exactly one `chr`-prefix toggle and nothing more.
- The reader holds one open FASTA stream and is single-thread-only for the life of
  the object.
- Pure host code — no CUDA, no compression, no external process.
