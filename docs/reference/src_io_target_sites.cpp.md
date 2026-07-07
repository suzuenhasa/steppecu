# `target_sites.cpp` reference

## 1. Purpose

`src/io/target_sites.cpp` reads and indexes the **target-site table** — the fixed
panel of GRCh38 positions the native VCF reader genotypes a sample against. It is a
pure host, io-leaf translation unit: no GPU, no CUDA header, just file parsing and a
couple of small in-memory indexes.

Concretely it owns three things, all declared in `target_sites.hpp`:

- `read_target_sites(path)` — parse the on-disk table into a `TargetSites`.
- `is_palindrome(a1, a2)` — the strand-ambiguous SNP test.
- `build_chrom_index(ts)` — build the per-chromosome position index the VCF
  interval-join runs against.

The table it reads is the Stage-1 → Stage-2 boundary artifact: a panel that has
already been rsID-joined, lifted from GRCh37 to GRCh38, and had each site's GRCh38
REF base pre-fetched — but with palindromes *still in*. This reader does not do
liftover, dbSNP cross-checks, or dedup; those stay upstream. Its whole job is to
turn that table into the two data structures the reader in `vcf_reader.cpp` needs,
and to build them **exactly** the way the Stage-0 Python oracle does so the two
implementations agree site-for-site.

---

## 2. The on-disk table format

The table is whitespace- or tab-separated, one site per line, in two accepted
shapes:

| Layout | Columns |
|---|---|
| 7-column (canonical) | `rsID  chrom  pos37  pos38  A1  A2  ref38` |
| 6-column | `rsID  chrom  pos38  A1  A2  ref38` (no `pos37`; it defaults to 0) |

The reader is lenient about framing and strict about content:

- **Blank lines and `#` comments are skipped.** A line that is empty, or whose first
  character is `#`, is ignored.
- **A header row is skipped by name.** If the first token is literally `rsID` or
  `rsid`, that line is treated as a header and dropped. This lets a
  human-readable table (the kind `write_target_table` emits, section 7) round-trip
  through the reader untouched.
- **Field count decides the layout, per line.** A record with 7 or more fields is
  read as the 7-column layout (extra trailing fields are ignored); exactly 6 fields
  is the 6-column layout; anything else (1–5 fields, after tokenizing) throws.

The allele and REF bases are upper-cased on the way in (`A1`, `A2`, `ref38`), and
an empty `ref38` token becomes `'.'` — the "REF unavailable" sentinel that
`TargetSite.ref38` also defaults to. Only the first character of each base field is
taken.

---

## 3. Parsing failures throw, they don't limp on

`read_target_sites` is a hard boundary: a table it can't make sense of is a
`std::runtime_error`, never a partially-populated `TargetSites`. Three things raise:

1. **Can't open the file** — `cannot open target table: <path>`.
2. **Wrong field count** — a record that is neither 6 nor 7 fields:
   `malformed record (expected 6 or 7 fields) at line <n>`.
3. **Non-numeric chrom or position** — `chrom`, `pos37`, or `pos38` that
   `std::stoi` / `std::stoll` can't parse throws `std::invalid_argument`, which is
   caught and re-thrown as `non-numeric chrom/pos at line <n>`.

Every message carries the 1-based line number (comments and blanks are counted, so
the number matches what you'd see in an editor). The `std::invalid_argument` catch
is narrow on purpose: it converts the numeric-parse failure into a clear, located
message rather than letting a raw library exception escape.

Note what is *not* validated here: base letters other than A/C/G/T, a `ref38` that
doesn't match `A1`/`A2`, out-of-range chromosomes, negative positions. This reader
trusts the upstream pipeline that produced the table for biological sanity and only
guards the mechanical parse. Its contract is "this table is well-formed and
indexable," not "this table is biologically correct."

---

## 4. Palindromes: detected here, dropped downstream

`is_palindrome(a1, a2)` is the single home for the strand-ambiguous test. A site is
a palindrome when its two alleles are a complementary pair on the same strand —
`{A,T}` or `{C,G}`, in either order, case-insensitively. Those are the sites where
you can't tell which strand a genotype call came from, so they're unsafe to merge.

Two design points matter:

- **It's shared, not duplicated.** Both `read_target_sites` (here) and the Stage-2
  native builder `build_target_sites` (in `target_build.cpp`) call this same
  function, so the drop policy is single-homed — there is exactly one definition of
  "palindrome" across the ingest path.
- **Detection and dropping are separated.** This file only *flags* palindromes
  (`TargetSite.palindrome`) and, crucially, *excludes them from the chromosome
  index* (section 5). It never removes them from `TargetSites.sites` — the panel
  keeps its full length and original order. The actual drop happens later, in
  `vcf_reader.cpp`'s panel-order resolution loop, which sees `s.palindrome`, records
  a `"palindrome"` drop reason, and emits a missing call. Keeping palindromes in the
  site list but out of the index is what lets the reader report "dropped: palindrome"
  per site while never trying to genotype one.

---

## 5. The per-chromosome index (`build_chrom_index`)

This is the load-bearing part of the file. `build_chrom_index` builds
`TargetSites.by_chrom`: for each chromosome, a `ChromIndex` holding two structures
that the VCF interval-join depends on. It is built over the **non-palindromic sites
only** — palindromes are skipped up front.

### 5a. `pos` — the sorted position array (duplicates kept)

Every non-palindromic site's `pos38` is pushed into `ChromIndex.pos`, then that
vector is sorted ascending. **Duplicate positions are deliberately kept** — if two
target sites lifted to the same GRCh38 coordinate, both entries stay in `pos`.

This array exists so the reader can answer "which target sites does this VCF record
cover?" with a binary search. A gVCF ref-confidence block spans an interval
`[pos, END]`, and the reader uses `std::lower_bound` / `std::upper_bound` on `pos`
(with an *inclusive* END) to mark every covered target as observed in a coverage
bitmap. Because the bitmap is indexed by position-in-`pos`, keeping duplicates means
a block covering a collided coordinate correctly marks *both* colliding sites as
covered — the count and the coverage stay honest at collisions.

### 5b. `slot` — the position → index map (last-wins)

`ChromIndex.slot` maps each `pos38` to its index in the sorted `pos` array. It is
reserved at twice the position count (a load-factor headroom for the hash map) and
filled by walking `pos` in order, `slot[pos[i]] = i`. When a position appears more
than once, later writes overwrite earlier ones, so a duplicated `pos38` **resolves
to its highest index in `pos`** — "last-wins."

`slot` is the reader's O(1) membership-and-lookup structure. On the explicit-variant
path it answers "is this VCF position one of my targets?" (`slot.find(pos)`); on the
panel-order resolution pass it turns a site's `pos38` back into its coverage-bitmap
index (`slot.at(s.pos38)`). The last-wins rule means colliding sites share one
resolution slot — again matching the oracle so the two agree at collisions rather
than diverging on which duplicate "owns" the position.

### 5c. Why it mirrors the oracle exactly

The whole point of this construction — non-palindromic-only, duplicates kept in
`pos`, last-wins in `slot` — is that it reproduces the Stage-0 Python oracle's
`ppos` / `slot` build (oracle.py:143, 236–237) bit-for-bit. The sort order, the
handling of collisions, and which sites are indexed all have to match, because the
oracle is the parity standard the native reader is validated against. A different
but "reasonable" choice here (dropping duplicates, first-wins, indexing palindromes)
would silently diverge from the oracle at exactly the awkward edge cases.

### 5d. Shared with the native builder

`build_chrom_index` is called both by `read_target_sites` (after parsing a table)
and by `build_target_sites` in `target_build.cpp` (after assembling sites natively
from a `.snp` panel + lift map + FASTA). Both go through this one function, so a
table read from disk and a table built in memory carry an identical index. The
function **overwrites** `ts.by_chrom` (it clears first), so it's safe to call on a
`TargetSites` that already has a stale index.

---

## 6. The `TargetSites` value and its invariants

The result is a `TargetSites` with:

- `sites` — every parsed site in **panel order** (the order they appear in the
  table), palindromes included, full length preserved.
- `by_chrom` — the per-chromosome index of section 5, over non-palindromic sites
  only; `has_chrom(c)` is a convenience membership check.

Invariants a caller can rely on after `read_target_sites` (or `build_target_sites`)
returns:

- `sites` and the input table are in 1:1 order (minus skipped blanks/comments/header).
- Every non-palindromic site's `(chrom, pos38)` is present in
  `by_chrom[chrom].slot`, and `slot`'s value is a valid index into
  `by_chrom[chrom].pos`.
- `by_chrom[chrom].pos` is sorted ascending and its length equals the number of
  non-palindromic sites on that chromosome (duplicates counted).
- No palindromic site appears in any `by_chrom` entry — but it is still in `sites`.
- Allele and REF bases in `sites` are upper-cased; `ref38` is `'.'` when the table
  had no base.

---

## 7. `write_target_table` (the round-trip)

`target_build.cpp` also defines `write_target_table`, which emits the canonical
7-column table with the exact `rsID\tchrom\tpos37\tpos38\tA1\tA2\tref38` header that
`read_target_sites` knows to skip. That closes the loop: a table built natively can
be written out and re-read, and the reader's header-skip and 7-column parse are
tuned to accept precisely what the writer produces. (The writer lives with the
native builder rather than here because it pairs with `build_target_sites`; it's
noted here so the read/write symmetry is visible from one place.)

---

## 8. Edge cases worth knowing

- **A chromosome with only palindromic sites** gets no `by_chrom` entry at all —
  `has_chrom` returns false for it — because palindromes are skipped before any
  entry is created. The reader treats "no index for this chromosome" as "no targets
  here," which is correct: every site on it would be dropped anyway.
- **A 6-column table** leaves `pos37 = 0` for every site. Nothing downstream keys on
  `pos37` for the join (the index is built on `pos38`), so a 6-column table
  genotypes identically to its 7-column form; `pos37` is provenance only.
- **Extra trailing columns** on a 7+-field record are ignored, not an error — a
  table with additional annotation columns still reads, as long as the first seven
  are the canonical ones.
- **An empty table** (all comments/blanks, or a lone header) is not an error: it
  yields a `TargetSites` with empty `sites` and empty `by_chrom`. Whether that's
  meaningful is the caller's call, not the parser's.
- **Duplicate positions across different chromosomes** never interact — the index is
  keyed by chromosome first, so `chr1:12345` and `chr2:12345` live in separate
  `ChromIndex` structures.
