# `vcf_reader.hpp` reference

## 1. Purpose

`src/io/vcf_reader.hpp` declares `VcfReader`, steppe's native reader for a single
sample's gVCF-style `.vcf.gz` — the sixth "reader arm" alongside the TGENO, PACKED-
ANCESTRYMAP, EIGENSTRAT, PLINK and ANCESTRYMAP paths. Its job is narrow and exact:
take one person's variant-call file and a fixed table of GRCh38 target sites, and
decide, site by site, what allele dosage that person carries at each site — or that
the site can't be called and is missing/dropped.

The thing that makes this reader special is that a gVCF doesn't spell out every
position. It records **variant records** where the sample differs from the
reference, and **reference-confidence blocks** — big `ALT=.` spans that say "across
this whole interval the sample simply matched the reference." So a target site is
answered not by looking it up directly but by an **interval join**: which record,
of all the records in the file, covers this site, and what does that record imply.

The whole design goal is a faithful port of the Stage-0 oracle
(`experiments/nikki-stage0/oracle.py`, `resolve()`). The header comment is emphatic
about this: every classification rule here — the hom-ref reconciliation, the depth
/genotype-quality floor, the multiallelic normalization, the rsID cross-check, the
palindrome drop, the strand-flip reconciliation — reproduces a specific oracle
branch so the native reader and the Python oracle agree call-for-call. The
state-machine implementation lives in `vcf_reader.cpp`; this header is the public
contract — the types, the options, and the resolution guarantees callers depend on.

This is a pure host, C++20 io-leaf. It links only zlib (via `GzipLineReader`), never
touches the GPU, and reports every failure as a `std::runtime_error`.

---

## 2. The output types

`genotype()` returns a `VcfIngestResult` bundling four things:

| Field | What it holds |
|---|---|
| `tile` | The canonical individual-major `SnpMajorTile` for the one sample — one dosage code per target site, in panel order. This is what feeds the shared device transpose and, downstream, the f2/f-stat engine. |
| `calls` | One `VcfSiteCall` per target site, again in panel order — the human-readable per-site report. |
| `counts` | A `VcfCounts` tally for the stderr summary and the recovered/dropped split. |
| `sample_id` | The sample this result is for (the resolved column name). |

### `VcfCall` — the resolved class

```
enum class VcfCall { Homref, Het, Homalt, Missing, Dropped };
```

The load-bearing rule, stated right on the enum: **this label is never recomputed
from the dosage.** It is the class the resolver committed to. A hom-ref block whose
REF equals the panel's A2 allele is `Homref` with dosage `0` — the "homref" comes
from *how the site was resolved*, not from the number `0`. Keeping the label and the
number independent is exactly what lets steppe match the oracle's report wording.

The distinction between `Missing` and `Dropped` also matters:
- **`Missing`** — the site exists in the panel but this sample can't be called there
  (no coverage, below the depth/quality floor, a half or `./.` genotype, an allele
  that isn't in the panel). It stays in the report as an NA.
- **`Dropped`** — the site is disqualified on principle, independent of this sample's
  data: a palindrome, an rsID that maps to a different variant, or a reference-base
  change. These are panel-integrity drops.

### `VcfSiteCall` — one report row

The report schema mirrors the oracle's exact column set (schema §9 of the Stage-0
spec): the site identity (`rsid`, `chrom`, `pos37`, `pos38`, panel alleles `a1`/`a2`),
the resolved `call`, the `dosage` (copies of panel A1 in `{0,1,2}`, or `-1` for NA),
the `source` (`refblock` | `variant` | `none`), a `flip` flag (1 when a strand flip
was applied), and a `drop_reason` string (`""` when called; otherwise `palindrome`,
`rsid_mismap`, `ref_change`, `below_floor`, `no_coverage`, and so on).

### `VcfCounts` — the tally

A flat struct of `long long` counters split three ways: the **called** sites (by
source and by class — `called_variant`, `called_refblock`, `homref`/`het`/`homalt`),
the **missing** sites broken out by cause (`missing_below_floor`, `missing_not_pass`,
`missing_non_panel`, `missing_no_coverage`, `missing_half_or_missing_gt`,
`missing_no_refbase`), and the **dropped** sites (`drop_palindrome`,
`drop_rsid_mismap`, `drop_ref_change`). Two bookkeeping counters — `records_seen` and
`variant_at_target` — track how much of the file was walked. The breakdown is fine-
grained on purpose: it's what tells a user *why* a run recovered fewer sites than
they hoped.

---

## 3. Construction and the `Options` floor

```
VcfReader(std::string vcf_path, const TargetSites& targets,
          std::string sample_id, Options opts);
VcfIngestResult genotype();
```

The constructor just captures its inputs; all the work happens in `genotype()`. The
`targets` reference is **borrowed, not owned** — the caller (see `cmd_ingest.cpp`)
must keep the `TargetSites` table alive for the reader's lifetime.

`Options` carries the quality floor and one scope switch:

| Option | Default | Meaning |
|---|---|---|
| `min_dp` | `8` | The depth floor. A reference block needs `MinDP` (or `FORMAT/DP`) at least this; a variant record needs `DP` at least this. Below it, the site is missing, not called. |
| `min_gq` | `20` | The genotype-quality floor for a variant record's `GQ`. |
| `autosomes_only` | `true` | Targets are chromosomes 1–22, so a non-1..22 CHROM is simply skipped. |

The floor is the H4 rule — coverage alone isn't enough; a site only counts as
observed if the underlying record clears both thresholds.

### Selecting the sample

`sample_id` names which column to read. The contract:
- **Empty string** means "the sole sample." If the `#CHROM` header carries more than
  one sample column, that's a `std::runtime_error` telling the user to pass a name —
  an empty string is not "pick the first one," it's "there had better be exactly one."
- **A non-empty name** is matched against the sample columns; a name not present in
  the header is a `std::runtime_error`.

The name that ends up in `VcfIngestResult::sample_id` is the resolved column name
either way.

---

## 4. The one-pass-then-resolve shape

`genotype()` runs in two phases, and understanding the split explains the memory
story:

1. **Stream the file once** to build coverage state. The reader can't answer a site
   until it has seen every record that might cover it, and a reference block can span
   far past the site it covers. So it makes a single pass, and as it goes it fills:
   - two per-chromosome coverage bitmaps — `pass_cov` (covered by a block that clears
     the floor) and `fail_cov` (covered only by a below-floor / non-PASS block) —
     indexed over the sorted target positions, and
   - a per-position **variant map** keeping, for each target position, the single
     best explicit variant record seen there.
2. **Resolve every target site in panel order**, consulting that state. This is the
   phase that produces the tile and the report rows.

Only variant records *at a target position* are kept; a variant off-panel is
discarded immediately. Reference blocks are collapsed into two bits per covered
position rather than stored as intervals. So the memory the reader holds is
proportional to the panel, not to the (potentially enormous) file.

---

## 5. The resolution precedence

This is the contract the whole reader exists to implement. For each panel site, in
strict order:

```
palindrome?  >  explicit variant record  >  passing ref block  >  failing ref block  >  no coverage
```

1. **Palindrome** — a strand-ambiguous A/T or C/G site is dropped up front
   (`drop_reason = "palindrome"`), `source` stays `"none"`, dosage NA. These sites
   never even had a `by_chrom` index entry built for them (see §7).
2. **Explicit variant record wins over any block.** If a variant record exists at
   the site, it is resolved and no block is consulted — a variant call is a
   stronger statement than "the sample matched reference across a span."
3. **Passing reference block** — the site sits inside a block that cleared the floor
   and the sample carries no variant there, so it is **hom-ref**. The dosage depends
   on which panel allele the GRCh38 reference base equals: `2` if REF is A1, `0` if
   REF is A2 (the H1 reconciliation), with a strand flip applied if the match is only
   on the complement.
4. **Failing reference block** — covered, but only by a block below the floor or not
   PASS. The site is `Missing` with `source = "refblock"`, `drop_reason =
   "below_floor"`. We know the sample was *sequenced* there but not *well enough*.
5. **No coverage** — no record of any kind touches the site. `Missing`, `drop_reason
   = "no_coverage"`.

### Inside the variant path

When a variant record resolves (step 2), it first clears the floor (PASS, `DP ≥
min_dp`, `GQ ≥ min_gq`) or the site is missing with the matching reason. Then the
diploid genotype is split, and each allele is reconciled against the panel's A1/A2 —
on the same strand first, then the complement. The dosage is the count of A1 copies,
and the class follows directly: two A1 copies is hom-ref-from-A1's-perspective, one
is het, zero is hom-alt. Anything that can't be normalized to a single panel base — a
spanning deletion `*`, an indel, a half-call, a `./.`, an allele index out of range,
an allele matching neither panel base even after complementing — resolves to
`Missing` with a specific reason (the M4 multiallelic-normalization rules), never a
silent wrong call.

When two variant records land on the same target position, the reader keeps the
**better** one: PASS beats non-PASS, then higher GQ wins — with the deliberate quirk
that a `GQ=0` field is treated as "no GQ" so it can't out-rank a record that carries
no GQ at all.

---

## 6. Lazy `source` / `call` assignment (the subtle part)

The header calls this out explicitly, and it is the easiest thing to get wrong.
The report's `source` and `call` fields reproduce the oracle's **lazy** field
assignment: `source` reads `"none"` until the resolver actually commits to a path.

The consequence that matters: the **rsID cross-check happens before `source` is set
to `"variant"`.** So when a variant record at a target position carries an `rs...` ID
that disagrees with the panel's rsID for that position, the site is dropped with
`source = "none"`, `call = Dropped`, `drop_reason = "rsid_mismap"` — *not* `source =
"variant"`. A reader that eagerly stamped `source = "variant"` the moment it found a
record would diverge from the oracle here. Likewise a hom-ref block with `REF == A2`
reports `call = "homref"` and dosage `0` — the label comes from the resolution path,
never derived back out of the `0`.

This is why §2 insists the label is never recomputed from the dosage: the report is a
record of *how* each site was decided, and matching the oracle means matching that
decision trail, not just the final number.

---

## 7. Contracts and invariants

- **`calls` and `tile.snp_major` are in panel order and the same length.** Every
  target site produces exactly one report row and exactly one tile code — including
  dropped and missing sites (they get `kMissingCode` in the tile). The two never
  drift out of lockstep.
- **A missing/dropped site is `kMissingCode` in the tile; a called site is its
  dosage.** The dosage byte written to the tile is the same `{0,1,2}` reported in the
  row.
- **The target index is built over non-palindromic sites only.** `TargetSites::
  by_chrom` (and its sorted `pos` array and last-wins `slot` map) deliberately exclude
  palindromes, mirroring the oracle's `ppos`/`slot` construction — so interval-join
  coverage and slot lookup agree with the oracle even at colliding lifted positions.
- **The reader does not do liftover, dbSNP-position cross-checks, or 1:1 dedup.** By
  the time a `TargetSites` reaches this reader it is already rsID-joined and lifted
  GRCh37→GRCh38 with `ref38` pre-fetched; those upstream steps stay orchestrated
  (shared with the oracle) at Stage 1. This reader's contract starts at "given clean
  targets, resolve them."
- **The final tile is packed for the shared device transpose.** After resolution each
  per-site code is packed into slot 0 of a single source byte
  (`src_bytes_per_record == 1`), with `n_individuals = 1`, `sel_rows = {0}`, and the
  sample label carried through — the same canonical individual-major shape the other
  reader arms hand to the device transpose.

---

## 8. Edge cases the reader handles deliberately

- **`chr`-prefixed chromosomes** are accepted: a leading `chr`/`CHR` is stripped
  before the CHROM is parsed to an autosome number. `X`/`Y`/`MT`/anything non-numeric
  is out of scope and skipped (M1).
- **Truncated records** — a data line with fewer columns than the sample column — are
  skipped rather than crashing the parse.
- **Reference-block END is inclusive.** A block's interval covers `[POS, END]`
  inclusive; the join uses `upper_bound(END)` so the END position itself is covered.
  A block with no `END` covers just its single POS.
- **Block depth falls back gracefully:** `MinDP` from INFO if present, otherwise
  `FORMAT/DP`. Absent depth means the block can't clear the floor and lands in
  `fail_cov`.
- **An unavailable reference base vs. a real `N`.** A target whose `ref38` is `'.'`
  (the FASTA fetch returned nothing) inside a passing block is `Missing` /
  `no_refbase`. A genuine assembly `'N'` is *not* `no_refbase`: it flows into
  reconciliation, matches neither panel allele, and drops as `ref_change` — exactly
  the oracle's distinction (its `no_refbase` fires only on an empty FASTA fetch).
- **Exact field-boundary matching.** INFO keys match up to `=` and FORMAT keys match
  a whole `:`-token (via `vcf_record.hpp`'s helpers), so a bare `DP` never matches
  inside `MinDP` and a FORMAT `DP` never hits `AD` — the boundary robustness the
  oracle got for free from bcftools accessors.

---

Related headers: `io/target_sites.hpp` (the `TargetSites`/`ChromIndex`/`TargetSite`
input table and the shared palindrome/index-build contract), `io/vcf_record.hpp` (the
field-exact text-parse helpers), `io/snp_major_tile.hpp` (the output tile layout),
and `io/gzip_line_reader.hpp` (the zlib line source). The `genotype()` state machine
itself is documented alongside the code in `vcf_reader.cpp`.
