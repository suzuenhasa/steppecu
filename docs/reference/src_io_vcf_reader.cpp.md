# `vcf_reader.cpp` reference

## 1. Purpose

`src/io/vcf_reader.cpp` is steppe's native gVCF-aware VCF reader — the "sixth
reader arm" alongside the TGENO / PACKEDANCESTRYMAP / EIGENSTRAT / PLINK /
ANCESTRYMAP genotype readers. Where those read a whole cohort out of a packed
genotype file, this one does something narrower and fussier: it takes **one
sample's** `.vcf.gz` and genotypes it against a fixed GRCh38 target panel,
producing exactly the two things the rest of the pipeline wants — a canonical
individual-major `SnpMajorTile` for that one sample (later handed to the shared
device transpose) and a per-site report row for every panel site.

This is a pure host translation unit. It touches no GPU, launches no kernel. It
gunzips and text-parses a VCF with zlib (via `GzipLineReader`) and walks it once.
All the genotype math that follows happens elsewhere; this file's whole job is to
turn a VCF plus a target panel into a clean per-site call.

The one design fact that governs everything below: this is a **faithful C++ port
of the Stage-0 oracle**, `experiments/nikki-stage0/oracle.py`. The oracle is the
Python reference that decided, sample by sample, what a "correct" call is for this
kind of consumer-genotyping VCF. This file reproduces it branch for branch —
including its quirks — so that the native reader and the oracle agree bit for bit.
Nearly every decision below exists because the oracle made it, and several exist
specifically to reproduce a subtle oracle behavior that a naive rewrite would get
*wrong*. Those are the "critique fixes" tagged `(fix a)`, `(fix b)`, `H1`..`M4`
in the source.

---

## 2. The three passes over the file

`VcfReader::genotype()` reads the VCF in three logical passes, though only one of
them actually streams the (large) body:

1. **Header pass — find the sample column.** Skip `##` meta lines; on the
   `#CHROM` header, locate which column carries our sample. If a `sample_id` was
   given, find it by name; if none was given, there must be exactly one sample
   (column 9) or the reader refuses with "pass `--sample` to select one". A file
   with no `#CHROM` header at all, or a header with fewer than 10 columns (no
   sample column), is a hard `std::runtime_error`.

2. **Body pass — build coverage + variant maps.** Stream every remaining record
   once, sorting each into one of two structures (section 4): the gVCF
   reference-block coverage bitmaps, or the per-position variant map.

3. **Resolution pass — call every target site in panel order.** Walk the target
   panel in its original order and, for each site, apply the interval-join
   precedence (section 3) to emit one report row and one packed genotype code.

Only pass 2 is O(file). The panel is small and fixed; passes 1 and 3 are cheap.

---

## 3. The interval-join precedence

This is the heart of the reader. For each target site the resolver tries a fixed
ladder of sources, highest precedence first, and stops at the first that applies:

1. **Palindrome drop.** If the site is strand-ambiguous (A/T or C/G), it is
   dropped outright — `drop_reason = "palindrome"`, source stays `"none"`, no
   attempt to read the VCF at all. (Palindromic sites never even make it into the
   per-chrom coverage index; see `target_sites.hpp`.)
2. **Explicit variant record** at this exact position (section 5). An explicit
   variant has **absolute precedence** over any overlapping reference block — a
   real SNP call always beats a coverage assertion.
3. **Hom-ref inside a *passing* reference block** (section 6, H1). The site is
   covered by a gVCF block that met the depth/PASS floor, so it is called
   homozygous-reference, reconciled to the panel's A1/A2.
4. **Covered only by a *failing* block** → `Missing` / `below_floor` (H4). The
   block covered the site but didn't clear the floor, so we know the site was
   *looked at* and found wanting — that's an informative missing, not absence.
5. **No coverage at all** → `Missing` / `no_coverage`.

The precedence order is the oracle's `resolve()`, and getting it exactly right —
variant over passing-block over failing-block over nothing — is what makes the
native reader's calls match.

---

## 4. What the body pass builds

Two structures, populated as the body streams:

- **Coverage bitmaps.** For each target chromosome, two `std::uint8_t` vectors
  (`pass_cov`, `fail_cov`) sized to that chrom's sorted target-position array.
  A gVCF reference-confidence block (a record with `ALT == "."`) marks every
  target position inside its `[POS, END]` interval — in `pass_cov` if the block
  cleared the floor (`FILTER==PASS` and depth ≥ `min_dp`), in `fail_cov`
  otherwise. The interval join uses `lower_bound`/`upper_bound` on the sorted
  position array, and the `END` boundary is treated as **inclusive** (`upper_bound`
  on `END`, oracle L1) — a block that ends exactly on a target position covers it.
  Depth is read as INFO `MinDP` when present, else FORMAT `DP`.
- **Variant map.** `chrom → (pos → VariantRec)`. An explicit variant record
  (`ALT != "."`) at a **target position** is stored, one per position. When two
  records land on the same target position, the tie-break (`better()`, section 5)
  keeps the stronger one. Variant records at non-target positions are ignored.

Chromosome names are normalized by stripping an optional `chr`/`Chr` prefix and
parsing the rest as an integer; anything that isn't a target autosome (X, Y, MT,
scaffolds, or a chrom not in the panel) is skipped (M1). Truncated records — fewer
columns than the sample column index — are skipped, not fatal.

---

## 5. The variant path (`resolve_variant`)

When a target site has an explicit variant record, two gates run before the
genotype is even parsed:

- **rsID cross-check (H3).** If the record's ID begins with `rs` and disagrees
  with the panel's rsID, the site is **dropped** as `rsid_mismap`. Crucially this
  is emitted *before* `source` is set to `"variant"`, so a mismapped site carries
  `source = "none"` — a faithful copy of the oracle's lazy field assignment
  (section 7).
- **The H4 floor.** `FILTER` must be `PASS`, FORMAT `DP` ≥ `min_dp` (default 8),
  FORMAT `GQ` ≥ `min_gq` (default 20). Failing PASS → `not_pass`; failing depth or
  quality → `below_floor`. Both are `Missing`, not dropped.

Past the floor, the diploid `GT` is parsed (either `/` or `|` separated). Then for
each of the two alleles:

- The allele index selects its base: `0` → REF, `n` → the *n*-th ALT. An index out
  of range, a spanning-deletion `*`, an indel, or any non-single-base allele makes
  the whole call `Missing` / `non_panel_allele` (M4, the multiallelic
  normalization). A half-call or missing GT (`.`) → `Missing` /
  `half_or_missing_gt`.
- Each single base is `reconcile()`d against the panel's A1/A2 (section 8). If a
  base matches neither allele on either strand → `Missing` / `non_panel_allele`.

The dosage is the count of **panel-A1** copies across the two alleles, and the
call label is named *from the A1 perspective*: two A1 copies is `Homref` (dosage
2), one is `Het` (dosage 1), zero is `Homalt` (dosage 0). If either allele needed a
strand complement to reconcile, `flip = 1` is recorded.

### The tie-break (`better`)

When two variant records collide on one position, `better()` prefers `FILTER==PASS`
first, then higher `GQ`. The one subtlety — tagged `(fix b)` — is that a `GQ` of
**0** is treated as `-1`, i.e. as *no* GQ. The oracle wrote `(a["gq"] or -1)`, where
Python's `0 or -1` evaluates to `-1`; so a record whose GQ is literally 0 must not
out-rank a record that carries no GQ field at all. The C++ reproduces that falsy-0
semantics exactly rather than the arithmetically-obvious `gq >= 0`.

---

## 6. The reference-block path (H1 reconciliation)

When no variant covers a site but a **passing** gVCF block does, the site is a
hom-ref call — but at what dosage? That depends on whether the reference base is
the panel's A1 or A2. The reader reconciles the panel's stored GRCh38 reference
base (`ref38`) against A1/A2:

- If REF reconciles to A1 → dosage **2** (two copies of A1).
- If REF reconciles to A2 → dosage **0**.

Either way the call label is `Homref` and `source = "refblock"`, with `flip` set if
a complement was needed.

### `no_refbase` vs `ref_change` (fix a)

There are two ways this can fail, and they are deliberately *not* the same:

- `ref38 == '.'` means the reference base was **unavailable** — the FASTA fetch
  returned nothing. That's `Missing` / `no_refbase`, and `source` stays `"none"`
  (the base was never in hand to attribute to the block).
- A real base — including a literal `N` in the assembly — that reconciles to
  *neither* A1 nor A2 is `Dropped` / `ref_change`, with `source = "refblock"`. The
  assembly's reference genuinely disagrees with the panel here.

A naive port would collapse both into one "no usable reference base" branch. The
oracle draws the line at *availability* (`no_refbase` fires only on the empty FASTA
fetch), so the port does too — an `N` flows into `reconcile()`, matches nothing, and
becomes a `ref_change` drop, not a `no_refbase` missing.

---

## 7. The lazy-field-assignment invariant

The single most error-prone thing to reproduce is the oracle's **lazy assignment**
of the `source` and `call` report fields (oracle.py:243–326). The rule:

- `source` is `"none"` until the resolver *commits* to a path. A site that is
  dropped before commitment — a palindrome, an rsID mismap, a `no_refbase` missing
  — carries `source = "none"` even though a variant record or a reference block was
  physically present. `source` describes the path *taken*, not the data *seen*.
- The `call` label is **never derived from the dosage**. A hom-ref-in-a-block site
  with REF==A2 is `call = "homref"`, `dosage = 0` — the label says homref and the
  dosage says zero A1 copies, and both are correct and independent. Any code that
  reconstructs the label from the number won't match the oracle at the REF==A2 case.

`VcfSiteCall` (declared in `vcf_reader.hpp`) is the report schema, and these two
fields are exactly where a plausible-looking rewrite drifts from the reference.
The header's own comment calls this out; this file honors it branch by branch.

---

## 8. Strand reconciliation (`reconcile`)

`reconcile(base, A1, A2)` answers "which panel allele is this base, and did I have
to flip strands to see it?" It tries the same strand first (base == A1, then base
== A2), then the complement (comp(base) == A1, then == A2), reporting `which`
(`+1` = A1, `0` = A2, `-1` = neither) and a `flip` flag. Complementing maps
A↔T and C↔G, and anything else complements to `N`. Bases are upper-cased first, so
soft-masked lowercase reference bases reconcile normally.

This is shared logic: both the variant path (per allele) and the reference-block
path (the single REF base) call it, so strand handling is single-homed.

---

## 9. The output tile

The resolution pass fills two parallel outputs per site: a `VcfSiteCall` report row
and a genotype code in `tile.snp_major`. The code is the dosage (`0`/`1`/`2`) for a
real call, or `kMissingCode` (3) for every `Missing` / `Dropped` outcome.

After the loop, each SNP's code byte is packed into a canonical source byte with
`pack_code_into_byte(0, 0, byte)` — the 2-bit code goes into slot 0 (MSB-first) of
a 1-byte-per-record SNP-major tile. The tile is then stamped as a single-sample,
single-population source (`n_individuals = 1`, `pop_offsets = {0, 1}`,
`pop_labels = {resolved_sample}`, one selected row). This is a *SNP-major source*
tile — the shared device transpose is what repacks it into the canonical
individual-major `GenotypeTile` the compute layer consumes. This file emits the raw
source; it does not transpose.

---

## 10. Contracts and invariants

- **One sample per run.** The reader genotypes exactly one sample. An empty
  `sample_id` is only valid when the file carries a single sample; multi-sample
  files require an explicit selection.
- **The panel is pre-normalized.** `TargetSites` arrives already rsID-joined and
  lifted to GRCh38, with palindromes still present but flagged, and `ref38`
  pre-fetched and upper-cased. The reader does no liftover, no dbSNP position
  cross-check, no dedup — those stay upstream (shared with the oracle). Its
  per-chrom coverage index and slot map are built over the **non-palindromic**
  sites only, so an interval join or slot lookup on a palindromic position simply
  doesn't exist (and can't, since palindromes are dropped before any lookup).
- **Positions can collide.** Liftover can map two rsIDs to one GRCh38 position; the
  slot map is last-wins, matching the oracle's `slot` construction, so the coverage
  bitmap index and the resolved site agree at colliding positions.
- **Every panel site produces exactly one report row and one tile code**, in panel
  order. The report and the tile are always the same length as `targets_.sites`.
- **Autosomes only.** Targets are chroms 1–22; any other CHROM in the VCF is
  skipped during the body pass.

---

## 11. Edge cases and failure modes

- **Missing `END` on a ref block** defaults the block to a single position
  (`END = POS`), so a block with no `END` covers just its own position.
- **Truncated records** (fewer fields than the sample column) are skipped
  *before* the `records_seen` counter is bumped, so a record too short to carry the
  sample column simply doesn't participate and isn't counted.
- **`INFO == "."`** short-circuits to "field absent" for `END`/`MinDP` lookups; a
  bare `DP=` never matches inside `MinDP=` and a FORMAT `DP` never hits `AD`,
  because the `vcf_record.hpp` helpers match keys exactly on the right delimiter
  (that field-boundary robustness bcftools gave the oracle for free).
- **Header failures are fatal**: no `#CHROM` line, a `#CHROM` with no sample
  column, an unknown `--sample` name, or an ambiguous unnamed multi-sample file all
  throw `std::runtime_error` with a message naming the file.
- **Everything else is soft.** A record on a non-target chrom, a non-target
  position, a non-integer position, a below-floor block, an out-of-range allele
  index — none of these are errors. They're the normal texture of a VCF, and each
  routes to the appropriate skip or `Missing` outcome.

The `VcfCounts` struct tallies every outcome (called-variant vs called-refblock,
the homref/het/homalt split, and each missing/dropped sub-reason) for the run
summary — every branch above bumps its own counter, so the summary reconciles
exactly with the per-site report.
