# `cmd_ingest.cpp` reference

## 1. Purpose

`src/app/cmd_ingest.cpp` is the handler behind `steppe ingest` — the command that
takes one sample's `.vcf.gz`, genotypes it against a table of GRCh38 target sites,
and turns the result into the artifacts the rest of the "place nikki among the
ancients" pipeline consumes. It is the thin orchestration layer: it validates the
flags, decides *what* work was actually asked for, calls the real workers in
`src/io/` and `src/core/`, and prints a one-line summary. The heavy lifting —
parsing the VCF, resolving each site, packing the tile, appending the panel — all
lives in the collaborators it drives.

There are four things a run can produce, and they can be combined:

- **the per-site report TSV** (`--report`) — the primary Stage-1 artifact, one row
  per target site in the Stage-0 oracle's exact schema;
- **the built target table** (`--emit-targets`, native mode only) — the 7-column
  GRCh38 table the native builder produced, dumped for reuse or inspection;
- **the raw canonical 2-bit tile bytes** (`--emit-tile`) — the bit-exact-check
  artifact, and the *only* output that touches a GPU;
- **the merged panel** (`--emit-merged`) — nikki appended as a size-1 population
  into an existing EIGENSTRAT-family panel (Stage 3).

Almost the whole file is pure host code. The report path never needs a device — the
Stage-1 block-correctness gate is a CPU-only job — and the merge path is host I/O.
The single seam that reaches the GPU is the `--emit-tile` transpose (section 8).

---

## 2. The two ways to name the target sites

Every run has to answer one question first: *what are the target sites?* There are
two mutually exclusive ways to supply them, and the file rejects any mix.

- **Legacy / Stage-1 (`--targets <table>`).** A pre-built GRCh38 target-site table
  is read straight off disk by `io::read_target_sites`. This is the back-compat
  path: the liftover and dbSNP cross-check already happened offline, upstream.
- **Native / Stage-2 (`--panel` + `--fasta` + `--lift`).** The table is built
  in-process by `io::build_target_sites` from the AADR EIGENSTRAT `.snp` panel, the
  GRCh38 FASTA (opened through `io::FaidxReader`, which expects a sibling `.fai`),
  and the orchestrated rsID→pos38 lift map. Native mode is the only mode that may
  also `--emit-targets` the built table, and it prints a build-counts line to stderr
  (`panel_total`, `autosomal`, `no_lift`, `dropped_dup`, `emitted`, …) so the reader
  can see exactly how the panel narrowed down.

The selector is deliberately simple: presence of *any* of `--panel/--fasta/--lift`
means native; presence of `--targets` means legacy. The validation block up top
turns every wrong combination into a specific, actionable message and a
`kExitInvalidConfig`:

- native **and** legacy flags together → "choose ONE target source";
- neither → "no target source";
- native but missing one of the three native inputs → "needs all of
  --panel, --fasta, --lift";
- `--emit-targets` without native mode → it's meaningful only for the native build.

---

## 3. Deciding what work to actually do

After the target source is settled, the file works out which of the four outputs
were requested, because that decides whether it needs to touch a VCF or a GPU at
all. The logic is a small ladder of booleans:

- **`want_genotype`** is true when `--report` or `--emit-tile` was given. Either one
  means "run the genotype pass", and that pass needs a VCF — so `want_genotype`
  without `--vcf` is a config error.
- **`want_merge`** is true when `--emit-merged` was given (section 7).
- **`need_genotype = want_genotype || want_merge`.** The merge needs nikki's calls
  too, so requesting *only* `--emit-merged` still forces the genotype pass on even
  though neither `--report` nor `--emit-tile` was named.

The "nothing to do" guard is the counterpart: if nothing wants genotyping and
neither `--emit-targets` nor `--emit-merged` was asked for, the command refuses with
a "nothing to do" message rather than silently succeeding. The one genuinely
empty-VCF case that *is* allowed is a bare **native `--emit-targets`**: building the
table needs no VCF, so when `!need_genotype` the function writes the table (already
done in section 2's build step) and returns `kExitOk` right there — that is the
gate-1 table-reproduction path.

`--min-dp` and `--min-gq` are validated to be non-negative here; their frozen
defaults (8 and 20) live in `IngestArgs` and are passed straight through to the
reader's `Options`.

---

## 4. The genotype pass

When `need_genotype` holds, the file constructs a `VcfReader` over the VCF, the
resolved `TargetSites`, the (optional) sample id, and the DP/GQ options, then calls
`reader.genotype()`. Everything hard about VCF genotyping — the gVCF reference-block
interval join, the hom-ref/panel reconciliation, the DP/GQ/PASS floors, the
multiallelic normalization, the rsID-mismap and palindrome drops, the strand-flip
logic — happens inside the reader and is documented at its own header. This file
just receives the `VcfIngestResult`: the canonical individual-major tile, the
per-site `calls` (panel order), the aggregate `counts`, and the resolved
`sample_id`.

Any exception the reader throws (open failure, malformed VCF, missing sample) is
caught, printed as `steppe ingest: <what>`, and mapped to `kExitIoError`. The same
catch-and-report shape guards the target build in section 2. The tile transpose in
section 8 is the one place that maps caught exceptions through
`exit_code_for_caught` instead, because a device fault there deserves a device/
runtime exit code rather than a flat I/O error.

---

## 5. The per-site report (`write_report`)

`--report` writes the calls out verbatim in the Stage-0 oracle schema (spec §9): a
fixed header line —

```
rsID  chrom  pos37  pos38  A1  A2  call  dosage  source  flip  drop_reason
```

— then one tab-separated row per `VcfSiteCall`. Two small faithfulness details
matter here and are easy to get wrong:

- **`call` is a label, never re-derived from `dosage`.** `call_str` maps the
  `VcfCall` enum (`homref`/`het`/`homalt`/`missing`/`dropped`) straight to its
  string. The reader assigns the label lazily as it resolves each site, so the
  report must print the label the reader committed to — a hom-ref block is
  `call=homref, dosage=0`, and that pairing comes from the resolver, not from
  reading the dosage back.
- **`dosage < 0` prints as `NA`.** A dosage of -1 is the "no call" sentinel; it is
  written as the literal `NA`, not as a negative number.

The writer checks the stream both at open and after writing, so a truncated write
surfaces as a failure rather than a silently short file.

---

## 6. Building the panel-aligned code vector (`build_panel_codes`)

This is the trickiest host routine in the file, and it exists only for the merge
path. The merge writer wants **one 2-bit code per SNP record in the panel `.geno`,
in panel `.snp` row order** — including the many panel rows that were never target
sites at all. `build_panel_codes` produces exactly that vector by streaming the
panel `.snp` and emitting one code per non-blank line.

The default for every row is `kMissingCode` (3). A real call is written onto a row
only when **all** of these hold:

1. the row's rsID is one of the in-scope target calls (looked up in a
   `rsID → call` map built once over `calls`);
2. the call is *covered* — its class is `homref`, `het`, or `homalt` (not missing,
   not dropped);
3. **the row's own `pos37` matches the call's `pos37`** — the polarity guard;
4. the dosage is in `{0, 1, 2}`.

Guard 3 is the load-bearing one, and it is why the merge path insists on a real
pos37 (section 7). A target rsID can appear on more than one panel row after a
messy lift (a duplicate or a mis-mapped position); keying the call onto the row by
**both rsID and pos37** ensures the call lands only on the panel row it was actually
built from, not on a homonymous collision elsewhere. Without it a dosage could be
stamped onto the wrong SNP.

Two robustness details:

- **A duplicate-rsID audit.** The routine counts how many *panel* rows carry a
  target rsID for the second time (`n_dup_panel_rsid`); that count is surfaced in
  the merge summary line as `panel_dup_rsid_hits` so a lift that scattered an rsID
  across panel rows is visible, not silent.
- **The interior-blank-line guard.** A blank line in the middle of the `.snp`
  desyncs the SNP axis — every subsequent code would land one row off — so the
  routine treats an interior blank as a hard error. A *trailing* blank at EOF is
  tolerated (it peeks for EOF and stops cleanly), which is the normal
  file-ends-in-newline case.

The `pos37` is parsed with `std::from_chars` and defaults to -1 when the column is
empty; an empty-string rsID never matches the map (it short-circuits the lookup).

---

## 7. The Stage-3 merge (`--emit-merged`)

`--emit-merged` appends nikki into an existing panel as a new size-1 population.
Its flag contract:

- it **requires** `--merge-into <panel prefix>` (the source `.geno/.snp/.ind` to
  append into) — without it, config error;
- it **requires** `--vcf` (you can't merge a sample you never genotyped);
- conversely `--merge-into` without `--emit-merged` is a config error — the prefix
  is meaningless with nothing to write.

There's a subtle correctness note baked into the validation comment: because
`build_panel_codes` keys on pos37 (section 6), the target source **must carry a real
pos37**. A 6-column `--targets` table (where pos37 defaults to 0) would fail the
pos37 guard on *every* row and produce an all-missing merge. The visible signature
of that mistake is a suspiciously low `called` count in the summary line — worth
knowing when a merge comes back empty.

The merge itself: `build_panel_codes` produces the aligned vector, then
`io::write_merged_panel` appends nikki (labelled with `result.sample_id` for both
the `.ind` id and its population) and returns a `MergeCounts`. The writer
auto-detects the on-disk format (TGENO / GENO-PA / EIGENSTRAT) and preserves it; the
file just reports which one it took, along with the SNP count, the individual count
before→after, nikki's called/missing split, and the duplicate-rsID hit count.

---

## 8. The canonical tile (`--emit-tile`) — the one GPU path

This is the only output that needs a device, and only because the raw canonical
2-bit tile is produced by the **shared device transpose**. The reader hands back an
individual-major `SnpMajorTile`; the on-disk canonical layout is the transpose of
that, and steppe has exactly one transpose implementation
(`core::transpose_snp_major`), which runs on a `ComputeBackend`. Reusing it here —
rather than writing a host transpose — keeps the packing bit-identical to every
other tile in the system.

So the tile branch:

1. parses `--device` into a `DeviceConfig` (section 9);
2. builds `device::Resources` and requires a first GPU (`require_first_gpu`); a box
   with no usable GPU here is a `kExitRuntimeError`;
3. runs the transpose on that GPU's backend;
4. writes `canon.packed` verbatim to the `--emit-tile` file as binary bytes,
   checking the stream at open and after the write.

Exceptions in this block flow through `exit_code_for_caught` so a real device fault
maps to the right runtime/device exit code.

---

## 9. `--device` parsing (`parse_device`)

A tiny local helper turns the `--device` string into a `DeviceConfig`. It strips all
whitespace, treats empty or `"auto"` as "use the default" (leaves the config
untouched, returns success), and otherwise splits the remainder on commas into a
list of integer ordinals via `std::stoi`. A token that isn't an integer is a
specific error (`--device ordinal '<tok>' is not an integer`) → `kExitInvalidConfig`.
Empty tokens (a trailing or doubled comma) are skipped rather than treated as zero.
This is a deliberately self-contained parse — `ingest` carries its own device flag
rather than going through the shared `RunConfig`, so this handler stays free of the
fit-command configuration machinery.

---

## 10. The summary line and exit codes

On a genotyping run that reaches the end, the file prints one stderr summary with
the sample id, the site count, the called split (variant vs ref-block, and within
that homref/het/homalt), the missing and dropped totals, and the raw records seen —
straight from `result.counts`. That line is the at-a-glance health check for a run.

Exit codes, all drawn from `core/config/exit_code.hpp`:

| Condition | Exit |
|---|---|
| Normal completion (including the native table-only build) | `kExitOk` |
| Any bad flag combination, a non-integer `--device`, or missing required inputs | `kExitInvalidConfig` |
| A target-build / VCF-read / report-write / merge failure (a caught `std::exception` on a host path) | `kExitIoError` |
| No usable first GPU for `--emit-tile` | `kExitRuntimeError` |
| A device/transpose exception on the `--emit-tile` path | whatever `exit_code_for_caught` maps it to |

---

## 11. Design notes and invariants

- **Self-contained args, no shared `RunConfig`.** `IngestArgs` is a flat struct of
  the flags this one command needs. `ingest` never merges into the fit commands'
  `RunConfig`, which is why it parses its own `--device` and keeps its validation
  local — the ingestion subsystem stays decoupled from the compute-config chain.
- **Host by default, GPU only when unavoidable.** Three of the four outputs are pure
  host work. The device is built only inside the `--emit-tile` branch, so a report or
  merge run never initializes CUDA — the Stage-1 gate can run anywhere.
- **The pos37 polarity guard is the correctness linchpin of the merge.** It is what
  keeps a call from being stamped onto the wrong panel row when a lift duplicated or
  mis-mapped an rsID. Any change to the target-table columns has to keep a real
  pos37 flowing to `build_panel_codes` or the merge silently goes all-missing.
- **Labels come from the resolver, not from dosage.** Both the report (`call`) and
  the merge (covered-vs-missing) trust the `VcfCall` the reader assigned; nothing in
  this file re-infers a call class from a dosage value.
- **The output vector length equals the panel `.geno` record count**, by
  construction — one code per non-blank `.snp` line — which is exactly the invariant
  `write_merged_panel` asserts (it throws if the code vector isn't panel-aligned).
