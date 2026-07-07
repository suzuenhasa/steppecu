# `cmd_ingest.hpp` reference

## 1. Purpose

`src/app/cmd_ingest.hpp` declares the front door for `steppe ingest` — steppe's
own gVCF-block-aware hardcall VCF reader, the sixth reader arm alongside the
five genotype-file readers. It exists to take one sample out of a `.vcf.gz`,
genotype it at a fixed table of GRCh38 target sites, and turn that into
artifacts the rest of steppe can use: a per-site report TSV, an optional raw
canonical 2-bit tile, and — the big one — an existing genotype panel with the
new sample appended as its own one-person population.

The header is deliberately tiny. It declares exactly two things: an `IngestArgs`
struct that carries every knob the command accepts, and the single entry point
`int run_ingest(const IngestArgs&)`. All the actual work — the validation, the
target-site build, the VCF pass, the report writer, the merge, the tile
transpose — lives in `cmd_ingest.cpp`. This document describes the contract the
header promises; read it alongside that `.cpp`.

Two design choices are baked into the header and worth stating up front:

- **Self-contained args, no shared config.** Unlike the fit and statistic
  subcommands (which flow through the `RunConfig` precedence merge in
  `cli_parse.cpp`), `ingest` takes a plain value struct straight from the
  command line. It does not touch `RunConfig`, so it can't pollute — or be
  polluted by — the shared configuration surface. Every field it needs is right
  here in `IngestArgs`.
- **GPU only through one narrow seam.** The whole report path is pure host code;
  it never opens a device. The command reaches the GPU in exactly one place —
  the canonical-tile transpose behind `--emit-tile` — and nowhere else.

---

## 2. The `IngestArgs` fields

Every field is a `std::string` (a path or label) except the two frozen integer
thresholds. An empty string means "not given."

| Field | Meaning |
|---|---|
| `vcf` | The `.vcf.gz` / `.vcf` to genotype. Required whenever any genotyping is asked for. |
| `targets` | A pre-built GRCh38 target-site table (the Stage-1 back-compat path). |
| `panel` | An AADR EIGENSTRAT `.snp` on GRCh37 — the source SNP set for the native table build. |
| `fasta` | A GRCh38 `.fa` (expects a sibling `.fai`) supplying the reference allele — the native `ref38`. |
| `lift` | The orchestrated rsID→pos38 lift map — the one input the native build defers to an external step. |
| `emit_targets` | Optional (native only): dump the built 7-column target table to this path. |
| `sample` | Optional: which sample id to genotype; defaults to the file's sole sample. |
| `report` | Optional: the per-site report TSV — the primary Stage-1 artifact. |
| `emit_tile` | Optional: the raw canonical 2-bit tile bytes. This is the one path that needs a device. |
| `merge_into` | Optional: the source panel **prefix** (`.geno`/`.snp`/`.ind`) to append the sample into. |
| `emit_merged` | Optional: the output merged-panel prefix. Requires `--merge-into`. |
| `device` | Optional: CUDA device ordinal(s), e.g. `"0"` or `"0,1"`; empty or `auto` means default. |
| `min_dp` | Ref-block MinDP / variant DP floor. Frozen default 8. |
| `min_gq` | Variant GQ floor. Frozen default 20. |

`min_dp` and `min_gq` are the only tunable quality thresholds; their defaults are
frozen for reproducibility and feed straight into `VcfReader::Options`.

---

## 3. Two ways to name the target sites (mutually exclusive)

The target-site set is what the VCF is genotyped against, and there are exactly
two ways to supply it — you must pick one:

- **(A) Stage-1, back-compat:** pass `--targets <table>`, a table someone built
  earlier. The command just reads it.
- **(B) Stage-2, native:** pass all three of `--panel`, `--fasta`, and `--lift`,
  and the command builds the table in-process — walking the panel `.snp`,
  pulling the GRCh38 reference allele from the FASTA, and applying the lift map
  to place each rsID at its GRCh38 position. The native build prints a full
  provenance line (panel total, autosomal kept, non-rsID, palindromic, duplicate
  rsIDs, lift hits/misses/dropped-dups, and final emitted count) to stderr, and
  can dump the assembled 7-column table with `--emit-targets`.

The command is "native" the moment **any** of `panel`/`fasta`/`lift` is set, and
"legacy" the moment `targets` is set. The validation rules that follow this are
strict: giving both a native input and `--targets` is an error, giving neither is
an error, and choosing native but leaving out any one of the three required
pieces is an error. `--emit-targets` is meaningful only on the native path and is
rejected on the legacy path.

---

## 4. What the command actually does (the four jobs)

`run_ingest` decides what to do from which outputs you asked for, not from a mode
flag. There are four jobs, and any combination that makes sense runs together:

1. **Build/read the target table.** Always happens. Native builds it; legacy
   reads it.
2. **Report** (`--report`): run the VCF pass and write the per-site TSV.
3. **Merge** (`--emit-merged` + `--merge-into`): run the VCF pass and append the
   sample as a one-person population into a copy of the panel.
4. **Tile** (`--emit-tile`): run the VCF pass, transpose the result to the
   canonical 2-bit layout on the GPU, and write the raw bytes.

Genotyping — the actual VCF pass — is needed by jobs 2, 3, and 4. A **bare native
`--emit-targets`** with none of those asks the command to build the table and
stop, without ever opening the VCF. That is the gate-1 "table reproduction" path:
no sample, no device, just the table.

Because the report path never touches a device, the Stage-1 block-correctness
gate — the check that steppe's gVCF ref-block handling matches the oracle — can
run entirely on the host.

---

## 5. The merge: appending one person to a panel

The most involved job is the Stage-3 merge, and its correctness hinges on one
rule. The panel being merged into has a fixed SNP axis (its `.geno` record
order, mirrored by its `.snp` rows). The command must produce, for the new
sample, exactly one 2-bit code per panel SNP row, in that same order — most of
them "missing," a few of them real calls.

The load-bearing invariant is the **double-key guard**: a real call is written
onto a panel row only when **both** the rsID **and** the GRCh37 position
(`pos37`) match that row. Matching on rsID alone is not enough. This is the guard
against duplicate or mis-mapped rsIDs polluting the wrong site — if an rsID
appears twice or a lift put it in the wrong place, the position check catches it
and that row stays missing rather than taking a call it doesn't own. On top of
that the call must actually be covered (homref/het/homalt) with a dosage in
`{0,1,2}`; anything else (missing, dropped, out-of-range) becomes the missing
code.

This is why a merge needs a target source that carries a **real** `pos37`. A
6-column table with `pos37 == 0` fails the position guard on every row, so every
site comes back missing — and the visible symptom of that mistake is a called
count near zero in the merge summary line.

The merge summary reports the detected panel format (TGENO / GENO/PA /
EIGENSTRAT), the SNP count, the individual count before and after (`N -> N+1`),
how many of the new sample's sites were called versus missing, and how many
duplicate-rsID hits the panel-uniqueness audit saw.

---

## 6. The tile path and the single GPU seam

`--emit-tile` is the only output that opens a device. It exists for the bit-exact
check: the VCF pass produces a SNP-major tile, and the canonical on-disk layout
is the transposed sample-major form. That transpose is the *shared* device
transpose the rest of steppe uses, so routing through it here proves the ingest
tile is byte-for-byte the same object the standard readers would produce.

The `--device` string is parsed into a `DeviceConfig` (empty/`auto` → default,
otherwise a comma-separated ordinal list), the first GPU is required, the
transpose runs, and the raw packed bytes are written straight to the tile file.
Everything about opening the device is confined to this branch; no other part of
`ingest` — including the whole report and merge machinery — ever needs a GPU to
be present.

---

## 7. Contracts and invariants

- **Exactly one target source.** Native (`panel`+`fasta`+`lift`) or legacy
  (`targets`), never both, never neither, and native means all three.
- **`--vcf` is mandatory for genotyping.** Any of `--report`, `--emit-tile`, or
  `--emit-merged` requires a VCF; asking for one without `--vcf` is rejected.
- **The command must have something to do.** With no genotype output *and* no
  `--emit-targets`/`--emit-merged`, `run_ingest` refuses rather than silently
  succeeding on a no-op.
- **`--emit-merged` ⇒ `--merge-into`** (and vice versa: `--merge-into` alone,
  with no `--emit-merged`, is rejected). A merge also forces the genotype pass on
  even if neither `--report` nor `--emit-tile` was asked for, because the merge
  consumes the calls.
- **`--emit-targets` is native-only.** It has no meaning on the legacy `--targets`
  path.
- **`--min-dp` / `--min-gq` must be ≥ 0.**
- **One code per panel row, in panel order.** The merge output vector length
  equals the `.geno` record count, and its ordering is the panel's, not the
  target table's.
- **Frozen defaults.** `min_dp = 8`, `min_gq = 20`, `--device` auto.

---

## 8. Edge cases and error handling

- **Interior blank line in the merge panel `.snp`.** A single trailing blank line
  at end-of-file is tolerated, but a blank line *between* records would desync the
  SNP axis, so it is a hard error rather than a skipped row.
- **Duplicate rsIDs in the panel.** These don't fail the run; they're counted and
  reported. The double-key position guard (section 5) is what keeps a duplicate
  from stealing a call it doesn't own, so the audit is informational.
- **Uncovered / out-of-range calls.** A site that resolved to missing or dropped,
  or whose dosage falls outside `{0,1,2}`, contributes the missing code to the
  merge vector — it is never forced into a real code.
- **All-missing merge.** The tell-tale of a `pos37`-less (6-column) target table:
  the guard rejects every row, the called count collapses to near zero. This is
  surfaced in the summary line rather than thrown, so it's diagnosable from the
  output.
- **Exit codes.** Validation failures return the invalid-configuration exit;
  file/read/write and target-build failures return the I/O-error exit; a device
  or transpose failure is mapped through the shared caught-exception classifier.
  A clean run — including the table-only native build — returns success.

---

## 9. Where the real work lives

The header is a declaration; the behavior is in `cmd_ingest.cpp` and its
collaborators: `io::build_target_sites` / `io::read_target_sites` /
`io::write_target_table` for the table, `io::VcfReader` for the genotype pass,
`io::write_merged_panel` for the Stage-3 merge, and `core::transpose_snp_major`
for the one GPU seam. If a detail here ever disagrees with those files, the code
is the authority.
