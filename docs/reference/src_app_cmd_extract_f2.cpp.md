# `cmd_extract_f2.cpp` reference

## 1. Purpose

`src/app/cmd_extract_f2.cpp` implements the `steppe extract-f2` command: the
precompute that turns a genotype triple (`.geno` + `.snp` + `.ind`, or the
equivalent PLINK/EIGENSTRAT files) into an on-disk `f2_blocks` directory that the
later fit commands (for example `steppe qpadm --f2-dir`) read. This is the one place
in steppe where file reading is wired directly into GPU compute.

The file is plain C++20 with no CUDA header of its own. It reaches the GPU only
through CUDA-free seams — the resource builder, the backend's decode entry, and the
library's extract entry — so the command can be compiled without pulling in the GPU
toolchain. It owns all of the program's standard output and standard error: the
library it calls never prints; it throws, and this command maps those throws to a
process exit code and a message.

The command itself does only the cheap, up-front work (validating inputs, sizing the
problem, and — at the end — writing the output directory). The expensive
genotype-to-`f2` chain lives in a separate library entry, `steppe::run_extract_f2`,
which this command calls once.

---

## 2. The extract chain: what the command owns versus delegates

The full path from genotype files to the written directory is a numbered chain. Some
steps run here in the command; the heavy middle runs inside the library entry.

| Step | What happens | Where it runs |
|---|---|---|
| 0 | Validate required inputs (the triple, `--out`, a population selection) | command |
| 1 | Open the `.geno`, read the `.ind` with the population selection | command (sizing/validation) |
| 2 | Read the `.ind` into groups sorted by label — this is the P-axis order and the `pops.txt` order | command |
| 3 | Read the `.snp` table (chromosome, genetic position, ref/alt alleles) | command (sizing/validation) |
| 4 | Read the canonical genotype tile (packed, population-contiguous) | library |
| 5 | The GPU decodes the tile to allele-frequency, variance, and count arrays | library |
| 6 | Apply the SNP filters (when non-default), subsetting the arrays and the SNP table | library |
| 7 | Assign SNPs to jackknife blocks along the genome | library |
| 8 | Compute the `f2` blocks on the GPU and copy the result to a host tensor | library |
| 9 | Write the `f2_blocks` directory: `f2.bin`, `pops.txt`, `meta.json` | command |

The reads in steps 1–3 that happen inside this command are for sizing and validation
only. The library re-reads the same triple internally when it does the real
decode/compute; those file reads are tiny next to the GPU work, so reading twice
costs almost nothing and keeps the two paths cleanly separated. The dry-run path
(section 6) uses only the command's own up-front reads and never enters the library.

---

## 3. Local helper functions

Four small helpers live in an anonymous namespace at the top of the file.

- **`tier_label` (two overloads)** — turns a memory-tier enum into a human word:
  `resident`, `host`, or `disk`. `resident` is the all-in-GPU-memory path; `host`
  and `disk` are the input-streaming tiers that keep GPU peak memory independent of
  the SNP count. One overload takes the tier enum the resolver returns; the other
  takes the mirror enum the library entry hands back after the real run, so both the
  planned tier and the actually-used tier can be echoed with the same words.
- **`pop_selection_str`** — a human echo of *how* populations were selected:
  `explicit:a,b,c` for a named list, `auto-top:K` for the top-K-by-sample-count
  mode, or `min-n:N` for the minimum-sample-count mode. The resolved labels
  themselves end up in `pops.txt`; this records the request that produced them and
  is stored in `meta.json`.
- **`validate_explicit_pops`** — checks that every name in an explicit `--pops` list
  actually appears in the resolved partition. This exists because the `.ind` reader
  *silently drops* an unknown label and only complains if the selection ends up
  completely empty. So a misspelled name among several good ones would otherwise pass
  unnoticed. This helper scans the resolved groups for each requested name and, on
  the first one missing, reports it and returns false so the caller can fail with a
  clear message naming the offending label.

---

## 4. Required inputs and validation (step 0)

Before any reading, the command rejects incomplete requests up front, each with a
specific message and the invalid-config exit code:

- **The genotype triple is required.** All of `--geno`, `--snp`, and `--ind` must be
  present (supplied directly or via a `--prefix`).
- **`--out DIR` is required** unless this is a dry run (a dry run produces no output
  directory, so it does not need one).
- **A population selection is required.** Exactly one of `--pops a,b,…`,
  `--auto-top-k K`, or `--min-n N` must be given. The "no selection" state is
  detected as the auto-top-K mode with K equal to zero, which is the default when the
  user named none of the three flags.

---

## 5. The up-front read: sizing and validation only

Steps 1–3 open the `.geno`, read the `.ind` with the selection, and read the `.snp`
table. The format of the `.geno` header pins which `.snp`/`.bim` and `.ind`/`.fam`
parsers to use, so the same command works across the supported formats.

Two dimensions are derived here without ever reading a genotype tile:

- **P** (the number of populations) is just the number of resolved groups.
- **M** (the number of SNPs) is the smaller of the header's SNP count and the number
  of rows in the `.snp` table — the common prefix the two axes share.

Deriving P and M from the partition and the header — rather than reading a tile —
is deliberate and keeps this up-front step format-agnostic. An individual-major tile
read targets one specific on-disk layout; doing that here would make a differently
laid-out prefix throw before the real run ever starts. The actual canonical-tile
read (which may involve an on-device transpose for the individual-major format)
happens later, inside the library entry.

After deriving the sizes the command guards against degenerate inputs: an empty
population selection (P ≤ 0) or no SNPs (M ≤ 0) is an invalid-config error, and a
`.snp` table with fewer rows than the tile spans is an I/O error — the `.snp` and
`.geno` SNP axes must agree. Any exception during these reads is caught and reported
as an input error.

The command also builds `pop_labels`, the P labels in P-axis index order (the same
order as `pops.txt`). This is the name-to-index map the compute engine itself does
not carry; it is written alongside the numeric result so downstream tools can map
population names back to matrix rows and columns.

Per-sample ploidy detection (the pseudo-haploid-versus-diploid decision) is *not*
done here anymore. It happens inside the library entry, which returns the per-kind
sample counts on its result so the command can echo them after the run.

---

## 6. The dry run (`--dry-run`)

A dry run reports the resolved sizes, the chosen memory tier, and the precision, then
returns success without doing any GPU compute. It prints the input paths, the SNP and
record counts, the population selection and resulting P, the precision (with its
mantissa-bit setting), the block size in both Morgans and centimorgans with the
resulting block count, and the active filters.

Two properties make the dry-run plan trustworthy:

- **The block count is computed with the real block-assignment rule** over the full
  SNP axis — cheap, CUDA-free arithmetic — so it is the same rule the real run uses.
- **The tier estimate routes through the exact same resolver the real run calls**,
  with the same precedence: an explicit `--tier` (from the config's force-tier field)
  wins, then the `STEPPE_FORCE_TIER` environment variable, then the automatic policy.
  So a `--tier X` verdict here is *exact*, and an automatic verdict uses the identical
  policy the run will use.

The automatic verdict is intentionally **conservative toward the resident tier**. It
feeds the resolver the full SNP count and the full-axis block count, both of which are
upper bounds: filtering only ever drops SNPs, so the real run's kept-SNP count and
block count are less than or equal to these. That means if the dry run says it will
stream, it will stream for real; and if it says resident at the full size, the run
still fits at the smaller kept size.

The free-GPU-memory probe used for the estimate goes through the resource builder's
reported capabilities rather than any direct CUDA call. On a machine with no visible
GPU, the tier estimate is simply skipped — the dry run remains a useful planning aid
even where no compute could run. The report also prints the free VRAM (free and
total) and free host RAM, and the size of one `f2` result slab per block (P × P ×
2 × 8 bytes, for the `f2` value and its paired variance in double precision) together
with the total result size at the estimated block count.

---

## 7. Source-provenance hashing (`--hash`) on a background thread

An optional SHA-256 of the whole source `.geno` file records provenance in
`meta.json`. It is **off by default** for a good reason: on a large panel (measured
at about 37 seconds of a roughly 41-second run on a 6.7 GB `1240K` `.geno`) the
whole-file read-and-hash dominated the command's wall time, yet it produces only a
provenance value. (It did once earn its keep by catching a corrupted reference file
via a hash mismatch, which is why the option exists at all.) The small `.snp` and
`.ind` hashes are cheap and are left to the directory writer.

When `--hash` is given, the hash is computed on a **background thread started before
the GPU work and joined just before `meta.json` is written**. Because the hash depends
only on the `.geno` path — nothing in the decode/filter/compute pipeline — its cost
overlaps the GPU wall time instead of adding to it.

The threading is written to be exception-safe in three layers:

- **The worker catches everything.** An exception escaping the top-level callable of a
  `std::thread` calls `std::terminate` — a hard crash with no diagnostic. The hash
  routine uses an exception-disabled stream and returns an empty string if the file
  cannot be opened, so the only realistic throw is an out-of-memory failure on its
  read buffer; the worker captures any such exception into a stored exception pointer
  and lets the main thread deal with it.
- **A scope-guard joins the thread no matter how the function exits.** A small
  RAII object joins the worker in its destructor, so an early return or an exception
  before the explicit join can never destroy a still-running thread (which would also
  terminate the process). Because the worker already swallows all exceptions, neither
  the guard's join nor the explicit join can ever observe an escaping exception.
- **A captured worker failure degrades rather than aborts.** If the worker stored an
  exception, the command prints a warning and writes `meta.json` with
  `source_hash_computed:false` — the provenance hash is non-essential, so it is not
  worth failing the whole extract over. Crucially, this path also clears the
  "hash source files" flag, which stops the directory writer from *re-hashing* the
  big file on the main thread — that would redo the very work that was backgrounded
  and could re-trigger the same out-of-memory failure inside the writer.

On the success path the computed hash is pre-filled into the metadata so the writer
skips re-hashing the large file. When `--hash` is not given, every hash field stays
empty and `meta.json` records `source_hash_computed:false`, so the absence is
recognizably deliberate rather than an accident.

---

## 8. Running the GPU chain and mapping exit codes

The command builds the GPU resources and, if no CUDA device is visible, fails with a
runtime error and a clear message — steppe is a GPU product and requires a
CUDA-capable device. Otherwise it calls `steppe::run_extract_f2` once with the
resolved paths, selection, filters, precision, block size, and ploidy mode. That one
call performs steps 4–8 of the chain.

The command keeps ownership of output and translates the library's exceptions into
exit codes:

- **A config-level fault** (an unknown population, an empty selection, or every SNP
  filtered out) surfaces as `std::invalid_argument` and maps to the invalid-config
  exit code.
- **Any other exception** maps through a shared helper that returns the right code —
  in particular a real device out-of-memory is reported as an out-of-memory/device
  failure rather than a generic error.

The ploidy mode passed to the library is translated from the config's three-way
setting into the library's own enum: forced pseudo-haploid, forced diploid, or
automatic detection.

---

## 9. The engaged precision handshake

The command passes the *requested* precision into the library, but records in
`meta.json` the precision the library actually *honored*. It does this by taking the
precision **kind** back from the library's result (its single source of truth for the
kind it used, including any internal downgrade) while keeping the **mantissa-bit
setting** from the requested config. The mantissa count is only meaningful for the
emulated double-precision mode and is unaffected by a downgrade, so the result does
not need to carry it.

This matters because the emulated mode can quietly downgrade to native double
precision when the emulation was compiled out; recording the honored kind means
`meta.json` reflects the arithmetic that actually ran, not merely what was asked for.
At present the library reports back the same kind it was given, so this is
behavior-preserving today; the handshake is in place so it stays correct if a
downgrade ever occurs.

---

## 10. Writing the output directory and the run summary

The final step assembles the metadata and writes the directory. The metadata records
the steppe version (injected at build time, with a non-release sentinel when built
standalone), the honored precision tag and mantissa bits, the block size, the block
and population counts, the total and kept SNP counts, every filter setting, the three
input paths, the population-selection string, and the source hashes (per section 7).

The directory writer produces `f2.bin` (the real `f2` values with their paired
variances), `pops.txt` (the population labels in P-axis order), and `meta.json`. A
writer failure is reported as an I/O error.

On success the command prints a short summary: the output path, P and the block
count, how many of the total SNPs were kept, the memory tier the run actually used,
the precision, the block size in both units, the ploidy mode with the per-kind sample
counts, and — when the writer produced one — an `f2_cache_id` for the artifact.
