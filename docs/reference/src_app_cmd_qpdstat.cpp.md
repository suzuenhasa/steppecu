# `cmd_qpdstat.cpp` reference

## 1. Purpose

`src/app/cmd_qpdstat.cpp` implements the `steppe qpdstat` command: a D-statistic /
f4 calculation over a set of four populations (a "quartet"). It is the code behind
`run_qpdstat_command`, the single entry point the command-line dispatcher calls.

The command has two very different back ends, chosen by which inputs the user gave:

- **The precomputed-statistics path** (`--f2-dir`). The user points at a directory
  of already-computed f2 blocks (the `f2.bin` file plus a `pops.txt` list of
  population names). From those, the command reports an f4 statistic for each
  quartet. This path does no new number-crunching of its own — it is a thin wrapper
  over the existing f4 code path.
- **The genotype path** (`--prefix`). The user points at a raw genotype dataset
  (`PREFIX.geno` / `PREFIX.snp` / `PREFIX.ind`, or the PLINK `.bed`/`.bim`/`.fam`
  equivalent). From the per-SNP genotypes the command computes the classic
  *normalized* D-statistic magnitude, which the f2-directory path cannot produce
  because it does not have the individual SNPs.

Both paths print the same table of columns — the four population names plus an
estimate, a standard error, a z-score, and a p-value — and both reuse the f4
emitter so the output schema is identical no matter which path ran.

The file is plain C++20 and lives in the application layer. It contains no GPU
(CUDA) code directly; see section 9.

---

## 2. Choosing a mode (dispatch order)

`run_qpdstat_command` picks exactly one of three behaviors, checked in this fixed
order. The order matters: the first matching condition wins.

1. **Genotype path** — if `--prefix` is set (non-empty), the command runs the
   genotype-reading normalized-D calculation (`run_qpdstat_prefix`, section 6) and
   returns. This branch is checked first so that supplying a genotype prefix always
   takes precedence.
2. **Sweep mode** — otherwise, if the "all combinations" flag is set
   (`--all-quartets`), the command hands off to the shared f-statistic sweep, asking
   it to scan every group of 4 populations. The sweep enumerates and scores the
   combinations on the GPU and keeps only the survivors that pass a significance
   filter. If the user also passed a `--pops` subset, the sweep is over just those
   populations; an empty subset means the whole f2 directory.
3. **Explicit-list f2-directory path** — otherwise, the command reads the f2 blocks
   directory, resolves the user's explicit quartet list, and reports f4 for each
   quartet (the default `--f2-dir` behavior).

The sweep and the explicit-list path are deliberately separate: the sweep generates
its own combinations, while the explicit path only computes the quartets the user
named.

---

## 3. Why the f2-directory path equals f4

The f2-directory path reports an f4 statistic and does nothing D-specific. This is
correct, not a shortcut, and rests on a verified fact about ADMIXTOOLS 2:

- ADMIXTOOLS 2's `qpdstat`, when run on precomputed f2 data rather than raw
  genotypes, produces results byte-identical to its `f4`. The "f4mode" switch that
  would distinguish a D-statistic from f4 is a no-op without per-SNP genotypes, so
  `qpdstat(f2dir, f4mode=TRUE)` equals `qpdstat(f2dir, f4mode=FALSE)` equals `f4`.

Because of that, the f2-directory path is full parity with ADMIXTOOLS 2's f2-data
`qpdstat` while adding zero new computation and zero new output code. It runs the
existing f4 engine once over the quartets and emits with the existing f4 emitter.
The reported estimate, standard error, z-score, and p-value are f4's, where the
z-score is estimate divided by standard error and the p-value is
`2 * (1 - Phi(|z|))` (Phi is the standard normal cumulative distribution). That
sign/z/p convention is exactly ADMIXTOOLS 2's D-statistic convention.

The genuinely different, per-SNP *normalized* D magnitude is only available on the
genotype path (section 6).

One detail of the f4 call: the f4 standard-error "fudge" term defaults to 0 here (a
bare f4 standard error), not the 1e-4 value that the qpAdm fit uses. The command
passes the option struct through with its default, so no fudge is added.

---

## 4. Building the quartet list

`build_quartet_names` turns the user's population arguments into a table where each
row is one quartet of four names. It accepts two mutually exclusive input styles and
prefers the first:

- **Row-aligned columns** — `--pop1`, `--pop2`, `--pop3`, `--pop4`, each a list of
  the same length. Row *k* across the four lists forms quartet *k*. If the four
  lists are not the same length, it fails with a message reporting the four lengths.
  Empty lists are also an error.
- **The `--pops` convenience** — a single flat list of names taken in groups of
  four: names 0–3 are the first quartet, names 4–7 the second, and so on. The list
  length must be a multiple of four, or it fails with a message. This is the easy way
  to express one quartet (`--pops A,B,C,D`) or several.

If neither style was supplied, it fails with a message explaining both options. The
function returns `false` and writes a human-readable reason into an `err` string on
any of these problems; the caller turns that into an "invalid config" exit code.

The genotype path and the f2-directory path both call this same builder, so the
quartet-input rules are identical across the two.

---

## 5. Resolving names to population indices

`resolve_quartets` maps each quartet of *names* to a quartet of *indices* into the
population axis (the ordered list of populations the run knows about). It does this
through a `PopResolver`, which knows the canonical spelling and position of every
population.

For each name it looks up the index; the first name that does not resolve makes the
whole function fail, with the offending name reported back in an `err` string. On
success it produces, per quartet, the four population-axis indices plus the four
*canonical* labels (the exact spelling from the population list). The canonical
labels — not the user's original typing — are what the emitter prints, so the
output always shows the dataset's own spelling of each population.

This resolver step is shared verbatim by both the genotype path and the
f2-directory path.

---

## 6. The genotype path (`--prefix`) and normalized D

`run_qpdstat_prefix` computes the per-SNP *normalized* D-statistic magnitude, which
the f2-directory path cannot produce. Conceptually, D is the mean over SNPs of a
numerator divided by the mean over SNPs of a denominator, both formed from per-SNP
allele frequencies, with the uncertainty estimated by a block jackknife over the
genome. The heavy lifting lives in `run_dstat`; this function is the front end that
prepares its inputs and formats its output.

The steps are:

1. **Expand the prefix into three file paths** in a format-aware way. An
   EIGENSTRAT-family prefix expands to `PREFIX.geno` / `PREFIX.snp` / `PREFIX.ind`;
   a PLINK prefix expands to `PREFIX.bed` / `PREFIX.bim` / `PREFIX.fam`. The actual
   on-disk format is detected when the genotype file is opened.
2. **Build the quartet name table** using the shared builder (section 4).
3. **Build the population union and resolve names** (section 7).
4. **Run the calculation**, passing the block size in Morgans (see below) and the
   GPU resources.
5. **Emit** the resulting table (section 8).

Several ADMIXTOOLS 2 parity choices are pinned inside `run_dstat` itself and are not
options here: the data are treated as forced-diploid, all SNPs are used
(`allsnps = TRUE`), and only autosomes are kept. Those are fixed to match
ADMIXTOOLS 2's `qpdstat` on genotypes.

**Block size units.** The jackknife groups SNPs into blocks along the genome. The
config surface speaks in centimorgans (`blgsize_cm`), but the block math works in
Morgans, so the command divides by 100 (100 centimorgans per Morgan) before handing
the value to `run_dstat`. ADMIXTOOLS 2's default block size is 0.05 Morgans, which
is 5 centimorgans.

---

## 7. The population union and the P-axis alignment invariant

This is the subtle part of the genotype path, and getting it wrong would silently
line up the wrong populations. The requirement is that the population indices this
command hands to `run_dstat` must refer to the exact same populations, in the exact
same order, that `run_dstat` itself decodes from disk. The two must agree perfectly.

The mechanism that guarantees this:

- **The union.** The command collects the distinct population names used across all
  quartets (in first-seen order — order does not matter here because the next step
  sorts). This union is the only set of populations that will be read. A four-
  population D-statistic over a dataset with tens of thousands of individuals (for
  example a ~27,594-individual panel) therefore decodes only the tiny handful of
  populations actually named, not the whole file.
- **The sorted partition.** The command reads the individual file restricted to
  exactly that union of populations (an "explicit" selection), which yields the
  populations **sorted ascending by label**. That sorted order *is* the population
  axis for this run.
- **The matching resolver.** The `PopResolver` is then built over that same sorted
  list. `run_dstat` internally performs the identical restricted-and-sorted read, so
  the indices the resolver produces line up one-for-one with what `run_dstat`
  decodes. The alignment is not a coincidence — both sides derive their order from
  the same restricted read plus the same ascending sort.

If opening the genotype file or reading the individual file fails, the command
reports an input error and returns an I/O error code. If the resolver itself is
invalid, it reports and returns an I/O error code.

---

## 8. Emitting results and the result shim

Both paths write the same table through the shared `emit_to_destination` helper,
which opens the destination (a `--out` file or standard output), writes, flushes,
and verifies the write. If the write is torn or short, it returns an I/O error code
rather than silently exiting success with a truncated file. The output format is CSV
by default, or TSV or JSON on request.

The actual row formatting is done by `emit_f4_result`, the same emitter the f4
command uses. The f2-directory path already holds an `F4Result` and passes it
straight in.

The genotype path holds a `DstatResult` instead, so it copies the fields across into
an `F4Result` shim before emitting: the four population names, the estimate,
standard error, z-score, p-value, the status, and the precision tag. The two result
types carry the same fields precisely so this copy is total — and reusing the f4
emitter this way keeps the genotype path's output schema byte-for-byte identical to
the f2-directory path's. This is the concrete reason the normalized-D result and the
f4 result look the same on screen: the D estimate/standard error/z-score/p-value are
reported in the same ADMIXTOOLS 2 D convention.

---

## 9. Exit codes and the no-CUDA rule

**Domain outcomes versus faults.** The command distinguishes an ordinary
statistical outcome from a real failure, and they exit differently:

- A **domain outcome** — for example a degenerate quartet whose covariance is not
  positive-definite — is recorded as a row (with the affected values reported as
  not-a-number where appropriate) and the process still exits success. The command
  computes what it can for every quartet and continues rather than aborting the
  whole batch. This "record and continue" behavior is why the final return maps the
  result status through `exit_code_for`, which turns domain statuses into a success
  code.
- A **fault** — bad population names or a missing/unreadable directory, missing
  genotype files, no CUDA device present, a device out-of-memory, or any file /
  format / CUDA-runtime error — returns a nonzero exit code. Faults on the device
  side are caught and routed through `exit_code_for_caught`, which maps a genuine
  device out-of-memory to its own code. Because steppe is a GPU product, a machine
  with no CUDA device is treated as a fault with a clear message, not a fallback.

**No CUDA in this file.** The command is application-layer C++ only and never
includes a CUDA header. It reaches the GPU exclusively through CUDA-free seams — the
resource builder, the f2-block uploader, and the f4 / dstat entry points — exactly
as the f4 command does. Keeping the GPU out of this translation unit is a
deliberate layering rule (enforced by a build-time check), and the command's `main`
owns all printing to standard output and standard error so the underlying library
never prints on its own.
