# `cmd_readv2.hpp` reference

## 1. Purpose

`src/app/cmd_readv2.hpp` declares the single entry point for the `steppe readv2`
command — the top-level function the command-line tool calls to run READv2
kinship estimation over a genotype triple. It is a header in the application
layer, so it sits above the compute library and below `main()`: `main()` parses
the command line into a configuration and hands that configuration to the one
function this header exposes.

```cpp
int run_readv2_command(const config::RunConfig& config);
```

That declaration is the whole header. Everything about *how* the command runs —
the per-individual resolve, the fail-fast validation, the GPU sweep, and the
streaming output — lives behind this seam in `src/app/cmd_readv2.cpp` (see
`src_app_cmd_readv2.cpp.md`). This doc describes the contract the header
promises; the `.cpp` doc describes the mechanics.

The header is deliberately plain C++20 with no GPU (CUDA) code in it. It reaches
the GPU only indirectly, through the narrow `run_readv2` seam
(`include/steppe/readv2.hpp`), which is itself CUDA-free. That keeps the whole
application layer buildable and unit-testable on a machine with no GPU toolkit
installed, and the boundary is enforced by the same build-time grep gate that
guards the rest of `src/app`.

This command is the READv2 Phase 0+1 surface: the Phase-0 frozen output schema
and the Phase-1 GPU sweep behind it.

---

## 2. What READv2 computes

READv2 is a **pseudo-haploid windowed-mismatch** kinship estimator. It shares no
math with the f-statistic or qpAdm commands — it is a genotype-path tool that
answers a different question: for every unordered pair of samples, how closely
are they related?

For each pair it walks the SNP axis in non-overlapping windows of a fixed SNP
count and, in each window, measures the proportion of comparable sites at which
the two samples' hardcalls differ (the allele-mismatch proportion, "P0"). It
then:

- averages that per-window mismatch across the windows to get **P0_mean**;
- normalizes P0_mean by an **all-pairs background** (the median across every pair
  by default, or the mean with `--norm mean`) to get **P0_norm** — normalizing
  against the cohort's own typical unrelated-pair mismatch is what makes the
  degree call robust to marker density and data quality;
- classifies a relatedness **degree** from P0_norm (one of four frozen tokens);
- and reports a **z** confidence.

Each surviving pair becomes one row in the frozen Phase-0 schema:

```
sampleA  sampleB  n_windows  n_overlap_sites  P0_mean  P0_norm  degree  z
```

"Frozen" means these columns, in this order, are a stability contract: the
command's output is byte-comparable against a committed golden, so the schema
does not drift.

---

## 3. The defining decision: per-individual modeling

The single most important thing to understand about this command is that READv2
operates **per individual, not per population**. Every other genotype-path
command in steppe collapses samples into named population groups and computes on
the groups. READv2 does the opposite: each selected Genetic ID is resolved to its
*own singleton* group — one sample, one sweep index — so the "all-pairs" sweep is
literally over every pair of individuals, not every pair of populations.

That is why the input is a raw genotype triple (`--prefix PREFIX.{geno,snp,ind}`)
and an optional `--samples` file of Genetic IDs rather than left/right population
lists. The `.ind` labels are not used to bucket samples together; they are the
pair labels. When a row is emitted, the two sweep indices are mapped back to
their Genetic-ID labels and canonicalized so that `sampleA <= sampleB`
lexicographically — the same canonical pair key the concordance validator uses —
so a pair appears exactly once, in a stable order, regardless of which sample the
sweep visited first.

---

## 4. The pipeline this entry point drives

Calling `run_readv2_command` runs a fixed sequence:

1. **Resolve the genotype triple.** `--prefix` is expanded to the
   `.geno` / `.snp` / `.ind` paths. The prefix is required; its absence is a
   configuration fault.
2. **Read the optional `--samples` restriction.** A file of Genetic IDs, one per
   line (blanks and surrounding whitespace ignored). If given, only those samples
   are swept; an empty or unopenable file is a fault.
3. **Per-individual resolve.** Each selected Genetic ID becomes a singleton group
   (section 3), yielding the list of per-sample labels.
4. **Fail-fast validation.** The command refuses obviously bad runs *before*
   touching the GPU: it needs at least two individuals to form a pair, and the
   pair count `C(N,2)` must sit under the enumeration cap unless `--sure` is
   passed. (Diploid/heterozygous samples are also rejected — READv2 v1 requires
   pseudo-haploid hardcalls — but that check happens deeper, inside the sweep, and
   surfaces here as an invalid-configuration exit; see section 6.)
5. **Build GPU resources and run the sweep.** The resolved device is set up, the
   first usable GPU is required, and the `run_readv2` seam streams SNP tiles into
   a resident device bit-matrix, runs the all-pairs `__popc` mismatch sweep, forms
   the background normalizer, and hands each surviving pair to a streaming sink.
6. **Emit.** Rows are written to one of three destinations (section 5).

The run knobs the command translates from the config into the sweep are:
`--window-snps` (the non-overlapping window size, default 1000), `--norm`
(`median` default | `mean` background), `--min-overlap` (drop a pair with too few
comparable sites), and autosomes-only restriction (on by default, the READv2
convention; `--no-auto-only` disables it).

---

## 5. Streaming output and the three sinks

READv2 never materializes all `C(N,2)` rows at once — the sweep emits pairs one at
a time in ascending pair-rank order, and the command formats and writes each row
as it arrives. This is what lets a large cohort's pair list stay bounded in
memory even when the pair count is enormous.

There are three output destinations, chosen by which flags are set:

| Condition | Destination |
|---|---|
| `--shard-dir DIR` set | A directory of sharded CSV/TSV/JSON, plus a manifest sidecar. Pairs are sharded by first-sample index (a fixed stride keeps every pair of a given first sample in one shard). |
| neither set | Standard output. |
| `--out FILE` set | A single output file (truncated). |

All three go through the same `Readv2ShardWriter`, so the row formatting is
identical across destinations; only the fan-out differs. The output format
(`--format csv|tsv|json`, default CSV) applies to all three.

---

## 6. Exit-code contract

`run_readv2_command` returns a process exit code as a plain `int`, one of the
named `CliExitCode` values, which `main()` returns straight to the OS. The rule
is the same "faults are nonzero, a real result is success" split the other
commands use:

| Situation | Code | Name |
|---|---|---|
| Success — the sweep ran and rows were emitted | `0` | `kExitOk` |
| Missing `--prefix`, an unknown `--format`, a bad/empty `--samples` file, fewer than two individuals, a pair count over the cap without `--sure`, or a rejected diploid/het input | `2` | `kExitInvalidConfig` |
| Cannot create `--shard-dir`, cannot open `--out`, or a population-resolver I/O failure, or an I/O/format error surfaced from the sweep | `4` | `kExitIoError` |
| A GPU-runtime failure (including no usable GPU), mapped from the caught exception | `3` / `5` | `kExitDeviceOom` / `kExitRuntimeError` |

Two contract details worth calling out:

- **The enumeration cap is a guardrail, not a limit on correctness.** `C(N,2)`
  grows quadratically, so a few thousand samples is millions of pairs. The command
  refuses to start a sweep past the cap and tells the user to either pass `--sure`
  or narrow the run with `--samples`. This mirrors the `--sure` gate on the
  f-statistic sweeps: the compute cost scales with the number of pairs, so the cap
  fires on the *enumeration* count.
- **A rejected input is a configuration fault, not a crash.** A diploid or
  heterozygous sample makes READv2 v1 undefined, so the sweep throws a fail-fast
  reject; the command catches it and returns `kExitInvalidConfig` with a clear
  message rather than producing a wrong answer.

On success the command also prints a one-line summary to standard error —
individual count, pair count, emitted count, and the background value — so a run
is self-documenting even when the rows themselves streamed to a file or shard
directory.

---

## 7. Edge cases and invariants

- **`z` is NaN for a single-window pair.** The z confidence needs at least two
  windows to have a spread to speak of; a pair with `n_windows < 2` reports `z` as
  NaN rather than a fabricated number. The frozen schema carries it through
  faithfully.
- **`degree` is one of four frozen tokens.** The relatedness class is a small
  closed vocabulary, not free text — part of what makes the output golden-stable.
- **Canonical pair identity.** `sampleA <= sampleB` lexicographically is an
  invariant of every emitted row (section 3), so downstream tools and the
  concordance validator can key on the pair without re-sorting.
- **`--samples` selects, it does not reorder semantics.** Restricting to a subset
  of Genetic IDs changes which pairs exist and, because the background is an
  all-pairs statistic, changes the normalizer — the background is computed over
  the *selected* cohort, not the whole file.
- **No GPU, clean failure.** With no usable device the command surfaces a specific
  runtime fault and a nonzero code rather than crashing or silently degrading —
  there is no CPU fallback path for READv2.
