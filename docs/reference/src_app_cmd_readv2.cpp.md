# `cmd_readv2.cpp` reference

## 1. Purpose

`src/app/cmd_readv2.cpp` is the host-side command behind `steppe readv2` —
pseudo-haploid, windowed-mismatch kinship. Given a genotype triple it computes,
for every unordered pair of selected individuals, how often their alleles
mismatch across non-overlapping SNP windows, normalizes that by an all-pairs
background, classifies a relatedness degree, and streams one frozen-schema row
per surviving pair.

This translation unit is the *front end* only. It parses and validates the run
configuration, resolves which individuals are in play, enforces the safety
gates, picks where the output goes, and then hands the actual work to the GPU
through a single seam — `run_readv2` (declared in `include/steppe/readv2.hpp`).
It contains no CUDA code itself; the GPU is reached only across that seam, the
same way `dates` and the f-stat sweeps reach it. Everything above the seam here
is plain host C++.

The single most load-bearing idea in this file is the **per-individual** framing
(section 2). Everywhere else in steppe the `.ind` sidecar's population column
collapses many samples into a population; READv2 does the opposite — each
selected Genetic ID becomes its own singleton group, and the all-pairs sweep
runs over individuals, not populations.

---

## 2. Per-individual resolution, not pop-collapse

The rest of steppe keys its partition on the GroupID / population column: many
samples fold into one population, and the sweep enumerates populations. READv2
is a kinship tool, so it needs the opposite — it compares *people*.

That is why this command calls `io::read_individual_partition` (from
`src/io/individual_partition.hpp`) rather than the ordinary `read_ind`. Every
retained `.ind`/`.fam` row becomes a **singleton** `PopGroup` labelled by its
Genetic ID (the sample identity), in genotype-record order. The resulting index
space — one index per individual — is exactly what the all-pairs `C(N,2)` sweep
enumerates. The command then lifts each singleton's label into `pop_labels`,
and that vector of Genetic IDs is the identity map the output sink uses to name
the two samples in each pair.

`read_individual_partition` is also where the first fail-fast checks live: an
unreadable sidecar, a `--samples` ID that doesn't exist, or a **duplicate**
Genetic ID among the retained samples all throw. The duplicate case matters —
two samples with the same id would make the name-to-index resolver ambiguous, so
it is rejected up front rather than silently mis-attributed.

---

## 3. The frozen output schema

Every emitted row carries the Phase-0 schema, in this fixed order:

```
sampleA  sampleB  n_windows  n_overlap  P0_mean  P0_norm  degree  z
```

- **sampleA, sampleB** — the two Genetic IDs, canonicalized so `sampleA <=
  sampleB` lexicographically (section 6).
- **n_windows** — how many non-overlapping SNP-count windows the pair had.
- **n_overlap** — the number of comparable (both-called) sites.
- **P0_mean** — the mean, over windows, of the per-window allele-mismatch
  proportion.
- **P0_norm** — `P0_mean` divided by the all-pairs background.
- **degree** — one of the four frozen relatedness tokens (computed on the
  device side, passed through here as a `const char*`).
- **z** — a confidence score; `NaN` when the pair has fewer than two windows.

The schema is frozen: the sink copies each field of the device-side
`Readv2PairRow` straight into the on-disk `Readv2OutRow` with no reinterpretation
beyond the label lookup and the A/B canonicalization.

---

## 4. Fail-fast input validation

The command validates aggressively before it ever touches the GPU, so a bad
invocation costs nothing. In order:

1. **`--prefix` is required.** Without a `PREFIX.{geno,snp,ind}` prefix it prints
   the usage line and exits invalid-configuration. On success the prefix is
   expanded into the three component paths via `io::resolve_genotype_triple`.
2. **`--format`** must parse to `csv | tsv | json`; an unknown token is an
   invalid-configuration exit.
3. **`--samples`** (optional) is read by `read_samples_file`: one Genetic ID per
   line, blanks and extra whitespace ignored (only the first token on a line is
   taken). An unopenable or *empty* samples file is an invalid-configuration
   exit — an empty restriction list is treated as a mistake, not "select
   nothing".
4. **N >= 2.** After resolution there must be at least two individuals to form a
   pair; one or zero is an invalid-configuration exit with a count in the
   message.
5. **The enumeration cap** (section 5).
6. **A valid resolver.** The `PopResolver` built from the labels must be valid
   (this is a redundant guard on the label set — duplicates were already
   rejected in section 2 — and any failure exits with an I/O error).

Note the two diploid/het cases are *not* caught here: a sample that isn't
pseudo-haploid is rejected deeper down, inside `run_readv2`, which throws
`std::invalid_argument`. That throw is caught in the run block and mapped to an
invalid-configuration exit (section 7). READv2 v1 requires pseudo-haploid
hardcalls, and this is where "you handed me diploid data" becomes a clean
user-facing error rather than a crash.

---

## 5. The `C(N,2)` enumeration cap

The number of pairs grows quadratically, so a large `--samples` set (or a whole
dataset) can enumerate an enormous number of pairs. This command mirrors the
f-stat sweep's `--sure` gate: `choose2_saturating(N)` computes `C(N,2)` in a
way that can't overflow (it saturates to `~0ULL` rather than wrapping), and if
that count exceeds `kFstatMaxComb` the run refuses to start unless `--sure` was
passed. The refusal message names the actual pair count and points the user at
either `--sure` or a tighter `--samples` restriction.

Like the f-stat cap, this bounds the *enumeration*, and the compute cost scales
with the number of pairs — so the cap is about the work, not the output. `--sure`
lifts it for a user who genuinely means to run the whole thing.

---

## 6. The sink: labelling and A/B canonicalization

The GPU emits rows keyed by the per-individual sweep indices `i` and `j`. The
sink (`make_sink`) is the small closure that turns each such `Readv2PairRow`
into an output row:

- It looks up the two Genetic-ID labels via the resolver (`label_at(i)`,
  `label_at(j)`).
- It picks the lexicographically smaller label as `sampleA` and the larger as
  `sampleB` (`li <= lj`). This canonical ordering matches the concordance
  validator's `pair_key`, so a READv2 result and its validator agree on how a
  pair is named regardless of which index the sweep happened to visit first.
- It copies the schema fields through unchanged (section 3).
- It passes the *sampleA-side* sweep index to the shard writer as the shard key,
  so that after canonicalization every pair sharing a first-sample lands in the
  same shard (section 8).

Rows arrive at the sink one at a time, in ascending pair-rank order. The full
`C(N,2)` set never materializes in memory — that streaming discipline is what
lets the tool scale to large individual sets without holding every pair.

---

## 7. Running the sweep and the three output destinations

Once validation passes, the command builds the device resources
(`device::build_resources`), requires the first GPU to be present
(`require_first_gpu`; a missing GPU is a runtime-error exit), assembles the
`Readv2Options` from the config (window size, `--norm` mean-vs-median,
`--min-overlap`, and the autosomes-only default), and populates a
`Readv2Manifest` for the sidecar. Then it dispatches to one of three output
paths, chosen by which flags were set:

1. **`--shard-dir DIR`** — the sharded path. It creates the directory (failure
   to create is an I/O-error exit), builds a `Readv2ShardWriter` over it with a
   fixed shard stride, runs the sweep, records the background into the writer for
   the manifest sidecar, and calls `finish`.
2. **Neither `--shard-dir` nor `--out`** — stream to standard output through a
   `Readv2ShardWriter` wrapping `std::cout`.
3. **`--out FILE`** — open the file truncating (an unopenable file is an
   I/O-error exit) and stream through a `Readv2ShardWriter` wrapping it.

All three follow the same shape: build a writer, wrap it in the section-6 sink,
call `run_readv2(...)`, then `writer.finish("readv2")` — and if `finish` returns
an exit code, return it. The only differences are the sink's destination and
whether the background is threaded into a manifest.

The whole run block is wrapped in a try/catch:

- `std::invalid_argument` is the fail-fast **input** reject (the diploid/het case
  of section 4) and maps to an invalid-configuration exit.
- Any other `std::exception` is an input-or-device error and is mapped by
  `exit_code_for_caught`, which picks the right exit class for the underlying
  fault.

On success it prints a one-line summary to standard error (individual count,
pair count, emitted count, and the background value) and returns the result's
own status exit code.

---

## 8. The shard stride

When sharding, the writer uses a fixed stride of **256** and is keyed by the
sampleA-side sweep index. The intent is twofold: keep every pair of a given
first-sample together in one shard (never split one sample's pairs across
files), and land roughly a few hundred first-samples per shard, which is a
reasonable default granularity for downstream tools that consume shards in
parallel. Because the shard key is the *canonicalized* sampleA index (section 6),
the grouping is stable with respect to how the sweep ordered `i` and `j`.

---

## 9. `Readv2Options` and the autosomes-only default

The command maps the config into the device-facing option struct:

| Option | Source | Meaning |
|---|---|---|
| `window_snps` | `--window-snps` | The non-overlapping SNP-count window size. |
| `norm` | `--norm` | Background normalizer: `mean` selects `Readv2Norm::Mean`, anything else (the default) is `Readv2Norm::Median`. |
| `min_overlap` | `--min-overlap` | Drop a pair with fewer than this fraction of comparable sites. |
| `autosomes_only` | `--auto-only` / `--no-auto-only` | Restrict the SNP axis to chromosomes 1–22. |

One default worth calling out: **autosomes-only is on by default** for READv2
(it follows the config's `filter().autosomes_only`, which the CLI defaults on for
this command), and `--no-auto-only` turns it off. Restricting to autosomes is the
READv2 convention — sex chromosomes would distort a mismatch-based kinship
signal — so the tool ships that restriction on rather than making every user
remember to add it.

---

## 10. Exit codes

| Condition | Exit |
|---|---|
| Normal completion | the result's status exit (`exit_code_for(result.status)`) |
| Missing `--prefix`; bad `--format`; unreadable/empty `--samples`; `N < 2`; over the pair cap without `--sure`; a diploid/het input reject | invalid-configuration (`kExitInvalidConfig`) |
| A resolver that reports invalid; a `--shard-dir` that can't be created; a `--out` that can't be opened | I/O error (`kExitIoError`) |
| No usable first GPU | runtime error (`kExitRuntimeError`) |
| Any other input/device fault during the run | whatever `exit_code_for_caught` classifies it as |

The pattern is the same one the rest of the app follows: everything that can be
decided from the inputs alone fails as an invalid-configuration exit before the
GPU is touched, filesystem problems are I/O errors, and only genuine
device/runtime faults escalate past that.
