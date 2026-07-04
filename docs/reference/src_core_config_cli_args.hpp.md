# `cli_args.hpp` reference

## 1. Purpose

`src/core/config/cli_args.hpp` defines `CliArgs`, the flat, mutable struct that
holds exactly what the user typed on the command line. It is one raw field per
flag — nothing computed, nothing validated, nothing resolved to an index or a
device handle yet. It is the surface the command-line parser fills in, and the
input the configuration builder later folds into a finished, frozen run
configuration.

steppe layers its settings from lowest to highest precedence:

1. compiled-in defaults,
2. a TOML configuration file,
3. environment variables (the `STEPPE_*` set),
4. the command line.

The command line sits at the top: whatever the user typed wins over everything
below it. `CliArgs` is that top layer, captured as data. The flow is:

- The command-line parser binds each flag into a field of `CliArgs`.
- `ConfigBuilder::merge_cli(const CliArgs&)` folds those fields down over the
  lower layers (defaults, then TOML, then environment).
- `build()` validates the merged result and freezes it into an immutable
  `RunConfig`.

`CliArgs` itself does no resolving. A label like a target population is carried
as a string and only turned into an index by the application against the
population list. A raw string like `--device 0,1` is carried verbatim and only
parsed into a device list inside the configuration builder. That builder is the
single place strings become validated structured values, so the mapping logic
never gets scattered across the parser.

### CUDA-free by contract

The header depends only on the C++ standard library — no GPU toolkit. It names
only standard types and the small CUDA-free enums declared here. That keeps it
lightweight enough to compile into the core library, the command-line tool, and
(later) the language bindings without any of them being forced to pull in the
device code. Because of this rule, none of the string-to-device or
string-to-precision translation lives here; it all happens in the configuration
builder, which is allowed to know about the GPU-facing types.

---

## 2. The "was it set?" sentinel

Almost every field is a `std::optional`. The optional is not there to allow a
missing value in the usual sense — it answers a very specific question: **did the
user actually pass this flag on the command line?**

- An **unset** optional means the user did not pass the flag, so the merge leaves
  whatever the lower layer already decided (an environment variable, the TOML
  file, or the compiled default) untouched.
- A **set** optional means the user did pass the flag, so its value overrides the
  lower layer.

This is why a plain value would be wrong here. If a field were a bare `int` or
`bool`, there would be no way to tell "the user explicitly asked for the default
value" apart from "the user said nothing" — and the second case must not clobber a
value that a TOML file or environment variable set. The optional keeps those two
situations distinct, making the precedence rule explicit for every single field.

A few fields are vectors or plain strings instead of optionals. Those use
**emptiness as the unset sentinel**: an empty list or empty string is itself the
natural no-op (an empty population list selects nothing, an empty path names no
file), so wrapping them in an optional would add nothing.

---

## 3. Command — the selected subcommand

`Command` is an enum recording which subcommand the user chose. It is only the
parsed selection; mapping it to a real entry point is the application's job. Its
presence here lets `build()` apply validation that depends on which command is
running.

| Value | Meaning |
|---|---|
| `None` | No subcommand was given. Bare `steppe` falls through to printing help. |
| `ExtractF2` | Precompute the f2 statistics from genotype files into an f2 directory. |
| `QpAdm` | A qpAdm admixture fit. |
| `QpWave` | A qpWave rank test. |
| `QpAdmRotate` | qpAdm run repeatedly over enumerated subsets of a source pool ("rotation"). |
| `F4` | The f4 statistic. |
| `F3` | The f3 statistic. |
| `F4Ratio` | The f4-ratio (an admixture proportion from a ratio of two f4 statistics). |
| `Qpdstat` | The D statistic. |
| `F4Sweep` | GPU-only sweep over every group of 4 populations (every `C(P,4)`), filtered on the device. |
| `F3Sweep` | GPU-only sweep over every group of 3 populations (every `C(P,3)`), filtered on the device. |
| `Qpfstats` | The genotype-path joint f2 smoother: a genotype prefix plus a population list produces a smoothed f2 directory. |
| `QpGraph` | A single-graph qpGraph fit: an f2 directory plus a graph edge-list produces the fit. |
| `QpGraphSearch` | Bounded exhaustive topology search: an f2 directory plus a population list enumerates candidate graphs. |
| `Dates` | Admixture dating: a genotype prefix, a target, and two left populations produce a date and standard error (using an FFT-based linkage-decay engine). |

---

## 4. PloidyMode — extract-f2 ploidy policy

`PloidyMode` selects how the f2 estimator treats each sample's ploidy when
extracting f2 statistics. It exists because the f2 estimate depends on whether a
sample is scored as diploid or pseudo-haploid.

| Value | Meaning |
|---|---|
| `Auto` | **The default.** Auto-detect each sample's ploidy from its own genotypes: a sample with any heterozygous call is treated as diploid, a sample with none as pseudo-haploid. This matches the `adjust_pseudohaploid = TRUE` policy[^at2]. |
| `Diploid` | Force every sample to diploid (the `--ploidy 2` override). This is also the legacy hardcoded behavior. |
| `PseudoHaploid` | Force every sample to pseudo-haploid (the `--ploidy 1` override). |

---

## 5. Global resource and precision knobs

These apply to every fit and extract command. All three are **raw, unparsed**
strings — the configuration builder is the single place that turns them into
validated structured values, and an unrecognized token is rejected there as an
invalid configuration.

| Field | Flag | Meaning |
|---|---|---|
| `command` | (the subcommand) | Which subcommand was selected (see `Command`). `None` means no subcommand — the app prints help. |
| `device` | `--device` | Which GPUs to use, verbatim and unparsed: `"auto"`, `"0"`, or `"0,1"`. The builder parses and validates it into the device list. **GPU-only:** there is no `"cpu"` value; passing `"cpu"` is rejected at build time. |
| `precision` | `--precision` | Which arithmetic mode the heavy matrix-multiply stages run in: `"emu40"`, `"emu32"`, `"fp64"`, or `"tf32"`. The builder maps it to the precision policy; an unknown token is an invalid configuration. |
| `config_path` | `--config FILE` | A TOML file to merge in *below* the command line but *above* the environment and compiled defaults. Empty means no file. |

---

## 6. qpAdm and qpwave inputs

The core inputs for the model-fitting commands. Population labels are carried as
strings; the application resolves them to indices against the population list,
because the fitting engine works purely on indices.

| Field | Flag | Meaning |
|---|---|---|
| `f2_dir` | `--f2-dir DIR` | The precomputed f2 directory — the "compute once, fit many times" artifact. Carried here; resolved and uploaded by the application. |
| `target` | `--target` | The target population label for qpAdm and qpadm-rotate. |
| `left` | `--left a,b,...` | The source population labels for qpAdm, or the full left set for qpWave (where the first entry is the reference). |
| `right` | `--right r0,r1,...` | The outgroup (right) population labels. The first entry is the base outgroup. |
| `pool` | `--pool a,b,c,...` | For qpadm-rotate: the pool of sources the application enumerates subsets of. |

---

## 7. Standalone f-statistic inputs

These carry the inputs for the standalone statistic commands (`f4`, `f4-ratio`,
`qpdstat`). The quartet-style commands accept two equivalent input shapes.

### Row-aligned columns

The f4 command takes four parallel columns; quartet number *k* is
`(pop1[k], pop2[k], pop3[k], pop4[k])`. This is the "no combinations" form — it
evaluates exactly the listed quartets, row by row, and matches f4
with combinations disabled[^at2]. All four columns must have the same length (the
application checks this). The f4-ratio command adds a fifth parallel column, so
tuple *k* becomes `(pop1[k], ..., pop5[k])` and the reported quantity is
`f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)`; all five columns must be the same length.

| Field | Flag | Meaning |
|---|---|---|
| `pop1, pop2, pop3, pop4` | `--pop1`…`--pop4` | The four row-aligned quartet columns for the f4 command. |
| `pop5` | `--pop5` | The fifth row-aligned column, for the f4-ratio command only. |

### The `--pops` convenience shorthand

Both the f4 and f4-ratio commands *also* accept a single flat `--pops` list (the
`pops` field, documented under extract-f2 inputs) read in fixed-size groups —
groups of 4 for f4, groups of 5 for f4-ratio. This lets a one-line f4 or f4-ratio
be written without the several `--popN` flags.

### The qpdstat magnitude prefix

`qpdstat_prefix` (`--prefix`, on the `qpdstat` command only) names the genotype
prefix for the normalized-D magnitude path. It is deliberately a **separate
field** from the extract-f2 `prefix` below so that a check for it does not trip
the configuration builder's extract-f2 expansion of `prefix` into `.geno`/`.snp`/
`.ind` file paths. That magnitude path is not yet implemented: when this field is
set, the command fails fast with a "not yet implemented" message. The alternative
route — qpdstat via `--f2-dir` — reports f4 instead, following the
convention on the f2 path[^at2] (proven byte-identical to its D-statistic f4 mode).

---

## 8. qpGraph and qpgraph-search options

Options for the single-graph fit and the bounded topology search.

| Field | Flag | Meaning |
|---|---|---|
| `graph` | `--graph FILE` | The admixture-graph edge-list file, for the `qpgraph` command only. The file has two whitespace- or comma-separated columns per line — parent then child — with an optional `from,to` header row and `#` comment lines skipped. The leaf nodes must be populations present in the f2 directory. |
| `numstart` | `--numstart N` | The number of independent restarts for the multistart fit — the axis the optimizer runs in parallel. |
| `diag_f3` | `--diag-f3 X` | The regularization added to the f3 covariance. Matches the `diag_f3` default of `1e-5`[^at2]. |
| `constrained` | `--constrained` / `--no-constrained` | Whether drift edges are constrained to be non-negative. Matches the parity default of on[^at2]. |
| `max_nadmix` | `--max-nadmix N` | For qpgraph-search: the ceiling on the number of admixture nodes to consider. The first version supports 0 or 1. |

---

## 9. qpAdm option overrides

These mirror the internal qpAdm options struct one-for-one, so a bare invocation
with none of them set reproduces the reference (golden) results. Each is unset by
default, meaning "use the options struct's own default."

| Field | Flag | Overrides |
|---|---|---|
| `fudge` | `--fudge` | The fudge factor. |
| `als_iterations` | `--als-iters` | The alternating-least-squares iteration count. |
| `rank` | `--rank` | The model rank. |
| `rank_alpha` | `--rank-alpha` | The rank-test significance level. |
| `allow_negative_weights` | `--allow-neg` / `--no-allow-neg` | Whether admixture weights may go negative. |
| `jackknife` | `--jackknife 0\|1\|2` | The jackknife policy. |
| `p_se_threshold` | `--p-se-threshold` | The standard-error threshold used with the p-value. |
| `se_require_p` | `--se-require-p` | Whether a p-value is required alongside the standard error. |

---

## 10. qpadm-rotate enumeration bounds

Bounds on the subset enumeration the rotation performs over the source pool.

| Field | Flag | Meaning |
|---|---|---|
| `min_sources` | `--min-sources` | The smallest source-subset size to enumerate. |
| `max_sources` | `--max-sources` | The largest source-subset size to enumerate. |

---

## 11. f4-sweep and f3-sweep controls

The sweep commands are GPU-only and evaluate *every* combination of populations,
then keep only the survivors of an on-device filter. The full result table is
never brought back to the host — only the small survivor set is.

| Field | Flag | Meaning |
|---|---|---|
| `sweep_all_combinations` | `--all-quartets` (f4 / qpdstat) / `--all-triples` (f3) | Turns on sweep mode. When set, the command enumerates every group of *k* populations over the `--pops` subset (empty means the whole f2 directory) instead of an explicit row-aligned list. When unset, it takes the byte-identical explicit-list path that produces the goldens. |
| `sweep_min_z` | `--min-z Z` | Keep items whose absolute z-score is at least `Z` (the on-device filter). Default `3.0`. Mutually exclusive with `--top-k`. |
| `sweep_top_k` | `--top-k K` | Keep the `K` items with the largest absolute z-score, using a bounded device-side reservoir so only about `K` results are ever resident. |
| `sweep_sure` | `--sure` | Lift the combination-count safety cap. A sweep over more combinations than the built-in limit refuses to run without this flag, because the cap guards *compute time*: every combination is computed in order to test it against the filter, so the work is proportional to the number of combinations regardless of how few survive. |
| `shard_dir` | `--shard-dir DIR` | Write the survivor table to a CSV file under `DIR` (created if absent) instead of to standard output or `--out`. At sweep scale the post-filter survivor set is the small output; the full combination table always stays on the device. |

---

## 12. extract-f2 inputs

The inputs for building an f2 directory from genotype files, plus how populations
are selected.

| Field | Flag | Meaning |
|---|---|---|
| `prefix` | `--prefix` | A shorthand that sets the genotype/SNP/individual files to `PREFIX.geno`, `PREFIX.snp`, `PREFIX.ind`. |
| `geno` | `--geno` | The genotype file, set explicitly. |
| `snp` | `--snp` | The SNP file, set explicitly. |
| `ind` | `--ind` | The individual file, set explicitly. |
| `out_dir` | `--out-dir` (canonical) / `--out` (alias) | The output f2 directory. Also used as `--out-dir` by the qpfstats command. |
| `pops` | `--pops` | An explicit population selection. (Also reused as the flat convenience list for the f4 and f4-ratio commands — see section 7.) |
| `auto_top_k` | `--auto-top-k` | Select populations automatically by taking the top *K*. |
| `min_n` | `--min-n` | Select populations by a minimum sample-count threshold. |
| `blgsize` | `--blgsize` | The jackknife block size, in **Morgans** (the parity convention[^at2]). Default `0.05`, which is 5 centimorgans. The configuration builder converts Morgans to centimorgans (multiplying by 100) into the centimorgan-stored run configuration. |
| `ploidy` | `--ploidy auto\|1\|2` | The ploidy policy (see `PloidyMode`). Default `Auto`, matching the pseudo-haploid adjustment[^at2]. |

---

## 13. Filter overrides

These override the quality-control filters applied while reading genotypes. The
continuously-valued thresholds are unset by default (leaving the lower layer's
value in place). The flag-gated filters carry a parity note where
relevant.

| Field | Flag | Meaning |
|---|---|---|
| `maf` | `--maf` | Minimum minor-allele frequency to keep a SNP. |
| `geno_max_missing` | `--geno-max-miss` | Maximum per-SNP missing-data fraction. |
| `mind_max_missing` | `--mind-max-miss` | Maximum per-sample missing-data fraction. |
| `autosomes_only` | `--auto-only` / `--no-auto-only` | Keep only autosomes. For extract-f2 the effective default is on, matching the parity default[^at2]. |
| `drop_monomorphic` | `--drop-mono` / `--no-drop-mono` | Drop SNPs with no variation. For extract-f2 the effective default is on, matching the poly-only behavior[^at2]. |
| `transversions_only` | `--transversions` | Keep only transversion SNPs. |
| `strand_mode` | `--strand-mode drop\|keep\|flip` | The strand-ambiguous-SNP policy, carried as a **raw, unparsed** token; the builder maps it at the single merge site and an unknown token is an invalid configuration. When unset the behavior is `drop` — the frozen default that drops palindromic (A/T and C/G) SNPs for merge safety and bit-identical parity. `keep` retains ambiguous SNPs (reproducing the parity default[^at2]). `flip` is a documented not-yet-implemented token that currently behaves like `keep` (no frequency-based reorientation yet). |

---

## 14. extract-f2 run controls

Controls over how the extract-f2 run executes — where its output lands and what
optional work it does. These are all parity-neutral: they move bytes or add
side work, never a reported number.

| Field | Flag | Meaning |
|---|---|---|
| `tier` | `--tier auto\|resident\|host\|disk` | Overrides where the f2 output is stored. A **raw, unparsed** token; the builder maps it at the single merge site (an unknown token is an invalid configuration). `auto` (the default) lets the runtime memory-tier policy decide; the other three pin the tier. This is the higher-precedence twin of the `STEPPE_FORCE_TIER` environment variable — the config field wins, the environment variable remains the fallback. Moving the tier changes only where bytes live, never a reported number. |
| `dry_run` | `--dry-run` | Report the tiers and sizes that would be used, without doing the actual compute. |
| `hash_source` | `--hash` | Compute the source-dataset provenance SHA-256 hashes of the genotype, SNP, and individual files. **Off by default**, because hashing the whole genotype file is a tens-of-seconds full-file read-and-compress that dominated the extract-f2 run yet only yields a provenance value. When on, the hashing is overlapped on a background thread with the GPU pipeline. When off, the metadata records that no source hash was computed and leaves the hash fields empty. |

---

## 15. Output

Where results go and in what format.

| Field | Flag | Meaning |
|---|---|---|
| `out_file` | `--out FILE` | The output file. Standard output if unset. |
| `format` | `--format csv\|tsv\|json` | The output format. Default `csv`. |

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
