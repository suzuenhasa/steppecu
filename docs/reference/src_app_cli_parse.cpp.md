# `cli_parse.cpp` reference

## 1. Purpose

`src/app/cli_parse.cpp` is the command-line front end for the `steppe` binary. It
turns the raw `argv` a user typed into a validated configuration and then hands off
to the real GPU compute for whichever subcommand was chosen.

It does exactly three things:

1. **Declares the whole command-line surface** — every subcommand (`qpadm`,
   `qpwave`, `f4`, `f3`, `f4-ratio`, `qpdstat`, `qpgraph`, `qpgraph-search`,
   `dates`, `qpfstats`, `f4-sweep`, `f3-sweep`, `qpadm-rotate`, `extract-f2`) and
   every flag each one accepts, together with its default and its meaning.
2. **Merges and validates** the parsed flags into a run configuration, applying a
   fixed precedence order (see section 3).
3. **Dispatches** to the matching `run_*_command` function, which is where the GPU
   work actually happens.

This is a pure host translation unit. It contains no CUDA code — the command-line
parsing library (CLI11) is named only here, and this file never includes a GPU
header. That isolation is deliberate and is checked by a build-time grep gate, so
the command-line layer stays cheap to compile and can't accidentally drag GPU code
into the parser.

Because the header for this file only declares the single entry point `run_cli`,
this `.cpp` is the authoritative and only place the full command-line contract is
written down.

---

## 2. How a subcommand runs (dispatch and exit codes)

`run_cli(argc, argv)` is the single entry point. It builds one CLI11 application,
registers every subcommand, parses, and either runs the chosen subcommand or prints
help.

The lifecycle for any one subcommand is:

1. CLI11 binds that subcommand's flags into a `CliArgs` struct (one plain
   value-holder per flag).
2. The subcommand's callback fires. It calls `build_config(...)` to run the
   precedence merge and validation (section 3).
3. If the configuration is invalid, the callback records the invalid-configuration
   exit code and returns.
4. Otherwise it calls the subcommand's `run_*_command(config)` — the real GPU
   compute — and records whatever that returns as the exit code.

Two mechanical details make this work reliably:

- **One `CliArgs` per subcommand, held in stable storage.** `run_cli` declares a
  separate `CliArgs` local for every subcommand and keeps all of them alive for the
  whole parse. CLI11 binds each flag to a field of these by reference (most flags bind
  their field directly), so the storage must outlive parsing. Only the selected subcommand's callback ever fires, so only
  the chosen subcommand's `CliArgs` carries user input; the rest stay at their
  defaults and are ignored.
- **One shared registration recipe, no `std::exit`.** Every subcommand is registered
  through a single `register_cmd` helper: it creates the subcommand, tags its command
  kind, runs a small setup lambda that attaches the flags, and installs a callback that
  builds the config and records the `run_*_command` result into a shared `code`. The
  callback returns normally — it does *not* call `std::exit` — so the stack unwinds and
  destructors run, and `run_cli` returns `code` after parsing.

**Bare invocation prints help.** The app is configured to require zero or one
subcommand. Zero subcommands (a bare `steppe` with no verb) is not treated as an
error: after parsing, `run_cli` prints the full help text and returns the success
exit code, so a bare invocation is a clean, documented no-op. One subcommand runs
that subcommand; after parsing, `run_cli` prints the help text only when no subcommand
was selected, then returns the recorded exit code. Two or more subcommands is a parse
error.

---

## 3. The configuration precedence chain (build_config)

`build_config` is the one place the parsed flags become a validated run
configuration. It runs a fixed four-stage merge, from lowest priority to highest:

1. **Defaults** — the built-in defaults.
2. **Config file** — a TOML file named by `--config`, merged only if that flag was
   given and non-empty.
3. **Environment** — any `STEPPE_*` environment variables.
4. **Command-line flags** — the `CliArgs` the user actually typed.

Later stages override earlier ones, so the command line always wins, then the
environment, then the config file, then the defaults. After merging, the builder
validates the result.

On failure the function prints the builder's own reason to standard error in the
form `steppe: invalid configuration: <reason>` and returns no value. The calling
subcommand callback maps that empty result to the invalid-configuration exit code.
On success it returns the finished configuration.

---

## 4. The version string

`steppe --version` prints a single version string, and `STEPPE_VERSION` is the sole
authority for it. The build injects the real value — the top-level project version —
into this translation unit through the app's CMake configuration.

If the macro is somehow absent (for example when this file is compiled standalone,
outside the normal build), the code falls back to the sentinel `"0.0.0+unknown"`.
That sentinel is deliberately **not** a real-looking version: it exists so that a
build with the version wiring missing announces itself as unknown rather than
silently impersonating a real release. The version is single-sourced on the project
version and nowhere hard-codes a duplicate.

---

## 5. Shared flag helpers and the byte-identical `--help` rule

Many flags appear on more than one subcommand. Rather than repeat the binding code,
this file factors each shared flag (or small group of flags) into a small helper
function that attaches it to a given subcommand and points it at that subcommand's
`CliArgs`.

### Why the help strings are parametrized

Several helpers take the flag's help text as a parameter instead of hard-coding it.
The reason is a strict invariant: **deduplicating a flag must not change any
subcommand's `--help` output by a single byte.** Two subcommands often share the
same flag but describe it slightly differently — for example the `--f2-dir` help is
more verbose on `qpadm` ("the f2_blocks directory (f2.bin + pops.txt + meta.json)")
than on `qpwave` ("the f2_blocks directory"), and the `--left` help is semantically
different between `qpadm` ("left source population labels") and `qpwave` ("left
population set; left[0] is the reference"). Passing each caller's exact current
string as a parameter lets the code be shared while every subcommand's help text
stays exactly what it was. Helpers whose help text is genuinely identical across all
callers (such as `--target`, always "Target population label") hard-code the string
and take no help parameter.

### Global flags (every subcommand)

`add_common_flags` attaches the resource and precision flags shared by all
subcommands:

| Flag | Values | Meaning |
|---|---|---|
| `--device` | `auto` \| `<ordinal>` \| `<ordinal>,<ordinal>` | Which CUDA device(s) to use. GPU-only — there is no `cpu` option, and passing `cpu` is rejected during configuration validation, not here. |
| `--precision` | `emu40` \| `emu32` \| `fp64` \| `tf32` | Matrix-multiply precision. Default `emu40`. |
| `--config` | path | A TOML config file. Sits below the command line but above the environment and defaults in the precedence chain (section 3). |

### Output flags (the commands that emit a table)

`add_output_flags` attaches the two result-destination flags shared by the fit and
statistic subcommands:

| Flag | Values | Meaning |
|---|---|---|
| `--out` | path | Output file. Standard output if omitted. |
| `--format` | `csv` \| `tsv` \| `json` | Output format. Default `csv`. |

### The qpAdm-family option flags

`add_qpadm_option_flags` attaches the model-fit knobs shared by `qpadm`, `qpwave`,
and `qpadm-rotate`. Their names mirror the underlying options so that a bare
invocation reproduces the reference results[^at2].

| Flag | Default | Meaning |
|---|---|---|
| `--fudge` | `1e-4` | The ridge constant added for numerical stability (parity default[^at2]). |
| `--als-iters` | `20` | Alternating-least-squares iteration count. |
| `--rank` | `-1` | The f4-matrix rank used for the fit; `-1` means auto (number of left populations minus one). |
| `--rank-alpha` | `0.05` | Significance level for the automatic rank decision. |
| `--allow-neg` / `--no-allow-neg` | on | Whether negative admixture weights are allowed. A paired boolean flag; default on. |
| `--jackknife` | — | Standard-error policy: `0` none, `1` feasible-only, `2` all. Level `2` is meaningful only for the rotation command. |
| `--p-se-threshold` | — | The p-value survivor gate used when jackknife policy is feasible-only. |
| `--se-require-p` | off | When set, feasible-only also requires the p-value to clear `--p-se-threshold`. |

The graph commands reuse a single `--fudge` binding of their own (with different
help text describing it as the graph edge-solve ridge); it is a distinct flag from
the qpAdm-family `--fudge` above, even though both write the same underlying field.

---

## 6. Population inputs for the standalone f-statistics

The bare-statistic commands (`f4`, `f3`, `f4-ratio`, and `qpdstat`) do not take a
target/left/right split. They take fixed-width tuples of populations. Each accepts
the populations two equivalent ways: as row-aligned columns (`--pop1`, `--pop2`, …),
or as a single flat `--pops` list read in groups of the tuple width. Every list is
comma- or space-separated.

### f4 quartets

`f4` takes quartets. Column meaning follows `admixtools::f4` with `comb = FALSE`[^at2], so
row `k` is the quartet `(pop1[k], pop2[k], pop3[k], pop4[k])`:

| Column | Role |
|---|---|
| `--pop1` | p1 — the first population. |
| `--pop2` | p2. |
| `--pop3` | p3 — the first outgroup (R0). |
| `--pop4` | p4 — the second outgroup (R1). |

`--pops` supplies quartets flat, in groups of 4: `p1,p2,p3,p4[,p1,p2,p3,p4,...]`.

### f3 triples

`f3` takes triples, `(C, A, B)` per row:

| Column | Role |
|---|---|
| `--pop1` | C — the apex / outgroup / target. |
| `--pop2` | A — the first argument. |
| `--pop3` | B — the second argument. |

`--pops` supplies triples flat, in groups of 3: `C,A,B[,C,A,B,...]`.

### f4-ratio 5-tuples

`f4-ratio` takes 5-tuples and computes `alpha = f4(p1,p2;p3,p4) / f4(p1,p2;p5,p4)`
(following `admixtools::qpf4ratio`[^at2]):

| Column | Role |
|---|---|
| `--pop1` | p1. |
| `--pop2` | p2. |
| `--pop3` | p3 — the numerator's third slot. |
| `--pop4` | p4 — the shared fourth slot. |
| `--pop5` | p5 — the denominator's third slot. |

`--pops` supplies 5-tuples flat, in groups of 5: `p1,p2,p3,p4,p5[,...]`.

`qpdstat` reuses the f4 quartet input verbatim (see section 8 for how it interprets
those quartets on its two paths).

---

## 7. The f-statistic sweep flags

`f4`, `f3`, and `qpdstat` can each switch from an explicit tuple list into a
**sweep** over every combination, and `f4-sweep` / `f3-sweep` are standalone
sweep-only commands. Both routes share the same on-device filter flags.

- **The enable flag.** `--all-quartets` (on `f4` and `qpdstat`) or `--all-triples`
  (on `f3`) turns on the sweep. It sweeps every combination — every C(P,4) quartet
  or C(P,3) triple — over the population subset named by `--pops`. An empty `--pops`
  means the whole f2 directory. When the enable flag is absent, the command stays on
  its explicit-tuple path and its output is byte-identical to before, so the
  reference results are unaffected.
- **The filter (`--min-z`, `--top-k`, `--sure`).**
  - `--min-z` keeps every item whose absolute z-score is at least this value
    (default 3.0). It is mutually exclusive with `--top-k`.
  - `--top-k` keeps the K items with the largest absolute z-score, using a
    bounded device-side reservoir that keeps only about K rows resident. It is
    mutually exclusive with `--min-z`.
  - `--sure` lifts the maximum-combinations safety cap; a sweep larger than the cap
    refuses to start without it.
- **`--shard-dir`** (sweep mode on the standalone-stat commands) writes the survivor
  table as a CSV under the named directory (created if absent) instead of to
  standard output or `--out`.

**The filter bounds the output, never the work.** Every combination is computed on
the GPU in order to test it against the filter, so a huge sweep is real GPU time
even when only a few results survive. The `--min-z` / `--top-k` filter only limits
how many rows are copied back and reported. The safety cap (`--sure`) exists
precisely because the compute cost scales with the number of combinations, not with
the number of survivors.

---

## 8. The subcommand catalog

Every subcommand is registered the same way, through the shared `register_cmd` recipe:
it gets a `CliArgs` (owned at `run_cli` scope), is tagged with its command kind, has its
flags attached (mostly through the shared helpers), and is given a callback that builds
the config and records its `run_*_command` exit code.

| Subcommand | What it does | Notable flags beyond the shared groups |
|---|---|---|
| `qpadm` | qpAdm fit over an f2_blocks directory. | `--f2-dir`, `--target`, `--left`, `--right` (right[0] = R0), plus the qpAdm option flags. |
| `qpwave` | qpWave rank sweep. No target; `left[0]` is the reference. | `--f2-dir`, `--left`, `--right`, plus the qpAdm option flags. |
| `qpadm-rotate` | qpAdm rotation: enumerate source subsets of a pool and fit each. | `--f2-dir`, `--target`, `--pool`, `--right`, `--min-sources` (default 1), `--max-sources` (`-1` = the whole pool), plus the qpAdm option flags. |
| `f4` | Standalone f4 statistic (est/se/z/p per quartet). | The quartet inputs (section 6) and the sweep flags (section 7). |
| `f3` | Standalone f3 statistic (est/se/z/p per triple). | The triple inputs (section 6) and the sweep flags (section 7). |
| `f4-ratio` | Standalone f4-ratio (alpha/se/z per 5-tuple). | The 5-tuple inputs (section 6). No sweep. |
| `qpdstat` | D-statistic with two paths (see below). | The f4 quartet inputs, the sweep flags, and `--prefix`. |
| `f4-sweep` | GPU-only sweep over every C(P,4) quartet, survivors only. | `--f2-dir` plus the standalone sweep flags. |
| `f3-sweep` | GPU-only sweep over every C(P,3) triple, survivors only. | `--f2-dir` plus the standalone sweep flags. |
| `qpgraph` | Single admixture-graph fit. | `--f2-dir`, `--graph` (the edge-list file), `--numstart` (default 10), `--diag-f3` (default 1e-5), `--constrained`/`--no-constrained` (on), and its own `--fudge`. |
| `qpgraph-search` | Bounded topology search over a leaf set. | `--f2-dir`, `--pops` (the leaf set, at least 3), `--max-nadmix` (0 or 1, default 1), `--numstart` (10), `--diag-f3` (1e-5), `--constrained`/`--no-constrained` (on), `--fudge`. |
| `dates` | Admixture dating from genotypes. | `--prefix` (a genotype triple; the `.snp` must carry a real centimorgan map), `--target`, and `--left` (exactly the two reference source populations). |
| `qpfstats` | Genotype-path joint f2 smoother that writes a smoothed f2 directory. | `--prefix`, `--pops` (the set to smooth over), `--out-dir`, `--blgsize` (jackknife block size in Morgans; default 0.05 = 5 cM). |
| `extract-f2` | Precompute an f2_blocks directory from genotypes. | A large filter/QC flag set — see section 9. |

### The two paths of `qpdstat`

`qpdstat` reads its four populations as quartets exactly like `f4`, but which
statistic it reports depends on which input it was given:

- **With `--f2-dir`** it reports f4 (the f2-cache path). This reproduces the
  f2-path convention and is byte-identical to the reference f4 mode[^at2].
- **With `--prefix`** it reads the genotype triple (`PREFIX.geno`, `PREFIX.snp`,
  `PREFIX.ind`) and reports the genotype-path normalized-D magnitude: the mean
  numerator over the mean denominator across per-SNP allele frequencies,
  block-jackknifed. This matches the reference genotype D-statistic with all-SNPs
  jackknifing[^at2].

The `--prefix` here (and on `dates` and `qpfstats`) is bound to a dedicated
genotype-prefix field, kept separate from the `extract-f2` `--prefix`. That
separation matters: the configuration builder expands the `extract-f2` prefix into
its component `.geno`/`.snp`/`.ind` paths, but these genotype-path commands need the
raw prefix left alone so their own dispatch can branch on it. So the two "prefix"
concepts do not share a field.

---

## 9. `extract-f2`'s genotype filters and defaults

`extract-f2` has the richest flag set because it is where genotypes are read and
filtered on the way to an f2_blocks directory. Several of its defaults intentionally
match the reference[^at2], and a few are frozen for reproducibility.

### Inputs and outputs

| Flag | Meaning |
|---|---|
| `--prefix` | A genotype triple prefix; sets `--geno`/`--snp`/`--ind` to `PREFIX.{geno,snp,ind}`. |
| `--geno`, `--snp`, `--ind` | Individual file paths that override the prefix expansion. |
| `--out-dir` (alias `--out`) | The output f2_blocks directory (holds `f2.bin`, `pops.txt`, `meta.json`). `--out` is a back-compatible alias. |
| `--pops` | An explicit population list. |
| `--auto-top-k` | Keep the K largest populations. |
| `--min-n` | Keep populations with at least N individuals. |

### Filter thresholds

| Flag | Default | Meaning |
|---|---|---|
| `--blgsize` | `0.05` Morgans (= 5 cM) | Jackknife block size, in Morgans (the parity convention[^at2]). |
| `--maf` | — | Minimum minor-allele frequency. |
| `--geno-max-miss` (alias `--maxmiss`) | — | Maximum per-SNP missing fraction. `--maxmiss` is the parity-ergonomic alias for the same field. |
| `--mind-max-miss` | — | Maximum per-sample missing fraction. |

### Flag-gated filters and their parity defaults

Note that `extract-f2` turns two of these on by default, to match the reference[^at2] — the
opposite of the underlying filter struct's own "everything off" defaults. This is
where those parity defaults are set.

| Flag | Default | Meaning |
|---|---|---|
| `--auto-only` / `--no-auto-only` | **on** | Keep only autosomes (chromosomes 1–22). Default on to match the reference; `--no-auto-only` disables it. |
| `--drop-mono` / `--no-drop-mono` | **on** | Drop monomorphic SNPs. Default on to match the reference polymorphic-only behavior; `--no-drop-mono` keeps them. |
| `--transversions` | off | Keep only transversion SNPs. |

### Enum-token flags

Three flags take a keyword and are mapped to a typed enum during configuration; an
unrecognized keyword is a configuration error.

- **`--strand-mode`** (`drop` \| `keep` \| `flip`) — the strand-ambiguous (A/T, C/G
  palindrome) SNP policy. `drop` is the default: it drops palindromes, which is the
  merge-safe, bit-identical frozen behavior. `keep` retains them, reproducing
  the reference default of keeping ambiguous SNPs. `flip` is a documented but
  not-yet-implemented token that currently behaves exactly like `keep` (no
  frequency-based reorientation yet). The raw keyword is carried through and mapped
  during configuration.
- **`--ploidy`** (`auto` \| `1` \| `2`) — the pseudo-haploid handling. `auto`
  (default) does per-sample detection (matching the reference pseudo-haploid
  adjustment); `1` forces pseudo-haploid; `2` forces diploid (the legacy hard-coded
  behavior). This flag validates its token inline and rejects anything else.
- **`--tier`** (`auto` \| `resident` \| `host` \| `disk`) — overrides the f2_blocks
  output memory tier. `auto` (default) uses the runtime tier-selection policy;
  `resident` forces the device-resident path (byte-identical to the default on small
  inputs); `host` and `disk` stream the input in SNP tiles so that the GPU peak
  memory stays independent of the number of SNPs, letting large runs complete that
  would otherwise exhaust device memory on the resident path. This command-line
  override takes precedence over the `STEPPE_FORCE_TIER` environment variable.

### Other flags

| Flag | Default | Meaning |
|---|---|---|
| `--dry-run` | off | Report the sizes, tier, and precision that a run would use, then stop without computing. |
| `--hash` / `--no-hash` | **off** | Compute a SHA-256 provenance hash of the source dataset. Off by default because hashing the whole genotype file is a tens-of-seconds whole-file read that would dominate the run for a provenance value that does not affect correctness. When on, it is overlapped on a background thread with the GPU decode/f2 pipeline. When off, `meta.json` records that the hash was skipped and leaves the hash fields empty as a deliberate-absence marker. |

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
