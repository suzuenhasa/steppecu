# `config_builder.cpp` reference

## 1. Purpose

`src/core/config/config_builder.cpp` implements `ConfigBuilder` — the one mutable
config object in steppe. It gathers the run's settings from several layers, then
validates all of them once and freezes the result into an immutable `RunConfig`
that the rest of the program treats as read-only.

Two properties shape everything in this file:

- **It accumulates raw, still-string state.** As layers are folded in, the builder
  just remembers what each layer set (for example the literal text `"0,1"` for
  `--device`, or `"emu40"` for `--precision`). Nothing is parsed until `build()`
  runs, and then it is parsed exactly once. This keeps the layer-merge logic to a
  simple field-by-field override and puts all the parsing and range-checking in one
  place.
- **It contains no GPU code.** The file includes no CUDA header and makes no CUDA
  call. It never prints — when something is wrong, `build()` stores a human-readable
  reason and returns a failure status, and the calling application is the one that
  prints it. All the checks here are *static*: token spelling, numeric ranges,
  flag conflicts, GPU-only rules. The live checks that need a real device — is this
  GPU actually present, is there enough free memory, can the backend honor this
  precision — are the application's job at run time, not this file's.

---

## 2. The layer merge and its precedence

Settings arrive from four sources, in increasing order of authority:

```
compiled defaults  <  TOML file  <  environment (STEPPE_*)  <  command line
```

A higher layer overrides only the individual fields it actually sets; an unset
field never wipes out a value a lower layer already placed. That is what makes the
builder a clean fold: each `merge_*` call just copies in the fields that layer
provides and leaves the rest alone.

The four fold methods each return the builder itself, so calls can be chained.

| Method | Layer | What it does |
|---|---|---|
| `with_defaults()` | compiled defaults (lowest) | Resets the accumulated state to empty. The real defaults live as ordinary struct defaults inside `RunConfig` (device config, qpAdm options, filter config, `csv` output format, 5 cM block size), so "nothing set" at the raw layer simply means `build()` falls through to those struct defaults. Calling this also clears any recorded TOML path, so a reused builder starts clean. |
| `merge_file(path)` | TOML file | Records a requested config-file path (an empty path is a genuine no-op). It does not read the file — this build has no TOML parser compiled in, so `build()` rejects any non-empty recorded path with a clear error instead of silently ignoring it. |
| `merge_env()` | `STEPPE_*` environment | Reads the documented `STEPPE_*` variables and overrides only the fields those variables set. |
| `merge_cli(args)` | command line (highest) | Folds in the parsed command-line arguments; each option that was actually given on the line overrides the layer below. |

### Environment variables read

`merge_env()` looks at exactly these keys:

| Variable | Sets |
|---|---|
| `STEPPE_DEVICE` | the `--device` selection |
| `STEPPE_PRECISION` | the `--precision` selection |
| `STEPPE_FORMAT` | the output format |
| `STEPPE_F2_DIR` | the f2 cache directory |
| `STEPPE_CONFIG` | a requested TOML config path (treated like `--config`) |

An environment variable that is unset *or* empty counts as "not provided" — so
`STEPPE_FORMAT=` (set but empty) behaves exactly like leaving it unset, and the
lower layer stays intact. Any `STEPPE_*` key not in this list is ignored, not an
error, so a forward-looking variable meant for a newer steppe does not break this
one.

### How the command-line fold works

`merge_cli()` treats three kinds of fields differently, but all follow the same
"only if it was set" rule:

- **Optional strings** (device, precision, output file, target, and so on) are
  copied only when the option carries a value.
- **List fields** (the left/right/pool/pops population lists, the `--pop1`..`--pop5`
  quartet columns) are copied only when non-empty.
- **Numeric and boolean options** (fudge, iteration counts, rank, the filter
  fractions, the many on/off flags) are copied only when the option was present.

The command name always carries through, because the argument parser always sets
it. A `--config` given on the command line is the highest-authority request for a
TOML file and overrides any path an environment variable recorded.

---

## 3. The string-parsing helpers

A handful of small, self-contained helpers do all the text handling without pulling
in a regular-expression engine or locale machinery:

| Helper | What it does |
|---|---|
| `to_lower(s)` | Lower-cases a string, byte by byte, for case-insensitive token matching. |
| `trim(s)` | Strips leading and trailing whitespace. |
| `split_csv(s)` | Splits a comma-separated list, trimming each piece and dropping empty pieces. Dropping empties means a trailing comma or a doubled comma (like `0,,1`) does not produce a spurious blank entry. |
| `parse_int(tok, out)` | Parses a base-10 integer from a token that must be *fully* consumed — any trailing junk, a partial number, non-numeric text, or an overflow all fail. This is the fail-fast gate that rejects a bad numeric token rather than silently reading part of it. |
| `env(key)` | Reads an environment variable as an optional value, where both "unset" and "set but empty" come back as "no value." |
| `parse_enum(tok, table)` | Case-folds the token and looks it up in a fixed `{token, value}` table, returning the matching enum value. It hands back a `std::optional`, so a token that matches no entry comes back as `nullopt` (no match) rather than a wrong-but-plausible default. |

---

## 4. `build()` — validate once, freeze, and fail fast

`build()` is where the accumulated raw state is parsed, range-checked, and turned
into the frozen `RunConfig`. It does one pass over the merged state.

On any problem it returns a failure carrying the `InvalidConfig` status and records
a single human-readable sentence explaining what was wrong; that sentence is
available afterward through `error_message()` for the application to print. Nothing
is ever silently coerced — a bad token or an out-of-range value stops the build.

The very first check handles the TOML case: if a config file was requested (by
`--config` or `STEPPE_CONFIG`) the build fails, because no TOML parser is compiled
in yet, and the user deserves a clear error rather than a confusing no-op where
their file was quietly ignored.

The rest of `build()` walks through each group of options. The sections below
describe each group.

---

## 5. Device selection (`--device`)

The `--device` text is turned into the list of GPU ordinals to use.

- `cpu` is explicitly rejected. steppe is a GPU product; its CPU path exists only as
  a development and test reference, not as a runtime a user can select. The error
  points the user at the real choices.
- `auto` (or an empty value) leaves the device list empty, which downstream means
  "auto-detect every visible GPU in enumeration order."
- Anything else is treated as a comma-separated list of ordinals. Each token must
  parse as an integer, must not be negative (CUDA ordinals start at 0), and must be
  distinct — a duplicate ordinal is rejected because the device set and its order
  must be unambiguous (that order is what fixes the deterministic order in which
  multi-GPU partial results are later combined).

---

## 6. Precision selection (`--precision`)

The precision token is matched case-insensitively against a canonical set plus a
few documented aliases, and mapped to a typed precision policy.

| Token(s) | Resulting precision |
|---|---|
| `emu40`, `emu`, `emulated_fp64` | Emulated double precision, 40 mantissa bits |
| `emu32`, `emulated_fp64_32` | Emulated double precision, 32 mantissa bits |
| `fp64`, `native` | Native double precision |
| `tf32` | TF32 tensor-core arithmetic |

An unrecognized token is rejected with an error that lists the valid spellings and
their aliases.

This exact token set — including the aliases — is deliberately kept in lockstep
with the Python interface's precision parser, so that a precision is spelled the
same way on the command line and in Python. Changing one without the other would
let the two surfaces drift apart.

---

## 7. Memory-tier override (`--tier`)

The `--tier` text pins where the f2 results are held, overriding steppe's automatic
choice:

| Token | Tier |
|---|---|
| `auto` | let the automatic policy decide (the default) |
| `resident` | keep everything in GPU memory |
| `host` | spill to host RAM |
| `disk` | spill all the way to disk |

An unknown token is rejected. This is the single, validated place that sets the
tier override from the command line. When `--tier` is absent the override stays at
`auto`, and the resolution logic then falls back — first to the `STEPPE_FORCE_TIER`
environment variable, then to the automatic tier-selection policy. The command-line
token spellings (`resident`/`host`/`disk`) intentionally mirror the environment
variable's token spellings; they are repeated here as plain text literals so this
config layer stays free of the GPU-side header that defines them.

---

## 8. Output format (`--format`)

The output format accepts `csv`, `tsv`, or `json` (case-insensitive); any other
value is rejected. The default, when the option is absent, is `csv`.

---

## 9. qpAdm fit options and their range checks

The qpAdm-specific options are copied into the fit-options struct, each with a range
check that rejects a nonsensical value up front:

| Option | Rule |
|---|---|
| `--fudge` | must be at least 0 |
| `--als-iters` | must be at least 1 |
| `--rank` | `-1` means "use the automatic default (one less than the number of left populations)"; any other negative value is illegal |
| `--rank-alpha` | must lie strictly between 0 and 1 |
| `--allow-negative-weights` | a plain on/off flag |
| `--jackknife` | must be 0 (none), 1 (feasible models only), or 2 (all models); the integer is mapped to the corresponding policy |
| `--p-se-threshold` | must lie in the closed range 0 to 1 |
| `--se-require-p` | a plain on/off flag |

---

## 10. Genotype filters and the parity defaults

The read-time genotype filters are copied into the filter struct, again with range
checks:

| Option | Rule |
|---|---|
| `--maf` | must lie between 0 and 0.5 |
| `--geno-max-miss` | must lie between 0 and 1 |
| `--mind-max-miss` | must lie between 0 and 1 |

Two flags get special default treatment for the `extract-f2` command specifically,
to match the parity behavior[^at2]:

- **Autosomes only.** For `extract-f2`, restricting to autosomes (chromosomes 1–22)
  is turned on by default, because the f2 extraction restricts to
  autosomes by default. Every other command keeps the struct default (off).
- **Drop monomorphic SNPs.** For `extract-f2`, dropping SNPs with no variation is
  also on by default, because f2 is built on the polymorphic subset only,
  dropping every SNP that is monomorphic across the analysis populations before it
  partitions the blocks. Matching this changes the kept-SNP count and the per-block
  SNP counts — which in turn changes the jackknife standard errors — to line up with
  the parity SNP set. It never moves the f2 point estimates, because a monomorphic SNP
  contributes nothing to any f2 difference; it only aligns the SNP set and the block
  partition. Every other command keeps the struct default (off).

Both of these `extract-f2` defaults can be overridden explicitly (for example
`--no-auto-only` or `--no-drop-mono`), and an explicit flag always wins.

The `--transversions-only` flag is a plain on/off copy.

The `--strand-mode` option chooses how to handle strand-ambiguous (palindromic)
SNPs:

| Token | Behavior |
|---|---|
| `drop` | drop palindromic SNPs — the default and the frozen, bit-identical original behavior |
| `keep` | keep them (this matches the parity default[^at2]) |
| `flip` | accepted as a documented, not-yet-implemented token; it is stored as `Flip` but currently behaves like `keep` — it does not drop palindromes and performs no frequency-based reorientation yet |

An unknown token is rejected. When the option is absent the mode stays at `drop`, so
the parity path is bit-identical to the pre-flag behavior.

---

## 11. Population-selection modes (mutually exclusive)

For `extract-f2`, there are three ways to choose which populations to use, and they
are mutually exclusive — supplying more than one is rejected:

| Option | Mode |
|---|---|
| `--pops` | an explicit list of population labels |
| `--auto-top-k` | the top *k* populations (by sample count); *k* must be at least 1 |
| `--min-n` | every population with at least *n* samples; *n* must be at least 1 |

If none of the three is given, the selection stays in its "nothing requested" state,
which the application surfaces as a clear failure when `extract-f2` actually needs a
selection.

---

## 12. Block size and the Morgans-to-centimorgans conversion

`--blgsize` sets the jackknife block width, and it is worth understanding the units
carefully.

The `blgsize` block width is expressed in **Morgans**[^at2] (its default of `0.05` Morgans
is the same as 5 centimorgans). To make a bare `--blgsize 0.05` reproduce the reference
block partition exactly, steppe's flag also speaks Morgans. But `RunConfig`
stores the block width in **centimorgans**, so the Morgans-to-centimorgans
conversion happens right here at the command-line seam, using a single named
conversion constant. The stored field is never reinterpreted as Morgans — only the
flag's input unit is Morgans. The value must be strictly greater than 0.

---

## 13. Other run knobs

### Ploidy policy

`--ploidy` controls the pseudo-haploid handling in f2 extraction. The default is
automatic, matching the per-sample pseudo-haploid detection[^at2]; passing `1`
or `2` forces a uniform pseudo-haploid or diploid ploidy for every sample. The
actual detection or forcing happens later, against the gathered genotypes.

### Rotate pool bounds

For the rotating source-enumeration workflow, `--min-sources` and `--max-sources`
bound how many sources each candidate model draws from the pool:

- `--min-sources` must be at least 1.
- `--max-sources` accepts `-1` to mean "up to the whole pool"; any other value must
  be at least 1.
- A concrete (non-`-1`) maximum must not be smaller than the minimum.

### f-statistic sweep filters

The f4-sweep and f3-sweep filter knobs:

- `--min-z` (minimum absolute z-score to keep) must be at least 0.
- `--top-k` (keep only the k most significant results) must be at least 1.
- `--min-z` and `--top-k` are mutually exclusive — a sweep uses one filter or the
  other, not both.
- `--sure` (the flag that lets an enormous all-combinations sweep proceed) and
  `--all-combinations` are plain on/off copies, and `--shard-dir` is carried as a
  path.

---

## 14. I/O paths, prefix expansion, and qpGraph inputs

The remaining fields are input and output paths carried through more or less
verbatim: the f2 directory, the target, the population lists (left, right, pool,
the `--pop1`..`--pop5` quartet columns, and the `--pops` convenience tuple), the
output file, and the output directory.

Two path behaviors are worth calling out:

- **The two `--prefix` flags are different.** The qpDstat command's `--prefix` is
  carried verbatim and is *not* expanded into genotype file paths — it is only a
  sentinel that command reads. The `extract-f2` `--prefix P`, in contrast, *is*
  expanded into the three genotype files. For an EIGENSTRAT-family dataset that is
  `P.geno`, `P.snp`, and `P.ind`; for a PLINK dataset it is `P.bed`, `P.bim`, and
  `P.fam`. Which extensions to use is chosen by probing the filesystem (a present
  `.geno` wins; otherwise a `P.bed` selects the PLINK triple). This step only picks
  the sibling file paths — the authoritative on-disk format is pinned later when the
  reader is constructed. An explicit `--geno`, `--snp`, or `--ind` overrides the
  corresponding path that the prefix would have produced.
- **qpGraph inputs.** The graph file path is carried through, and its options are
  range-checked: `--numstart` (the number of random restarts) must be at least 1,
  and `--max-nadmix` (the ceiling on admixture nodes, which this version supports at
  0 or 1) must be at least 0. `--diag-f3` and `--constrained` are plain copies.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
