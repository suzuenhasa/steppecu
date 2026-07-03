# `run_config.hpp` reference

## 1. Purpose

`src/core/config/run_config.hpp` defines `RunConfig` — the single, frozen bundle of
settings that the command-line tool hands to the compute layer. Everything a run
needs to do its work (which subcommand, which GPUs, which precision, which files,
which populations, which filters) is resolved into this one object once, up front,
and then never changed.

Think of it as the finished, validated form of everything the user typed on the
command line, everything set through environment variables, and everything read from
a config file — all merged, type-checked, and locked down. After it is built, nothing
downstream can accidentally reinterpret a raw flag string or apply a filter twice: the
rest of the program only ever sees valid, already-resolved values.

`RunConfig` exposes read-only accessors and nothing else. There are no setters. The
only code that can fill in its fields is the builder that also validates them, so a
`RunConfig` that exists is, by construction, a `RunConfig` that passed validation.

---

## 2. How it is built and frozen

`RunConfig` is produced by exactly one thing: `ConfigBuilder::build()`. The builder is
the single place that turns the three raw input layers (command-line flags, environment
variables, config file) into typed values, validates them, and fails fast on anything
bad. If a device string is malformed, or a required population selection is missing, or
a precision token is unknown, the failure happens right there in `build()` — before any
GPU work, before any file is opened.

Two design choices keep the object immutable after that point:

- **Const accessors only.** Every public method is a `const` getter that returns the
  stored value (or a const reference to it). There is no public way to mutate a field
  once the object exists.
- **The builder is a friend.** `ConfigBuilder` is declared a `friend class`, which lets
  it — and only it — write the private fields during `build()`. This gives the "set once,
  then const forever" behavior without exposing any mutating method to the outside world.

The result is the frozen-configuration contract: validate once, freeze into an immutable
value, and treat the config as constant everywhere downstream.

---

## 3. CUDA-free by contract

`run_config.hpp` deliberately includes no GPU or CUDA headers. It is built only from
CUDA-free pieces: the public config structs (`DeviceConfig`, `FilterConfig`,
`Precision`), the qpAdm option struct (`QpAdmOptions`), the population selection type
(`io::PopSelection`), the `Command` enum, and standard-library strings and vectors.

This matters for two reasons:

- **It compiles and unit-tests with no GPU and no CUDA toolkit installed.** The whole
  `core/config` layer can be exercised on a plain machine.
- **It only *holds* resolved settings; it never *probes* hardware.** `RunConfig` stores
  the resolved `DeviceConfig` (which GPUs the user asked for, at what precision), but the
  live check of how much VRAM is actually free, whether two GPUs can talk to each other,
  and so on is the application layer's job at run time. The value object is a record of
  intent, not a snapshot of the machine.

---

## 4. What it holds: four validated structs plus carried paths

`RunConfig` is, at its core, four already-mapped-and-validated configuration structs plus
a set of input/output strings that the application consumes verbatim.

The four structs are the real entry-point types that the compute functions actually take,
rather than an abstract stand-in. The command-line flags are mapped straight onto them:

| Accessor | Type | What it carries |
|---|---|---|
| `command()` | `Command` | Which subcommand was selected (echoed from the parsed args so the app dispatches on the frozen config, not on the raw arguments). |
| `device()` | `DeviceConfig` | The resolved device and precision policy — which GPUs, in what order, at what arithmetic precision. |
| `qpadm_options()` | `QpAdmOptions` | The resolved per-call qpAdm options (fudge, rank, jackknife policy, and so on). |
| `filter()` | `FilterConfig` | The resolved on-the-fly quality-control filters applied while reading genotypes. |
| `pop_selection()` | `io::PopSelection` | The resolved population selection used by extract-f2. |

Everything else on `RunConfig` is a carried input/output value — a path, a name list, or
a scalar knob — resolved and stored so the command implementation can read it directly.

---

## 5. The selected command

`command()` returns a `Command` enum value naming which subcommand the user invoked. The
full set is: `None` (no subcommand — the app prints help), `ExtractF2`, `QpAdm`, `QpWave`,
`QpAdmRotate`, `F4`, `F3`, `F4Ratio`, `Qpdstat`, `F4Sweep`, `F3Sweep`, `Qpfstats`,
`QpGraph`, `QpGraphSearch`, and `Dates`.

The value is echoed from the parsed arguments into the frozen config so that dispatch, and
any command-specific validation, keys off the immutable object rather than re-reading the
raw command line.

---

## 6. The four resolved config structs

### `device()` — `DeviceConfig`

The resolved device and precision policy. The `devices` list is parsed from `--device`
and the precision from `--precision`. An empty `devices` list means "auto-enumerate every
visible GPU." This is exactly what the application feeds to the resource builder at run
time — the value object holds the policy, the app runs the live probe.

### `qpadm_options()` — `QpAdmOptions`

The resolved qpAdm per-call options (fudge factor, allele-sharing switch, rank,
rank-alpha, jackknife policy, and the rest). These default to the `QpAdmOptions` struct's
own defaults, which is deliberate: a bare invocation with no extra flags reproduces the
reference results.

### `filter()` — `FilterConfig`

The resolved quality-control filters applied on the fly during extract-f2. Every default
is a no-op, so unless the user asks for a filter, nothing is dropped.

### `pop_selection()` — `io::PopSelection`

The resolved population selection for extract-f2. The default state — an automatic top-K
selection with `k` left unset — is the "no selection requested" state. When extract-f2
needs an explicit selection, the application surfaces that unset default as a fail-fast
rather than guessing.

---

## 7. Carried input and output paths

These accessors return values resolved verbatim from the command line for the application
to consume directly.

| Accessor | Type | Default | Meaning |
|---|---|---|---|
| `f2_dir()` | `string` | empty | The precomputed f2 cache directory the fit commands read from. |
| `target()` | `string` | empty | The qpAdm/DATES target population. |
| `left()` | `vector<string>` | empty | The left (source) population names. |
| `right()` | `vector<string>` | empty | The right (outgroup/reference) population names. |
| `pool()` | `vector<string>` | empty | The candidate pool of source populations for rotation. |
| `out_file()` | `string` | empty | The output file path. |
| `format()` | `string` | `"csv"` | The output format. |
| `geno()` | `string` | empty | The genotype file path (an EIGENSTRAT-style triple with `snp` and `ind`). |
| `snp()` | `string` | empty | The SNP file path. |
| `ind()` | `string` | empty | The individual/sample file path. |
| `out_dir()` | `string` | empty | The output directory (for commands that write a directory of results, like an f2 cache). |
| `shard_dir()` | `string` | empty | Where a sweep writes its survivor CSV; empty means write to standard output or `--out`. |

### `qpdstat_prefix()`

The `--prefix` value for the qpdstat command — the genotype prefix for the normalized-D
("Part B") path. It is carried verbatim, not expanded into separate geno/snp/ind paths.
Empty means unset (the f2-directory-based f4-reporting path is used). A non-empty value
currently causes the qpdstat command to fail fast, because that Part-B path is not yet
implemented.

---

## 8. f4 and f4-ratio population columns

The `f4` and `f4-ratio` commands accept populations two different ways, and `RunConfig`
carries both so the command implementation can pick whichever the user actually used.

### Row-aligned columns

| Accessor | Type | Meaning |
|---|---|---|
| `pop1()` | `vector<string>` | Column 1 of the quartets (`--pop1`). |
| `pop2()` | `vector<string>` | Column 2 (`--pop2`). |
| `pop3()` | `vector<string>` | Column 3 (`--pop3`). |
| `pop4()` | `vector<string>` | Column 4 (`--pop4`). |
| `pop5()` | `vector<string>` | Column 5 (`--pop5`), used only by `f4-ratio`. |

These are row-aligned: quartet number *k* is `(pop1[k], pop2[k], pop3[k], pop4[k])`, so
the four (or five) vectors are read together by index. For `f4-ratio`, the fifth column
adds the denominator population, and the ratio for row *k* is
`f4(p1, p2; p3, p4) / f4(p1, p2; p5, p4)`. All default to empty.

### The `--pops` convenience

| Accessor | Type | Meaning |
|---|---|---|
| `pops()` | `vector<string>` | The raw `--pops` labels: names given in groups (4 at a time for `f4`, 5 at a time for `f4-ratio`), read directly by the command. |

`pops()` is a convenience form: instead of five separate column flags, the user lists the
names in order and the command groups them. The application picks whichever of the two
forms is non-empty. Note that `pops()` is distinct from `pop_selection()` — extract-f2
maps its selection into `PopSelection`, whereas the `f4` command reads these raw `--pops`
names directly.

---

## 9. extract-f2 controls

| Accessor | Type | Default | Meaning |
|---|---|---|---|
| `blgsize_cm()` | `double` | `5.0` (`kDefaultBlockSizeCm`) | The jackknife block size in centimorgans. 5 cM matches ADMIXTOOLS 2's default of 0.05 Morgans. |
| `ploidy()` | `PloidyMode` | `Auto` | The ploidy policy. `Auto` matches ADMIXTOOLS 2's per-sample auto-detection (a heterozygous call means diploid; none means pseudo-haploid). `--ploidy 2` and `--ploidy 1` force a uniform ploidy for every sample. |
| `min_sources()` | `int` | `1` | The smallest source-set size the rotation considers. |
| `max_sources()` | `int` | `-1` | The largest source-set size; `-1` means "up to the whole pool." |
| `dry_run()` | `bool` | `false` | When true, report the planned memory tiers, sizes, and precision and then exit without doing any compute. A planning aid; the default is a real run. |
| `hash_source()` | `bool` | `false` | When true, compute the source-dataset provenance SHA-256 hashes. Off by default because hashing the whole genotype file dominated extract-f2's cost and is a provenance value, not a correctness one. When requested, the command overlaps that large hash with the GPU pipeline on a background thread. |

---

## 10. f-statistic sweep controls

These knobs govern the all-combinations f4/f3 sweeps — the commands that enumerate every
possible quartet or triple over a chosen population subset and filter the results on the
GPU.

| Accessor | Type | Default | Meaning |
|---|---|---|---|
| `sweep_all_combinations()` | `bool` | `false` | When true (`--all-quartets` / `--all-triples`), route the f4/f3/qpdstat command to the GPU sweep over every combination of the `--pops` subset, instead of the explicit-list path. |
| `sweep_min_z()` | `double` | `3.0` | The minimum absolute z-score to keep a result, used in MinZ mode. |
| `sweep_top_k()` | `int` | `-1` | When greater than 0, switch to TopK mode and keep the K results with the largest absolute z-score. When `-1`, use MinZ mode with the `sweep_min_z()` threshold instead. |
| `sweep_sure()` | `bool` | `false` | When true (`--sure`), lift the safety cap on the maximum number of combinations a sweep will attempt. |

The filter mode is decided by `sweep_top_k()`: a positive value means TopK (keep the K
largest `|z|`), otherwise MinZ (keep everything with `|z|` at or above `sweep_min_z()`).

---

## 11. qpGraph controls

These carry the settings for the single-graph qpGraph fit and the bounded topology search.

| Accessor | Type | Default | Meaning |
|---|---|---|---|
| `graph_file()` | `string` | empty | The admixture-graph edge-list file path (`--graph`). Empty means unset. |
| `qpgraph_numstart()` | `int` | `10` | The number of random restarts for the fit (`--numstart`); matches the qpGraph option default. |
| `qpgraph_diag_f3()` | `double` | `1e-5` | The diagonal value added to the f3 covariance for stability (`--diag-f3`); matches the ADMIXTOOLS 2 default. |
| `qpgraph_constrained()` | `bool` | `true` | Whether the fit constrains admixture weights (`--constrained`); matches the ADMIXTOOLS 2 default. |
| `qpgraph_max_nadmix()` | `int` | `1` | The ceiling on admixture nodes for the topology search (`--max-nadmix`); the first version supports 0 or 1. |
