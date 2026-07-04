# `config_builder.hpp` reference

## 1. Purpose

`src/core/config/config_builder.hpp` declares `ConfigBuilder`, the one and only
mutable configuration type in steppe. Its whole job is to collect settings from
several sources, layer them in a fixed priority order, validate everything once,
and then hand back a frozen, immutable `RunConfig` that the rest of the program
reads but can never change.

The flow is always the same: construct a builder, fold in each source of
settings from lowest priority to highest, then call `build()`. `build()` is the
single place that checks the accumulated settings for legality, converts the raw
command-line strings into real typed values, and — if everything is valid —
seals the result. After `build()` succeeds, the configuration is `const` and can
be shared freely.

The builder itself contains no GPU code and makes no GPU calls (see section 7).
It performs only the checks that can be done without a device, which is why it
can be unit-tested on a machine with no GPU at all.

---

## 2. The four config layers and their precedence

Settings arrive from four sources. They are layered so that a higher-priority
source overrides a lower one, but only for the fields it actually specifies. In
increasing priority:

| Priority | Layer | Where it comes from |
|---|---|---|
| Lowest | Compiled defaults | The real struct defaults baked into the program |
| ↓ | TOML file | A config file the user points at |
| ↓ | Environment | The `STEPPE_*` environment variables |
| Highest | Command line | The flags the user typed |

The key rule: **an unset field never clobbers a lower layer.** Each higher layer
overrides only the fields it genuinely set. "Was this actually set?" is tracked
with a per-field sentinel — an `std::optional` that is present on the command
line, a present environment variable, or a present TOML key. A field the user
left alone falls through to whatever the layer below it decided. This is why
passing a value that happens to equal the default is different from passing
nothing: passing nothing leaves a lower-layer override intact, while a bare value
with no sentinel would silently overwrite it.

The list-valued fields (the left, right, pool, and pops population lists)
override when they are non-empty; an empty list means "not specified" and leaves
the lower layer alone.

---

## 3. The layer API

Four methods fold the four layers in. Each returns a reference to the builder, so
calls chain in precedence order. They only accumulate raw state — none of them
validates or parses anything; that is `build()`'s job (section 4).

| Method | What it folds in |
|---|---|
| `with_defaults()` | Seeds the compiled defaults — the real struct defaults for the device, qpAdm-options, filter, and population-selection settings, plus the default output format and the default jackknife block size. This is the lowest layer. It is idempotent and meant to be called once, up front. |
| `merge_file(path)` | Folds a TOML config file over the current state, below the environment and command line. An empty path is a no-op (no file was requested). A non-empty path is recorded and checked at `build()` time (see section 6 for the current TOML status). |
| `merge_env()` | Folds the `STEPPE_*` environment variables over the current state — above TOML, below the command line. It reads only the documented `STEPPE_*` keys (for example the device, precision, and format keys). An unset variable leaves the lower layer intact; a set one overrides it. **Unknown `STEPPE_*` keys are ignored, not treated as errors** — a forward-compatibility key meant for a newer steppe is deliberately not this build's concern. |
| `merge_cli(args)` | Folds the parsed command-line arguments over the current state — the highest priority. Only the fields that were actually set on the command line override the lower layers; an unset field is left as-is. The population list fields override when non-empty. |

A typical call site is a single chain:
`builder.with_defaults().merge_file(path).merge_env().merge_cli(args).build()`.

---

## 4. `build()` — validate once, then freeze

`build()` is the fail-fast gate. It does three things in one pass:

1. **Parses** the raw string knobs (things like `--device 0,1`,
   `--precision emu40`, `--jackknife 1`) into the real typed structs.
2. **Range-checks** the numeric values.
3. **Freezes** the validated result into an immutable `RunConfig`, or reports the
   first violation it finds.

Validation happens exactly once, here, and nowhere else downstream. Everything
past this point sees only valid configuration. A bad token, an out-of-range
value, or a GPU-only violation is rejected outright — it is never silently
coerced into something legal.

`build()` returns a value-or-error result (`BuildResult<RunConfig>`). On success
it holds the frozen `RunConfig`; on any violation it holds the error status
`InvalidConfig`, and the human-readable reason is available separately through
`error_message()` (section 5). Success or failure is checked with `has_value()`
or by treating the result as a boolean.

### Why the return type is `BuildResult` and not `std::expected`

`BuildResult<RunConfig>` is a small, GPU-code-free stand-in for the standard
library's `std::expected`. The design intent is `std::expected<RunConfig,
Error>`, but that type is C++23 and this code compiles as C++20, where it is not
available. `BuildResult` provides the same slice of the interface the callers use
(`has_value()`, `value()`, `operator*`, `operator->`, and an `error()` accessor),
so failure sites read the same as they would with `std::expected`. When the
toolchain moves to C++23 it can be swapped out with no changes at the call sites.
The error category for every static violation the builder can find is
`InvalidConfig`.

---

## 5. What `build()` parses and range-checks

These are the conversions and bounds `build()` enforces. Every failure below is
reported as `InvalidConfig`.

### Token parsing (raw string → typed value)

| Raw knob | Becomes | Rules |
|---|---|---|
| `--device` | The device list in `DeviceConfig` | `auto` or empty means auto-enumerate every visible GPU; `0` or `0,1` are explicit device ordinals. `cpu` is **rejected** — steppe is GPU-only. A negative, duplicate, or non-numeric ordinal is rejected. |
| `--precision` | The precision policy | `emu40` and `emu32` map to emulated double precision with 40 or 32 mantissa bits; `fp64` maps to native double precision; `tf32` maps to the TF32 screening mode. An unknown token is rejected. |
| `--jackknife` | The jackknife policy | Must be `0`, `1`, or `2`; out of range is rejected. |
| `--format` | The output format | Must be `csv`, `tsv`, or `json`; unknown is rejected. |
| `--strand-mode` | The strand-ambiguous-SNP policy in the filter | `drop`, `keep`, or `flip`; an unknown token is rejected. |
| `--tier` | The memory-tier override in `DeviceConfig` | `auto`, `resident`, `host`, or `disk`; an unknown token is rejected. |

### Numeric range checks

| Value | Allowed range |
|---|---|
| `fudge` | greater than or equal to 0 |
| `als_iterations` | greater than or equal to 1 |
| `rank_alpha` | strictly between 0 and 1 |
| `blgsize` (jackknife block size) | greater than 0 |
| The filter fractions (missing-data thresholds and similar) | between 0 and 1 inclusive |
| `min_sources` | greater than or equal to 1 |

### A unit conversion worth knowing

The command line takes the jackknife block size (`--blgsize`) in Morgans, which
matches the parity convention[^at2] (its default is `0.05` Morgans). The frozen
`RunConfig` stores this value in centimorgans instead. `build()` performs the
Morgans-to-centimorgans conversion (multiplying by 100) in this single place, so
the two unit conventions can never drift apart.

### The TOML check

A TOML path that was recorded by `merge_file()` but cannot be parsed (because no
TOML parser is compiled into this build) is rejected here rather than silently
ignored. See section 6.

---

## 6. The TOML seam: recorded now, parsed later

The builder accepts a `--config` TOML file today, but the actual TOML parser is
not yet part of the build. `merge_file()` is wired as an honest "no-op when
absent" seam:

- An **empty** path means no file was requested — a clean no-op.
- A **non-empty** path is recorded. Because there is no TOML parser compiled in
  yet, `build()` reports it as `InvalidConfig` rather than pretending to have
  read it. This makes the missing feature loud instead of silently dropping a
  file the user explicitly asked for.

The parser body and the full TOML handling land together in a later milestone.
Until then, the contract is: a requested-but-unreadable config file is an error,
never a silent skip.

---

## 7. CUDA-free by contract

This builder includes no device header and makes no GPU call. That is a
deliberate boundary, not an accident of the current code. The builder does the
**static** validation — token legality, numeric ranges, the GPU-only rule, and
flag conflicts — none of which needs a device present.

The **live** checks are somebody else's job. Whether a chosen precision can
actually be honored by the selected backend, how much GPU memory is really free,
and whether the requested devices exist are all runtime facts that the
application layer probes at run time through a separate, also-GPU-code-free
capabilities seam. Keeping those out of the builder is what lets the whole
config layer compile and be unit-tested with no GPU and no GPU toolkit
installed — the same reason the collaborating config structs are GPU-code-free.

---

## 8. Internal state and thread-safety

Internally the builder keeps its accumulating layers in one command-line-shaped
state object. Each `merge_*` call is a simple field-by-field override into that
object, and `build()` does the one parse-and-validate pass over the merged
result. The recorded TOML path and a flag marking whether a non-empty path was
given are held alongside it, plus the last failure reason string.

A `ConfigBuilder` is **not thread-safe** — it is meant to be owned and driven by
a single caller. This is not a limitation in practice: the whole point is to fold
the layers, call `build()`, and then work with the frozen `RunConfig`, which
*is* `const` and freely shareable across threads.

### `error_message()`

`error_message()` returns the human-readable reason the most recent `build()`
failed, or an empty string if `build()` succeeded or has not run. The application
prints this string to standard error; the library itself never prints. This keeps
`printf`-style output out of the library while still giving the user a specific,
readable explanation of what was wrong with their configuration.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
