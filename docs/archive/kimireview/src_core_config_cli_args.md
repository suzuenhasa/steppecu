This is a clean, architecture-following data header, but it is not as disciplined as it first looks. A senior reviewer would nod at the design intent, then point out that the "every field is optional" contract is only half true, and that the file has started to absorb implementation trivia that does not belong in a parse-time struct.

## What's genuinely good

- **The `std::optional` precedence sentinel is the right call.** Lines 16-20 and 64-67 explain clearly why an unset optional must not override lower-precedence config layers. That is a real configuration-design competence signal.
- **CUDA-free boundary is explicit and respected.** Lines 10-14 and 24-26 keep the header dependent only on the C++ standard library; device tokens stay raw strings and are parsed/validated in `ConfigBuilder`, not here.
- **Namespace and include guard are correct.** `STEPPE_CORE_CONFIG_CLI_ARGS_HPP` (line 21) and `namespace steppe::config` (line 28) follow project convention.
- **Aggregate POD design.** No constructors, no virtuals, no methods — this is meant to be filled by CLI11 and consumed by `ConfigBuilder`. That separation of concerns is sound.

## What a senior developer would flag

**The "every field is optional" claim is contradicted by the actual fields.**

Lines 64-67 say:

```cpp
/// The flat, parse-time CLI surface. Every field is the std::optional "was it set?"
/// sentinel for the §9 precedence merge (an UNSET field does NOT override the lower
/// layer). The vector/string fields that default to empty use emptiness as the
/// "unset" sentinel where an empty value is itself the no-op (e.g. an empty pop list).
```

But `command` at line 70 is a bare enum, not optional:

```cpp
Command command = Command::None;
```

So the struct really has *three* sentinel strategies: `std::optional` for scalars, `empty()` for vectors, and a dedicated enum value for `command`. A senior reviewer would ask why `command` is not `std::optional<Command>` with no default, or at least would push the comment to stop claiming "every field is optional."

**The vector quartet design for `f4` and `f4-ratio` is fragile duplication.**

Lines 107-113 and 140-145:

```cpp
std::vector<std::string> pop1, pop2, pop3, pop4;
...
std::vector<std::string> pop5;
```

These are row-aligned columns plus a separate `pops` field (line 188) that the parser must split into groups of 4 or 5. The comment says "the app validates" length equality, which means this struct exports an invariant ("all four/five columns must have the same length") that it cannot enforce. A stronger design would be a single `std::vector<std::array<std::string, 4>>` or a dedicated `F4Args` sub-struct parsed before reaching this layer.

**`Command` is starting to look like a product roadmap rather than an implemented command set.**

Lines 34-50 list thirteen subcommands, several of which are annotated with not-yet-implemented or GPU-only caveats:

```cpp
F4Sweep,  ///< GPU-only all-combinations f4 sweep (every C(P,4); on-device filter).
...
QpGraphSearch,  ///< topology SEARCH v1 (--f2-dir + --pops -> exhaustive bounded enumeration).
Dates,    ///< admixture DATING (--prefix + --target + --left{2} -> the date + SE; cuFFT LD engine).
```

A parse-time enum should track what the *CLI can actually dispatch today*. If half these commands are stubs, the enum becomes a source of stale comments and `switch` `default` traps downstream. The `Dates` line even drags `cuFFT` into the supposedly CUDA-free header — only as a comment, but it signals scope creep.

**`qpdstat_prefix` carries too much implementation detail.**

Lines 132-138:

```cpp
/// `--prefix PATH` (the `qpdstat` command ONLY) — the genotype prefix for the
/// normalized-D MAGNITUDE (Part B; not yet implemented). DISTINCT from `prefix` below
/// (extract-f2's geno/snp/ind triple prefix) so the qpdstat guard checks it WITHOUT
/// triggering ConfigBuilder's extract-f2 prefix->geno/snp/ind expansion. When SET, the
/// qpdstat command fails fast with a "Part B not yet implemented" message; the --f2-dir
/// path reports f4 (the AT2 f2-path convention, proven byte-identical to qpdstat f4mode).
```

This is a wall of implementation reasoning that belongs in `ConfigBuilder` or the app, not in a flat CLI args struct. A config header should say what the flag means; the failure-fast behavior and provenance hash notes should live where the logic actually is.

**Magic defaults are documented but scattered.**

Defaults like `--min-z` `3.0` (line 169), `diag_f3` `1e-5` (line 123), and `blgsize` `0.05` (line 191) appear only in comments. They are explained, but a senior reviewer would want them as named constants or, better, defaulted in `QpAdmOptions` / `FilterConfig` and left `std::nullopt` here so the single source of truth is not a comment.

**Raw string enums (`device`, `precision`, `tier`, `format`) are intentional but still a minor hazard.**

Lines 78, 82, 210, and 221 store enum-like CLI tokens as `std::string`. The architecture says `ConfigBuilder` is the single parse site, which is good, but it means this struct can carry invalid tokens deep into the call stack before anyone checks them. Typed wrappers (e.g. `std::optional<RawDeviceToken>`) would make the "parse me later" intent self-documenting.

## The "slop" test

**Not slop.** The file is coherent, follows project conventions, and the comments explain *why* the design choices were made rather than just restating the field names. There is no obvious copy-paste drift, no leaked raw pointers, no mixed `printf`/`std::cout`, and no CUDA footguns in a header that correctly stays CUDA-free.

## What it actually looks like

This looks like **solid mid-senior systems code written by someone who understands the project's layering and config-precedence rules.** The author made a deliberate, defensible choice to keep CLI parsing thin and validation centralized, and the file reads as though it was written against an architecture document — which it explicitly cites.

At the same time, it has the mild bloat that happens when a header starts absorbing every caveat about how downstream code behaves. A senior reviewer would say: "Good bones, but tighten the invariants, stop using the struct comments as a design notebook, and make the sentinel strategy consistent before it grows more commands."

**Verdict:** B+. Clean, correct, and well-intentioned, but the mixed sentinel story and growing enum/command surface keep it short of exemplary.