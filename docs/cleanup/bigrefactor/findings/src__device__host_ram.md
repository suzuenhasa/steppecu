# Review findings — src__device__host_ram

Files: src/device/host_ram.cpp

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/host_ram.cpp:12 — `#include <cstring>` is unused: no `str*`/`mem*` function from `<cstring>` is called anywhere in the file (the only `<cstring>`-adjacent symbol, `std::size_t`, comes from `<cstddef>` already pulled in via `<cctype>`/`<sys/sysinfo.h>`/`tier_select.hpp`); `iequals` uses a hand-rolled loop rather than `strcmp`/`strncasecmp`. Suggested: drop the `<cstring>` include.

## Group 5 — Hardcoded values / magic numbers

- [5.1][LOW] src/device/host_ram.cpp:49-51 — the STEPPE_FORCE_TIER env-string contract (`"resident"`, `"host"`, `"disk"`) is parsed via bare string literals at the call site. These are the one parse home (grep confirms no live-literal duplication elsewhere; f2_blocks_multigpu.cpp only does `std::getenv("STEPPE_FORCE_TIER")` and delegates here), so there is no drift bug today — but they implicitly mirror the `OutputTier` enum arms and the `DeviceConfig::ForceTier` switch above (44-47), and a future tier rename would silently miss this string table. Suggested (optional hygiene): hoist the three tier tokens to named `constexpr` strings (or a single tier-name table) next to the `OutputTier` enum so the env spelling and the enum stay coupled.
- All other Group 5 tasks clean for this unit: 5.2 — no hardcoded sizes/bounds; every dimension (`P`/`M`/`n_block`) and the VRAM/host-RAM figures are parameters, and all tier fractions/workspace sizes (`kResidentTierVramFraction`, `kHostTierRamFraction`, `kCublasWorkspaceBytes`) are named constants in config.hpp consumed by tier_select.hpp, not literals here. 5.3 — no duplicated numeric constant (the only literal, the `1u` mem_unit fallback at line 20, is a self-documenting "no scaling" guard for a kernel-reported unit and appears once). 5.4 — no hardcoded paths/IDs/device ids (the env-var name `"STEPPE_FORCE_TIER"` is read in the orchestrator, not here; this helper takes the already-read value as a param per the no-global design). 5.5 — no literal `32` (or any warp-size-ambiguous constant) anywhere in the file.

## Group 6 — Naming

No Group 6 issues found.
Notes (context-applied, NOT flagged): 6.1 — `si` (line 18) is the idiomatic name for the `struct sysinfo` it wraps, tight scope; `s`/`i`/`c` (lines 27-31) are short-scope locals in the tiny `iequals` helper (input string, loop index, lowered char) and read clearly; `unit` (line 20), `lower_lit` (lowercase literal) are descriptive; `P`/`M`/`n_block` (line 40) are the project's canonical domain-math dimension names (pops/markers/blocks per §12 spec), not opaque. 6.2 — names match behavior: `free_host_ram_bytes` returns free+buffer RAM in bytes, `resolve_output_tier` resolves a tier; no count/index or list/map mislabel. 6.3 — consistent in-file convention: snake_case functions/locals, PascalCase types (`OutputTier`, `DeviceConfig`); the `P`/`M` caps vs `n_block` snake mix is the codebase-wide domain-math convention, not a one-file inconsistency. 6.4 — `iequals`, `lit`, `vram`/`ram` are standard idioms; no nonstandard abbreviation.

## Group 7 — Duplication

- [7.1][LOW] src/device/host_ram.cpp:49-51 — three near-identical `if (iequals(env_value, "<token>")) return OutputTier::<X>;` lines differing only by the (string literal, returned enum) pair; a repeated 3-line shape parameterized by a constant. Self-contained (only env-parse home — Group 5 §5.1 confirms no off-file literal drift), so harmless today, but it is a hand-unrolled lookup table. Suggested (optional hygiene, dovetails with the §5.1 token-hoist): drive it from one `{const char*, OutputTier}` table and loop, so adding/renaming a tier touches a single row.
Notes (context-applied, NOT flagged): 7.1 — the `ForceTier` switch arms (44-47) are an enum→enum structural mapping; a `switch` is the idiomatic non-duplicated form, not a copy-paste block, so not flagged. 7.2 — no loop-invariant or repeated expression: in `iequals` (30-34) `s[i]`/`lower_lit[i]`/the `tolower` chain are necessarily per-iteration and computed once each; in `free_host_ram_bytes` the two casts (21-22) wrap distinct operands. 7.3 — the two `static_cast<std::size_t>` at 21-22 operate on different fields (`si.freeram` vs `si.bufferram`) and the widen-before-multiply is the correct overflow-safe form; not foldable. No `sizeof` in the file. 7.4 — no macro/helper-collapsible boilerplate; the switch and the if-chain are two distinct resolution stages and folding them would harm clarity.

## Group 8 — Comments

- [8.3][LOW] src/device/host_ram.cpp:21-22 — `free_host_ram_bytes` returns `si.freeram + si.bufferram` but neither the function name ("free") nor any comment explains why reclaimable buffer-cache (`bufferram`) is counted as free RAM; this is an intentional tier-policy decision (buffer cache is reclaimable under memory pressure) that drives `select_output_tier` host-RAM sizing, and a future reader could mistake it for a bug or "miss" missing `si.sharedram`/cached. Suggested: add a one-line rationale (e.g. "buffer cache is reclaimable, so count it as available for the host-RAM tier") and note the deliberate omission of other reclaimable fields.
- [8.3][LOW] src/device/host_ram.cpp:20 — the `si.mem_unit ? si.mem_unit : 1u` fallback documents the *what* via the ternary but not the *why*: older Linux kernels (pre-2.3.23) report `mem_unit == 0` meaning "fields are already in bytes", a non-obvious ABI quirk; absent rationale a reader may read the `1u` branch as dead. Suggested: add a short note that `mem_unit==0` is the legacy "values already in bytes" kernel convention.
- 8.1 clean — no restating-the-code comments; the closest, line 47 `// fall through to env, then auto`, adds the non-obvious fall-through *destination* (where control goes after `break`), which is rationale, not a restatement. 8.2 clean — no stale comments: the header (1-8) correctly describes the live `sysinfo(2)` probe + `resolve_output_tier` home, line 42's "frozen precedence" matches the switch→env→auto order, line 52's "unset ⇒ ignored ⇒ automatic" matches `iequals` returning false on null (27). 8.4 clean — no TODO/FIXME/HACK/XXX markers in the file (grep-confirmed).

## Group 9 — Constants & configuration

- [9.2][LOW] src/device/host_ram.cpp:49-51 — the env-tier config contract (the three tunable knobs `"resident"`/`"host"`/`"disk"`) is buried inline inside the `resolve_output_tier` policy logic rather than surfaced at the file top / a config struct next to the `OutputTier` enum and the `DeviceConfig::ForceTier` switch (44-47) it mirrors. These string tokens ARE configuration (the public STEPPE_FORCE_TIER spelling), so embedding them mid-function tangles the knob with the resolution control-flow. No drift bug today (this is the single parse home per the existing §5.1 note), so LOW. Suggested (optional, dovetails with §5.1/§7.1): hoist the tier tokens to named `constexpr` strings (or a `{const char*, OutputTier}` table) declared with the enum so the configuration surface is visible at a glance and stays coupled to the enum.
- 9.1 clean — no should-be-const left mutable: `unit` (20) and `c` (31) are already `const`; `struct sysinfo si{}` (18) is necessarily mutable (filled by `sysinfo(&si)`); the loop index `i` (29) is mutated in the `for`. Nothing here is `constexpr`-promotable that isn't already (all true constants live in config.hpp per §5.2). 9.3 clean — no positional booleans: `resolve_output_tier` takes a named `ForceTier` enum (not a bool flag) and dimension/size params; the `switch` uses named enum arms (`ForceTier::Resident`, etc.); `select_output_tier` (53) and `iequals` (27) are called with no boolean arguments. No `foo(true,false,...)`-style call anywhere in the file.

## Group 10 — Initialization

No Group 10 issues found.
Notes (context-applied, NOT flagged): 10.1 — every variable is declared at its point of first use and initialized in the same statement: `struct sysinfo si{}` (18) is value-initialized immediately before `sysinfo(&si)` (19); `const std::size_t unit` (20), `std::size_t i = 0` (29), and `const char c` (31) all carry their initializer at declaration. No late/distant declaration and no uninitialized-then-assigned variable anywhere in the file. 10.2 — no fragile zero-init dependency: the `{}` on `si` (18) value-zeroes the POD but the code does NOT lean on any pre-zeroed field — it gates on `sysinfo(&si) != 0` (19) and only reads `freeram`/`bufferram`/`mem_unit` on success, and `mem_unit` additionally has its own `? : 1u` fallback (20). The `{}` is defensive, not load-bearing. `iequals` reads only initialized locals; the two free functions have no static/global state and rely on no zero-init.
