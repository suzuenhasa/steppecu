I read through this carefully. This is **not slop** — it's clearly written by someone who understands the project conventions and the CLI command pattern. A senior developer would find it competent and mostly consistent, but there are a few polish issues and one real convention mismatch that would get flagged.

## What's genuinely good

- **Clear structural decomposition.** The file is organized into six numbered sections (read f2 dir, resolve names, enumerate subsets, multi-GPU warning, build/upload/run, emit results). This makes the command's control flow easy to follow and mirrors the documented `cli-bindings.md` flow.
- **Correct lexicographic subset enumeration.** The combination walker in `enumerate_pool_subsets` (lines 54–76) is the standard combinatorial algorithm and the comments explicitly tie it to `golden_rot_generate.R`'s `combn(pool,2)` + `combn(pool,3)` ordering. The `model_index` counter is dense and deterministic — exactly what you want for reproducible golden-file tests.
- **Appropriate use of `std::span`.** Passing `models` and `results` as `std::span<const QpAdmModel>` / `std::span<const QpAdmResult>` (lines 185, 205, 214) is good C++20 hygiene and avoids imposing ownership semantics on the engine.
- **Consistent error-handling philosophy.** The code distinguishes config errors, IO errors, runtime/device faults, and per-model domain outcomes. The comment at lines 218–219 explicitly states the "RECORD-AND-CONTINUE" contract — a useful invariant for maintainers.
- **RAII device resource handling.** `device::Resources`, `device::DeviceF2Blocks`, and `std::ofstream` are stack-owned, and the device path is wrapped in a single `try/catch(const std::exception&)` (lines 173–191). No raw pointers or manual cleanup.
- **Reuses existing abstractions.** It doesn't reinvent pop resolution, f2-dir loading, output formatting, or the qpadm search engine — it composes them.

## What a senior developer would flag

**Mix of C-style `std::fprintf(stderr, ...)` with C++ `std::cout` output:**

```cpp
std::fprintf(stderr, "steppe qpadm-rotate: --f2-dir is required\n");          // line 86
// ...
emit_rotation_table(std::cout, fmt, ...);                                      // line 205
```

This is the most visible convention mismatch. The file includes both `<cstdio>` and `<iostream>`, and uses `std::fprintf` for diagnostics but `std::cout` for formatted results. In a job-application showcase, this looks like the project never settled on an output strategy. Pick one: either stream everything through `std::ostream&` (and inject `std::cerr` where needed) or document why this split is intentional.

**Cramped one-line error returns (lines 117–121):**

```cpp
if (!t.ok) { std::fprintf(stderr, "steppe qpadm-rotate: %s\n", t.error.c_str()); return cfg::kExitInvalidConfig; }
if (!rt.ok) { std::fprintf(stderr, "steppe qpadm-rotate: %s\n", rt.error.c_str()); return cfg::kExitInvalidConfig; }
if (!pl.ok) { std::fprintf(stderr, "steppe qpadm-rotate: %s\n", pl.error.c_str()); return cfg::kExitInvalidConfig; }
```

These are much harder to scan than the preceding multi-line validations (lines 103–114). They also introduce inconsistent brace style. A senior reviewer would ask to break these across lines like the others.

**Magic sentinel `-1` for "whole pool":**

```cpp
int hi = (config.max_sources() == -1) ? pool_n : config.max_sources();  // line 133
```

`-1` apparently means "use the entire pool," but the constant is not named in this file. If `RunConfig` exposes a named constant (e.g., `kMaxSourcesUnlimited`), use it here; otherwise the next reader has to guess the convention.

**The `right_n` convention is fragile (lines 201–202):**

```cpp
// nr convention: right[0] == R0, so right_n == right.size()-1 (== metadata.nr).
const int right_n = static_cast<int>(right_idx.size()) - 1;
```

The comment admits this is a convention, and the calculation can underflow if `right_idx` is ever empty. The code validates `config.right().empty()` earlier, but `right_n` being `size - 1` is a subtle invariant that belongs closer to the `QpAdmModel` / `QpAdmResult` types, not inlined in every command.

**Output format is validated *after* building the model list (lines 142–157):**

If `config.format()` is invalid, the code has already enumerated all combinations and validated names. Moving the format check earlier would fail faster and avoid wasted work. Minor, but a senior reviewer notices ordering like this.

**TODO embedded in a user-facing warning (lines 162–168):**

```cpp
std::fprintf(stderr,
             "steppe qpadm-rotate: WARNING: %zu devices requested; the rotation "
             "runs single-GPU-preferred (TODO(multigpu-host-bounce): the rotation "
             "is host-bounce-capped on no-P2P consumer cards). Use --device 0.\n",
             config.device().devices.size());
```

User-facing strings are the wrong place for internal TODO markers. Move the TODO to a comment or issue tracker; keep the warning concise.

**Every `QpAdmModel` copies `right_idx` and `target_idx` (lines 60–65):**

```cpp
m.target = target_idx;
m.right = right_idx;
```

This is fine when `right` is small, but for large rotation pools the repeated allocation and copy of `right` into every model is wasteful. If `QpAdmResult` supports it, a shared reference or a single right-vector owned outside the model list would be cleaner. As written it's a small scalability blemish.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted code with stale comments, missing error handling, or algorithms that only work by accident. This file has none of that. The comments are dense but accurate, the math is correct, and the error handling follows a documented contract.

## What it actually looks like

This looks like **solid, workmanlike CLI glue code written by someone who knows the project conventions and the qpadm command pattern.** It isn't trying to be clever; it's trying to be correct, reproducible, and consistent with the existing `cmd_qpadm.cpp` path. The combination enumeration is the only new algorithmic logic, and it's implemented carefully.

A senior C++ reviewer would say: "Competent, but polish the output style and clean up the one-liners before I'd call it production-clean." A CUDA reviewer wouldn't have much to say here because the GPU is reached through the same seams as the existing qpadm command — which is exactly the right layering for this file.

The main risk in a showcase context is that **the `fprintf` / `cout` split and the cramped error lines make the code look less finished than it actually is.** These are easy fixes and would raise the perceived quality significantly.

**Verdict:** B+ — respectable production CLI code, one polish pass away from A-.
