# `host_device.hpp` reference

## 1. Purpose

`src/core/internal/host_device.hpp` is the single home for three small
preprocessor helpers that the rest of the library shares:

1. **`STEPPE_HD`** — a qualifier that lets one function definition compile both
   as ordinary CPU code and as GPU code.
2. **`STEPPE_DEBUG_ONLY(...)`** — a wrapper for statements that should run only
   in debug builds and vanish in release builds.
3. **`STEPPE_ASSERT(cond, msg)`** — a precondition check that runs only in debug
   builds.

All three were previously open-coded — retyped by hand — in several different
places. Collecting them here means there is exactly one definition of each, so
the copies can never quietly drift apart.

The header contains no CUDA code. It uses only the C++ preprocessor and the
standard `<cassert>`, so it compiles fine whether or not the CUDA compiler is in
use. That is what lets both the CPU-only parts of the library and the GPU
kernels include it from one shared definition. It is an internal header — not
part of the library's published interface.

---

## 2. STEPPE_HD — the host/device qualifier

`STEPPE_HD` marks a function so that the same source can be built for both the
CPU and the GPU:

- When the file is compiled by the CUDA compiler, `STEPPE_HD` expands to
  `__host__ __device__`, which tells the compiler to generate the function for
  **both** the CPU and the GPU.
- When the file is compiled by a plain host compiler, `STEPPE_HD` expands to
  nothing, so the same function is an ordinary CPU function that can be
  unit-tested on the CPU.

This is what lets a single shared per-element primitive — for example the code
that estimates an f2 value or decodes an allele frequency — be written once and
compiled into both the CPU library and the GPU kernels from one definition.

### Why the `#ifndef` guard

The macro is defined inside an `#ifndef STEPPE_HD` guard. If a translation unit
that includes this header already defined `STEPPE_HD` (for instance because it
also pulled in another header that set it), including this file is a harmless
no-op rather than a redefinition warning or error.

That guard also fixes a real latent bug. Before this file existed, `STEPPE_HD`
was `#define`d byte-for-byte identically in two different headers, with neither
one undefining it afterward. Two separate definitions of the same macro are only
legal as long as they stay exactly the same text — the moment one copy is edited
and the two drift apart, the program becomes ill-formed. Moving the definition
here, behind the guard, removes that hazard: there is now one definition, and
re-seeing it is safe.

---

## 3. STEPPE_DEBUG_ONLY — debug-only statements

`STEPPE_DEBUG_ONLY(...)` expands its argument only in debug builds. In release
builds (where `NDEBUG` is defined) it expands to nothing.

It follows the same contract as the standard `assert`: in a release build the
body is removed entirely and — exactly like `assert` — its argument is **not
evaluated**. Because of that, the argument must not contain side effects the
program actually needs; anything essential must live outside the macro.

The call site always supplies the terminating semicolon, so the expansion is
always a complete, well-formed statement either way:

- In a release build it becomes the no-op `((void)0)`.
- In a debug build it becomes the argument itself.

Use it to wrap any statement that should run only in debug builds — for example
an extra GPU synchronize whose only job is to pin down which kernel faulted —
instead of writing `#if defined(NDEBUG)` by hand at each such site.

---

## 4. STEPPE_ASSERT — debug-only precondition check

`STEPPE_ASSERT(cond, msg)` checks a precondition, but only in debug builds.

- **In a debug build** it is the standard `assert((cond) && (msg))`. If the
  condition is false the program prints the message and the source location and
  aborts — which is what you want under a debugger or the CUDA sanitizer.
- **In a release build** (`NDEBUG`) the check is compiled out, so it costs
  nothing at runtime.

### The release-build subtlety

When the check is compiled out, `cond` and `msg` do not simply disappear. They
still appear in the source, but only inside an unevaluated `sizeof`
(`(void)sizeof(cond), (void)sizeof(msg), (void)0`). This has two deliberate
consequences:

1. `cond` and `msg` are **still not evaluated** at runtime — `sizeof` only asks
   about their type, never runs them. So, as with `STEPPE_DEBUG_ONLY`, the
   condition and message must be free of needed side effects.
2. They are still marked as **"used"** by the compiler.

That second point is the reason for the trick. A variable that exists only to be
checked in an assertion would, once the assertion is compiled away, look unused —
and the compiler would emit an "unused variable" warning. Because the
release build is configured to treat warnings as errors, that warning would
break the build. Referencing the names inside `sizeof` keeps them counted as
used, so a variable that is touched only by a `STEPPE_ASSERT` does not trip that
warning.

### When to use it

Use `STEPPE_ASSERT` for preconditions that are documented but not otherwise
enforced, on hot paths where a throwing runtime check would be too expensive but
a silently wrong answer would be unacceptable. It is the cheap, debug-only way to
enforce "this must be true here" without paying for it in release.
