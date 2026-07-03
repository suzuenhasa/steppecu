# `log.hpp` reference

## 1. Purpose

`src/core/internal/log.hpp` is the single logging facade for the steppe library.
The rule it exists to enforce is: library code never writes diagnostics with a bare
`printf` or `std::cout`. Instead, every diagnostic goes through a `STEPPE_LOG_*`
macro, so the destination, the level, and the eventual async policy can all be
changed in one place — behind the macro — without touching any caller.

Today the facade realizes exactly **one** level: a warning sink, `STEPPE_LOG_WARN`.
That is the only level the code currently needs. The richer levels
(`STEPPE_LOG_INFO`, `STEPPE_LOG_ERROR`, and so on) are reserved names for a future
structured-logging backend; they are not defined in this file yet.

The one warning level serves a specific job: it is where the resource-cleanup code
reports a failure that happens while tearing a resource down. The move-only wrappers
that own GPU resources (`DeviceBuffer`, `Stream`, `Event`, `CublasHandle`) can't
throw from their destructors, but they also must not swallow a nonzero destroy
status in silence. So when a teardown call fails, they route the error string here.
This single sink replaced three separate `fprintf(stderr, ...)` warning macros that
had been copied into the device-buffer, stream, and handle headers and had already
started to drift apart from each other. Consolidating them into one macro is the
whole reason the file exists.

---

## 2. The `STEPPE_LOG_WARN` macro

`STEPPE_LOG_WARN(fmt, ...)` is the one warning sink. It takes a printf-style format
string followed by the matching arguments — the classic `%s`, `%d` style, **not** a
`{}` brace style.

A typical call:

```cpp
STEPPE_LOG_WARN("cudaFree at teardown: %s", cudaGetErrorString(e));
```

When it emits, it prints a single line to standard error, prefixed with a fixed tag
so warnings are easy to spot and grep for:

```
[steppe][warn] cudaFree at teardown: <message>
```

The prefix (`[steppe][warn] `) and the trailing newline are added by the macro; the
caller supplies only the message format and its arguments.

The printf-style format string is deliberate: it is the single seam a future
structured-logging backend would swap out. Because every warning in the library
flows through this one macro, changing what a warning *does* — send it somewhere
else, buffer it, make it async — is a one-line change here, not a sweep across the
codebase.

---

## 3. Debug and release behavior

The macro has two completely different definitions depending on the build:

| Build | What `STEPPE_LOG_WARN` does |
|---|---|
| Debug (no `NDEBUG`) | Pulls in `<cstdio>` and expands to code that prints the `[steppe][warn]` line to standard error. |
| Release (`NDEBUG` defined) | Expands to `((void)0)` — it does nothing at all and compiles away to nothing. |

So a release build is **silent at teardown**: teardown warnings simply do not exist
in release binaries. This matches the behavior of the three older macros this file
replaced — the change was a consolidation, not a behavior change. Anything that was
silent before is still silent, and anything that printed in a debug build still
prints.

The debug arm is written as a `do { ... } while (0)` block wrapping three `fprintf`
calls (the prefix, then the caller's formatted message, then the newline). The
`do/while(0)` wrapper is the standard trick that lets a multi-statement macro be used
like a single statement — for example, as the body of an `if` with no braces —
without surprising anyone.

---

## 4. Invariants callers must respect

Two properties of this macro are easy to get wrong and worth stating plainly.

**The arguments are not always evaluated.** In a release build the macro expands to
nothing, which means its arguments are never evaluated — exactly like a disabled
`assert`. Therefore the format arguments must be free of any side effect the program
actually relies on. Write:

```cpp
STEPPE_LOG_WARN("status: %s", cudaGetErrorString(e));   // fine: pure read
```

but never hide a needed action inside the argument list, because in a release build
that action never runs.

**It is safe to call from a `noexcept` destructor.** The destructors of the RAII
resource wrappers are `noexcept` and must never throw. `std::fprintf` does not throw,
so this macro is safe to call from those destructors — which is precisely the
situation it was built for. That is what lets "fail fast" during normal operation
coexist with "still report the problem" during teardown, without a destructor ever
being able to throw.

---

## 5. Where it lives and what uses it

The file is pure preprocessor plus `<cstdio>` in its debug arm. It contains no CUDA
code, so it compiles cleanly whether or not the compiler is the CUDA compiler. That
is why it lives under `core/internal/` and is pulled in through the shared internal
interface target that the device-side CUDA headers depend on. The GPU-resource
wrappers include it and route their teardown failures into `STEPPE_LOG_WARN`.

It is an internal header. It is not part of the public, installed interface of the
library — callers outside the library never see it or link against it. It is a
build-time-only convenience shared among the library's own translation units.
