# `nvtx.hpp` reference

## 1. Purpose

`src/core/internal/nvtx.hpp` is the one place steppe defines its optional
profiler-timeline annotation macro. It exposes a single name to the rest of the
codebase:

- `STEPPE_NVTX_RANGE(name)` — mark a named span on the NVIDIA profiler timeline.

NVTX (the NVIDIA Tools Extension) is a lightweight way to label regions of a
program so that when you record a run in a profiler, the timeline shows named
bars — "decode", "f2 GEMM batch", "jackknife", "qpAdm fit" — instead of an
undifferentiated wall of kernels. A marker is purely descriptive: it annotates
when a phase starts and ends on the host side and does no computation of its own.

The macro is **off in every shipping build**. Only a dedicated profiling build
turns it on. When it is off, the header adds nothing to the compiled program at
all — not even a single instruction. That "zero-overhead off" promise is the
central design point of the file, and it is structural rather than a matter of the
optimizer being clever (see section 3).

---

## 2. The `STEPPE_NVTX_RANGE(name)` macro — the contract

`STEPPE_NVTX_RANGE(name)` opens a named annotation range that automatically closes
at the end of the enclosing block. It follows the same lifetime pattern as a
`std::lock_guard`: the range opens where you write the macro and closes on its own
when execution leaves the surrounding scope. There is no matching "end" macro to
call, and no way to forget to close it.

- `name` is a string literal — for example `STEPPE_NVTX_RANGE("decode");`.
- Placing the macro at the top of a function or block marks that whole block.
- When the block exits (including via an early return or an exception), the range
  closes.

That is the entire public surface of the header: one macro, string in, a
self-closing timeline span out.

---

## 3. The two build states

The macro has exactly two forms, chosen at compile time by whether the
`STEPPE_NVTX` preprocessor symbol is defined.

### Off — the shipping default

When `STEPPE_NVTX` is **not** defined (the default for every normal build), the
macro expands to nothing and the header includes no NVTX or CUDA header at all.
The result is that the compiled object code is byte-for-byte identical to a build
that never included this header, and inspecting the compiled symbols (`nm`) shows
no NVTX symbols whatsoever. Nothing is emitted and then optimized away — nothing
is emitted in the first place. This is why the "zero-overhead off" description is
structural, not aspirational: it holds regardless of optimization level.

### On — the profiling build only

`STEPPE_NVTX` is defined only by the dedicated profiling build, which passes
`-DSTEPPE_NVTX=ON`. That definition is applied **privately** to the GPU code
module (`steppe_device`), so it never leaks into other parts of the build.

In this state the header pulls in the header-only NVTX version 3 C++ API and the
macro expands to an `nvtx3::scoped_range` object. Two things make this cheap to
enable:

- NVTX v3 is header-only and dependency-free. It ships inside the CUDA toolkit's
  `nvtx3/` include directory, so turning the feature on needs no `find_package`
  and no extra library to link — including the header is enough.
- The `nvtx3::scoped_range` type opens the range when it is constructed and closes
  it when it is destroyed, which is what gives the macro its automatic
  end-of-scope behavior.

---

## 4. Parity safety — why a marker can never change a result

steppe guarantees reproducible numeric results by running its statistics on a
single stream in a fixed order. An NVTX marker is a passive host-side timeline
annotation only. It launches no device work, touches no stream, performs no
synchronization, and runs no kernel or math. Because it never interacts with the
computation, enabling the profiling build cannot alter any number steppe reports.
The reproducibility contract is untouched whether the markers are compiled in or
compiled out.

---

## 5. Unique per-range variable names (the token-paste detail)

When the macro is on, each expansion creates a local `nvtx3::scoped_range` object,
and each of those objects needs a distinct variable name so that two markers in
the same scope do not collide. The header builds that name from the current source
line number (`__LINE__`), producing names like `steppe_nvtx_range_48`.

Doing this correctly needs two layers of concatenation:

- `STEPPE_NVTX_CONCAT_IMPL(a, b)` performs the actual token paste with `##`.
- `STEPPE_NVTX_CONCAT(a, b)` calls the first one.

The extra layer is required because of a preprocessor rule: if `__LINE__` were
pasted directly with `##`, the paste would suppress macro expansion and you would
get the literal text `__LINE__` glued on instead of the line number. Routing it
through the outer macro forces `__LINE__` to expand to its numeric value *before*
the paste happens, so the name ends up line-derived and unique. This is a standard
preprocessor idiom; the two-level structure exists solely to defeat that
paste-suppression rule.

---

## 6. Where to place markers

Markers belong only at **coarse phase boundaries** — the handful of major stages
such as decode, the batched f2 matrix-multiply, the jackknife, and the qpAdm fit.
Keep them out of per-block and per-quartet inner loops.

The reason is about measurement fidelity, not correctness. Even though a marker
does no device work, placing one inside a hot inner loop would add host-side
annotation churn on every iteration and could shift the relative timing of the
kernels you are trying to observe. Restricting markers to the outer phase
boundaries keeps a profiling build's kernel timeline representative of a normal
run.

---

## 7. Placement in the build and visibility

The header lives in `core/internal/` and is consumed through the
`steppe::core_internal` interface target. It is designed to compile in two very
different contexts:

- It can be compiled by a plain C++ compiler with no CUDA at all. The off path
  includes no CUDA or NVTX header, and the on path includes only the host-side C++
  NVTX wrapper, which needs no CUDA runtime.
- It can also be compiled by the CUDA compiler.

Being compilable both ways is why it sits in the shared internal core rather than
alongside the GPU code. It is deliberately **not** part of steppe's public
interface: the markers are used only inside the GPU code's own translation units,
so no library user or public header ever sees this macro.
