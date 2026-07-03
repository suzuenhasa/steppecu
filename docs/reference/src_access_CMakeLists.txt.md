# `src/access/CMakeLists.txt` reference

## 1. Purpose

This file builds `steppe_access`, a tiny static library that holds the two
host-side helper pieces both the command-line tool **and** the Python bindings
need in order to work with an already-computed f2 result on disk. It exists only
to give those two helpers a single shared home so neither consumer has to keep
its own copy.

The two helpers are:

1. **The f2 result reader** — the code that opens a directory of computed f2
   blocks (the `f2.bin` file plus its `pops.txt` companion) and loads it back
   into memory.
2. **The population-name resolver** — the code that maps a human-readable
   population name to its numeric index on the population axis of the f2 tensor.

Before this library existed, both helpers were compiled directly into the
command-line executable. That left no shared, linkable place for anything else
to reuse them. But the Python bindings module needs the exact same two pieces:
it also has to read an f2 directory, and it also has to turn population names
into indices, because the compute layer underneath speaks only in numeric
indices and knows nothing about names. Rather than duplicate the source into a
second build target, both the command-line app and the bindings module link this
one small library. There is exactly one copy of each helper.

The library does no computation of its own. It contains no GPU code and pulls in
nothing from the GPU layers — see section 3 for how that is enforced.

---

## 2. What each source file compiles (and what is deliberately left out)

`steppe_access` is built from two `.cpp` files.

| Source file | What it does |
|---|---|
| `f2_dir_io.cpp` | Reads a directory of pre-computed f2 blocks back into memory: the `f2.bin` binary block file (steppe's on-disk f2-block cache format) together with the `pops.txt` file that lists the population labels. This is the loader for the "compute f2 once, fit many models later" workflow. |
| `pop_resolver.cpp` | Maps a population name to its index on the population axis. The compute layer is index-only, so this is the translation between the names a user types and the positions the math actually uses. |

Both source files physically live in `src/app/`, next to the command-line
command flows that originally grew them, not in `src/access/` alongside this
build file. The build file just reaches over and compiles them from there. Their
headers are therefore still included with an `app/...` prefix everywhere, and
moving the *compilation* into this library did not change any of those include
lines.

### What is deliberately not here

There is a third small helper next to these two in `src/app/` —
`result_emit.cpp` — and it is intentionally **excluded** from this library. That
file serializes results into the command-line tool's text output (CSV, TSV, and
JSON). The Python bindings do not reuse it, because the bindings return native
Python objects instead of formatted text. Since only one of the two consumers
needs it, it stays a private part of the command-line app rather than being
promoted into this shared library.

---

## 3. The CUDA-free rule (what it links, and what it deliberately does not)

`steppe_access` is a plain C++ target with no GPU code in it at all. It declares
no CUDA language support, has no `.cu` sources, and must never include a CUDA
toolkit header. This is not a soft preference — a single leaked CUDA header would
turn this host-only compile into a hard build failure, and a structural build
check greps for exactly that so the rule cannot quietly erode.

Staying CUDA-free is possible because the two helpers only ever reach code that
is itself CUDA-free. It links exactly two dependencies, both of which are
interface/seam targets that expose headers without dragging in any GPU code:

- **`steppe::api`** (public) — the CUDA-free public surface. It provides the
  error type and the f-statistics types the helpers use (`steppe/error.hpp`,
  `steppe/fstats.hpp`).
- **`steppe::core_internal`** (public) — carries the `src/` include root, which
  is what makes the `app/...` and `device/...` include prefixes resolve. In
  particular it lets `f2_dir_io.cpp` reach the on-disk f2-block format
  definition (`device/f2_disk_format.hpp`), which is a header-only, CUDA-free
  description of the file layout.
- **`steppe::warnings`** (private) — the shared compiler-warning settings, not
  propagated to anything that links this library.

Both public dependencies are pure interface/seam targets, so linking them brings
in headers and settings but no CUDA toolkit and no GPU runtime. Just as
important is what this library does **not** link: it links no CUDA and does not
link the command-line argument parser. It stays a small, host-only leaf by
construction.

---

## 4. Include-path convention and language standard

**Include prefixes.** Because `steppe::core_internal` sets the include root to
`src/`, the sources include their headers with directory prefixes that name the
subfolder they live in — `app/...` for the two helpers' own headers and
`device/...` for the shared f2-block format header. This is the same convention
the other libraries use, which keeps `#include` lines uniform across the
codebase and is exactly why relocating the compilation into this library left
those lines untouched.

**Language standard.** The library is compiled as C++20, and that requirement is
public, so any target that links `steppe_access` also compiles as at least
C++20.
