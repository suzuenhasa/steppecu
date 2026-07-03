# `src/app/CMakeLists.txt` reference

## 1. Purpose

`src/app/CMakeLists.txt` is the build recipe for the `steppe` command-line
program — the executable a user actually runs from a shell. It does four things:

1. Obtains the command-line parsing library the app depends on (CLI11), preferring
   an already-installed copy and fetching a pinned version only if none is found.
2. Declares the executable target and lists every source file that makes up the CLI,
   including one file per subcommand.
3. Links the executable against the rest of steppe's libraries in a specific
   arrangement that is required for the GPU kernels to link correctly.
4. Bakes the project version string into the binary so `steppe --version` reports it.

The single most important property of this build file is that the CLI executable is
a **plain C++ target, not a CUDA target**. It compiles as ordinary host C++20 with no
GPU compiler involved, and it must never include a CUDA toolkit header. Section 2
explains why that constraint exists and how it enforces itself.

---

## 2. The CUDA-free constraint

The CLI is built as an ordinary C++20 program. This build file deliberately does
**not** enable CUDA as a language for the target, ships **no** `.cu` source files,
and does **not** turn on CUDA separable compilation. The target must not pull in any
CUDA toolkit header, directly or transitively.

### How the CLI reaches the GPU without CUDA headers

The app still needs to run GPU work — it just does so at arm's length, through a
small set of entry points that are written to expose no CUDA types in their
signatures. These CUDA-free seams are the only doorways the CLI uses to reach the GPU
layer:

- building the GPU resources a run needs,
- launching the qpAdm fit,
- uploading the precomputed f2 blocks to the device,
- reading and writing the on-disk f2 block format.

Because these entry points speak only in plain C++ types, the CLI can call them while
staying entirely free of CUDA code.

### Why the constraint enforces itself

This arrangement needs no separate linter or policy check to stay honest. If a CUDA
runtime header ever leaked into one of the CLI's includes, this host-only compile
would simply fail to build — an ordinary C++ compiler cannot process a CUDA toolkit
header. So the layering rule ("the app never touches CUDA directly") is guaranteed by
the compile itself: a violation is a hard build error, not a warning someone has to
notice.

---

## 3. The CLI11 dependency

The CLI uses **CLI11 version 2.4.x** (pinned to 2.4.2) to parse command-line
arguments. This build file obtains it the same no-network-first way the rest of the
repository obtains its other third-party dependencies:

1. **Try `find_package` first.** If a compatible CLI11 (2.4 or newer, in CONFIG mode)
   is already installed on the machine, use it and log which version was found. This
   path needs no network access.
2. **Fall back to fetching.** If no installed copy is found, pull in the project's
   package-fetching helper and download CLI11. The fetch is logged so it is obvious
   from the build output that a download happened.

### Why the fetch is pinned to a commit, not a tag

The fetched version is pinned two ways at once. It names the version (2.4.2) so the
fetcher's version bookkeeping lines up with the `find_package` requirement above, and
it also pins the exact immutable commit that the `v2.4.2` release tag points at. The
commit pin matters because a git tag is movable — someone could re-point `v2.4.2` at
different code later — whereas a commit hash always refers to exactly the same source.
Pinning the commit makes the build reproducible regardless of what happens to the tag
upstream.

### Why this lives here and not in a top-level file

All of the CLI11 handling and the package-fetching machinery are confined to this one
subdirectory on purpose. Keeping them here makes CLI11 a **private** dependency of the
app subtree: the core numerical library and the GPU/device library never see it and
never build against it. That preserves the layering rule that the lower libraries do
not depend on anything the command-line front end happens to use.

---

## 4. The executable and its sources

The build declares one executable target (internally named `steppe_app`) and then
renames its output file to `steppe`, so what a user installs and runs is called
`steppe`, not `steppe_app`. It is also given an alias so other parts of the build can
refer to it by a qualified name.

### The source files

Each subcommand of the CLI is its own source file, alongside a few shared pieces
(argument parsing, result formatting, and the program entry point):

| Source file | What it provides |
|---|---|
| `main.cpp` | The program entry point. |
| `cli_parse.cpp` | Command-line argument parsing. |
| `cmd_qpadm.cpp` | The qpAdm GPU fit command. |
| `cmd_qpgraph.cpp` | Single-graph qpGraph fit — takes an f2 directory plus a graph edge-list and runs the fit. |
| `cmd_qpwave.cpp` | The qpWave rank-sweep command (no target population; the first left population acts as the reference). |
| `cmd_f4.cpp` | The standalone f4 statistic — estimate, standard error, z-score, and p-value per quartet. |
| `cmd_f3.cpp` | The standalone f3 statistic — the same four outputs per triple. |
| `cmd_f4ratio.cpp` | The standalone f4-ratio — an alpha value plus its standard error and z-score, per five-population tuple. |
| `cmd_fstat_sweep.cpp` | The GPU-only f4/f3 sweep over every combination of populations, computed on the device with a significance (absolute-z) and top-K filter applied. |
| `cmd_qpdstat.cpp` | The D-statistic / f4 command driven from an f2 directory. |
| `cmd_qpfstats.cpp` | The genotype-path joint f2 smoother — takes a genotype prefix and a population list and produces a smoothed f2 directory. |
| `cmd_dates.cpp` | Admixture dating — takes a genotype prefix, a target, and two left populations and returns a date plus its standard error. |
| `cmd_rotate.cpp` | The qpAdm rotation — enumerates subsets of a population pool and fits them in batches. |
| `cmd_extract_f2.cpp` | The genotype-to-f2-blocks precompute step; a thin wrapper over the shared extract library. |
| `result_emit.cpp` | Formats a qpAdm result as tidy CSV, TSV, or JSON. |

### Two source files that live here but are compiled elsewhere

The f2-directory reader and the population-name-to-index resolver are **not** in the
list above, even though their `.cpp` files physically sit in this directory next to
the commands that first needed them. They were moved into a shared library (see
section 5) so that the Python bindings can reuse the exact same code instead of
duplicating it. That shared library compiles those two files; the CLI executable links
the shared library rather than compiling the two files a second time.

---

## 5. Library links

The executable links, privately, against steppe's core numerical library, its I/O
library, its public API library, its device (GPU) library, two shared helper
libraries, the CLI11 parser, and the project's shared warning-flag target.

### The device-linking arrangement

The most subtle part of the link list is that the CLI executable links the **device**
library even though the CLI itself runs no GPU kernels of its own and device-links
nothing. There are two reasons it appears here:

1. **To resolve the CUDA-free factory and resource-building symbols.** The seams from
   section 2 are implemented in the device library, so the executable must link it to
   find those symbols — the same reason the core library links it.
2. **To let the GPU kernels finish their device-side linking.** The core library is
   configured to *not* resolve device symbols itself. Because of that, the GPU
   kernels' device-side link step only completes in a final program that links **both**
   the core library and the device library together. The CLI executable is exactly
   such a program, so linking both here is what lets the device library's
   relocatable-device-code kernels get device-linked. The executable itself remains a
   pure host target; it contributes no device code of its own.

### The two shared helper libraries

Beyond the core/io/api/device libraries, the executable links two shared libraries so
that the CLI and the Python bindings run the identical code:

- **The access library** provides the f2-directory reader and the
  population-name-to-index resolver (the two files described at the end of section 4).
- **The extract library** provides the genotype-to-f2 extraction chain and the
  f2-directory writer.

Keeping both as shared libraries, rather than as CLI-only source files, is what lets
the bindings reuse them without a second copy.

---

## 6. Compile features and the version string

The target is compiled with the C++20 standard.

The version string that `steppe --version` prints is baked in at compile time from the
project version declared once at the top level of the build. It is defined as a
compile-time macro so that there is a **single home** for the version number: the app
source contains no second hardcoded version literal that could drift out of sync with
the real project version.
