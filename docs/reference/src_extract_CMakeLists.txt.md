# `src/extract/CMakeLists.txt` reference

## 1. Purpose

This build file defines `steppe_extract`, a static library that is the single
shared home for two things:

1. **The genotype-to-f2 extract chain** — the full pipeline that turns a raw
   genotype dataset (a prefix pointing at genotype/individual/SNP files) into an
   in-memory set of f2 statistics. The chain is: decode the genotypes, apply the
   SNP filters, assign SNPs into jackknife blocks, run the multi-GPU tiered f2
   computation, and copy the result back to the host.
2. **The f2-blocks directory writer** — the code that serializes that result to
   disk as an f2-blocks cache directory (the on-disk format whose files carry the
   `STPF2BK1` identifier), including the real value-pair data and its sidecar
   metadata files.

Both pieces are compiled once here and linked by two different consumers: the
command-line tool and the Python bindings. Neither consumer owns the code; they
both link this one library.

---

## 2. Why this library exists

The extract chain originally lived *only* inside the command-line executable — it
was the body of the CLI's `extract_f2` command, with no separate, linkable home.
That became a problem because the Python bindings need the exact same chain: given
a genotype prefix, build an f2-blocks directory. There was no way for the bindings
to reuse the CLI's copy.

The situation is made worse by how the project is built. The command-line sources
are compiled *only* when the CLI is switched on in the build. A Python-wheel build
turns the CLI off and the Python bindings on — so in that configuration the extract
chain, living inside the CLI, would not be compiled at all, and the bindings would
have nothing to call.

`steppe_extract` solves both problems at once. It is the one copy of the chain,
independent of whether the CLI is being built. The CLI command becomes a thin
wrapper that calls into this library, and the Python binding calls the same entry
point. One implementation, two callers — no duplicated logic and no
build-configuration hole.

A related detail: the two `.cpp` source files this library compiles physically
live in the CLI source directory (`src/app/`), right next to the command flows that
originally grew them, and their internal include paths still point at that
directory. This build file simply reaches over and compiles those files here. The
CLI target then *links* this library instead of compiling the same files a second
time, so the code is never built twice.

---

## 3. What it builds

The library is a `STATIC` archive named `steppe_extract`, built from two source
files:

| Source file | What it provides |
|---|---|
| `../app/extract_f2_core.cpp` | The genotype-to-f2 chain — the implementation of the public extract API (its entry point is the run-extract function the CLI and bindings both call). |
| `../app/f2_dir_writer.cpp` | The f2-blocks directory writer — writes the real value-pair data plus the metadata sidecar files. |

It also defines the namespaced alias `steppe::extract`, so other build targets link
against `steppe::extract` rather than the bare target name.

The C++ standard is pinned to C++20, and that requirement is public — anything that
links this library also compiles as C++20.

---

## 4. Link dependencies and the CUDA-free rule

`steppe_extract` links the same set of internal libraries the command-line
executable uses, split into one public and several private dependencies:

| Dependency | Visibility | What the chain gets from it |
|---|---|---|
| `steppe::api` | PUBLIC | The public headers — the configuration, extract, and f-statistics interfaces. Public because they appear in this library's own interface. |
| `steppe::core` | PRIVATE | The multi-GPU tiered f2 computation. |
| `steppe::io` | PRIVATE | The genotype/individual/SNP readers, ploidy detection, and the SNP filter. |
| `steppe::device` | PRIVATE | The resource builder and device handles (the factory that stands up GPU resources, the resources object, the f2-blocks output type, and the decode backend handle). |
| `steppe::warnings` | PRIVATE | The shared compiler-warning settings. |

### It reaches the GPU, but it is still a pure host target

There is a sibling reader library, `steppe::access`, that is genuinely CUDA-free
and links only the API and an internal-core target. `steppe_extract` is different:
its chain *does* reach the GPU. But it reaches the GPU only through CUDA-free seams
— it links `steppe::device` purely to resolve the plain-C++ factory and handle
symbols (the same route the CLI and the bindings use). It does not itself perform
any device linking, because the core library it depends on turns device-symbol
resolution off (see section 6). The net effect is that `steppe_extract` is a pure
host archive.

Because of that, this remains an ordinary C++20 host target. It declares no CUDA
language, compiles no `.cu` files, and pulls in no CUDA toolkit header. This is a
hard structural rule, not a style preference: if a CUDA header were to leak into one
of its source files, this host-only compile would fail outright. A separate build
check greps for exactly that leak, so the boundary is enforced mechanically.

---

## 5. The stamped version string

The library is compiled with a private definition, `STEPPE_VERSION`, set to the
top-level project version. The directory writer stamps this string into the
metadata file it writes alongside the f2-blocks data.

The point of routing it through the compile definition is single-homing: the
version lives in exactly one place — the top-level project's declared version — and
flows from there into the writer. Previously the CLI command passed this same
version string in by hand; now it comes from the one project-level source of truth,
so the stamped version can never drift out of sync with the actual build.

---

## 6. The device-symbol resolution invariant

This library sets one target property, and it is the most load-bearing line in the
file:

```
set_target_properties(steppe_extract PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
```

### What it does and why it is required

`steppe_extract` links `steppe::device` privately. That device library is built
with separable CUDA compilation, and its own core dependency deliberately turns
*device-symbol resolution off*. This library links the same edge, so it must mirror
that same "off" setting.

The critical fact is that this property does **not** propagate automatically from
the core library to this sibling target. Each target decides it independently. So if
this line were missing, the default behavior would kick in: CMake would generate a
device-link step for `steppe_extract` and archive its output object into the static
library. That object registers the GPU code (the fatbin) from the CUDA backend.

The failure then surfaces at the very end of the build. The final executable that
links this library *also* device-links the device library, so it registers that same
GPU code a second time. The two registrations collide and the host link fails with
an "undefined reference" error naming the CUDA backend's fatbin wrapper symbol.

With the property set to off, no device-link step is generated for this library, it
stays a pure host archive with no embedded GPU-code registration, and the consuming
executable device-links the device library exactly once — the same clean path the
core library and the CUDA test targets already rely on.

This is a correctness invariant, not a tuning knob: the setting must stay off, and
it must stay aligned with the core library's own setting.
