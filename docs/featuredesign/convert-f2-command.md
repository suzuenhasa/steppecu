# Feature design — an easy CLI command for the f2 `.rds` converter (`convert-f2`)

**Status: scoped, ready to build.** The AT2 ⇄ steppe `.rds` converter
([`export_f2_rds`/`import_f2_rds`](rds-converter.md), merged in `a536fd2`) currently exists
**only as a Python API**. This scopes the "easy way to run it as a command" — so a user can
convert without writing a Python snippet.

Grounded against the real code: the CLI is **CLI11 2.4.2**, all subcommands are registered in
`src/app/cli_parse.cpp::run_cli`; `steppe_app` is a **host-only (no-CUDA)** target that already
links the f2-dir reader (`app::read_f2_dir`, `src/app/f2_dir_io.hpp:64`) and writer
(`app::write_f2_dir`, `src/app/f2_dir_writer.hpp:108`). **zlib is NOT currently a dependency**
(no `find_package(ZLIB)`, no `<zlib.h>` anywhere) — it would be new.

---

## The fork: which "command"?

The converter is **pure format translation, no GPU** — so the two homes are a *Python* command
(ships in the wheel) or a *native* subcommand of the `steppe` binary. They serve different users:

| | **A. Python console-script `steppe-rds`** | **B. Native `steppe convert-f2`** |
|---|---|---|
| Command | `steppe-rds export f2_dir out_rds` | `steppe convert-f2 --export --f2-dir … --out-dir …` |
| Ships in | the **wheel** (`pip install steppe`) | the **binary** (the curl-able fatbinary CLI) |
| Needs Python? | **yes** | **no** — self-contained with the binary |
| Directions | **both** (export=stdlib, import=`pyreadr`) — already built | **export** straightforward; **import** is the hard half (needs a C++ RDS *reader*) |
| New deps | none (import reuses the `[rds]` extra) | **zlib** (`find_package(ZLIB)`) + a C++ port of the RDS serializer |
| Effort | **S** (~40 LOC + 2 pyproject lines, no rebuild) | **M–L** (see §B) |
| Reuses | the existing `export_f2_rds`/`import_f2_rds` verbatim | `read_f2_dir` / `write_f2_dir` seams; re-implements the serializer |

**Recommendation: ship A now, offer B as a follow-on.** A is ~1 hour, delivers *both*
directions today with zero rebuild, and gives a real shell command. B is the "proper"
binary-native command (matches `steppe qpadm …`), but it's much larger and its import half
needs a hand-rolled RDS reader in C++. Since the verify-in-AT2 user almost always already has
R+Python, A covers the stated goal; B is worth it only if a **Python-free** `steppe convert-f2`
is explicitly wanted for binary-only boxes.

> Note the command can't be named plain `steppe` from the wheel — that would collide with the
> binary on `PATH`. Hence `steppe-rds` (or `steppe-convert`).

---

## A. Python console-script `steppe-rds`  — **recommended, S**

The converter logic already exists and is golden-gated. This is just a CLI front door.

### Mechanism
Add an argparse `main()` that dispatches to the existing facade, and register it as a
console-script entry point so `pip install steppe` puts `steppe-rds` on `PATH`.

```bash
steppe-rds export  f2_dir  out_rds_dir            # steppe STPF2BK1 -> AT2 read_f2() dir
steppe-rds import  at2_rds_dir  out_f2_dir        # AT2 .rds dir   -> steppe STPF2BK1
#   export: no third-party dep.   import: needs `pip install steppe[rds]` (pyreadr).
```

### Files (~S, ~50 LOC, no CUDA rebuild)
| File | LOC | change |
|---|---|---|
| `bindings/steppe/_rds.py` | ~40 | **add** `main(argv=None)` — argparse with `export`/`import` subparsers (positional `src`, `dst`; `--counts ones\|vpair` for export; `--type f2` for import); calls `export_f2_rds`/`import_f2_rds`; maps the lazy-import `pip install steppe[rds]` error to a clean stderr message + nonzero exit. |
| `pyproject.toml` | ~2 | add `[project.scripts]` → `steppe-rds = "steppe._rds:main"`. |
| `tests/python/test_py_convert_rds.py` | ~25 | **add** a CLI smoke test: `subprocess`/`runpy` invoke `steppe-rds export <fixture> <tmp>`, assert the `.rds` dir is produced + re-reads bit-exact (reuses the existing round-trip asserts); arg-error paths return nonzero. |
| `docs/commands.md`, `docs/cheatsheet.md` | ~8 | document the `steppe-rds export/import` one-liners next to the Python-API section. |

### Gate
Reuses the merged round-trip gate (the CLI only wraps the API): the smoke test proves the entry
point dispatches + produces a valid dir; the existing 11 converter tests already prove
bit-exactness + the AT2/qpAdm round-trip. **No box/R needed for A.**

---

## B. Native `steppe convert-f2` subcommand  — follow-on, M–L

A Python-free command inside the binary. Feasible because `steppe_app` is host-only and already
links both f2-dir seams — but the RDS (de)serializer must be re-implemented in C++.

### What it takes
1. **`enum class Command::ConvertF2`** in `src/core/config/cli_args.hpp` (+ fields: a mode flag,
   an input dir, `--out-dir`).
2. **`src/app/cmd_convert_f2.{hpp,cpp}`** — `[[nodiscard]] int run_convert_f2_command(const config::RunConfig&)`,
   shaped like `cmd_f4.cpp`. Export: `read_f2_dir(f2_dir)` → walk pairs → emit `<p1>/<p2>_f2.rds`
   + `block_lengths_f2.rds` + a subdir per pop. Import: parse the AT2 `.rds` dir → `F2BlockTensor`
   → `write_f2_dir(...)` (nonzero-vpair sentinel, per §4 of [rds-converter.md](rds-converter.md)).
3. **Registration block + `#include`** in `cli_parse.cpp::run_cli` (reuse `add_f2_dir_flag`,
   `add_common_flags`; add `--export/--import` + `--out-dir`).
4. **`src/app/CMakeLists.txt`** — add `cmd_convert_f2.cpp`; add `find_package(ZLIB REQUIRED)` +
   `target_link_libraries(steppe_app PRIVATE ZLIB::ZLIB)`.
5. **A C++ RDS/XDR (de)serializer** — the load-bearing new code:
   - **Writer** (export): port `_rds.py`'s `_write_rds_matrix` / `_write_rds_int_vector` —
     big-endian XDR REALSXP `[n·2]` + the `dim`/`dimnames` ATTRIB pairlist, then `deflate` via
     zlib with a fixed gzip header (`mtime=0`) for byte-determinism. **~200 LOC, well-specified**
     (the Python version is byte-identical to R `saveRDS(version=2)`; port it 1:1).
   - **Reader** (import): parse gzip'd XDR RDS back to a `[n,2]` matrix. **This is the harder
     half** — no `pyreadr` equivalent in C++. Either hand-roll a minimal RDS reader (only the one
     shape AT2 emits) or vendor `librdata`. **Recommendation: ship `convert-f2 --export` first**
     (the verify-in-AT2 path, the whole point), and either defer import to Python (A) or add the
     C++ reader in a later pass.

### Files (~M–L)
| File | LOC | change |
|---|---|---|
| `src/app/rds_serializer.{hpp,cpp}` | ~350 | **new** — the C++ XDR/gzip RDS writer (+ optional reader) |
| `src/app/cmd_convert_f2.{hpp,cpp}` | ~180 | **new** — the subcommand handler |
| `src/core/config/cli_args.hpp` | ~6 | `Command::ConvertF2` + fields |
| `src/app/cli_parse.cpp` | ~25 | registration block + `#include` |
| `src/app/CMakeLists.txt` | ~4 | `cmd_convert_f2.cpp` + ZLIB |
| `tests/…` | ~60 | export-byte-exact vs the Python reference + the R end-to-end gate |

### Gate
Same "verified in ADMIXTOOLS 2" proof as the merged converter: `steppe convert-f2 --export` the
9-pop cache → `admixtools::read_f2` + `qpadm` matches steppe-native to ~1e-11 (the numbers are
already captured in the merge commit). Plus a byte-exactness check against the Python
serializer's output (they must produce identical `.rds` bytes).

---

## Risks
- **A**: none of substance — it wraps golden-gated code; only the arg-parsing/exit-code surface
  is new (covered by the smoke test).
- **B**: (1) the C++ serializer must stay byte-identical to R `saveRDS(version=2)` — gate it
  against the Python reference bytes *before* the R step; (2) zlib gzip determinism (fixed
  header, `mtime=0`); (3) the import RDS reader is the real cost — keep export-first.
- **Naming**: `steppe-rds` (A) vs `steppe convert-f2` (B) are different commands; if both ship,
  document that they're equivalent and the binary one is Python-free.
