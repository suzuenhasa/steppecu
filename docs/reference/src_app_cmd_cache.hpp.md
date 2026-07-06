# `cmd_cache.hpp` reference

## 1. Purpose

`src/app/cmd_cache.hpp` declares the entry points for the `steppe cache`
command-line subcommand — a small family of **host-only** cache inspectors:

| Function | Backs | What it does |
|---|---|---|
| `run_cache_ls(root)` | `steppe cache ls [ROOT]` | Tabulate every STPF2BK1 f2 cache found under `ROOT`, header-only. |
| `run_cache_show(dir)` | `steppe cache show <DIR>` | Print one cache's header facts, an integrity mark, and its raw `meta.json`. |
| `run_cache_verify(dir, check_sources)` | `steppe cache verify <DIR>` | Re-hash `f2.bin` / `pops.txt` against the content-address stamped in `meta.json`. |

These are the native C++ mirror of the `steppe-cache` Python tool
(`bindings/steppe/_cache.py`): the same STPF2BK1 header read and the same
content-address re-hash, exposed as a subcommand of the main binary so a user
who only installed the CLI can still inspect and verify their caches.

The header is deliberately thin — three free functions, each returning an exit
code. Everything about *how* a cache is read lives behind these seams in
`src/app/cmd_cache.cpp` (see `src_app_cmd_cache.cpp.md`).

---

## 2. Why these functions take plain strings, not a `RunConfig`

Every other `steppe` subcommand parses its flags into a validated `RunConfig`
and dispatches to a GPU `run_*_command`. The cache inspectors do **not**: they
touch no device, allocate no GPU memory, and read no genotypes. They only open a
64-byte file header and hash a couple of files. So they are dispatched straight
from the `cache` subcommand in `cli_parse.cpp` with plain path arguments — no
config build, no device selection, no precision policy. This is what "host-only"
means here, and it is enforced structurally: this translation unit includes no
CUDA header (the same build grep-gate that guards the rest of `src/app`).

---

## 3. Contract and exit codes

Each function returns a `config::CliExitCode`:

- `kExitOk` (0) — the listing completed, or the cache is well-formed and its
  stored content-address re-hashes correctly. A missing `ls` root is **not** an
  error (it lists nothing and returns 0, matching the Python tool).
- `kExitInvalidConfig` (2) — `show` / `verify` were pointed at something that is
  not a readable STPF2BK1 cache, or `verify` found a size / hash / header
  mismatch.

`run_cache_verify`'s `check_sources` flag is accepted for parity with the
Python tool but is a reserved no-op on the native path: re-hashing the original
`geno` / `snp` / `ind` source files is left to `steppe-cache`. See
`src_app_cmd_cache.cpp.md` §for the per-check semantics and the exact parity
rules the implementation follows.
