# `cmd_cache.cpp` reference

## 1. Purpose

`src/app/cmd_cache.cpp` is the native, host-only f2-cache inspector behind
`steppe cache ls | show | verify`. It is the compiled C++ twin of the
`steppe-cache` Python tool (`bindings/steppe/_cache.py`) — same header-only read
strategy, same size arithmetic, same content-address re-hash, same wording on the
lines it prints — so a user gets identical answers whether they reach for the
bundled Python entry point or the `steppe` binary itself.

It touches no GPU. There is no CUDA header here, no `RunConfig`, no `build_config`
precedence merge. The three entry points are dispatched straight from the `cache`
subcommand's callbacks in `cli_parse.cpp` (see that file's `cache` block), which
is why their signatures take a plain `std::string` path rather than a merged
configuration. That host-only isolation is deliberate and is enforced by the same
build-time grep gate that guards the rest of the app target: this translation unit
must stay free of GPU code.

Everything this file does is on-disk inspection. It never computes an f2, never
launches a kernel, and — the load-bearing rule of the whole tool — never reads the
multi-gigabyte f2 payload.

---

## 2. What an f2 cache is on disk

A cache is a directory. Three files matter:

| File | What it holds |
|---|---|
| `f2.bin` | A 64-byte `STPF2BK1` header, then the f2 slab, the vpair slab, and an int32 block-sizes trailer. This is the big one — often multiple gigabytes. |
| `pops.txt` | The population labels, one per line. |
| `meta.json` | An optional, open-ended provenance record. May carry the full schema, a minimal import record, or be absent entirely. |

A directory *is* a cache exactly when it contains a readable `f2.bin` whose header
carries the `STPF2BK1` magic and an FP64 `dtype` (section 3). That is the only
membership test — `ls` walks a tree and reports every directory that passes it,
`show` and `verify` insist on it before doing anything else.

---

## 3. The header-only read strategy

The single most important design rule: **only ever read the 64-byte header.**

`read_header(dir, h)` opens `f2.bin`, reads exactly `kF2DiskHeaderSize` (64) bytes
into an `F2DiskHeader`, and checks two things: the eight-byte `STPF2BK1` magic and
that the `dtype` word is FP64. The dtype gate matters because the section-4 size
arithmetic hard-codes an 8-byte element — a non-FP64 cache is one this reader
declines rather than mis-measures, exactly as the Python `_read_header_only` does.
(The `version` word is deliberately *not* checked — the Python reader doesn't
either, and tightening it here would be a new divergence.) It never seeks past byte
64. The f2 slab, the vpair slab, and the block-sizes trailer — the gigabytes of
payload — are never faulted into memory here.

That is what makes `ls` over a tree of large caches cheap: scanning fifty caches
reads fifty 64-byte headers, not fifty multi-GB files. `show` is instant for the
same reason. The header carries everything the tool reports about shape — the
population count `P` and the block count `n_block` are read straight from it and
treated as authoritative.

The one place a full file *is* read is `verify`, and only because integrity
verification is the one job that genuinely requires re-reading the bytes (section
6). Everything short of `verify` stays header-only.

---

## 4. The size arithmetic

`expected_f2_bytes(P, n_block)` reproduces the exact byte count the extract writer
lays down for an `f2.bin`:

```
expected = 64  +  2 * P * P * n_block * 8  +  4 * n_block
           ^^^     ^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^
           header  two P*P*n_block FP64 slabs  int32 block-sizes trailer
                   (the f2 slab + the vpair slab)
```

The 64 is the header. The middle term is the two FP64 slabs — f2 and vpair — each
`P * P * n_block` doubles at 8 bytes apiece, hence the leading `2` and the trailing
`* 8`. The last term is the `n_block` int32 block sizes at 4 bytes each.

For the committed 9-population golden (`docs/examples/f2_9pop/`, `P = 9`,
`n_block = 710`) this is `64 + 920160 + 2840 = 923064` bytes, which is exactly the
size that golden's `f2.bin` reports on disk. That equality is the tool's cheapest,
strongest correctness signal: it confirms shape without hashing a single byte of
payload. The arithmetic runs in `std::uint64_t` throughout so a large cache can't
overflow the product.

`show` and `verify` both compare the actual on-disk `f2.bin` size against this
expected value and mark it `OK` or `MISMATCH` / `FAIL`.

---

## 5. The content-address re-hash

The extract writer stamps a content address for the payload and the labels into
`meta.json`: `f2_cache_id` for `f2.bin` and `pops_sha256` for `pops.txt`. Both are
stored in the form `"sha256:<hex>"`.

There is one wrinkle the tool has to get right. The project's `sha256_file` helper
(shared with `f2_dir_writer`) returns a **bare** hex digest with no prefix. The
stored id, however, is `"sha256:"`-prefixed. So `content_id(path)` computes the
bare digest and prepends the literal `"sha256:"` before comparing:

```
content_id(p)  ==  "sha256:" + sha256_file(p)
```

That single prefix is the whole reconciliation — it makes the recomputed value
directly comparable to the string parsed out of `meta.json`, with no normalizing on
the stored side.

---

## 6. Pulling the stored id out of meta.json (the regex, and why)

`meta.json` is an open, best-effort record — a cache may carry the full schema, a
minimal four-field import record, or nothing at all — and this file links no JSON
library. So `extract_id(meta, key)` reaches for the one value it needs with a small
regex instead of a parser:

```
"<key>"\s*:\s*"([^"]*)"
```

It matches the quoted key, the colon (with optional whitespace either side), and
captures the **raw quoted value** — whatever string is stored, not only a
well-formed `sha256:` one. That breadth is the point: it mirrors the Python tool's
`meta.get(key)`, where a present-but-wrong value (a tampered id, a truncated hash,
a value missing the `sha256:` prefix) is a *mismatch* that fails verify, not
silently read as "no stored id". A genuinely absent key returns an empty string —
the one case that is treated as "nothing to check against" (section 7).

A companion `extract_int(meta, key)` does the same for an integer field (`P`,
`n_block`), returning `std::nullopt` when the field is absent — that's what backs
the header-vs-meta cross-check in verify (section 10).

`meta.json` itself is **not** hashable and is never attested. Verify speaks to the
payload and the labels — the things a content address can pin down — not to the
provenance record wrapped around them. Every verify run says so in its own closing
line.

---

## 7. Report-only when there's no stored id

A stored id can be legitimately absent — the `.rds`-import path, for instance,
writes no `f2_cache_id`. That absence is **not** a verification failure.

When `extract_id` comes back empty, `verify` prints an informational `[--  ]` line
that recomputes and shows the id ("no stored id; computed sha256:…") and does *not*
count a failure. A cache with nothing to check against still passes. A failure is
only ever a stored id that is present and *disagrees* with the recomputed value —
that, and only that, prints `[FAIL]` and bumps the failure count. This mirrors the
Python tool's three-state check exactly: `ok`, `fail`, or informational `info`.

---

## 8. `run_cache_ls`

`steppe cache ls [ROOT]` tabulates every cache under a root.

- When the argument is empty the root defaults via `default_root()`, which honors
  `$STEPPE_F2_DIR` — the same env the fit commands read — so `ls` finds the cache
  the rest of steppe already points at, falling back to the current directory. A
  missing root is **not** an error: the error-code-constructed iterator is simply
  the end iterator, so the scan yields nothing and `ls` reports "none" and exits
  successfully, the same way the Python tool's `os.walk` quietly yields nothing.
- It walks the tree with a `recursive_directory_iterator` that skips
  permission-denied subtrees rather than throwing, and advances with an error code
  so a mid-walk filesystem error stops the scan cleanly instead of aborting. Each
  per-entry `is_directory` / `relative` query uses the error-code overload too, so
  a single un-stattable entry is skipped, never thrown — the tree scan `ls` exists
  for stays robust to one bad symlink.
- For each directory it does the section-3 membership test: is there an `f2.bin`,
  and does `read_header` accept its magic? Directories that fail are silently
  skipped — a tree full of unrelated folders just doesn't clutter the table.
- Each surviving cache prints one row: its path relative to the root, `P` and
  `n_block` from the header, and a human-readable directory size. The size is the
  summed bytes of `f2.bin` + `pops.txt` + `meta.json` (`dir_bytes`), rendered by
  `human_size` into `B`/`K`/`M`/`G`/`T` units.
- If the walk finds no caches at all, it says so on standard error but still exits
  successfully — an empty tree is a clean, documented no-op, not an error.

---

## 9. `run_cache_show`

`steppe cache show <DIR>` pretty-prints one cache.

It insists the directory is a real `STPF2BK1` cache first (`read_header`); if not,
it stops with an invalid-configuration exit and a `not an STPF2BK1 cache` message.
Then it prints, in the exact wording the Python tool uses:

- `P (populations)` and `n_block`, each annotated `(from f2.bin header)` so the
  reader knows these came from the authoritative header, not from `meta.json`.
- The `f2.bin size` line: the actual on-disk size, the expected size from the
  section-4 arithmetic, and an `[OK]` / `[MISMATCH]` mark comparing them.
- The raw `meta.json`, dumped verbatim under a `--- meta.json ---` banner, or
  `(none)` when there is no `meta.json` (or it's empty). The record is shown as-is
  rather than reformatted — whatever the writer stamped is what the user sees.

`show` is header-only: it reads the 64 bytes, stats the file for its size, and
slurps the small text `meta.json`. It never reads the payload.

---

## 10. `run_cache_verify`

`steppe cache verify <DIR>` re-checks integrity against the stored content
addresses. This is the one command that reads whole files, because hashing is the
job.

Same gate first: the directory must be a real `STPF2BK1` cache or it exits
invalid-configuration. Then it runs up to five checks, printing an aligned status
line for each and tallying failures:

1. **`f2.bin size`** — actual on-disk size versus the section-4 expected size.
   `[OK  ]` or `[FAIL]`.
2. **`header P` / `header n_block`** — run only when `meta.json` carries those
   redundant fields, each is compared against the authoritative header value. A
   `meta.json` that contradicts the header is a `[FAIL]`. This mirrors the Python
   `_verify_checks` header cross-check, and catches a provenance record that has
   drifted from the payload it describes.
3. **`f2_cache_id`** — the section-5 recomputed `"sha256:"+bare` digest of
   `f2.bin` versus the id parsed from `meta.json`. `[OK  ]` on match, `[FAIL]` on
   mismatch, or the informational `[--  ]` "no stored id" line of section 7 when
   `meta.json` carries none.
4. **`pops_sha256`** — the same three-state check for `pops.txt`, run only when a
   `pops.txt` is actually present.

It closes with the fixed reminder that verify attests the `f2.bin` payload and the
`pops.txt` labels, and that the `meta.json` record itself is not hashable (section
6). If any check failed it writes a `verify FAILED (N check(s))` summary to
standard error and exits invalid-configuration; otherwise it exits successfully.

### The `--check-sources` flag

`run_cache_verify` takes a `check_sources` bool, but on the native path it is
deliberately a no-op (it's cast to void). Re-hashing the original genotype
`geno`/`snp`/`ind` source files — which can be enormous — is left to the
`steppe-cache` Python tool, where `--check-sources` does the full source-file walk.
The flag is accepted here so the two front ends share a command surface, but the
heavy source-hashing work has one home, not two.

---

## 11. Exit codes

The tool uses just two outcomes, drawn from `core/config/exit_code.hpp`:

| Condition | Exit |
|---|---|
| Normal completion — including an `ls` whose root is missing or holds no caches | success (`kExitOk`) |
| A `show` / `verify` directory that isn't an FP64 `STPF2BK1` cache; a verify with at least one failed check | invalid-configuration (`kExitInvalidConfig`) |

A failed integrity check is treated as an invalid-configuration exit rather than a
device or runtime fault, which fits the tool's nature: there is no GPU here to
fault, only on-disk facts that either line up or don't.
