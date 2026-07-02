# Feature design — AT2 ⇄ steppe f2 `.rds` converter (`import_f2_rds` / `export_f2_rds`)

**Status: scoped, ready to build.** An *optional* converter so an f2 cache can move between
steppe's `STPF2BK1` format and ADMIXTOOLS 2's `read_f2()` `.rds` directory — primarily so a
user can **export a steppe f2 cache and verify steppe's results inside ADMIXTOOLS 2 / R**.

Grounded against the real formats: the AT2 side was verified by loading the `.rds` objects in
**R 4.3.3 + admixtools 2.0.10** and reading `read_f2()` source on box5090; the steppe side from
`src/device/f2_disk_format.hpp` + `src/app/f2_dir_{writer,io}.cpp`.

---

## 1. There is no fundamental blocker

steppe's f2 values and jackknife-block partition are already **bit/tolerance-identical to AT2**
on the same `blgsize`/`setblocks` partition. So this is **pure container translation** — no
compute, no GPU, no numerical work. The only difference is the on-disk shape.

## 2. The one real catch: reading is easy, *writing* `.rds` arrays is blocked

- **Import (AT2 → steppe): tooling-easy.** `pyreadr` reads AT2's per-pair matrices (no R
  needed); steppe's `f2.bin` writer already exists (`tests/python/conftest.py`).
- **Export (steppe → AT2): the load-bearing blocker.** Every off-the-shelf library
  (`librdata`, `pyreadr`) can only *write* a column-oriented R **data.frame** — none can emit
  AT2's **numeric matrix with `dim`/`dimnames`**, and `librdata`'s writer has no gzip path at
  all. **Fix:** AT2's target shape is tiny and fully specified, so a **~150-line hand-rolled
  RDS serializer** (pure stdlib: `struct` big-endian XDR + `gzip`) writes it exactly, with zero
  heavy dependencies. Bit-exactness is free (XDR + gzip are lossless).

## 3. Format mapping (verified)

| steppe `STPF2BK1` | AT2 `read_f2` directory |
|---|---|
| `f2.bin`: 64-byte header, then P×P×n_block FP64 f2 tensor | one subdir **per pop**; each unordered pair `<p1>/<p2>_f2.rds` = numeric matrix `[n_block, 2]`, col 1 = f2, `colnames c("f2","counts")`, gzip'd |
| int32 `block_sizes` trailer | `block_lengths_f2.rds` = integer vector `[n_block]` (per-block SNP counts) |
| `pops.txt` (P-axis order) | subdir names = pops; `read_f2` derives `dimnames` from them |
| FP64 `vpair` (per-block valid-SNP counts) | **not stored** in AT2's f2 family (its "counts" col is all `1.0`) |
| `meta.json` (advisory) | `cache_metadata.json` + `.f2_cache_id` (present, but **not** read by `read_f2`) |

**AT2 `read_f2()` contract (from source):** `pops = list.dirs(recursive=FALSE)` — so a subdir
must exist for **every** pop (even the last-alphabetical one, holding only its zero self-pair);
files are keyed under the **C-locale-smaller** pop of each pair; the diagonal self-pair matrix
is all `0.0`; `block_lengths_f2.rds` is mandatory (`read_f2` aborts without it). f2 alone is
enough for qpAdm/qpGraph — the parallel `_ap`/`_fst` families are optional.

## 4. Three semantic edge cases (must get right, all cheap)

1. **Pop ordering** — AT2 C-locale-sorts `dimnames`; the writer keys each pair under
   `p1 = C-locale-smaller` and **creates a subdir for every pop**. Import sorts with Python
   `sorted()` (byte order == C-locale, not locale-aware).
2. **Diagonal** — steppe's `f2[i,i]` is the nonzero within-pop het correction; AT2 self-pairs
   are all `0.0`. Export writes zeros on the diagonal; import zeros steppe's. Harmless for
   f4/qpAdm (off-diagonal only) — assert **off-diagonal** in the gate.
3. **`vpair` is import-lossy** (asymmetric) — AT2's f2 "counts" col is `1.0`, not real counts,
   so steppe's `vpair` can't be recovered from f2-only `.rds`; fill a **nonzero sentinel**
   (`block_sizes` broadcast), never zeros (zeros trip the `vpair==0` missing-block detector and
   break `maxmiss>0`). True `vpair` needs the `_ap` family. **Export is unaffected.**

## 5. Mechanism — Python facade behind an optional `[rds]` extra

Pure format translation, no GPU → it lives in the **Python facade**, base wheel stays
numpy-only. Different tools per direction (no single library does both):
- **Import:** `pyreadr` reads the per-pair matrices → write `STPF2BK1` in pure Python.
- **Export:** the hand-rolled RDS/XDR serializer (pure stdlib) → AT2's exact array shape.

An `Rscript`/`rpy2` sidecar is reserved **only** as an optional byte-for-byte cross-check for
users who already have R — never on the critical path (it would force an R install). A C++
`convert-f2` subcommand (vendored `librdata` reader + the same hand-rolled writer) is the
heavier alternative if a Python-free CLI is ever wanted.

## 6. Files (≈ **Medium**, ~570 LOC)

| File | LOC | change |
|---|---|---|
| `bindings/steppe/_rds.py` | ~200 | **new** — hand-rolled RDS XDR (de)serializer (`_write_rds_matrix`, `_write_rds_int_vector`, `_read_*` via `pyreadr` lazy-import with a clear `pip install steppe[rds]` error) |
| `bindings/steppe/__init__.py` | ~150 | `export_f2_rds(f2, out_dir, *, counts='ones'|'vpair', write_ap=False)` + `import_f2_rds(rds_dir, out_dir, *, type='f2')`; lazy-import `_rds`; add to `__all__` |
| `pyproject.toml` | ~4 | add `rds = ["pyreadr>=0.5"]` (import-only dep; **export needs no dep**) |
| `tests/python/test_py_convert_rds.py` | ~160 | **new** — round-trip gates (CI, no R) |
| `tests/r/verify_export_rds.R` | ~55 | **new** — box5090-only end-to-end AT2 gate |
| `docs/commands.md` | ~15 | document `export_f2_rds`/`import_f2_rds` + the "verify in AT2" flow |

## 7. Direction & golden gate

**Do export first** — it's the stated goal (verify in AT2) *and* the harder engineering half
(the serializer). Import is a smaller follow-on that reuses the pop-order/diagonal logic in
reverse (with the `vpair` caveat).

**Export gate (the literal "verified in ADMIXTOOLS 2" proof):**
1. *CI proxy, no R* — export a small steppe f2 dir → re-read every `<p1>/<p2>_f2.rds` with
   `pyreadr`, assert col 1 == steppe `f2[i,j,:]` **bit-exact** and `block_lengths_f2.rds` ==
   `block_sizes`, and a subdir exists for all P pops.
2. *End-to-end, box5090 (R 4.3.3 + admixtools 2.0.10)* — `admixtools::read_f2(exported_dir)`
   returns `[P,P,n_block]`; assert off-diagonal == steppe's f2 and `dimnames` == sorted pops;
   then run `admixtools::qpadm/f4` on the exported cache and assert est/se **match steppe's
   native fit on the same cache**.

**Import gate:** real AT2 dir → `import_f2_rds` → `STPF2BK1` → `extract_f2` self-consistency at
~1e-12 (off-diagonal only; do **not** assert on `vpair`).

**Test small.** Use the 9-pop cache (`/workspace/data/qpgraph_9pop_stpf2bk1`, 45 files) so the R
round-trip is seconds, not minutes.

## 8. Risks

- Serializer correctness (endianness, RDS XDR v2, the attribute pairlist, NaN/Inf) — caught by
  the pyreadr re-read byte-check *before* the R step.
- Pop-order / diagonal conventions — caught by the R `dimnames` + off-diagonal asserts.
- `O(P²)` tiny-file fan-out (45 at P=9; ~295k at P=768) — emit only the `f2` family by default;
  scale-guard on large P.
- Float exactness is a **non-issue** (XDR + gzip lossless).
