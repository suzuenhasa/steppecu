# Massive f-stat output: how to handle billions of result rows

**Status:** RESEARCH ONLY. No implementation, no `src/` changes. This document is the
research record + a recommended architecture for the user to decide on. Branch
`phase2-fit-engine` @ `da40d6c` (the workflow note cited `8751759`; actual HEAD is
`da40d6c` — recorded for honesty).

**The user's two explicit questions, answered up front:**

1. **Does compression help?** Only marginally on the part that matters. The float payload
   (`est/se/z/p`) is near-incompressible: **measured ~1.04–1.06× with generic zlib, ~1.13–1.20×
   with the float-aware BYTE_STREAM_SPLIT transform** (local benchmark below; matches the
   published FP64-entropy ceiling of ~1.25–1.33×). The columns that DO compress are the
   integer pop-index columns (~2.2× random, up to ~770× when written in enumeration order),
   not the statistics. **Compression is a ~1.1–1.3× rounding error on a multi-TB problem; it
   does not change the order of magnitude.**

2. **How is this handled in general?** No prior tool ever hit this wall, so there is no
   "billions-of-rows f-stat store" precedent to copy. The three reference tools handle "lots
   of output" by (a) **streaming each row as it is computed** and never holding a table
   (DReichLab AdmixTools C), (b) **refusing above a hard combination cap** (ADMIXTOOLS 2 R,
   `maxcomb = 1e6`), and (c) **collapsing the combinatorics structurally** (Dsuite, one
   canonical row per trio). The general big-data answer for the rare full dump is
   **stream → shard → columnar-on-disk → query-don't-load (DuckDB/Arrow predicate pushdown)**.
   The dominant lever, verified across all angles, is **FILTERING** (write only `|z|>t` /
   top-K), which buys 1–3 orders of magnitude where compression buys ~1.1×.

---

## 1. The problem, quantified

A standalone f-stat result is a table with one row per quartet/triple:
`pop1..pop4` (indices) + `est, se, z, p` (FP64). The numeric payload is **32 B/row**
(four FP64); with int32 indices it is **48 B/row**. The row count is combinatorial.

`C(P,4)` quartets (AT2's default enumeration emits **3×** that for the three topology
orientations — verified, qpDstat.c:567/:659 prints three `result:` lines per quad):

| P (pops) | `C(P,4)` rows | ×3 (AT2 ordered) | raw 32 B/row | ×3 raw |
|---|---|---|---|---|
| 100  | 3.92 M | 11.8 M | 0.0001 TB | 0.000 TB |
| 300  | 331 M | 992 M | 0.011 TB | 0.032 TB |
| 500  | 2.57 B | 7.72 B | 0.082 TB | 0.247 TB |
| 1000 | 41.4 B | 124 B | **1.33 TB** | 3.98 TB |
| 2000 | 665 B | 1.99 T | 21.3 TB | 63.8 TB |
| 2500 | 1.62 T | 4.87 T | 52.0 TB | 156 TB |
| 4266 (architecture §11.4 top-end pop count) | 13.8 T | 41.3 T | **441 TB** | 1.32 PB |
| 5000 | 26.0 T | 78.0 T | 832 TB | 2.50 PB |

f3 triples `C(P,3)` are ~12× smaller but still 166 M rows (5.3 GB) at P=1000, 20.8 B
(666 GB) at P=5000.

**Verified by `python3 math.comb` this session.** The premise is exactly correct: from
~P=500 the full table is multi-TB; at the architecture's own top-end pop count (4266) it is
**~0.4–1.3 PB**. It cannot be a single RAM object in C++ *or* Python — a billions-row
numpy/pandas object is impossible regardless of how it is produced. **At the high end the
full table can never be the deliverable: it must be filtered, or it must never be fully
materialized.** steppe's own boxes are 2×5090 / 2×RTX-PRO-6000 workstations, not a PB array,
so the full table at P>~1500 is physically off-box.

steppe's **compute** already honors a bounded-memory contract (diagonal-jackknife +
item-tiling, OOM-free and linear to 1M items — project memory). The **output contract** does
not: the result struct and the binding both assume the whole table fits in host RAM.

---

## 2. Per-angle findings (with citations)

### 2.1 Prior art — how the reference tools handle volume

**Three distinct, verified strategies; none is "one giant RAM table":**

**(a) DReichLab AdmixTools C (`qpDstat`) = pure streaming, no output array.**
Verified directly on box5090 (`/workspace/AdmixTools_src/src/qpDstat.c`):
- The enumeration requires an explicit population set — it never auto-explodes to `n^4`:
  `qpDstat.c:254-255` is fatal if both `poplistname` and `popfilename` are NULL.
- Two modes (README.Dstatistics:44-45): `poplistname` = "Program will run the method for
  **all quadrapules**"; `popfilename` = "Enter a list of tests you want to perform — 4
  populations on each line" (the targeted, hypothesis-driven path).
- The all-quad path is a nested `for a<b<c<d` loop over `numeg` (`qpDstat.c:506-511`).
- Each result is `printf`'d the instant it is computed, **inside** the loop
  (`qpDstat.c:567`, and `:659` for the explicit-quad-list loop) — there is **no accumulation
  array**. README.Dstatistics:52: "The program will write all the output to stdout."
- The documented mitigation for scale is **row-range sharding**, not compression:
  `locount/hicount` = `-L lo -H hi` (`qpDstat.c:96`, `:319`), and README.Dstatistics:33-34
  warns that the all-quad mode "uses an **excessive amount of memory**. A solution … is to
  make multiple runs with -L lo -H hi set." The z-score is printed but **never gates output**
  — there is no significance filter.

**(b) ADMIXTOOLS 2 R (`qpdstat`) = hard refusal above 1e6 + fully in-RAM tibble.**
Verified verbatim from `R/qpdstat.R` (raw GitHub):
```r
out = expand_grid(pop1, pop2, pop3, pop4)
maxcomb = 1e6
if(ncomb > maxcomb & !sure) {
  stop(paste0('If you really want to compute ', ncomb,
              ' f-statistics, run this again with "sure = TRUE".'))
}
```
The result is one in-memory tibble (`pop1..pop4, est, se, z, p`); **no streaming, no
flush-to-disk, no significance filter**. The `maxcomb = 1e6` cap (≈ `C(45,4)`) is an explicit
"you probably did not mean this" guard. This is the single strongest prior-art signal that
the full all-quartets table is generally **not wanted**.

**(c) Dsuite (`Dtrios`) = combinatoric collapse + streamed text.**
Verified from github.com/millanek/Dsuite + Malinsky 2021 (Mol Ecol Resour). Computes **one D
per unordered trio** (`C(n,3)`, outgroup fixed) by orienting P1/P2 so nABBA≥nBABA — collapsing
the ordering blow-up. Each trio (every one — no significance filter) is written to a tab-text
file; `DtriosCombine` sums counts across separate runs (a shard/merge pattern). Tutorials run
**tens** of species (20 sp = 1140 trios), not thousands.

**Honest nuance:** none of the three *filters* by significance — but none ever *reaches*
billions of rows either (R caps at 1e6, Dsuite runs tens of pops). The absence of a filter is
a consequence of never hitting the wall, not evidence that billions-of-rows is a desired
artifact. **steppe is the first tool that makes `C(2500,4)` = 1.6 T quartets computable, so
steppe is the first that MUST set an output policy — there is no precedent to copy.**

### 2.2 Compression — does it help? (measured, not guessed)

Local benchmark this session (`python3` + numpy + zlib, on realistic f-stat-shaped columns:
`est~N(0,3e-3)`, `se=|N(2.5e-4,8e-5)|`, `z=est/se`, N=2M rows):

| What | ratio | B/row |
|---|---|---|
| `est` FP64, zlib-6 | **1.043×** | 7.7 |
| `se` FP64, zlib-6 | 1.064× | 7.5 |
| `z` FP64, zlib-6 | 1.037× | 7.7 |
| `est` FP64, BYTE_STREAM_SPLIT + zlib-6 | **1.143×** | 7.0 |
| `se` FP64, BSS + zlib-6 | 1.197× | 6.7 |
| `z` FP64, BSS + zlib-6 | 1.126× | 7.1 |
| int32 pop-index (random, P=1000), zlib-6 | 2.238× | 1.8 |
| int32 pop-index **sorted (enumeration order)**, zlib-6 | **773×** | ~0.0 |
| FP32 lossy (store `est` as f32) | 2.00× | 4.0 |
| **CSV 17-sig-digit (current steppe `fmt_double`)** | **0.37× (2.7× BLOAT)** | 86.6 |

**Verified principles behind the numbers:**
- A 52-bit FP64 mantissa of noisy jackknife estimates is near-maximum entropy. Lossless FP
  compression has a theoretical ceiling around ~1.25–1.33× (FCBench arXiv:2312.10301:
  median 1.225 single / 1.202 double; the float-compression literature consensus). My
  benchmark independently reproduces ~1.04–1.20×.
- BYTE_STREAM_SPLIT (Parquet encoding **ID 9**, applies to **FLOAT, DOUBLE, INT32, INT64,
  FIXED_LEN_BYTE_ARRAY**): verified verbatim from the Apache Parquet spec — "This encoding
  **does not reduce the size of the data** but can lead to a significantly better compression
  ratio and speed when a compression algorithm is used afterwards." My benchmark confirms the
  ~7–18% uplift over plain zlib.
- The **integer index columns** compress well (low cardinality → dictionary/RLE; Parquet
  RLE_DICTIONARY = ID 8, "All Physical Types"). Sorted-in-enumeration-order they nearly
  vanish (the outer loop variable is constant across long runs).
- **Pcodec / Pco** (arXiv:2502.06112) reports **44–158% higher ratio than Zstd-Parquet** on
  numeric sequences — but **those benchmarks are Taxi / time-series data with
  autocorrelation**, not high-entropy f-stat mantissas. **SPECULATIVE for steppe:** Pco may
  beat BSS+Zstd on f-stats but is unverified on this data and is a non-standard dependency.

**Conclusion (direct answer):** compressing the floats is worth ~1.1–1.2×. The real shrink
levers are, in order: **FILTERING** (1–3 orders of magnitude), **binary-vs-ASCII** (current
CSV is 2.7× *bigger* than raw binary; switching to binary is an immediate ~2.7× win *before*
any codec), **dropping z/p** (recompute on read: `z=est/se`, `p=2·pnorm_upper(|z|)` — removes
16 of 32 numeric B/row for free), and **lossy FP32** (2×, but a numerics-policy decision —
see §6). Compression of the float payload is a footnote.

### 2.3 Filtering — the order-of-magnitude lever (quantified)

`|z|>t` survival under the pure-null background (two-sided `2·(1−Φ(t))`; real data has signal
so a touch higher), at `C(1000,4)` = 41.4 B rows:

| threshold | null fraction | survivors @ P=1000 | size |
|---|---|---|---|
| `|z|>2` | 4.55e-2 | 1.88 B rows | 60 GB |
| `|z|>3` | 2.70e-3 | 112 M rows | **3.6 GB** |
| `|z|>4` | 6.33e-5 | 2.6 M rows | 84 MB |
| `|z|>5` | 5.73e-7 | 23.7 K rows | <1 MB |
| `|z|>6` | 1.97e-9 | ~82 rows | KB |

`|z|>3` (the canonical f4 significance threshold) shrinks 1.33 TB → ~3.6 GB (≈380×); `|z|>5`
→ KB. **Multiple-testing math reinforces this:** with 10⁹–10¹² simultaneous tests a
Bonferroni-corrected `0.05/N` threshold corresponds to `|z|≈6.0–6.5`, and FDR/Benjamini–
Hochberg is the preferred correction at that scale — so under *any* defensible correction
only thousands-to-millions of rows are interpretable; the rest is provably noise the
researcher will never read. **Filtering composes with streaming**: apply the predicate in the
emit loop, never buffer the rejects. A bounded top-K min-heap (size K) gives a fixed-memory
"most significant K" regardless of N.

**SPECULATIVE:** the exact survivor fraction on **real AADR** all-quartets output is not
measured (no run done — this is a research doc). The order-of-magnitude conclusion
(filtering ≫ compression) is robust; the precise 380× at `|z|>3` depends on the real
significant fraction.

### 2.4 Big-data patterns + Python read-back

The universal robust pattern (verified across Arrow/Parquet/DuckDB docs):
- **Producer** = bounded-memory streaming writer. Arrow C++ `parquet::arrow::FileWriter` /
  `WriteRecordBatch` / `StreamWriter` write row-groups incrementally, never materializing the
  full table (arrow.apache.org/docs/cpp/parquet.html). pyarrow `ParquetWriter.write_batch` in
  a loop is the Python equivalent.
- **Sharded/partitioned** output (many row-group-sized files, not one PB file): enables
  parallel multi-GPU/multi-thread writes + partition pruning on read. Mirrors AT2's `-L/-H`
  and Dsuite's `DtriosCombine`.
- **Columnar + predicate pushdown** for the consumer. **Verified from DuckDB docs:**
  projection pushdown — "only the columns required for the query are read"; filter pushdown —
  "the filter will be pushed down into the scan, and can even be used to skip parts of the
  file using the built-in zonemaps [row-group min/max]"; glob — `SELECT * FROM 'dir/*.parquet'`.
  So `SELECT * FROM 'fstats/*.parquet' WHERE abs(z)>5` over a multi-TB dataset returns only
  the handful of significant rows into a small DataFrame, **out-of-core, never loading the
  file**.

**Python read-back per option (you can NEVER return a billions-row DataFrame):**
- **CSV/TSV:** `pandas.read_csv(chunksize=)` or DuckDB `read_csv_auto`; full scan every time,
  no column/row skipping. Worst.
- **Parquet/Arrow:** `pq.ParquetFile(...).iter_batches(batch_size, columns=[...])` (chunked,
  projected, bounded memory); `read_table(columns=, filters=[('z','>',3)])` (projection +
  predicate pushdown via row-group stats); `pyarrow.dataset` for the sharded glob; DuckDB SQL
  for the cleanest TB-scale path. cuDF/kvikIO can read Parquet straight to GPU (general
  knowledge, **not re-verified this session**).
- **Filtered / top-K output:** fits in RAM by construction → return directly as a DataFrame.

**Caveats (verified):** Parquet `memory_map=True` "won't help much with resident memory
consumption" because compressed data must be decoded — true zero-copy mmap is only for
*uncompressed* Arrow IPC/Feather (arrow.apache.org/docs/python/parquet.html). **Row-group
pruning only helps if output is written sorted by the filter key** (z); on unsorted output
every row-group spans the full z range and min/max stats prune poorly — getting pushdown
leverage requires sort-by-z or significance-bucket partitioning on write.

### 2.5 steppe reuse — what exists, and the gap

**The central finding:** steppe's streaming/tiering machinery exists, but it is built entirely
for the **f2_blocks INPUT cache**, not for f-stat **results**. The result path is 100%
RAM-resident with no streaming seam.

**Reusable (verified in-tree):**
- **`src/device/f2_disk_format.hpp`** — the only on-disk format precedent: 64-byte versioned
  header (magic `STPF2BK1`, version, dtype, P, n_block, region offsets) + raw FP64 regions,
  little-endian (lines 21-42), with the `detail::slab_offset` **uint64-widen-before-multiply**
  discipline (lines 51-55) that a multi-TB seekable file needs. **Reuse the header/versioning/
  offset discipline, not the layout** — it is a dense positional `[P²·n_block]` tensor with no
  per-row keys; an f-stat table needs `pop1..pop4` keys per row.
- **`src/device/cuda/block_sink.cuh`** — `StagingRing`: a triple-buffered
  (`kStreamStagingSlots = 3`, line 46) pinned ring + background writer thread that overlaps
  GPU compute / D2H / disk pwrite, with fail-fast event sync (line 95), backpressure
  (`acquire_slot`, line 206), and a **`DrainFn` callback seam** (line 125) abstracting "where
  the bytes land" (HostRam memcpy vs Disk pwrite). **This is the exact producer/consumer shape
  a streamed f-stat emitter wants; the `DrainFn` seam is the natural injection point for a
  Parquet/CSV row-group writer.** It is CUDA-private (`.cuh`, `cudaStream_t`, P²-slab-shaped),
  so it is the **design blueprint**, not the literal class.
- **`src/device/tier_select.hpp`** — `OutputTier {Resident, HostRam, Disk}` (lines 33-38),
  selected by runtime free-VRAM/free-RAM probes, with `STEPPE_FORCE_TIER` override. The same
  "fits RAM → in-memory; else spill to disk" policy is exactly the output-mode switch an
  f-stat result needs (small/targeted → DataFrame; billions → stream to disk).
- **`src/app/result_emit.cpp`** — the CSV emitters (`emit_f4_csv` :465-480, f3 :516-530,
  f4ratio :566-581) **already stream row-by-row to a `std::ostream`** (one `os << row` per
  quartet — no giant string), and the format primitives (`fmt_double` 17-sig-digit :25,
  `csv_quote`, NA sentinel) are reusable verbatim. **But the streaming stops at the ostream:**
  they consume a fully-materialized `F4Result`. JSON (`emit_f4_json` :482-508) is a structured
  object, **not** cleanly streamable — exclude it for massive output.
- **`bindings/module.cpp`** — `f2_to_numpy` (capsule deleter, zero-copy view, DLPack
  deferral noted :14-21) is the template for read-back: **return a path / lazy handle, never a
  DataFrame**.

**MUST CHANGE (the gap, verified):**
- **`include/steppe/f4.hpp:46-60`** — `F4Result` is 9 fully-materialized `std::vector`s
  (`p1..p4` int + `est/se/z/p` double). `src/core/qpadm/f4.cpp:75-78,101-104,122-134`
  `reserve()`s and `assign()`s all to full N and writes `res.est[ks]=…` per row.
  `dstat.cpp` is identical. **At billions of rows these vectors are multi-TB and cannot
  exist** — even though compute already tiles, the result aggregation re-materializes the
  whole table.
- **`bindings/module.cpp:229` `f4_to_dict`** copies every vector into an `nb::dict`; the
  `__init__.py` facade builds a `pd.DataFrame` from it. **A billions-row dict/DataFrame is
  impossible** (the R reference hits the same wall — confirmed: it returns a tibble).

---

## 3. Option comparison

| Option | Write memory | Size vs raw f64 | Python read-back | Verdict |
|---|---|---|---|---|
| **CSV/TSV stream** (current shape) | O(tile) if streamed from tiles | **2.7× BIGGER** (17-sig ASCII) | full scan, `chunksize=`; no skip | Human-readable small-N only; worst at scale |
| **CSV + gzip** | O(tile) | ~1.3× bigger than raw binary | full scan | Still bigger than binary; no pushdown |
| **Columnar Parquet** (BSS floats + dict indices + Zstd) | O(row-group) incremental | ~0.5–0.9× (binary win + ~1.2× float codec + index compression) | **projection + predicate pushdown, iter_batches, DuckDB glob** | **Best interop default** |
| **Arrow IPC / Feather (uncompressed)** | O(batch) | ~1.0× (binary, no codec) | zero-copy mmap (the one format where mmap helps) | Alt when mmap zero-copy matters |
| **Sharded files** (orthogonal) | per-shard bounded | (same as chosen format) | partition pruning / glob | Multi-GPU write layout; compose with Parquet |
| **mmap positional binary** | O(1) if row index computable from quartet | 1.0× (uncompressed) | numpy.memmap view | Only fits fixed-enumeration-order, keyless schemes; niche |
| **FILTERING (`|z|>t` / top-K)** | O(tile) + O(K) | **1/380× at `|z|>3`, 1/400000× at `|z|>5`** | result fits RAM → direct DataFrame | **The dominant lever; default path** |
| **Queryable store (DuckDB over Parquet)** | (writes Parquet) | (Parquet) | SQL out-of-core, returns small result | The read-back answer at TB scale |
| **Zarr / HDF5** | chunked | ~codec | chunk reads | **Wrong shape** — result is a queried *table*, not a dense tensor; do not use |

**Key trade-off truths:**
- The biggest single size win over the *current* path is **binary instead of 17-digit ASCII**
  (~2.7×), available the moment you leave CSV — before any codec.
- **Compression of the floats is ~1.1–1.2× and is not the answer** (§2.2).
- **Filtering is 1–3 orders of magnitude** and is the only thing that makes the full table
  problem disappear (§2.3).
- Parquet's value is *not* primarily bytes — it is **predicate/projection pushdown** (the
  read-back story) + the binary-vs-ASCII win + modest index compression.

---

## 4. Recommended architecture

A **filter-first, stream-and-shard-for-the-opt-in-full-case, columnar-on-disk,
query-don't-load** design. Concretely, four layers:

**(1) PRIMARY = targeted sets + filter-on-write (the 99% path).**
- A **targeted quartet/trio list** (mirror AT2 `popfilename`: O(list) not O(P⁴)) — the
  hypothesis-driven case that is most real usage.
- For sweeps, a **streaming emitter wired into the existing tile loop** that writes only rows
  passing a predicate: `--min-abs-z` / `--max-p` / `--top-k` (bounded min-heap, O(K) memory).
  This converts multi-TB → MB–GB and makes the billions-row in-RAM object problem vanish for
  the realistic case. Filtering is composed into the emit loop; rejects are never buffered.

**(2) FORMAT = columnar Parquet, written one RecordBatch per tile.**
- Float columns (`est`, `se`) with **BYTE_STREAM_SPLIT + Zstd** (captures the available
  ~1.1–1.2×; free decode-speed win per the spec).
- **Pop identity as int16/int32 index columns + a `pops.txt` sidecar** (P<32767 fits int16),
  **not** repeated name strings — indices dict/RLE-compress to near-nothing in enumeration
  order; names are pure overhead.
- **Consider dropping `z`/`p` on disk** (derive on read: `z=est/se`, `p=2·pnorm_upper(|z|)`)
  — halves the numeric payload for free. (Open decision §7.)
- `write_statistics=True` + a chosen `row_group_size`; **sort/partition by `z`** (or a
  significance bucket) so row-group min/max stats actually prune on read.
- Keep CSV/TSV as a human-readable *small-N* convenience only, explicitly warned as the
  largest and slowest-to-query.

**(3) FULL UNFILTERED DUMP = opt-in, streamed, sharded, gated.**
- Replace the RAM-resident `F4Result` on the sweep path with a **streaming sink** (keep the
  in-RAM struct for small/targeted results). The sink reuses the **`StagingRing` blueprint**
  (block_sink.cuh) — GPU produces result chunks, background thread writes them — with a
  Parquet row-group writer in the `DrainFn` seam, sharded one Parquet file per
  shard/GPU/pop1.
- Add an **AT2-2-style safety cap** (`maxcomb = 1e6` analogue): refuse an unfiltered
  all-quartets sweep above a configurable row budget unless the user passes an explicit
  override flag *and* a target shard directory (mirrors `sure = TRUE` and AT2-C's `-L/-H`
  expectation).

**(4) PYTHON READ-BACK = query, don't load.**
- Bindings must **not** promise `to_dataframe()` on a sweep result. Ship Parquet; document/wrap
  a **DuckDB-over-Parquet** path (predicate + projection pushdown reads only matching rows off
  disk into a small DataFrame) plus `pyarrow.dataset` `iter_batches` / `filters=`. A full
  billions-row DataFrame is an explicit **non-goal**. The filtered/top-K output, by
  construction, fits in RAM and *can* be returned directly.

**steppe reuse map:** `StagingRing` triple-buffer + background-writer + `DrainFn` seam
(blueprint for the result sink); `tier_select` Resident/HostRam/Disk + `STEPPE_FORCE_TIER`
(the small→DataFrame / huge→disk mode switch); `f2_disk_format` header/versioning/uint64-offset
discipline (template for any steppe-native shard format); `result_emit` row-by-row ostream +
`fmt_double` primitives (CSV path verbatim); `f2_to_numpy` capsule/path-return pattern (the
read-back seam). **Do NOT** introduce Zarr/HDF5 (wrong shape).

---

## 5. Verified vs speculative

**VERIFIED (real source, this session):**
- Combinatorics & `|z|>t` survival: `python3 math.comb` / `statistics.NormalDist` (§1, §2.3).
- AT2-C streams per-row, requires explicit pop set, shards by `-L/-H`, warns "excessive
  memory": box5090 `qpDstat.c:96,254-255,506-511,567,659,319` + `README.Dstatistics:33-34,44-45,52`.
- AT2-2 R `maxcomb = 1e6` + `sure` stop + `expand_grid` in-RAM tibble + no significance
  filter: verbatim from `R/qpdstat.R` (raw GitHub).
- Parquet BYTE_STREAM_SPLIT (ID 9; FLOAT/DOUBLE/INT32/INT64/FLBA; "does not reduce the size …
  better compression ratio and speed … afterwards"), RLE_DICTIONARY ID 8: Apache Parquet spec.
- Float compression ratios (~1.04–1.20× f64; 2.2×/773× int index; 2.7× CSV bloat; FP32 2×):
  **measured locally this session** (numpy + zlib + BSS transform).
- FP64 entropy ceiling ~1.25–1.33×: FCBench arXiv:2312.10301 (corroborates the local measure).
- Pcodec 44–158% over Zstd **on numeric/time-series benchmarks**: arXiv:2502.06112 (the
  caveat — not f-stats — is part of the verified finding).
- DuckDB projection + filter pushdown (zonemaps) + glob, out-of-core: DuckDB docs (verbatim).
- Parquet `memory_map` "won't help much with resident memory": arrow.apache.org/docs/python/parquet.html.
- steppe code: `f4.hpp:46-60`, `f4.cpp:75-134`, `result_emit.cpp:25,465-581`,
  `f2_disk_format.hpp:21-55`, `block_sink.cuh:46,95,125,206`, `tier_select.hpp:33-38`,
  `module.cpp:229,720`. No Arrow/Parquet/Zstd dependency exists today (grep empty).

**SPECULATIVE / NOT VERIFIED (needs a steppe-specific benchmark or a decision):**
- The **actual** compression ratio of *real AADR* `est/se/z/p` under BSS+Zstd or Pcodec —
  my numbers are on a defensible synthetic model; must benchmark real output before quoting
  a steppe ratio (per the real-data-only memory rule).
- The **real** `|z|>t` survivor fraction on AADR (the 380×/400000× are null-background
  estimates; real data has signal → somewhat higher).
- Whether f-stat floats have enough cross-quartet correlation (shared pops) for Pco to beat
  BSS+Zstd — plausible, unverified.
- That FP32 / mantissa-truncation is numerically safe (jackknife SE carries few sig figs —
  very likely, but a numerics-policy sign-off is required; steppe's emulated-FP64 policy is
  strict, so lossy storage is a decision, not a default).
- Write **throughput** of Parquet (esp. Zstd-9/BSS, CPU-heavy) vs raw-binary vs CSV at GPU
  emit rates — the emitter could become the bottleneck; not measured.
- cuDF/kvikIO Parquet-to-GPU read-back — general knowledge, not re-verified this session.
- mmap-positional keyless scheme — feasible only for fixed-enumeration-order; not designed here.

---

## 6. Open decisions (the user must choose before any implementation)

1. **Default policy: filter-first vs full-dump.** Recommend filter-first (`--min-abs-z` /
   `--top-k`) as the default, full unfiltered dump opt-in + size-gated. Confirm.
2. **Safety cap value.** Adopt an AT2-2-style `maxcomb`? What threshold (1e6? higher, since
   steppe *can* compute more)? What override flag spelling?
3. **On-disk format.** Parquet (interop, pushdown, ecosystem dependency: Arrow/Parquet C++) vs
   a steppe-native binary shard format (reusing `f2_disk_format` discipline, zero new
   dependency, but no ecosystem read-back) vs Arrow IPC/Feather (zero-copy mmap). Recommend
   Parquet for read-back, but it adds a build dependency steppe does not have today.
4. **Drop `z`/`p` on disk and recompute on read?** Halves numeric payload; costs a derive step
   and a tiny convention lock-in.
5. **Lossy FP32 / mantissa-truncated storage mode?** Big size win (2–2.8×) but a numerics
   sign-off against the parity tolerance is required. Opt-in only.
6. **Sharding key.** By `pop1`? By chromosome/region? By significance bucket (best for
   pushdown)? Affects multi-GPU write parallelism and read pruning.
7. **Read-back surface.** Ship Parquet + document DuckDB/pyarrow query, or also wrap a
   `steppe.read_fstats(path, filter=...)` helper? Confirm `to_dataframe()` on a sweep is a
   non-goal.

**Honest thin spots:** no pop-gen precedent for a queryable f-stat store (novel for steppe);
all compression/survivor ratios need a real-AADR benchmark before being quoted to users; write
throughput at GPU emit rate is unmeasured and could move the format decision.
