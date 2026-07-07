# `readv2_shard_writer.cpp` reference

## 1. Purpose

`src/app/readv2_shard_writer.cpp` is the streaming, per-pair row writer behind
`steppe readv2`. It takes the relatedness rows the GPU produces — one per
unordered pair of individuals — and lays them down on disk as they arrive, in
whichever of the two shapes the user asked for.

The whole design turns on one fact about READv2's output: there is **one row per
pair, with no filter**. For `N` individuals that is `C(N,2)` rows — for a few
thousand samples, millions of rows — and unlike the f-stat sweep (which keeps
only a bounded top-K survivor table in host memory) there is nothing to prune.
So the rows cannot be collected into a `std::vector` and written at the end; they
would not fit. This writer exists to stream every row straight to its destination
the instant it is produced, buffering nothing.

Like the rest of the `app` layer, this translation unit is CUDA-free. It touches
no GPU, launches no kernel, and computes no statistic. It is pure output
plumbing: take a finished row, serialize it, write it, and at the end close the
file cleanly and report whether the write actually landed.

---

## 2. The two modes

The writer has two constructors, one per mode, and a row is routed differently in
each.

**Single-stream mode** — `Readv2ShardWriter(std::ostream&, fmt, info)`. Every row
goes to one stream: a `--out FILE`, or standard output when no file was given.
This constructor writes the header (csv/tsv) or the JSON prologue to that stream
*immediately*, at construction, so the header lands before any row.

**Shard-directory mode** — `Readv2ShardWriter(dir, fmt, shard_stride, info)`. Rows
are partitioned across many files named `readv2.NNNNN.<ext>` inside `dir`, split
by the *first* sample's index (section 4). This constructor writes nothing at
construction — it cannot, because it does not yet know which shards will exist.
Each shard file gets its own prologue lazily, the first time a row lands in it.
The caller must have created `dir` already; the writer opens files inside it but
does not create the directory itself. Note the `shard_stride` is clamped to at
least 1 in the initializer list, so a zero or negative stride can never cause a
divide-by-zero in `append`.

A single boolean, `sharded_`, records which constructor ran, and `append` /
`finish` branch on it. Everything else — the serialization, the number
formatting, the schema — is shared between the two modes, so a row looks
byte-identical whether it went to one file or was scattered across shards.

---

## 3. Streaming, and why the header is written up front

The prologue (`write_prologue`) is written the moment a destination stream first
exists — at construction for single-stream mode, at first-row-in-shard for shard
mode. That ordering is what makes streaming possible: because the header is
already down, each row can be appended immediately with no need to remember
anything about the ones before it except a single "is this the first row?" flag.

For **csv/tsv** the prologue is just the frozen column header line
(`readv2_emit_header`), and each row is one `readv2_emit_row` line after it.
Nothing is wrapped or delimited; a row is a line.

For **JSON** the output is a single object, opened in the prologue with the
run-level facts (`n_individuals`, `n_pairs`, `norm`, `window_snps`, a
`"status":"ok"`) and an open `"rows":[` array, then each row is a `{...}` object
inside that array. Because it streams, the writer inserts the comma *before*
every row except the first — it cannot look ahead to know whether more rows
follow, so a leading comma keyed off the per-stream `first` flag is the only
comma strategy that works without buffering. `finish` closes the array and object
with `]}`.

### The `background` caveat

The JSON prologue deliberately does **not** carry the `background` value. The
background is the all-pairs median normaliser, and it is only known *after* every
pair has been reduced — but rows are streaming out during that very same pass.
Putting it in the prologue would mean buffering all `C(N,2)` rows until the pass
finished, which is exactly what this writer refuses to do. So the background is
reported elsewhere instead — in the shard manifest sidecar (section 5) and on
standard error — and the frozen per-row schema is untouched by its absence. This
is a documented, load-bearing trade-off, not an oversight.

---

## 4. Sharding by first-sample index

In shard mode, `append(row, sampleA_index)` computes the shard number as

```
shard = sampleA_index / shard_stride
```

with integer division, and the file is `readv2.<shard, zero-padded to 5>.<ext>`.
The `sampleA_index` passed in is the index of whichever sample became `sampleA`
after the caller canonicalised the pair (`sampleA < sampleB` lexically), so the
partition is stable and reproducible.

The default stride is 256 (set by the caller in `cmd_readv2.cpp`), which keeps a
few hundred first-samples' worth of pairs together in each file. The point of
sharding on the *first* sample is locality: every pair that shares a first sample
lands in the same shard, so a consumer can process one first-sample's full row of
relationships by reading a single shard rather than scanning the whole output.

Shards are created lazily and cached in a `std::map<int, Shard>`. The first time a
given shard number is seen, the writer opens its file (binary, truncating),
writes the prologue into it, and remembers the stream plus its own `first` flag.
Every later row for that shard reuses the cached stream. Each shard file is fully
self-contained — its own header/prologue, its own rows, its own closing `]}` for
JSON — so any single shard can be read on its own without the others.

If a shard file cannot be opened, the writer does **not** throw or abort mid-run.
It records the reason in `worst_error_` and returns, dropping that row (and any
later row for the same missing shard, since the shard never gets cached). The
error is surfaced at `finish` time (section 6). This keeps a single bad file from
tearing down a long run before the failure can be reported cleanly.

---

## 5. The manifest sidecar (shard mode only)

When a shard run finishes cleanly, `finish` writes one extra file,
`readv2.manifest.json`, into the shard directory. It is the run's index: the
things a downstream reader needs to interpret the scattered shard files as one
result. It records `n_individuals`, `n_pairs`, the actual `n_shards` written
(`shards_.size()`), the `shard_stride` used, `window_snps`, `norm`, the
`background` value that could not live in the row stream (section 3), and a
`schema` string spelling out the frozen column order:

```
sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z
```

The `background` reaches the manifest through `set_background`, which the caller
invokes after `run_readv2` returns (the value is only known then). Single-stream
mode writes no manifest — there is nothing to index, the one file is the whole
result — so `set_background` matters only for the shard path.

---

## 6. Finishing, verification, and I/O error reporting

`finish(cmd)` is where the output is closed and, crucially, *verified*. Writing a
row can silently fail — a full disk, a closed pipe on the reader's end, a short
write — and none of that shows up until the stream is flushed and its state
checked. So `finish` does not just close brackets; it flushes and then checks
`good()`, and turns a bad stream into an `kExitIoError` exit code with a message
on standard error. A run that streamed millions of rows into a broken pipe must
fail loudly, not exit zero.

The single-stream path: close the JSON `]}` if JSON, flush, and if the stream is
no longer good, report `write failed (short write / closed pipe)` and return the
I/O-error code; otherwise return `nullopt` (success).

The shard path, in order:

1. If a shard file failed to open earlier (`worst_error_` non-empty), report that
   stored reason now and return the I/O-error code — the deferred failure from
   section 4 surfaces here.
2. Otherwise close and flush every shard file, checking each `good()`; a failed
   shard flush is an I/O-error exit.
3. Write the manifest sidecar (section 5), checking that its open, its write, and
   its flush all succeeded — each failure is its own I/O-error exit with a
   distinct message.
4. Return `nullopt` on full success.

`finish` is marked `[[nodiscard]]` and returns `std::optional<int>`: `nullopt`
means "all writes verified, carry on", and a value is an exit code the caller
should return immediately. The caller (`cmd_readv2.cpp`) does exactly that:
`if (const auto rc = writer.finish("readv2")) return *rc;`.

---

## 7. Contracts and invariants

- **Nothing is buffered.** At most one row's worth of text is in flight at a time
  (plus the OS stream buffer). Peak host memory does not grow with the number of
  pairs — the entire reason the class exists.
- **The header/prologue always precedes the rows** in every file, because it is
  written before any row can be appended to that stream.
- **Row serialization is mode-independent.** csv/tsv rows and JSON row objects are
  produced by the shared `readv2_emit_*` helpers, so a row is identical whether it
  went to a single stream or a shard, and it matches the frozen schema exactly.
- **The shard directory must already exist**; the writer opens files in it but
  does not `mkdir` it (the caller does).
- **`shard_stride` is always at least 1** — clamped in the constructor — so the
  division in `append` is safe for any caller-supplied stride.
- **Shard files are self-contained**; each can be read without the others, and the
  manifest, not any single file, records how many there are.
- **A verified close is part of the contract.** A successful `finish` (`nullopt`)
  means every byte was flushed and every stream ended `good()`; success is never
  reported for a write that did not land.

---

## 8. Edge cases

- **A shard file that will not open** does not crash the run; the row is dropped,
  the reason is remembered, and `finish` reports it as an I/O error (section 4/6).
- **A closed pipe / full disk mid-stream** is caught at `finish` by the `good()`
  check, not silently ignored.
- **`background` genuinely unknown at prologue time** is handled by design — it is
  never promised in the streamed rows or the JSON prologue, only in the manifest
  and on stderr (section 3).
- **Zero or negative `shard_stride`** is clamped to 1 rather than rejected, so the
  worst a bad stride does is put every pair in shard 0 — never a crash.
- **A `background` never set** in shard mode simply serialises as its default
  (0.0) in the manifest; `set_background` is the caller's responsibility.
