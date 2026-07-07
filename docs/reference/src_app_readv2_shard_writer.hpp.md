# `readv2_shard_writer.hpp` reference

## 1. Purpose

`src/app/readv2_shard_writer.hpp` declares `Readv2ShardWriter` — the little
component that gets READv2's per-pair results out of memory and onto disk as they
are produced. It is the streaming output sink for the `steppe readv2` command. The
declaration lives here; the behavior is implemented in the sibling
`readv2_shard_writer.cpp`, and this one doc covers both, since the header is just
the public face of that implementation.

The whole thing exists because of one awkward fact about READv2's output shape.
Unlike the f-stat sweep — which runs a top-K filter and so only ever keeps a bounded
survivor table in host memory — READv2 emits **one row per unordered pair with no
filter at all**. That's `C(N, 2)` rows. For a run of a few thousand individuals
that number is in the millions, and buffering them all into a `std::vector<Row>`
before writing would blow up host memory at exactly the scale people care about. So
this writer never buffers: every row streams to its destination the instant it is
produced.

Like the rest of the app layer, this translation unit is CUDA-free. It touches no
GPU, launches no kernel, and computes no statistic — it only knows how to format a
finished row and put it somewhere. The numbers themselves arrive already computed in
a `Readv2OutRow` (see `readv2_emit.hpp`).

---

## 2. Two output modes

A single `Readv2ShardWriter` is either a **single-stream** writer or a **shard-
directory** writer, chosen by which constructor you call, and it stays that way for
its whole life.

- **Single-stream mode** — `Readv2ShardWriter(std::ostream& os, fmt, info)`. Every
  row goes to one `std::ostream`: a file behind `--out FILE`, or standard output.
  The header (for csv/tsv) or the JSON prologue is written to that stream *right
  away*, in the constructor, before any row arrives.
- **Shard-directory mode** — `Readv2ShardWriter(std::string dir, fmt, shard_stride,
  info)`. Rows are partitioned across many files named `readv2.NNNNN.<ext>` inside
  `dir`, and a `readv2.manifest.json` sidecar is written at the end. This is the
  mode `--shard-dir` selects, and it exists so a very large run's output can be
  produced (and later consumed) in independent pieces rather than one enormous file.

The `sharded_` flag records which mode is active; almost every method branches on it
once at the top.

### The manifest struct

`Readv2Manifest` is the small bag of run-level facts recorded in the JSON prologue
and the shard manifest: `n_individuals`, `n_pairs`, the `background` value, the
`norm` label (default `"median"`), and `window_snps`. It describes the *run*, not
any one row, so it is passed in once at construction and stamped into the prologue
and the sidecar.

---

## 3. What gets written, and when (the prologue)

Both modes open every output file with a **prologue** via the shared
`write_prologue` helper, so a single-stream file and each shard file are all self-
contained and openable on their own:

- **csv/tsv** — the frozen header line from `readv2_emit_header`: the exact column
  order `sampleA sampleB n_windows n_overlap_sites P0_mean P0_norm degree z`.
- **json** — a prologue object that opens the run metadata and then an unterminated
  `"rows":[` array, ready for row objects to be streamed in.

The single-stream writer emits its prologue in the constructor. A shard writer can't
do that — it doesn't know which shards it will need until rows arrive — so it emits
each shard's prologue lazily, the first time a row lands in that shard (section 5).

### The one thing the streamed prologue leaves out

The JSON prologue deliberately does **not** carry `background`. That value is the
all-pairs median, and it is only known *after* every pair has been reduced — but
rows stream out *during* that same reduction pass. Putting `background` in the
prologue would therefore force the writer to hold back all `C(N, 2)` rows until the
median settled, which is exactly the buffering this component exists to avoid. So
the frozen row schema stays untouched, and `background` is reported two other ways
instead: in the shard manifest sidecar, and on stderr. `set_background()` is the
setter the caller uses to hand that value over once it's known, purely so the
manifest can record it.

---

## 4. Appending a row

`append(row, sampleA_index)` is the hot path — called once per pair. `row` is the
already-formatted, already-canonicalized (`sampleA < sampleB`) output row.
`sampleA_index` is the index of whichever sample became `sampleA`, and it is used
only to decide which shard the row belongs to.

In single-stream mode `append` just writes the row straight to the one stream and
returns. All the interesting logic is in shard mode.

The actual formatting is factored into `write_row_to(os, first, row)`, which both
modes share:

- **json** — writes a comma before every row *except the first* into that stream
  (tracked by the `first` flag), then the row object. This is what keeps each file's
  `rows` array well-formed while streaming, without knowing the row count up front.
- **csv/tsv** — just writes the row; no separator bookkeeping needed.

After writing, `first` is flipped to `false`. Each destination — the single stream,
or each shard — owns its own `first` flag, so every file's JSON array is comma-
correct independently.

---

## 5. How sharding works

The shard for a row is a plain integer division:

```
shard = sampleA_index / shard_stride
```

So all pairs whose `sampleA` falls in the same `shard_stride`-wide band of sample
indices land in the same file. The `readv2` command currently uses a `shard_stride`
of 256, so shard 0 holds sampleA indices 0–255, shard 1 holds 256–511, and so on.
(As a safety belt, a non-positive stride passed to the constructor is clamped up to
1, so the division can never divide by zero.)

Shards are created **lazily and on demand**. The writer keeps a `std::map<int,
Shard>` keyed by shard number. When a row's shard isn't in the map yet, `append`:

1. Formats the file name `readv2.NNNNN.<ext>` (five zero-padded digits) in `dir_`.
2. Opens it truncating, in binary mode.
3. Writes that shard's prologue (header or JSON prologue).
4. Inserts the open stream into the map with a fresh `first = true`.

Only shards that actually receive a row are ever created, so a sparse run doesn't
litter the directory with empty files. Using a `std::map` (rather than a hash map)
also means the shards iterate in ascending numeric order at `finish()` time, which
keeps the manifest and the on-disk file set tidy and deterministic.

Each `Shard` is `{ std::unique_ptr<std::ostream> os; bool first; }` — the stream is
owned by the map and closed when the writer is destroyed, and `first` is that
shard's private JSON-comma flag.

---

## 6. Errors don't throw — they're remembered

Shard mode has to open files while it runs, and a file open can fail (a full disk, a
permission problem). Rather than throw from the middle of the streaming loop,
`append` records the *last* such failure in `worst_error_` and returns quietly. The
run keeps going; the failure is reported once, later, at `finish()`. This keeps the
per-row hot path branch-light and defers all the error reporting to one place.

Note the header comment says the shard `dir` "must already exist." The `readv2`
command satisfies that by calling `create_directories` on `--shard-dir` before
constructing the writer, so a missing directory is caught there with its own
message; by the time this writer opens a shard file, the directory is expected to be
present, and a failure to open a file inside it is treated as the I/O error above.

---

## 7. `finish()` — close out, verify, and report

`finish(cmd)` is the mandatory closing call. It returns an **exit code on failure**,
or `std::nullopt` on success — the `[[nodiscard]]` on it is there so a caller can't
accidentally ignore a write failure. `cmd` is just the command name (`"readv2"`)
woven into any error message.

### Single-stream

It closes the JSON arrays (`]}` plus a newline) if the format is JSON, flushes, and
then checks `stream_->good()`. That post-flush check is the whole point: a streamed
write can silently fail on a short write or a closed pipe (think `steppe readv2ss |
head`), and this is where that gets caught and turned into an I/O-error exit instead
of a false success.

### Shard mode

1. If a shard file failed to open during the run, the remembered `worst_error_` is
   printed now and `finish` returns the I/O error — no partial manifest is written.
2. Otherwise it walks the shards in order, closes each one's JSON array, flushes,
   and checks `good()`; any bad shard is an I/O error.
3. It writes the `readv2.manifest.json` sidecar — a single JSON object recording
   `n_individuals`, `n_pairs`, `n_shards`, `shard_stride`, `window_snps`, `norm`,
   the `background` handed in via `set_background`, and the frozen `schema` string.
   The manifest is flushed and `good()`-checked too.

Any failure at any step short-circuits to the I/O-error exit with a message on
stderr; a clean run returns `std::nullopt`.

---

## 8. Contracts and invariants

- **Mode is fixed at construction.** A writer is single-stream or sharded for life;
  the constructor you pick decides it.
- **No row is ever buffered.** Rows are formatted and written on each `append`. Host
  memory stays flat in the number of pairs — that is the reason this class exists.
- **Every output file is self-contained.** Each stream (the single stream, or each
  shard) gets its own prologue and, in JSON, its own comma-tracked, properly closed
  `rows` array.
- **`finish()` must be called exactly once, and its result must be honored.** It is
  the only place a streamed write failure is detected and the only place the shard
  manifest is written. Skipping it means an unterminated JSON file and no manifest;
  ignoring its return means missing a real I/O error.
- **The frozen row schema is never widened here.** Run-level facts that can't fit the
  per-row streaming model (notably `background`) live in the prologue and manifest,
  never as extra row columns.
- **App-layer, CUDA-free.** No GPU code, no kernels — just formatting and I/O.

---

## 9. Edge cases

- **A closed pipe / short write** on the single stream is caught by the post-flush
  `good()` check in `finish` and surfaced as an I/O-error exit, not a silent success.
- **A shard file that won't open** doesn't throw and doesn't abort the run; the last
  such failure is remembered and reported at `finish`, which then declines to write a
  manifest that would misrepresent an incomplete output.
- **A non-positive `shard_stride`** is clamped to 1, so sharding can't divide by zero
  — every row just lands in shard 0.
- **A sparse run** creates only the shards that actually receive rows; unused shard
  files are never opened.
- **`background` reported before it exists.** Because the median is only known after
  the streaming pass, the writer accepts it after the fact through `set_background`
  and records it only in the manifest/stderr, keeping the streamed rows untouched.
