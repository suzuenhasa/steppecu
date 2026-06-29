# Cross-cutting review: the `steppe` output/emit layer

This is a read-only cross-cutting review focused on how the CLI writes data.

## What's genuinely good

There **is** a centralized serializer for the main result types, and the commands that use it are better for it.

- **`result_emit.cpp/hpp` is the right idea.** It owns the `OutputFormat` enum, `parse_output_format`, NaN→`NA`/`null` sentinels, CSV quoting, JSON escaping, status/precision string mapping, and 17-digit double formatting for round-trip safety.

  ```cpp
  // result_emit.cpp:25-31
  [[nodiscard]] std::string fmt_double(double v) {
      if (std::isnan(v)) return "NA";
      std::ostringstream o;
      o.precision(17);
      o << v;
      return o.str();
  }
  ```

- **The `qpadm`/`qpwave`/`f4`/`f3`/`f4ratio`/`qpadm-rotate` family all route through it.** They do not inline CSV/JSON construction. `cmd_f4.cpp:193-202`, `cmd_qpadm.cpp:142-151`, `cmd_rotate.cpp:205-216`, etc.

- **The stderr/stdout split is respected.** Faults go to `stderr` via `std::fprintf(stderr, ...)`, data goes to `stdout` or `--out`. `architecture.md §10` is followed in spirit.

- **`f2_dir_writer.cpp` actually checks post-write stream state**, unlike most commands:

  ```cpp
  // f2_dir_writer.cpp:429-439
  if (!o) return fail(...);
  o.write(...);
  if (!o) return fail(Status::InvalidConfig, "write_f2_dir: f2.bin write failed (disk full?): ...");
  ```

- **Binary mode on output files** avoids CRLF translation surprises on Windows.

## What a senior developer would flag

### 1. The `--out`/stdout sink boilerplate is duplicated in every command

Every command repeats the same `if (out.empty()) std::cout else std::ofstream` block instead of using a helper:

```cpp
// cmd_qpadm.cpp:141-151
if (config.out_file().empty()) {
    emit_qpadm_result(std::cout, fmt, result, target_label, left_labels);
} else {
    std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "steppe qpadm: cannot open --out file: %s\n",
                     config.out_file().c_str());
        return cfg::kExitIoError;
    }
    emit_qpadm_result(out, fmt, result, target_label, left_labels);
}
```

The same pattern appears in `cmd_f4.cpp:192-202`, `cmd_f3.cpp:188-198`, `cmd_f4ratio.cpp:197-207`, `cmd_qpwave.cpp:139-149`, `cmd_rotate.cpp:204-216`, `cmd_dates.cpp:131-141`, `cmd_qpgraph.cpp:197-207`, `cmd_qpdstat.cpp:341-351`, and again at `cmd_qpdstat.cpp:223-233`. A senior dev would factor this into an `OutputSink` that returns an `std::ostream&` and the error code.

### 2. Post-write failures are ignored almost everywhere

The commands check **open** success but never verify that the full emission succeeded. If the disk fills, a pipe closes, or a network mount flakes after the file is opened, the command exits as if everything worked:

```cpp
// cmd_f4.cpp:195-202
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
if (!out) { ... return kExitIoError; }
emit_f4_result(out, fmt, result, l1, l2, l3, l4);
// <-- no out.good() / !out check; no flush
```

`cmd_fstat_sweep.cpp:189` calls `out.flush()`, but still doesn't check the stream afterward. Only `f2_dir_writer.cpp` and the meta.json write there (`:511-513`) check after emitting.

For `std::cout` paths there is no flush or exception-bit check either.

### 3. Number formatting is inconsistent across features

- `result_emit` uses 17 significant digits for round-trip fidelity.
- `cmd_dates.cpp:50-55` uses fixed `%.6f`.
- `cmd_qpgraph.cpp:95-97` and `cmd_fstat_sweep.cpp:57,69` use default `operator<<` formatting (whatever the current locale/precision gives you).
- `cmd_extract_f2.cpp:252` uses `%.4g` for dry-run summaries.
- `f2_dir_writer.cpp:476-507` builds `meta.json` with default `ostringstream` formatting for doubles.

So the same double can print as `0.12345678901234568`, `0.123457`, `0.123456`, or `0.1235` depending on which command emitted it. There is no project-wide `DoubleFormatter`.

### 4. CSV/JSON escaping and label quoting are not uniform

`result_emit` quotes labels:

```cpp
// result_emit.cpp:89-97
[[nodiscard]] std::string csv_quote(const std::string& s) { ... }
```

But `cmd_dates.cpp:70-73` emits raw labels with no quoting:

```cpp
os << "target" << sep << "source1" << sep << "source2" << sep ...
os << target << sep << src1 << sep << src2 << sep ...
```

`cmd_qpgraph.cpp:121-123` also emits raw labels into CSV:

```cpp
os << r.admix_from[j] << sep << r.admix_to[j] << sep << "admix" << sep << r.weight[j] << "\n";
```

`cmd_fstat_sweep.cpp:68` does the same. A population name containing a comma or quote will break these CSVs.

JSON is equally ad-hoc:

```cpp
// cmd_dates.cpp:58-60
os << "  \"target\": \"" << target << "\",\n";
os << "  \"source1\": \"" << src1 << "\",\n";
os << "  \"source2\": \"" << src2 << "\",\n";
```

No escaping at all. `cmd_qpgraph.cpp:98` is the same.

### 5. Header-row construction is duplicated even inside `result_emit`

`emit_csv` builds four separate header literals by hand (`:160-161`, `:172-174`, `:184-185`, `:190-191`). `emit_f4_csv`, `emit_f3_csv`, and `emit_f4ratio_csv` each build their own near-identical header row. There is no `write_csv_header(os, {"col1","col2",...}, sep)` helper, so adding a column means touching multiple string-concatenation sites.

### 6. Some features bypass the shared outputter entirely

This directly confirms the user's suspicion.

| Feature | Emitter location | Uses `result_emit`? |
|---|---|---|
| `qpadm`/`qpwave`/`f4`/`f3`/`f4ratio`/`qpadm-rotate` | `result_emit.cpp` | Yes |
| `dates` | `cmd_dates.cpp:48-74` | **No** |
| `qpgraph` | `cmd_qpgraph.cpp:82-129` | **No** |
| `qpgraph-search` | `cmd_qpgraph.cpp:216-248` | **No** |
| `f4-sweep`/`f3-sweep` | `cmd_fstat_sweep.cpp:44-71` | **No** |
| `extract-f2` dry-run/summary | `cmd_extract_f2.cpp:246-284,406-421` | **No** (different concern) |
| `qpfstats` summary | `cmd_qpfstats.cpp:112-115` | **No** |

`dates` is especially jarring because it reimplements NaN handling, CSV header construction, JSON string construction, and status mapping from scratch rather than calling `result_emit::fmt_double`/`status_str`.

### 7. File-path handling is inconsistent

`f2_dir_writer.cpp:404-409` and `cmd_fstat_sweep.cpp:169-181` use `std::filesystem::create_directories` with an `std::error_code`. Good.

But every `--out FILE` path is passed as a plain `std::string`:

```cpp
// cmd_f4.cpp:195
std::ofstream out(config.out_file(), std::ios::binary | std::ios::trunc);
```

There is no `std::filesystem::path` validation, no parent-directory creation, and no warning that `trunc` will silently overwrite an existing file. If the parent directory is missing, the open fails and the user gets a generic "cannot open" message.

### 8. JSON is built by string concatenation everywhere

Even `result_emit` builds JSON by hand with `os << "{\n"`. There is no `JsonBuilder` or similar, so mistakes like missing commas, unescaped strings, or `NaN` literals are easy to introduce. `cmd_qpgraph.cpp:82-129` and `cmd_dates.cpp:56-67` are the most fragile because they do not share the `json_quote` primitive.

## The user's suspicion: "repeat the CSV/emit differently per feature"

**Verified and expanded.**

It is not universal chaos—the `qpadm`/`fstat` family is clearly meant to be centralized and mostly is. But the layer is **leaky**:

1. **Later commands bypass `result_emit`** because it is too specific to qpadm-shaped results. `dates`, `qpgraph`, and the sweep commands each rolled their own emitters with their own quoting/number/NaN rules.
2. **Even inside `result_emit`, the CSV schema is rebuilt per emitter** rather than being table-driven. The header rows are copy-pasted string literals.
3. **The output *sink* (file vs stdout, open, flush, error check) is copy-pasted per command**, not abstracted.

So the repeat is real in two dimensions: some features repeat the whole emitter, and the shared emitter repeats header/schema construction.

## Slop test

**Not slop.** Slop would be magic numbers, no error checking, copy-pasted code with stale comments, or obviously wrong formatting. This code has:

- Deliberate NaN/Inf sentinels.
- A real `OutputFormat` abstraction.
- Structured error returns.
- Binary mode and `std::error_code` in the places that matter for big files.
- Comments explaining *why* the output shape matches AT2/golden files.

The problem is not carelessness; it is that the centralization stopped halfway. The shared pieces are good, but they were not pulled up far enough to cover newer commands.

## What it actually looks like

This looks like a project where `result_emit` was introduced as a shared serializer for the qpadm/fstat family, but later commands (`dates`, `qpgraph`, `sweep`) and dry-run output grew their own emitters because `result_emit` was not generic enough for tables, single-row summaries, or arbitrary JSON shapes. A senior reviewer would say:

> "You already built half an outputter. Finish it: one `OutputSink` for stdout/file, one `CsvRowWriter` for header + quoting + separators, one `JsonWriter` for escaping and nulls, and one `DoubleFormatter` for the whole project. Then force every command through it."

## Verdict

**B-/B.**

The emit layer is **competent and centralized where it counts**, but the cross-cutting output contract is **not enforced project-wide**. The commands that bypass `result_emit` introduce real inconsistencies in number formatting, CSV quoting, JSON escaping, and NaN handling. The duplicated `--out`/`stdout` boilerplate and the lack of post-write failure checks are the kind of technical debt that will quietly corrupt output files on full disks.

**Recommended next step:** Add a small `app/output_sink.hpp` plus a `TableEmitter`/`JsonEmitter` used by every command, and retrofit `dates`, `qpgraph`, and `fstat_sweep` to use it. That would turn this into a solid A-.
