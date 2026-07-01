# I/O, CLI & Output UX Review — `steppe`

**Scope:** `src/io/*.{cpp,hpp}`, `src/app/cmd_*.cpp`, `src/app/cli_parse.{cpp,hpp}`, `src/app/result_emit.{cpp,hpp}`, `src/app/f2_dir_writer.{cpp,hpp}`, `src/app/f2_dir_io.cpp`, and `docs/kimireview/cross_cutting_output_layer_review.md`.

**Method:** First-principles read-only assessment. No source files were modified.

---

## 1. Input format support and auto-detection

### What is already strong

- `GenoReader` implements a real multi-format probe chain in one constructor (`src/io/geno_reader.cpp:244-330`):
  1. Reads the first `kGenoHeaderBytes` and tries `parse_geno_header` for packed `TGENO`/`GENO` (`src/io/eigenstrat_format.cpp:21-130`).
  2. Falls back to a full ASCII scan for EIGENSTRAT (`src/io/geno_reader.cpp:104-149`).
  3. Checks the PLINK `.bed` magic `0x6c 0x1b 0x01` and derives geometry from `.bim`/`.fam` line counts (`src/io/geno_reader.cpp:285-309`).
  4. Probes one line for ANCESTRYMAP triples and derives geometry from `.snp`/`.ind` counts (`src/io/geno_reader.cpp:181-240`).
- `resolve_genotype_triple` centralizes the “which sibling files exist?” decision for the `--prefix` path: `.geno` wins over `.bed` (`src/io/genotype_source.cpp:18-37`).
- `read_snp_table` / `read_ind_partition` dispatch the parser based on the detected `GenoFormat`, so downstream commands do not re-spell `.snp` vs `.bim` (`src/io/genotype_source.cpp:39-50`).
- Per-sample ploidy auto-detection (`src/io/ploidy_detect.cpp:16-45`) is cleanly isolated and scans the same tile bytes the decoder will use.

### What a senior engineer would flag

- **No user override.** There is no `--geno-format` / `--format` input flag. If both `prefix.geno` and `prefix.bed` exist, `.geno` wins silently (`genotype_source.cpp:25`). A user with a converted PLINK triple sitting next to an EIGENSTRAT triple has no way to force PLINK without deleting files, and gets no warning that auto-resolution happened.
- **No compressed input.** No VCF/BCF, no gzip/bgzip. For modern population-genetics pipelines that is a real gap.
- **Auto-detection is a chain of full-file scans.** The EIGENSTRAT probe scans the whole `.geno` to validate rectangularity and legal chars (`geno_reader.cpp:104-149`). That is correct but expensive; there is no fast-path “trust the `.snp` count” mode.
- **No clear format reporting to the user.** A successful run never tells the user which format was detected. When debugging a misnamed file, the user only finds out via an error.
- **ANCESTRYMAP geometry comes from sibling line counts, but the line-count helper silently ignores blank/interior issues** that the stricter `read_snp`/`read_ind` will later reject (`geno_reader.cpp:36-54` tolerates blank lines at EOF but does not validate them exhaustively).

---

## 2. Reader architecture and modularity

### What is already strong

- The `io` layer is genuinely CUDA-free and returns plain POD structs (`GenotypeTile`, `SnpMajorTile`, `SnpTable`, `IndPartition`) that cross the layer boundary without leaking implementation (`src/io/genotype_tile.hpp`, `src/io/snp_major_tile.hpp`).
- `GenoReader` is move-only and owns the file path, and per-format read methods are separate: `read_tile` for TGENO, `read_snp_major_tile` for GENO, plus text/binary variants (`src/io/geno_reader.hpp:82-194`).
- Fail-fast is pervasive: malformed records carry 1-based line numbers (`src/io/snp_reader.cpp:140-205`, `src/io/plink_reader.cpp:151-205`), non-finite genetic positions are rejected (`src/io/snp_reader.cpp:125-136`), and checked multiplication prevents silent allocation overflow (`src/io/geno_reader.cpp:423-445`).
- Exception-type contract discipline: allocation failures are translated to `std::runtime_error` so callers can catch one type (`src/io/geno_reader.cpp:463-473`).

### What a senior engineer would flag

- **The caller must know which `read_*` method to call.** The header format is known after construction (`header().format`), but there is no `GenoReader::read_canonical_tile(...)` that dispatches internally. Every app caller (now and in the future) has to branch on `GenoFormat`. That is exactly the kind of branch that drifts out of sync.
- **Massive duplication in the SNP-major readers.** `read_snp_major_tile`, `read_eigenstrat_snp_major_tile`, `read_plink_snp_major_tile`, and `read_ancestrymap_snp_major_tile` repeat the same selection/reorder build (`src/io/geno_reader.cpp:563-584`, `690-705`, `837-858`, `988-1002`), the same checked-multiply + allocation pattern, and the same tile struct setup. Only the per-byte decode differs. A single templated `read_snp_major_tile<Decoder>` would halve the file size and remove the risk of inconsistent guards.
- **EIGENSTRAT vs PLINK `.ind`/`.snp` logic is duplicated but not shared.** `read_ind` and `read_fam` share the same selection semantics but are separate files (`src/io/ind_reader.cpp`, `src/io/plink_reader.cpp`). The selection algorithm (Explicit / AutoTopK / MinN + final sort) is copy-pasted. A common `select_groups` helper would remove this.
- **`geno_reader.cpp` is 1,132 lines**, largely because the four SNP-major variants each re-implement the same boilerplate. That is a maintainability tax.

---

## 3. CLI command consistency and discoverability

### What is already strong

- CLI11 is confined to a single TU (`src/app/cli_parse.cpp`), matching the architecture rule that the library never prints (`src/app/cli_parse.hpp:16-25`).
- Shared helpers exist for global flags (`add_common_flags`: `--device`, `--precision`, `--config`, `cli_parse.cpp:80-92`), output flags (`add_output_flags`: `--out`, `--format`, `cli_parse.cpp:95-102`), and QpAdm-style options (`add_qpadm_option_flags`, `cli_parse.cpp:107-127`).
- Subcommand coverage is broad: `qpadm`, `qpwave`, `qpadm-rotate`, `f4`, `f3`, `f4-ratio`, `qpdstat`, `dates`, `qpgraph`, `qpgraph-search`, `qpfstats`, `extract-f2`, `f4-sweep`, `f3-sweep`.
- `--help` works per subcommand because of CLI11; bare `steppe` prints the top-level help (`cli_parse.cpp:747-754`).

### What a senior engineer would flag

- **`--out` means different things.** For most commands it is an output file; for `extract-f2` it is an output directory; for `qpfstats` the directory flag is renamed `--out-dir`. Users have to read help for every command to know which semantics apply. A product-level CLI should either make `--out` always a file and use `--out-dir` for directories, or document the divergence very loudly.
- **Not every command supports `--format`.** `qpfstats` and `extract-f2` have no `--format` flag because they emit directories/dry-run text. `qpgraph` and `dates` support `--format` in parsing but ignore the shared emitter (see §4). A user learning the tool will assume `--format json` works the same everywhere; it does not.
- **No global observability flags.** There is no `--quiet`, `--verbose`, `--progress`, or `--log`. Long-running commands (`extract-f2`, `qpfstats`, sweeps, `qpgraph-search`) emit either nothing or only post-hoc summaries. In a GPU product that can run for minutes, that feels raw.
- **`--prefix` is overloaded onto the same config field (`qpdstat_prefix`) for `dates`, `qpdstat`, and `qpfstats`.** That is pragmatic but confusing in the source and in docs; a user sees `--prefix` in three commands with subtly different semantics (genotypes only vs genotypes + normalized-D vs smoothing).
- **Help strings are preserved byte-identical via parameterized helpers** (`add_f2_dir_flag`, `add_right_flag`, etc.). That is fine for regression, but it means the help text is scattered and hard to keep consistent as new flags are added.
- **No command examples.** `--help` lists flags but never shows a canonical invocation like `steppe extract-f2 --prefix v66 --pops A,B,C --out ./f2`.

---

## 4. Output formatting consistency (CSV/JSON/TSV/number formatting)

### What is already strong

- `result_emit.cpp` is the right centralization point for the main result family. It owns:
  - `OutputFormat` enum and `parse_output_format` (`result_emit.cpp:605-610`).
  - 17-digit double formatting for round-trip safety (`result_emit.cpp:25-31`).
  - NaN → `"NA"` in CSV/TSV and `null` in JSON (`result_emit.cpp:25-41`).
  - CSV quoting and JSON escaping (`result_emit.cpp:89-114`).
  - Status and precision string mapping (`result_emit.cpp:45-68`).
- `qpadm`, `qpwave`, `qpadm-rotate`, `f4`, `f3`, and `f4-ratio` all route through it.
- CSV sections are labeled with `# section: NAME` comments, which makes downstream parsing predictable (`result_emit.cpp:159-191`).

### What a senior engineer would flag

This is the layer’s biggest weakness: **several commands bypass the shared emitter entirely**, and even the shared emitter builds JSON by hand.

| Command | Emitter location | Uses `result_emit`? |
|---|---|---|
| `qpadm` / `qpwave` / `f4` / `f3` / `f4-ratio` / `qpadm-rotate` | `result_emit.cpp` | Yes |
| `dates` | `cmd_dates.cpp:48-74` | **No** — uses `%.6f`, raw labels, no CSV quoting/JSON escaping |
| `qpgraph` | `cmd_qpgraph.cpp:82-129` | **No** — default `operator<<` formatting, raw labels |
| `qpgraph-search` | `cmd_qpgraph.cpp:216-248` | **No** |
| `f4-sweep` / `f3-sweep` | `cmd_fstat_sweep.cpp:44-71` | **No** — raw labels, default doubles, custom JSON |
| `extract-f2` dry-run / summary | `cmd_extract_f2.cpp:246-284,406-421` | **No** — `%.4g`, `printf` |
| `qpfstats` summary | `cmd_qpfstats.cpp:112-115` | **No** — `%.4g`, `printf` |

Concrete consequences:

- **Number formatting is inconsistent across the product.** The same double can appear as `0.12345678901234568` (`result_emit`), `0.123457` (`cmd_dates.cpp:53`), `0.123456` (default `operator<<` in `cmd_qpgraph.cpp:95-97`), or `0.1235` (`cmd_extract_f2.cpp:252`). There is no `DoubleFormatter` used everywhere.
- **CSV/JSON escaping is inconsistent.** `result_emit` quotes labels; `cmd_dates.cpp:70-73` and `cmd_qpgraph.cpp:121-123` emit raw labels. A population name containing a comma or quote will break those CSVs.
- **JSON is string-concatenated everywhere**, including `result_emit`. There is no `JsonWriter`. That makes it easy to introduce missing commas, unescaped strings, or `NaN` literals. `cmd_dates.cpp:58-65` and `cmd_qpgraph.cpp:93-114` are the most fragile because they do not even share `json_quote`.
- **Header construction is duplicated even inside `result_emit`.** `emit_csv` builds four header literals by hand (`result_emit.cpp:160-161`, `172-174`, `184-185`, `190-191`), and `emit_f4_csv`/`emit_f3_csv`/`emit_f4ratio_csv` each build their own. A `CsvRowWriter` helper would make adding a column a one-line change.

This confirms and expands the finding in `cross_cutting_output_layer_review.md`: the output layer is centralized where it counts, but the centralization stopped halfway.

---

## 5. Error messages to users

### What is already strong

- Almost every error is prefixed with `steppe <cmd>:` (e.g., `cmd_f4.cpp:110`, `cmd_f4.cpp:115`, `cmd_f4.cpp:123`).
- Config validation failures print `steppe: invalid configuration: <reason>` (`cli_parse.cpp:71-72`).
- Device faults are caught and printed with the exception `what()` (`cmd_f4.cpp:176-181`).
- `f2_dir_writer.cpp` returns detailed error reasons with paths (`f2_dir_writer.cpp:429`, `439`, `467`, `469`, `511`, `513`).
- `read_f2_dir` gives specific diagnostics: bad magic, unsupported version, truncated regions, pops.txt mismatch (`f2_dir_io.cpp:79-159`).

### What a senior engineer would flag

- **File-open errors are generic.** `cmd_f4.cpp:197-199` prints `cannot open --out file: <path>` but does not include `errno`/filesystem reason or suggest a fix. If the parent directory is missing, the user has to guess.
- **No `--help` suggestion on invalid CLI input.** A mistyped flag or missing required argument produces CLI11’s default error; there is no curated “try `steppe <cmd> --help`” hint.
- **No machine-readable error mode.** When `--format json` is requested, errors are still plain text on stderr. A polished product should optionally emit a JSON error object.
- **Some errors leak internal config field names.** For example, `qpdstat` reuses `config.qpdstat_prefix()` for `--prefix`; if validation fails, the user does not see `--prefix` in the message unless the command author remembered to write it. This is mostly handled, but it is fragile.
- **`write_f2_dir` maps write failures to `Status::InvalidConfig`** (`f2_dir_writer.hpp:79-83`, `f2_dir_writer.cpp:324-330`). That is a taxonomy mismatch: a disk-full write is an I/O error, not a configuration error. The app exit code becomes `kExitInvalidConfig` instead of `kExitIoError`.

---

## 6. File path handling and overwrite behavior

### What is already strong

- Directory outputs create parent directories with `std::error_code`: `write_f2_dir` (`f2_dir_writer.cpp:404-409`) and `--shard-dir` in sweeps (`cmd_fstat_sweep.cpp:170-176`).
- Output files are opened in binary mode (`std::ios::binary`), avoiding CRLF surprises on Windows (`cmd_f4.cpp:195`, `cmd_fstat_sweep.cpp:182`).
- `f2_dir_writer.cpp` takes `std::filesystem::path` rather than raw strings (`f2_dir_writer.hpp:97`).

### What a senior engineer would flag

- **`--out` file paths are plain `std::string`.** Every command opens `config.out_file()` directly (`cmd_f4.cpp:195`, `cmd_qpadm.cpp:144`, etc.). There is no path validation, no parent-directory creation, and no normalization.
- **Silent overwrite.** All file outputs use `std::ios::trunc` (`cmd_f4.cpp:195`). The user gets no warning that an existing file will be destroyed. A product-level tool should either require `--force` for overwrite or at least emit a clear message.
- **No atomic writes.** Output is written directly to the target path. If the process crashes or a disk fills, the user is left with a partial file. The f2 dir is written as three separate files (`f2.bin`, `pops.txt`, `meta.json`) with no atomic commit; a crash can leave a readable-but-incomplete cache.
- **Post-write failures are ignored.** Commands check `out.is_open()` but not the stream state after emission (`cmd_f4.cpp:201` — no `out.good()` or flush check). Only `f2_dir_writer.cpp` checks `if (!o)` after the final write (`f2_dir_writer.cpp:439`). This means a full disk or closed pipe can produce a silent zero-byte or truncated output. `cmd_fstat_sweep.cpp:189` calls `out.flush()` but still does not check the stream afterwards.
- **No symlink / same-file guard.** Nothing prevents `--out` from pointing at one of the input files.

---

## 7. The f2 directory format (STPF2BK1) design

### What is already strong

- The format is well documented in code: 64-byte header, `f2` slab, `vpair` slab, `block_sizes` int32 trailer, column-major `i + P*j` within each block (`src/app/f2_dir_writer.hpp:10-20`).
- The reader validates magic, version, and dtype (`f2_dir_io.cpp:83-97`), so format drift is detectable.
- REAL `vpair` values are written, enabling the future NA/drop-pair path (`f2_dir_writer.cpp:432-434`).
- Provenance is thorough: `meta.json` records version, precision, block size, filters, source paths, optional SHA-256s, and a `source_hash_computed` boolean so missing hashes are deliberate rather than suspicious (`f2_dir_writer.cpp:474-508`).
- The f2 payload is content-addressed with a SHA-256 `f2_cache_id` (`f2_dir_writer.cpp:445-446`).
- SHA-256 hashing is self-contained, streaming, and even has a SHA-NI fast path (`f2_dir_writer.cpp:60-232`), so the optional `--hash` flag is practical.

### What a senior engineer would flag

- **No schema version in `meta.json`.** The binary `f2.bin` has a version, but the JSON does not. If `meta.json` grows new fields, consumers have no clean way to know which version they are reading.
- **JSON is hand-built with `std::ostringstream`.** Filter doubles use default formatting (`f2_dir_writer.cpp:486-488`), so values like `blgsize_cm` may render with inconsistent precision. There is no shared JSON builder (same issue as §4).
- **No checksum for `pops.txt` or `meta.json`.** Only `f2.bin` is SHA-256’d. A corrupted `pops.txt` (e.g., swapped labels) silently changes results.
- **No atomic directory commit.** The three files are written independently. A crash mid-write can produce a directory that passes `read_f2_dir` but has stale metadata.
- **`meta.json` records `source_hash_computed: false` by default**, which is honest, but there is no separate field explaining *why* hashing was skipped. A future user may wonder whether the omission was intentional or a bug.
- **The binary header is 64 bytes and mostly opaque to external tooling.** That is fine for `steppe` consumers, but a product-level spec should publish the layout so third-party readers can exist.

---

## 8. Logging vs user output separation

### What is already strong

- The architecture rule is respected in spirit: errors/warnings go to stderr, data goes to stdout or `--out` (`architecture.md §10`).
- Device-unavailable messages and warnings go to stderr (`cmd_f4.cpp:166-169`, `cmd_rotate.cpp:162-168`).
- Sweep commands report summary counts to stderr, keeping stdout clean for the survivor table (`cmd_fstat_sweep.cpp:208-209`).

### What a senior engineer would flag

- **Human summaries are written to stdout, not stderr.** `qpfstats` prints `steppe qpfstats: wrote smoothed f2 dir ...` to stdout (`cmd_qpfstats.cpp:112-115`). `extract-f2` prints its post-run summary to stdout (`cmd_extract_f2.cpp:406-421`). When a user runs `steppe extract-f2 ... > out.txt`, the summary pollutes the captured data stream. These are log messages and should go to stderr, or to a dedicated `--log` file.
- **`--dry-run` output also goes to stdout.** `extract-f2 --dry-run` writes its plan to stdout (`cmd_extract_f2.cpp:246-284`). That is useful, but it should optionally be machine-readable JSON and should not be mixed with a future data stream.
- **No log levels.** There is no `--verbose` to see which format was detected, which tier was selected, or how many blocks were assigned. There is no `--quiet` to suppress even warnings.
- **No progress reporting.** Long GPU operations (`run_extract_f2`, `run_qpfstats`, sweeps, `qpgraph-search`) give no indication of progress. For a tool that can run for many minutes, that is a UX gap.
- **Help fallback uses `std::printf`** (`cli_parse.cpp:753`), which is stdout. That is acceptable for help, but it reinforces the pattern that stdout is treated as the general-purpose channel.

---

## 9. What would make the CLI/output layer feel like a polished product

Below is a concrete, prioritized plan. The goal is not cosmetic: it is to make the tool predictable, trustworthy, and pleasant for both interactive users and pipeline authors.

### A. Enforce one shared output layer (highest impact)

- Create `app/output_sink.hpp` with:
  - `class OutputSink { std::ostream& os; bool owns; }` that handles stdout vs file, creates parent directories, opens atomically to a temp file, flushes, checks stream state, and renames on success.
  - `CsvRowWriter`: header row from a vector of names, delimiter-aware, calls the shared quoting routine.
  - `JsonWriter`: a tiny RAII builder that handles commas, escaping, and `null`/`NaN` so no command ever hand-builds JSON again.
  - `DoubleFormatter`: one place that decides precision (17 digits for round-trip, configurable fixed for user-facing values).
- Retrofit `cmd_dates.cpp`, `cmd_qpgraph.cpp`, `cmd_fstat_sweep.cpp`, `cmd_extract_f2.cpp`, and `cmd_qpfstats.cpp` to use these primitives.
- Make every command that emits data call `emit_with_sink(config.out_file(), emitter)` so post-write errors cannot be missed.

### B. Fix the `--out` semantics

- For file-emitting commands, keep `--out` as a file path and create parent directories automatically.
- For directory-emitting commands, rename `--out` to `--out-dir` everywhere (`extract-f2`, `qpfstats`) and reject `--out` with a migration message. Consistency is more important than preserving the current flag name.
- Add `--force` / `--no-clobber` to control overwrite; default to warning or requiring `--force` for existing files.

### C. Add observability and progress

- Add `--quiet`, `--verbose`, and `--log <file>` global flags in `add_common_flags`.
- Route all human-readable summaries, progress, and warnings to stderr (or the log file), never stdout.
- Add progress hooks for long operations: at minimum a block counter for `extract-f2`/`qpfstats`, a topology counter for `qpgraph-search`, and an enumerated counter for sweeps.
- Print the detected input format when `--verbose` is set.

### D. Improve CLI discoverability

- Add example invocations to subcommand help text.
- Add a `steppe --list-formats` or document format detection in help.
- Emit a curated error message like `Run "steppe <cmd> --help" for usage.` on invalid config.

### E. Make auto-detection safer

- Add `--geno-format {tgeno,geno,eigenstrat,plink,ancestrymap,auto}` (default `auto`) so users can force a format.
- When auto-resolving a PLINK triple, print a verbose message: `Detected PLINK .bed/.bim/.fam for prefix ...`.
- Warn when both `.geno` and `.bed` siblings exist and `.geno` is chosen.

### F. Harden file and directory I/O

- Use atomic writes: write to `<path>.tmp.<pid>`, fsync/flush, then `std::filesystem::rename`.
- Check stream state after every emit and after flush; return a real `kExitIoError` with `errno`/message on failure.
- Add a directory manifest / lockfile for f2 dirs so a partially written directory is rejected as corrupt.

### G. Improve the f2 dir format

- Add a `meta_schema_version` field to `meta.json`.
- Add a SHA-256 of `pops.txt` to `meta.json` so label corruption is detectable.
- Optionally publish the binary layout in `docs/design/f2-blocks-format.md` for external consumers.

### H. Unify error output

- Add `--format json` support for errors: when the user requests JSON, emit a structured error object on stderr (or a separate `--error-file`) with fields `command`, `exit_code`, `message`, `path`, `line`.
- Stop mapping I/O write failures to `InvalidConfig`; use `IoError` consistently.

### I. Reduce reader duplication

- Introduce `GenoReader::read_canonical_tile(const IndPartition&, std::size_t snp_begin, std::size_t snp_end)` that internally dispatches on `header().format`.
- Refactor the four SNP-major readers into one templated core plus format-specific byte decoders.
- Share the EIGENSTRAT/PLINK `.ind`/`.snp` selection algorithm in a common helper.

---

## Verdict

The I/O layer is **solid at the bottom and leaky at the top**. The readers are careful, fail-fast, and well-layered; the f2 directory format is well designed; and `result_emit.cpp` is the right shared abstraction for the main result family. The CLI is functional and broad.

Where it falls short of a polished product is **consistency**: several commands bypass the shared emitter, number formatting differs by command, file outputs silently overwrite and ignore post-write failures, and human summaries leak into the data stream. These are not slop — the code is clearly intentional — but they are the kind of inconsistencies that make a tool feel like a collection of scripts rather than a single product.

**Current grade: B.** With the plan above — especially a unified `OutputSink` + `CsvRowWriter`/`JsonWriter` forced on every command, atomic writes, and stderr-only logging — this would become a solid **A** output layer.
