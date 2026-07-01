# Error Handling & Robustness — First-Principles Assessment

This is a read-only, first-principles review of how `steppe` surfaces, propagates, and recovers from failures. The scope is the code that exists today, with line-level references.

## 1. Error taxonomy (Status enum, Error struct, exceptions)

### What exists

The public error vocabulary is intentionally small and lives in one place:

- `include/steppe/error.hpp` defines `enum class Status` with six values:
  - `Ok`, `DeviceOom`, `RankDeficient`, `NonSpdCovariance`, `ChisqUndefined`, `InvalidConfig`.
  - The three **domain outcomes** (`RankDeficient`, `NonSpdCovariance`, `ChisqUndefined`) are deliberately surfaced as ordinary per-model values, not faults.
  - The taxonomy matches architecture.md §10, except the richer C enum `steppe_status_t` / internal `Error` struct described there is not yet present; the C++ code uses only `Status`.

The actual propagation mechanisms are three different things depending on layer:

| Layer | Mechanism | File / Line |
|---|---|---|
| Config build | `BuildResult<T>` (C++20 stand-in for `std::expected<T, Status>`) | `src/core/config/build_result.hpp:36` |
| CLI dispatch | `std::optional<RunConfig>` + `error_message()` string | `src/core/config/config_builder.hpp:120`, `src/app/cli_parse.cpp:60` |
| I/O read paths | ad-hoc `{ bool ok; Status status; std::string error; ... }` structs | `src/app/f2_dir_io.hpp:50`, `src/app/pop_resolver.hpp:26` |
| CUDA runtime | Typed exceptions (`CudaError`, `CublasError`, `CusolverError`, `CufftError`) | `src/device/cuda/check.cuh:55` |
| qpAdm fit | `Status` field inside result structs (`QpAdmResult.status`, `F4Result.status`, etc.) | `src/core/qpadm/qpadm_fit.cpp:82-84`, `src/app/result_emit.cpp:45` |

### Gaps

- There is **no single internal `Error` struct** carrying `{category, status, message}` as architecture.md §10 prescribes. The C++ side strings together ad-hoc `error` fields, `error_message_` in the builder, and exception `what()` strings.
- The `Status` enum omits the architecture §10 categories `STEPPE_ERR_IO_FORMAT` and `STEPPE_ERR_CUDA_RUNTIME`; those are currently represented as `std::runtime_error` strings or as `InvalidConfig`/`kExitRuntimeError` at the CLI.
- `exit_code_for(Status)` (`src/core/config/exit_code.hpp:54`) handles only the six existing enum values and maps any unknown future value to `kExitRuntimeError`. That is fine, but the mapping from I/O faults to `InvalidConfig` vs. `kExitIoError` is done by hand in every command rather than by a centralized status category.

## 2. Consistency of error propagation (Status vs exceptions vs asserts)

### What exists

The project has a clear, documented rule:

- **Domain outcomes** ride in result structs as `Status` values and do **not** throw.
- **Faults** (config, I/O, CUDA runtime, OOM) are fail-fast and throw or return `false`+reason.
- **Asserts** (`STEPPE_ASSERT`) are debug-only precondition checks (`src/core/internal/host_device.hpp:80`).

This distinction is largely honored:

- `qpadm_fit.cpp` returns `Status::NonSpdCovariance` / `RankDeficient` as values at lines 82–92.
- `config_builder.cpp` returns `unexpected(Status::InvalidConfig)` with a message at line 213.
- `check.cuh` throws typed exceptions for CUDA faults, never calls `std::exit`.
- `STEPPE_CUDA_CHECK_KERNEL()` after every kernel launch (`f2_block_kernel.cu:359`, `cuda_backend.cu` passim).

### Gaps

- **Inconsistent I/O error representation.** `read_f2_dir` returns a `F2DirResult` with embedded `Status::InvalidConfig` (`src/app/f2_dir_io.cpp:26`), but `read_snp` throws `std::runtime_error` (`src/io/snp_reader.cpp:143`), and `count_text_records` silently returns `0` on failure (`src/io/geno_reader.cpp:38`). A caller cannot distinguish "file not found" from "malformed record" without parsing English strings.
- **Inconsistent config validation timing.** Some checks run in `ConfigBuilder::build()` (e.g. `--device` parsing, `config_builder.cpp:227`), but some validation is duplicated or delayed in the command files:
  - `cmd_qpadm.cpp:46` checks `target.empty()`.
  - `cmd_extract_f2.cpp:135` checks genotype triple emptiness.
  - `cmd_f4.cpp:109` checks `--f2-dir` emptiness.
  These are reasonable command-specific checks, but there is no central place that guarantees a `RunConfig` is complete and valid after `build()`.
- **Exception translation is manual.** Every command wraps `build_resources`/`upload_f2_blocks_to_device`/`run_*` in its own `try/catch (const std::exception&)`, prints `e.what()`, and returns `kExitRuntimeError`. There is no shared `steppe_status_t steppe_run(...)` boundary doing the mapping.
- **`std::runtime_error` is overloaded.** It is used for:
  - CUDA-fault translation in `cmd_qpadm.cpp:118`.
  - "not implemented" backend sentinel in `qpadm_fit.cpp:187`.
  - Data-format errors in `snp_reader.cpp:131`.
  - Model-index invariant violations in `model_search.cpp:209`.
  This makes it impossible for a caller to distinguish recoverable from unrecoverable failures without string inspection.

## 3. Validation boundaries (build-time vs runtime)

### What exists

The architecture is explicit: `ConfigBuilder::build()` validates once and freezes into an immutable `RunConfig` (architecture.md §9). The builder does a good job on static validation:

- Token legality (`--device`, `--precision`, `--tier`, `--format`) at `config_builder.cpp:227–317`.
- Numeric ranges (`--fudge >= 0`, `--rank-alpha` in `(0,1)`, etc.) at lines 319–521.
- Flag conflicts (`--min-z`/`--top-k` mutual exclusion, pop-selection modes, `min_sources`/`max_sources`) at lines 398–466.

### Gaps

- **No live VRAM / precision honorability validation in `build()`.** The header and architecture §9 say `build()` should include the device-memory budget, but `config_builder.cpp:209` explicitly states: "the LIVE device/VRAM probe is the app's job through build_resources". That splits the validation boundary: `build()` is pure static, and runtime failures (OOM, unsupported precision) surface later as `kExitRuntimeError`. This is documented but means a user can pass a config that `build()` accepts and then fail mid-run.
- **No semantic validation of file existence at build time.** `--f2-dir`, `--prefix`, `--out`, etc. are strings. The builder does not check that the referenced paths exist or are writable. That is left to each command's runtime, so error messages vary.
- **Model-level validation (pop names, quartet shapes) happens in commands, not in `RunConfig`/`ConfigBuilder`.** This is architecturally correct (names are an app concern per cli-bindings.md §1), but it means there is no single preflight that catches all user errors before GPU work begins.

## 4. I/O error handling (disk full, permissions, missing files)

### What exists

The project correctly avoids `printf` in library code and routes errors to the app for printing (architecture.md §10). Some positive patterns:

- `f2_dir_writer.cpp` checks `std::ofstream` state after each write and reports "disk full?" explicitly (lines 429, 439, 467, 469, 511, 513).
- `f2_dir_io.cpp` checks directory existence, magic, version, dtype, shape, and truncation before reading data (lines 61–159).
- `snp_reader.cpp` validates token count, non-finite genetic positions, blank-line placement, and integer overflow with line numbers (lines 131–181).
- `sha256_file` returns an empty string on failure (`f2_dir_writer.cpp:358`) rather than throwing, and the caller records `""` + `source_hash_computed:false`.

### Gaps

- **No `std::error_code` plumbing for filesystem errors.** `f2_dir_io.cpp:62` captures `std::error_code` but then returns a generic `InvalidConfig` message without forwarding `ec.message()`. On `EACCES`/`ENOSPC` the user sees "--f2-dir is not a directory" instead of "Permission denied".
- **Disk-full detection is best-effort.** `std::ofstream::operator bool` after `write` may not surface a short write until flush/close. The code does check `!o` after the sequence (good), but there is no explicit `flush()` before the final check, and `std::ofstream` destructors can swallow errors.
- **Output directory creation races.** `f2_dir_writer.cpp:405` uses `fs::create_directories(dir, ec)` and checks `ec`, which is correct. However, the command-level check for `--out` emptiness happens before the writer runs, so the error path is two hops away.
- **No atomic / transactional writes.** `f2.bin`, `pops.txt`, and `meta.json` are written independently. A crash after `f2.bin` but before `meta.json` leaves a half-valid cache directory that `read_f2_dir` will accept (it does not require `meta.json`). This is documented as intentional (`f2_dir_io.hpp:17`) but is a robustness trade-off.
- **`std::printf` in `cmd_extract_f2.cpp:246` and `cli_parse.cpp:754` is app-level and therefore allowed by architecture §10, but it bypasses the `STEPPE_LOG_*` facade that the rest of the library is supposed to use.**

## 5. CUDA error handling and recovery

### What exists

This is the strongest area of the codebase:

- **Single home for CUDA checks.** `src/device/cuda/check.cuh` defines `STEPPE_CUDA_CHECK`, `STEPPE_CUDA_WARN`, `CUBLAS_CHECK`, `CUSOLVER_CHECK`, `CUFFT_CHECK`, and `STEPPE_CUDA_CHECK_KERNEL()`.
- **Typed exceptions with source location.** `CudaError`, `CublasError`, `CusolverError`, `CufftError` carry `file:line:function` and the failing expression (`check.cuh:55–202`).
- **OOM is distinguished.** `DeviceBuffer` throws `CudaError(cudaErrorMemoryAllocation, ...)` on size overflow (`device_buffer.cuh:70`), and `STEPPE_CUDA_CHECK` would surface `cudaErrorMemoryAllocation` from `cudaMalloc`. The public boundary is supposed to map that to `STEPPE_ERR_DEVICE_OOM` (architecture §10); today the CLI maps all exceptions to `kExitRuntimeError` (5) instead of `kExitDeviceOom` (3).
- **Capability degrade is non-throwing.** `STEPPE_CUDA_WARN` (`check.cuh:292`) returns the status and logs once, used for P2P probes / recoverable states.
- **Post-launch kernel checks.** `STEPPE_CUDA_CHECK_KERNEL()` calls `cudaGetLastError()` and, in debug, `cudaDeviceSynchronize()` (`check.cuh:325`).
- **RAII teardown never throws.** Every wrapper (`DeviceBuffer`, `Stream`, `Event`, `CublasHandle`, `CusolverDnHandle`, `CufftPlan`, `GesvdjInfo`) routes destroy failures to `STEPPE_LOG_WARN` (`device_buffer.cuh:145`, `stream.hpp:92`, `handles.hpp:211`).
- **Device-agnostic free design.** `DeviceBuffer::reset()` intentionally does not record/restore `cudaSetDevice`, relying on `cudaFree` being pointer-device-aware (`device_buffer.cuh:112`). This is well-documented but depends on an unspecified CUDA invariant.

### Gaps

- **No recovery from `cudaErrorMemoryAllocation`.** The code throws and the CLI exits. Architecture §11.2 says `build()` should reject over-budget configs up front, but because the live probe is in `build_resources`, the OOM can still happen during allocation. There is no fallback to a smaller tile/chunk or to the HostRam/Disk tier at allocation time.
- **`kExitRuntimeError` is used for CUDA faults instead of `kExitDeviceOom`.** `cmd_qpadm.cpp:118`, `cmd_extract_f2.cpp:344`, `cmd_f4.cpp:176` all catch `std::exception` and return `kExitRuntimeError`. A `cudaErrorMemoryAllocation` should map to `kExitDeviceOom` (3).
- **Asynchronous CUDA errors may not attribute to the right launch in release.** `STEPPE_CUDA_CHECK_KERNEL()` forces sync only in debug. That is a deliberate performance choice, but it means release builds rely on the next synchronous API call to surface the error, which can obscure which kernel failed.
- **No CUDA context teardown ordering guarantee.** `DeviceBuffer::reset()` treats `cudaErrorCudartUnloading` / `cudaErrorContextIsDestroyed` as benign (lines 153–154). That is reasonable, but static/global CUDA objects could still observe ordering issues if any are ever introduced.

## 6. User-input validation and helpful error messages

### What exists

The CLI error messages are generally concrete and actionable:

- `--device cpu is not supported: steppe is a GPU product...` (`config_builder.cpp:231`).
- `--precision 'X' is unknown (use emu40 | emu32 | fp64 | tf32)` (`config_builder.cpp:278`).
- `--pops contains an unknown population 'X' (not present in ...)` (`cmd_extract_f2.cpp:171`).
- `f2.bin truncated reading the f2 region (... doubles): <path>` (`f2_dir_io.cpp:121`).
- `io::read_snp: malformed genetic position "..." at line N` (`snp_reader.cpp:131`).

The builder centralizes many range/format messages, and the app prefixes them with the subcommand name.

### Gaps

- **Error-message ownership is split.** The library sets `error_message_` in the builder and strings in result structs, but the app is responsible for printing. This works, but there is no enforced schema (error code + message + context), so downstream programmatic consumers cannot parse errors reliably.
- **`--help` is printed to stdout on bare invocation, while errors go to stderr.** That is conventional, but the no-subcommand path returns `kExitOk` (`cli_parse.cpp:754`), which is fine for interactive use but may surprise scripts.
- **Some validation messages leak internal field names.** E.g. `write_f2_dir: pop_labels has N entries but f2 tensor P=M...` (`f2_dir_writer.cpp:386`). That is clear enough for a developer but less friendly than "the population list in pops.txt does not match f2.bin".
- **No suggestions for typos.** Unknown pop labels name the offending label but do not suggest near matches.
- **CLI11 validation for `--ploidy` throws `CLI::ValidationError` from a lambda (`cli_parse.cpp:714`).** That is the correct CLI11 idiom, but it is the only place in the file that throws through the option callback; all other options defer validation to `ConfigBuilder::build()`.

## 7. Invariants and contracts (asserts vs validation)

### What exists

- `STEPPE_ASSERT` is debug-only and used for cheap, hot-path preconditions (`host_device.hpp:80`). Examples:
  - `M <= INT_MAX` narrowing invariant in `f2_block_kernel.cu:384`.
  - cuBLAS/cuSOLVER device-ordinal assertions in `handles.hpp:180`.
- `STEPPE_DEBUG_ONLY` gates the forced sync in `STEPPE_CUDA_CHECK_KERNEL()` and other debug-only probes.
- Runtime validation is used when the check has user-facing meaning:
  - `CudaBackend::compute_f2` throws `std::runtime_error` for `M > INT_MAX` (`cuda_backend.cu:346`) because it is a reachable user-input scaling limit, not merely an internal invariant.
  - `scatter_into_slots` throws on out-of-range `model_index` (`model_search.cpp:209`) because it guards deterministic re-sort.

### Gaps

- **Asserts are removed in release, but some checks are safety-critical and should survive.** The `M <= INT_MAX` assert in `f2_block_kernel.cu:384` is duplicated by a throwing check upstream in `cuda_backend.cu:346`, which is good. But not every assert has a release counterpart. For example, `CublasHandle::assert_on_creation_device` is debug-only; in release a handle used on the wrong device will fail with `CUBLAS_STATUS_ARCH_MISMATCH` or silently corrupt results.
- **`STEPPE_ASSERT` macro does not support formatting.** It takes a literal string message, which is acceptable but less informative than a formatted error.
- **Contracts around `F2BlockTensor` shape invariants are checked late.** `write_f2_dir` validates `P`, `n_block`, vector sizes, and `block_sizes` length (lines 380–402), which is good. But consumers like `upload_f2_blocks_to_device` assume the host tensor is well-formed without re-validating.
- **`PopResolver::label_at` uses `.at()` (throws on OOB) (`pop_resolver.hpp:59`).** This is a contract check, but it throws `std::out_of_range` with no context, which the CLI does not translate to a user message.

## 8. Thread safety and const-correctness issues

### What exists

- `ConfigBuilder::build()` is `const` and the builder is documented as not thread-safe (`config_builder.hpp:53`). The frozen `RunConfig` is immutable and freely shareable.
- `run_qpadm_search` uses per-device `std::jthread` workers and writes results into pre-sized slots with non-overlapping indices, so no lock is needed (`model_search.cpp:290–315`).
- `std::atomic_flag` is used for one-shot capability warnings (`f2_block_kernel.cu:253`, `handles.hpp:580`).
- `DeviceBuffer` and the RAII wrappers are move-only, which prevents accidental shared mutable state.

### Gaps

- **`ConfigBuilder::error_message_` is `mutable` and returned by reference (`config_builder.hpp:120`).** A `const` object returns a non-const reference to a mutable member. While the builder is single-threaded by contract, this is a const-correctness wart: `error_message()` is declared `const noexcept` but can be observed to change after `build()` calls.
- **`F2Replication::per_device` holds borrowed pointers to objects inside `rep.owned` and `f2` (`model_search.cpp:192–247`).** The lifetimes are correct (all live inside the same scope), but the pointer indirection makes the contract easy to violate in future edits.
- **`std::thread` in `cmd_extract_f2.cpp:302` hashes the source `.geno` while the GPU pipeline runs.** The thread is joined by `ThreadJoiner` before the function exits, which is correct. However, if `sha256_file` throws inside the thread, `std::terminate` is called. The lambda does not wrap the call in a try/catch.
- **CUDA stream/handle wrappers are not thread-safe by design.** Each backend owns one stream and one handle. `run_qpadm_search` threads each use their own backend (`resources.gpus[g].backend`), so this is safe. But nothing in the type system prevents two threads from sharing one `ComputeBackend`.
- **`Stream::synchronize()` is `const` but mutates CUDA stream state (`stream.hpp:81`).** Semantically the stream state is owned by the device, but the method is non-mutating from the host-object perspective; this is a minor const-correctness choice.
- **The CLI stores one `CliArgs` per subcommand and captures references in CLI11 lambdas (`cli_parse.cpp:323–336`).** The lambdas outlive parse because the `CliArgs` objects are in stable storage. This is correct but fragile: adding a dynamically allocated subcommand later could break the reference lifetime.

## Concrete plan to make error handling A+

The following changes are ordered by impact and can be landed incrementally.

### P1 — Centralize the internal error type

1. Introduce `struct Error { Status status; std::string message; }` in a CUDA-free internal header, matching architecture.md §10.
2. Replace `BuildResult<T>` with `std::expected<T, Error>` when the project moves to C++23, or extend `BuildResult<T>` to carry a message today.
3. Make `ConfigBuilder::build()` return `Error` with context (file/flag/value) so commands stop printing a single concatenated string.
4. Unify `F2DirResult`, `ResolveResult`, and similar `{ ok, status, error }` structs behind a `Result<T>` template that carries `Error`.

### P2 — Complete the status taxonomy and exit-code mapping

1. Add `Status::IoError` and `Status::CudaRuntime` to `include/steppe/error.hpp`.
2. Update `exit_code_for(Status)` to map them to `kExitIoError` (4) and `kExitRuntimeError` (5).
3. Add a translation layer `Status exception_to_status(const std::exception&)` that inspects `CudaError::status()` and returns `DeviceOom` for `cudaErrorMemoryAllocation`, `CudaRuntime` for other CUDA errors, and `IoError` for I/O-related `std::runtime_error`s.
4. Change every command's `catch (const std::exception&)` to call `exception_to_status` and return the correct exit code.

### P3 — Move live resource validation into `ConfigBuilder::build()`

1. Add a `BackendCapabilities` argument or probe hook to `build()` so it can reject unhonorable precision / over-budget VRAM before the CLI dispatches.
2. Keep the hook CUDA-free by injecting a `std::function<BackendCapabilities(const DeviceConfig&)>` or by passing a precomputed capabilities object from the app.
3. Document the new boundary: `build()` is static + capability-aware; runtime allocation failures are still possible but become exceptional rather than expected.

### P4 — Harden I/O error handling

1. Forward `std::error_code::message()` into user-facing I/O errors.
2. Flush output streams explicitly before the final `!out` check, or switch to `std::ostreambuf_iterator`/POSIX `write` for explicit short-write detection.
3. Consider writing `f2_blocks` output to a temporary directory and atomically renaming it on success, so a crash mid-write cannot leave a half-valid cache.
4. Add a unit test that exercises ENOSPC/EACCES via `tmpfs` quotas or mocked streams.

### P5 — Strengthen asserts and release contracts

1. Audit every `STEPPE_ASSERT` and ensure a release check exists for safety-critical invariants (device ordinal, shape agreement, narrowing casts).
2. Convert the `M <= INT_MAX` assert to a release `if (...) throw` inside `run_f2_gemms` so the narrowing site is guarded even if a future caller bypasses `cuda_backend.cu`.
3. Replace `PopResolver::label_at` `.at()` with a checked accessor that returns `Status` or throws a domain-specific exception with the offending index.

### P6 — Improve user-facing diagnostics

1. Add a small `format_validation_error` helper that includes the offending value, the constraint, and (for pop names) the nearest neighbor from the valid set.
2. Ensure all error paths name the file/command that produced them consistently: `steppe <command>: <category>: <message>`.
3. Make `--help` behavior explicit in the CLI contract and document that bare invocation exits 0.

### P7 — Thread safety and lifetime

1. Make `ConfigBuilder::error_message()` return by value or `const std::string&` without `mutable` (the message should be set only in `build()`, which can be non-const internally).
2. Wrap the `geno_hash_thread` lambda in a try/catch and store any exception in a `std::exception_ptr`, then rethrow after join.
3. Add a compile-time or runtime guard that `ComputeBackend` methods are not called concurrently (e.g. a debug-only thread-id assert).
4. Replace borrowed-pointer `F2Replication` with `std::vector<std::reference_wrapper<const DeviceF2Blocks>>` or a small owning view to make the lifetime contract explicit.

### P8 — Observability

1. Extend `STEPPE_LOG_WARN` to `STEPPE_LOG_INFO`/`ERROR` as architecture §10 foresees.
2. Route app-level prints through the same facade, with a CLI sink installed in `main()`.
3. Add a small structured-log field (`[tag: ...]`) to capability warnings so CI can grep for `emu_tuning_unavailable`, `cusolver_emu_fp64_unavailable`, etc.

### Tests to add

- A config-builder unit test that asserts every `InvalidConfig` path produces a non-empty, useful message.
- A `read_f2_dir` test for each truncation/magic/version/dtype/pops-mismatch path.
- A CUDA-fault translation test that constructs `CudaError(cudaErrorMemoryAllocation)` and verifies the CLI would return `kExitDeviceOom`.
- A multi-threaded `run_qpadm_search` stress test with empty shards and out-of-range `model_index`.
- An I/O fault injection test using a custom streambuf that fails after N bytes.

## Summary grade

- **CUDA fault handling:** A- (excellent centralization, misses OOM→exit-code mapping).
- **Domain-outcome handling:** A (clearly distinguished from faults).
- **Config validation:** B+ (good static validation, live resource probe deferred).
- **I/O robustness:** B (covers common cases, lacks atomicity and fine-grained status codes).
- **Error taxonomy / propagation consistency:** B- (multiple ad-hoc mechanisms, no central `Error` type).
- **User messages:** B+ (mostly clear, no structured/programmatic form).
- **Asserts/contracts:** B (good debug discipline, some release gaps).
- **Thread/const safety:** B (correct by convention, no type-system enforcement).

Overall current grade: **B+**. Implementing P1–P3 would raise it to **A-**; P4–P8 plus the listed tests would make it **A+**.
