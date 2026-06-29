# Flow & Cognitive Coupling Assessment — `steppe`

**Scope:** first-principles read-only trace of five end-to-end flows in the C++/CUDA
`steppe` codebase, plus the cross-cutting patterns that make the code feel like it
"bounces around".  Sources inspected include `src/app/main.cpp`,
`src/app/cli_parse.cpp`, the `cmd_*.cpp` files, `src/core/qpadm/*`,
`src/core/fstats/*`, `src/device/backend.hpp`, `src/device/cuda/cuda_backend.cu`,
`src/device/cuda/*_kernel.cu`, `src/io/*`, `docs/architecture.md` §4, and the
previous review notes `01_architecture_and_layering.md` and
`05_io_cli_and_output_ux.md`.

---

## 1. Executive summary

The codebase is **functionally coherent but not organically flowing**.  The
layering is *structurally* enforced (CUDA is private to `steppe_device`, the seam
is CUDA-free, no global state), but the control flow still ricochets between
layers because several pieces of orchestration live in the wrong layer, the
`ComputeBackend` seam has become a monolith, and the CLI commands duplicate the
same five-step ballet (`read_f2_dir`, `build_resources`, `upload`, `run_*`,
`emit`).

The overall pattern is: **thin-ish commands that are still too thick**, a
**library layer (`core`) that does I/O it promised not to**, and a **device seam
that every feature has to grow**.  A reader trying to follow a single command
must hold 8–12 files and three conceptual layers in mind at once.  The code is
careful, well-commented, and parity-obsessed, but the *flow* earns a **C+**:
works, layer-correct on the surface, yet fights the reader rather than carrying
them.

---

## 2. Flow-by-flow trace

### 2.1 Simplest happy path: `steppe f4`

**Path:**

1. `src/app/main.cpp:16-25` — `main()` catches everything and maps to
   `kExitRuntimeError`; all real work is delegated to `run_cli()`.
2. `src/app/cli_parse.cpp:484-487` — the `f4` subcommand is bound; its callback
   calls `build_config()` then `std::exit(run_f4_command(*config))`.
3. `src/app/cmd_f4.cpp:99-207` — the command body:
   * checks sweep mode (`cmd_f4.cpp:104`),
   * validates `--f2-dir` (`cmd_f4.cpp:109-117`),
   * reads the f2 dir via `read_f2_dir()` (`f2_dir_io.cpp:57`),
   * resolves pop names via `PopResolver` (`cmd_f4.cpp:127-155`),
   * calls `device::build_resources()` + `upload_f2_blocks_to_device()` +
     `steppe::run_f4()` in one big `try` block (`cmd_f4.cpp:163-182`),
   * parses `--format` and opens `--out` (`cmd_f4.cpp:185-202`),
   * emits via `emit_f4_result()` (`result_emit.cpp:652`).
4. `src/core/qpadm/f4.cpp:158-170` — public `run_f4()` overloads pick the primary
   backend (`f4.cpp:43-45`) and call `run_f4_impl()`.
5. `src/core/qpadm/f4.cpp:53-143` — `run_f4_impl()`:
   * flattens quartets,
   * calls `core::qpadm::assemble_f4_quartets()` (`f4.cpp:94`),
   * calls `core::qpadm::jackknife_diag()` (`f4.cpp:119-120`),
   * computes `est/se/z/p`.
6. `src/device/backend.hpp:1145-1163` — `assemble_f4_quartets` virtual is
   declared; the CUDA override lives in `cuda_backend.cu:3163-3221` and calls
   `launch_assemble_f4_quartets_gather()` in `qpadm_fit_kernels.cu:1928`.
7. `src/app/result_emit.cpp:482-516` / `:652-669` — CSV/JSON emission.

**Layer bouncing:**

* The command does **name resolution and file I/O**, then calls the library, then
  does **output formatting**.  That split is correct, but the command is still
  ~210 lines because it owns the full orchestration.
* `src/core/qpadm/f4.cpp` is a clean orchestrator, yet it has to know that
  `assemble_f4_quartets` and `jackknife_diag` are the right two seams.  That
  knowledge is duplicated for f3, f4-ratio, qpwave, qpadm.
* `ComputeBackend` declares `assemble_f4_quartets` in the *device* header
  (`backend.hpp:1145`) even though the math is a pure f4 contraction; this forces
  the device seam to grow for every standalone stat.

**Decision duplication:**

* `--f2-dir` empty check: repeated in `cmd_f4.cpp:109`, `cmd_qpadm.cpp:77`,
  `cmd_qpgraph.cpp:135`, `cmd_f3.cpp`, `cmd_qpwave.cpp`, etc.
* `--out` empty → stdout vs file open: repeated in every command that emits
  (`cmd_f4.cpp:192-202`, `cmd_qpadm.cpp:141-151`, `cmd_qpgraph.cpp:197-207`).
* `parse_output_format()` is called in every command even though
  `ConfigBuilder::build()` already validated it.
* No-GPU check + message: repeated verbatim in at least six commands
  (`cmd_f4.cpp:165-169`, `cmd_qpadm.cpp:108-112`, `cmd_qpgraph.cpp:166-170`,
  `cmd_extract_f2.cpp:330-334`).

**Temporal coupling:**

* `upload_f2_blocks_to_device()` must be called on a backend whose current
  device matches the `device_id` argument.  The function guards this internally
  (`device_f2_blocks.cu:92-100`), but the caller has no type-level guarantee.

**Cognitive load:** ~11 files to follow one quartet statistic.

---

### 2.2 Precompute flow: `extract-f2` (there is no lazy precompute)

**Important finding:** only `extract-f2` precomputes.  `f4` and `qpadm` require an
existing `--f2-dir`; there is **no command that decides to precompute missing f2
blocks on demand**.  The "precompute decision" is therefore just: the user ran
`extract-f2`.

**Path:**

1. `cli_parse.cpp:742-743` → `run_extract_f2_command(*config)`.
2. `cmd_extract_f2.cpp:133-423` — the command:
   * validates the genotype triple + pop selection (`:135-153`),
   * does an **up-front** read of `.ind`/`.snp` for sizing and `--dry-run`
     (`:155-209`),
   * optionally starts a background SHA-256 thread (`:297-311`),
   * calls `steppe::run_extract_f2()` (`:336-339`),
   * builds `F2DirMeta` by hand (`:364-397`),
   * calls `write_f2_dir()` (`:399-404`),
   * prints a human summary to **stdout** (`:406-422`).
3. `src/app/extract_f2_core.cpp:73-328` — `run_extract_f2()`:
   * opens `GenoReader`, reads `.ind`/`.snp`,
   * calls `core::read_canonical_tile()` (`:109-110`) to get a canonical
     `io::GenotypeTile`,
   * builds a `DecodeTileView` (`:154-163`),
   * calls `backend.detect_sample_ploidy_device()` for counts (`:172-173`),
   * on the CUDA path calls `backend.decode_af_compact_filter()`
     (`:204-212`) which runs the regime-B keep-mask + column compaction on the
     device and returns a compacted `DeviceDecodeResult`,
   * copies the compacted `Q/V/N` back to host (`:221`),
   * calls `core::assign_blocks()` (`:294-296`),
   * calls `core::compute_f2_blocks_multigpu_tiered()` (`:307-308`),
   * returns `dev_f2.to_host()` as an `F2BlockTensor` (`:312`).
4. `src/core/stats/read_canonical_tile.cpp:57-113` — dispatches on
   `GenoFormat` and, for every non-TGENO format, calls a backend transpose via
   `backend.transpose_to_canonical()` (`:38`).  This is the **cross-layer bridge
   that architecture.md §4 says should live in `app`**.
5. `src/device/cuda/decode_compact_kernel.cu:116-169` — launches
   `launch_autosome_keep_mask` / `launch_regimeb_keep_mask` /
   `launch_compact_columns_gather`.
6. `src/core/fstats/f2_blocks_multigpu.cpp:443-549` —
   `compute_f2_blocks_multigpu_tiered()` decides Resident/HostRam/Disk via
   `resolve_output_tier()` and calls the appropriate backend seam.
7. `src/device/cuda/cuda_backend.cu:2612-5616` — the backend implements
   `compute_f2_blocks_device`, `compute_f2_blocks_streamed`, etc.
8. `src/app/f2_dir_writer.cpp:373-521` — `write_f2_dir()` writes `f2.bin`,
   `pops.txt`, and hand-rolled `meta.json`.

**Layer bouncing (the big one):**

* `core/stats/read_canonical_tile.cpp` is in `core` but it **reads genotype tiles
  and calls `backend.transpose_to_canonical()`**.  This is exactly the wiring
  point `architecture.md:239` says belongs in `app`.  It makes `core` depend on
  `io` (confirmed by `src/core/CMakeLists.txt:153` linking `steppe::io`).
* `extract_f2_core.cpp` is in `src/app/` but is compiled by `src/extract/` (per
  `src/extract/CMakeLists.txt:25-26`).  The directory ownership is inverted.
* `cmd_extract_f2.cpp` owns the dry-run tier estimate (`:236-285`) *and* the real
  tiered compute, so the tier logic is split between app planning and core
  execution.

**Decision duplication:**

* `validate_explicit_pops()` is implemented twice: once in
  `cmd_extract_f2.cpp:118-129` (returns bool) and again in
  `extract_f2_core.cpp:57-69` (throws).  The comments admit they are the same.
* The `--dry-run` path recomputes `assign_blocks` and `resolve_output_tier` with
  the same inputs the real run will use (`cmd_extract_f2.cpp:238-267`).  That is
  intentional but still duplicates policy.
* `extract_f2_core.cpp:196` checks `backend.capabilities().device_count > 0` to
  decide CPU-vs-CUDA filter path, duplicating the "am I resident?" intent that
  the tiered entry already encoded.

**Temporal coupling / hidden ordering:**

* The background hash thread must be joined before `write_f2_dir` reads
  `meta.geno_sha256`.  The ordering is enforced by `ThreadJoiner` RAII
  (`cmd_extract_f2.cpp:308-311`), but the coupling is implicit in the data flow.
* `decode_af_compact_filter` requires the caller to set
  `view.detect_ploidy_on_device` vs `view.sample_ploidy` correctly.  There is no
  type that encodes "Auto vs forced ploidy"; it is a pair of fields with a
  protocol.
* `DeviceF2Blocks::to_host()` and `upload_f2_blocks_to_device()` both push/pop
  the current CUDA device internally.  This is safe but means device affinity is
  not visible in the type.

**Mixed abstraction levels:**

* `cmd_extract_f2.cpp` interleaves CLI validation, a `--dry-run` sizing report,
  thread management, SHA-256 provenance, and output metadata assembly.  It is ~300
  lines of high-and-low abstraction in one function.
* `f2_dir_writer.cpp` couples directory creation, binary serialization, SHA-256,
  and hand-built JSON in one TU.

**Cognitive load:** ~14 files for one genotype-to-f2 run.

---

### 2.3 Fit flow: `qpadm` / `qpgraph`

#### `qpadm` single-model

**Path:** `cli_parse.cpp:353-357` → `cmd_qpadm.cpp:75-157` →
`include/steppe/qpadm.hpp:170` → `src/core/qpadm/qpadm_fit.cpp:301-311` →
`run_qpadm_impl()` (`:260-273`) → `assemble_f4()` (`:265`) + `run_impl()`
(`:53-199`) → `jackknife_cov()`, `gls_weights()`, `se_from_loo()`,
`run_rank_sweep()`, `run_popdrop()` → CUDA backend overrides in
`cuda_backend.cu:3030-5616` and kernels in `qpadm_fit_kernels.cu`.

#### `qpadm-rotate` / S8 search

**Path:** `cmd_rotate.cpp` builds model list → `run_qpadm_search()`
(`include/steppe/qpadm.hpp:192`) → `src/core/qpadm/model_search.cpp:252-323` →
`replicate_f2()` (`:215-248`) + `plan_model_shards()`
(`model_search_core.cpp:9-31`) + `fit_shard()` (`model_search.cpp:113-149`) →
`fit_models_batched()` CUDA override (`cuda_backend.cu:4916-5362`) or per-model
fallback `fit_models_batched_default()` (`model_search.cpp:70-79`).

#### `qpgraph`

**Path:** `cli_parse.cpp:379` → `cmd_qpgraph.cpp:133-209` →
`src/core/qpadm/qpgraph_fit.cpp:173-185` → parse model, `assemble_f3_triples()`,
`jackknife_cov()`, `qpgraph_fit_fleet()` (`cuda_backend.cu:3619-3786`) → kernels
in `qpgraph_fit_kernels.cu:474-539`.  Topology enumeration for search is host-side
in `qpgraph_enumerate.cpp:268-489`.

**Layer bouncing:**

* `device/backend.hpp:624-651` declares `core::qpadm::fit_models_batched_default`
  and `core::qpadm::model_in_small_path` **inside a device header**.  A device
  layer header referencing core-layer symbols is a layering inversion.
* `model_search.cpp:33-36` documents that `assemble_f4` caches `tot_line_` as a
  per-backend member that `jackknife_cov` consumes inside `run_impl`; the two
  calls must be adjacent on the *same* backend instance.  This is temporal
  coupling hidden in a comment rather than a type.
* `qpgraph_enumerate.cpp` is host-pure graph theory + 1-WL hashing that lives in
  `core/qpadm/` but is consumed by `qpgraph_fit.cpp`; the separation is
  reasonable, but the file mixes AT2-emulation, graph algorithms, and
  isomorphism hashing at once.

**Decision duplication:**

* The small-path envelope predicate `model_fits_small_path` is defined in
  `qpadm_bounds.hpp`, checked in `model_search.cpp:88-93`, and sized in the
  kernel.  That is single-sourced, but the *partition into small/large* is done
  in `fit_shard()` (`model_search.cpp:119-130`) while the *backend* also has a
  `model_fits_small_path` dispatch (`backend.hpp:643-650`).
* G==1 fast paths exist in `compute_f2_blocks_multigpu.cpp:231`,
  `compute_f2_blocks_multigpu_device()` (`:387`), and `run_qpadm_search()`
  (`model_search.cpp:270`).  The intent "single device needs no fan-out" is
  repeated.

**Temporal coupling:**

* `replicate_f2()` (`model_search.cpp:215-248`) materializes `f2` to host and
  re-uploads to every GPU.  The big TODO at `:167-191` admits this is a
  gather-then-scatter round-trip caused by the fact that precompute combines to
  one device and rotation then re-broadcasts.  The ordering is forced by the
  current architecture, not by the problem.
* `fit_shard()` catches the sentinel from `fit_models_batched` and falls back to
  the per-model default (`model_search.cpp:134-140`).  A backend that forgot to
  override fails at runtime, not compile time.

**Mixed abstraction levels:**

* `cmd_qpgraph.cpp:82-129` hand-builds CSV and JSON output from raw labels and
  `operator<<` doubles, bypassing `result_emit.cpp` entirely (already flagged in
  `05_io_cli_and_output_ux.md`).
* `cuda_backend.cu` contains decode, f2 GEMMs, qpadm fit, qpfstats, dates, and
  qpgraph fleet all in one backend implementation file (~5600 lines).  The seam
  is monolithic.

**Cognitive load:** qpadm rotation: ~15 files.  qpgraph search: ~10 files plus
graph-theory internals.

---

### 2.4 Multi-GPU f2 path

**Path:** `extract_f2_core.cpp:307-308` calls
`compute_f2_blocks_multigpu_tiered()` → `f2_blocks_multigpu.cpp:443-549`.
For a Resident result it calls `compute_f2_blocks_device()`; for HostRam/Disk it
calls `compute_f2_blocks_streamed()`; for multi-GPU it delegates to
`compute_f2_blocks_multigpu_device()` (`:372-428`) or the host-staged
`compute_f2_blocks_multigpu()` (`:213-364`).

Inside multi-GPU:

1. `f2_blocks_multigpu_core.cpp:135-155` — `plan_multigpu_shards()` uses
   `core::block_ranges()` + `device::plan_block_shards()`.
2. `f2_blocks_multigpu_core.cpp:68-131` — `fan_out_shards()` launches one
   `std::jthread` per GPU, builds local `block_id` and zero-copy `MatView`s,
   then calls a `ShardSeam`.
3. The seam calls either:
   * `compute_f2_blocks_resident()` (P2P path), producing `DevicePartial`s,
   * `compute_f2_blocks()` (host path), producing `F2BlockTensor` partials,
   * `compute_f2_blocks_into()` (direct path), writing into a caller buffer.
4. For P2P: `device/cuda/p2p_combine.cu:142-318` assembles resident partials via
   `place_partials_into()` (`:86-138`) using `cudaMemcpyPeerAsync`.
5. For host-staged: `core/fstats/f2_combine.cpp:26-125` `combine_f2_partials_host()`
   places partials in fixed g order.

**Layer bouncing:**

* The combine gate (`select_p2p_combine`, `f2_blocks_multigpu.cpp:126-129`) is
  single-sourced and well isolated, but the decision depends on config, caps,
  *and* a runtime probe (`can_access_peer`) recorded during `build_resources`.
  The caller must trust that `Resources::gpus[0].caps` is the right authority.
* `compute_f2_blocks_multigpu_device()` falls back to the host-staged function on
  no-P2P hardware (`f2_blocks_multigpu.cpp:425-427`), which then re-uploads to a
  `DeviceF2Blocks`.  The data path bounces: device → host → device, exactly as
  the TODO warns.

**Temporal coupling:**

* `place_partials_into()` relies on the caller having already validated that
  shards tile `[0, n_block_full)` exactly.  Validation is shared
  (`validate_resident_partials`), but the fact that the combine writes every slab
  exactly once is a protocol, not a type.
* The peer-enable in `p2p_combine.cu:123` must happen while the root device is
  current; the `DeviceGuard` at `:161-179` enforces this, but device affinity is
  again implicit.
* `DevicePartial` handles must outlive the combine; the caller frees them after
  the combine returns (`p2p_combine.cu` comments at `:84-85`, `:314-317`).

**Cognitive load:** Multi-GPU precompute is arguably the most carefully
single-sourced part of the codebase, but a reader still needs
`f2_blocks_multigpu.cpp`, `f2_blocks_multigpu_core.cpp`, `f2_combine.cpp`,
`p2p_combine.cu`, `shard_plan.hpp`, `device_partial.hpp`, and `device_f2_blocks`
to understand one call.

---

### 2.5 Status / error propagation flow

**How a CUDA kernel error becomes a CLI exit code:**

1. Kernel launch wrappers call `STEPPE_CUDA_CHECK_KERNEL()`
   (`src/device/cuda/check.cuh:325-330`), which checks `cudaGetLastError()` and,
   in debug builds, synchronizes.
2. `STEPPE_CUDA_CHECK()` (`check.cuh:279-280`) throws `CudaError` (or
   `CublasError`, `CusolverError`, `CufftError`) carrying file:line and the
   original `cudaError_t`.
3. The exception unwinds through the backend implementation (`cuda_backend.cu`),
   through `core` orchestrators, and is caught in the app command, e.g.
   `cmd_f4.cpp:176-181` or `cmd_extract_f2.cpp:344-347`.
4. The command prints `steppe <cmd>: device error: <what()>` and returns
   `kExitRuntimeError` (value 5).
5. `main.cpp:19-24` catches any stragglers and also returns
   `kExitRuntimeError`.

**Domain outcomes (RankDeficient / NonSpdCovariance / ChisqUndefined):**

* These are **not exceptions**.  They ride as `Status` values inside result
  structs (`QpAdmResult`, `QpWaveResult`, `F4Result`, `QpGraphResult`).
* The CLI maps the result's `status` through
  `src/core/config/exit_code.hpp:54-71`, which returns `kExitOk` (0) for the
  three domain outcomes.

**Config validation:**

* `ConfigBuilder::build()` returns an optional; on failure `cli_parse.cpp:70-73`
  prints the reason and the callback calls `std::exit(kExitInvalidConfig)`.
* Command-specific checks (missing `--f2-dir`, bad pop name, empty `--left`) also
  return `kExitInvalidConfig` directly, without ever creating a `Status`.

**Status transformations:**

* A CUDA kernel error is transformed **three times**: `cudaError_t` → typed
  exception → command-level `kExitRuntimeError`.
* A domain outcome is transformed **twice**: numeric kernel/devInfo signal →
  `Status` enum value → `kExitOk` via `exit_code_for()`.
* **There is no mapping from CUDA OOM to `Status::DeviceOom` or to
  `kExitDeviceOom` in the CLI.**  `Status::DeviceOom` exists in the public enum
  (`include/steppe/error.hpp:28`) and `exit_code.hpp` maps it, but the app catch
  blocks treat *all* exceptions as `kExitRuntimeError`, so `kExitDeviceOom` is
  dead code in the CLI path.
* I/O errors from `read_f2_dir` or `write_f2_dir` are returned as a struct with
  an error string; the command maps them to `kExitIoError`.  Notably,
  `write_f2_dir` uses `Status::InvalidConfig` internally for disk-full/write
  failures (`f2_dir_writer.cpp:429`, `439`, etc.), which then becomes
  `kExitInvalidConfig` instead of `kExitIoError`.

---

## 3. Cross-cutting flow problems

1. **The `ComputeBackend` god interface.**  `backend.hpp:670-1853` declares ~40
   pure/semi-pure virtuals covering decode, precompute, fit, sweeps, dates,
   qpfstats, qpgraph, and capabilities.  Every new GPU feature touches this one
   seam, forcing `core`, `device`, `app`, and bindings to recompile together.
   Alternative backends (e.g., a mock for unit tests) are expensive.

2. **`core` does I/O.**  `core/stats/read_canonical_tile.cpp` and
   `extract_f2_core.cpp` read genotype files and call device transposes.
   `src/core/CMakeLists.txt:153` links `steppe::io`.  This contradicts
   `architecture.md:239` ("the `app` layer is the only place that wires `io`
   output into compute").

3. **App commands are copy-paste orchestrators.**  Every fit command repeats:
   validate `--f2-dir`, `read_f2_dir`, `PopResolver`, `build_resources`, no-GPU
   check, `upload_f2_blocks_to_device`, `run_*`, format parse, `--out` open,
   emit.  That repetition is exactly what makes the CLI layer feel bouncy.

4. **Output layer is half-centralized.**  `result_emit.cpp` handles qpadm/qpwave
   well, but `cmd_dates.cpp`, `cmd_qpgraph.cpp`, `cmd_fstat_sweep.cpp`, and
   `cmd_extract_f2.cpp` all hand-build JSON/CSV.  Number formatting, quoting,
   and NaN handling differ by command (detailed in
   `05_io_cli_and_output_ux.md`).

5. **`RunConfig` is an app config object in `core`.**
   `src/core/config/run_config.hpp:40-179` holds CLI strings like `--out`,
   `--format`, `--graph`, `--shard-dir` alongside compute config.  `core` should
   not know output filenames.

6. **Directory ownership smells.**  `src/access/CMakeLists.txt:24-25` compiles
   `../app/f2_dir_io.cpp` and `../app/pop_resolver.cpp`; `src/extract/CMakeLists.txt:25-26`
   compiles `../app/extract_f2_core.cpp` and `../app/f2_dir_writer.cpp`.
   Shared library code should live under its own directory.

7. **Status enum vs. exceptions mismatch.**  The public `Status` enum
   (`include/steppe/error.hpp`) lacks `IoError` and `CudaRuntime`, yet the
   architecture doc (`architecture.md:682-689`) promised a richer C ABI.  The
   CLI maps exceptions straight to exit codes, bypassing `Status`.

8. **Multi-GPU fit is honestly deferred.**  `model_search.cpp:167-191` documents a
   ~3.8 s / 8.7 GB host bounce on consumer GPUs and caps speedup at ~1.21×.  The
   root cause is the precompute gather-then-rotation-scatter pattern.

---

## 4. What "organic" would look like

For each major flow, the ideal shape is:

* **CLI commands are thin shells.**  A command parses args, builds a small
  domain request object, calls one library function, and emits via a shared sink.
  It should not know about `DeviceF2Blocks`, `upload`, or `read_f2_dir` unless
  that command is uniquely about I/O.

* **`core` is pure compute orchestration.**  Genotype I/O and format dispatch
  belong in `app` or a dedicated `extract` library.  `core` should receive
  already-decoded `Q/V/N` views and block partitions.

* **A split device seam.**  Instead of one `ComputeBackend`, have role-specific
  interfaces (`DecodeBackend`, `PrecomputeBackend`, `FitBackend`,
  `SweepBackend`, `QpgraphBackend`).  A thin facade can aggregate them for
  existing callers.  Missing overrides become compile errors, not runtime
  sentinels.

* **One output abstraction.**  An `OutputSink` owns stdout-vs-file, atomic writes,
  parent-dir creation, stream-state checks, and CSV/JSON formatting.  Every
  command emits through it.

* **A single status/error value type.**  Library calls return a `Result<T>`
  carrying either a value or a `Status` (including `IoError`, `CudaRuntime`,
  `DeviceOom`).  The CLI maps *one* status to an exit code.

* **Precompute is explicit and centralized.**  Either `extract-f2` is the only
  precompute path (current reality) and that is clearly documented, or a
  `PrecomputeEngine` is shared by any command that needs to materialize f2 from
  genotypes.  Either way, `F2BlockTensor` creation happens in one place.

* **Multi-GPU fit avoids the host bounce.**  Per-device precompute (each GPU
  builds its own full f2 from the genotype stream) eliminates `replicate_f2()`
  entirely.

---

## 5. Concrete refactor targets

The smallest changes with the biggest flow payoff:

1. **Factor the fit-command boilerplate.**  Add `app/command_runner.hpp` with a
   helper like
   ```cpp
   template<class Fn>
   int run_fit_command(const RunConfig& cfg, Fn&& compute);
   ```
   that owns `read_f2_dir` → `build_resources` → `upload` → `compute` → emit and
   maps exceptions/status to exit codes.  Apply it to `f4`, `f3`, `f4-ratio`,
   `qpadm`, `qpwave`, `qpgraph`, and rotate.

2. **Move genotype I/O out of `core`.**  Relocate `read_canonical_tile.cpp` and
   `run_extract_f2` (currently `src/app/extract_f2_core.cpp`) into
   `src/extract/`.  Make `core` depend only on `MatView`/`BlockPartition`, not on
   `io` headers.  Update `core/CMakeLists.txt` to drop the `steppe::io` link.

3. **Move shared app files to their libraries.**  Move `f2_dir_io.cpp`,
   `pop_resolver.cpp`, `f2_dir_writer.cpp`, and `extract_f2_core.cpp` into
   `src/access/` and `src/extract/`; remove `../app` source references from
   `src/access/CMakeLists.txt` and `src/extract/CMakeLists.txt`.

4. **Slim the backend seam.**  Split `ComputeBackend` into focused interfaces.
   Move `fit_models_batched_default` / `model_in_small_path` declarations out of
   `device/backend.hpp` and into `core/qpadm/model_search.hpp`, passing a
   callback/delegate into the backend instead.

5. **Centralize output.**  Create `app/output_sink.hpp` with `OutputSink`,
   `CsvRowWriter`, `JsonWriter`, and `DoubleFormatter`.  Convert `qpgraph`,
   `dates`, `extract-f2` summary/dry-run, and `fstat_sweep` to use it.

6. **Fix the status/exit-code taxonomy.**  Add `Status::IoError` and
   `Status::CudaRuntime` to the public enum; make `write_f2_dir` return `IoError`
   on disk failures; map library `Status` values to the existing exit codes in
   `exit_code_for()`.  Stop catching all exceptions as `kExitRuntimeError`.

7. **Remove duplicate `validate_explicit_pops`.**  Keep one throwing version in
   the extract library and have the CLI call it, converting the exception to the
   appropriate exit code.

8. **Resolve the multi-GPU host bounce.**  Implement per-device precompute for S8
  rotation so each GPU builds its own `f2_blocks`, eliminating `replicate_f2()`
  (`model_search.cpp:215-248`) and the TODO at `:167-191`.

9. **Honour the C ABI promise or remove it.**  Either implement the opaque-handle
   C boundary described in `architecture.md:908-914`, or document that
   `include/steppe/` is a same-toolchain C++ convenience API.

---

## 6. Verdict

**Overall grade: C+.**

The project is **structurally layered** and **parity-correct**, which is the
hard part.  But the flow is not organic: control and data bounce between `app`,
`core`, `io`, and `device` because orchestration responsibilities are misplaced,
the `ComputeBackend` seam is a monolith, and CLI commands duplicate the same
five-step dance.  A reader must juggle too many files and too many implicit
ordering contracts to understand a single command.

The good news is that the worst coupling is concentrated and fixable: most of it
is the command boilerplate, the backend seam, and the `core` ↔ `io` entanglement
from genotype-path entry points.  Addressing the concrete targets above —
especially a shared command runner, a split backend seam, and moving genotype I/O
out of `core` — would lift this from a C+ flow to a B+ or A- flow without
changing any numeric results.

**Bottom line:** the code is correct, but it does not yet read like a single
product.  It reads like a collection of carefully parity-gated pipelines that
happen to share a CLI.
