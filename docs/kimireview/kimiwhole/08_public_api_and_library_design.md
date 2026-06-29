# Public API and Library Design Review — steppe

Read:
- `/home/suzunik/steppe/include/steppe/*.hpp` (public C++ surface)
- `/home/suzunik/steppe/src/device/backend.hpp` (internal ComputeBackend seam)
- `/home/suzunik/steppe/bindings/module.cpp` (nanobind compiled binding)
- `/home/suzunik/steppe/bindings/steppe/__init__.py` (pure-Python facade)
- `/home/suzunik/steppe/docs/design/cli-bindings.md` (design contract)
- `/home/suzunik/steppe/docs/kimireview/bindings_module.md` (prior review)

This review treats the installed `include/steppe/*.hpp` headers, the internal
`backend.hpp` seam, and the Python bindings as one library surface and assesses
it from first principles.

---

## 1. Public C++ API surface clarity and completeness

### What's good

- **The CUDA-free seam is well defended.** Every installed public header
  (`include/steppe/qpadm.hpp:12-15`, `f4.hpp:22-26`, `fstats.hpp:9-15`,
  `extract.hpp:9-17`) states the same contract: no CUDA headers, plain standard
  C++20, name→index resolution owned by the caller. This is the single most
  important architectural decision in the library surface and it is consistent.

- **Entry points are narrow and task-oriented.** `run_qpadm`, `run_qpwave`,
  `run_qpadm_search` (`qpadm.hpp:170-237`), `run_f4`/`run_f3`/`run_f4ratio`
  (`f4.hpp:70-81`, `f3.hpp:69-80`, `f4ratio.hpp:74-85`), `run_extract_f2`
  (`extract.hpp:91-99`), `run_qpfstats` (`qpfstats.hpp:99-105`), `run_dstat`
  (`dstat.hpp:83-89`), `run_dates` (`dates.hpp:118-121`), and the sweep/search
  variants (`fstat_sweep.hpp:79-86`, `qpgraph_search.hpp:118-127`) give users a
  complete f-statistics workflow without exposing the device layer.

- **Index-only compute seam.** Populations are passed as `std::vector<int>` /
  `std::span<const int>` indices into the f2 P axis, never strings, which keeps
  the core fast and unambiguous (`qpadm.hpp:106-119`). The design doc
  (`cli-bindings.md §1`, lines 79-95) documents why this separation is necessary.

- **Host-oracle parity overloads.** Most entries come in both
  `device::DeviceF2Blocks&` (GPU primary, zero D2H) and `F2BlockTensor&` (host
  parity) forms (`qpadm.hpp:170-181`, `f4.hpp:70-81`, `qpgraph.hpp:134-146`).
  This is an excellent testing/architecture pattern.

### What a senior API designer would flag

- **The surface is large and flat.** There are ~16 public headers and they all
  expose top-level free functions in `namespace steppe`. There is no
  sub-namespace grouping (e.g. `steppe::fit`, `steppe::stat`,
  `steppe::precompute`). As the library grows, discoverability will suffer and
  symbol collision risk rises.

- **Result structs are inconsistent in scope.** `QpAdmResult` (`qpadm.hpp:124-163`)
  is a kitchen sink: single-model weights, rank-sweep vectors, rankdrop, popdrop,
  status, precision tag, and model index. `QpWaveResult` (`qpadm.hpp:212-222`)
  duplicates the rankdrop shape. `F4Result`/`F3Result`/`F4RatioResult`
  (`f4.hpp:46-60`, `f3.hpp:50-64`, `f4ratio.hpp:56-69`) are tidy tables.
  `QpGraphResult` (`qpgraph.hpp:85-126`) mixes vectors of different semantic
  groups (weights, edges, leaves, worst residual). The public surface would be
  cleaner with nested table structs (`WeightsTable`, `RankDropTable`,
  `PopDropTable`, `EdgeTable`) composed into result types.

- **Magic `-1` sentinels are still exposed.** `QpAdmModel::target = -1`,
  `model_index = -1`, `QpAdmOptions::rank = -1` (`qpadm.hpp:61,69,109,118,162`)
  are pragmatic for aggregates, but they leak invalid states into the public
  contract. A public API should prefer `std::optional<int>` or a small `Index`
  type with an explicit `Unset` value.

- **`Precision` is a struct but is treated like an enum in user code.**
  `Precision::Kind` plus `mantissa_bits` (`config.hpp:284-331`) is flexible, but
  most callers only want `"fp64"`, `"emu"`, or `"tf32"`. The public surface
  forces users to construct a `Precision{Kind::EmulatedFp64, 40}` rather than
  offering named factory functions (`Precision::emulated_fp64()`,
  `Precision::native()`, `Precision::tf32()`).

- **`FilterConfig` exposes internal policy comments as API behavior.**
  `FilterConfig::geno_max_missing` vs the AT2 population-axis `maxmiss` is
  carefully documented (`config.hpp:490-496`, `extract.hpp:19-24`), but the
  public type still has a field whose meaning differs from its name. This is a
  long-term support burden; a dedicated `ExtractFilterConfig` or clearer field
  names (`sample_max_missing`, `pop_max_missing`) would reduce confusion.

- **`fstats.hpp` exposes a raw `size()` helper but no indexing helpers.**
  `F2BlockTensor` documents the flat layout `i + P·j + P·P·b`
  (`fstats.hpp:35-46`), but users must hand-roll index math. A small
  `double f2_at(int i, int j, int b) const` and a checked accessor would improve
  ergonomics and prevent off-by-one errors.

- **`SweepResult` and `QpGraphSearchResult` expose parallel-array tables with no
  row accessor.** `SweepResult::keys[r]` is an `std::array<int,4>` where the
  fourth slot is unused for f3 (`fstat_sweep.hpp:61-62`).
  `QpGraphSearchResult::candidates` is a vector of `QpGraphCandidate`
  (`qpgraph_search.hpp:75-112`), which is better, but the design still leans on
  the caller to keep parallel arrays aligned.

---

## 2. ABI stability considerations

### What's good

- **Value types dominate the public surface.** `QpAdmModel`, `QpAdmResult`,
  `F2BlockTensor`, `QpGraphEdge`, etc., are plain aggregates of `std::vector`
  and scalar fields. This avoids the ABI hazards of virtual functions,
  `std::string` inline layouts across compilers (still a hazard, but a contained
  one), and hidden vtables.

- **No inline virtual functions in public headers.** The public API headers do
  not define virtual dispatch surfaces; the device abstraction (`backend.hpp`)
  is an internal seam, not installed as public API.

- **The design doc explicitly defers the C ABI** (`cli-bindings.md §1`
  constraint 5, lines 65-70; §9.4). The project understands that the installed
  cross-toolchain boundary is a separate milestone (`M(abi-1)`), which is the
  right call for ABI stability.

### What a senior API designer would flag

- **`std::vector` in public structs is not ABI-stable across compiler/stdlib
  changes.** `F2BlockTensor` (`fstats.hpp:47-78`) and every result struct embed
  `std::vector<double>` / `std::vector<int>` / `std::vector<std::string>`. For a
  library that may ship as a wheel and be loaded into arbitrary Python
  processes, this is acceptable *inside* one build, but it is not a SemVer-frozen
  ABI. If the long-term goal is a stable installed C/C++ ABI, the public structs
  need opaque handles + C accessors, not inline STL containers.

- **`enum class` underlying types are mostly implicit.** `Status`
  (`error.hpp:22-49`) and `Precision::Kind` (`config.hpp:286-320`) do not
  specify `int`/`std::uint32_t` explicitly. For a future C ABI this is a
  footgun; adding `enum class Status : int` is a one-line improvement.

- **`QpAdmResult` appends fields in a monolithic struct.** The comment calls it
  "non-breaking" (`qpadm.hpp:20-23, 141-142`), but for an ABI-stable layout
  (size/offset sensitive) every append is a break. A future stable ABI should
  version the result via opaque handles or nested sub-structs.

- **`JackknifePolicy` uses `enum class : int` with frozen numeric mapping
  (`qpadm.hpp:48-52`), which is good.** This should be the pattern for all public
  enums.

- **The nanobind binding is compiled against the C++ headers in-process
  (`cli-bindings.md §9.4`), so Python ABI stability is not currently a design
  goal.** That is fine, but it should be stated explicitly in the public docs
  (it is not).

---

## 3. Result types and their ergonomics

### What's good

- **Domain outcomes are values, not exceptions.** `Status` (`error.hpp:22-49`)
  carries `Ok`, `DeviceOom`, `RankDeficient`, `NonSpdCovariance`,
  `ChisqUndefined`, `InvalidConfig`. This is exactly right for model-space
  search and sweep workflows where thousands of models may fail individually.

- **`se.empty()` is the unambiguous "not computed" sentinel** (`qpadm.hpp:127-130`).
  This is honest and avoids the `NaN`/`0` ambiguity common in genomics APIs.

- **Parallel-array tables are simple to bind and emit.** The binding layer can
  convert `F4Result` directly into a pandas DataFrame (`bindings/__init__.py:232-264`).
  The shape mirrors AT2's `$weights`/`$rankdrop`/`$popdrop` conventions
  (`cli-bindings.md §5.2`).

### What a senior API designer would flag

- **`QpAdmResult` mixes scalar, vector, and table semantics in one flat struct.**
  A user who only wants `weight`/`p` still carries `rankdrop_*`, `popdrop_*`,
  `rank_chisq`, etc. The Python facade partially fixes this with DataFrame
  accessors (`QpAdmResult.weights`, `.rankdrop`, `.popdrop` in
  `__init__.py:127-174`), but the C++ user has no equivalent convenience.

- **`QpGraphResult` lacks a clear separation between admixture weights and drift
  edges.** `weight`, `weight_lo`, `weight_hi`, `admix_from`, `admix_to`,
  `edge_length`, `edge_from`, `edge_to` are all top-level vectors
  (`qpgraph.hpp:89-104`). A nested `struct AdmixtureWeights` and `struct
  DriftEdges` would make the API self-documenting.

- **`DatesResult` mixes scalars, vectors, and LOO diagnostics loosely**
  (`dates.hpp:81-104`). `curve_cm`/`curve_corr` are a table; `loo_date_gen` /
  `loo_weight` are another table; `date_gen`/`se`/`fit_error_sd` are scalars.
  Nesting them would improve clarity.

- **`F2BlockTensor` has no symmetry-enforcing API.** The tensor is symmetric per
  block (`fstats.hpp:35-46`), but the struct gives direct mutable access to the
  underlying `std::vector<double>`. A public API that wants to guarantee
  invariants should either expose const accessors or document loudly that
  symmetry is the caller's responsibility.

- **`SweepResult` returns survivors in "per-chunk lex order"**
  (`fstat_sweep.hpp:56-60`). For `TopK` the doc says it is sorted by `|z|`
  descending, but for `MinZ` it is not. A public API should guarantee a stable
  sort order or expose an explicit `sort_by` parameter.

- **`QpGraphSearchResult::candidates` exposes every enumerated topology, which is
  excellent for parity testing** (`qpgraph_search.hpp:86-93`), but there is no
  lookup by `hash` or `edges`; users must linear-scan. A `std::unordered_map`
  helper or an explicit `find_candidate(std::uint64_t hash)` would help.

---

## 4. Separation between public API and implementation details

### What's good

- **The layering is structurally enforced.** `include/steppe/*.hpp` never
  includes CUDA headers. `device::DeviceF2Blocks` and `device::Resources` are
  forward-declared (`qpadm.hpp:37-40`, `f4.hpp:25-39`, etc.). The real headers
  live in `src/device/`, which is private to the build.

- **`backend.hpp` is an internal seam, not a public header.** It is correctly
  placed under `src/device/` rather than `include/steppe/`. It exposes the
  `ComputeBackend` abstract class (`backend.hpp:670-1853`) with many virtuals,
  which is appropriate for an in-tree dependency-injection seam but would be a
  poor public ABI.

- **Public headers do not leak internal concepts.** Terms like
  `F4Blocks`, `JackknifeCov`, `GlsWeights`, `RankSweep`, `PopDropRow` are
  confined to `backend.hpp` and `src/core/`. The public user sees only the final
  result tables.

### What a senior API designer would flag

- **`src/device/backend.hpp` is enormous (1857 lines) and mixes many roles.** It
  is simultaneously the device abstraction, the S3-S8 fit-engine POD contract,
  the genotype decode view contract, the DATES seam, the qpfstats seam, and the
  sweep seam. A public-library reviewer would split this into
  `device/compute_backend.hpp`, `device/fit_engine_pods.hpp`,
  `device/decode_views.hpp`, etc. Even though it is internal, its complexity
  leaks into every core/device TU.

- **Public headers include `core/internal/views.hpp` indirectly only through
  `backend.hpp`, but `extract.hpp` forward-declares `io::PopSelection` and
  `device::Resources` (`extract.hpp:37-43`).** This is fine, but the public API
  still depends on internal forward-declared types whose real definitions live
  in private headers. For a stable public ABI these seams need opaque handles
  with C accessors.

- **`FilterConfig` lives in `config.hpp` and is used by both public
  `run_extract_f2` and internal filtering code.** That is the right single
  source, but the struct mixes user-facing thresholds (`maf_min`,
  `autosomes_only`) with internal file paths (`prune_in_path`). A small split
  between "filter predicate" and "filter source" would clarify ownership.

- **The Python bindings include `app/f2_dir_io.hpp`, `app/pop_resolver.hpp`,
  `io/geno_reader.hpp`, `io/ind_reader.hpp` directly** (`module.cpp:40-61`).
  These are internal app/io headers, not public C++ API. The binding layer is
  therefore acting as an app layer, which is correct per the architecture
  (`cli-bindings.md §1` constraint 1), but it means the Python API is tightly
  coupled to the CLI's internal file formats.

---

## 5. Python bindings design and consistency with CLI

### What's good

- **The two-layer design is right.** `steppe._core` (compiled) does marshalling
  only; `steppe` (pure Python) does pandas/dataclass shaping
  (`module.cpp:1-7`, `__init__.py:1-13`). This keeps the compiled module free of
  pandas and lets `import steppe` succeed without pandas installed.

- **The Python API mirrors the CLI command set.** `read_f2`, `extract_f2`,
  `qpfstats`, `qpadm`, `qpwave`, `qpadm_search`, `qpgraph`, `qpgraph_search`,
  `f4`, `f3`, `f4ratio`, `qpdstat`, `dstat`, `dates` (`__init__.py:29-53`)
  correspond directly to CLI subcommands (`cli-bindings.md §4.1`).

- **Names are resolved Python-side against `f2.pops`.** Unknown names raise a
  clean `KeyError` (`module.cpp:121-128`). This matches the CLI's pop-name
  resolution against `pops.txt` (`cli-bindings.md §4.3`).

- **Domain outcomes stay on the result object.** `Status` is a Python enum
  (`__init__.py:57-71`), not an exception, consistent with the C++ design.

### What a senior API designer would flag

- **The Python API is not generated from the C++ headers; it is hand-written.**
  Every kwarg default in `__init__.py` (`qpadm` at line 499-510, `extract_f2`
  at line 404-417) must stay synchronized with `QpAdmOptions` and
  `FilterConfig`. There is no schema or stub generator. Over time this will
  drift.

- **`qpadm_search` returns either `list[QpAdmResult]` or a single DataFrame
  depending on `as_dataframe`** (`__init__.py:940-985`). Polymorphic return
  types are a usability footgun; type hints cannot express "this function
  returns A or B." Prefer a dedicated `qpadm_search_table()` helper.

- **`f4`, `f3`, `f4ratio`, `qpdstat`, `dstat` have an `as_dataframe` flag that
  returns either the result object or `.table`** (`__init__.py:760-937`). Same
  polymorphic-return problem. A cleaner API is `f4(...).table` always, or two
  functions.

- **`Status` enum values are lower-case strings in `_core` and Python enum
  members in the facade** (`__init__.py:62-66`). The mapping is trivial but
  brittle; if `Status` gains a value, both `module.cpp:230-240` and
  `__init__.py` must be updated.

- **Python docstrings live only in `__init__.py` and `module.cpp` docstrings;**
  there is no generated Sphinx/ReadTheDocs surface. For an A+ library, API docs
  should be generated from the Python module and cross-linked to the C++ design
  docs.

- **`F2Blocks.to_numpy()` and `vpair_to_numpy()` are copy-by-default**
  (`module.cpp:1025-1041`, `__init__.py:383-392`). That is safe, but for large
  tensors (P=768, n_block~800 ⇒ ~3.7 GiB) users will want a zero-copy view.
  The design doc defers DLPack (`cli-bindings.md §5.3`), but the public API
  should at least document the copy cost and roadmap.

- **`qpgraph_search` exposes `candidates` as a DataFrame only when pandas is
  installed** (`__init__.py:691-704`). The property silently requires pandas
  despite the class itself not needing it. That is consistent with the lazy
  pandas design, but it means the full search result is not inspectable without
  pandas.

---

## 6. Memory ownership semantics across the API boundary

### What's good

- **`F2BlockTensor` owns its data.** `std::vector<double> f2`, `vpair`,
  `block_sizes` (`fstats.hpp:47-78`) are value members. There is no hidden
  device pointer or external allocator.

- **`DeviceF2Blocks` never crosses into Python.** The binding uploads the host
  tensor inside each call (`module.cpp:192-203`) and lets the device object be
  destroyed on scope exit. This is exactly the ownership model the design doc
  calls for (`cli-bindings.md §5.3`, spike risks #1 and #2).

- **The C++ public API takes references, not owners.** `run_qpadm` takes
  `const device::DeviceF2Blocks&` and `device::Resources&` (`qpadm.hpp:170-173`).
  The caller owns both; the function borrows them for the call duration. This
  is simple and safe.

### What a senior API designer would flag

- **`F2Handle` in the bindings is allocated with raw `new`** (`module.cpp:397`,
  `847`, `910`). The prior `bindings_module.md` review (lines 13-19) already
  flagged this. nanobind's `rv_policy::take_ownership` requires a pointer, but
  a `std::unique_ptr<F2Handle>` + a small `release_to_python()` helper would
  make the transfer exception-safe and self-documenting.

- **`f2_to_numpy` copies element-by-element** (`module.cpp:1031-1032`). The
  prior review flagged this too. For a multi-gigabyte tensor, `std::memcpy` or
  `std::copy` (which the compiler can vectorize) is a straightforward fix.

- **`QpAdmResult` and friends return large vectors by value.** This is correct
  C++11+ move semantics, but the binding layer converts them into Python lists
  (`result_to_dict` at `module.cpp:255-291`), then the Python facade converts
  those lists into pandas DataFrames (`__init__.py:127-174`). For large search
  results this is `C++ vector → nb::list → Python list → pandas Series`. A
  direct numpy/pandas construction from `std::span` would reduce copies.

- **There is no explicit resource lifetime API in Python.** `F2Blocks` holds a
  cached `Resources` (`module.cpp:82`), but the user cannot close or reset it.
  If the user changes `device` mid-session, the cached resources stay bound to
  the old device. A `F2Blocks.close()` or context-manager protocol would help.

- **`run_extract_f2` and `run_qpfstats` have dual return modes** (`module.cpp:743-915`):
  either a path string or a new `F2Handle`. Union return types are awkward in
  both C++ and Python. A dedicated `write_f2_dir()` vs `extract_f2_to_handle()`
  split would be clearer.

---

## 7. Config/options structs design

### What's good

- **Single source of truth for constants.** `config.hpp` collects named
  constants: `kDefaultMantissaBits=40`, `kFstatMaxComb`,
  `kDefaultBlockSizeCm=5.0`, `kAutosomeChromMin/Max=1/22`,
  `kMaxVramUtilizationFraction=0.80`, etc. (`config.hpp:39-274`). This is
  exactly what a config header should do.

- **`DeviceConfig` separates override intent from discovered capability**
  (`config.hpp:337-473`). The big comment block (`config.hpp:364-387`) is an
  unusually clear explanation of why `devices`, `enable_peer_access`,
  `prefer_p2p_combine`, `deterministic`, and `force_tier` are intent knobs while
  probe results live in `Resources`.

- **`QpAdmOptions` is small and explicit** (`qpadm.hpp:57-100`). Defaults match
  AT2 parity constants and are documented (`fudge=1e-4`, `als_iterations=20`,
  `rank_alpha=0.05`).

- **`DatesOptions` and `QpGraphOptions` are domain-specific and self-contained**
  (`dates.hpp:60-77`, `qpgraph.hpp:56-80`).

### What a senior API designer would flag

- **`DeviceConfig` is a large, mutable bag of fields.** It has 12+ members
  (`config.hpp:337-473`). For a public config type, a builder pattern or
  immutable struct with named setters would prevent half-initialized configs.
  The CLI design doc proposes a `ConfigBuilder` (`cli-bindings.md §4.5,
  §9 question 1`), but it does not exist in the public API yet.

- **`FilterConfig` defaults are "no-op" but fields have misleading names.**
  `geno_max_missing` defaults to `1.0` (keep everything), but in `extract_f2`
  the *real* maxmiss is the population-axis coverage, handled internally
  (`extract.hpp:19-24`, `config.hpp:490-496`). A cleaner public API would
  expose `pop_coverage_min` and `sample_missing_max` as separate concepts.

- **`Precision` requires the user to know about `mantissa_bits`.** The public
  surface should offer named factories so callers do not need to read
  `kDefaultMantissaBits` comments. Example:
  ```cpp
  static Precision fp64() noexcept;
  static Precision emulated_fp64(int mantissa_bits = kDefaultMantissaBits);
  static Precision tf32();
  ```

- **`QpAdmOptions::jackknife` is ignored by `run_qpadm` and `run_qpwave`**
  (`qpadm.hpp:46-47, 83-88`). That is documented, but it is still a config field
  that does not apply to the function it is passed to. Splitting into
  `QpAdmOptions` (single fit) and `QpAdmSearchOptions` (search) would remove the
  surprise.

- **No validation hooks in public config types.** `DeviceConfig` can request
  `stream_count > 1` with `deterministic = true`, which the doc says `build()`
  rejects (`config.hpp:434-455`), but that validation lives in `build_resources`,
  not in the struct. Public config structs should offer a `validate()` or
  `is_valid(std::string& reason)` method for early, host-side checks.

---

## 8. Discoverability and documentation of API

### What's good

- **Headers are heavily commented.** Nearly every struct, member, and function
  has a doxygen-style block explaining intent, defaults, and AT2 parity. For a
  domain-heavy library this is valuable.

- **`__all__` in `__init__.py` is explicit** (line 29-53). Python users can
  `from steppe import *` and get the intended surface.

- **The design doc (`cli-bindings.md`) is a readable contract.** It maps CLI
  flags to C++ types, Python kwargs, and result shapes.

### What a senior API designer would flag

- **There is no generated API reference.** Doxygen/Sphinx/Breathe is not wired
  (no `Doxyfile`, no `docs/api/`). A user must read headers or the design doc
  to learn the API. For an A+ library, generated HTML docs with cross-references
  are expected.

- **The header comments are too long for quick scanning.** `config.hpp` is 553
  lines, ~70% comments. The signal-to-noise ratio is low for a user who just
  wants to know "how do I call qpadm?" Comments should be tiered: a one-line
  brief, then an optional detailed block.

- **No quick-start example in the repo.** There is no `examples/` directory with
  a minimal `qpadm` script in C++ and Python. The design doc has snippets
  (`cli-bindings.md §5.1`), but they are embedded in prose.

- **`backend.hpp` comments are excellent for maintainers but overwhelming for
  new library users.** Because it is internal this is less critical, but the
  sheer length (1857 lines) makes the seam hard to navigate.

- **Error messages are not user-tested.** The binding raises `ValueError` for
  almost every fault (`module.cpp:89, 200-201, 700-701`). A user who passes a
  bad precision string gets a clear message (`module.cpp:177-179`), but a CUDA
  OOM is also a `ValueError`. More specific exception types (`MemoryError`,
  `RuntimeError`, custom `SteppeError`) would improve debugging.

---

## 9. What would make this an A+ library API

Concrete improvements, ordered by impact:

1. **Introduce a stable C ABI layer for the installed boundary.**
   - Add `include/steppe/steppe_c.h` with opaque `steppe_f2_t`,
     `steppe_result_t`, and `steppe_status_t`.
   - Keep the current C++ convenience headers for in-process use, but mark
     them as "same-toolchain only" in docs.
   - This addresses ABI stability (§2) and separation (§4).

2. **Refactor result structs into composed, self-documenting tables.**
   - In `qpadm.hpp`, nest `WeightsTable { target, left, weight, se, z }`,
     `RankDropTable`, `PopDropTable` inside `QpAdmResult`.
   - In `qpgraph.hpp`, nest `AdmixtureWeights` and `DriftEdges`.
   - Add checked row accessors (`weights.at(i)`, `rankdrop.at(r)`) and
     `empty()` / `size()` helpers.

3. **Add named factories for `Precision`.**
   ```cpp
   // config.hpp
   struct Precision {
       static Precision fp64() noexcept;
       static Precision emulated_fp64(int bits = kDefaultMantissaBits);
       static Precision tf32() noexcept;
       // ... existing members ...
   };
   ```
   This removes the `Precision{Kind::EmulatedFp64, 40}` incantation from user
   code.

4. **Split search-specific options from single-fit options.**
   - Create `QpAdmSearchOptions : QpAdmOptions { JackknifePolicy jackknife; ... }`.
   - Remove `jackknife`, `p_se_threshold`, `se_require_p` from `QpAdmOptions`
     (`qpadm.hpp:57-100`) so single-fit callers are not confused.

5. **Introduce a public `ConfigBuilder` / validation layer.**
   - As proposed in `cli-bindings.md §9 question 1`.
   - Provide `DeviceConfig::validate() -> std::optional<std::string>` and
     `FilterConfig::validate()` so callers can fail fast before allocating GPUs.

6. **Improve Python API ergonomics.**
   - Remove polymorphic return types: `qpadm_search(...)` always returns
     `list[QpAdmResult]`; add `qpadm_search_table(...)` for the DataFrame.
   - Same for `f4`/`f3`/`f4ratio`/`qpdstat`/`dstat`: remove `as_dataframe`;
     users call `.table`.
   - Add `F2Blocks.close()` and context-manager support (`with steppe.read_f2(...)`).
   - Replace raw `new F2Handle` with a `std::unique_ptr` transfer helper
     (`module.cpp:397, 847, 910`).
   - Use `std::memcpy` / `std::copy` in `f2_to_numpy` (`module.cpp:1031-1032`).
   - Map `std::bad_alloc` to `MemoryError` and CUDA runtime errors to a custom
     `SteppeDeviceError` instead of blanket `ValueError`.

7. **Generate API documentation.**
   - Add a Doxygen/Sphinx/Breathe target.
   - Publish a quick-start example under `examples/` showing C++ and Python
     `read_f2` → `qpadm` → inspect `weights`/`p`.

8. **Tighten public enum ABI.**
   - Add explicit underlying types: `enum class Status : int`,
     `enum class Precision::Kind : int`, `enum class JackknifePolicy : int`
     (already done; extend to all public enums).

9. **Add indexing helpers to `F2BlockTensor`.**
   ```cpp
   double f2_at(int i, int j, int b) const noexcept;
   double& f2_at(int i, int j, int b);
   std::span<const double> block(int b) const; // [P*P]
   ```
   This makes the documented layout (`fstats.hpp:35-46`) usable without
   hand-rolled math.

10. **Clarify ownership in the public docs.**
    - State explicitly that `F2BlockTensor` owns host memory.
    - State that `DeviceF2Blocks` is a private RAII handle borrowed by `run_*`
      functions.
    - State that the Python wheel requires a CUDA-capable GPU and does not
      provide a CPU runtime path.

---

## Verdict

The steppe public API is **solid B+ work**: the CUDA-free layering is correct,
the domain-outcome model is right, the host-oracle parity overloads are
excellent, and the Python/C++ split is well conceived. It is not yet an A+
library API because the result structs are too flat, the Python surface has
polymorphic return types, the bindings leak C-isms (raw `new`, manual copy
loops), there is no stable installed ABI, and the documentation, while
extensive, is not yet generated or example-driven. Ship after addressing the
high-impact items in §9; the C ABI layer and the result-struct refactoring are
the two highest-leverage changes.
