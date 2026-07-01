# Wave 2 Code Review Synthesis

## 1. Executive Summary

This wave covers 13 files across the Steppe C++/CUDA project and the overall quality is **competent, work-in-progress systems/research code — mostly B territory, none of it slop, but none of it senior-dev polished either.** The authors clearly understand the genomics domain, the project's frontend/backend split, and the CUDA/host boundary. The math is generally faithful to ADMIXTOOLS-2, memory ownership is mostly modern (vectors, spans, unique_ptr), and the architecture comments show real design intent.

The most common issues are **contract inconsistencies around error handling** and **C-isms that slip through a C++20 codebase.** Several files declare a "status-as-value, never exceptions" architecture, then turn around and use `std::vector::at()` (which throws), swallow `std::runtime_error` broadly, or return half-initialized objects with an `.error` string. Secondary but pervasive: **over-commenting and copy-paste drift** — many files are saturated with scaffolding notes, cleanup-ticket citations, and repeated boilerplate that should have been factored out. The lowest-grade file (`cpu_backend.cpp`, B-) is penalized for being a 2,200-line monolith with duplicated assembler logic and hidden mutable state.

In short: the bones are good, the algorithms are trustworthy, but before this goes in front of a senior staff engineer it needs one hardening pass on error boundaries, one refactoring pass on duplication, and one editorial pass on comments.

---

## 2. Cross-Cutting Themes

- **Exception-vs-Status schizophrenia.** `dates.cpp`, `f3.cpp`, `qpgraph_fit.cpp`, and especially `qpadm_fit.cpp` all talk about returning `Status` values, but then use `.at(kPrimaryGpu)` (throws `std::out_of_range`), or `catch (const std::runtime_error&)` and silently continue. Pick one model and make it honest.
- **Unguarded primary-backend access.** `primary_backend()` or `resources.gpus[0].backend` appears in at least four files (`dates.cpp`, `f4.cpp`, `f3.cpp`, `qpgraph_fit.cpp`) with no null/empty check. In a project that claims GPU-backed compute is optional/configurable, this is a footgun.
- **`std::nan("")` as a C-ism.** Repeated in `dates.cpp`, `f4.cpp`, `f3.cpp`, `qpgraph_fit.cpp`, and `cpu_backend.cpp`. It works, but `std::numeric_limits<double>::quiet_NaN()` is the idiomatic C++20 spelling; mixing the two in one TU is sloppy.
- **Cast noise and signed/unsigned drift.** `qpgraph_model.cpp` is the worst offender, but `model_search_core.cpp` narrows `std::size_t` to `int` for a device index, and `qpgraph_fit.cpp` is littered with `static_cast<std::size_t>` around every index. This usually signals that the data structure should have been signed all along.
- **Copy-paste drift.** `cpu_backend.cpp` has three near-identical f4/f3 assemblers; `geno_reader.cpp` repeats the same selection/reorder block across four SNP-major readers; `plink_reader.cpp` duplicated `read_ind`/`read_fam` logic but introduced a real cap bug; `module.cpp` has cloned `run_f4_py` / `run_qpdstat_py` bodies. This is the project's biggest maintainability risk.
- **Comment bloat and defensive scaffolding.** `snp_reader.cpp`, `cpu_backend.cpp`, and `module.cpp` cite cleanup tickets, markdown sections, and spike-risk essays inline. Useful during refactor; noise in committed code.
- **C-isms in C++20 bindings/entry points.** `module.cpp` uses raw `new F2Handle()` with nanobind ownership transfer; `main.cpp` uses `std::fprintf` instead of `<iostream>`. Both work, both look dated.
- **Hidden state and stale contract drift.** `cpu_backend.cpp` has `tot_line_` set only inside `assemble_f4*` and read inside `jackknife_*`, and its `survivor_blocks` rule is documented differently in two places. These are exactly the bugs that pass tests today and explode next quarter.

---

## 3. Per-File Verdict Table

| File | Grade | Key Concern |
|---|---|---|
| `src/core/stats/dates.cpp` | B+ | `STEPPE_ASSERT` used for user-input validation; unchecked backend dereference at line 141. |
| `src/core/qpadm/qpadm_fit.cpp` | B | Swallows all `std::runtime_error`s at lines 187–190; mixes Status and exceptions. |
| `src/core/qpadm/f4.cpp` | B+ | Unguarded `primary_backend`; stale `jackknife_cov` vs `jackknife_diag` comment. |
| `src/core/qpadm/f3.cpp` | B+ | `primary_backend` can throw despite status-value claim; downstream seam status unobserved. |
| `src/core/qpadm/model_search_core.cpp` | B+ | `G == 0` throws `std::runtime_error`; narrows `size_t` device index to `int`. |
| `src/core/qpadm/qpgraph_fit.cpp` | B+ | Unchecked `gpus[0]` assumption; heavy cast noise; C-style `std::nan("")`. |
| `src/core/qpadm/qpgraph_model.cpp` | B+ | Relentless `static_cast<std::size_t>` noise; `std::function` for local recursive DFS. |
| `src/device/cpu/cpu_backend.cpp` | B− | 2,200-line monolith; copy-paste f4/f3 assemblers; hidden `tot_line_` state. |
| `src/io/geno_reader.cpp` | B+ | Copy-pasted selection/reorder boilerplate across four reader functions. |
| `src/io/plink_reader.cpp` | B | `read_fam` increments `row` past `n_records_present`, inflating total count vs. `.bed`. |
| `src/io/snp_reader.cpp` | B+ | Comment bloat; redundant `SnpTable::count`; `string_view` split not used. |
| `src/app/main.cpp` | B+ | Uses `std::fprintf` in a C++20 file; catch-all swallows exception identity. |
| `bindings/module.cpp` | B+ | Raw `new F2Handle()` transfers; duplicated precision-tag ternary; cloned `run_f4_py`/`run_qpdstat_py`. |

---

## 4. Top 5 Concrete Fixes Before Senior Review

### 1. `src/core/qpadm/qpadm_fit.cpp` lines 187–190 — stop swallowing `std::runtime_error`
The review calls this "the worst thing in the file" and a "textbook footgun." The current code catches *every* `std::runtime_error` and ignores it, papering over OOM, corrupt covariance, or logic bugs just to handle a temporary "not implemented" backend path.

```cpp
} catch (const std::runtime_error&) {
    // backend has no rank_sweep override yet ...
}
```

**Fix:** Introduce a typed `NotImplementedError` (or better, make `run_rank_sweep` / `run_popdrop` return a `Status` or `std::optional<Result>`) and catch only that. Do not use exceptions for feature flags.

### 2. `src/io/plink_reader.cpp` lines 242–245 — fix the partial-file row-count bug
`read_fam` increments `row` for records beyond `n_records_present`, but `read_ind` does not. This makes `part.n_individuals_total` exceed the `.bed` axis for partial files, directly contradicting the header contract.

```cpp
if (row >= n_records_present) {
    ++row;          // BUG: inflates total past .bed axis
    continue;
}
```

**Fix:** Remove the `++row;` (or mirror `ind_reader.cpp` exactly), and add a test that feeds a `.fam` longer than the `.bed`.

### 3. `src/device/cpu/cpu_backend.cpp` — extract a shared f4/f3 assembler helper
The file repeats the same survivor-block setup, `f2at` lambda, compaction loop, and `compute_loo_and_total` call across `assemble_f4`, `assemble_f4_quartets`, and `assemble_f3_triples` (roughly lines 1001–1191). This is ~80 lines of duplication and the source of drift.

**Fix:** Factor the common skeleton into a private template/helper that takes the per-mode index mapping as a policy. This also eliminates the stale-comment risk between the three functions.

### 4. Guard `primary_backend` / `gpus[0]` access everywhere
Multiple files (`dates.cpp:141`, `f4.cpp:44`, `f3.cpp:47`, `qpgraph_fit.cpp:39-42`) assume `resources.gpus` is non-empty and `.backend` is non-null:

```cpp
return *resources.gpus.at(kPrimaryGpu).backend;
```

**Fix:** Either return `Status::NoDevice` / `InvalidConfig` from these helpers, or assert with a clear message and document that the caller must validate. Do not let `.at()` throw if the rest of the file claims status-as-value semantics.

### 5. `bindings/module.cpp` lines 397, 847, 910 — replace raw `new F2Handle()` with a centralized owned-pointer helper
Raw `new` scattered through binding entry points is a leak/exception-safety risk and looks like 2012-era pybind:

```cpp
auto* h = new F2Handle();
```

**Fix:** Add a small factory (`F2Handle* make_f2_handle(...)` returning the pointer, or a `std::unique_ptr` with a nanobind-aware deleter) so ownership transfer happens in one place and exceptions before the cast cannot leak. While in the file, also collapse the duplicated `precision_tag` ternary and deduplicate `run_f4_py` / `run_qpdstat_py`.

---

## 5. Honest Bottom Line

None of these files will make a senior developer walk out of the room. The math is right, the architecture is coherent, and the ownership model is mostly modern. But the wave is carrying a consistent set of rough edges: **error-handling contracts that aren't self-consistent, defensive comments that substitute for cleaner code, and duplicated blocks that have already started to drift.** The two files that most urgently need attention before any showcase are `qpadm_fit.cpp` (the swallowed exception) and `cpu_backend.cpp` (the monolithic oracle with hidden state and copy-paste assemblers). Fix those five items above and this moves from "competent research code" to "code you can confidently hand to a staff engineer."
