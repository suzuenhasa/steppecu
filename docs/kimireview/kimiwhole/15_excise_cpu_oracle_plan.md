# Plan: Excise the CPU Oracle from Mainline Code

This plan moves the CPU backend from a shipping abstraction into an internal test-only oracle. The production codebase will expose only the CUDA path. Parity testing continues, but against an oracle that lives in `tests/oracle/` rather than `src/device/`.

---

## 1. Goal and non-goals

### Goal

- Mainline `src/` contains only the production CUDA compute path.
- `ComputeBackend` is a GPU-focused seam, not a CPU/GPU dual abstraction.
- Public headers (`include/steppe/*.hpp`) expose only the CUDA-backed API.
- Parity tests still compare GPU output to a CPU oracle, but the oracle is a test fixture.

### Non-goals

- We are not deleting the CPU math. It stays as the correctness reference.
- We are not changing any numeric results.
- We are not rewriting CUDA kernels. Their comments can still cite the CPU oracle.

---

## 2. High-level file moves

```
Current                                          Target
─────────────────────────────────────────────────────────────────────────────
src/device/cpu/cpu_backend.cpp          →      tests/oracle/cpu_oracle.cpp
src/device/backend_factory.hpp          →      src/device/backend_factory.hpp
                                                 (remove make_cpu_backend)
src/device/backend.hpp                  →      src/device/backend.hpp
                                                 (slim to GPU-only virtuals)
include/steppe/f4.hpp etc.              →      include/steppe/f4.hpp etc.
                                                 (remove host-oracle overloads)
src/core/qpadm/f4.cpp etc.              →      src/core/qpadm/f4.cpp etc.
                                                 (remove host-oracle branches)
tests/reference/test_f2_equivalence.cu  →      tests/reference/test_f2_equivalence.cu
                                                 (use tests/oracle helpers)
```

---

## 3. Step-by-step plan

### Phase 1 — Create the test-only oracle tree

Create:

```
tests/oracle/
  ├── cpu_oracle.hpp          // public oracle interface for tests
  ├── cpu_oracle.cpp          // moved+trimmed cpu_backend.cpp
  ├── qpadm_oracle.hpp        // qpAdm-specific oracle helpers
  ├── qpadm_oracle.cpp
  ├── qpgraph_oracle.hpp
  ├── qpgraph_oracle.cpp
  ├── fstat_oracle.hpp        // f4/f3/f4-ratio
  ├── fstat_oracle.cpp
  └── detail/                 // shared oracle internals
      ├── small_linalg.cpp
      ├── small_linalg.hpp
      └── ...
```

**Action:** Move `src/device/cpu/cpu_backend.cpp` to `tests/oracle/cpu_oracle.cpp`.

**Trimming rule:** Remove anything that implements the `ComputeBackend` interface. The oracle does not need virtuals, `device_id`, stream handling, or capability reporting. It becomes a set of free functions:

```cpp
// tests/oracle/cpu_oracle.hpp
#pragma once
#include <steppe/fstats.hpp>
#include <steppe/f3.hpp>
#include <steppe/f4.hpp>
#include <steppe/f4ratio.hpp>
#include <steppe/qpadm.hpp>
#include <steppe/qpgraph.hpp>
#include <span>

namespace steppe::oracle {

[[nodiscard]] F4Result compute_f4_oracle(
    const F2BlockTensor& f2,
    std::span<const Quartet> quartets);

[[nodiscard]] F3Result compute_f3_oracle(
    const F2BlockTensor& f2,
    std::span<const Triple> triples);

[[nodiscard]] F4RatioResult compute_f4ratio_oracle(
    const F2BlockTensor& f2,
    std::span<const F4Ratio> ratios);

[[nodiscard]] QpAdmResult fit_qpadm_oracle(
    const F2BlockTensor& f2,
    const QpAdmOptions& opts);

[[nodiscard]] QpGraphResult fit_qpgraph_oracle(
    const F2BlockTensor& f2,
    const QpGraphOptions& opts);

// etc.

} // namespace steppe::oracle
```

### Phase 2 — Slim `ComputeBackend`

Edit `src/device/backend.hpp`.

**Remove** virtuals that exist only for the CPU oracle:

- `assemble_f4(host F2BlockTensor)` overloads.
- `assemble_f4_quartets(host F2BlockTensor)` overloads.
- `assemble_f3_triples(host F2BlockTensor)` overloads.
- `fit_qpadm(host F2BlockTensor)` overloads.
- `fit_qpgraph(host F2BlockTensor)` overloads.
- `decode_af_compact_*` host-pointer overloads.
- Any method whose comment says "CpuBackend oracle door" or "host-oracle overload".

**Keep** only the production GPU virtuals:

- Device-resident f2 methods.
- Device decode methods.
- Device fit/sweep methods.
- Capability queries that matter for GPU scheduling.

**Rename consideration:** You may keep `ComputeBackend` as the name, or rename it to `CudaBackendInterface` / `GpuBackend`. Keeping `ComputeBackend` is less churn, but renaming makes the CPU excision explicit.

### Phase 3 — Remove CPU factory from public API

Edit `src/device/backend_factory.hpp`.

**Remove:**

```cpp
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cpu_backend();
```

**Keep:**

```cpp
[[nodiscard]] std::unique_ptr<ComputeBackend> make_cuda_backend(int device_id = 0);
[[nodiscard]] int visible_device_count();
```

If any mainline code still calls `make_cpu_backend()`, that is now a compile error — fix it to call the oracle from `tests/oracle/` or remove the branch.

### Phase 4 — Clean public headers

For each of these files, remove the host-oracle overload:

- `include/steppe/f4.hpp`
- `include/steppe/f3.hpp`
- `include/steppe/f4ratio.hpp`
- `include/steppe/qpadm.hpp`
- `include/steppe/qpwave.hpp`
- `include/steppe/qpgraph.hpp`
- `include/steppe/qpgraph_search.hpp`
- `include/steppe/dates.hpp`
- `include/steppe/qpfstats.hpp`

**Before:**

```cpp
[[nodiscard]] F4Result run(const Resources& r,
                           const F2BlockTensor& f2,
                           std::span<const Quartet> quartets);

/// HOST-ORACLE overload...
[[nodiscard]] F4Result run(const ComputeBackend& be,
                           const F2BlockTensor& f2,
                           std::span<const Quartet> quartets);
```

**After:**

```cpp
[[nodiscard]] F4Result run(const Resources& r,
                           const F2BlockTensor& f2,
                           std::span<const Quartet> quartets);
```

The public API now always goes through `Resources` (which holds the CUDA backend).

### Phase 5 — Remove host-oracle branches from `core`

Edit these `core` files and delete the host-oracle code paths:

- `src/core/qpadm/f4.cpp` — remove the `run_f4_impl(const ComputeBackend&, const F2BlockTensor&, ...)` overload or convert it to use `Resources`.
- `src/core/qpadm/f3.cpp` — same.
- `src/core/qpadm/f4ratio.cpp` — same.
- `src/core/qpadm/qpgraph_fit.cpp` — same.
- `src/core/qpadm/qpadm_fit.cpp` — same.
- `src/core/stats/dstat.cpp` — remove CPU-oracle branches.
- `src/core/stats/qpfstats.cpp` — remove CPU-oracle branches.
- `src/core/stats/dates.cpp` — remove CPU-oracle branches.

**Pattern:** Find blocks like:

```cpp
// S3 — host-oracle assemble (the CpuBackend reads host memory directly).
```

and replace them with a single production path:

```cpp
auto f4blocks = backend.assemble_f4_quartets(dev_f2, quartets);
```

If a helper function in `core` was only used by the CPU path, delete it or move it to `tests/oracle/`.

### Phase 6 — Update reference tests

Every `tests/reference/*.cu` file that does:

```cpp
#include "device/backend_factory.hpp"
auto cpu = steppe::device::make_cpu_backend();
```

changes to:

```cpp
#include "oracle/cpu_oracle.hpp"
```

and:

```cpp
auto cpu_res = steppe::oracle::compute_f4_oracle(f2, quartets);
auto gpu_res = steppe::run(resources, f2, quartets);
diff(cpu_res, gpu_res);
```

The test structure stays the same; only the namespace and call site change.

### Phase 7 — Update CMake

#### `src/device/CMakeLists.txt`

Remove:

```cmake
src/device/cpu/cpu_backend.cpp
```

from the `steppe_device` sources.

#### `tests/CMakeLists.txt`

Add a new test target:

```cmake
add_library(steppe_oracle STATIC
    oracle/cpu_oracle.cpp
    oracle/qpadm_oracle.cpp
    oracle/qpgraph_oracle.cpp
    oracle/fstat_oracle.cpp
    oracle/detail/small_linalg.cpp
    # ... other oracle internals
)

target_link_libraries(steppe_oracle PUBLIC steppe_core)
```

Link reference tests against `steppe_oracle`:

```cmake
target_link_libraries(test_f2_equivalence PRIVATE steppe_oracle steppe_device)
```

#### `src/core/CMakeLists.txt`

Ensure `core` no longer links anything CPU-backend-specific. It should only need `steppe_device` (for the abstract `ComputeBackend` interface) and `steppe_f2` (once f2 engine exists).

### Phase 8 — Clean comments and docs

- CUDA kernel `.cuh` comments can still reference `cpu_oracle.cpp` as the spec. That is documentation, not coupling.
- Update `architecture.md` to state that the CPU oracle is test-only.
- Remove or rewrite comments in `core` and `device` that say "CpuBackend oracle door" — they are now misleading.

---

## 4. What breaks and how to fix it

| Breakage | Fix |
|---|---|
| `make_cpu_backend()` no longer exists | Reference tests include `oracle/cpu_oracle.hpp` and call free functions. |
| Host-oracle overloads removed from public headers | Tests call oracle directly; mainline code always uses `Resources`. |
| `ComputeBackend` slimmed | CPU-only virtuals deleted; remaining virtuals are GPU-only. |
| `core` files had CPU branches | Delete branches; keep single production path. |
| `backend.hpp` no longer includes CPU symbols | Move shared math helpers (`small_linalg.hpp`, etc.) to `core/internal/` or `tests/oracle/detail/`. |
| CMake link errors | Add `steppe_oracle` target and link tests to it. |

---

## 5. Resulting architecture

```
Production mainline:
  src/app/        → CLI commands (use Resources)
  src/core/       → pure math orchestration (use ComputeBackend + F2BlockTensor)
  src/device/     → CUDA backend only
  src/f2/         → f2 engine (future)
  include/steppe/ → public C++ API (CUDA-backed only)

Test-only:
  tests/oracle/   → CPU oracle free functions for parity testing
  tests/reference/→ parity tests that compare oracle vs. CUDA
```

The public story becomes:

> "`steppe` is a CUDA-accelerated population-genetics compute engine. The CPU implementation is retained internally as a correctness oracle for our test suite."

---

## 6. Effort estimate

- Phase 1 (create oracle tree): 1 day
- Phase 2 (slim backend): 1–2 days
- Phase 3–4 (factory + public headers): 1 day
- Phase 5 (core cleanup): 2–3 days
- Phase 6–7 (tests + CMake): 2 days
- Phase 8 (comments + docs): 1 day

**Total: roughly 1–1.5 weeks of focused refactoring.**

Most of the work is mechanical deletion and namespace changes. The risk is low because the numeric math does not change.

---

## 7. Optional: go further

If you want to make the excision even cleaner:

- **Rename `ComputeBackend` to `CudaBackend`.** Makes it impossible for future code to reintroduce a CPU implementation as a "backend."
- **Remove the abstract interface entirely.** If the only production implementation is CUDA, you could have `core` call `CudaBackend` directly. This removes one abstraction layer but reduces test flexibility.
- **Move `tests/oracle/` to a separate repo or submodule.** If the oracle grows large, keep it out of the main repo entirely. Not recommended while you're actively iterating.

The minimal version above is enough to make the architecture honest.
