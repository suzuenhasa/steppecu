# steppe — NAMING / STYLE STANDARD

Status: DRAFT for user approval. Governs the **Phase C** big-refactor cleanup (the ~480 LOW
findings). Authored read-only from the codebase and cross-checked against the big-refactor
review's per-unit findings under `docs/cleanup/bigrefactor/findings/`.

---

## 1. Purpose & scope

This document is the rulebook for resolving the Phase-C **LOW** findings **consistently**:

- **Group 6** — naming (the body of the work; ~51 live LOW after Phase B)
- **Group 5** — magic numbers / unnamed literals (residual LOW are naming-shaped)
- **Group 9** — config / hoist-to-`constexpr` / positional booleans
- **Group 10** — initialization (4 items)
- **Group 3** — dead code / stray includes (LOW items)

It is **DESCRIPTIVE first**: it codifies the conventions steppe already follows so cleanups
land *toward* the existing dominant spelling, not toward a new one. It is **prescriptive only**
where the review found genuine drift (the items tagged **MIXED** below). The codebase is already
strongly self-consistent (30/61 units clean on naming, 47/61 on config, 57/61 on init), so the
job is to bring stragglers into line — **not** to restyle the tree.

Each rule is tagged:
- **[DOMINANT]** — the code already does this near-universally; codify and bring stragglers in.
- **[MIXED]** — a real inconsistency the review flagged; this is where the standard makes a call.

> **Reference findings by CONTENT, not by line number.** Phase B shifted `cuda_backend.cu` by
> ~+200 lines vs the findings docs. Line numbers below are current-as-mined spot checks, not
> stable anchors.

---

## 2. THE CONVENTIONS

### 2.1 Types — `PascalCase` [DOMINANT]

All `struct` / `class` / `enum class` names are `PascalCase`. No exceptions found.

```cpp
struct  Precision;            // include/steppe/config.hpp
struct  QpAdmResult;          // include/steppe/qpadm.hpp
struct  F2Result, F4Blocks;   // src/device/backend.hpp
class   CublasHandle;         // src/device/cuda/handles.hpp
class   ComputeBackend;       // src/device/backend.hpp
```

**Rule:** known domain acronyms are **title-cased inside** the PascalCase name, never all-caps:
`qpAdm → QpAdm`, `GLS → Gls`, `f2/f4 → F2/F4`, `cuSOLVER-Dn → CusolverDn`.
So `F2Result`, `GlsWeights`, `CusolverDnHandle`, `GesvdjInfo` — never `F2RESULT` or `GLSWeights`.
*Rationale:* one casing rule for every type makes types greppable and unambiguous at the seam.

### 2.2 Functions / methods / free functions — `snake_case` [DOMINANT]

Every free function, member function, and `constexpr` helper is `snake_case`. No `camelCase`
methods anywhere.

```cpp
run_qpadm(...);  run_qpwave(...);          // include/steppe/qpadm.hpp
compute_f2_blocks_device(...);             // src/device/backend.hpp (virtual)
gls_weights_loo_batched(...);              // src/device/backend.hpp
read_snp(...);  parse_geno_header(...);    // src/io/*
get();  device_id();  status_name();       // accessors (handles.hpp, check.cuh)
constexpr cdiv(...);  constexpr grid_for(...);  // src/core/internal/launch_config.hpp
```

**Rule:** acronyms are **lowercase** in function names: `gls_weights`, `assemble_f4`,
`qpadm_dof`, `pchisq_upper`. *Rationale:* matches AT2/BLAS verb vocabulary and keeps the seam
diffable against the R/AT2 source.

### 2.3 Variables & parameters — `snake_case` [DOMINANT]

Descriptive locals/params are `snake_case`: `block_sizes`, `block_offsets`, `bucket_members`,
`compute_se`, `weights_feasible`, `inv_pivot`.

**Single letters are allowed ONLY for** (a) AT2 / linear-algebra **math dimensions** or
(b) **tight in-loop counters**. See §3.2 for the protected-vocabulary allowlist and §4 for the
loop-counter exception. A *derived* value read far from its definition gets a descriptive name
(this is the largest Group-6 theme — see §4 DO/DON'T).

### 2.4 Struct / class data members — TWO sub-conventions, both [DOMINANT]

This split is **load-bearing** and Phase C must preserve it:

- **2.4a — Public / POD struct fields: `snake_case`, NO trailing underscore.** These are
  value-type contract fields crossing the seam.
  ```cpp
  int mantissa_bits;  int stream_count;  bool enable_peer_access;   // config.hpp
  int model_index;    double rank_chisq;                            // qpadm.hpp
  std::size_t n_block; int block_id; std::vector<...> pop_offsets;  // backend.hpp
  ```
- **2.4b — Private class members: `snake_case_` WITH trailing underscore.** Encapsulated state
  on behaviorful classes (RAII wrappers, the pimpl).
  ```cpp
  h_, ws_, ws_bytes_, device_id_, prev_, info_, promoted_;          // handles.hpp
  stream_, blas_, solver_, workspace_, solve_precision_,            // cuda_backend.cu Impl
  batched_dispatch_count_, tot_line_, pinned_in_, stage_f2_;
  ```

**Rule:** the trailing `_` is the marker of an *encapsulated private class member*; plain
`snake_case` is for *public POD/struct fields*. **Phase C must not add `_` to a POD field nor
strip it from a class member.** *Rationale:* the `_` instantly distinguishes owned mutable state
from a value-contract field at the use site.

### 2.5 Constants / `constexpr` — `kPascalCase`, `inline constexpr`, homed in `config.hpp` [DOMINANT]

Every named *value* constant is `k` + PascalCase, declared `inline constexpr`, single-homed in
`include/steppe/config.hpp` (domain bounds live in their domain header). Measured **40/40 = 100%**
of actual value-constants follow this. Every constant carries a `///` doc stating value, units,
parity status, and which spike literal it replaced.

```cpp
inline constexpr int         kDefaultMantissaBits   = 40;                  // config.hpp:44
inline constexpr int         kGesvdjMaxDim          = 32;                  // config.hpp:94
inline constexpr std::size_t kCublasWorkspaceBytes  = 64u*1024u*1024u;     // config.hpp:100
inline constexpr std::size_t kFitBudgetHeadroomBytes = (std::size_t)512<<20; // config.hpp:113
inline constexpr int kQpMaxNl, kQpMaxNr, kQpMaxR;     // src/core/qpadm/qpadm_bounds.hpp
```

**Rules:**
- A `constexpr` **function** is `snake_case`, not `k`-prefixed (it is a function, not a value):
  `resident_tensor_bytes(...)`, `per_block_chunk_bytes(...)` in `vram_budget.hpp`. Correct as-is.
- **Bake the unit into the name** where ambiguous: `kDefaultBlockSizeCm`, `kCublasWorkspaceBytes`,
  `kFitBudgetHeadroomBytes`.
- **Constants are NOT macros.** Numeric/policy constants use `kPascalCase inline constexpr`,
  never an `ALL_CAPS` `#define`.
- Compile-time constants must be `inline constexpr`, **not** `const`. [MIXED — Group 9.1]:
  `pchisq.hpp` has a few `const int kMaxIter` / `const double kEps` that should become
  `inline constexpr` (name already correct).
- **Single-source the value.** A LOW rename must never fork a single-homed constant
  (`kQpMax*`, `kGesvdjMaxDim`, `qpadm_dof()`).

### 2.6 Enums — `enum class`, `PascalCase` type, `PascalCase` enumerators [DOMINANT]

Always scoped (`enum class`); never bare `enum`. Enumerators are `PascalCase`.

```cpp
enum class Kind        { Fp64, EmulatedFp64, Tf32 };           // config.hpp
enum class ForceTier   { Auto, Resident, HostRam, Disk };      // config.hpp
enum class JackknifePolicy : int { None=0, FeasibleOnly=1, All=2 };  // qpadm.hpp
```

**Rule:** add an explicit `: int` and explicit values **only** when the integer mapping is
user-facing or frozen (e.g. `JackknifePolicy`). *Rationale:* scoped enums prevent implicit
int conversions and namespace the enumerators.

### 2.7 Namespaces — nested `steppe::<layer>::<sub>`, flattened form, commented close [DOMINANT]

```cpp
namespace steppe::device { ... }  // C++17 flattened — NOT nested steppe { device { } }
}  // namespace steppe::device     // close always commented, two spaces before //
}  // namespace                    // anonymous (TU-private)
}  // namespace detail             // impl-only helpers behind macros
```

Layers: `steppe` (public), `steppe::core` / `::core::qpadm` / `::core::internal`,
`steppe::device`, `steppe::io` / `::io::filter`. `detail` (nested in a layer) holds the
free-function machinery behind the CHECK macros (`steppe::device::detail::cuda_check`).
Anonymous `namespace {` holds TU-private kernels/helpers.

**One sanctioned special case:** `namespace core::qpadm {` (relative, dropping the leading
`steppe`) is allowed **only** when already inside `namespace steppe {` (a forward-decl seam in
`backend.hpp`). Flattened-absolute is the default.

### 2.8 Files — `snake_case`, extension by CUDA-visibility role [DOMINANT]

| Ext    | Role | Examples |
|--------|------|----------|
| `.hpp` | host-pure / CUDA-free header | `config.hpp`, `backend.hpp`, `snp_reader.hpp` |
| `.cuh` | device-private CUDA header   | `check.cuh`, `qpadm_fit_kernels.cuh`, `device_buffer.cuh` |
| `.cu`  | CUDA TU (kernel bodies + `<<<>>>`) | `cuda_backend.cu`, `f2_block_kernel.cu` |
| `.cpp` | host TU | `qpadm_fit.cpp`, `cpu_backend.cpp`, `geno_reader.cpp` |

**Rule:** the kernel body and any `<<<>>>` live **only** in a `.cu`. A header/impl pair shares
its base name (`f2_block_kernel.cuh` ↔ `.cu`). Documented carve-out: `handles.hpp` is `.hpp`
**despite** including `cublas_v2.h` (matches architecture.md §4's table) — but it is still
PRIVATE to `steppe_device` and says so in its header.

### 2.9 CUDA kernels, launch wrappers & `.cu`/`.cuh` pairing [DOMINANT]

The seam is a fixed triple:

```
launch_<op>(...)        // host wrapper, void return, declared in .cuh, defined in .cu
   └─ <op>_kernel<<<>>> // __global__, snake_case + _kernel suffix, body in .cu only
        └─ dev_<helper> // __device__ helper, dev_ prefix
```

```cpp
void launch_decode_af(...);   void launch_qpadm_fit_models_batched(...);   // *.cuh
__global__ void decode_af_kernel(...);  __global__ void als_kernel(...);   // *.cu
__device__ dev_lu_factor(...);  __device__ dev_jacobi_svd_V(...);          // *.cu
```

**Rules:**
- `launch_*` = a wrapper that issues `kernel<<<grid,block,0,stream>>>(...)` and ends with
  `STEPPE_CUDA_CHECK_KERNEL()`. `run_*` = a wrapper that issues **library** calls
  (cuBLAS/cuSOLVER), no custom `<<<>>>` (e.g. `run_f2_gemms`).
- `__launch_bounds__(N)` sits on the line above the kernel name, pinned to the kernel's sole
  launch block size.
- Model-batched rotation variants append `_models` / `_models_batched` to **both** kernel and
  wrapper (`launch_qpadm_loo_models_batched` → `qpadm_loo_models_kernel`).
- [MIXED — Group 6.3] **`launch_` wrapper params keep the `d`-prefix** (they are device pointers
  visible to host code); a `__global__` kernel MAY drop it (everything inside a kernel is device).

### 2.10 Macros — `STEPPE_` prefix, `ALL_CAPS` [DOMINANT]

All project macros are `STEPPE_`-prefixed and `ALL_CAPS`.

```cpp
STEPPE_CUDA_CHECK   STEPPE_CUDA_CHECK_KERNEL   STEPPE_CUDA_WARN   // check.cuh
STEPPE_HD   STEPPE_ASSERT   STEPPE_DEBUG_ONLY                     // host_device.hpp
STEPPE_LOG_WARN                                                   // log.hpp
```

- **Include guards:** `STEPPE_<PATH_IN_UPPER_SNAKE>_<EXT>`, mirroring the repo path; `.cuh`
  uses the `_CUH` suffix. `#endif  // STEPPE_..._HPP` (two spaces before `//`).
  e.g. `STEPPE_DEVICE_CUDA_HANDLES_HPP`, `STEPPE_DEVICE_CUDA_CHECK_CUH`.
- **Documented carve-out — DO NOT "fix":** `CUBLAS_CHECK` / `CUSOLVER_CHECK` are **NOT**
  `STEPPE_`-prefixed (check.cuh:233,240). This is intentional (mirrors the spike vocabulary;
  there is exactly one of each). Leave them.
- **Rule:** new macros are `STEPPE_`-prefixed `ALL_CAPS`; **numeric constants are not macros**
  (use §2.5).

### 2.11 Doc-comments & architecture.md `§`-cross-refs [DOMINANT]

- `///` (Doxygen) = API / declaration doc, on every struct, field, method, constant. Trailing
  `///<` for one-liner field docs. `@param` / `@return` on multi-arg virtuals.
- `//` = the file-header block and in-body rationale / "why" comments. Never `/* */` blocks.
- **File header block shape:** `// <repo-relative path>` → blank `//` → one-line PURPOSE with
  `(architecture.md §N; ROADMAP §M)` refs → rationale → a **LAYERING note** stating the library
  and CUDA-visibility (e.g. `// THIS HEADER IS CUDA-FREE BY CONTRACT.` or
  `// PRIVATE to steppe_device (architecture.md §4 layering)`).
- **The `§` / ROADMAP / cleanup cross-refs are the project's traceability backbone — KEEP them.**
  They are not noise to strip; every non-trivial decision cites its spec section.

### 2.12 `const` / `[[nodiscard]]` / `noexcept` [DOMINANT]

- `[[nodiscard]]` on **every** value-returning query/factory/pure predicate (`get()`,
  `device_id()`, `capabilities()`, the fit virtuals). Add it to any non-void return whose
  result must be used.
- `noexcept` on accessors, moves, and teardown helpers. **Ctors that call CHECK macros may
  throw and are deliberately NOT `noexcept`** (handles.hpp:431 — "a ctor may throw; the dtor
  may not"). Destructors **never throw**; a nonzero teardown status routes to `STEPPE_LOG_WARN`.
- `const` locals are the norm (already overwhelmingly applied).
- RAII wrappers are move-only: deleted copy, `noexcept` move via `std::exchange`, private
  `destroy()/restore() noexcept`, moved-from owns nothing. This is the template for any RAII fix.

---

## 3. FP64 / PRECISION & CUDA-SPECIFIC NAMING

### 3.1 Pointer-locality prefixes — `d`/`d_` device, `h_` host [DOMINANT]

The review calls this "applied with rare consistency." Never leave a bare pointer where
host/device locality is ambiguous.

- **Device pointers:** `d`/`d_` prefix. Two in-convention sub-styles coexist by design and
  **must not be normalized into each other:**
  - tight CamelTail for kernel-arg pointers: `dQ`, `dV`, `dXmat`, `dQinv`, `dLoo`, `dScratch`
  - snake for index/host arrays: `d_left`, `d_block_sizes`, `d_status`, `d_se`
- **Host staging:** `h_` prefix — `h_left`, `h_se`, `h_weight`, `h_offsets_tile`.

**Do NOT** rewrite `dXmat` ↔ `d_xmat`. Preserve the local file's choice.

### 3.2 Protected parity vocabulary — RENAME FORBIDDEN (the #1 guardrail)

These names are **frozen for §12 oracle-diffability** against AT2 / cuSOLVER / cuBLAS. 30/61
units were clean precisely because the review excluded these. **Phase C must not "clean" them:**

- AT2 / linear-algebra symbols: `P` (pops), `M` (SNPs), `Q` `V` `N` `S` `R`, `A` `B` (rank-r
  factors), `Qinv` `Qf`, `r` `nl` `nr` `nb` `rmax`, `f2` `f4`, `Vpair`.
- SPMG device index `g` / `G`; loop counters `i j k p q b g`; de-collision doubles `kk ll kr kc ip`.
- **Name-after-AT2 trumps casing:** `seed_AB`, `opt_A`, `opt_B` keep the trailing capital
  on purpose; `tot_line_` matches AT2 `tot`. This is **not** casing drift.
- The **RHS/LHS inversion** (`solve(rhs,lhs)` mirroring R) and the **lowercase-field vs
  uppercase-param** split (`q`/`v`/`n` fields ↔ `Q`/`V`/`N` contract params, backend.hpp).
- FP64 / Ozaki literals and parity-frozen thresholds (e.g. `1e-30`, `kGesvdjMaxDim==32`):
  **name only, never change the value.**

### 3.3 CUDA-specific rules

- **Never change a type to win precision.** Do **not** rewrite `double` → `float` (or
  `int64_t` → `long`), and do not narrow accumulators. FP64-by-design is the law (§12).
- **`STEPPE_HD` for anything that must compile in both host and device** (`host_device.hpp`).
  Shared per-element primitives (`cdiv`, `grid_for`, the f2 estimator) use `STEPPE_HD`, never raw
  `__host__ __device__`. Raw `__device__` / `__global__` / `__restrict__` / `__launch_bounds__`
  are spelled directly, but **only inside `.cu` / `.cuh`** (they never cross to host TUs).
- **Name warp / grid constants, don't inline them:** `(m*m+31)/32*32` → `kWarpSize`;
  grid clamp `65535` → `kMaxGridDimX`; per-kernel block sizes `256/128/64` → `kBlock*`.
  These are occupancy tuning, explicitly **not** correctness-load-bearing.
- A duplicated single-letter-type device-restore guard (`struct G { int d; }`, now at **3** sites)
  should be replaced by a shared `DeviceGuard` RAII helper (resolves the findings at once).

---

## 4. DO / DON'T quick reference (the common Group-6 friction)

| Theme | DON'T (the drift) | DO (the fix) | Notes |
|---|---|---|---|
| **One concept, two spellings in one TU** (largest theme) | `bytes_per_rec` ↔ `bytes_per_record`; `vp` ↔ `vpair`; `n_ind` ↔ `n_individuals`; `e` ↔ `err` for `cudaError_t` | Spell it the way the **member/field it feeds** is spelled: `bytes_per_record`, `vpair`, `n_individuals`, `err` | One-side-only rename, zero behavior change |
| **One letter, two meanings in one TU** | `M` = SNP count AND widened `nl·nr`; `t` = Jacobi tangent AND swap-temp | Reserve `M` for the **SNP axis**; name the widened model-dim `mz` / `m_sz`; disambiguate `t` locally | Per Group 6.4.2 |
| **Cryptic derived local read far away** | `s` = `(nb-1)/sqrt(nb)` read 60+ lines later; `wc` next to verbose `als` | `jackknife_scale` / `se_scale`; `weight_chisq`. Rename the **matching kernel param too** | Coordinate both sides for parity traceability |
| **Legit single letter — LEAVE IT** | — | Tight-loop counters `i/j/k/p/b/g`; AT2 math symbols `P/M/Q/V/N/A/B/r/nl/nr` | **No action.** §3.2 |
| **Misleading role (noun vs verb/bool)** | `fit` is an int block-count; `part_slabs` holds an element count; `*_region_` holds a byte offset | `block_count`; `part_elems`; `*_offset` when it is an offset | Nouns for counts/extents, predicates for bools |
| **Unnamed literal** | bare `32`, `65535`, `1e-30`, status code `6` | Name it (`kWarpSize`, `kMaxGridDimX`, `Status::RankDeficient`). **Parity-frozen ⇒ name, keep value** | §2.5, §3.3 |
| **Positional booleans** | a 2-bool, 18-arg call where transposition silently mis-classifies (`assemble_result`) | Named flags aggregate / `enum class` | A **single** named+defaulted bool with a `/*name=*/` call annotation is fine |
| **Ad-hoc abbreviation** | two suffixes for one concept (`Pl` vs `Pp`); `free_vram` vs `free_host` | Pick one suffix per concept within the TU | One-side-only |

---

## 5. WHAT NOT TO TOUCH (non-goals)

Phase C is a LOW-cleanup, **not** a restyle. Explicitly out of bounds:

1. **Protected parity vocabulary (§3.2).** Do not rename `P/M/Q/V/N/A/B/r/nl/nr/Vpair`, the
   RHS/LHS inversion, the lowercase-field/uppercase-param split, `seed_AB`/`opt_A`/`opt_B`,
   FP64/Ozaki literals, or `1e-30`-class frozen thresholds. Lock-step only, if ever.
2. **The POD-vs-member underscore split (§2.4).** Do not add `_` to a POD field; do not strip
   `_` from a private class member.
3. **The two in-convention pointer styles (§3.1).** Do not normalize `dXmat` ↔ `d_xmat`.
4. **Documented carve-outs (§2.10).** Do not rename `CUBLAS_CHECK`/`CUSOLVER_CHECK` to
   `STEPPE_*`; do not "fix" `handles.hpp` being `.hpp`-with-CUDA.
5. **Correct existing names.** Do not re-case a domain acronym inside an already-correct
   PascalCase type (`F2Result` stays `F2Result`).
6. **Established public API.** No mass renames of `run_qpadm` / public struct fields / enumerators
   — that is an ABI/API break, out of scope for a LOW pass.
7. **Tight-loop counters `i/j/k/p/b/g` are FINE.** They are the loop-counter exception, not cryptic.
8. **Parity-frozen values.** Name a frozen literal, but **never change its magnitude** (§12).
9. **Roadmapped scaffold.** Do not delete seam-ahead-of-consumer symbols
   (`CusolverMathModeScope::promoted()`, `streamed_working_set_bytes`); mark `[[maybe_unused]]`
   or leave a note instead of removing.
10. **Behavior.** Every Group-6 rename is one-side-only and behavior-preserving. Touch the math
    in **no** rename.

---

## 6. How Phase C applies this

- **Group 6 (~51 LOW):** apply §4. Bring the straggler spelling **to** the field/member spelling
  (one concept = one spelling), resolve the `M`/`t` overloads (§4 row 2), and give cryptic
  derived locals descriptive names — coordinating any matching kernel param so both sides stay
  parity-traceable. Respect every §5 carve-out.
- **Group 5 (residual LOW):** name frozen sentinels/thresholds per §2.5 + §3.3 — `kWarpSize`,
  `kMaxGridDimX`, `Status::RankDeficient`, the `1e-30` Jacobi floor. **Name only; keep the value.**
- **Group 9:** hoist compile-time `const` → `inline constexpr` (§2.5); hoist a configurable
  buried literal (env-var key, cache path, file mode) to a file-top `constexpr`; resolve the one
  multi-bool call (§4 last rows). Many Group-9 lines are the same as Group-5 — resolve together.
- **Group 10 (4):** initialize at declaration; producers must establish out-param contracts
  unconditionally (size + zero / full-write on every path); value-type scalar members get an
  in-class default (the `int target = -1` pattern). One-line policy, no naming rule.
- **Group 3:** one IWYU sweep (preserve the 3-group include order + the trailing `// symbol`
  comments); delete dead stores; **keep** roadmapped scaffold. One-line policy, no naming rule.

**Bottom line:** the standard is ~80% Group 6. Its load-bearing half is three rules —
**protected-vocabulary allowlist (§3.2)**, **one concept = one spelling (§4)**, and
**one letter = one meaning (§4)** — plus the descriptive convention tables (§2). The Group 5/9
MED drift is already committed (config.hpp confirms `kGesvdjMaxDim` / `kFitBudget*` / `kFeeder*`
are live); Phase C's 5/9 work is naming/hoisting hygiene, not correctness.
