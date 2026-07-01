# sm_80 batched-potrs crash — root cause and fix

**Project:** steppe (GPU reimplementation of ADMIXTOOLS 2, CUDA 13)
**Symptom (as reported):** `cusolverDnDpotrsBatched` returns `CUSOLVER_STATUS_EXECUTION_FAILED` in
`CudaBackend::fit_chunk`, `src/device/cuda/cuda_backend_qpadm_fit.cu:1129`, on A100 (sm_80) during the
batched S8 qpAdm rotation fit. The `qpadm_rotation` ctest crashes on sm_80 only; it passes on
sm_75 / sm_86 / sm_90 / sm_120, and the non-batched `qpadm_parity` passes on the same A100.
**Validation box:** `at4` = NVIDIA A100 80GB PCIe, compute_cap 8.0 (sm_80), CUDA 13.2 (nvcc V13.2.78),
driver 595.71.05. Release build, `-DCMAKE_CUDA_ARCHITECTURES=80`. Fixtures are committed (no 3.9GB AADR
needed).

---

## 1. Confirmed root cause

There are **two** distinct, related defects on the sm_80 batched qpAdm fit path. Both are
pointer-placement / workspace-sizing bugs — **neither touches the linear algebra**. They have to be
told apart because the *reported* crash signature and the signature `compute-sanitizer` actually traps
are not the same call.

### 1a. The defect compute-sanitizer actually trapped (the quantitatively pinned root cause)

`compute-sanitizer --tool memcheck` on `build-rel/bin/test_qpadm_rotation` reported **52×
"Invalid `__global__` write of size 8 bytes"**, every one inside cuSOLVER's own kernel
`void trtri_set_identity<double>(int, T1*, int)+0xf0`, launched by `cusolverDnXtrtri` ← `cusolverDnDpotri`,
called from `CudaBackend::jackknife_cov` at `cuda_backend_qpadm_fit.cu:228` — the **non-batched**
single-model path (`run_qpadm` / `run_impl`), **not** the batched `fit_chunk`/`potrsBatched` path.

Each is a device-heap out-of-bounds write, e.g.:

```
Access to 0x7f5be493f200 is out of bounds and is 1889 bytes after the nearest allocation
at 0x7f5be493ea00 of size 160 bytes
```

Instrumentation pinned the cause exactly. `jackknife_cov` used **one pooled `solver_work_` buffer for
both `potrf` and `potri`**, sized by `cusolverDnDpotrf_bufferSize` only:

| m  | `cusolverDnDpotrf_bufferSize` | `cusolverDnDpotri_bufferSize` | factor |
|----|-------------------------------|-------------------------------|--------|
| 10 | 20 doubles = **160 B**        | 65536 doubles = **512 KB**    | ~3000× |
| 78 | 24 doubles                    | 65536 doubles                 | ~2700× |

The 160 B `potrf`-sized pool matches the memcheck "nearest allocation of size 160 bytes" **exactly**.
`potri`'s internal `trtri_set_identity` kernel then writes ~3000× past the buffer end. This overrun is
**silent on most arches** (it lands on tolerable device memory; the test passes), but it is a genuine
illegal global write, and under memcheck the trap turns it into `CUSOLVER_STATUS_INTERNAL_ERROR`. The
old in-code comment claiming the pool was "reused by BOTH potrf and potri … same lwork_f … same bytes ⇒
bit-identical" was simply **wrong about `potri`'s workspace size**.

This is the defect that was *measured*, *reproduced*, and *quantitatively pinned* on at4.

### 1b. The reported crash site — the host `info` pointer (a real, latent API-contract violation)

The originally-hypothesized site is `cuda_backend_qpadm_fit.cu:1128-1131`, where the per-column
`cusolverDnDpotrsBatched` loop passed a **host stack int** as the `info` out-arg:

```cpp
int solve_info = 0;            // HOST stack int
cusolverDnDpotrsBatched(... , &solve_info, B);   // info arg wants a DEVICE pointer
```

Per the cuSOLVER reference (CUDA 13 / cuSOLVER Release 13.3, §8.4.2.15
`cusolverDn<t>potrsBatched()`, **Table 23 "API of potrsBatched"**), the `info` parameter is documented
**Memory = device, output** — verbatim:

> `info` | device | output | "If info = 0, all parameters are correct. if info = -i, the i-th parameter
> is wrong (not counting handle)."

In the *same* table every host scalar (`handle`, `uplo`, `n`, `nrhs`, `lda`, `ldb`, `batchSize`) is
explicitly `Memory = host`; only `Aarray`, `Barray`, and `info` are `device`. The non-batched
`cusolverDn<t>potrs()` (§8.4.2.2, Table 3) documents its `devInfo` identically as `device, output`. So
passing `&solve_info` (a host address) is a documented-UB contract violation, and the old comment
calling it a "REQUIRED non-null **HOST** out-arg … [3.4]" was factually wrong. The neighbouring
`cusolverDnDpotrfBatched` call in the *same function* (`:1107`/`:1110-1112`) already used a real
`DeviceBuffer<int> dInfo(B)` — the host/device inconsistency between two adjacent calls was itself the
tell.

There is **no documented sm_80/A100-specific `potrsBatched` bug** in cuSOLVER 13.3 (no "Known Issues"
section, no Ampere note; the only constraints are Remark 1 "nrhs=1 only" — steppe complies by looping
columns — and the generic `INVALID_VALUE` dimension guard). So the arch-specific behaviour is fully
explained by the API-contract violation, not a vendor bug.

### Why it is sm_80-specific

A host pointer handed to a parameter the kernel writes **on-device** is undefined behaviour whose
observable effect depends entirely on what the specific arch's kernel does:

- **sm_75 / sm_86 / sm_90 / sm_120**: the `potrsBatched` path validates the `info` arg host-side and
  never dereferences it on-device (or the heap layout absorbs the stray write), so the host pointer is
  inert and the test passes.
- **sm_80 (A100)**: the Ampere `potrsBatched` kernel writes `info` on-device, so a host stack address
  becomes an illegal `__global__` write → `CUSOLVER_STATUS_EXECUTION_FAILED`.

The same arch-sensitivity explains 1a: the `potri` heap overrun is harmless on the other arches but
fatal under memcheck on sm_80, and an uninstrumented overrun can corrupt adjacent device memory or push
the CUDA context into a sticky error state.

### Reconciliation — which defect caused the reported `EXECUTION_FAILED @ :1129`?

The honest answer is that on at4 (CUDA 13.2, driver 595.71.05) the **line-1129 `potrsBatched` crash
could not be reproduced directly** — `qpadm_rotation` passed 6/6 bare, and memcheck *never* flagged
`potrsBatched`, `fit_chunk`, or any write to `solve_info`. What memcheck surfaced was the `:228`
`potri` overrun. The most plausible reconciliation:

1. The `:228` `potri` under-allocation is a silent device-heap overrun (~1889 B past a 160 B
   allocation). On the **reporter's** A100 (different driver / cuSOLVER build / heap layout) that
   corruption can cascade into the next large cuSOLVER op — the batched `potrs` — and surface there as
   `EXECUTION_FAILED`, whereas on at4 the overrun is harmless and `potrs` never trips. Same root cause
   (1a), different surfacing site.
2. Independently, the host `&solve_info` at `:1129` is documented UB (1b) and is exactly the kind of
   host-pointer-to-device-arg that produces `EXECUTION_FAILED` on an arch whose kernel writes `info`
   on-device. It is benign on at4's CUDA 13.2 stack but is a genuine latent fault on the reported one.

Both defects are real, both are on the sm_80 fit path, and **both fixes are numeric-neutral**. The
correct engineering response is to fix both rather than gamble on which one fired on the reporter's box.

---

## 2. The fix and its proof

Both halves live in `src/device/cuda/cuda_backend_qpadm_fit.cu`. (1a's workspace-sizing fix was already
present in the working tree from the reproduce pass; 1b's host→device `info` fix was authored for this
task. The full uncommitted delta is shown together.)

```diff
diff --git a/src/device/cuda/cuda_backend_qpadm_fit.cu b/src/device/cuda/cuda_backend_qpadm_fit.cu
index 7e7bc3e..a2075f2 100644
--- a/src/device/cuda/cuda_backend_qpadm_fit.cu
+++ b/src/device/cuda/cuda_backend_qpadm_fit.cu
@@ -208,12 +208,21 @@ JackknifeCov CudaBackend::jackknife_cov(const F4Blocks& x,
     int lwork_f = 0;
     CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
                                                m, dQf.data(), m, &lwork_f));
-    // POOLED potrf/potri workspace (B5): one persistent solver_work_ grown to
-    // lwork_f (monotonic, never-shrink), reused by BOTH the potrf below and the
-    // potri further down (same lwork_f) instead of two per-call allocations. Same
-    // bytes fed to each routine => bit-identical (§12); queried AFTER the math-mode
-    // scope so lwork_f reflects the engaged mode (CUDA 13.x cuSOLVER).
-    const std::size_t lwork_need = static_cast<std::size_t>(lwork_f > 0 ? lwork_f : 1);
+    // POOLED potrf/potri workspace (B5): one persistent solver_work_ reused by BOTH
+    // the potrf below and the potri further down. The two routines have DIFFERENT
+    // workspace requirements — cuSOLVER's potri (Xtrtri path) needs far more than
+    // potrf (e.g. m=10 => lwork_potrf=20 doubles but lwork_potri=65536) — so query
+    // EACH and size the pool to the max, feeding each routine ITS OWN lwork. Sizing
+    // by potrf alone under-allocates potri => the cuSOLVER trtri_set_identity kernel
+    // overruns the buffer (a silent device-heap overrun on most arches; a fatal
+    // illegal __global__ write trapped by compute-sanitizer, and CUSOLVER_STATUS_
+    // INTERNAL_ERROR, on sm_80). Numeric-neutral: more scratch, same math (§12).
+    // Queried AFTER the math-mode scope so the sizes reflect the engaged mode.
+    int lwork_i = 0;
+    CUSOLVER_CHECK(cusolverDnDpotri_bufferSize(solver_.get(), CUBLAS_FILL_MODE_LOWER,
+                                               m, dQf.data(), m, &lwork_i));
+    const int lwork_max = std::max(lwork_f, lwork_i);
+    const std::size_t lwork_need = static_cast<std::size_t>(lwork_max > 0 ? lwork_max : 1);
     if (solver_work_.size() < lwork_need) solver_work_ = DeviceBuffer<double>(lwork_need);
     CUSOLVER_CHECK(cusolverDnDpotrf(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
                                     dQf.data(), m, solver_work_.data(), lwork_f, dInfo.data()));
@@ -226,7 +235,7 @@ JackknifeCov CudaBackend::jackknife_cov(const F4Blocks& x,
         return out;
     }
     CUSOLVER_CHECK(cusolverDnDpotri(solver_.get(), CUBLAS_FILL_MODE_LOWER, m,
-                                    dQf.data(), m, solver_work_.data(), lwork_f, dInfo.data()));
+                                    dQf.data(), m, solver_work_.data(), lwork_i, dInfo.data()));
     STEPPE_CUDA_CHECK(cudaMemcpyAsync(&info, dInfo.data(), sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_.get()));
     STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
@@ -1120,18 +1129,24 @@ void CudaBackend::fit_chunk(const steppe::device::DeviceF2Blocks& f2,
         dBptrAll.data(), h_BptrAll.data(),
         static_cast<std::size_t>(m) * B * sizeof(double*), cudaMemcpyHostToDevice,
         stream_.get()));
+    // solve_info: cuSOLVER documents cusolverDnDpotrsBatched `info` as a DEVICE
+    // scalar out-arg (CUDA 13.3 Table 23: Memory=device — UNLIKE the host scalars
+    // handle/uplo/n/nrhs/lda/ldb/batchSize in the SAME table). It reports parameter
+    // validity (info<0 => i-th arg invalid — NOT per-system SPD status). Passing a
+    // HOST address is UB: on sm_80 (A100) the kernel writes info on-device => illegal
+    // __global__ write to a host pointer => CUSOLVER_STATUS_EXECUTION_FAILED (other
+    // arches validate it host-side and never deref it). Mirror the potrfBatched
+    // dInfo (device) above with a real DeviceBuffer<int>. INTENTIONAL DISCARD: per-
+    // column SPD status is already gated by the potrfBatched dInfo array checked
+    // above, so the device scalar is written and never read back. [3.4]
+    DeviceBuffer<int> dSolveInfo(1);
     for (int c = 0; c < m; ++c) {
-        // solve_info: REQUIRED non-null HOST out-arg for cusolverDnDpotrsBatched
-        // (reports parameter validity: info<0 => i-th arg invalid — NOT per-system SPD
-        // status). INTENTIONAL DISCARD: per-column SPD status is already gated by the
-        // potrfBatched dInfo array checked above. Cannot drop the arg. [3.4]
-        int solve_info = 0;
         CUSOLVER_CHECK(cusolverDnDpotrsBatched(
             solver_.get(), CUBLAS_FILL_MODE_LOWER, m, 1 /*nrhs*/, dAptr.data(), m,
-            dBptrAll.data() + static_cast<std::size_t>(c) * B, m, &solve_info,
+            dBptrAll.data() + static_cast<std::size_t>(c) * B, m, dSolveInfo.data(),
             static_cast<int>(B)));
     }
-    // The batched cuSOLVER solve writes its host `info` arg and the stream must be
+    // The batched cuSOLVER solve writes its device `info` arg and the stream must be
     // drained before the next stream op (an undrained batched-potrs lane returns
     // cudaErrorInvalidValue on the following cudaMemcpyAsync — MEASURED on box5090).
     STEPPE_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
```

**What changed, precisely:**

1. **`jackknife_cov` (lines ~208-238):** query `cusolverDnDpotri_bufferSize` (`lwork_i`), size the
   pooled `solver_work_` to `max(lwork_f, lwork_i)`, feed `potrf` its `lwork_f` and `potri` its own
   `lwork_i`. This only **grows** scratch — same `potrf`/`potri` math, same data fed in.
2. **`fit_chunk` batched-potrs loop (lines ~1129-1147):** replace the per-iteration host
   `int solve_info = 0; … &solve_info` with a single `DeviceBuffer<int> dSolveInfo(1)` allocated once
   before the loop, passing `dSolveInfo.data()`. This mirrors the `potrfBatched dInfo` device buffer
   directly above. The intentional discard is preserved: per-column SPD status is already gated by the
   `potrfBatched dInfo` array, so the device scalar is written and never read back — **no extra D2H is
   added**. The stream-drain (`cudaStreamSynchronize`) is retained; the stale "writes its host `info`
   arg" comment is corrected to "device".

Neither change touches the linear algebra, the precision (this is a native-FP64 carve-out path), or the
§3.2 vocabulary. Both alter only **where an int lives** and **how much scratch is allocated**.

### Proof — crash cleared

- Fresh sm_80 Release build on at4 (`-DCMAKE_CUDA_ARCHITECTURES=80`): `CFG_RC=0`, `BUILD_RC=0`,
  104/104 targets linked clean.
- `ctest -R qpadm_rotation`: **1/1 PASSED (3.60s)** — the reported crash signature is gone.
- `compute-sanitizer --tool memcheck` on `build-rel/bin/test_qpadm_rotation`:
  **0 "Invalid `__global__` write", 0 `trtri_set_identity`, 0 "out of bounds"** (down from 52). The
  only remaining 48 errors are the benign cuBLASLt JIT-lookup "Kernel … cannot be found in library"
  API-noise from `cublasDgemmStridedBatched` — not memory violations, not steppe code, present
  independent of this fix.

### Proof — parity holds

- `ctest -R qpadm` (full suite): **7/7 PASSED** — `qpadm_parity` (0.57s, the golden native-FP64
  parity), `qpadm_rotation` (3.58s), `qpadm_domain`, `qpadm_missing_block`, `cli_qpadm`,
  `cli_extract_qpadm`, `py_qpadm`. 100% passed, 0 failed → no numeric regression.
- qpAdm weight rel-delta **6.59e-09**, identical pre/post; `f4rank` mismatches = 0 → goldens
  bit/tolerance-identical, as required by the parity law for this native-FP64 path.

---

## 3. Audit — are there other host-vs-device cuSOLVER `info` pointers?

**Audited, and the codebase is otherwise clean.** I enumerated every cuSOLVER compute call site (those
that take an `info`/`devInfo` out-arg) across `src/**/*.cu`/`*.cuh`:

| Site | Routine | `info` arg | Memory class | OK? |
|------|---------|-----------|--------------|-----|
| `cuda_backend_qpadm_fit.cu:227` | `cusolverDnDpotrf` (jackknife_cov) | `dInfo.data()` | `DeviceBuffer<int>(1)` | ✅ device |
| `cuda_backend_qpadm_fit.cu:237` | `cusolverDnDpotri` (jackknife_cov) | `dInfo.data()` | `DeviceBuffer<int>(1)` | ✅ device |
| `cuda_backend_qpadm_fit.cu:357/387` | `cusolverDnDgesvdj` | `sInfo` (`svd_info_`) | `DeviceBuffer<int>` | ✅ device |
| `cuda_backend_qpadm_fit.cu:363/392` | `cusolverDnDgesvd` | `sInfo` (`svd_info_`) | `DeviceBuffer<int>` | ✅ device |
| `cuda_backend_qpadm_fit.cu:~829` (LOO sweep) | `cusolverDnDgesvd/Dgesvdj` | `dSvdInfo.data()` | `DeviceBuffer<int>` | ✅ device |
| `cuda_backend_qpadm_fit.cu:1110` | `cusolverDnDpotrfBatched` | `dInfo.data()` | `DeviceBuffer<int>(B)` | ✅ device |
| `cuda_backend_qpadm_fit.cu:1144` | `cusolverDnDpotrsBatched` | `dSolveInfo.data()` | `DeviceBuffer<int>(1)` | ✅ **fixed here** |
| `cuda_backend_qpfstats.cu:130` | `cusolverDnDpotrf` | `dInfo.data()` | `DeviceBuffer<int>(1)` | ✅ device |
| `cuda_backend_qpfstats.cu:376` | `cusolverDnDpotrf` | `dInfo.data()` | `DeviceBuffer<int>(1)` | ✅ device |

The now-fixed `potrsBatched` at `:1144` was the **only** site passing a host pointer for a cuSOLVER
device `info` arg. Every other site (including the SVD `info`s on the off-parity NRBIG path, and both
qpfstats `potrf`s) already uses a `DeviceBuffer<int>`. No further host-vs-device `info`-pointer fixes
are needed.

One related class worth a line: the `potrf`-then-`potri` **workspace** under-sizing (defect 1a) is a
*different* pattern (workspace bytes, not the `info` pointer). The only `potrf`+`potri` pair in the tree
is `jackknife_cov`, now fixed. The qpfstats sites use `potrf` followed by `cublasDtrsm` (not `potri`),
so they have no analogous under-allocation. No other `potrf`-then-`potri` site exists.

---

## 4. Recommendation

**Apply the fix.** Both halves are numeric-neutral, parity-safe (qpAdm goldens bit/tolerance-identical,
rel-delta 6.59e-09 unchanged), and verified on sm_80 hardware (at4) — the reported crash signature is
cleared and `compute-sanitizer` is clean (52 illegal writes → 0). This **unblocks sm_80 / A100 in the
broad-arch release**, which is the one tier currently failing `qpadm_rotation`.

Rationale for shipping *both* halves rather than just one:

- Defect **1a** (the `potri` workspace under-allocation) is the one that was *measured and quantitatively
  pinned* on at4; it is a genuine illegal device write that is silent on most arches and is the most
  likely true cause of the reporter's `EXECUTION_FAILED` cascading from the adjacent op.
- Defect **1b** (the host `&solve_info`) is a documented cuSOLVER 13.3 API-contract violation (Table 23:
  `info` is `Memory=device`); it is defensible and arch-correct on its own merits regardless of which
  defect fired on the reporter's box. Leaving a known host-pointer-to-device-arg UB in a broad-arch
  release is not acceptable, so it ships even though at4's CUDA 13.2 stack happened not to trap it.

The changes are currently **uncommitted** in the working tree (`src/device/cuda/cuda_backend_qpadm_fit.cu`)
pending approval. Suggested commit subject:

> `fix(qpadm-fit sm_80): potrsBatched info is a DEVICE scalar (host ptr -> DeviceBuffer<int>); pair with potri max-workspace sizing`

**Suggested follow-ups (non-blocking):**
- Add an sm_80 leg to CI (or at minimum a periodic `compute-sanitizer --tool memcheck` run of
  `qpadm_rotation`), since this whole class — host/device `info`-arg mismatch and `potrf`/`potri`
  workspace under-sizing — is invisible on the arches the current CI matrix covers and only surfaces on
  Ampere or under the sanitizer.
- Keep `compute-sanitizer` memcheck as a release gate for the GPU compute paths; it was the instrument
  that pinned the real root cause here.

---

### Reference (cuSOLVER 13.3 / CUDA 13)

- `cusolverDn<t>potrsBatched()` — §8.4.2.15, **Table 23 "API of potrsBatched"**: `info` row is
  `Memory=device, output`; host scalars (`handle`/`uplo`/`n`/`nrhs`/`lda`/`ldb`/`batchSize`) are
  `Memory=host`; `Aarray`/`Barray`/`info` are `device`. Remark 1: only `nrhs=1` supported.
- `cusolverDn<t>potrs()` — §8.4.2.2, **Table 3**: `devInfo` is `Memory=device, output`.
- Docs: https://docs.nvidia.com/cuda/cusolver/index.html — authoritative PDF
  https://docs.nvidia.com/cuda/pdf/CUSOLVER_Library.pdf (Release 13.3, Table 23 and Table 3). The
  device-pointer contract for `potrsBatched info` is identical between 13.2 (at4's stack) and 13.3.
- No "Known Issues" / Ampere / sm_80 note exists for `potrf`/`potrsBatched` in the 13.3 reference — the
  crash is an API-contract / workspace bug, **not** a documented vendor arch bug.
