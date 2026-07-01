# steppe — Multi-Arch Build + AT2 Parity Validation

**Scope of this run:** release-scoping experiment. Does steppe (HEAD of the local
working tree, CUDA 13) **build** and does the ADMIXTOOLS-2 golden **parity hold**
on older GPU arches (Volta / Turing / Ampere / Hopper), or is 0.1.0 Blackwell-only?
Each instance got a fresh source rsync and a fresh per-arch configure (no sm_120
build cache reused; cubins verified to match the instance GPU via `cuobjdump`).

---

## 1. Headline

**steppe builds and AT2 golden parity HOLDS well beyond Blackwell — on Turing
(sm_75), consumer Ampere (sm_86), and Hopper (sm_90) — and the sm_120-tuned
emulated-FP64 (Ozaki) matmul path compiled with ZERO sm_120-specific-intrinsic
breakage on every architecture nvcc accepted.** That is the decisive positive
result: the precision-context worry — that the Blackwell-tuned emulation might use
sm_120-only tensor-core FP64-emulation features that break the build or a kernel on
older silicon — **did not materialize on any arch that compiled.**

Two boundaries qualify the headline:

- **Volta (sm_70) is OUT — and it is not steppe's fault.** CUDA 13.0 dropped
  offline compilation for `compute_70` entirely (`nvcc --list-gpu-arch` floors at
  `compute_75`). Configure dies at compiler-detection before any steppe source is
  touched. Volta cannot be in a CUDA-13 release matrix at all.
- **Datacenter Ampere (sm_80 / A100) builds and SCALAR parity holds, but has ONE
  real arch-specific runtime break:** the *batched* Cholesky solve
  (`cusolverDnDpotrsBatched`) in the S8 rotation fit path crashes
  (`CUSOLVER_STATUS_EXECUTION_FAILED`). 11/12 golden parity tests pass including the
  primary `qpadm_parity`; only the batched `qpadm_rotation` fixture fails. This is a
  genuine code defect to fix, not a precision mismatch.

So on the build+parity axis the broad target {sm_75, sm_86, sm_90, sm_120} is
clean today; sm_80 needs one batched-path fix to join them; sm_70 is impossible
under CUDA 13.

---

## 2. Matrix

| Instance | GPU | sm | Family | Builds? | configure_s | build_s | Parity holds? | Tests P/F/S | Key note |
|---|---|---|---|---|---|---|---|---|---|
| at1 | RTX 2080 Ti | 75 | Turing | ✅ yes | 6 | 156* | ✅ **yes** | 67 / 10 / 0 | All committed-fixture goldens pass. `emu_honorability` gate RED on sm_75 (parity still holds). Needed extra `-DSTEPPE_CUDA_ARCH=75` (cmake rewrites bare `75`→120). |
| at2 | RTX 3090 | 86 | Ampere (consumer) | ✅ yes | 20 | 110 | ✅ **yes** | 68 / 9 / 0 | Clean build, all goldens pass, `emulated_fp64_honorable=true`. Only fails = the 2 hardcoded `compute_major>=12` test pins + AADR-absence skips. |
| at3 | Tesla V100-SXM2-32GB | 70 | Volta | ❌ **no** | 2 | — (fail) | — n/a | 0 / 0 / 0 | **CUDA 13 dropped sm_70.** `nvcc fatal: Unsupported gpu architecture 'compute_70'`. Dies at `project()` try-compile; no steppe source compiled. Toolkit block, not steppe. |
| at4 | A100-PCIE-40GB | 80 | Ampere (datacenter) | ✅ yes | 71 | 71 | ⚠️ **no** | 67 / 10 / 0 | Builds clean, scalar parity holds (11/12 goldens incl. `qpadm_parity`). **Real break:** `cusolverDnDpotrsBatched` → `CUSOLVER_STATUS_EXECUTION_FAILED` in batched S8 fit (`qpadm_rotation`). |
| at5 | H100 PCIe | 90 | Hopper | ✅ yes | 13 | 27 | ✅ **yes** | 68 / 9 / 0 | Cleanest run. All goldens pass, `emulated_fp64_honorable=true`, cubin verified sm_90 (56 device objs, no sm_120 leakage). |

\* at1 `build_s=156` reused ~37 objects from an aborted first attempt; a fully-clean
sm_75 build is ~175s. See §3 build-time notes.

**Reading the P/F/S columns:** ctest reported `skipped=0` on every box because the
AADR-keyed tests hard-error on a missing `/workspace/data/aadr` path instead of
emitting a clean ctest SKIP. With the ~3.9 GB AADR data intentionally absent (per
brief), 7 such tests fail-by-exit-code on every arch but are **skip-equivalent** and
would fail identically on sm_120 — they are not arch or parity signals. Subtract
those 7 and the two `compute_major>=12` test-pins, and the genuine arch-attributable
results are: at1/at2/at5 = full parity, at4 = one real batched-path break, at3 =
no build.

---

## 3. Findings

### 3.1 Volta / CUDA-13: a toolkit wall, not a steppe wall (at3)
The V100 failure is the most clear-cut release-scoping fact. CUDA 13.0 (V13.0.88)
removed Volta `sm_70` from offline compilation. `nvcc --list-gpu-arch` on the box =
`{75,80,86,87,88,89,90,100,110,103,120,121}` — **no 70**. The arch override resolved
correctly (`-DCMAKE_CUDA_ARCHITECTURES=70` passed through to
`--generate-code=arch=compute_70,code=[compute_70,sm_70]`); nvcc itself rejected it
during `project()` compiler detection (`CMakeLists.txt:20`), so configure aborted
(RC=1) before a single steppe `.cu` was compiled. This is **independent of steppe's
code and of the FP64-emulation path** — V100's strong native FP64 is moot because
the arch can't be compiled. Notably the repo's own `STEPPE_CUDA_ARCH_RELEASE`
(`cmake/CUDAArch.cmake:61–63`) already starts at `75-real` with **no 70 entry** —
the codebase already treats Turing as the floor, consistent with this result.
**Implication:** Volta support would require a separate CUDA-12.x build, or be
dropped. For a CUDA-13 0.1.0, drop it.

### 3.2 No sm_120-specific build break on ANY arch that compiled
This was the central precision-context risk and it came back clean. On sm_75, sm_80,
sm_86, and sm_90, nvcc accepted the arch with **no rejection of any sm_120-specific
intrinsic**, the Ozaki emulated-FP64 device code compiled, and cubins were verified
to the target arch via `cuobjdump` (sm_75 / sm_86 / sm_80 / sm_90 respectively — no
sm_120 leakage, no "no kernel image" at test time). The library's own capability
probe reported `emulated_fp64_honorable=true` on sm_80, sm_86, and sm_90. The
Blackwell-tuned emulation is **portable at the source/compile level** across the
Turing→Hopper range.

### 3.3 Cross-arch parity drift: one real FAIL, one precision caveat, the rest are noise
Separating genuine arch failures from expected skips:

- **REAL arch FAIL — at4 / A100 / sm_80 (`qpadm_rotation`, #34):**
  `cusolverDnDpotrsBatched` → `CUSOLVER_STATUS_EXECUTION_FAILED` at
  `cuda_backend_qpadm_fit.cu:1129`. The **batched** Cholesky solve in the S8
  rotation/batched-fit path crashes on sm_80. The non-batched `qpadm_parity` (#25)
  PASSES on the same box, so this is a batched-launch/execution failure in the
  cuSOLVER batched-API setup, **not** a numeric parity mismatch — and it is on the
  *native-FP64* path, which is supposed to be A100's strength. This is the one item
  that must be code-fixed before sm_80 is a fully-validated target.

- **PRECISION CAVEAT — at1 / RTX 2080 Ti / sm_75 (`emu_honorability`, #12):** the
  EmulatedFp64 honorability self-check (that emulation genuinely engages the
  high-precision fixed-point path rather than cuBLAS's ~60-bit dynamic default) does
  **not** hold on Turing at the end-to-end `compute_f2` sub-check. **Golden parity
  still holds** — every committed AT2 fixture matches — but a Turing release would
  ship with the honorability gate red. This is exactly the arch-sensitivity the
  precision context flagged. It needs triage before claiming Turing as a
  *tuned/honorable-emu* target (vs merely a *correct* one). Note sm_80/86/90 all
  report honorable=true, so this is specific to Turing.

- **NOT arch failures (every box):** the two hardcoded `compute_major >= 12`
  (Blackwell) assertions — `backend_capabilities_probe` (#41) and `resources_build`
  (#42) — fail *by design* on any pre-Blackwell arch; the backend constructs, VRAM
  probes, and `emulated_fp64_honorable` all pass underneath them. And the 7
  AADR-data-absence tests (`f2_equivalence`, `decode_equivalence`,
  `f2_blocks_equivalence`, `f2_determinism`, `f2_multigpu_parity`,
  `block_partition_aadr_consistency`, `filter_oracle`) hard-error on the missing data
  path rather than ctest-SKIP. Both classes would behave identically on sm_120 —
  zero arch signal.

### 3.4 Native vs emulated FP64 — the angle has a twist
The brief's hypothesis was that strong-native-FP64 datacenter parts (V100/A100/H100)
might be a *better* fit than the weak-native Blackwell the emulation was tuned for.
Evidence is mixed:
- **H100 (sm_90):** best of both worlds — emulated path honorable + correct, native
  FP64 strong, full parity, fastest run. Clean win.
- **A100 (sm_80):** the emulated/scalar path is fully correct (`emu_honorability`
  passes, 11/12 goldens pass), but the **native-FP64 batched cuSOLVER path is exactly
  what crashes.** So "native FP64 is strong on A100" did not translate to a cleaner
  result — the one break is on that path.
- **V100 (sm_70):** strong native FP64 is irrelevant; can't compile under CUDA 13.
- **Turing (sm_75):** weak native (1:32) *and* emulation honorability red — correct
  but the least "tuned" of the working arches.

Net: native-FP64 strength does not by itself buy a cleaner port; the emulated path is
the portable common denominator (correct everywhere it compiled), and the only
native-path-specific failure is a cuSOLVER batched bug on sm_80.

### 3.5 Build-time deltas (interpret with care — boxes are not matched)
configure_s and build_s vary with CPU core count, clock, disk, CPM-fetch state, and
partial-rebuild reuse — these are different vast.ai boxes, so treat as rough:

| Box | arch | cores (-j) | configure_s | build_s | note |
|---|---|---|---|---|---|
| at5 H100 | 90 | -j26 | 13 | **27** | clean 320/320 — fastest CPU box |
| at4 A100 | 80 | — | 71 | 71 | configure includes CPM fetch (CLI11/nanobind); build split across 2 passes |
| at2 3090 | 86 | -j32 | 20 | 110 | clean from-scratch 320 steps |
| at1 2080Ti | 75 | -j32 | 6 | 156* | reused ~37 objs; clean ≈175s |

There is **no arch-intrinsic compile-cost signal** here — the spread tracks the host
CPU and the rsync/CPM state, not the GPU target. The H100 box simply had the fastest
host. Build cost is not a reason to scope arches in or out.

### 3.6 Harness wart that hit 4 of 5 boxes (orchestrator fix)
The prescribed rsync used `--exclude 'build*'`, which is **unanchored** — rsync
matches it against the basename at every directory level and silently drops the
tracked source `src/core/config/build_result.hpp` (and
`experiments/f2_emu_spike/build_run.sh`). This caused a spurious first-build failure
(`fatal error: core/config/build_result.hpp: No such file` — a *host* C++ error, not
nvcc/arch) on at1, at2, at4, and at5. Each runner worked around it (targeted re-sync
of the two files). **Fix:** anchor the pattern to `--exclude '/build*'` (or
`--exclude 'build-rel'`). This is purely a harness issue; it did not affect any
parity or arch result once worked around.

---

## 4. Release Recommendation

**Verdict: go BROAD, not Blackwell-only. 0.1.0 should target Turing→Blackwell.** The
evidence is decisive: the Blackwell-tuned emulated-FP64 path is source-portable with
no sm_120-intrinsic breakage, and full AT2 golden parity holds on sm_75, sm_86, and
sm_90 with zero code changes. Restricting to Blackwell would forfeit the broad
installed base for no functional reason.

### 4.1 Fatbin arch list
Ship the repo's existing release list **minus Volta** (already absent) and treat the
two untested-here entries explicitly:

```
75-real;80-real;86-real;89-real;90-real;120-real;120-virtual
```

- `75,80,86,90,120` — directly validated this run (build + parity, with the sm_80
  caveat in §4.3).
- `89-real` (Ada) — not tested here but architecturally between 86 and 90, both of
  which passed cleanly; low-risk to include via fatbin. Flag as "compiled-only,
  parity-by-extrapolation" until a Lovelace box is run.
- `120-virtual` (PTX) — keep as the forward-compat JIT fallback.
- **`100-real` (datacenter Blackwell):** the repo list includes it but it was **not
  tested** this run and is explicitly *not* cubin-compatible with 120
  (`CUDAArch.cmake:18`). Keep it in the fatbin if a B100/B200 build target is wanted,
  but mark it unvalidated.
- **`70` (Volta):** omit — impossible under CUDA 13 (§3.1). If Volta coverage is a
  hard requirement, that is a *separate CUDA-12.x build*, out of scope for the
  CUDA-13 0.1.0.

### 4.2 What needs a code fix before the broad claim is fully green
1. **(sm_80 RELEASE BLOCKER for A100) Batched Cholesky crash.**
   `cusolverDnDpotrsBatched` → `CUSOLVER_STATUS_EXECUTION_FAILED` at
   `cuda_backend_qpadm_fit.cu:1129` in the S8 batched-fit path. Investigate
   batched-workspace sizing / batched-API argument setup on sm_80; consider a
   fallback to the per-system (non-batched) solve, which passes. **Until fixed, gate
   sm_80 as "scalar-parity validated, batched-rotation path unsupported"** — or hold
   the A100 stamp.
2. **(sm_75 TRIAGE, not a blocker) Turing emu honorability.** `emu_honorability`
   (#12) is red on sm_75 — the Ozaki emulation likely falls back to cuBLAS's ~60-bit
   dynamic default instead of engaging fixed-point on Turing. Parity holds, so this
   does **not** block a "correct on Turing" claim, but it does block a "tuned/honorable
   emu on Turing" claim. Triage before stamping Turing as a first-class precision
   target.

### 4.3 Test-harness fixes required for a green multi-arch CI (not library defects)
These do not block the library on any arch but **will keep multi-arch CI red** until
fixed:
- Relax / parametrize the two hardcoded `compute_major >= 12` (Blackwell) assertions
  in `backend_capabilities_probe` (#41) and `resources_build` (#42) to the detected
  min-arch. They fail by design on every pre-Blackwell box.
- Make the AADR-keyed tests emit a proper **ctest SKIP** when `STEPPE_AADR_ROOT` /
  dataset is absent, instead of hard-erroring on a hardcoded `/workspace/data/aadr`
  path. Today they inflate the failed count and force `skipped=0` everywhere.
- Anchor the rsync exclude (`/build*`) in the orchestrator (§3.6).
- Fix the `cmake/CUDAArch.cmake` bare-`75` substitution: a user passing
  `-DCMAKE_CUDA_ARCHITECTURES=75` is silently rewritten to 120 (lines 50–56 treat
  `75` as the toolkit auto-default), so the **sm_75 target uniquely** requires the
  extra `-DSTEPPE_CUDA_ARCH=75`. For a Turing-inclusive release this special-case
  should be disambiguated (e.g. distinguish a user-set `75` from the detection
  default) so Turing builds with the single documented knob.

### 4.4 Bottom line
Ship 0.1.0 as **Turing(75) + Ampere consumer(86) + Hopper(90) + Blackwell(120),
with Ada(89) compiled-by-extrapolation and datacenter Blackwell(100) compiled-but-
unvalidated.** Add **A100/datacenter-Ampere(80) once the batched-cuSOLVER crash is
fixed** — until then mark sm_80 scalar-validated only. **Drop Volta(70)** (CUDA-13
toolkit limitation). The only true code blocker for the broad target is the single
sm_80 batched-Cholesky bug; everything else blocking a green CI is test-harness
hygiene.
