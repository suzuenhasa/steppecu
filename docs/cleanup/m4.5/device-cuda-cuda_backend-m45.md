# M4.5 unit review (ADVERSARIAL second pass) — `src/device/cuda/cuda_backend.cu`

Delta under audit: the M4.5 multi-GPU additions over the `main` body — `device_id` / `cudaSetDevice` threading (ctor + `set_and_return_device` + `guard_device`), the `capabilities()` probe, and the EmulatedFp64-honorable tagged degrade routed through it. The M0–M4 + B1–B27 cleanup of this file is on `main`, reviewed at 7/10 in `docs/cleanup/device-cuda-cuda_backend.md` (its F1/F2/F3/F4/F11/F12/F13a/F14/F15/F19/F20 HIGH/MED findings have all landed — re-verified against the current source). This pass audits the M4.5 delta isolated as:

1. ctor `device_id` parameter + `set_and_return_device` member-init hook (lines 94–108, 569–582);
2. `guard_device()` at the head of `compute_f2` / `compute_f2_blocks` / `decode_af` (114, 227, 402, 549–561);
3. `capabilities()` — the new probe (470–547);
4. the factory `make_cuda_backend(int device_id)` (609–611).

Everything below each compute method's degenerate guard (GEMM orchestration, bucket loop, VRAM budget) is M0–M4 code reviewed previously and out of scope except where the delta touches it (the `guard_device` hot-path cost, the data-bouncing question, the per-chunk sync).

This is the SECOND, adversarial pass. The first pass (score 9/10, 22 findings, 5 perf) is re-verified finding-by-finding against the source and the official NVIDIA docs below. **One first-pass finding (Corr-2) is materially WRONG on its load-bearing CUDA-behavior citation and is corrected here; Perf-1 is strengthened (CUDA-12 eager-init); two new substantiated findings are added (an unnecessary device-bounce in the probe revealed by the `cudaMemGetInfo`-vs-`cudaGetDeviceProperties` current-device asymmetry, and a per-call `guard_device` redundant with the immediately-following GEMM-path device assert).** Net: +2 added, −1 corrected-not-deleted (Corr-2 kept but its rationale rewritten), 0 deleted-as-false.

Files read fully for context: the target; `device/backend.hpp` (the `ComputeBackend` + `BackendCapabilities` contract, with the M4.5 `capabilities()` default base impl); `device/cuda/check.cuh` (`STEPPE_CUDA_WARN`/`STEPPE_CUDA_CHECK`, the `[[nodiscard]]` on `cuda_warn`); `device/cuda/handles.hpp` (`CublasHandle` device-ordinal record-and-assert + `MathModeScope`); `device/cuda/device_buffer.cuh` (the RAII allocate-on-current-device contract); `device/cuda/f2_block_kernel.cuh` (`emulation_honorable`/`engage_f2_precision`); `src/device/resources.cpp` (the dominant `capabilities()` caller — `make_cuda_backend(ordinal)` then immediate `capabilities()`); `src/core/fstats/f2_blocks_multigpu.cpp` (reads `gpus[0].caps.can_access_peer`, zero-copy `col_off` sub-views); `include/steppe/config.hpp` (`Precision` field order, `kDefaultMantissaBits`, `kMaxVramUtilizationFraction`); `tests/reference/test_backend_capabilities_probe.cu` (the gate, incl. `device_is_pro_tier` re-querying `cudaGetDeviceProperties` for `prop.name`); the prior `docs/cleanup/device-cuda-cuda_backend.md`.

### Verified against authoritative NVIDIA docs (cited at each use)

- **CUDA 12.0+ `cudaSetDevice` now EXPLICITLY initializes the runtime + primary context** after changing the current device (previously deferred to the first runtime call): "As of CUDA 12.0, `cudaSetDevice()` will now explicitly initialize the runtime after changing the current device for the host thread… the `cudaInitDevice()` and `cudaSetDevice()` calls initialize the runtime and the primary context associated with the specified device." [CUDA Runtime API — Device Management / Initialization](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html). Material to `set_and_return_device`, `guard_device`, and the Perf-1 "short-circuit" claim. **The docs do NOT state that a `cudaSetDevice` to the already-current device is detected and short-circuited.**
- **`cudaDeviceCanAccessPeer(&can, dev, dev)` (SAME device) returns `cudaSuccess` with `can == 0` — it does NOT return `cudaErrorInvalidDevice`.** Confirmed empirically on the NVIDIA forum (`cudaDeviceCanAccessPeer(&can, 0, 0)` → `ret == cudaSuccess`, `can == 0`) [NVIDIA Developer Forums — "Peer access not supported between a device and itself"](https://forums.developer.nvidia.com/t/peer-access-not-supported-between-a-device-and-itself/167665) and the runtime-API peer page documents only `cudaSuccess`/`cudaErrorInvalidDevice` without keying the latter to the self-pair [CUDA Runtime API §6.14 Peer Device Memory Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html). **This corrects the first-pass Corr-2 claim.**
- **`cudaMemGetInfo(size_t* free, size_t* total)` reports for the CURRENT device** (no device argument in the signature) [CUDA Runtime API §Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html). Material: the probe's `cudaSetDevice(device_id_)` is REQUIRED by `cudaMemGetInfo` (it has no device arg) but NOT by `cudaGetDeviceProperties(&prop, device_id_)` (explicit device arg).
- **`cudaDeviceGetAttribute` fetches one scalar attribute; `cudaDevAttrComputeCapabilityMajor` (75) / `cudaDevAttrComputeCapabilityMinor` (76) are valid attributes** [CUDA Runtime API — Device Management / `cudaDeviceAttr`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html). Material to Perf-3 (the lean alternative to the full `cudaGetDeviceProperties` struct).
- **cuBLAS context is bound to the CUDA context current at `cublasCreate`** (cuBLAS §2.1.2) — confirms why `set_and_return_device` must precede `blas_` construction.
- **`cublasSetStream` unconditionally resets the cuBLAS workspace to the default pool** (cuBLAS §2.4.7) — the F1/X-1 invariant `CublasHandle::set_stream` re-applies; the M4.5 delta does not touch it (still bound once in ctor).
- **CUDA grid axis caps `(2^31−1, 65 535, 65 535)`** [CUDA C++ Programming Guide §Compute Capabilities] — relevant only to confirm no launch is in the delta.

---

## Role & layering

This TU is the only place a host caller meets the GPU f2/decode path; it implements the CUDA-free `ComputeBackend` seam, and CUDA stays PRIVATE to `steppe_device` (§4). The M4.5 delta does NOT disturb that: `capabilities()` returns the CUDA-free `BackendCapabilities` POD by value (only `int`/`std::size_t`/`bool` cross the seam, `backend.hpp:144-185`); the factory still returns `std::unique_ptr<ComputeBackend>` with `device_id` defaulted in the CUDA-free `backend_factory.hpp`; and the combine *policy* (the P2P-vs-host fork, the fixed `g=0..G-1` sum) lives correctly ABOVE this seam in `core/fstats/f2_blocks_multigpu.cpp` and `device/p2p_combine.*`. This file only *probes* and *binds*, never *combines* — exactly the per-device-instance contract (`backend.hpp:193-202`) and cleanup §(2).1–§(2).4 split.

The delta is, on the whole, very good: the member-init-order argument for `set_and_return_device` is correct and load-bearing; the probe's save/restore makes `capabilities()` genuinely `const`-and-device-neutral within a host thread; every probe field is parity-neutral by construction (§12), so the locked bit-identity is untouched. The deductions concentrate on (a) two **performance** points the user flagged — `guard_device` issues an unconditional `cudaSetDevice` on every compute entry on a doc-UNSUPPORTED "short-circuit" claim that CUDA-12 eager-init makes worse, and the probe pays a `cudaSetDevice` round-trip wider than the one query (`cudaMemGetInfo`) that actually needs it; (b) a redundant `cudaGetDeviceProperties` full-struct query for two `int`s; (c) a **corrected** false-positive on the `peer==device_id_` skip rationale; (d) a soon-to-be-live thread-locality contract gap; and (e) comment-density/DRY-of-prose debt. None breaks parity.

---

## Score: 9/10 — the multi-GPU delta is clean, correct, and parity-neutral; what holds it off 9.5–10 is an unconditional per-call `cudaSetDevice` on a doc-unsupported "no-op" claim (made worse by CUDA-12 eager-init), a probe that bounces the current device wider than `cudaMemGetInfo` needs while full-struct-querying for two ints, a corrected same-device-peer rationale, and prose duplication

The M4.5 delta is materially better than the 7/10 body it sits on and is the right shape: a thin device-binding + observability layer above an untouched parity-locked compute core. The deductions are senior-bar polish, not correctness or parity defects. The first pass earned its 9; this pass keeps it at 9 but reattributes the points: the single most-important *substance* change is correcting the `cudaErrorInvalidDevice` false citation (Corr-2) — a review that asserts wrong CUDA behavior is exactly what the QUALITY MANDATE forbids — and tightening Perf-1 with the CUDA-12 eager-init fact. Fix Perf-1/Perf-3/Perf-6 (the device-bounce narrowing), Edge-2, and Read-1 and the unit is a 9.5–10.

---

## Findings

### Performance (first-class this pass)

#### Perf-1 [MED perf, PARITY-SAFE: yes] `guard_device()` issues an unconditional `cudaSetDevice` on every compute entry; the comment claims the runtime "short-circuits a redundant set" — UNSUPPORTED by the docs, and CUDA-12 eager-init makes a redundant set potentially non-trivial
Location: `guard_device()` (549–561), called at the head of `compute_f2` (114), `compute_f2_blocks` (227), `decode_af` (402).

The comment (555–557) asserts: *"On the single-GPU path this re-selects device 0 — a cheap no-op-equivalent (the runtime short-circuits a redundant set) and ZERO behavior change."* This is an unverified CUDA-behavior claim. The official Device-Management page does **not** state that `cudaSetDevice` detects "already current" and short-circuits. Worse, **CUDA 12.0 changed `cudaSetDevice` to explicitly initialize the runtime + primary context** when changing the current device ([CUDA Runtime API — Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)); while the primary context is already initialized by `set_and_return_device` at construction, the comment's confident "the runtime short-circuits a redundant set" is precisely the kind of from-memory device-behavior assertion this audit must reject. The absolute cost is tiny (the f2 GEMMs dwarf it; the forum-reported single-`cudaSetDevice` overhead is sub-millisecond once the context exists), so this is a *claim-correctness* and *unconditional-on-single-GPU* finding, not a hot-loop regression.

Why it matters: §11.4 needs the `cudaSetDevice` ONLY when the ambient device might differ from `device_id_` (the multi-GPU interleave the per-device-instance contract permits). On single-GPU it is provably redundant, and the justification must not rest on an unverified short-circuit. The QUALITY MANDATE forbids asserting CUDA behavior from memory.

Concrete fix: keep `guard_device` unconditional (it is correct and the multi-GPU model needs it), but **rewrite the comment** to the honest, doc-backed claim — "no documented synchronization; NOT verified to short-circuit the already-current case, so it is a real (cheap, post-init) `cudaSetDevice` every entry, justified by the multi-GPU interleave contract; CUDA-12 `cudaSetDevice` re-runs runtime init bookkeeping ([Device Management])." If a measured single-GPU regression ever appears, the principled optimization is `int cur; cudaGetDevice(&cur); if (cur != device_id_) cudaSetDevice(device_id_);` — but `cudaGetDevice` is itself a runtime call, so the conditional only wins if `cudaGetDevice` is materially cheaper than a post-init `cudaSetDevice` (plausible: it reads thread-local state), which must be **measured** before claiming. Comment fix now; conditional-set is a profile-driven follow-up.

Severity: med. Effort: S (comment) / M (measured conditional). PARITY-SAFE: yes — device selection moves no arithmetic bits (§12).

Adversarial check: is `guard_device` itself wrong / removable? No. The per-device-instance contract (`backend.hpp:198`) plus the `CublasHandle` device-ordinal debug-assert (`handles.hpp:147-159`) require `device_id_` current before any GEMM, and a single process legitimately interleaves G backends' calls on one host thread. Removing it reintroduces the F20 hazard. The finding is narrowly about the comment's overclaim + the unconditional nature on single-GPU.

#### Perf-2 [LOW perf, PARITY-SAFE: yes] `capabilities()` always pays a `cudaGetDevice` + `cudaSetDevice(device_id_)` + `cudaSetDevice(entry_device)` round-trip even when `device_id_` is already current
Location: `capabilities()` 483–485 and the restore at 545.

The dominant caller (`resources.cpp:91-92`) constructs `make_cuda_backend(ordinal)` — which leaves `ordinal` current via `set_and_return_device` — then *immediately* calls `capabilities()`, so on entry the current device IS `device_id_`; the set-to-`device_id_` and the restore-to-`entry_device` are both redundant in that path. It is genuinely defensive (the probe is documented `const` + device-neutral, callable from any ambient context), so it is **correct**; this is build-time-only, O(G) per process, negligible. The defensive design is the right call — a `const` query must not leak a `cudaSetDevice`. **Not** recommended for optimization on its own; see Perf-6, which is the *narrowing* of this round-trip that IS worth doing.

Severity: low. Effort: trivial. PARITY-SAFE: yes.

#### Perf-3 [LOW perf, PARITY-SAFE: yes] `capabilities()` queries the FULL `cudaDeviceProp` struct to read only `prop.major`/`prop.minor`
Location: 490–493: `cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device_id_); caps.compute_major = prop.major; caps.compute_minor = prop.minor;`.

`cudaGetDeviceProperties` populates the entire `cudaDeviceProp` (name, all attributes, clocks, etc.); the probe uses two `int` fields. CUDA exposes `cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, device_id_)` / `…Minor` which fetch exactly the requested scalar — **verified** the enums exist (`cudaDevAttrComputeCapabilityMajor` = 75, `…Minor` = 76) and are valid for `cudaDeviceGetAttribute` ([CUDA Runtime API — `cudaDeviceAttr`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)). Once-per-process-per-device, so the absolute saving is negligible — but it is precisely the "query the whole struct for one field" anti-pattern, and this very probe's `cudaMemGetInfo` comment cites the codebase's "exactly the datum needed" value (00-overview §(2).1).

Counter-consideration (why it might stay): the downstream tier gate `test_backend_capabilities_probe.cu` (`device_is_pro_tier`) re-queries `cudaGetDeviceProperties` for `prop.name` (`std::strstr(prop.name, "PRO")`). So the full struct IS needed elsewhere — see Read-2: if a `device_name`/`tier` field were added to the POD, the full query would become justified and the test would stop re-querying. As long as the POD carries only `compute_major/minor`, two `cudaDeviceGetAttribute` calls are the leaner path.

Concrete fix: either swap to two `cudaDeviceGetAttribute` calls, OR (Read-2) keep the full query and capture `prop.name` into the POD to eliminate the downstream re-query (net fewer system-wide queries). Severity: low. Effort: S. PARITY-SAFE: yes.

#### Perf-4 [N/A — no data bouncing of STATISTIC data introduced by the delta; explicitly justified]
The M4.5 delta moves no statistic data. `capabilities()` reads scalar device attributes only; `guard_device` moves nothing; the factory allocates the same backend. The H2D/D2H copies and per-chunk `cudaStreamSynchronize` in `compute_f2_blocks` (311–397) and the H2D/D2H in `compute_f2`/`decode_af` are M0–M4 code reviewed previously (prior F4/F18-perf) — not the delta. The per-device sharding in the *consumer* (`f2_blocks_multigpu.cpp:139-143`) uses **zero-copy column sub-views** (`Q.data + col_off`) — no per-device re-upload, no host round-trip of the SNP matrix — verified correct and that unit's concern. The one device-state "bounce" the delta DOES introduce (the probe's `cudaSetDevice` round-trip) is Perf-2/Perf-6, not a data copy. N/A for statistic-data bouncing.

#### Perf-5 [N/A — no missing grid-stride / launch-config issue in the delta]
No kernel launch is in the M4.5 delta. The prior F2/F3 (grid.z / feeder y-axis vs the 65 535 cap) were closed in `vram_budget.hpp::max_blocks_per_chunk` (z-clamp to `kMaxGridZ`) and the kernel-TU `grid_z_extent`/`grid_for` helpers (architecture.md §7 line 470 confirms), none of which is in this file's delta. N/A.

#### Perf-6 [LOW–MED perf, PARITY-SAFE: yes] NEW — the probe sets the current device WIDER than the single call that needs it (`cudaMemGetInfo`); `cudaGetDeviceProperties` and the peer scan don't depend on the current device, so the save/restore could shrink to one query's scope
Location: `capabilities()` — `cudaSetDevice(device_id_)` (485) guards the whole body (491–542) before the restore (545).

Verified against the docs which of the probe's queries actually require `device_id_` to be CURRENT:
- `cudaGetDeviceCount(&count)` (478): process-global, no current-device dependence.
- `cudaGetDeviceProperties(&prop, device_id_)` (491): takes an **explicit device argument** — does NOT read the current device ([CUDA Runtime API — Device Management]).
- `cudaMemGetInfo(&free_b, &total_b)` (501): **no device argument — reports the CURRENT device** ([CUDA Runtime API — Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)). **This is the ONLY call in the probe that needs `device_id_` current.**
- `cudaDeviceCanAccessPeer(&access, device_id_, peer)` (523): takes **two explicit device arguments** — does NOT read the current device.
- `emulation_honorable(emu_probe)` (542): a host predicate over a build macro — no CUDA current-device dependence.

So the `cudaSetDevice(device_id_)`/restore pair exists *solely* for `cudaMemGetInfo`. Today it brackets the entire body, which is harmless but over-broad: it makes the probe look like it depends on the current device for its property/peer reads when it does not. The minimal, intention-revealing form is to bracket only the `cudaMemGetInfo` call (set, read free/total, restore) and let the count/properties/peer/honorability reads run on the ambient device. That both narrows the device-state side-effect window and documents-by-construction which datum is current-device-relative.

Why it matters: §2 separation-of-concerns / intent-revealing code, and it removes an over-claim a future reader could mis-trust ("the probe pins the device for all its work" — it only needs to for one call). Parity-neutral (no arithmetic). It does NOT reduce the number of `cudaSetDevice` calls in the dominant caller (still one set + one restore), so the perf gain is nil; the value is *correctness-of-intent* and a tighter side-effect window — hence LOW–MED, leaning LOW.

Concrete fix: move `cudaGetDevice(&entry)`/`cudaSetDevice(device_id_)` … `cudaMemGetInfo` … `cudaSetDevice(entry)` into a tight 4-line block (or a `MemInfo probe_vram_on(device_id_)` helper that does the save/restore internally), and drop the device switch around the count/properties/peer/honorability reads. Severity: low–med. Effort: S. PARITY-SAFE: yes.

Adversarial check: is the wide bracket actually load-bearing? No — none of the other queries reads the current device (verified per-call above). The only risk is if a future field were added that DOES need the current device (e.g. a default-allocation probe); then bracketing the whole body is forward-defensive. That is a legitimate reason to keep the current shape, so this is a *suggestion* (LOW), explicitly noting the defensive counter-argument — not a defect.

### Correctness & bugs

#### Corr-1 [GOOD] The `set_and_return_device` member-init-order argument is correct and load-bearing
Location: ctor (94–95) `: device_id_(set_and_return_device(device_id))`, `device_id_` declared FIRST (582), `blas_`/`workspace_` after (597–598).

C++ initializes non-static members in **declaration order** regardless of the member-init-list order, and the ctor body runs after all members init. `device_id_` is declared first, so `set_and_return_device(device_id)` (which calls `cudaSetDevice(device_id)` and returns the ordinal) runs *before* `blas_` (`CublasHandle` → `cublasCreate`, context binds to the device current at the call, cuBLAS §2.1.2) and *before* `workspace_` (`DeviceBuffer<std::byte>` → `cudaMalloc` on the current device, `device_buffer.cuh:74` "allocate on the current device"). So both the cuBLAS context and the workspace VRAM bind to `device_id`, and `CublasHandle::device_id()` records `device_id` via its own `cudaGetDevice` (`handles.hpp:78`). Exactly correct; the comment (78–93) explains it faithfully. No bug.

Adversarial check: could `set_and_return_device` throw (invalid ordinal) and leave a half-constructed object? It throws via `STEPPE_CUDA_CHECK` *during* `device_id_`'s init, before `blas_`/`workspace_` construct — so no member needs unwinding except the trivially-destructible `int`, and `make_unique` frees the allocation as the exception propagates. Note the CUDA-12 eager-init fact STRENGTHENS this: `cudaSetDevice` now initializes the primary context, so an invalid ordinal or an init failure surfaces *here* at construction (fail-fast) rather than being deferred to the first compute call — exactly the §2 behavior wanted. Correct and exception-safe.

#### Corr-2 [CORRECTED from first pass — the skip is correct but the first pass's CITED RATIONALE is WRONG] The `peer == device_id_` skip is fine, but it does NOT prevent a `cudaErrorInvalidDevice` WARN, because the same-device pair does not error
Location: `capabilities()` 516–517 (`if (peer == device_id_) continue;`).

The first pass asserted, with a citation, that "`cudaDeviceCanAccessPeer` **returns `cudaErrorInvalidDevice` when the two device arguments are the same**" and concluded the skip is "load-bearing" because without it the self-pair "would route a real error through `STEPPE_CUDA_WARN` — emitting a spurious WARN line on every probe." **This is factually wrong.** Verified: `cudaDeviceCanAccessPeer(&can, dev, dev)` returns **`cudaSuccess` with `can == 0`** — confirmed on the NVIDIA forum (`cudaDeviceCanAccessPeer(&can, 0, 0)` → `ret == cudaSuccess`, `can == 0`) [NVIDIA Developer Forums — "Peer access not supported between a device and itself"](https://forums.developer.nvidia.com/t/peer-access-not-supported-between-a-device-and-itself/167665); the runtime-API page documents only `cudaSuccess`/`cudaErrorInvalidDevice` without keying `cudaErrorInvalidDevice` to the self-pair [CUDA Runtime API §6.14](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html). So even **without** the skip, the self-pair would take the `s == cudaSuccess && access != 0` FALSE branch (access==0) and emit **no WARN** — the first pass's "spurious WARN per probe" hazard does not exist.

What the skip actually buys: it avoids one meaningless self-comparison runtime call and makes the intent ("scan the OTHER devices") explicit. That is a clean, correct micro-optimization and a readability win — keep it — but it is **cosmetic/intent-clarifying, not error-avoiding.** The in-code comment (513–514) is accurate ("the self-pair is skipped: cudaDeviceCanAccessPeer is defined for distinct devices"); the first-pass *review prose* over-claimed the mechanism. This is the one finding where the prior review asserted CUDA behavior from memory and got it wrong; correcting it is the most important substance change in this pass.

Severity: the code is GOOD (no change needed); the finding is a review-prose correction. PARITY-SAFE: yes.

Adversarial check on my own correction: could a *driver* return `cudaErrorInvalidDevice` for the self-pair on some platform? The forum repro and the absence of any doc keying the error to the self-pair make `cudaSuccess`/`can==0` the documented/observed behavior; the code's `s == cudaSuccess && access != 0` guard is robust either way (a hypothetical `cudaErrorInvalidDevice` would WARN-and-continue, not throw), so the skip is belt-and-suspenders regardless. My correction stands; the code is unaffected.

#### Corr-3 [LOW, PARITY-SAFE: yes] `count` is both the reported `device_count` and the peer-scan upper bound — correct and DRY, but the coupling is implicit
Location: `capabilities()` 477–479 + the peer loop `for (int peer = 0; peer < count; ++peer)` (516).

The scan bounds by `count` read *before* the `cudaSetDevice(device_id_)` switch — correct (device count is process-global, not per-current-device, verified by `cudaGetDeviceCount` having no current-device dependence) and DRY (reused, not re-queried). Nit: a reader must notice `count` is both the reported `device_count` and the scan bound. A one-word comment at 516 ("over the `count` visible devices") makes it explicit. Trivial.

Severity: low. Effort: trivial. PARITY-SAFE: yes.

#### Corr-4 [GOOD] `emulated_fp64_honorable` is probed with the SAME predicate the GEMM path consults — they cannot disagree
Location: 540–542; predicate `emulation_honorable` (`f2_block_kernel.cuh:53`), also consulted by `engage_f2_precision`/`f2_compute_type`.

The probe builds `Precision{Kind::EmulatedFp64, kDefaultMantissaBits}` and calls the *exact* `emulation_honorable` that `run_f2_gemms`→`engage_f2_precision` uses. So "would an EmulatedFp64 run be honored as fixed-slice Ozaki, or downgraded?" is answered identically by the probe and the compute path; `test_backend_capabilities_probe.cu` asserts the equality. The aggregate-init field order is **verified correct**: `Precision` is `{Kind kind; int mantissa_bits;}` (`config.hpp:188,195`), so `{Kind::EmulatedFp64, kDefaultMantissaBits}` maps `kind ← EmulatedFp64`, `mantissa_bits ← 40`. Closes the prior F11 passthrough correctly. Good.

### Edge cases & failure modes

#### Edge-1 [GOOD] Empty / single-GPU / no-peer cases all handled in the probe
- **`count==1`**: the peer loop's only candidate `peer==device_id_` is skipped, `can_peer` stays false ⇒ `can_access_peer==false`; the consumer degrades to host-staged; `test_backend_capabilities_probe.cu` asserts `!can_access_peer` on a single device. Correct.
- **All-isolated (P2P driver-disabled, the 5090 tier)**: every `cudaDeviceCanAccessPeer` returns `cudaSuccess`/`access==0`, `can_peer` stays false ⇒ host-staged. The documented tagged degrade (§11.4), not a fault. Correct — and consistent with the verified same-device/no-peer `cudaSuccess` semantic (Corr-2).
- **`device_id_` out of range**: caught at construction by `set_and_return_device`'s `cudaSetDevice` throw (now eager-init, so it fails at construction, not deferred), so `capabilities()` is never reached on a bad ordinal. The §9 "configured device N not among visible" *grade* of message is `build_resources`' job (sibling `resources.md`), not here. Correct division.

#### Edge-2 [MED, PARITY-SAFE: yes] The `const` save/restore probe is documented as device-neutral but is only PER-HOST-THREAD-correct; the §11.4 "one host thread per device" model (explicitly day-one) will make that live
Location: `capabilities()` save/restore (483–485, 545); doc claim (465–469) "callable from any ambient-device context".

The CUDA current device is **per-host-thread** state. The save/restore is correct and side-effect-free *within one host thread*. Today the entire device/core combine path is single-threaded (no `std::thread`/`std::async`/OpenMP in `src/device` or `src/core/fstats` — verified), so it is safe. But architecture.md §11.4 (line 713) and the §0 objective state **"one host thread per device, per-device CUDA streams"** as a day-one SPMG model, and the sibling `resources.md` defers a per-device-thread `Resources` build as a coming workflow. The doc's framing ("leaves no side effect, callable from any context") invites a future reader to call one backend's `capabilities()` from *another* device's worker thread; the save/restore would transiently switch *that thread's* current device to `device_id_` and back — racing nothing in CUDA (current device is thread-local) but surprising a `guard_device`-less assumption if a call pattern ever shares a thread across devices.

Why it matters: this is the trust-bearing gap the sibling reviews flag for the threaded model converging on §11.4. Not a bug today; an under-documented invariant the coming workflow can trip.

Concrete fix: one doc line on `capabilities()` — "the save/restore is per-host-thread-correct (the CUDA current device is thread-local); under the §11.4 one-thread-per-device model, the restore pins nothing for *other* threads — call from the owning thread or any thread, never assume it fixes the device process-wide." Severity: med (contract precision on the path the threaded workflow will exercise). Effort: trivial. PARITY-SAFE: yes.

Adversarial check: speculative? The architecture *names* the per-device-thread model in-scope day-one (§0, §11.4 line 713) and the sibling review tracks the threaded build as planned — documenting the thread-locality of the device-current contract is warranted, not gold-plating. Kept as MED-doc.

#### Edge-3 [LOW] `capabilities()` re-trusts the ctor's ordinal validation implicitly
Location: 485, 491. `device_id_` was validated at construction (`set_and_return_device` threw if `cudaSetDevice` rejected it — and CUDA-12 eager-init means that check is real at construction), so `capabilities()` cannot fail on validity. Not a live bug. Optional: `STEPPE_ASSERT(device_id_ >= 0 && device_id_ < count)` after reading `count` would document the validated-ordinal invariant. Polish, not a defect.

Severity: low. Effort: trivial. PARITY-SAFE: yes.

### Numerical / precision (§12)

#### Num-1 [N/A — correctly parity-neutral by construction; the bit-identity gate is untouched]
The M4.5 delta performs **no arithmetic**. `device_id`, `guard_device`, every `BackendCapabilities` field, and the factory are data-movement / observability / binding levers only — §12 parity-neutral by construction, exactly as `backend.hpp:132-138` and the file banner (33–34) state. The locked bit-identity rests on the compute path (feeder → 3-GEMM → assemble, the fixed-slice Ozaki engagement, the fixed `g=0..G-1` combine in the consumer), none of which this delta changes. `test_f2_multigpu_parity.cu` proves memcmp-identity across G for both combine tiers via `make_cuda_backend(device_id)` — the path the delta provides. The one place the delta *could* have leaked parity (running a backend's GEMMs on the wrong device) is precisely what `guard_device` *prevents* (the F20 fix), with `CublasHandle`'s debug assert as the belt to its suspenders. Explicitly justified N/A.

### CUDA idioms / RAII / async semantics / launch config / occupancy (§7)

#### Cuda-1 [GOOD] `STEPPE_CUDA_WARN` vs `STEPPE_CUDA_CHECK` split exactly along the §11.4 capability-tier law
Location: `STEPPE_CUDA_CHECK` on `cudaGetDeviceCount`/`cudaGetDevice`/`cudaSetDevice`/`cudaGetDeviceProperties`/`cudaMemGetInfo` (478/484/485/491/501/545), `STEPPE_CUDA_WARN` on the single recoverable call `cudaDeviceCanAccessPeer` (522–523).

Textbook realization of `check.cuh`'s CAP-1/CAP-2 contract: a backend that cannot enumerate/select/query its OWN device is a real fault (throw); "cannot peer-access" is an EXPECTED budget-tier degrade (WARN + branch, never throw). The `const cudaError_t s = STEPPE_CUDA_WARN(...)` correctly consumes the `[[nodiscard]]` return (`check.cuh:134`) and the `s == cudaSuccess && access != 0` guard (524) treats ANY non-success status as "no peer", never trusting an unwritten `access`. Given the corrected Corr-2 (the self-pair returns `cudaSuccess`/access==0, not an error), the WARN here only fires on a *genuine* probe failure (a real driver error querying a distinct peer) — exactly the recoverable-status intent. Exemplary. Keep.

#### Cuda-2 [GOOD] `guard_device` / `set_and_return_device` are the ONLY `cudaSetDevice` callers; the RAII wrappers stay record-and-assert
The backend is the legitimate *owner* that SELECTS the device; `CublasHandle` only *records and asserts* it (`handles.hpp:147-159`, never `cudaSetDevice`); `DeviceBuffer` allocates on whatever is current. The delta keeps that separation clean — no RAII wrapper gained a hidden `cudaSetDevice`. §7 "no global mutable state in the wrappers" honored. Good.

#### Cuda-3 [LOW] `capabilities()` is a ~77-line sequence of raw runtime calls rather than small named helpers; the peer-scan is the one extractable sub-unit
Location: 470–547. Six concerns inline: device-count, save/restore device, compute-capability, VRAM, peer-scan, honorability. Readable (heavily commented), but the peer-scan loop (515–529) is the one sub-unit with real logic (the skip + WARN-branch + early `break`) and could be a `static bool probe_any_peer(int device_id, int count)` free function — host-CUDA, unit-testable against a mock count/self-pair, shrinking `capabilities()` to a flat field-fill. Pairs with Perf-6 (a `probe_vram_on(device_id_)` helper that owns the save/restore). Minor; not a monolith on the M4 `compute_f2_blocks` scale.

Severity: low. Effort: S. PARITY-SAFE: yes.

### Magic numbers & hardcoded values (§4)

#### Mag-1 [GOOD] No magic number in the production probe
The probe reports `prop.major/minor` verbatim (492–493); the "≥ 12" sm_120 floor lives only in the test (`test_backend_capabilities_probe.cu`), correct because one sm_120 build serves both boxes (§0) so the value is observability, not a dispatch key (comment 487–489 says exactly this). The honorability probe uses the named `kDefaultMantissaBits` (`config.hpp:44`). No literal `0`/`1` survives except loop bounds. Clean.

### Decomposition / single-responsibility / function size (§2)

#### Dec-1 [GOOD] The delta's additions are each single-purpose and small
`set_and_return_device` (one statement + return), `guard_device` (one statement), `capabilities()` (a field-fill probe). The ctor stays small (device-set folded into the member-init). Right decomposition for a binding+probe layer; the only sub-extraction worth considering is the peer-scan + a vram-probe helper (Cuda-3, Perf-6). Good.

### Readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density

#### Read-1 [MED — the dominant cleanliness debt] Comment density on the delta is very high and partly re-narrates contracts single-homed elsewhere
Location: ctor doc (72–93, 22 lines), `capabilities()` doc (450–469) + per-field inline blocks, `guard_device`/`set_and_return_device`/`device_id_` docs (549–582).

The delta adds ~120 lines, a large majority prose. Much is load-bearing *why* (the member-init-order argument; the peer-skip rationale — which, post-Corr-2, should be stated as "intent-clarifying, the self-pair is a no-op `cudaSuccess`/0" NOT "avoids a WARN"; the WARN-routing justification). But several blocks re-state contracts with a single authoritative home:
- The per-device-instance contract is narrated in the banner (9–20), the ctor doc (72–77), `guard_device` (550–554), `device_id_` (574–581), AND `set_and_return_device` (565–568) — five copies in one file, atop `backend.hpp:193-202` and `backend_factory.hpp`.
- The "`can_access_peer` true ⇒ P2P, false ⇒ host-staged tagged degrade" gate is narrated here (505–514), in `backend.hpp:166-175`, in `f2_blocks_multigpu.cpp:156-167`, and in `p2p_combine.hpp` — four+ homes for one gate.

Why it matters: §2 readability; a reviewer must reconcile multiple copies to be sure none drifted. Keep the *mechanism* comments verbatim; collapse the *contract restatements* to one-line cross-refs ("per-device-instance contract: see `backend.hpp:193`").

Concrete fix: trim contract-restating prose to cross-refs; **and** fix the peer-skip comment block (511–514) so it no longer leans on any "self-pair errors" implication — state the verified `cudaSuccess`/`access==0` semantic. Severity: med (largest cleanliness debt; a drift risk; now also a correctness-of-prose item via Corr-2). Effort: M. PARITY-SAFE: yes.

#### Read-2 [LOW] `cudaDeviceProp prop{}` is queried but `prop.name` (needed downstream for tier detection) is discarded; the POD carries no human tier
Location: 490–493; downstream `test_backend_capabilities_probe.cu` (`device_is_pro_tier`) re-queries `cudaGetDeviceProperties` solely for `prop.name`.

Either (a) **Perf-3 path**: stop fetching the full struct here (use `cudaDeviceGetAttribute`), accepting the test fetches its own properties; or (b) **consolidation path**: since the full struct is already fetched here, capture `prop.name` (or a derived `bool is_datacenter_tier` / a fixed `char device_name[256]`) into `BackendCapabilities`, eliminating the downstream re-query and giving `Resources`/logs a human tier string for the §10 capability tag (`last_combine_path` is an enum, not a tier name). (b) is the cleaner system-level move but grows the CUDA-free POD by a fixed buffer (keep it POD-trivial — no `std::string` — for the seam) and is arguably beyond M4.5 scope. Flag for consideration.

Severity: low. Effort: S. PARITY-SAFE: yes.

#### Read-3 [GOOD] const-correctness, `[[nodiscard]]`, `noexcept`, `override` on the delta
- `capabilities()` is `[[nodiscard]] … const override` — matches `backend.hpp:304`. Correct: it mutates no member; the device save/restore is thread-local CUDA state, not object state, so `const` is honest.
- `guard_device()` is `const` (mutates no member; the `cudaSetDevice` is external state) — correct, and callable from the `const` methods. Verified `capabilities()` does NOT call `guard_device` (it does its own save/restore because it must *restore* the entry device) — no redundant double-set. Good.
- `set_and_return_device` is `[[nodiscard]] static` — correct (callable during first-member init, no `this`).
- `make_cuda_backend(int device_id)` is `[[nodiscard]]` matching the factory decl.
- Correctly NOT `noexcept` anywhere (they call throwing `STEPPE_CUDA_CHECK`). Consistent.

### Layering / API / ABI (§4)

#### Lay-1 [GOOD] The CUDA-free seam survives the delta intact
`capabilities()` returns `BackendCapabilities` by value — CUDA-free POD (`backend.hpp:144-185`). The factory's `int device_id` is a plain int defaulted in the CUDA-free `backend_factory.hpp`. No CUDA type leaks; `resources.cpp` consumes `caps.device_count`/`caps.can_access_peer` with no `<cuda_runtime.h>` (verified). Combine policy stays above the seam. Textbook §4.

#### Lay-2 [GOOD] The factory default `device_id = 0` lives in the header, defined unqualified here — ODR-correct
`backend_factory.hpp` declares the default; `cuda_backend.cu:609` defines `make_cuda_backend(int device_id)` without repeating it. Verified no duplicate default / ODR hazard. Every zero-arg call site binds to device 0 unchanged.

### Testability (§13)

#### Test-1 [GOOD] The probe + binding are gated on a live device; honorability equality asserted against the library predicate
`test_backend_capabilities_probe.cu` pins `compute_major>=12`, `total>0`, `free<=total`, `device_count==cudaGetDeviceCount`, `emulated_fp64_honorable==emulation_honorable(EmulatedFp64)`, device-0/device-1 construction, and the tier-keyed `can_access_peer` (PRO ⇒ true; GeForce ⇒ false-is-expected). `test_resources_build.cu` pins per-device binding order; `test_f2_multigpu_parity.cu` proves the device-bound path is memcmp-identical across G. Thorough.

#### Test-2 [LOW] The peer-scan branch is exercised only on a live multi-GPU box
The skip + WARN-branch + early-break runs only with `count>=2`. On 1-GPU CI the loop body never runs (only candidate is `device_id_`, skipped). Cuda-3's `probe_any_peer(device_id, count)` extraction would let the skip-self / all-isolated arms be unit-tested host-side behind a CUDA-call seam. Minor — the live gate covers the real path. Folds into Cuda-3.

Severity: low. Effort: S. PARITY-SAFE: yes.

### Capability-tier coherence (§(2))

#### Cap-1 [GOOD] The tag is off the numeric payload; the probe is the input, the discovered tag lives on `Resources`
`capabilities()` produces only the *probe input* (`BackendCapabilities`), which `build_resources` stores in `PerGpuResources::caps` (`resources.cpp:92`, the ONE probe owned at construction, §(2).1). The *which-path tag* (`last_combine_path`) is set by the consumer on `Resources` (`f2_blocks_multigpu.cpp:183,200`), never on `F2BlockTensor` (§(2).2). The override knobs (`prefer_p2p_combine`) live on `DeviceConfig`. The P2P answer is NON-throwing (§(2).4). This file holds up its quarter cleanly: it probes, it does not decide or tag. The honorability field is correctly *on the probe* (a build fact) while the combine path is *off it* (a runtime tag). Good.

#### Cap-2 [LOW, cross-unit] `can_access_peer == true iff reaches AT LEAST ONE peer` is the right semantic for the root, but the field name doesn't convey "any-peer"
Location: 515–529; field `BackendCapabilities::can_access_peer`.

The probe sets `can_access_peer = true` if `device_id_` can reach *any* one peer (early `break`). For the combine *root* (gpus[0]) this is right — it needs ≥1 peer to pull from (comment 511–513). The consumer gates P2P on **only the root's** bool (`f2_blocks_multigpu.cpp:172`, `gpus[0].caps.can_access_peer`). On an *asymmetric* fabric (root reaches A but not B) the root's `can_access_peer==true` would green-light a P2P combine that then fails its `cudaMemcpyPeer` from B — caught fail-fast by `p2p_combine`'s throwing `STEPPE_CUDA_CHECK` on the DMA, so not a parity/correctness break, just a later/worse error than a per-peer gate. A consumer/contract nuance, surfaced while auditing what this probe hands downstream; the probe's any-peer semantic is correctly documented here.

Concrete fix (cross-unit, optional): rename to `can_access_any_peer`, or have the P2P gate verify each owning peer's reachability (defense-in-depth; the transport already fails-fast). Severity: low. Effort: S. PARITY-SAFE: yes.

---

## Considered & rejected (incl. rejected-for-parity)

- **REJECTED-FOR-PARITY: skip `guard_device` on the single-GPU path (`device_id_==0`) to save the per-call `cudaSetDevice`.** Tempting for Perf-1, but `guard_device` also satisfies the `CublasHandle` device-ordinal debug-assert and protects against a host that changed the ambient device between calls (the multi-GPU interleave the contract permits). Skipping it on device 0 reintroduces the F20 hazard for any process interleaving a device-0 and a device-1 backend on one host thread. Parity is unaffected either way (device selection moves no bits), but the *correctness contract* requires the guard. Reject removal; keep the comment fix + measured-conditional follow-up.
- **REJECTED-FOR-PARITY: move per-device combine into the probe / have `capabilities()` pre-enable peer access.** Peer-enabling is the P2P transport's job (`p2p_combine.hpp`, lazily); doing it in the probe leaks device state out of a `const` query and is wasted on the host-staged tier. The probe must stay a pure parity-neutral observation; pulling combine work in risks coupling the locked fixed-order sum to the binding layer. Rejected; the §(2)/§11.4 split (probe here, combine above) is correct.
- **Rejected: cache `capabilities()` so repeated calls don't re-probe.** Called O(G) times per process at build, never on the hot path; caching adds mutable state to a `const` method for no measurable gain.
- **Rejected (corrected, not deleted): the first pass's Corr-2 "the skip avoids a spurious `cudaErrorInvalidDevice` WARN".** Not deleted because the skip itself is good; *corrected* because the cited CUDA behavior is wrong — `cudaDeviceCanAccessPeer(dev,dev)` returns `cudaSuccess`/`access==0`, so no WARN would fire even without the skip. See Corr-2.
- **Rejected: `2u * pm` / `2u * two_pm` could overflow `std::size_t` in the delta.** Not the delta — those are M0–M4 lines; casts to `size_t` precede the multiply, and at the ceiling (P≈4266, M≈584k) `2*pm ≈ 5e9 ≪ 1.8e19`. Out of scope and not an issue anyway.
- **Rejected: `cudaGetDevice`-then-conditional-`cudaSetDevice` in `guard_device`.** Listed in Perf-1 as a *measured-only* follow-up, not a recommendation: `cudaGetDevice` is itself a runtime call, so the conditional only wins if it is materially cheaper than a post-init `cudaSetDevice` — unverified; the QUALITY MANDATE forbids asserting it from memory. Profile first.
- **Rejected: add a null-check guard on `make_cuda_backend`.** `std::make_unique` never returns null (throws `bad_alloc`). No action.
- **Rejected: `capabilities()` should validate the §9 duplicate/out-of-range ordinal contract.** That is `build_resources`' job (a backend bound to one ordinal cannot see the configured set). The probe validates only its own device (via the ctor's eager-init `cudaSetDevice` throw). Out of scope.
- **Rejected: `device_count`/peer-scan could race a concurrent device hot-plug.** steppe is single-process; device count is stable for the process lifetime (§11.4). No race.
- **Rejected: shrinking the probe's `cudaSetDevice` to skip it entirely (rely on the ambient device for `cudaMemGetInfo`).** No — `cudaMemGetInfo` reports the *current* device (verified: no device argument), so without the set it would report the ambient device's VRAM, not `device_id_`'s — a real bug. The narrowing in Perf-6 keeps the set around `cudaMemGetInfo`; it does NOT remove it.

---

## What it takes to reach 10/10

All PARITY-SAFE (none touches arithmetic or the fixed combine order):

1. **Perf-1 (med, S):** rewrite the `guard_device` comment to the honest, doc-backed claim — drop "the runtime short-circuits a redundant set"; note CUDA-12 `cudaSetDevice` re-runs runtime init bookkeeping and is NOT verified to short-circuit. Keep the guard unconditional. The measured-conditional `cudaGetDevice`-gate is a profile-driven follow-up, not required.
2. **Corr-2 + Read-1 (med, S/M):** fix the peer-skip comment (511–514) to the *verified* same-device semantic (`cudaSuccess`/`access==0`, the skip is intent-clarifying not error-avoiding), and collapse the five in-file restatements of the per-device contract + the P2P gate to one-line cross-refs (keep the load-bearing member-init-order / WARN-routing mechanism comments). This is the largest standing cleanliness + prose-correctness debt.
3. **Perf-6 (low–med, S):** narrow the probe's `cudaSetDevice` save/restore to bracket only `cudaMemGetInfo` (the sole current-device-relative call), letting count/properties/peer/honorability run on the ambient device — or extract a `probe_vram_on(device_id_)` helper that owns the save/restore.
4. **Perf-3 + Read-2 (low, S):** either swap `cudaGetDeviceProperties` for two `cudaDeviceGetAttribute` calls, OR capture `prop.name` into the POD so the downstream tier-keying stops re-querying — pick the system-level-leaner and document it.
5. **Edge-2 (med, trivial):** add the per-host-thread caveat to the `capabilities()` device-neutrality doc, ahead of the §11.4 one-thread-per-device workflow.
6. **Cuda-3 + Test-2 (low, S):** extract `probe_any_peer(device_id, count)` so the skip-self / all-isolated branches are unit-testable behind a CUDA-call seam; flatten `capabilities()` to a field-fill.
7. **Corr-3 (low, trivial):** one-word comment coupling `count` to the peer-loop bound; **Edge-3 (low, trivial):** optional `STEPPE_ASSERT(0 <= device_id_ < count)`.
8. **Cap-2 (low, S, cross-unit):** consider renaming `can_access_peer` → `can_access_any_peer`, or per-owning-peer reachability in the gate (defense-in-depth; the transport already fails-fast).

After 1–5 land, this is a clean 9.5; 6–8 plus the POD/tier-name consolidation (Read-2) close it to 10.

---

## Good patterns to keep

- **The `set_and_return_device` member-init-order hook** — selecting the device while initializing the first-declared member so it is sequenced before `blas_`/`workspace_`. A precise, correct solution to the cuBLAS-context-binding ordering problem (cuBLAS §2.1.2), now *strengthened* by CUDA-12 eager-init (the device + primary-context fail fast at construction). Textbook §7.
- **`STEPPE_CUDA_WARN` on `cudaDeviceCanAccessPeer` only, `STEPPE_CUDA_CHECK` on every genuine fault** — the §11.4 capability-tier law with surgical precision; the tagged degrade never faults, the real faults never WARN-and-continue. Given the verified same-device `cudaSuccess`/0 semantic, the WARN here fires only on genuine probe failures — exactly the recoverable-status intent.
- **The `peer == device_id_` skip** — correct and intent-clarifying (the self-pair is a meaningless `cudaSuccess`/`access==0` no-op; the skip avoids the wasted call and reads as "scan the others"). Keep it; just state the rationale accurately (Corr-2).
- **The honorability probe consults the SAME `emulation_honorable` predicate the GEMM path uses** — probe and compute path can never report different EmulatedFp64 honorability; the test asserts the equality. The right way to close the prior F11 passthrough.
- **The `const` + save/restore device-neutral probe** — `capabilities()` leaves no `cudaSetDevice` side effect within a host thread, so it is callable from any ambient-device context. Correct (add the per-host-thread caveat, Edge-2; narrow the bracket to `cudaMemGetInfo`, Perf-6).
- **Parity-neutral by construction, stated and true** — every delta field is data-movement/observability/binding only; the locked bit-identity (proven by `test_f2_multigpu_parity.cu`) is untouched, and the delta is the binding layer that *prevents* the wrong-device parity hazard (the F20 fix) rather than introducing one.
- **The CUDA-free seam through the delta** — `BackendCapabilities` POD by value, `int device_id` defaulted in the CUDA-free factory header, combine policy kept above the seam. §4 dependency direction preserved.
