# Unit review (ADVERSARIAL SECOND PASS) — `core/fstats/f2_blocks_multigpu` (the SPMG precompute entry point / orchestrator)

Files audited in full:
- `src/core/fstats/f2_blocks_multigpu.hpp` (the public-to-core decl + the parity contract doc)
- `src/core/fstats/f2_blocks_multigpu.cpp` (`compute_f2_blocks_multigpu`, the orchestrator)

Context RE-READ line-by-line to ground or refute every first-pass claim (not taken on faith from the draft): `device/resources.hpp`/`.cpp` (the injected bundle, `CombinePath` tag, `build_resources`, `device_count()` return type), `device/shard_plan.hpp`/`.cpp` (`plan_block_shards`/`DeviceShard`, the `block_sizes`/`ranges` parallel-input contract), `core/fstats/f2_combine.hpp`/`.cpp` (the host-staged baseline — the zero-init + fixed-order `+=`), `device/p2p_combine.hpp` AND `device/cuda/p2p_combine.cu` (the P2P fast-path — it DOES call `cudaDeviceEnablePeerAccess`, and it is per-partial-synchronous), `device/backend.hpp` (the `ComputeBackend` seam + `BackendCapabilities`), `device/cuda/cuda_backend.cu:221-398` (`compute_f2_blocks` — CONFIRMED each per-device call is a self-contained, blocking, full-shard H2D-from-pageable upload + feeder + grouped GEMM + trailing `cudaStreamSynchronize`), `core/domain/block_partition_rule.hpp` (`block_ranges`/`BlockRange`, the throwing `block_id.size() < M` guard, the `M<=0||n_block<=0 ⇒ {}` early-out), `core/internal/views.hpp` (`MatView`), `include/steppe/config.hpp` (`Precision`, `DeviceConfig`, the TWO-knob banner for `prefer_p2p_combine` vs `enable_peer_access`), `core/internal/host_device.hpp` (`STEPPE_ASSERT`), `core/internal/log.hpp` (**`STEPPE_LOG_WARN` — CONFIRMED a no-op under NDEBUG**), `core/fstats/f2_from_blocks.cpp` (the sibling orchestrator's `validate_qvn`/`validate_partition` convention), `src/core/CMakeLists.txt` (steppe_core does NOT link a threads library), `tests/reference/test_f2_multigpu_parity.cu` (the locked bit-identity gate — CONFIRMED it covers only balanced real AADR, both `prefer_p2p_combine` arms, no degenerate geometries). Architecture §2/§4/§7/§9/§10/§11.4/§12/§13 re-read. CUDA host-API behaviour verified against the CUDA Programming Guide §3.4 "Programming Systems with Multiple GPUs" and the NVIDIA "How to Overlap Data Transfers in CUDA C/C++" blog (cited inline); a fabricated quote in the first pass is corrected below.

---

## Role & layering

`compute_f2_blocks_multigpu` is the single-node multi-GPU (SPMG) precompute entry point (architecture §5 S2, §11.4): it takes the injected `Resources&` bundle + the FULL Q/V/N + the shared `BlockPartition` + a `Precision`, and returns the combined `[P × P × n_block]` `F2BlockTensor`. It does exactly: (a) Q/V/N+partition debug-validation; (b) `G < 1` throw; (c) `G == 1` fast-path that forwards verbatim to the lone backend; (d) `G >= 2` block-aligned shard → per-device `compute_f2_blocks` over zero-copy sub-views → (e) capability-gated fixed-order combine (host-staged baseline vs the opt-in device-resident P2P).

**Layering is correct and is the unit's strongest property — VERIFIED, not assumed.** I read all three `device/` headers it includes (`resources.hpp`, `p2p_combine.hpp`, `shard_plan.hpp`) and confirmed each carries a "CUDA-FREE BY CONTRACT" guarantee and includes no `<cuda_runtime.h>`; the CUDA bodies (`cuda_backend.cu`, `p2p_combine.cu`) live in `steppe_device` and are reached only through the CUDA-free decls. The `.cpp` includes no CUDA header, issues no GEMM, allocates no device memory. The combine *policy* (the §4 gate) is host-side intent over the injected `Resources` and rightly lives here. The fixed g=0..G-1 order, the block-aligned whole-block shard, and the zero-init combine — the three pillars of the §12 bit-identity the locked parity test proves — all hold. CMake confirms `steppe_core` links `steppe::device` PRIVATE and archives no nvcc dlink object; the unit physically cannot pull CUDA.

**Verdict shift from the first pass.** The first pass scored 8/10 and was *substantially correct* on the dominant fact (P1: the G devices run strictly sequentially). My second pass CONFIRMS P1, P2, P3, P5, C1, C2, D1, R1, T1, T2, CT2, and the capability-tier coherence findings against the actual code and the cited docs. It also finds **three things the first pass got wrong or missed**, two of them material:
- **NEW high-value miss (W1): the §11.4-mandated degrade WARN is a NO-OP in release.** `STEPPE_LOG_WARN` (log.hpp:44) expands to `((void)0)` under NDEBUG. The "architecture-mandated tagged degrade" at .cpp:196-199 therefore produces NO output in any production (Release) build — the exact build that ships to the budget box where the degrade ALWAYS fires. The first pass repeatedly praised this WARN (CT1, "precise and non-spammy") without noticing it is compiled out. This is a real §11.4 ("explicit logged fallback") gap.
- **CORRECTION (CU2 is built on a fabricated premise):** the first pass asserts "`STEPPE_NVTX_SCOPE` is in `core/internal/nvtx.hpp` (host-pure)". **There is no `nvtx.hpp` and no `STEPPE_NVTX_SCOPE` anywhere in the tree** (`grep -rn NVTX src include` → empty). The recommendation is not "add one include"; it is "first build an NVTX facade, then use it". Downgraded and re-scoped below.
- **NEW (W2): P1's fan-out needs a build change the first pass omitted** — `steppe_core` does not link `Threads::Threads`/pthread, so `std::jthread` would not link as-is.

These do not move the structural conclusion. The unit is correct, parity-safe, and cleanly layered; it is held back by the unrealized concurrency (P1), a release-silent degrade (W1), redundant host materializations (P2/P3/P4), a debug-assert convention that diverges from the sibling (C1/C2), the half-enforced two-knob gate (CT2), and a missing host-only test (T1). Score: **8/10**, same number as the first pass but for a tightened reason set.

---

## Score: 8/10 — clean, correct, parity-safe layering and an honest G==1 fast path; held back by a fully-sequential G-device drive (the headline multi-GPU speedup is unrealized), a §11.4 degrade WARN that is compiled out in release, several duplicated host buffers, a debug-validation that diverges from the sibling orchestrator, a half-enforced two-knob P2P gate, and casting noise

The orchestrator is correct and parity-safe — the locked memcmp gate proves it across G and precision. But this is the *performance pass*, and the dominant performance fact is structural and unaddressed: **the per-device GEMM work is issued strictly one device at a time**, each `compute_f2_blocks` blocking on its own trailing `cudaStreamSynchronize(stream_)` (cuda_backend.cu:397) before the next device's call is even *issued*. On the 2-GPU box this makes wall-clock `Σ_g time(g)`, not `max_g time(g)` — roughly halving the achievable precompute throughput vs. the milestone's stated goal. The header pre-justifies this as a "later performance workflow", which is a defensible scoping decision but means the unit is a *parity scaffold*, not yet the *speedup* the milestone promises — and the review must grade it on the perf axis it is explicitly being graded on. With P1's concurrency lever pulled (parity-safe), the release-silent WARN fixed (W1), the redundant materializations removed (P2/P3/P4), the validation aligned to the sibling (C1/C2), and the gate made knob-coherent (CT2), this is a 9.5–10 unit.

---

## Findings

### Performance (first-class this pass)

#### P1 — [HIGH] The G devices are driven strictly sequentially; the headline multi-GPU speedup is unrealized — CONFIRMED
**Location:** `compute_f2_blocks_multigpu`, the per-device loop, .cpp:130–154 (`for (g …) partials[g] = resources.gpus[g].backend->compute_f2_blocks(...)`).

**Issue (verified against cuda_backend.cu line-by-line).** Each `CudaBackend::compute_f2_blocks` is fully self-contained and **blocking**: `guard_device()` binds its own device (cudaSetDevice, :227), it uploads the entire per-shard Q/V/N H2D (`pm = P*M_g` doubles, :321–326), runs the feeder (:328) with an interior `cudaStreamSynchronize` (:333), runs the size-grouped strided-batched GEMMs over the buckets, copies the result D2H (:391–395), and ends with `cudaStreamSynchronize(stream_)` (:397) before returning a HOST `F2BlockTensor` (:398). The orchestrator calls this G times in a plain host loop, so device `g+1`'s work is not even *issued* until device `g`'s sync has returned. The two RTX 5090s (or any G devices) run their GEMMs **serially**.

**Why it matters / doc grounding.** The CUDA Programming Guide §3.4 "Programming Systems with Multiple GPUs" states each device has its own default stream and "commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device" — i.e. the hardware can run device g and g+1 concurrently; the blocking host loop forfeits exactly this. (NOTE — correction to the first pass: it quoted "Kernel calls and asynchronous memory copying functions don't block the CPU thread … Calls to the next GPU will be executed concurrently" as verbatim §3.4 text; I could not confirm that exact sentence on the current (v13) §3.4 page, so I cite the verified per-device-default-stream wording instead. The substance is unchanged — the doc supports cross-device concurrency; the issue is real.) This is the milestone's whole point (§11.4: the precompute is the shardable phase). For a balanced even SNP split across G=2 devices, near-2× is on the table; the unit delivers ~1× plus combine overhead.

**The blocker is the seam, not the loop.** `compute_f2_blocks` returns a host tensor and blocks internally, so the orchestrator cannot overlap the devices without either (a) per-device host threads each driving one blocking call, or (b) a non-blocking seam variant (`compute_f2_blocks_async`). Option (a) is the smallest change above the seam: one thread per `g`, each running the existing blocking call into its own pre-sized `partials[g]` slot, joined before the combine. Each backend is bound to its own device, owns its own stream/handle, and the partials land in distinct vector slots ⇒ **no shared mutable state** (§2); per-device bits are fixed by the block-aligned shard regardless of concurrency.

**Concrete fix (parity-safe).** Pre-size `partials` (already done, :129), pre-build each device's sub-view and `block_id_local` (or build them inside the worker — see P4), launch G workers each invoking `compute_f2_blocks(...)` into its own slot, join all before the combine, keep the combine fixed g=0..G-1. **Capture each worker's exceptions** via a per-worker `std::exception_ptr` and rethrow the first after join (a `std::thread`/`jthread` that lets an exception escape calls `std::terminate` — verified C++ contract). See **W2**: this also needs a `Threads::Threads` link on `steppe_core`.

**Adversarial self-check.** Could threading reorder the combine? No — the combine reads `partials[g]` in fixed g order *after* the join barrier; GEMM bits are independent of wall-clock slot. Could it break the locked test? The test calls the same entry point; if `partials[g]` content is preserved (it is — distinct slots, move-assigned) the memcmp passes. Could two backends contend on a global cuBLAS/CUDA state? Each backend `guard_device()`s its own device and owns its own handle/stream; `cudaSetDevice` is per-host-thread state (each worker thread sets its own current device), so two workers on two threads do not clobber each other's current-device. Verified safe.

**Severity:** high. **Effort:** M. **Parity-safe?** yes (execution-concurrency only).

#### P5 — [LOW, can't-fix-here] Per-device Q/V/N is uploaded H2D from *pageable* host memory, so the per-shard upload cannot overlap compute — CONFIRMED
**Location:** sub-views `Qg/Vg/Ng` (.cpp:141–143) → `compute_f2_blocks`, which `cudaMemcpyAsync`s them H2D (cuda_backend.cu:321–326) from the caller's host pointers.

**Issue (doc-grounded).** The Q/V/N storage is the caller's plain host arrays (the parity test passes `std::vector<double>`). The NVIDIA "How to Overlap Data Transfers in CUDA C/C++" guide states "The host memory involved in the data transfer must be pinned memory" to overlap; a `cudaMemcpyAsync` from pageable memory does not overlap with other work. So even within one device the shard H2D does not overlap the feeder/GEMMs, and under P1's fan-out the G uploads would contend on the same pageable source without DMA overlap.

**Scoping.** This is NOT the orchestrator's bug — it only forwards `MatView`s; pinning is the producer's responsibility (the `io` streamer, §11.1). But the orchestrator is where the full Q/V/N first gets sub-viewed per device, so it is the natural place to *document* the pinned-staging expectation.

**Concrete fix.** Add a one-line contract note in the header `@param Q,V,N` that for overlapped multi-GPU ingest the caller should provide pinned host storage (§11.1), else the per-device H2D is synchronous. No code change in this unit.

**Severity:** low (not fixable in this file). **Effort:** S (doc). **Parity-safe?** yes (pinning never changes bits).

#### P2 — [LOW-MED] `device_ids` is rebuilt on every P2P call from data already pinned in `Resources` — CONFIRMED
**Location:** .cpp:179–182 (`std::vector<int> device_ids(G, 0); for … device_ids[g] = resources.gpus[g].device_id;`).

**Issue.** `device_ids[g] == resources.gpus[g].device_id`, immutable for the `Resources` lifetime (set once in build_resources, resources.cpp:90). Rebuilding this G-vector per call is a tiny heap allocation on a cold (once-per-precompute) path — negligible throughput cost but a redundant materialization of state the bundle already holds in fixed order. The P2P seam takes `std::span<const int>`, so it does not even need an owning vector.

**Concrete fix.** Best: have `Resources` expose a cached contiguous `device_ids` span built once in build_resources (the seam wants a span; the fixed order is a `Resources` invariant) and pass it straight through. The vector already constructs with `(G, 0)` and overwrites every slot, so even the local form is fine if kept — but the cache removes the allocation entirely. (Note: the `(G,0)` init then full overwrite is itself a minor redundancy — `reserve(G)`+`push_back` avoids the zero-fill; trivial.)

**Severity:** low-med. **Effort:** S. **Parity-safe?** yes (same values, same order).

#### P3 — [MED] `block_sizes` is materialized solely to feed `plan_block_shards`, duplicating data already in `ranges` — CONFIRMED
**Location:** .cpp:104–108 (`std::vector<int> block_sizes(n_block); for (b …) block_sizes[b] = ranges[b].size();`) feeding `plan_block_shards(block_sizes, ranges, G)` at :110–114.

**Issue (verified against shard_plan.cpp).** The planner is handed BOTH `ranges` and a parallel `block_sizes` derived purely from `ranges[b].size()`. shard_plan.cpp then re-asserts `block_sizes.size() == ranges.size()` (:35, throws) — a check that can only fire on the orchestrator's own bug, since the orchestrator built one from the other three lines earlier — and computes `total_snps += block_sizes[b]` (:59), which is exactly `Σ ranges[b].size()`. So `block_sizes` carries zero new information; it is a denormalization the planner does not need.

**Concrete fix (parity-safe).** Change `plan_block_shards` to derive each block size from `ranges[b].size()` internally (it already iterates `ranges` for `ranges[b].begin`/`end`), dropping the `block_sizes` parameter, the `block_sizes.size() != ranges.size()` throw, and the orchestrator's :104–108 loop. The plan is byte-identical (`ranges[b].size()` equals the placed value). **Cross-unit caveat:** the fix touches `plan_block_shards` (a separate unit); logged here because the orchestrator is the caller that creates the redundancy. If the planner signature is frozen, the in-unit fallback is `block_sizes.reserve(n_block)`.

**Severity:** med (DRY/cleanliness). **Effort:** S–M. **Parity-safe?** yes.

#### P4 — [LOW] `block_id_local` is rebuilt per device with a per-element subtract — CONFIRMED, with an unflagged underflow invariant
**Location:** .cpp:146–150 (`std::vector<int> block_id_local(M_local); for (k…) block_id_local[k] = partition.block_id[s0+k] - sh.b0;`).

**Issue.** For each device the orchestrator allocates an `M_local`-element vector and fills it from the contiguous global slice `block_id[s0..s1)` minus the constant `sh.b0`. Correct and necessary (the seam wants dense zero-based local ids), but it is an allocation + O(M_local) host pass *inside the (currently serial) device loop*. Under P1's fan-out it should move into the worker (each thread owns its slice — off the issue-critical path). Additionally: the subtract `block_id[s0+k] - sh.b0` cannot underflow because every column in `[s0, s1)` belongs to a block in `[b0, b1)` (the block-aligned shard guarantee), so `block_id[s0+k] >= sh.b0` — this invariant is documented in prose at :122–124 but is not asserted; a one-line `STEPPE_ASSERT(block_id_local[k] >= 0 && block_id_local[k] < n_block_local)` would make the contract checkable.

**Concrete fix.** Under P1, build `block_id_local` inside each worker. Add the bound assert. If P1 is deferred, acceptable as-is.

**Severity:** low. **Effort:** S. **Parity-safe?** yes (identical local ids).

#### P6 — [LOW, rejected-for-parity as a "fix"] `combine_f2_partials_host` re-zeroes then overwrites every owned slab — CONFIRMED as DELIBERATE
**Location:** the call at .cpp:201; f2_combine.cpp:97–100 (`assign(total, 0.0)`) + the `+=` loop (:110–128).

**Assessment (verified).** The combine zero-fills the full `[P×P×n_block]` f2+vpair, then `+=`'s each compact partial. For disjoint block-aligned shards this is a placement (each slab written once onto 0.0), so the f2/vpair arrays are touched twice (`2·P²·n_block` doubles). f2_combine.hpp:37–42 is explicit that the `+= onto 0.0` is *deliberate* — it is literally the §12 fixed-order sum and bit-matches the P2P on-device add (p2p_combine.cu:82–92, `acc[...] += src[k]` onto a `cudaMemset(0)` accumulator). So the redundant zero-touch is the price of provable host==P2P byte-identity. **The "skip the zero-fill / pure-placement" optimization is REJECTED-FOR-PARITY** (it would diverge the two tiers' arithmetic shape and gains only one MB-scale streaming write that is explicitly off the critical path, §11.4). Logged so a future reader does not "discover" and break the two-tier neutrality. The current code is correct.

**Severity:** low. **Effort:** N/A. **Parity-safe?** current code yes; the "fix" no-rejected-for-parity.

#### P7 — [INFO, cross-unit] The P2P combine is itself fully sequential with a per-partial full-device sync — perf debt the orchestrator inherits but does not own
**Location:** p2p_combine.cu:223–338 (the per-device combine loop, with `cudaDeviceSynchronize()` at :316 inside each peer iteration).

**Assessment.** When the orchestrator selects P2P (.cpp:173–188), the chosen transport pulls each peer partial with `cudaMemcpyPeer` and place-adds it, fully sequentially, with a *full* `cudaDeviceSynchronize()` per peer partial (:316, present for a real buffer-lifetime fence — the peer source buffer is freed at scope exit and would race the cross-device DMA). The peer pre-stage (H2D upload onto the owning device, then peer pull) is also a documented "performance wart" (p2p_combine.cu:20–25). This is real perf debt, but it is **the P2P unit's, not this orchestrator's** — the orchestrator only selects the path and hands it host partials in fixed order. Logged for cross-unit visibility so the P2P review owns it (streamed copy+compute overlap, eliding the H2D pre-stage by keeping partials device-resident, are P2P-unit changes — all parity-neutral since the transport only moves bytes). Out of scope for this file.

---

### Correctness & bugs

#### W1 — [HIGH] The §11.4-mandated tagged degrade WARN is compiled out under NDEBUG (release) — NEW, missed by the first pass
**Location:** .cpp:196–199 (`if (prefer_p2p && !can_access_peer) STEPPE_LOG_WARN("P2P combine unavailable …");`), against log.hpp:44–53.

**Issue.** `STEPPE_LOG_WARN` is `((void)0)` under `NDEBUG` (log.hpp:44 — verified) and only emits to stderr in debug builds. The architecture mandates an *explicit logged fallback* on a capability degrade (§11.4 "non-throwing tagged-degrade path … records the fallback"; config.hpp:278–283 "the backend DEGRADES — non-throwing and tagged"). On the **budget box this degrade ALWAYS fires** (`can_access_peer == false` on the 5090), and production builds are Release (NDEBUG). So in the exact build+box where the degrade is the normal case, **the user gets no diagnostic at all** — the "tagged degrade WARN" is silent precisely where it matters. The first pass praised this WARN twice (CT1: "precise and non-spammy"; the Good-patterns list) without noticing it never runs in release.

**Why it matters.** A silent fallback in release violates the §11.4 "explicit logged fallback" contract and defeats the observability intent (a user who set `prefer_p2p_combine=true` and expected the fast path has no signal it degraded). The out-of-band `last_combine_path` tag still records it (so a *test* can read it — and the parity test does), but a *running CLI* gets nothing. NOTE: this is partly an upstream limitation — `log.hpp` only realizes a debug-removed WARN today (the spdlog `STEPPE_LOG_INFO/ERROR` levels "land with the logging milestone"). So the cleanest fix may be milestone-gated.

**Concrete fix.** Two options: (a) once the logging milestone lands a release-surviving level, route the degrade through it (it is a runtime capability event, not a teardown WARN — it should survive NDEBUG). (b) Interim: at least record that the only release-visible signal is the `last_combine_path` tag, and document in the header that callers must read `resources.last_combine_path` to detect a degrade in release (since the WARN is debug-only). Do NOT silently rely on a no-op macro for an architecture-mandated diagnostic.

**Severity:** high (architecture-contract gap on the always-degrading box; observability). **Effort:** S (interim doc) / M (proper release-surviving log level, milestone-coupled). **Parity-safe?** yes (logging never touches the numeric path).

#### C1 — [LOW] The `s0 < 0 ? 0 : s0` / `M < 0 ? 0 : M` / `n_block < 0 ? 0 : n_block` clamps encode impossible states and are internally inconsistent — CONFIRMED
**Location:** `col_off` (.cpp:140, `s0 < 0 ? 0 : s0`); `block_id_local` size (:146, `M_local < 0 ? 0 : M_local`); `block_ranges` span length (:101, `M < 0 ? 0 : M`); `block_sizes` size (:104, `n_block < 0 ? 0 : n_block`).

**Issue.** `DeviceShard` guarantees `s0 <= s1`, `b0 <= b1` for a well-formed plan; `block_ranges` produces non-negative `begin`/`end`; `STEPPE_ASSERT(Q.M >= 0)` (:70) and the build_resources/plan_block_shards invariants make `s0`, `M_local`, `M`, `n_block` all `>= 0` on any reachable path. The `?:` clamps therefore never trigger — and they are inconsistent: `MatView Qg{Q.data + col_off, P, M_local}` (:141) carries the **unclamped** `M_local`, so the clamp on `col_off` (:140) and on the vector size (:146) guard against a negative `M_local` that the `MatView` extent itself does not. If `M_local` could ever be negative this code is already wrong at :141 regardless of the clamps; if it cannot (the real case), the clamps are dead and add casting noise (see T-cast). This is "fail-soft on impossible state" where §2 wants fail-fast.

**Concrete fix.** Drop the `?:` clamps; add one up-front `STEPPE_ASSERT` that the plan is well-formed (`sh.s0 <= sh.s1 && sh.b0 <= sh.b1` per shard, mirroring `DeviceShard::empty()`'s invariant) — matching the sibling's debug-assert convention. Then `col_off = (size_t)P * (size_t)s0`, `block_id_local(M_local)`, `block_ranges(span(block_id.data(), M), …)` with no ternaries.

**Severity:** low. **Effort:** S. **Parity-safe?** yes (well-formed inputs ⇒ identical results; clamps were unreachable).

#### C2 — [LOW-MED] No fail-fast that `partition.n_block` is consistent with `block_id` before forwarding; diverges from the sibling orchestrator — CONFIRMED
**Location:** the validation block .cpp:66–73 vs. `f2_from_blocks.cpp:103–121` (`validate_partition`).

**Issue (verified against the sibling).** `f2_from_blocks::compute_f2_blocks` runs `validate_partition` which debug-asserts `M<=0 || n_block > 0` (:114), `M<=0 || n_block <= M` (:116), and an O(M) dense/non-decreasing scan (:118). `compute_f2_blocks_multigpu` only debug-asserts `block_id.size() == M` (:72) and then relies on `core::block_ranges` to *throw* on a bad partition (block_partition_rule.hpp:221 throws on `block_id.size() < M`; :239/:244 throw on out-of-range/non-decreasing). That is functionally safe — the throw is real even under NDEBUG. But the two sibling entry points have **divergent fail-fast behaviour for the same contract** (§8 DRY drift): the single-GPU path attributes a malformed partition to its own input with a file/line debug-assert at the orchestration boundary; the multi-GPU path defers to a downstream `runtime_error` from inside `block_ranges`. Crucially, the `n_block > 0` / `n_block <= M` sanity the sibling asserts is **not covered downstream**: a `partition.n_block` *larger* than the true block count (but `block_id` still valid and in range) would build an over-long `ranges`/`block_sizes` and the planner would emit trailing empty shards — silently wrong block count rather than an asserted breach. (`block_ranges` only validates `id < n_block`, not `n_block <= max(id)+1`.)

**Concrete fix.** Promote the sibling's `validate_partition` (currently in `f2_from_blocks.cpp`'s anonymous namespace) to a shared `core/internal` helper consumed by both orchestrators — closing the §8 DRY gap. At minimum add `STEPPE_ASSERT(M <= 0 || partition.n_block > 0)` and `STEPPE_ASSERT(M <= 0 || (long)partition.n_block <= M)` here.

**Severity:** low-med. **Effort:** S (asserts) / M (shared helper). **Parity-safe?** yes (validation only).

#### C3 — [INFO] `G < 1` throw on an unsigned `device_count()` — unreachable via build_resources, correctly kept as defense-in-depth
**Location:** .cpp:76–81.

**Assessment — not a defect.** `device_count()` returns `std::size_t` (resources.hpp:113), so `G < 1` is exactly `G == 0`. `build_resources` guarantees `gpus.size() >= 1` (resources.cpp:73–77), so this is unreachable for a canonically-built `Resources`. But `Resources` is an aggregate a caller could default-construct with empty `gpus`, so the throw is a legitimate entry-boundary fail-fast (§2). Keep it. Minor readability nit: on an unsigned type `G == 0` reads clearer than `G < 1` (the `< 1` invites a "can it be negative?" double-take that the unsigned type forecloses). The throw message duplicates build_resources's own guard text — acceptable (two call sites).

---

### Edge cases & failure modes

#### E1 — [LOW] Empty-partition (`n_block == 0`, `M == 0`) takes the full `G >= 2` planning path and does G no-op device round-trips — CONFIRMED correct, but wasteful
**Location:** whole `G >= 2` branch with `n_block == 0`.

**Assessment (traced).** With `M == 0`/`n_block == 0`, `G >= 2`: `block_ranges` returns `{}` (block_partition_rule.hpp:209–210 early-out), `block_sizes` empty, `plan_block_shards` returns G empty `{0,0,0,0}` shards (shard_plan.cpp:48–50), every device gets `M_local==0`/`n_block_local==0` and the backend early-returns an empty `F2BlockTensor` (cuda_backend.cu:239), and `combine_f2_partials_host` produces an empty tensor (validation passes: `covered == 0 == n_block_full`). **Correct.** But it still constructs G backends' worth of work (G blocking `compute_f2_blocks` calls that each early-return after a `guard_device()` cudaSetDevice) plus the full planning machinery for a no-op. A cheap `if (n_block == 0) return {};` (or an explicit empty `F2BlockTensor` with `P` set) right after the `G == 1` check would short-circuit without G device round-trips.

**Severity:** low. **Effort:** S. **Parity-safe?** yes (empty in ⇒ empty out on every path).

#### E2 — [INFO] `n_block < G` (more devices than blocks) — trailing devices get empty shards; correct, but untested at this entry point
**Assessment — correct.** `plan_block_shards` produces trailing empty shards (shard_plan.cpp:105–115 + the value-init `{0,0,0,0}` tail), the orchestrator hands those devices `M_local==0`, the backend early-returns, the combine places nothing (`part.n_block==0`, the `+=` loop runs zero iterations). No defect. But the locked parity test only exercises `n_block >> G` (see T2).

#### E3 — [INFO] Width chain `int`/`long`/`size_t` is correct but noisy — CONFIRMED
**Assessment.** `b0/b1` are `int` (DeviceShard), `n_block` fits `int` per the domain (O(1e3)), so `n_block_local = sh.b1 - sh.b0` (:135) is safe. `col_off` correctly promotes to `size_t` before the `P * s0` multiply (:139–140), matching MatView's "promote to long before multiplying by P" rule (views.hpp:50, 66–67). `M_local = s1 - s0` is `long` (DeviceShard `s0/s1` are `long`). Values are all correct; the *mix* of `int` (P, n_block_local), `long` (M, s0, s1, M_local), `size_t` (col_off) within ~15 lines is the readability noise called out in T-cast.

---

### Numerical / precision (§12)

#### N1 — [INFO] The orchestrator forwards `precision` verbatim and performs no arithmetic — exactly correct for §12
**Assessment — no defect, justified.** Zero floating-point math here: it shards, dispatches, and the combine sums onto exact zeros (`x + 0.0`, in f2_combine.cpp / p2p_combine.cu, not in this file). `precision` is passed unchanged to every `compute_f2_blocks` (:90 for G==1, :153 for G>=2) and never inspected. This is exactly §12/§11.4 — the orchestrator is parity-neutral; the per-device GEMM precision is the backend's lever. **Good.**

#### N2 — [INFO] Fixed g=0..G-1 combine order preserved on both forks; the gate is parity-neutral — CONFIRMED
**Assessment — no defect.** Both `combine_f2_partials_host` (:201) and `combine_f2_partials_p2p` (:184) receive `partials_span`/`shards_span` in the same g-indexed order; both sum in fixed g=0..G-1 onto a zero-init full tensor (f2_combine.cpp:110, p2p_combine.cu:223 — verified identical arithmetic). The `use_p2p` gate (:171–172) selects transport only. The `last_combine_path` tag is set on `Resources` (:183, :200), never on the tensor — matching §12 / cleanup §(2).2. **Good — this is the §12 law, held.**

---

### CUDA idioms / RAII / async semantics (§7)

#### CU1 — [INFO] No CUDA resource owned here; all RAII lives below the seam — correct by design
**Assessment — no defect.** The orchestrator owns no `DeviceBuffer`/`Stream`/handle; those are PRIVATE to `steppe_device` inside `CudaBackend`. The only §7 lever relevant here is P1 (driving the per-device streams concurrently from the host) — achievable by owning *threads*, not streams. Recorded under Performance.

#### CU2 — [LOW, RE-SCOPED — first-pass premise was fabricated] No observability range on the multi-GPU phases; AND the NVTX facade the fix needs does not exist
**Location:** whole function.

**Issue + correction.** §10 mandates NVTX phase ranges so the SPMG overlap (or, today, the serialization) is visible on the Nsight timeline. The orchestrator wraps G `compute_f2_blocks` calls plus a combine with no enclosing phase scope, so profiling the SPMG path shows G unlabeled back-to-back ranges and an invisible combine — a real gap, especially since P1's serialization is the headline finding and you'd want Nsight to *show* it. **BUT the first pass's concrete fix is wrong:** it says "`STEPPE_NVTX_SCOPE` is in `core/internal/nvtx.hpp` (host-pure)". I grepped the whole tree (`grep -rn 'NVTX\|nvtx' src include`) — **there is no `nvtx.hpp` and no `STEPPE_NVTX_SCOPE` macro anywhere.** So the fix is not "add an include"; it is "build a CUDA-free NVTX facade in `core/internal/` first (host-pure, no-op when disabled, the §8/§10 pattern), THEN wrap the shard-compute and combine phases here." That is a larger, cross-cutting effort, and arguably belongs to a profiling/observability workflow, not this unit.

**Concrete fix.** When the NVTX facade exists, wrap `G >= 2` shard+compute in a `f2_shard_compute` range and the combine in a `f2_combine` range. Until then, this is a documented gap, not an in-unit one-liner.

**Severity:** low. **Effort:** M (facade must be built first — NOT S as the first pass claimed). **Parity-safe?** yes (no-op on the numeric path).

---

### Magic numbers & hardcoded values (§4)

#### M1 — [INFO] No magic numbers — CONFIRMED
**Assessment — no defect.** The only literals are `0` (init), `1` (the `G < 1` device floor — a true minimum, not a tunable), and the `?:` clamp sentinels (C1 recommends removing). No block-size/fraction/bucket-base appears here (those are correctly the backend/config's). Clean per §4.

---

### Decomposition / single-responsibility / function size (§2)

#### D1 — [MED] One ~150-line function doing six jobs; would read better as a short composition — CONFIRMED
**Location:** the whole body (.cpp:50–202).

**Issue.** The function does: (1) Q/V/N+partition validation (:66–73); (2) device-count fail-fast (:76–81); (3) G==1 fast-path (:88–91); (4) shard planning (:100–114); (5) per-device sub-view compute (:129–154); (6) capability-gated combine (:168–201). Each is a coherent unit with its own doc paragraph. The sibling `f2_from_blocks.cpp` keeps validation in named free functions; this inlines everything. At ~150 lines / six responsibilities it is past where extraction aids readability and (critically) testability.

**Concrete fix.** Extract `validate_inputs(resources, Q, V, N, partition)` (1–2), `compute_partials(resources, Q, V, N, ranges, shards, precision) -> vector<F2BlockTensor>` (5 — the natural home for P1's fan-out), and `select_and_combine(resources, partials, shards, P, n_block) -> F2BlockTensor` (6 — the §4 gate). Top-level then reads as a ~15-line composition. This makes job 5 (the perf loop) and the gate predicate independently host-testable against a fake backend (T1).

**Severity:** med. **Effort:** M. **Parity-safe?** yes (pure refactor; identical call sequence).

---

### Readability, naming, const-correctness, attributes (§7/§9)

#### R1 — [LOW-MED] `[[nodiscard]]` on a function that also mutates `Resources::last_combine_path` — the tag write is an undocumented side effect — CONFIRMED
**Location:** decl `[[nodiscard]] … compute_f2_blocks_multigpu(steppe::device::Resources& resources, …)` (hpp:66); the writes `resources.last_combine_path = …` at .cpp:183, :200.

**Issue.** The signature reads as a "give me the tensor" function (`[[nodiscard]]` return) but it also *writes* `resources.last_combine_path`. The header `@param resources` (hpp:53–56) documents one reason `resources` is non-const ("driving each backend->compute_f2_blocks mutates the backend's device-side scratch") — but NOT the second (the tag write). A reader auditing "what does this mutate on `Resources`" finds the tag write undocumented at the signature. The tag-on-Resources design is an accepted §(2).2 decision (out-of-band, off the tensor — correct), but the const-correctness story is muddied.

**Concrete fix.** Expand `@param resources` to explicitly list `last_combine_path` as a written field ("MUTATED: records which combine transport this run used in `resources.last_combine_path`"). A result-envelope alternative (`struct { F2BlockTensor tensor; CombinePath path; }`) is a larger API change; doc-ing it is the pragmatic fix.

**Severity:** low-med. **Effort:** S (doc). **Parity-safe?** yes.

#### R2 — [TRIVIAL] `const std::span<const F2BlockTensor>` outer-const is redundant — CONFIRMED, ignorable
**Location:** .cpp:168–169.

**Assessment.** `partials_span`/`shards_span` are built unconditionally and used in both branches (fine — both need them). The outer `const` on a local span is redundant (a span's view is already immutable; the inner `const T` is the meaningful one). Harmless; not worth churn except as part of D1's extraction.

#### R3 — [LOW] `@throws` is documented only in the .cpp prose, not the header contract — CONFIRMED
**Assessment.** The function legitimately throws (`runtime_error` on `G<1` (:78) and on a malformed partition via `block_ranges`; `CudaError` from a backend or the P2P DMA) and correctly is not `noexcept`. The degrade-vs-throw split for P2P is in the .cpp combine comment (:156–199) but not in the header `@return`/`@throws`. Add a `@throws` clause to the header (`runtime_error` on no-device/malformed-partition; `CudaError` on a device fault; **never throws to degrade P2P** — the degrade is non-throwing).

**Severity:** low. **Effort:** S. **Parity-safe?** yes.

#### T-cast — [LOW] Casting noise: `static_cast` chains and `?:` clamps cluster within the per-device loop — CONFIRMED
**Location:** :100–101, :104, :106–107, :139–140, :146, :148–149 (the `static_cast<std::size_t>(… < 0 ? 0 : …)` family).

**Issue.** Removing the C1 clamps eliminates the bulk of the ternary-inside-cast noise. What remains (`static_cast<std::size_t>(P) * static_cast<std::size_t>(s0)`, `static_cast<std::size_t>(k)`) is legitimate width-promotion that the MatView contract requires. After C1 the casts read as plain promotions rather than "promote a possibly-negative-clamped value", which is the readability win. No width *bug* exists (E3); this is purely noise reduction coupled to C1.

**Severity:** low. **Effort:** S (falls out of C1). **Parity-safe?** yes.

---

### Layering / API / ABI (§4)

#### L1 — [INFO] CUDA-free split is clean and VERIFIED; the combine policy is in the correct layer — CONFIRMED, the unit's best quality
**Assessment — no defect.** Re-verified by reading every included header: `core/fstats/f2_combine.hpp` (core, host), `device/p2p_combine.hpp` (CUDA-free decl — includes only `<span>`, `steppe/fstats.hpp`, `device/shard_plan.hpp`), `device/resources.hpp`/`shard_plan.hpp` (both "CUDA-FREE BY CONTRACT", no `<cuda_runtime.h>`), `core/domain/block_partition_rule.hpp` (host-pure). No CUDA header, no cuBLAS. The combine *policy* (the `use_p2p` gate) is host-side intent over `Resources` and rightly lives in this `core` orchestrator; the *transports* are below the seam (one in `core`, one as a CUDA-free decl into a `.cu`). Dependency direction `core -> device(decl)` respected; CMake confirms `steppe::device` is linked PRIVATE. **This is exactly the §4 design.**

#### L2 — [LOW] The three `device/` includes are CUDA-free decls — worth a one-line header note
**Assessment.** `f2_blocks_multigpu.cpp` includes `device/resources.hpp`, `device/p2p_combine.hpp`, `device/shard_plan.hpp` — all CUDA-free-by-contract device-layer headers, so no §4 violation (core may name CUDA-free device decls — the `ComputeBackend` seam pattern). The unit's header could note that its `device/` includes are the CUDA-free decls (the way `f2_from_blocks.cpp` notes "includes only the CUDA-free seam"). Cleanliness only.

---

### Testability (§13)

#### T1 — [MED] The orchestrator's host-pure logic is only exercised end-to-end on a GPU (the parity `.cu`); none of it is unit-tested GPU-free — CONFIRMED
**Location:** the entire unit; the only test is `tests/reference/test_f2_multigpu_parity.cu` (GPU-required).

**Issue (verified against the test).** §13/§2: host-pure logic should be CPU-unit-testable with no GPU. The orchestrator's host logic — the G==1 forward, the shard→sub-view→local-id transform, the `use_p2p` gate selection, the `device_ids` build, the validation fail-fasts — depends on the backend ONLY through the `ComputeBackend` interface, and there is a `CpuBackend` precisely so the pipeline is GPU-free-testable (§8). Yet there is **no host-only unit test**: the entry point is only run through the CUDA backend in a `.cu`. The parity test (read in full) exercises only `prefer_p2p_combine` true/false on balanced real AADR; the gate logic, empty-shard handling, `n_block < G`, and the local-id transform are unverified except as a side effect.

**Why it matters.** On the budget box (no peer access) the P2P branch is never taken, so a regression in the gate (e.g. flipping `prefer_p2p && can_access_peer`) is invisible there. The host logic is the part most likely to drift and the cheapest to test.

**Concrete fix.** Add `tests/unit/test_f2_blocks_multigpu.cpp` (host GoogleTest) with a *fake* `ComputeBackend` recording its `(Q,V,N,block_id,n_block)` and returning a synthetic compact partial. Assert: (a) G==1 forwards the FULL partition unchanged; (b) G==2 hands device g exactly its shard's columns + dense zero-based local ids; (c) the combine reassembles the synthetic partials; (d) the `use_p2p` *predicate* picks P2P iff `prefer_p2p_combine && can_access_peer` (test the predicate, not the `.cu` transport — extract it per D1); (e) empty-shard and `n_block < G` correct placements; (f) under P1, race-freedom under TSan. Needs D1's extraction + a fake backend; both cheap.

**Severity:** med. **Effort:** M (enabled by D1). **Parity-safe?** yes (tests only).

#### T2 — [LOW] The locked GPU parity test does not cover `n_block < G`, single-block, or empty-partition — CONFIRMED
**Assessment (verified against the test).** The memcmp gate proves the balanced common case across precisions and both combine tiers, but the degenerate geometries (E1/E2 — `n_block` ∈ {0,1}, `n_block < G`) are not asserted bit-identical. These are exactly the corners where empty-shard early-return and trailing-empty-shard placement could regress unnoticed. Cover them in the T1 host test (cheaper, no GPU — the shard/combine logic is host-pure; only the per-device GEMM bits need the GPU and those are covered by the balanced case), or add a tiny synthetic GPU fixture.

**Severity:** low. **Effort:** S. **Parity-safe?** yes.

#### T3 — [INFO] D1 is the enabler for T1 — a coupled pair
**Assessment.** T1 is blocked on D1 (extracting `compute_partials` / the gate predicate) because the current monolith exposes only the GPU-coupled whole. Treat D1+T1 as one work item. Highest-leverage cleanliness+testability change after P1.

---

### Capability-tier coherence (§11.4)

#### CT1 — [PARTIALLY-CONFIRMED, but see W1] The probe-once / tagged-degrade / which-path-recording design — the tag is off the numeric payload (correct); the WARN half is release-silent (defect, W1)
**Assessment.** The gate reads the *probed* `gpus[0].caps.can_access_peer` (set once in build_resources, never re-probed here — correct), ANDs it with the *intent* `config.prefer_p2p_combine`, and records the outcome in `resources.last_combine_path` (out-of-band, never on the tensor — correct, §12/cleanup §(2).2). The genuine-degrade branch fires only when the user preferred P2P but the device cannot peer (:196), not on a deliberate baseline choice — correct §11.4 intent. **BUT** the diagnostic it emits is compiled out in release (W1), so the "explicit logged fallback" is only half-realized. The first pass marked CT1 a clean "no defect"; my pass splits it: the *tag* discipline is correct and exemplary; the *logged* half is a release-silent gap (tracked as W1). The tag (`CombinePath`, off the tensor) is the part to keep.

#### CT2 — [LOW-MED] The gate checks `prefer_p2p_combine && can_access_peer` but ignores `enable_peer_access`; the two documented-distinct knobs are half-enforced — CONFIRMED
**Location:** .cpp:171–172 (`use_p2p = resources.config.prefer_p2p_combine && resources.gpus[0].caps.can_access_peer`).

**Issue (verified against config.hpp AND p2p_combine.cu).** config.hpp carefully distinguishes two knobs (the OVERRIDE-KNOB banner, :229–260): `enable_peer_access` = "the MAY-WE knob: whether the backend is permitted to call cudaDeviceEnablePeerAccess at all" (:255–260), and `prefer_p2p_combine` = "the WHICH-PATH knob … once peer access IS available, prefer the device-resident combine" (:262–287). The orchestrator's gate consults `prefer_p2p_combine` and the probe `can_access_peer`, but **NOT** `enable_peer_access`. And the P2P transport **does** call `cudaDeviceEnablePeerAccess(owning_device, 0)` (p2p_combine.cu:265 — confirmed). So a user who sets `prefer_p2p_combine=true` and `enable_peer_access=false` (coherent: "I prefer P2P logically but forbid enabling peer access") will still select the P2P path, which then calls `cudaDeviceEnablePeerAccess` — directly contradicting the knob the user set to false. The two-knob design is only half-enforced here; the stricter knob is silently ignored.

**Concrete fix.** Add `&& resources.config.enable_peer_access` to the `use_p2p` gate, and update the genuine-degrade WARN condition correspondingly (a user who set `enable_peer_access=false` is a deliberate baseline choice, like `prefer_p2p_combine=false` — no WARN). OR explicitly document that `prefer_p2p_combine` supersedes `enable_peer_access` and reconcile both config docs. As written they are inconsistent.

**Severity:** low-med (config-contract coherence; unusual combination but a real inconsistency). **Effort:** S. **Parity-safe?** yes (both transports are bit-identical; this only changes which transport runs — the parity test's `prefer_p2p_combine=false` arm already proves host-staged == P2P).

---

## Considered & rejected (incl. rejected-for-parity)

- **Skip the zero-fill in `combine_f2_partials_host` / do a pure placement (disjoint shards)** — REJECTED-FOR-PARITY. The `+= onto 0.0` is the deliberate arithmetic that makes the host path bit-identical to the on-device P2P add (f2_combine.hpp:37–42; p2p_combine.cu:82–92; locked-test claim (4)). Eliding the zero-touch saves one MB-scale streaming write off the critical path (§11.4) and risks diverging host from P2P. Forbidden. (Logged as P6.)

- **Move the host combine on-device via a fused reduction kernel** — REJECTED (wrong layer + pointless). The host combine is in `core` and must stay CUDA-free (§4); the P2P tier already *is* the on-device combine. The combine is kB–MB, off the critical path.

- **NCCL AllReduce / a tree-reduce for the combine to overlap it with compute** — REJECTED-FOR-PARITY. §12 is categorical: the parity reduction is the fixed g=0..G-1 host/device order, never AllReduce (its order varies with G). The orchestrator correctly never does this.

- **Make `compute_f2_blocks` return device-resident partials and combine on GPU 0 without the D2H round-trip (even on the host path)** — REJECTED (seam + scope, NOT parity). The host-staged path D2H's each partial inside `compute_f2_blocks`, then the combine re-handles host memory; a device-resident pipeline would elide that — but that IS the P2P fast-path's job. On a no-peer box (the 5090) you cannot keep partial g on device 0 from device g without a host bounce anyway, so the D2H is inherent to the *portable* baseline, not removable waste. Byte-identical if done, but out-of-scope and already-solved-by-P2P-where-possible. (P2P's own H2D pre-stage wart is P7.)

- **Replace the G==1 fast-path with "run the G>=2 path with one shard"** — REJECTED (the explicit structural-invariance choice). The G==1 early-return (:88–91) makes single-GPU invariance *structural* — zero shard/combine code on the value path, bit-for-bit the existing result. `plan_block_shards` would handle G==1 (one full range), but routing it through the combine adds a no-op placement + an extra tensor copy for zero benefit. Keep the fast-path. (A *good pattern*, below.)

- **Cache `block_id_local` on `Resources`/the plan permanently** — REJECTED. `block_id_local` is shard-AND-partition-specific and transient; caching it on `Resources` would be state that depends on the partition, which `Resources` does not (and should not) know. Keep it local (built in the worker under P1). `device_ids` (P2) is different — it is partition-independent per-device identity and DOES belong cached on `Resources`.

- **Re-probe `can_access_peer` symmetric (both directions) before P2P** — REJECTED (the probe is gpus[0]-only by design; §(2).1 "one probe, owned at construction"). The orchestrator trusts the probe; the P2P transport re-confirms by failing-fast on the DMA (p2p_combine.cu STEPPE_CUDA_CHECK on cudaMemcpyPeer). Adding a re-probe here duplicates the capability probe. Correct as-is.

- **[First-pass CU2 fix] "wrap in STEPPE_NVTX_SCOPE from core/internal/nvtx.hpp"** — REJECTED-AS-STATED (fabricated premise). There is no `nvtx.hpp` / `STEPPE_NVTX_SCOPE` in the tree. The observability *gap* is real (re-scoped in CU2), but the fix is "build the facade first", not "add an include" — and it likely belongs to a profiling workflow, not this unit.

- **[First-pass implied] route the degrade WARN through `STEPPE_LOG_WARN` and consider it logged** — REJECTED as a *complete* fix. `STEPPE_LOG_WARN` is a NDEBUG no-op (W1); on the always-degrading budget box in release it logs nothing. The proper fix is a release-surviving log level (milestone-coupled) or, interim, document that the `last_combine_path` tag is the only release-visible signal.

---

## What it takes to reach 10/10

**Performance (the headline gap):**
1. **P1 — fan out the G per-device `compute_f2_blocks` calls across G host threads, join before the combine.** The single change that turns the unit from a parity scaffold into the milestone's speedup. Parity-safe (execution-concurrency only). Capture worker exceptions via `exception_ptr` and rethrow the first on join. Requires **W2** (link `Threads::Threads` on `steppe_core`). (Effort M.)
2. **P5 — document the pinned-host-memory expectation** for overlapped ingest in the header `@param` (async H2D is synchronous from pageable memory; §11.1). (Effort S, doc only.)
3. **P2/P3/P4 — remove the redundant materializations:** cache `device_ids` on `Resources` (P2); drop the `block_sizes` projection by deriving it inside `plan_block_shards` from `ranges` (P3, planner-side); build `block_id_local` inside the worker (P4, falls out of P1). (Effort S–M.)
4. **CU2 — observability:** when an NVTX facade exists (it does NOT today), add `f2_shard_compute` / `f2_combine` phase ranges so the serialization-vs-overlap is visible on Nsight (§11.3). Lower priority precisely because the facade must be built first. (Effort M.)

**Correctness & contract:**
5. **W1 — make the §11.4 degrade diagnostic survive release.** Route the "P2P unavailable -> host-staged" degrade through a release-surviving log level (when the logging milestone lands), or interim-document that `last_combine_path` is the only release signal. Do not rely on a NDEBUG no-op for an architecture-mandated fallback log. (Effort S interim / M proper.)
6. **CT2 — fold `enable_peer_access` into the `use_p2p` gate** (or document that `prefer_p2p_combine` supersedes it) so the two documented-distinct knobs are coherent and the P2P transport never enables peer access against `enable_peer_access=false`. (Effort S.)
7. **C1 — delete the `?:` clamps,** replace with one well-formed-plan `STEPPE_ASSERT`; the `MatView` extent and the buffer sizes then agree, and the casting noise (T-cast) collapses to plain promotions. (Effort S.)
8. **C2 — close the §8 DRY gap with the sibling:** promote `f2_from_blocks.cpp`'s `validate_partition` (the `n_block` sanity + dense scan) to a shared `core/internal` helper consumed by both entry points. (Effort S–M.)
9. **R1/R3/L2 — complete the header contract:** document that `last_combine_path` is mutated; add `@throws`; note the three `device/` includes are CUDA-free decls. (Effort S.)
10. **E1 — add an `n_block == 0` early-out** so the degenerate case skips G device round-trips. (Effort S.)

**Cleanliness & testability:**
11. **D1 — decompose into `validate_inputs` / `compute_partials` / `select_and_combine`** so the top-level reads as a ~15-line composition and the perf loop + the gate predicate are independently testable. (Effort M.)
12. **T1+T3 — add a host-only `tests/unit/test_f2_blocks_multigpu.cpp`** with a fake `ComputeBackend`: G==1 forward, per-device sub-view + local-id correctness, gate predicate, empty/`n_block<G` placements, race-freedom under TSan. (Effort M, enabled by D1.)
13. **T2 — cover the degenerate shard geometries** (`n_block` ∈ {0,1}, `n_block < G`) in the host test (no GPU needed for the shard/combine logic). (Effort S.)

Meeting **1, 5, 6, 8, 11, 12** (the perf lever + the release-silent-degrade fix + the knob-coherence fix + the DRY close + the decomposition that unlocks the host test) is what moves this from a correct-but-serial scaffold to a 10/10 multi-GPU orchestrator.

---

## Good patterns to keep

- **The CUDA-free layering is exemplary (L1), and VERIFIED.** The orchestrator reaches the GPU only through the `ComputeBackend` seam and two CUDA-free combine decls; it includes no CUDA header, owns no device memory, issues no GEMM. Re-confirmed by reading every included header and the CMake PRIVATE link. This is the §4 design realized.

- **The G==1 structural fast-path (:88–91).** Forwarding verbatim to the lone backend makes single-GPU invariance *structural* (zero shard/combine code on the value path, bit-for-bit the existing result) rather than a property to re-prove. Keep it as a guarded early-return.

- **The fixed g=0..G-1 combine order, preserved identically on both forks (N2),** with the transport selected by a parity-neutral gate. The order is the §12 law and the unit never lets the transport touch it. Confirmed identical arithmetic in `f2_combine.cpp` and `p2p_combine.cu`.

- **The out-of-band `last_combine_path` TAG — recorded on `Resources`, never on the numeric `F2BlockTensor` (CT1).** This keeps the parity diff comparing only math-produced bits, exactly the §12 / cleanup §(2).2 discipline. (Note: the *tag* is the part to keep; the *WARN* alongside it is release-silent — fix W1.)

- **Block-aligned whole-block sharding via the single-source `block_ranges` inverse + `plan_block_shards` (:100–114).** The orchestrator never re-derives the block→column mapping; it consumes the one home (§8), which is the structural reason the parity holds. It leans on it rather than reimplementing the GEMM (design §0).

- **Dense doc comments that state the parity argument inline** (header :30–50; .cpp :13–29). The "why this is bit-identical" reasoning lives where the code lives, so a future editor sees the parity contract before touching the loop. (R1/R3 only ask for two missing clauses.)
