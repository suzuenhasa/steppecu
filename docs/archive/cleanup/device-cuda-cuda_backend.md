# Review — `src/device/cuda/cuda_backend.cu` (unit: device-cuda-cuda_backend)

Reviewer: senior CUDA/C++ review pass against `docs/architecture.md` (§2, §4, §7, §8, §9, §10, §11, §12, §13), `docs/ROADMAP.md` (§0, §4, §5, §6), `docs/TODO.md` (the `wn01sl1wz` cleanup backlog + the re-verified CAPABILITY-TIER section).
Scope: the GPU `ComputeBackend` implementation — `CudaBackend` with `compute_f2`, `compute_f2_blocks` (the M4 per-block deliverable on branch `m4-perblock-f2`), `decode_af`, and the `make_cuda_backend()` factory. Read fully alongside `backend.hpp`, `device_buffer.cuh`, `handles.hpp`, `stream.hpp`, `check.cuh`, `f2_block_kernel.{cuh,cu}`, `f2_blocks_kernel.{cuh,cu}`, `decode_af_kernel.cuh`, `config.hpp`, `fstats.hpp`, `views.hpp`, and `cpu_backend.cpp`.

Method note: every CUDA-behavior claim (cuBLAS workspace/stream semantics, `cudaMemcpyAsync` synchronicity, grid-dimension limits, `cublasGemmEx` integer widths) is checked against the official NVIDIA docs and cited inline. Each finding is adversarially re-questioned; dismissed candidates are in "Considered & rejected."

---

## Role & layering

This TU is the only place a host caller meets the GPU f2/decode path. It implements the **CUDA-free** `ComputeBackend` seam (`backend.hpp`), and CUDA is `PRIVATE` to `steppe_device` (arch §4). The layering is correct: it includes only device-private CUDA headers (`check.cuh`, `device_buffer.cuh`, `handles.hpp`, the three kernel `.cuh`s) plus CUDA-free public headers (`config.hpp`, `fstats.hpp`, `backend.hpp`, `views.hpp`). It owns no raw `cudaMalloc`/`cublasCreate` — everything is `DeviceBuffer<T>` / `CublasHandle` (arch §2, §7; this TU is correctly **not** on the allocation allowlist). It exposes the concrete type only through `make_cuda_backend()` returning `std::unique_ptr<ComputeBackend>` (arch §9). The orchestration-vs-kernel split is honored: no `<<<>>>` here, only `launch_*` / `run_*` wrappers (arch §7).

The *skeleton* is right and the discipline is high. The problems concentrate in: (a) a load-bearing determinism bug where the §12 cuBLAS workspace is silently discarded; (b) two latent launch-failure modes from y/z grid axes exceeding 65535; (c) the default-stream / device-id resource debt the TODO already flags as the M4.5-blocking theme; (d) `compute_f2_blocks` being a ~170-line monolith mixing host int-math, VRAM budgeting, and device orchestration; (e) missing edge-case guards; and (f) a missing capability-tier observability hook. None break the *happy-path numerics* (M4 is green vs the CPU oracle on real AADR), but several are exactly the latent correctness/determinism hazards the §12 contract is meant to forbid.

---

## Score: 7/10 — structurally exemplary, but a determinism-voiding workspace bug, two 65 535-axis launch traps, a host-synchronous default stream, and a monolithic M4 method keep it well short of the 9.5–10 bar

The RAII/DRY/narrow-wrapper discipline is genuinely strong. But F1 (silent §12 determinism defeat) and F2/F3 (latent grid-dimension launch failures at production SNP counts) are *correctness* class against the project's own laws, not style nits; and `compute_f2_blocks` violates §2 single-responsibility badly enough that the TODO already lists it as a pre-M4.5 calcification risk. A senior reviewer cannot sign off at 9+ with a silently-defeated `cublasSetWorkspace` on the bit-stable path and an unclamped y-axis that fails at ~1M SNPs.

---

## Findings

### (1) Correctness & bugs

**[F1 — HIGH, before-M4.5: YES] `cublasSetStream` inside the GEMM wrappers unconditionally discards the ctor's `cublasSetWorkspace`, voiding §12 emulated-FP64 determinism.**
Location: ctor (lines 48–53) binds the workspace; the defeat happens downstream in `f2_block_kernel.cu:242` (`run_f2_gemms`) and `f2_blocks_kernel.cu:190` (`run_f2_gemms_group`), both reached from `compute_f2` (line 86) and `compute_f2_blocks` (line 263).
The ctor does `CUBLAS_CHECK(cublasSetWorkspace(blas_.get(), workspace_.data(), workspace_.bytes()))` — exactly the §12 requirement that "fixed-point emulation voids the run-to-run bit-wise guarantee unless an adequate workspace is supplied" (config.hpp `kCublasWorkspaceBytes`: "an explicit workspace is REQUIRED for run-to-run reproducibility of emulated FP64"). But the cuBLAS docs state `cublasSetStream()` **"unconditionally resets the cuBLAS library workspace back to the default workspace pool"** ([cuBLAS, cublasSetStream](https://docs.nvidia.com/cuda/cublas/index.html)). Both GEMM wrappers call `cublasSetStream(handle, stream)` on entry, so the *first* f2 call throws away the 64 MiB `workspace_` and reverts to the default pool. Worse: since `stream_` is currently `nullptr`, the workspace is never re-bound after the reset — so the bit-stable guarantee the project advertises is **not in force in any run today**.
Why it matters: the entire reason `workspace_` exists is defeated (arch §12; ROADMAP §0 determinism). This is the highest-value correctness fix in the file.
Fix: bind stream+workspace once and never lose it — fold both into `CublasHandle` (set stream in ctor, then `cublasSetWorkspace`), and remove the per-call `cublasSetStream` from the GEMM wrappers; or, if the wrappers must keep it, call `cublasSetWorkspace` immediately after each `cublasSetStream`. Composes with F4 (owning `Stream`).
Severity: HIGH. Effort: M. Before-M4.5: YES — M4.5 adds a per-device Fp64 parity-recompute on the same handle (TODO theme 1), magnifying an unscoped, workspace-losing handle.

**[F2 — HIGH, before-M4.5: YES] Grouped scatter kernel sets `grid.z = n_in_group` with no clamp to the 65 535 z-axis limit.**
Location: `launch_assemble_blocks_group` (`f2_blocks_kernel.cu:231-233`) launched from `compute_f2_blocks` line 266; `grid.z = static_cast<unsigned>(n_in_group)`.
CUDA grid dimensions are capped at `(2^31−1, 65535, 65535)` for x/y/z on all compute capabilities including Blackwell sm_120 ([CUDA grid dim limits, 2147483647 × 65535 × 65535](https://forums.developer.nvidia.com/t/maximum-grid-size-2147483647/35211)). `n_in_group` is `nb` (≤ `max_blocks`, which is VRAM-bounded), so today it is small and the launch succeeds — but nothing in this TU or the kernel clamps it. A bucket of >65 535 blocks-in-chunk (a small-P / huge-`n_block` regime, or a future budget bump) would silently exceed the z limit and throw `cudaErrorInvalidConfiguration` from the post-launch `cudaGetLastError()`. It is latent, not active, but it is an unguarded device-launch invariant.
Why it matters: arch §2 fail-fast / §7 launch-config correctness. The mitigation is mechanical and the TODO already lists "clamp grid.z to 65535" in backlog B.
Fix: clamp the z extent and loop the kernel over z-tiles, or assert `nb <= 65535` at the launch site (cheap; `max_blocks` is already chosen so a chunk fits VRAM, so the assert essentially never trips but documents the invariant). Better: tile the batch axis the same way chunks already tile.
Severity: HIGH (latent silent launch failure on an unvalidated axis). Effort: M. Before-M4.5: YES.

**[F3 — HIGH, before-M4.5: YES] The feeder puts the SNP count on `grid.y` (capped 65 535) instead of `grid.x` (capped 2^31), so `compute_f2_blocks` over all M fails at ~1M SNPs.**
Location: `compute_f2_blocks` calls `launch_f2_feeder(..., P, M, stream_)` (line 203) over **all** M; the feeder grid is `f2_block_kernel.cu:230-231`: `grid(grid_for(P), cdiv(M, kCdivBlock))` → `grid.x = cdiv(P,16)`, `grid.y = cdiv(M,16)`.
At the real AADR scale in the headers (M=584 131), `grid.y = ceil(584131/16) ≈ 36 508` — under 65 535, so M4 passes today. But a 1.05M-SNP dataset gives `grid.y ≈ 65 670 > 65 535` → `cudaErrorInvalidConfiguration`. The decode path got the orientation right (decode_af_kernel coalesces on the SNP axis and the SNP count is the large axis there); the *feeder* puts the large axis (M, the SNP count) on the **y** dimension, which is exactly the one with the 65 535 cap. The `compute_f2` (single-block) path is bounded by a single block's SNP count so it is safe; the exposure is the M4 whole-M feeder.
Why it matters: arch §7 launch-config; this is the "unsafe-axis-orientation twin" of the decode kernel. 1M+ SNP datasets are explicitly in scope (arch §11.1 "1M+ SNPs"), so this is reachable on real data, not pathological.
Fix: orient the feeder grid so the SNP axis is `grid.x` (the 2^31 axis) and P is `grid.y` (P ≤ ~4266, always under 65 535), swapping the in-kernel index mapping accordingly; or tile the y axis. The fix is in `f2_block_kernel.cu`, but this TU is the caller that drives M into the unsafe axis, so it is the natural site to flag.
Severity: HIGH (reachable launch failure on in-scope data). Effort: M. Before-M4.5: YES (M4.5/M5 push M higher per device, not lower).

**[F4 — HIGH, before-M4.5: YES] Default `cudaStream_t stream_ = nullptr` makes every `cudaMemcpyAsync` host-synchronous and leaves no per-device lane (the RAII `Stream` exists, unused).**
Location: member `stream_ = nullptr` (line 338); all three methods use it for every `cudaMemcpyAsync` and launch.
(a) Semantics/perf: the H2D uploads copy from `Q.data` (pageable `std::vector` host memory). Per the CUDA Runtime docs, `cudaMemcpyAsync` "might be synchronous with respect to host" for "transfers between device memory and pageable host memory," and a pageable source is staged into pinned memory by the driver which "may synchronize with the stream" ([CUDA Runtime API — API synchronization behaviors](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)). So "Async" buys nothing here regardless of stream. (b) Architecture: a literal null `cudaStream_t` is the legacy default stream; the fully-written RAII `Stream` (stream.hpp) is unused. The TODO calls this "Default-stream debt … RAII Stream exists, unused ⇒ every cudaMemcpyAsync is host-synchronous + no lane to per-device-ify" — its #1 pre-M4.5 item.
Why it matters: arch §7 ("one Stream per independent lane"; RAII for all CUDA resources) and §11.1 (the streaming pipeline needs a dedicated copy stream). M4.5/M5 cannot per-device-ify work that runs on the global null stream.
Fix: owning `Stream stream_;` member declared **before** `blas_` (destruction order — the stream must outlive the handle that binds it), construct `blas_{stream_.get()}`, route all memcpys/launches/syncs through `stream_`. This forces the F1 fix to be done right (workspace bound after the stream).
Severity: HIGH. Effort: M. Before-M4.5: YES.

**[F5 — MED] Integer truncation of `M` (`long`) to `int` for the GEMM `k` is silent and unchecked in `compute_f2`.**
Location: `compute_f2` → `run_f2_gemms(..., P, M, ...)` (line 86); inside `f2_block_kernel.cu:248` `M` is narrowed `const int Mi = static_cast<int>(M)` and passed as `cublasGemmEx`'s `k`. `cublasGemmEx` takes **`int`** for m/n/k/lda/ldb/ldc; there is no `cublasGemmEx_64` (the `_64` variants exist only for a documented subset) ([cuBLAS 64-bit Integer Interface](https://docs.nvidia.com/cuda/cublas/index.html)). So `compute_f2` silently produces wrong results for `M > INT_MAX` (~2.1e9). `compute_f2_blocks`'s per-block GEMM `k` is the bounded `s_pad`, so the batched path is safe; the exposure is the M0 whole-block path.
Why it matters: `MatView::M` is `long` precisely "so a [P×M] view over a large SNP block does not overflow a 32-bit count" (views.hpp) — yet that width is discarded one call deeper with no guard (arch §2 fail-fast; ROADMAP §4 keeps `size_t` indexing mandatory above P≈32k). 2.1e9 SNPs in one block is unrealistic, so latent, but it is silent corruption rather than a clean error.
Fix: precondition-assert `M <= INT_MAX` in `compute_f2` (or document the single-block cap). The streaming M5 path keeps blocks below the int limit anyway.
Severity: MED. Effort: S. Before-M4.5: no.

**[F6 — LOW–MED] `decode_af`'s packed-tile copy multiplies a `uint8_t` count by `sizeof(uint8_t)`, masking the idiom and diverging from the `.bytes()` convention used elsewhere.**
Location: `decode_af`, lines 304–314: `cudaMemcpyAsync(dPacked.data(), tile.packed, packed_bytes * sizeof(std::uint8_t), ...)` where `DeviceBuffer<std::uint8_t> dPacked(packed_bytes)`. Correct (the product equals `packed_bytes`), but it reads as if a unit conversion happens (it does not), and it is the only memcpy not using the buffer's own `.bytes()` (the workspace uses `.bytes()`).
Why it matters: arch §7 uniform idiom / readability; `DeviceBuffer::bytes()` exists and is self-evidently correct and `T`-change-proof.
Fix: `cudaMemcpyAsync(dPacked.data(), tile.packed, dPacked.bytes(), ...)`.
Severity: LOW–MED. Effort: S. Before-M4.5: no.

### (2) Edge cases & failure modes

**[F7 — MED] `compute_f2` has NO input-validation guard, unlike its two siblings.**
Location: `compute_f2`, lines 55–103.
`compute_f2_blocks` (line 130) guards `if (P <= 0 || M <= 0 || n_block <= 0) return out;`; `decode_af` (line 298) guards `if (P <= 0 || M <= 0) return out;`. `compute_f2` does neither. With negative `P`/`M`, `static_cast<std::size_t>` of a negative wraps to ~1.8e19, and `DeviceBuffer(huge)` throws `cudaErrorMemoryAllocation` — fail-fast, but reported as OOM, not InvalidConfig. `Q.data == nullptr` with `pm > 0` faults inside `cudaMemcpyAsync`. Inconsistent with the siblings.
Why it matters: arch §2 fail-fast + cross-method consistency.
Fix: add `if (P <= 0 || M <= 0) return {};` and ideally a non-null precondition on `Q.data/V.data/N.data` to `compute_f2`.
Severity: MED. Effort: S. Before-M4.5: no.

**[F8 — MED] `compute_f2_blocks` trusts `block_id` invariants without validation; a malformed `block_id` is an out-of-bounds host (and later device) write.**
Location: lines 136–147 (block-layout scan). `block_offsets[(size_t)b] = s;` / `out.block_sizes[(size_t)b] = ...` index by `b = block_id[s]` with **no** check that `0 <= b < n_block`. The contract (backend.hpp) says dense, `0..n_block-1`, non-decreasing — but a caller passing `n_block` smaller than `max(block_id)+1`, or a negative id, writes out of bounds of the host `std::vector`s (heap corruption), then the scatter kernel indexes `dF2_all` out of bounds. Also: an *absent* block id in `[0,n_block)` (a gap) leaves `block_offsets=0`, `block_sizes=0`; the bucket loop emits a zero-size block padded with `V=0` → f2=0/vpair=0 — plausibly correct-by-accident but never asserted. The CPU oracle has the identical unchecked scan, so it is symmetric (not a divergence), but symmetric heap corruption is still heap corruption.
Why it matters: arch §2 fail-fast; §13 (a property test should reject a malformed partition as a clean error).
Fix: assert `0 <= block_id[s] < n_block` in the scan; fold into the extracted `block_ranges()` helper (F13) where it is unit-tested once.
Severity: MED. Effort: S. Before-M4.5: ideally with F13.

**[F9 — LOW] `max_blocks` cast to `int` can wrap before the clamp; self-heals only via `nb_total`.**
Location: lines 238–243. `max_blocks = static_cast<int>(chunk_budget / per_block_bytes)`; if the `size_t` quotient exceeds `INT_MAX` the cast wraps negative, then `if (max_blocks < 1) max_blocks = 1;` clamps to 1 (catastrophically slow) before `nb_total` caps it. Benign in effect (a small bucket re-caps it), but fragile. No divide-by-zero: `per_block_bytes = (4·P·s_pad + 4·slab)·8 ≥ 8` given the `P<=0`/`M<=0` guard and `s_pad≥1` from `ceil_bucket`.
Fix: `max_blocks = static_cast<int>(std::min<std::size_t>(chunk_budget / per_block_bytes, static_cast<std::size_t>(nb_total)))` with a `max(...,1)`.
Severity: LOW. Effort: S. Before-M4.5: no.

**[F10 — LOW] Redundant negative-`n_block` ternaries duplicate the `n_block <= 0` guard.**
Location: lines 125–130. `(n_block < 0 ? 0 : n_block)` appears three times *before* the `if (P <= 0 || M <= 0 || n_block <= 0) return out;` guard, so it defends the pre-guard assigns against negative `n_block` — but the guard then returns the partially-populated `out`. Same intent expressed twice.
Fix: hoist the guard to the top (after setting `out.P`/`out.n_block`); the body can then assume `n_block >= 1` and the ternaries drop.
Severity: LOW. Effort: S. Before-M4.5: no.

### (3) Numerical / precision vs §12

**[F11 — MED, before-M4.5: YES] The backend is a silent passthrough for the `STEPPE_HAVE_EMU_TUNING=OFF` + `EmulatedFp64` foot-gun.**
Location: `compute_f2` / `compute_f2_blocks` forward `precision` straight into `engage_f2_precision` (lines 86, 214, 263).
Per `f2_block_kernel.cu:201-217`, with `STEPPE_HAVE_EMU_TUNING==0`, `engage_f2_precision` still sets `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` but omits the FIXED/EAGER/mantissa pins — so cuBLAS uses *dynamic* mantissa control, which §0/§12 identify as the rejected "trap" (~60-bit, no speedup, **and** voids §12 bit-stability), while the run still reports `EmulatedFp64`. arch §9's `build()` validator is supposed to reject exactly this ("EmulatedFp64 … on a toolkit/arch where the emulation math mode is unavailable — fall back to native Fp64 or error"). The backend, which receives the user's `Precision`, is the natural enforcement point.
Why it matters: arch §9 (build-time rejection), §12 (the determinism/parity law). Silent degrade to the trap is the worst outcome: loses the speedup *and* breaks bit-stability while claiming the safe mode. TODO flags it "[high]."
Fix: primary fix is in `engage_f2_precision`/`build()` (`#error` / throw `INVALID_CONFIG`); at minimum the backend should emit a capability tag (F18) when EmulatedFp64 is engaged without tuning.
Severity: MED (engagement is the primary site; the backend is complicit as a passthrough). Effort: S. Before-M4.5: YES.

**[F12 — MED, before-M4.5: YES] One shared `blas_` whose math mode is mutated per call is not re-entrant-safe; M4.5's Fp64 parity-recompute will interleave engagements.**
Location: `engage_f2_precision` mutates global handle state (math mode, mantissa control) at lines 214 / inside the GEMM wrappers. One handle + one host thread is fine today, but M4.5's per-device Fp64 parity-recompute on the same handle (TODO theme 1) will interleave EmulatedFp64 and PEDANTIC engagements with no save/restore.
Why it matters: §12 determinism hazard the *one-shared-handle* design choice in this TU creates.
Fix: the TODO's RAII `MathModeScope` (engage once per compute call, restore prior mode). Lives in the engagement helper, but the backend's single-handle design is what surfaces it.
Severity: MED. Effort: M. Before-M4.5: YES.

Positive: the numerator/divide stays native FP64 (delegated to `launch_assemble_f2` / `launch_assemble_blocks_group`), and the precision is forwarded unchanged — the backend correctly does not second-guess §12's operation-by-conditioning policy.

### (4) CUDA idioms / RAII / stream & async / launch config / occupancy vs §7

- **[F1]** workspace lost on `cublasSetStream` — the §7/§12 handle-state gap.
- **[F2]/[F3]** y/z grid axes unclamped/mis-oriented vs the 65 535 cap — the §7 launch-config gaps.
- **[F4]** owning `Stream` unused — the central §7 RAII gap.
- **Positive:** all device memory is `DeviceBuffer<T>`; the handle + workspace are members created once (arch §7 "create once; reuse"); no per-iteration handle creation; post-launch checks live inside the `launch_*` wrappers (verified in the kernel TUs), so this TU correctly relies on them and never re-checks.

### (5) Magic numbers & hardcoded values vs §4 / ROADMAP §4

**[F13a — HIGH, before-M4.5: YES] `0.80` VRAM-utilization literal (line 230) AND the budget omits the cuBLAS workspace.**
Location: lines 226–230. `chunk_budget = 0.80 * free_b`. (a) `0.80` is a tuning fraction, not a math constant — ROADMAP §4: "No literal may survive except true mathematical constants." (b) `cudaMemGetInfo` is called after the resident set is allocated (so `free_b` excludes the resident tensors — good), but the chunk slabs are sized at `0.80*free_b` while the GEMMs will also touch the 64 MiB `workspace_` (`kCublasWorkspaceBytes`); the budget never reserves it. The 20% margin usually absorbs 64 MiB, but the budget is *nominally* unsound — on a tight device the chunk can crowd out the workspace and OOM mid-GEMM, the exact failure §11.2 says to reject up front.
Why it matters: arch §11.2 ("rejects over-budget configs up front rather than failing mid-stream"), §9 fail-fast, ROADMAP §4. TODO flags both as one "[high]" item.
Fix: `kMaxVramUtilizationFraction` in config.hpp (doc-commented with measured rationale, matching the existing constants); `chunk_budget = kMaxVramUtilizationFraction * (free_b > kCublasWorkspaceBytes ? free_b - kCublasWorkspaceBytes : 0)`. Best folded into a host-pure `max_blocks_per_chunk()` (F15).
Severity: HIGH. Effort: S. Before-M4.5: YES.

**[F13b — LOW] The `4u` slab-footprint literals in `per_block_bytes` (lines 238–240).** `4u·P·s_pad + 4u·slab` encodes a specific layout (Qg+Vg+Sg = 4·P·s_pad inputs; Gg+Vpairg+Rg = 4·slab outputs) whose derivation lives in a comment, not a named helper. Borderline-structural, but the TODO ("factor `per_block_chunk_bytes()` (kill bare `4u`)") agrees. The `2*P`/`2u*pm`/`2u*pp` factors are true structural constants of the 2P-row stack (ROADMAP §4 exempts the structural `2`), so keep those literal.
Severity: LOW. Effort: S. Before-M4.5: no (fold into F15).

### (6) Decomposition / single-responsibility / function size vs §2

**[F14 — HIGH, before-M4.5: YES] `compute_f2_blocks` is a ~170-line monolith mixing six concerns; three are pure host int-math buried in a CUDA TU.**
Location: lines 113–284. It does: (1) output sizing/zeroing, (2) the host block-range scan, (3) power-of-2 bucketing + sort, (4) resident-tensor alloc + metadata upload, (5) the feeder pass in an inner scope, (6) VRAM budget + chunked per-bucket gather→GEMM→assemble loop. Concerns (2), (3), and (6)'s budget math are **host-pure integer logic** needing no CUDA, currently untestable without a GPU because they live in a `.cu`. arch §2 (single-responsibility; keep orchestration thin) and §13 (host-pure numerics unit-testable GPU-free) are both strained. M4.5 will *copy* this structure per device, calcifying it. TODO lists it verbatim as a pre-M4.5 high item.
Fix: extract host-pure helpers — `block_ranges()` (→ `core/domain/`, F13), `bucket_blocks_by_padded_size(sizes, base)`, `max_blocks_per_chunk(free, P, s_pad, slab, fraction, ws_bytes)` (consumes the F13a fix) — each unit-tested on the host. Method then reads: validate → sizes → ranges → upload → feeder → per-bucket launch (~40 lines).
Severity: HIGH. Effort: M. Before-M4.5: YES.

**[F15 — HIGH, before-M4.5: YES] The block-range scan is duplicated verbatim between `cuda_backend.cu:136-147` and `cpu_backend.cpp:214-226`.**
Location: lines 136–147 here; `cpu_backend.cpp:214-226` (confirmed identical: same `while (s<M)` / inner `while (e<M && block_id[e]==b)` scan, differing only in storing `offsets+sizes` vs `begin+end`). DRY/layering violation — the SNP→block partition is a *domain decision* that arch §2/§8 require to live once in `core` (the canonical single-home example), and §4 keeps domain logic out of `device`. The two backends agree only because the copies are hand-synced.
Why it matters: arch §2/§8/§4. TODO: "Block-range scan duplicated in BOTH backends … Fix: `block_ranges(...)` in `core/domain/block_partition_rule.{hpp,cpp}`."
Fix: `std::vector<BlockRange> block_ranges(const int* block_id, long M, int n_block)` (host-pure) in `core/domain/block_partition_rule`; both backends call it; unit-test it; fold in the F8 bounds-assert.
Severity: HIGH. Effort: M. Before-M4.5: YES.

Positive: `compute_f2` (49 lines) and `decode_af` (47 lines) are appropriately sized and single-purpose. The precision engagement is correctly hoisted *once* before the M4 bucket loop (line 214) rather than re-derived per group — the "engage precision once, shared by all paths" pattern the TODO wants preserved.

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density

- **const-correctness:** locals consistently `const` (`P`, `M`, `pm`, `slab`, `total`, etc.); the lambdas and comparator are const-clean. Good.
- **`[[nodiscard]]`:** the three overrides and the factory carry it, matching `backend.hpp`. Good.
- **`noexcept`:** correctly absent — the methods call throwing `STEPPE_CUDA_CHECK`/`CUBLAS_CHECK` (arch §7/§10). Consistent.
- **Comment density:** very high and genuinely explanatory — the VRAM-budget rationale (lines 171–178, with concrete P=768/M=584k GB figures) is excellent and matches config.hpp's "every constant doc-commented with measured rationale" standard.
- **[F16 — LOW] Naming:** `Bucket::s_pad` is fine; the comparator params `a`/`c` (skipping `b` to dodge the loop `b`) are slightly confusing. Cosmetic.
- **[F17 — LOW] Verified-honest comment:** lines 168–169 say the bucket sort is "cosmetic / smallest first." Confirmed: buckets write disjoint `dF2_all` slabs and one chunk is resident at a time, so order affects neither results nor VRAM peak. No fix; noting it was checked.

### (8) Performance

- **[F4]** the null-stream host-synchronous copies are the main perf issue (and a §7 issue). On pageable-source H2D the "Async" is a no-op ([CUDA Runtime sync behaviors](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)); M5's pinned double-buffer (TODO ⚡ #2) is the real fix, with the dedicated `Stream` as prerequisite.
- **[F18-perf] Per-chunk `cudaStreamSynchronize` (line 271) fully serializes chunks** — no overlap between a chunk's D2H/assemble and the next chunk's gather. Deliberate ("one chunk resident at a time" for VRAM) and fine at M4 scale (a handful of chunks); revisit with M5. Severity LOW, before-M4.5 no.
- **Positive:** never materializes the `[SNP×pop×pop]` intermediate (arch §5 S2, §11.1); the feeder-raw-inputs inner scope (lines 193–209) frees `dQ_raw/dV_raw/dN_raw` before the bucket loop with a measured 35.8 GB > 32 GB OOM justification — exactly the §11.2 VRAM discipline.

### (9) Layering / API / ABI vs §4

- TU is layering-clean (see "Role & layering").
- **[F19 — LOW, before-M4.5: YES] `make_cuda_backend()` takes no `DeviceConfig`/`device_id`** — ties into F20. M4.5 needs `(int device_id, ...)`; changing the factory signature later touches `Resources` wiring, so land it with F20. Confirm the factory declaration lives in a CUDA-free header so `core`/`Resources` can call it without a CUDA include (arch §4). Effort: S now / M later.

### (10) Testability vs §13

- **[F14]/[F15]** are the testability findings: the host int-math (block-range scan, bucketing, chunk-budget) is currently un-unit-testable because it lives in a `.cu`. Extracting to `core` host-pure helpers makes them GoogleTest-able with no GPU (arch §13). High value.
- The backend is correctly diffable against `CpuBackend` at the `F2BlockTensor` seam (the existing M4 reference test).

### (11) Capability tiers (PRO-6000-capable vs budget-5090)

**[F20 — MED, before-M4.5: YES] No device selection, no capability probe, and no capability-tagged log line — yet this is where the budget-vs-capable VRAM decision is made.**
Location: whole class (no `device_id`/`cudaSetDevice`) + the `compute_f2_blocks` VRAM path (lines 226–273).
(a) Device selection: the backend ignores `DeviceConfig::devices` and never calls `cudaSetDevice` — it runs on the ambient current device. arch §9 (resources injected, not ambient) and §11.4 SPMG (`cudaSetDevice` to switch per device) both require it; TODO "[med] No device_id / cudaSetDevice … The SPMG prerequisite."
(b) Capability observability: the TODO capability-tier section mandates "a capability probe + capability-tagged results (every run records which path it took + why it degraded)." Two tier-relevant facts are decided here yet logged nowhere: the VRAM regime (a 96 GB PRO 6000 runs far larger `n_block`/P single-shot; a 32 GB 5090 chunks aggressively) and the EmulatedFp64-without-tuning silent degrade (F11). Neither is tagged. When M4.5's `cudaMemcpyPeer` device-resident combine vs host-staged combine lands, this is the TU that must log "P2P combine unavailable → host-staged fixed-order combine." The fix is *observability + device selection* (parity-neutral per the TODO), not new math.
Why it matters: TODO capability-tier contract ("degrade with an EXPLICIT logged tag"); arch §9 (injected resources), §10 (observability — and §10 forbids `printf`/`cout`, yet this TU has no logging facade at all; the teardown sinks are still stderr stubs).
Fix: thread `device_id` into the factory/ctor + `cudaSetDevice`; add `STEPPE_LOG_INFO` capability tags at the VRAM-budget decision and precision engagement once `log.hpp` is wired into `steppe_device`.
Severity: MED. Effort: M. Before-M4.5: YES (device selection is the SPMG prerequisite; the combine tag is an M4.5 deliverable).

---

## Considered & rejected

- **"`2u * pm` could overflow `std::size_t`."** Rejected. Casts to `size_t` happen *before* the multiply (lines 61–64, 179–180), so the product is 64-bit; at the ceiling (P≈4266, M≈584k) `2*pm ≈ 5e9` ≪ 1.8e19. (The `int` truncation of `M` for the GEMM *k* is the separate real issue — F5.)
- **"`cudaMemGetInfo` called too early / races other allocations."** Rejected for single-process: it is called after the resident set is allocated (line 227), so `free_b` reflects post-resident free memory — exactly what the chunk budget needs. steppe is single-process per §11.4. The unsoundness is the un-reserved workspace (F13a), not the timing.
- **"The `0.80` fraction risks OOM because cuBLAS GEMM allocates scratch beyond the workspace."** Partially rejected: with a user workspace bound (after F1), cuBLAS uses it for scratch; the 20% margin plus the 64 MiB workspace covers typical scratch. The defect is nominal (workspace not reserved, F13a), not an observed crash — kept as F13a, not escalated.
- **"Does F5's `int Mi` truncation bite the feeder in `compute_f2_blocks` too?"** Rejected: the feeder takes `long M` and grids with `cdiv(M, ...)` in `long`; only the cuBLAS GEMM path narrows to `int`, and in the M4 path the GEMM `k` is the per-block `s_pad`, not `M`. So the feeder over all M is integer-safe (its problem is the *y-axis count*, F3, not int width). F5 is scoped to `compute_f2`.
- **"Member destruction order of `stream_`, `blas_`, `workspace_`."** Checked: reverse-declaration order → `workspace_`, then `blas_`, then `stream_`. `~CublasHandle` runs (and `cublasDestroy` synchronizes per §7) before the workspace frees — safe. **When F4 lands**, `Stream stream_` must be declared *before* `blas_` so the stream outlives the handle that binds it (flagged in F4).
- **"`out.block_sizes` filled before the `n_block<=0` guard leaks."** Rejected — RAII `std::vector`, no leak; only the redundant ternary (F10, LOW).
- **"`Tf32` mishandled on the f2 path."** Rejected for this TU: `f2_compute_type` maps Tf32→native `CUBLAS_COMPUTE_64F` and `engage_f2_precision`'s `else` sets PEDANTIC, so Tf32 silently runs native FP64 — a `f2_block_kernel.cu` design choice (Tf32 is screening-only; the f2 numerator is cancellation-prone), not a backend bug. The backend just forwards. Out of scope here.
- **"Missing final `cudaDeviceSynchronize` in `compute_f2_blocks`."** Rejected — `cudaStreamSynchronize(stream_)` (line 282) is present and sufficient; result vectors are safe to read after it returns.
- **"`cudaMemcpyAsync` of `out.f2`/`out.vpair` to pageable host then immediate sync — could the copy be reordered before the assemble kernel?"** Rejected — same (null) stream as the kernels, so stream ordering guarantees the assemble completes before the D2H; the trailing `cudaStreamSynchronize` then makes the host read safe.

---

## What it takes to reach 10/10

In rough priority (HIGH/MED are pre-M4.5 unless noted):

1. **F1 — restore the determinism workspace.** Stop the per-call `cublasSetStream` from silently dropping `cublasSetWorkspace`; bind stream+workspace once. The single most important fix — makes the advertised §12 bit-stability actually true.
2. **F2 + F3 — make the launch configs 65 535-safe.** Clamp/tile the grouped scatter `grid.z`; re-orient the feeder so the SNP count (M) is on `grid.x` (2^31 axis), not `grid.y`. F3 is reachable on in-scope 1M+ SNP data.
3. **F4 — owning `Stream stream_`** declared before `blas_`; all copies/launches/syncs through it. Pays the TODO's #1 default-stream debt and is F1's natural home.
4. **F14 + F15 — decompose `compute_f2_blocks`;** extract host-pure `block_ranges()` (→ `core/domain`, shared with CpuBackend), `bucket_blocks_by_padded_size`, `max_blocks_per_chunk`; unit-test GPU-free. Method shrinks to ~40 lines.
5. **F13a — fix the VRAM budget:** reserve `kCublasWorkspaceBytes`, promote `0.80` → `kMaxVramUtilizationFraction`.
6. **F20 + F19 — thread `device_id` + `cudaSetDevice`** through the factory/ctor (SPMG prerequisite; land the signature change now) and add capability-tagged logging.
7. **F11 + F12 — close the EmulatedFp64-without-tuning foot-gun observably** and add the `MathModeScope` for M4.5's coexisting Fp64 recompute.
8. **F7 + F8 — the missing input-validation guards** (`compute_f2` P/M/null; `block_id` in-range assert, folded into `block_ranges`).
9. **F5, F6, F9, F10, F13b, F16 — the LOW/MED hygiene set** (int-`k` guard, `dPacked.bytes()`, chunk-clamp width, redundant ternary, named `per_block_chunk_bytes`, naming).

After 1–6 land, this is comfortably a 9+. 10/10 also wants the capability-tagged logging (F20) and the host-pure helpers under unit test (F14/F15) so the int-math is provably correct GPU-free.

---

## Good patterns to keep

- **RAII for every CUDA resource** — `DeviceBuffer<T>` for all device memory, `CublasHandle` + `DeviceBuffer<std::byte>` workspace as members created once. Textbook §7. (Just fix the stream — F4.)
- **"Engage precision once, shared by all paths"** — `engage_f2_precision` hoisted before the M4 bucket loop (line 214), the same helper used by the M0 path. The DRY single-source-of-precision the TODO wants preserved.
- **Never materializing `[SNP×pop×pop]`** — only the reduced `[P×M]`/`[2P×M]`/`[P×P]`/`[2P×P]` buffers plus the resident `[P×P×n_block]` tensor (§5 S2, §11.1).
- **The feeder-raw-inputs inner scope** (lines 193–209) freeing `dQ_raw/dV_raw/dN_raw` before the bucket loop, with a *measured* VRAM justification — exactly the §11.2 discipline, documented with real numbers.
- **Narrow `launch_*` / `run_*` wrappers** — no `<<<>>>` in this TU; post-launch checks delegated to the kernel wrappers. Clean §7 host/device separation.
- **High, honest comment density** — every non-obvious choice (the size-grouped strided-batched design, the chunking-for-VRAM, the contiguous-block-id assumption, the "cosmetic" sort) is explained with its arch-§ cite and measured rationale, matching config.hpp's exemplary standard.
- **CUDA-free seam respected** — results copied into plain `std::vector` `F2Result`/`F2BlockTensor`/`DecodeResult` before crossing back to `core`; the factory returns the abstract `ComputeBackend`.
