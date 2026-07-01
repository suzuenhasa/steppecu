# Review — `src/device/cuda/check.cuh` — M4.5 delta (unit: device-cuda-check-m45)

Adversarial second pass focused on the **M4.5 delta** to the project's single
CUDA/cuBLAS error-checking home: the new **non-throwing `STEPPE_CUDA_WARN`** —
`detail::cuda_warn` (lines 117–149), the `STEPPE_CUDA_WARN` macro (171–182), and the
doc that describes it (file-header 18–24, function block 117–133, macro block 171–182).
The pre-M4.5 review (`docs/cleanup/device-cuda-check.md`, 8.5/10) is the baseline; the
CAP-1/CAP-2 non-throwing path it *recommended* is exactly what this delta lands, and the
`STEPPE_DEBUG_ONLY` / `core/internal/log.hpp` single-homes the prior review wanted now
exist and this file consumes them (lines 39–40, 200–201). This pass judges ONLY the M4.5
addition and the lines it touches, against the focus question:

> Is `STEPPE_CUDA_WARN` **DRY** with `STEPPE_CUDA_CHECK`, **warning-clean**, **correct in
> returning the status**, and is the **"log once" claim real**?

The first-pass M4.5 review (this doc's predecessor) scored 8.5/10 with 13 findings + 1
perf finding. This pass **re-verified every one against the actual source and the official
NVIDIA Runtime-API docs**, reframed E-1 (its "swallowed / confusing downstream fault"
framing is too strong — the call site already clears the sticky error AND fails fast at the
throwing `cudaMemcpyPeer`, under a caller contract that makes the residual a contract bug),
**confirmed** the two headline findings (D-1 DRY, D-2 "log once"), and **added three new
findings** the prior pass missed — the most important being **N-1: `check.cuh` never
documents the sticky-last-error caveat of `STEPPE_CUDA_WARN`**, a real footgun the
`p2p_combine.cu:265` author rediscovered the hard way and open-coded an 11-line comment +
manual `cudaGetLastError()` clear to work around.

Standards judged against: `docs/architecture.md` §2 (DRY/single-source), §4 (layering,
named constants), §7 (CUDA idioms, fail-fast, RAII, post-launch checks), §10 (log taxonomy
/ `STEPPE_LOG_*` never `printf`), §11.4 (capability-tiered combine + *an explicit logged
fallback*), §12 (parity — the WARN path must be parity-neutral); `docs/ROADMAP.md` §4/§5/§6.
Every load-bearing device-behavior claim is cited to the official CUDA Runtime API docs.

**Re-verified facts (grep + read + official docs, this pass):**
- `STEPPE_CUDA_WARN` has exactly **three** production call sites (grep-confirmed; the other
  matches are comment mentions), all parity-neutral and off the statistic path:
  - `cuda_backend.cu:523` — `STEPPE_CUDA_WARN(cudaDeviceCanAccessPeer(&access, …))`, the
    capability probe, called **once per peer** in the `for (peer … count)` loop (516–528)
    inside `capabilities()`. The status is *captured* (`const cudaError_t s = …`, :522) and
    branched on (:524) — honoring `[[nodiscard]]`.
  - `p2p_combine.cu:265` — `(void)STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(owning, 0))`,
    the expected-`cudaErrorPeerAccessAlreadyEnabled` degrade, in the `g=0..G-1` combine
    loop, **immediately followed by `(void)cudaGetLastError()` (:278)** that clears the
    sticky error the WARN'd status leaves behind.
  - `p2p_combine.cu:165` — `~DeviceGuard(){ (void)STEPPE_CUDA_WARN(cudaSetDevice(dev)); }`,
    a device-restore in a `noexcept` RAII destructor.
- `capabilities()` is itself probed **once per device** by `Resources::build`
  (`resources.cpp:92`, "the ONE probe, owned here", in the `for (ordinal …)` loop), so on a
  no-peer budget box the same "cannot access peer" degrade is WARN'd **G·(count−1)** times
  per build (confirmed — substantiates D-2's spam claim structurally).
- A dedicated, no-shadow unit test exists: `tests/reference/test_cuda_check.cu` directly
  includes the production `check.cuh` and pins WARN-does-not-throw + yields-status,
  WARN(success) quiet pass-through, and the CHECK-vs-WARN contrast (the prior pass's T-1
  shadow gap is closed).
- The "log once" sibling is real and next door: `f2_block_kernel.cu:221
  warn_emulated_fp64_downgraded_once()` uses `static std::atomic_flag … test_and_set(...)`
  so the tag fires **at most once per process**, explicitly "thread-safe (M4.5 multi-GPU
  may engage from more than one host thread)". `cuda_warn` has **no** such flag (D-2).
- `STEPPE_LOG_WARN` is debug-only: under `NDEBUG` it is `((void)0)` and does **not**
  evaluate its arguments (`log.hpp:44–54`) — confirmed; this is what makes the
  `[[maybe_unused]]` necessary and the "build-mode-independent status return" claim true.
- **Official-docs facts grounding this pass** (CUDA Runtime API, fetched 2026-06):
  - `cudaDeviceEnablePeerAccess` documented returns are `cudaSuccess`,
    `cudaErrorInvalidDevice` ("if `cudaDeviceCanAccessPeer()` indicates that the current
    device cannot directly access memory from peerDevice"),
    `cudaErrorPeerAccessAlreadyEnabled` ("if direct access … has already been enabled"),
    and `cudaErrorInvalidValue` ("if flags is not 0") — `group__CUDART__PEER`.
  - `cudaGetLastError` "Returns the last error … and resets it to `cudaSuccess`" and "may
    also return error codes from previous, asynchronous launches"; `cudaPeekAtLastError`
    does **not** reset — `group__CUDART__ERROR`.
  - `cudaGetErrorName`/`cudaGetErrorString` return a non-null string even for unknown codes
    ("'unrecognized error code' is returned") — `group__CUDART__ERROR`; so the WARN line's
    `%s` for those two is always NULL-safe.
  - `std::source_location::line()` is `constexpr uint_least32_t line() const noexcept`
    (cppreference, `<source_location>`), so `%u` + `static_cast<unsigned>` is the correct,
    width-matching `-Wformat` fix.

## Role & layering

`STEPPE_CUDA_WARN` is the correct home for the M4.5 capability-degrade diagnostics. It is
the non-throwing sibling of the throwing `STEPPE_CUDA_CHECK`, lives in the same
CUDA-private `.cuh` (so it stays `PRIVATE` to `steppe_device`, §4), routes its one line
through the §10 single sink `STEPPE_LOG_WARN` (no bare `printf` — correct), and is
documented as fault-vs-recoverable-disjoint so the statistic path uses only the throwing
checker and §12 parity is identical on both capability tiers. The placement is exactly
right: §11.4 mandates the P2P fast-path be a `cudaDeviceCanAccessPeer`-gated fallback
**"with an explicit logged fallback"** (verbatim, architecture.md §11.4), and the expected
degrades (`cudaDeviceCanAccessPeer` answering "no", `cudaErrorPeerAccessAlreadyEnabled`)
now have a sanctioned non-throwing, logged path instead of being forced through the
throwing checker (which would convert a graceful degrade into a hard fault). That is the
single most important correctness property of the delta, and it is met. The `[[nodiscard]]`
(line 134) is the right contract — the return is the product, the caller must branch — and
it is honored at the one branching site (`cuda_backend.cu:522`); the two intentionally
discarding sites cast to `(void)`.

The CUDA-free/device split stays clean: the only new dependency, `STEPPE_LOG_WARN` from
`core/internal/log.hpp`, is the §8/§10-sanctioned CUDA-free `core_internal` INTERFACE
target consumed by the device layer — no CUDA header leaks into `core`, no layering
inversion, the non-throwing status never crosses the public C ABI (it selects an internal
combine tier whose two paths are bit-identical).

The remaining issues, in descending substance: a **newly found** undocumented
sticky-last-error caveat that every side-effecting WARN site must work around (N-1, the
prior pass missed this entirely); the **DRY duplication** of `CudaError`'s message format
(D-1); the **false "log once"** doc claim that a sibling in the same layer *does* implement
(D-2); a reframed discard-site finding (E-1) whose real merit is a missing named-discard
helper; and a handful of LOW idiom/precision/test nits.

## Score: 8/10 — the WARN path is correct, parity-neutral, [[nodiscard]]-honest, warning-clean, and tested; held back by (a) an UNDOCUMENTED sticky-last-error caveat that turns a WARN'd side-effecting status into a phantom fault at the next post-launch check — the home stays silent while the call site open-codes an 11-line workaround (N-1, NEW); (b) a duplicated message-format string that re-implements `CudaError`'s formatting in the very file whose reason to exist is DRY (D-1); (c) a "log once" doc claim that is false as written — once per *invocation*, G·(count−1) times on a no-peer box — while a sibling implements the real per-process "once" (D-2)

Half a point below the prior pass's 8.5 because N-1 is a genuine new MED-severity contract
gap (a sticky-error footgun the home does not document, which the call site had to discover
and work around) that materially affects the focus question "is it correct in returning the
status" — the answer is *yes for the return value, but the home omits the side-effect
contract that comes with using `STEPPE_CUDA_WARN` on a side-effecting runtime call.* On the
four focus questions:
- **correct in returning the status — YES** (verified against docs + the test;
  build-mode-independent, `return status;` sits outside the `if`), **but** the doc omits
  that a WARN'd non-success status from a *side-effecting* call also leaves the sticky
  last-error set (N-1).
- **warning-clean — YES** (`[[maybe_unused]]` + `static_cast<unsigned>` are the exactly
  correct two annotations for `-Wextra -Werror`; verified the `%u`/`uint_least32_t` match).
- **DRY with `STEPPE_CUDA_CHECK` — NO, partially** (duplicates `CudaError`'s message format
  byte-for-byte as a `printf` literal — D-1).
- **is the "log once" claim real — NO** (D-2: once-per-call, and the doc's own example loop
  calls it `count−1` times per `capabilities()`, G times per build).

None of the four defects is a parity risk and none is a correctness bug in the shipped
binary (the sticky error is cleared at the one site that needs it). They are exactly the
"accreted warts" this pass is meant to find. Fixing N-1/D-1/D-2 plus the small polish items
takes it to 9.5+.

---

## Findings

### (8) Performance — FIRST CLASS THIS PASS

The WARN path is **off every hot/statistic path by construction** (§12; verified — all
three call sites are a capability probe, the off-critical-path combine peer-enable, and a
teardown guard), so throughput surface is near-zero. There are no copies, no kernels, no
streams, no allocations in this header, so the five perf sub-axes the mandate flags are:

- **Data bouncing — N/A** (no H2D/D2H/D2D in this header; that is `p2p_combine.cu`'s domain).
- **Missing grid-stride — N/A** (no kernels here).
- **Sequential P2P / streams — N/A** (no transfers here).
- **Needless sync / default-stream / hot-path alloc — N/A** (none here; the WARN line is
  release-compiled-out and the success path is a single compare-and-return).
- **Type-casting noise — N/A as *noise*** (the one cast, `static_cast<unsigned>(loc.line())`
  at :144, is a *correct* `-Wformat` fix matching `%u` to `uint_least32_t`, not noise — see
  W-1). Explicitly N/A on all five for this header.

**P-1 [LOW, effort S, PARITY-SAFE: yes] — failure branch lacks `[[unlikely]]`.**
Location: `detail::cuda_warn`, line 138 `if (status != cudaSuccess)`.
The hot/expected path is `status == cudaSuccess` → fall through to `return status;` (one
compare, one return, inlined). The failure branch calls `cudaGetErrorName`/`String` +
`STEPPE_LOG_WARN`; under `NDEBUG` the whole `STEPPE_LOG_WARN(...)` body and its argument
evaluation vanish (`log.hpp:45` → `((void)0)`), so in release the failure branch is *also*
just `if + return` — **no release-mode cost on either path.** The only note is the missing
`[[unlikely]]` (mirrors the prior `check.cuh` review's A-2 for `cuda_check`): documents
intent, can keep the success path in icache. Negligible benefit (not hot), zero risk. Worth
doing only paired with the same hint on `cuda_check`/`cublas_check` for consistency (they
were left bare in the prior pass too). Severity LOW, effort S.

### (1) Correctness & bugs

**N-1 [MED, effort S, PARITY-SAFE: yes] — NEW: `check.cuh` never documents the
sticky-last-error caveat of `STEPPE_CUDA_WARN`; a WARN'd non-success status from a
*side-effecting* runtime call leaves the per-thread sticky error set, which the very next
`cudaGetLastError()` / `STEPPE_CUDA_CHECK_KERNEL()` surfaces as a phantom fault. The home
stays silent; the call site rediscovered it and open-codes an 11-line workaround.**
Location of the gap: `cuda_warn` doc-block 117–133 and the macro doc 171–182 (they describe
"YIELDS the status so the caller can branch" but say nothing about the sticky side effect).
Location of the workaround: `p2p_combine.cu:266–278` — a 13-line comment + a manual
`(void)cudaGetLastError()`.

This is the headline NEW finding and the prior pass missed it entirely. The mechanism,
verified against the docs: a CUDA Runtime call that returns a non-success status records it
as the per-thread sticky last-error (CUDA Runtime API `group__CUDART__ERROR`:
`cudaGetLastError` "Returns the last error that has been produced by any of the runtime
calls … and resets it to `cudaSuccess`"). `cuda_warn` only **reads the return value** to
branch; it deliberately does **not** call `cudaGetLastError()` to consume the sticky state
(correct — a checker should not silently clear errors it did not raise). So after
`STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(owning, 0))` returns
`cudaErrorPeerAccessAlreadyEnabled` (the *expected, tolerated* status on the 2nd+ combine),
the sticky last-error is **still set** — and the next `STEPPE_CUDA_CHECK_KERNEL()` (which
calls `cudaGetLastError()`, line 199) would surface that stale, already-handled status as
if THIS kernel launch failed, throwing a `CudaError` for a fault that never happened.

The `p2p_combine.cu:265` author hit exactly this and worked around it with the explicit
`(void)cudaGetLastError()` at :278 plus a 13-line comment (:266–277) explaining the trap
verbatim: *"Left uncleared, the NEXT cudaGetLastError() — inside the place-add's
STEPPE_CUDA_CHECK_KERNEL below — would WRONGLY surface this stale, already-handled status as
if the kernel launch failed."* That is precisely the contract the **home** should state once
— and it does not. `check.cuh`'s WARN doc even contrasts itself against `STEPPE_CUDA_CHECK`
and explains the build-mode-independent return, but never mentions that a WARN on a
side-effecting call leaves a landmine for the next post-launch check.

Why it matters: §2 single-home — the caveat belongs at the checker, not rediscovered per
call site; the next author who uses `STEPPE_CUDA_WARN` on, say, a tolerated
`cudaErrorPeerAccessNotEnabled` from a future `cudaDeviceDisablePeerAccess`, before a
kernel, will get a phantom `CudaError` with no warning that the contract required a
`cudaGetLastError()` clear. The `cuda_backend.cu:523` probe site is *safe* only because
`cudaDeviceCanAccessPeer` returns `cudaSuccess` (the "cannot" is in the out-param, not the
status), so no sticky error is left — but that is luck of which API it wraps, not something
the doc tells you to reason about.

Concrete fix (doc-only, parity-safe): add to the `cuda_warn` doc-block (and one line on the
macro) a caveat: *"`STEPPE_CUDA_WARN` reads the returned status to branch; it does NOT
consume the per-thread sticky last-error. If the WARN'd call is a side-effecting runtime
call (not a pure query like `cudaDeviceCanAccessPeer`) and you tolerate its non-success
status, you MUST `(void)cudaGetLastError()` before the next post-launch check
(`STEPPE_CUDA_CHECK_KERNEL`) or it will misattribute the stale status to the next launch —
see `p2p_combine.cu`."* Optionally provide a `STEPPE_CUDA_WARN_CLEAR(expr)` convenience
macro that WARNs then clears (`cudaGetLastError()`) for the tolerate-and-continue pattern,
so the call site does not hand-roll the clear. Severity MED (it is a real, demonstrated
footgun that the milestone's marquee feature already tripped over once), effort S,
parity-safe (observability/error-state only; no number changes). *Adversarial check — is
this a real gap or am I over-reading?* It is real: a maintainer reading only `check.cuh`
sees "YIELDS the status, branch on it" with zero indication a tolerated non-success leaves
the sticky error set; the existing 13-line call-site comment is direct evidence the contract
was non-obvious and had to be discovered. Confirmed, not a false positive.

**No correctness bug in the shipped behavior.** The status return is correct and
build-mode-independent (verified): `cuda_warn` returns `status` unconditionally at line 148,
*outside* the `if`, so the branched value is identical in debug and release — only the WARN
*line* is `NDEBUG`-gated. The dedicated test `test_cuda_check.cu:61–80,87–119` pins this
(yields the status, never throws, `cudaSuccess` quiet pass-through, CHECK-on-same-status
throws-contrast). The doc-block claim (126–129) that the status return is
"build-mode-INDEPENDENT" is true and verified. The shipped binary does the right thing; the
defects are the documentation/DRY/idiom items below (and N-1's missing-caveat).

### (2) Edge cases & failure modes

**E-1 [LOW (was MED — DOWNGRADED), effort S, PARITY-SAFE: yes] — Two of the three call
sites `(void)`-discard the `[[nodiscard]]` return; the prior pass's claim that the `:265`
discard "swallows" a real fault and lets it "re-surface as a confusing downstream
`cudaMemcpyPeer` fault" is too strong — the site already clears the sticky error AND
fails-fast at the throwing `cudaMemcpyPeer`, under a caller contract that makes a residual
`cudaErrorInvalidDevice` a *contract-violation bug*, not a swallowed-and-lost one.**
Location of the contract: `cuda_warn` `[[nodiscard]]`, line 134. Discarding sites:
`p2p_combine.cu:165` (DeviceGuard dtor) and `p2p_combine.cu:265` (peer-enable).

Re-verified against the source and the docs, and **downgraded** from the prior pass:
- `p2p_combine.cu:165` — the `~DeviceGuard()` teardown restore. Discarding is **defensible**:
  a destructor cannot meaningfully act on a failed `cudaSetDevice`, the WARN line is the
  diagnostic, and the dtor is `noexcept`. *Acceptable* — but the `(void)` cast is the same
  noise the prior `check.cuh` review's B-2 flagged for the throwing path; a named helper
  reads better (A-2 below).
- `p2p_combine.cu:265` — `(void)STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(owning, 0))`.
  The prior pass rated this MED, claiming a genuine `cudaErrorInvalidDevice` is "swallowed
  and only re-surfaces as a confusing downstream `cudaMemcpyPeer` fault." **That framing is
  too strong, and I downgrade it.** Re-reading the full site (lines 252–319) and the header
  contract (`p2p_combine.hpp:64–76`): (1) the caller has **already** verified
  `gpus[0].caps.can_access_peer && G>=2` before calling (the §11.4 degrade decision is made
  *before* this routine), so a `cudaErrorInvalidDevice` here — documented by CUDA as
  returned "if `cudaDeviceCanAccessPeer()` indicates that the current device cannot directly
  access memory from peerDevice" (`group__CUDART__PEER`) — means the caller violated its
  own promise: it is a *contract bug*, not an expected status to branch on. (2) The very
  next line `(void)cudaGetLastError()` (:278) clears the sticky error (so it does NOT
  re-surface at the place-add's `STEPPE_CUDA_CHECK_KERNEL`). (3) The subsequent
  `cudaMemcpyPeer` IS routed through the **throwing** `STEPPE_CUDA_CHECK` (:297–300): with
  peer access genuinely unavailable the DMA fails and **fails-fast there** — exactly as the
  site comment (:262–264) and the header (`p2p_combine.hpp:71–76`) state by design. So the
  fault is *not lost*; it surfaces fail-fast at the DMA, attributed to the DMA. The only
  real merit is **diagnostic locality** — the error points to `cudaMemcpyPeer` rather than
  to the peer-enable that was the true cause. That is worth a one-line capture-and-branch
  for a sharper message, but it is LOW (the binary already fails-fast on the right
  condition), not MED.

Concrete fix (LOW): at `:265`, capture and branch for a sharper diagnostic —
`const cudaError_t s = STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(owning, 0));` then, if the
codebase wants the early pinpoint, `STEPPE_ASSERT(s == cudaSuccess || s ==
cudaErrorPeerAccessAlreadyEnabled, "p2p combine: peer not reachable despite can_access_peer
contract")` — a debug precondition (cheap, off the hot path, and parity-neutral) that
catches the contract violation at the real cause rather than relying on the downstream DMA.
For the dtor site, leaving `(void)` is fine; a named discard helper (A-2) makes the intent
read as intent, not as a `[[nodiscard]]` suppression. (Call-site finding, in scope because
the header's `[[nodiscard]]` is what makes it visible.) Severity LOW (downgraded), effort S.

**E-2 [LOW, effort S, PARITY-SAFE: yes] — `cudaErrorNotReady` from query/poller APIs is
the wrong fit for `STEPPE_CUDA_WARN` (it would log on every not-ready iteration), and the
doc does not say so.**
The prior `check.cuh` review's B-3 flagged that `cudaStreamQuery`/`cudaEventQuery`
legitimately return `cudaErrorNotReady`. `STEPPE_CUDA_WARN` is *a* non-throwing path, but
its doc frames it around *capability degrades* (peer access). A poller returning
`cudaErrorNotReady` is **not** a degrade; it is the normal "not done yet" answer that should
produce **no** WARN line. Routing a busy-poll through `STEPPE_CUDA_WARN` would emit a WARN
on every not-ready iteration (thousands), and would also leave a sticky error each time
(N-1). No current site does this (verified: no `cudaStreamQuery`/`cudaEventQuery` in the
tree), so it is latent — but when M5's double-buffered pipeline lands the poller,
`STEPPE_CUDA_WARN` is the wrong tool and a third, **silent** predicate (e.g.
`cuda_is_not_ready(status)`, no log, no sticky-clear concern) will be wanted. Fix: one
comment line noting `STEPPE_CUDA_WARN` is for *recoverable-but-noteworthy* statuses
(degrades worth a tag), **not** for high-frequency expected statuses like
`cudaErrorNotReady`, which need a non-logging predicate. Severity LOW (latent), effort S.

**E-3 [LOW, effort S, PARITY-SAFE: yes] — no guard that `expr`/`loc` survive into the
WARN line; harmless via the macro, but `cuda_warn` is `detail::`-reachable directly.**
`detail::cuda_warn` passes `expr` straight to `STEPPE_LOG_WARN("… '%s' …", … expr …)`.
`expr` is always `#expr` (a non-null string literal) via the only sanctioned path (the
macro), so this is safe today — identical to the `CudaError` ctor's borrow (prior review
B-2, rated LOW). Noted for symmetry: a direct `detail::cuda_warn(s, nullptr)` would feed a
`nullptr` to `%s` (UB). `cudaGetErrorName`/`String` are NULL-safe (verified — they return
"unrecognized error code" for unknown codes, `group__CUDART__ERROR`), so only `expr` is the
concern. The same one-line "`expr` is `#expr`, never null" invariant comment the prior
review recommended for the ctors covers `cuda_warn`. Severity LOW.

### (6) Decomposition / single-responsibility (§2 DRY) — the core focus finding

**D-1 [MED, effort S, PARITY-SAFE: yes] — `cuda_warn` re-implements `CudaError`'s message
format as a second source of truth, in the one file whose entire purpose is DRY.**
Location: `cuda_warn` `STEPPE_LOG_WARN(...)` format, lines 143–146, vs `CudaError` ctor,
lines 55–58.

This is the headline DRY finding and the direct answer to "is it DRY with
`STEPPE_CUDA_CHECK`." The same diagnostic message is spelled in two syntaxes:

```cpp
// CudaError ctor (55–58): std::string concatenation
loc.file_name() + ":" + std::to_string(loc.line()) + " (" + loc.function_name()
  + "): '" + expr + "' -> " + cudaGetErrorName(status) + ": " + cudaGetErrorString(status)

// cuda_warn (143–146): printf format
STEPPE_LOG_WARN("%s:%u (%s): '%s' -> %s: %s",
  loc.file_name(), static_cast<unsigned>(loc.line()), loc.function_name(), expr,
  cudaGetErrorName(status), cudaGetErrorString(status));
```

The shape `"file:line (func): 'expr' -> name: string"` is now defined **twice**. The
header's opening comment (6–8) brags "There is exactly ONE STEPPE_CUDA_CHECK … and every
CUDA … *fault* call routes through them," and the file exists *because* §2 mandates one
home — yet the M4.5 delta added a second, hand-kept copy. The author even documented the
duplication (line 119 "same file:line:function + error name/string as CudaError"; line 173
"(same shape)") rather than removing it. If the format changes (e.g. the prior review's B-1
fixed-buffer rewrite, or appending the cuBLAS-style description), the two silently drift —
the B-1 fix would touch `CudaError` and leave `cuda_warn`'s `printf` copy stale.

Why it matters: §2 ("implemented exactly once … consumed through one target"); §8
single-home. Concrete fix (preferred, dovetails with the prior B-1): extract one
`noexcept` `snprintf`-based helper both consume —
`detail::format_cuda_error(char* buf, std::size_t n, status, expr, loc)` writing the
canonical string — then `CudaError`/`CublasError` fill their `buf_` from it (also fixing
B-1's OOM-path allocation) and `cuda_warn` logs it (`STEPPE_LOG_WARN("%s", buf)`).
Lighter-weight minimum: factor just the *format-string literal* into one
`inline constexpr const char* kCudaErrorFmt = "%s:%u (%s): '%s' -> %s: %s";` (a named
constant per §4) referenced by both the `STEPPE_LOG_WARN` call and a `snprintf` in
`CudaError`. Severity MED (DRY-in-the-DRY-file is precisely the §2 smell, and the delta
introduced it), effort S, parity-safe (formatting is off the statistic path).

*Adversarial check — is the duplication justified?* The two genuinely differ in *mechanism*
(`std::string` concat allocates/throws; `printf`/`snprintf` does not, which is what the
`noexcept`/release-silent WARN wants), so you cannot call `CudaError::what()` from
`cuda_warn` without constructing a `CudaError` (which can throw `std::bad_alloc` — wrong for
a non-throwing path). That argues against "reuse `CudaError`" but **not** against "share a
`snprintf` helper / format literal," which is `noexcept` and serves both. The duplication is
*not* justified. Confirmed, not a false positive.

**D-2 [MED, effort S, PARITY-SAFE: yes] — the "log once" / "logs ONE STEPPE_LOG_WARN line"
claim is FALSE as written: `cuda_warn` logs once **per invocation**, not once per condition
or per process — and a sibling in the same layer implements the real "once."**
Location of the claim: lines 18–20 ("logs one STEPPE_LOG_WARN line"), 118–119 ("emits
exactly ONE STEPPE_LOG_WARN line"), 172–173 ("logs ONE STEPPE_LOG_WARN line"). Code:
138–147 — an unconditional log inside `if (status != cudaSuccess)`, no dedup/once-flag.

The direct answer to "is the 'log once' claim real": **no.** "One line" is ambiguous between
"one line *per call*" (true) and "one line *ever*, deduplicated" (false, and what "log once"
idiomatically means in this codebase). The evidence it reads as the latter and is wrong:
1. The codebase has a *real* "log once" helper next door — `f2_block_kernel.cu:221
   warn_emulated_fp64_downgraded_once()` uses `static std::atomic_flag … test_and_set` so
   the capability tag fires **at most once per process**, explicitly "thread-safe (M4.5
   multi-GPU may engage from more than one host thread)". That is the project's established
   meaning of "warn once" for a capability degrade. `cuda_warn` has **no** such flag.
2. The doc's own canonical example is a loop: `cuda_backend.cu:516–528` calls
   `STEPPE_CUDA_WARN(cudaDeviceCanAccessPeer(...))` once **per peer**. On a no-peer box (the
   documented budget-5090 degrade case) every peer is "cannot," so the probe emits
   `count−1` WARN lines per `capabilities()` call — and `capabilities()` is probed once per
   device by `Resources::build` (`resources.cpp:92`), so a G-GPU budget box logs the same
   "no peer access" degrade **G·(count−1)** times per build, not once.
3. `p2p_combine.cu:265` WARNs `cudaErrorPeerAccessAlreadyEnabled` once **per peer partial
   per combine call** — repeated every combine.

So the degrade is logged **O(peers × probes)** times, not once. Why it matters: §11.4 wants
"an explicit logged fallback" (singular, a tag), and §10 forbids log spam; the comment
promises "exactly ONE" and the code delivers a multiplicity that grows with G and call
frequency. Same family as the prior `check.cuh` review's C-1 contract overstatement.

Concrete fix — pick one:
- **(a) Honest wording (minimal):** "logs one STEPPE_LOG_WARN line" → "logs **one line per
  invocation**" everywhere (18–20, 118–119, 172–173). Cheapest, makes the claim true.
- **(b) Make "once" real (matches the sibling):** if the *intent* was process-once dedup,
  that belongs at the **call site** (like `warn_emulated_fp64_downgraded_once`), not in
  `cuda_warn` — a `static` flag inside `cuda_warn` would dedup *all* statuses across *all*
  sites globally, swallowing a second, different degrade (REJECTED below). So keep
  `cuda_warn` per-call + reword (a), and have the probe loop WARN once with a summary after
  the loop, or wrap it in a call-site once-guard mirroring the sibling.

Recommended: (a) in `check.cuh` + a note that per-degrade dedup is a call-site concern.
Severity MED (false documented contract on the milestone's marquee feature), effort S,
parity-safe. *Adversarial check:* could "one line" defensibly mean "per call"? In isolation
yes — but read against the sibling `…_once()` helper and §11.4's singular "an explicit
logged fallback," a maintainer reasonably reads it as dedup and is surprised by the G-scaled
spam. Confirmed worth fixing.

### (4) CUDA idioms / RAII / warning-cleanliness (§7) — the "warning-clean" focus finding

**W-1 [INFO / no action — verified CLEAN] — the WARN path is warning-clean under
`-Wextra -Werror`, and the two annotations that make it so are exactly correct.**
- `[[maybe_unused]]` on `expr` and `loc` (135–136): under `NDEBUG`, `STEPPE_LOG_WARN(...)`
  → `((void)0)` and **does not evaluate its arguments** (`log.hpp:45`; matches
  `assert`/`STEPPE_DEBUG_ONLY` semantics), so in release `expr`/`loc` are referenced by
  nothing inside the `if` body and `-Wunused-parameter` (under `-Wextra`) would fire. The
  `[[maybe_unused]]` is the correct minimal suppression, and the comment (130–133) explains
  exactly this. Note `status` stays used (`if`/`return`), correctly *not* annotated. Verified.
- `static_cast<unsigned>(loc.line())` (144): `std::source_location::line()` returns
  `uint_least32_t` (verified, cppreference `<source_location>`: `constexpr uint_least32_t
  line() const noexcept`), and `%u` expects `unsigned int`. Where `uint_least32_t` is wider
  than `unsigned int` the cast is the right `-Wformat` match; where equal it is a no-op.
  **Correct.** (Contrast `CudaError`, which uses `std::to_string(loc.line())` and sidesteps
  the specifier — a D-1-adjacent inconsistency, but each is correct in its own idiom.)
No warning-cleanliness defect. This sub-question is a clean PASS.

**W-2 [LOW, effort S, PARITY-SAFE: yes] — `%u` for the line number is a
narrowing-of-convenience; `PRIuLEAST32`/`std::to_string` would be width-exact, but
`unsigned` is fine for a line number.** A source line cannot realistically exceed
`UINT_MAX`, so the cast cannot lose information in practice; the pedantically-exact form
would be `<cinttypes>` `PRIuLEAST32` or `std::to_string` (as `CudaError` does). Not worth
the include. Noted only because the two siblings format the *same field* two different ways
(folds into D-1). Severity LOW.

**A-1 [LOW, effort S, PARITY-SAFE: yes] — NEW: the `STEPPE_CUDA_WARN` macro inherits the
non-variadic shape the prior `check.cuh` review flagged as S-1, so the same latent
top-level-comma fragility now applies to the M4.5 macro too.**
Location: macro `STEPPE_CUDA_WARN(expr)`, lines 181–182.
The prior `check.cuh` review's S-1 noted the expression-checker macros take a fixed single
`expr` parameter rather than variadic `(...)`/`#__VA_ARGS__`. **Not a bug today** (commas
inside a function-call's parens are one macro argument; only a *top-level* comma would
break, and a single function-call never has one). But the M4.5 delta added
`STEPPE_CUDA_WARN(expr)` with the identical non-variadic shape, so whatever consistency fix
S-1 recommends for `STEPPE_CUDA_CHECK`/`CUBLAS_CHECK` should land on `STEPPE_CUDA_WARN` in
the same edit — the prior pass's S-1 listed two macros and the delta silently added a
third. Severity LOW, effort S; bundle with the S-1 hardening.

### (3) Numerical / precision (§12)

**N/A — and verified parity-neutral.** `cuda_warn` performs no arithmetic and is on no
statistic path. Its three call sites are `cudaDeviceCanAccessPeer` /
`cudaDeviceEnablePeerAccess` / `cudaSetDevice` — pure capability/observability levers §11.4
/ §12 designate parity-neutral (host-staged and P2P combines are bit-identical regardless of
what the probe found). The status it yields *selects* a combine tier
(`f2_blocks_multigpu.cpp:172,184,201`), but **both tiers are proven bit-identical** (parity
gate met per the brief), so the WARN path cannot change a reported number. Explicitly N/A,
verified against §12.

### (5) Magic numbers (§4)

**Clean for the delta** — `cuda_warn` introduces no literals; `%u`/`%s` are conversion
specifiers, not tunables. The *one* literal the D-1 fix would introduce (the shared format
string) must be a named `inline constexpr const char*` (e.g. `kCudaErrorFmt`) per §4 /
ROADMAP §4, not a bare repeated literal — flagged so D-1's fix lands it named.

### (7) Readability, naming, const-correctness, [[nodiscard]]/noexcept, comment density

Strengths (verified): `[[nodiscard]]` on `cuda_warn` is the correct, load-bearing contract
(the return is the product) and is honored at the branching site; the comment at 130–133
explaining the `[[maybe_unused]]`/`NDEBUG` interaction is genuinely useful; the doc density
is high and arch-citing, consistent with the file.

**R-1 [LOW, effort S, PARITY-SAFE: yes] — `cuda_warn` is `inline` but not `noexcept`, while
it is called from a `noexcept` destructor (`p2p_combine.cu:165`).**
`cuda_warn` cannot throw: its only operations are an integer compare, the const-char runtime
string lookups (`noexcept` C functions), and `STEPPE_LOG_WARN`'s `fprintf` (which
`log.hpp:37–38` documents as non-throwing and "safe to call from the `noexcept` RAII
destructors"). So it is *de facto* `noexcept`. Marking it explicitly (a) documents
destructor-safety that `~DeviceGuard()` relies on, and (b) lets
`bugprone-exception-escape`/`cppcoreguidelines` verify the dtor stays non-throwing through
this call. Contrast `cuda_check`/`cublas_check`, which **must not** be `noexcept` (they
throw by design) — so `noexcept` on `cuda_warn` cleanly distinguishes the non-throwing
sibling at the signature level. Severity LOW, effort S. *Adversarial check:* could the
`fprintf` chain throw? `std::fprintf` is `noexcept`-in-practice (C function); the `do/while`
is plain. Safe.

**A-2 [LOW, effort S, PARITY-SAFE: yes] — NEW (split from E-1): there is no named
deliberate-discard helper, so the two intentional discards read as `(void)`-cast
`[[nodiscard]]` suppressions rather than as intent.**
Location: `p2p_combine.cu:165` and `:265`. Verified no such helper exists (grep:
`cuda_warn_discard`/`STEPPE_CUDA_WARN_VOID`/`warn_discard` → none). The `[[nodiscard]]` is
correct and the `(void)` discards are legitimate at these two sites (a dtor cannot act on
the status; the peer-enable's tolerated status is handled by the sticky-clear), but a
one-word named helper — `STEPPE_CUDA_WARN_VOID(expr)` (warn, deliberately discard) or, even
better, `STEPPE_CUDA_WARN_CLEAR(expr)` (warn + `cudaGetLastError()` clear, which *also*
fixes N-1's footgun at the side-effecting site) — makes the intent self-documenting and
removes the `(void)` noise the prior `check.cuh` review's B-2 flagged for the throwing path.
Severity LOW, effort S; pairs with N-1 (the `_CLEAR` variant subsumes the manual clear).

**R-2 [LOW, effort S, PARITY-SAFE: yes] — doc redundancy: the WARN behavior is described in
THREE places (file-header 18–24, function block 117–133, macro block 171–182) with
overlapping content that can drift.** The three blocks restate "logs one line, yields the
status, NDEBUG-gated, capability degrades, not the statistic path" — and it is *where* the
D-2 "one line" claim is repeated three times, so a fix must touch all three (and the N-1
caveat must be added consistently). The prior review's "dense comments" strength taken
slightly too far for a 16-line function. Fix: keep the macro-site usage doc (171–182)
authoritative, trim the function-internal block to the `[[maybe_unused]]`/`NDEBUG` mechanics
it uniquely explains, and let the file header give the one-sentence summary. Severity LOW.

### (9) Layering / API / ABI (§4)

**Clean.** `cuda_warn` stays in `::steppe::device::detail`; the macro `STEPPE_CUDA_WARN` is
global-scope like its siblings; the only new dependency (`STEPPE_LOG_WARN` from
`core/internal/log.hpp`) is the §8/§10-sanctioned CUDA-free `core_internal` INTERFACE target
consumed by the device layer — no CUDA header leaks into `core`, no layering inversion. The
non-throwing status is internal; it never crosses the public C ABI (it selects an internal
combine tier). The combine *policy* (which tier) lives correctly in
`core/fstats/f2_blocks_multigpu.cpp` (§4 dependency direction), reading the parity-neutral
`can_access_peer` POD field; `check.cuh` provides only the *mechanism* (the non-throwing
checker). Consistent with §4/§16. No finding.

### (10) Testability (§13)

**T-OK [no action — a point in favor]** — the prior pass's T-1 (no direct test; the .cu test
shadowed the production macro) is **closed**: `tests/reference/test_cuda_check.cu` directly
includes the production `check.cuh` (no shadow) and pins the four WARN/CHECK properties as
values. One gap remains:

**T-1 [LOW, effort S, PARITY-SAFE: yes] — the test does not assert the WARN line is
*emitted* (only that the status is yielded and nothing is thrown), nor the per-call count
(which would lock D-2's wording), nor the sticky-error contract (N-1).**
`test_cuda_check.cu` checks control flow (no throw, correct return) but not the observable:
that a debug build writes exactly one `[steppe][warn]` line on a non-success status and
**nothing** on `cudaSuccess`. The "quiet pass-through" claim (test comment :29) is asserted
by construction, not observed. Capturing stderr and asserting the line count would also be
the regression guard for the D-2 "log once per call" wording and the D-1 shared-format fix.
Separately, the N-1 sticky-error contract is untestable as-is (no assertion that a WARN'd
non-success leaves `cudaGetLastError() != cudaSuccess` and that a pure-query WARN does not) —
a tiny addition would pin the very contract N-1 wants documented. Fix: redirect stderr
(`freopen`/a pipe) around a debug-build WARN, assert one line on failure / zero on success;
add a `cudaSetDevice(bad)` WARN + `cudaGetLastError()` assertion to pin the sticky contract.
Severity LOW (the value path is covered), effort S.

### (11) Capability-tier coherence

**Coherent — verified.** The probe (`cuda_backend.cu:523`) yields the status, leaves
`access = 0` on a non-success, and the tag lands in `caps.can_access_peer` (a bool on the
`BackendCapabilities` POD) — the capability tag is recorded on the **POD payload, off the
numeric/statistic payload** (the §11.4 / workflow `wxz1fiiln` "tag off the statistic"
rule). The combine-tier selection reads `can_access_peer`
(`f2_blocks_multigpu.cpp:172,196`), and both tiers are bit-identical. The capability-tier
blemishes are D-2 (the degrade is *logged* more than the "once" the comment promises) and
N-1 (the side-effecting peer-enable's sticky-error caveat is undocumented) — the *tag* and
the tier-selection are correctly placed and parity-neutral. No further finding.

---

## Considered & rejected (incl. rejected-for-parity)

- **"`cuda_warn` should `static`-dedup its log so D-2's 'once' is true in code."** REJECTED.
  A `static std::atomic_flag` *inside* `cuda_warn` would dedup across **all** statuses and
  **all** call sites globally — a second, *different* degrade (e.g. a real
  `cudaErrorInvalidDevice` after a benign `cudaErrorPeerAccessAlreadyEnabled`) would be
  silently swallowed. Per-degrade dedup is a **call-site** concern (as
  `warn_emulated_fp64_downgraded_once` correctly implements). The checker must stay
  stateless. D-2's fix is honest wording + optional call-site once-guard.
- **"`cuda_warn` should call `cudaGetLastError()` itself to clear the sticky error (fix N-1
  in code)."** REJECTED as the default. A checker that silently consumes the sticky error
  would *mask* a real, unrelated prior fault that happened to be pending — wrong for a
  generic checker. N-1's fix is to **document** the caveat at the home and offer an *opt-in*
  `STEPPE_CUDA_WARN_CLEAR` for the tolerate-and-continue pattern; the bare
  `STEPPE_CUDA_WARN` must keep its read-only contract.
- **"`cuda_warn` should reuse `CudaError::what()` for its message (full DRY)."** REJECTED.
  Constructing a `CudaError` to borrow `what()` invokes `std::string` concat that allocates
  and can throw `std::bad_alloc` — wrong for a non-throwing, release-silent path. The
  correct DRY fix (D-1) is a shared `noexcept` `snprintf` helper / format literal.
- **"The `:265` discard swallows a real fault that re-surfaces as a confusing downstream
  error (prior pass's MED framing)."** PARTIALLY REJECTED → DOWNGRADED to LOW (E-1). The
  site clears the sticky error (:278) and fails-fast at the throwing `cudaMemcpyPeer`
  (:297–300); under the caller's pre-verified `can_access_peer` contract a residual
  `cudaErrorInvalidDevice` is a contract bug, not a lost fault. The merit is diagnostic
  locality (point at the cause, not the DMA) — LOW, a debug `STEPPE_ASSERT`, not a MED
  swallow.
- **"The `(void)STEPPE_CUDA_WARN(...)` in `~DeviceGuard()` is a `[[nodiscard]]` violation
  that should branch."** REJECTED for that site. A destructor cannot act on a failed
  `cudaSetDevice` and must stay `noexcept`; the WARN line is the diagnostic. (A named
  discard helper, A-2, makes the deliberate discard read as intent.)
- **"`cuda_warn` should throw on a status NOT in the expected-degrade set (be a hybrid)."**
  REJECTED-FOR-DESIGN. The point is a *non-throwing* sibling; the throwing decision belongs
  at the call site (E-1's branch), not inside a generic checker that would need each site's
  expected-set.
- **"Moving/summarizing the per-peer WARN changes behavior / parity."** REJECTED-as-not-a-
  parity-concern. The WARN line is pure observability (§12 parity-neutral); de-duping or
  summarizing changes only stderr text, never a number or the g=0..G-1 combine order. So
  D-2's call-site fix is parity-safe.
- **"`STEPPE_CUDA_WARN`'s `%u` line-number cast hides a width bug."** REJECTED. A source line
  number fits `unsigned` with astronomical margin; the cast is a `-Wformat` fix matching `%u`
  to `uint_least32_t` (verified), not a lossy narrowing of real data. (W-2 keeps it a LOW
  consistency nit.)
- **"`cuda_warn` not being `[[unlikely]]`/`[[gnu::cold]]` is a perf regression on the
  capability path."** REJECTED as material. Off every hot loop; the release failure branch
  is also just `if + return` (the log compiles out). P-1 keeps `[[unlikely]]` as a LOW
  intent-documentation nit.
- **"The WARN path could break §12 by logging from a parity-critical reduction."** REJECTED.
  Verified all three sites are capability/teardown, none on the `f2_blocks`/combine
  arithmetic; the combine *order* (g=0..G−1) and bytes are untouched by any WARN. Parity
  intact.

---

## What it takes to reach 10/10

In priority order (the first three are the substantive M4.5-delta items):

1. **Document the sticky-last-error caveat at the home (N-1, NEW)** — add to `cuda_warn`'s
   doc that it reads-but-does-not-consume the sticky error, so a tolerated non-success from
   a *side-effecting* call must be followed by `(void)cudaGetLastError()` before the next
   post-launch check (point at `p2p_combine.cu`). Optionally add a `STEPPE_CUDA_WARN_CLEAR`
   convenience macro (warn + clear) so the call site stops hand-rolling the 11-line
   workaround. This is the most important fix — it closes a demonstrated footgun the
   milestone already tripped over.
2. **Kill the message-format duplication (D-1)** — one `noexcept` `snprintf` format helper
   (or at minimum a named `constexpr kCudaErrorFmt` literal) consumed by both
   `CudaError`/`CublasError` and `cuda_warn`, so the "file:line (func): 'expr' -> name:
   string" shape has exactly one home; also unblocks the prior B-1 fixed-buffer
   `CudaError` without leaving the WARN copy stale. The §2-DRY-in-the-DRY-file fix.
3. **Make the "log once" claim honest (D-2)** — reword "logs one STEPPE_LOG_WARN line" →
   "logs **one line per invocation**" in all three doc blocks (18–20, 118–119, 172–173), and
   note per-degrade dedup is a call-site concern (the `warn_emulated_fp64_downgraded_once`
   pattern). Optionally summarize the per-peer probe WARN into one line after the loop in
   `cuda_backend.cu` so a no-peer budget box does not emit G·(count−1) identical lines.
4. **Mark `cuda_warn` `noexcept` (R-1)** — documents and verifies the destructor-safety
   `~DeviceGuard()` relies on, and signature-level-distinguishes the non-throwing sibling
   from the throwing `cuda_check`/`cublas_check`.
5. **Name the deliberate discard / sharpen the `:265` diagnostic (A-2 + E-1)** — a
   `STEPPE_CUDA_WARN_VOID`/`_CLEAR` macro so the two `(void)` sites read as intent, and a
   debug `STEPPE_ASSERT` at `:265` so a `can_access_peer`-contract violation is pinpointed at
   the peer-enable, not the downstream DMA.
6. **Assert the WARN observable + sticky contract in the test (T-1)** — capture stderr,
   assert one line on failure / zero on success (locks D-1/D-2), and assert a side-effecting
   WARN leaves the sticky error while a pure-query WARN does not (locks N-1).
7. **Document the recoverable-but-noteworthy vs expected-poll distinction (E-2)** — note
   `cuda_warn` is for *degrades worth a tag*, not high-frequency `cudaErrorNotReady` pollers
   (which need a silent non-logging predicate at M5).
8. **Minor polish:** `[[unlikely]]` on the failure branch (P-1, paired with the same on
   `cuda_check`/`cublas_check`); make `STEPPE_CUDA_WARN` variadic with the other macros
   (A-1, fold into the prior S-1); trim the triple-documented WARN behavior to one
   authoritative block (R-2); one-line `expr`-is-`#expr`-never-null invariant note (E-3);
   width-exact line formatting folded into D-1 (W-2).

## Good patterns to keep

- **The non-throwing sibling exists at all, in the right place** — `STEPPE_CUDA_WARN` next to
  `STEPPE_CUDA_CHECK` is exactly the §11.4 "explicit logged fallback" mechanism and what the
  prior review's CAP-1/CAP-2 recommended. The fault-vs-recoverable split is the correct
  architecture. Preserve.
- **`[[nodiscard]]` on `cuda_warn`** — the return *is* the product; the attribute enforces
  the branch and `cuda_backend.cu:522` honors it. Textbook.
- **Build-mode-independent status return, NDEBUG-only log** — `return status;` sits outside
  the `if`, so the value is identical in debug and release while only the diagnostic line is
  release-silent. The test pins it. Preserve exactly.
- **Warning-cleanliness done right** — `[[maybe_unused]]` on the release-unused `expr`/`loc`
  + `static_cast<unsigned>` to match `%u`/`uint_least32_t`: the minimal, correct two
  annotations for `-Wextra -Werror`, each documented. Keep.
- **Routing through the single `STEPPE_LOG_WARN` sink** (§10, no bare `printf`) — keeps the
  sink/level/async policy swappable. Keep.
- **A dedicated, GPU-free, no-shadow unit test for the delta** (`test_cuda_check.cu`) pinning
  WARN-vs-CHECK as distinct paths — closes the prior shadow gap. Keep and extend with the
  stderr-observable + sticky-contract assertions (T-1).
- **The capability tag lands on the `BackendCapabilities` POD, off the numeric payload** —
  the §11.4 / `wxz1fiiln` "tag off the statistic" rule, correctly observed. Keep.
- **The combine policy lives in `core/fstats/f2_blocks_multigpu.cpp`, not in `check.cuh`** —
  the header provides only the non-throwing *mechanism*; the tier *decision* reads the
  parity-neutral POD field in the right layer (§4). Clean separation. Keep.
