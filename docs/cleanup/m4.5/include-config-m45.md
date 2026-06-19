# M4.5 unit review (ADVERSARIAL 2nd pass) — `include/steppe/config.hpp`

The M4.5 delta: `prefer_p2p_combine` + the MULTI-GPU OVERRIDE-KNOB BANNER. Knob-type separation
(override-intent vs discovered), defaults, doc accuracy, CUDA-free.

Reviewer: adversarial second pass over the M4.5 multi-GPU delta to the public config contract. Every
first-pass finding was re-verified against the actual source and the official CUDA docs; the
load-bearing C-1 claim was traced all the way into the `.cu` definition; two new findings were added
and two first-pass claims were corrected (counts / "no tracking home").

**Scope is narrow and deliberate**: the M4.5 additions to `include/steppe/config.hpp` — the new
`prefer_p2p_combine` field (lines 262–287), the new banner (lines 229–253), and the M4.5-era touches
to the adjacent fields it interacts with (`enable_peer_access` 255–260, `deterministic` 292–310,
`devices` 202–209). The pre-M4.5 body (named constants, `Precision`, `FilterConfig`) was reviewed in
`docs/cleanup/include-config.md` (9.0/10) and is **not re-litigated** here except where an M4.5 change
touches it or a prior finding is now closed by M4.5 (those closures are noted so the score is honest),
plus two pre-M4.5 *in-file* inconsistencies I surfaced while verifying and record (out of delta scope).

Files read **fully** this pass (re-verified, not trusted from the first draft): the target
`include/steppe/config.hpp` (391 lines); `tests/unit/test_config.cpp` (the static-assert contract gate);
`src/core/fstats/f2_blocks_multigpu.cpp` (the **sole runtime consumer** of `prefer_p2p_combine`, the §4
combine gate at 156–202); `src/device/p2p_combine.hpp` (the device-resident combine the knob selects);
`src/device/cuda/p2p_combine.cu` (the *definition* — line 265, the actual `cudaDeviceEnablePeerAccess`
call, which is what makes C-1 a real veto violation and not merely a doc nit); `src/device/resources.hpp`
(the DISCOVERED tag `last_combine_path`/`CombinePath`); `src/device/backend.hpp`
(`BackendCapabilities::can_access_peer`); `src/device/vram_budget.hpp` (the closed `kMaxVramUtilizationFraction`);
`tests/reference/test_f2_multigpu_parity.cu` (the locked parity test that flips the knob). Cross-checked
`architecture.md` §9/§11.4/§12, `ROADMAP.md` §M4.5, `docs/cleanup/00-overview.md` §(2).2/§(2).3.

Standard judged against: `architecture.md` §2 (DRY/single-source, separation, fail-fast,
no-global-mutable-state, typed-immutable-config), §4 (layering app→api→core→device; CUDA PRIVATE to
steppe_device; config is the `steppe_api` payload), §7 (CUDA idioms — mostly N/A in a CUDA-free header),
§9 (typed immutable config / injected Resources; the `DeviceConfig` field listing + `build()` validation
list), §11.4 (SPMG tile/block sharding + host-side fixed-order combine + capability-tiered
`cudaMemcpyPeer` combine; NEVER NCCL AllReduce), §12 (the bit-identity parity law). CUDA host-API claims
were looked up and are cited inline.

## Role & layering

`config.hpp` is the top of the dependency DAG and the principal payload of the `steppe_api` INTERFACE
target (§4). The M4.5 delta does not change that role: `prefer_p2p_combine` is a plain `bool`, the banner
is comment-only, and the file remains CUDA-free (includes only `<cstddef>/<string>/<vector>`, re-verified —
no `<cuda_runtime.h>`, no CUDA type leaked by the new material). The §4 promise that CUDA is
PRIVATE-to-device therefore still holds end-to-end: the new knob is consumed by
`core/fstats/f2_blocks_multigpu.cpp` (a `core` TU) as a plain `bool` field of `resources.config`, with no
reverse dependency, and the device-resident combine it gates (`p2p_combine.cu`) reads only the *resolved*
path, never this header's enum.

**The intent/discovery split this delta encodes is the architecturally correct one, and it is realized
correctly across the unit boundary** — I re-verified the discovered half against the real layout:
*override-intent* (`devices`, `enable_peer_access`, `prefer_p2p_combine`, future `enable_gds_ingest`)
lives here on `DeviceConfig`; *discovered capability* (`can_access_peer` on `backend.hpp:175`, free/total
VRAM, emulated-FP64-honorable) and the *which-path tag* (`CombinePath::None/HostStaged/P2pDeviceResident`
on `resources.hpp:46–59`, the field `Resources::last_combine_path` `resources.hpp:110`, mutated only by
the entry point at `f2_blocks_multigpu.cpp:183,200`) live on `Resources`, never on `DeviceConfig` and
never on the numeric `F2BlockTensor`. The banner is **not aspirational** — it describes the real layout.
This delta **closes prior `include-config.md` finding 11.1** ("the natural home for the capability-tier
override knobs; today it has only `enable_peer_access`") and **closes 9.1** (`deterministic` is now
present, 292–310). Those closures justify the high score; the residue below is doc-accuracy plus one real
code/doc contradiction (C-1) that I traced to a genuine runtime veto violation in the `.cu`.

The new material is small but **load-bearing**: it is the public, ABI-facing expression of the M4.5
capability tier, the single place a user forces the host-staged parity baseline, and the contract the
locked parity test pins (`test_f2_multigpu_parity.cu:338,343` flips exactly this field). Held to the
interface-design + doc-accuracy bar (the file has no executable logic to be algorithmically wrong).

## Score: 9.5/10 — the intent/discovery split is textbook and the parity-neutrality reasoning is vendor-accurate; held off 10 by one real, now-confirmed-runtime doc/code veto contradiction (C-1) plus doc-completeness and a spec-listing drift

Rationale: this is the cleanest part of the M4.5 set on the dimension that matters most for a public
config header — it gets the *taxonomy* right (intent here, discovery on `Resources`, tag off the numeric
payload), correctly and explicitly marks the new lever parity-NEUTRAL with the right §12 reasoning (the
transport only moves bytes; software fixes the `g=0..G-1` order; never NCCL AllReduce), and the
`cudaMemcpyPeer`-is-byte-exact claim it rests on is vendor-accurate (verified below against the CUDA
Runtime API docs). The real deductions are unchanged in substance from the first pass but I have
*strengthened C-1*: the `prefer_p2p_combine` doc (278–282) says the gate prefers P2P "when
`enable_peer_access` is set AND the runtime `cudaDeviceCanAccessPeer` probe succeeds", but the only
runtime gate (`f2_blocks_multigpu.cpp:171–172`) reads `prefer_p2p_combine && can_access_peer` and **never
reads `enable_peer_access`** — and worse than a mere over-specification, the P2P path it admits then
*unconditionally calls* `cudaDeviceEnablePeerAccess` (`p2p_combine.cu:265`), which is the *exact*
operation `enable_peer_access=false` is documented two fields up to forbid ("whether the backend is
permitted to call `cudaDeviceEnablePeerAccess` at all"). So the header does not merely over-promise a
precondition — it documents a veto the code actively violates. Score held at 9.5 (the contradiction is a
MED public-contract/runtime mismatch on a parity-adjacent knob; it cannot regress parity because both
combine paths are bit-identical, but it is a real lie the header tells about runtime behavior). The
remaining half-point is doc-completeness (R-1 redundancy, DRY restatement count, Perf-2 data-bounce
clause) and a spec-drift I newly surfaced (Spec-1: the canonical §9 `DeviceConfig` listing was never
updated to include `prefer_p2p_combine`).

---

## Findings

### Performance (FIRST-CLASS this pass)

The M4.5 delta to this file is **pure declaration** — one `bool` field and comments. There are no
kernels, copies, streams, syncs, allocations, casts, or grid math *in this file* to optimize, so the
classic perf hunt (data bouncing / grid-stride / sequential-P2P / default-stream / hot-path alloc /
casting noise) is **N/A inside `config.hpp`**. What this file *can* do for performance is shape the
override surface so the downstream P2P/host-staged perf decisions are expressible and not foot-gunned.
I hunted the perf-shaping axis hard:

**Perf-1 [INFORMATIONAL — confirmation, not a defect] `prefer_p2p_combine` is a zero-cost `bool` and the
knob it gates is correctly OFF the bandwidth critical path.** The doc (273–276) correctly states the
combine traffic is kB–MB and off the critical path, matching architecture.md §11.4 verbatim ("Because
this cross-GPU traffic is kB–MB and off the critical path, it is architectural cleanliness, not a
throughput lever — steppe is deliberately P2P/NVLink-insensitive", arch:717). The knob is a *cleanliness /
data-movement* lever, not a throughput lever, and the doc says exactly that ("a data-movement/cleanliness
lever, never a reported number", 276). Correct framing; do not let a future edit re-sell
`prefer_p2p_combine` as a speed knob. **No action.**

**Perf-2 [LOW — doc accuracy touching the perf decision] The doc does not name the actual data-movement
asymmetry — the P2P path elides the host round-trip the host-staged baseline pays.** Location: 262–287.
The doc explains *what* the P2P path does ("GPU 0 pulls each peer's partial via a byte-exact DMA and sums
… on-device") and that it is parity-neutral, but frames the choice purely as cleanliness. The genuine
asymmetry (which I confirmed against `combine_f2_partials_host` vs `combine_f2_partials_p2p`): the
host-staged baseline performs G device→host gathers of each compact partial + a host-side FP64 placement
sum + (in the broader §11.4 vision) a broadcast, whereas the P2P path keeps everything device-resident
(one `cudaMemcpyPeer` per non-root partial + an on-device placement-add). On the budget 5090 this is moot
(P2P driver-disabled, host-staged is the only path); on the capable PRO-6000 tier the P2P path *does*
avoid a host round-trip of the `2·P²·B` doubles. *Why it matters:* the prompt flags data bouncing as
first-class; the public contract that gates the choice should name the asymmetry so a reader understands
why the capable tier prefers it — it is a *latency/host-traffic* win on the combine even though the
combine is off the *streaming* critical path. *Concrete fix:* one clause on the `prefer_p2p_combine` doc:
"the device-resident path elides the G device→host partial gathers and the host-side FP64 placement sum
the host-staged baseline pays — a host-round-trip (data-bounce) saving on the combine, parity-neutral
because both sum the same bytes in the same order." Severity: low. Effort: S. PARITY-SAFE: yes (doc only).

### Correctness & bugs

**C-1 [MED — the one real defect in the delta; STRENGTHENED vs first pass to a confirmed runtime veto
violation] The `prefer_p2p_combine` doc says the gate reads `enable_peer_access`; the runtime gate does
NOT — and the admitted P2P path then unconditionally calls the very API `enable_peer_access=false` is
documented to forbid.** Location: 278–282 (the "Default `true` (capable-first)" paragraph), falsified by
the sole consumer `f2_blocks_multigpu.cpp:171–172` and `src/device/cuda/p2p_combine.cu:265`.

The header states verbatim (278–282): *"Default `true` (capable-first): prefer the device-resident
combine when `enable_peer_access` is set AND the runtime `cudaDeviceCanAccessPeer` probe succeeds;
otherwise the backend DEGRADES."* But the runtime gate is two-term:
```cpp
// f2_blocks_multigpu.cpp:171-172
const bool use_p2p =
    resources.config.prefer_p2p_combine && resources.gpus[0].caps.can_access_peer;
```
— it reads `prefer_p2p_combine` and the *discovered* `can_access_peer`, and **never reads
`enable_peer_access`.** I confirmed by `grep` that **no production code reads `DeviceConfig::enable_peer_access`
at all** — every hit outside this header is either a doc comment (`resources.hpp:101`, `backend.hpp:168`)
or a default-pinning `static_assert` in `test_config.cpp:86`. The field is a documented MAY-WE veto wired
to nothing.

This is worse than the first pass framed it ("over-specifies a precondition the code does not enforce").
The P2P path the two-term gate admits **then calls the forbidden API**:
```cpp
// src/device/cuda/p2p_combine.cu:265
(void)STEPPE_CUDA_WARN(cudaDeviceEnablePeerAccess(owning_device, 0));
```
So a user who sets `enable_peer_access = false` (the documented MAY-WE veto: "whether the backend is
permitted to call `cudaDeviceEnablePeerAccess` at all", 255–258) but leaves `prefer_p2p_combine = true`
will pass the gate, enter `combine_f2_partials_p2p`, and **call `cudaDeviceEnablePeerAccess` — the exact
operation the veto exists to forbid.** The two override knobs are documented as cleanly distinct (255–287),
but the code both (a) ignores the permission knob in the gate and (b) actively violates it in the path it
admits. *Why it matters:* §2 fail-fast / §9 typed contract — a public header is a *promise*; promising a
precondition (`enable_peer_access` gates the path) the code not only fails to honor but actively
contradicts is worse than omitting it, because the next engineer reasons from the contract. This is the
config-doc face of `device-resources` finding K2; that review put the *code* fix in the consumer.

*Adversarial check — is the doc right and the code the bug, or vice-versa?* The banner (255–260) is
explicit that `enable_peer_access` is the permission to call `cudaDeviceEnablePeerAccess` "at all", and
the P2P combine *does* call it — so a `false` permission with the P2P path taken is a genuine
contradiction the design intends to forbid. The *intended* design is the three-term gate; the code is the
latent defect and the doc states the intent. But as it stands **today the header is inaccurate about
current runtime behavior**, which is the defect in *this* unit. *Concrete fix (this file):* if the
consumer is not changed, soften the doc to "prefer the device-resident combine when the runtime
`cudaDeviceCanAccessPeer` probe succeeds" and add "NOTE: `enable_peer_access` is the separate MAY-WE veto;
the current combine gate (`f2_blocks_multigpu.cpp`) does NOT yet AND it into the path decision, and the
P2P path calls `cudaDeviceEnablePeerAccess` regardless — see resources K2." If the consumer IS changed
(recommended, and the right fix): widen the gate to `prefer_p2p_combine && enable_peer_access &&
can_access_peer`, after which this header's doc is correct as-written. The two MUST agree. Severity: med.
Effort: S. PARITY-SAFE: **yes** — both combine paths are bit-identical, so AND-ing in `enable_peer_access`
only changes *which transport*, never a reported number (§12); the fix cannot regress parity either way.

**C-2 [LOW] The doc implies a `G >= 2` gate component that this header's field cannot express and that the
code enforces STRUCTURALLY, not in the gate.** Location: implied by 268–272 ("GPU 0 pulls each peer's
partial … and sums in the fixed `g = 0..G-1` order"); the sibling docs `resources.hpp:41,57` and
`p2p_combine.hpp:66` state the gate as `… && G >= 2`. In the actual consumer the `G >= 2` condition is
NOT in the `use_p2p` expression (171–172) — it is guaranteed because the `G == 1` fast path already
returned at `f2_blocks_multigpu.cpp:88–91` before the gate runs. So "`G >= 2`" is a *structural*
precondition, not a gate term. *Why it matters:* minor, but the §11.4 gate is documented as a three-part
predicate in two sibling docs while the *code* gate is two-part with G handled upstream — a reader
reconciling the copies (see DRY-1) could think the gate is missing a check. `config.hpp` does **not** state
the predicate as such (it is the cleanest of the homes), so this is a note that `config.hpp` should NOT
acquire the over-stated three-part form when it gets the cross-ref tidy. Severity: low. Effort: S.
PARITY-SAFE: yes (doc only). *Adversarial:* not a defect in `config.hpp` today — flagged so a "make
config.hpp restate the full gate" edit does not import the inaccuracy.

**C-3 [N/A — verified correct against the official docs] The parity-neutrality claim and its
`cudaMemcpyPeer`-is-byte-exact premise are vendor-accurate.** Lines 272–276 assert both combine paths
"sum in the SAME fixed device order and are therefore BIT-IDENTICAL … the transport only moves bytes;
software fixes the order; NEVER NCCL AllReduce." I verified the load-bearing premise against the CUDA
Runtime API docs: `cudaMemcpyPeer`/`cudaMemcpyPeerAsync` are documented as "Copies memory between two
devices" with **no arithmetic, no re-rounding, no data transformation** — a raw byte transfer (CUDA
Runtime API — Memory Management group, `cudaMemcpyPeer`/`cudaMemcpyPeerAsync`). When peer access is not
enabled the copy still completes, staged through host (CUDA Programming Guide §3.4 "Programming Systems
with Multiple GPUs"; the `CUDA_ERROR_PEER_ACCESS_NOT_ENABLED` semantics confirm direct P2P requires an
explicit `cudaDeviceEnablePeerAccess`, otherwise data is staged via host). Because the copy is
bit-preserving and the *summation* order is software-fixed at `g=0..G-1` in both paths (host-staged in
`f2_combine.cpp`, device-resident in `p2p_combine.cu`), the two are bit-identical by construction — exactly
the §12 argument, and exactly what the locked parity test proved by `memcmp`
(`test_f2_multigpu_parity.cu:453`, the explicit two-tier-neutrality `bit_equal(candG2h, candG2p)` check).
The "NEVER NCCL AllReduce" caveat is correct (AllReduce order varies with G/buffer size, arch:91/§12).
**No defect; this is the single most important claim in the delta and it is correct.**

### Edge cases & failure modes

**E-1 [LOW] `prefer_p2p_combine` is an unguarded `bool` with no `build()` validation home — same latent
gap as every other field here (no `ConfigBuilder`/`RunConfig` exists yet).** Location: 287. A `bool` has
no out-of-range value, so this is benign for the field itself. But the doc promises a *degrade* behavior
("otherwise the backend DEGRADES — non-throwing and tagged — to the host-staged fixed-order combine")
realized in the *consumer*, not validated here. The architecture spec defines `ConfigBuilder::build()` as
the `std::expected<RunConfig, Error>` validation choke point (arch:592–598); it does not exist yet (M0–M4.5
stop at the device slice). The M4.5-relevant residue: §9's `build()` validation list does not yet mention
`prefer_p2p_combine`, and there is no enforced relationship between `prefer_p2p_combine` and
`enable_peer_access` anywhere (this is C-1's root — `build()` could normalize/WARN on the
`enable_peer_access=false, prefer_p2p_combine=true` contradiction, resolving C-1 at the validation layer).
*Concrete fix:* when `build()` lands, add `prefer_p2p_combine` to the §9 list with the cross-knob rule. For
now, a one-line doc note that the cross-knob interaction is `build()`-validated-TBD. Severity: low. Effort:
S. PARITY-SAFE: yes.

**E-2 [N/A — verified correct] Single-GPU / G==1 interaction with the knob is benign.** With
`devices.size() == 1` (or empty→auto-enumerate to one device) the consumer returns via the G==1 fast path
(`f2_blocks_multigpu.cpp:88–91`) **before** the gate ever reads `prefer_p2p_combine`, so the knob is inert
on single-GPU regardless of value — and `last_combine_path` is left at its value-initialized `None` on
that path (the `CombinePath::None` doc, `resources.hpp:47–49`, explicitly covers "the G==1 single-GPU fast
path"). The config doc correctly speaks only of the multi-GPU combine; the knob defaulting `true` is
harmless on a 1-GPU box. **No config.hpp defect.** Verified.

**E-3 [N/A — verified] The default `true` is safe on the budget 5090 (P2P driver-disabled).** On the 5090
`can_access_peer == false`, so `use_p2p == false` regardless of `prefer_p2p_combine`, and the code takes
the host-staged baseline AND emits the tagged WARN ("P2P combine unavailable (no peer access) ->
host-staged fixed-order combine", `f2_blocks_multigpu.cpp:197–198`) precisely because the user *preferred*
P2P. The doc's "capable-first … otherwise DEGRADES — non-throwing and tagged" (277–282) exactly matches
this realized behavior. Default-`true` is the correct capable-first stance and is non-throwing on the
budget box. **Verified correct — no defect.** (One staleness nit folded into Doc-2 below: the doc's
example for "peer access disabled" is "stock-driver GeForce", which on the *current* box reality IS the
2× 5090 — the example is accurate, not stale; the PRO-6000 capable box is spun down but the doc never
claims it is present.)

### Numerical / precision vs §12

**N-1 [N/A — and it is the right design] The knob is parity-neutral by construction and the doc says so
correctly.** `prefer_p2p_combine` selects a *transport* for an already-fixed-order sum; it changes no
arithmetic, no accumulation order, no precision. The doc states this in the strongest correct form
(272–276). This is the §12 line the entire parity push was protecting; the delta is on the right side of
it — it adds a lever that moves bytes, never one that reduces. The proven `memcmp` bit-identity across both
tiers (the locked gate, `test_f2_multigpu_parity.cu:453`) is the empirical confirmation that the doc claim
holds. **No precision defect; design correct, doc accurate.**

**N-2 [N/A — verified correct] The `deterministic` field (closed prior 9.1) correctly states the
multi-GPU combine rule in §12 terms.** Lines 292–310 add `deterministic = true` with a doc that includes
"the multi-GPU partials are combined in the fixed host-side device order pinned by `devices` (§11.4) rather
than a non-deterministic AllReduce." This is the field prior `include-config.md` 9.1 flagged as missing; it
now exists, is type/default-pinned by `test_config.cpp:41–44` (`static_assert` bool + default true), and
its doc correctly couples it to the M4.5 combine. **Closure verified — good.** Caveat under L-1: I
re-confirmed by `grep` that **no source reads `DeviceConfig::deterministic`** yet (the field is declared but
its documented §12 rules are not yet enforced anywhere; the only `deterministic` hits in source are
unrelated locals/comments).

### CUDA idioms / RAII / stream & async semantics vs §7

**CUDA-1 [N/A] No CUDA in the delta.** `prefer_p2p_combine` is a plain `bool`; the banner is prose. No
resources, streams, allocations, handles, or launch math are declarable or violatable in this CUDA-free
header. The async/stream semantics of the P2P combine the knob gates live in `p2p_combine.cu` and are that
unit's concern. The §7 idioms are upheld vacuously here. Re-verified the header includes no CUDA header and
leaks no CUDA type via the new material.

### Magic numbers & hardcoded values vs §4 / ROADMAP §4

**MN-1 [N/A — and a prior `high` is now CLOSED] No new magic numbers in the delta; M4.5 resolved the prior
`kMaxVramUtilizationFraction` defect.** The delta introduces a `bool` and comments — zero literals.
Separately re-verified that prior `include-config.md` finding 5.1 (the `high`: `0.80` was a live magic
literal in `cuda_backend.cu`) **is closed by M4.5**: `kMaxVramUtilizationFraction` exists (config.hpp
90–111), is consumed in `vram_budget.hpp`, and carries `static_assert(kMaxVramUtilizationFraction > 0.0 &&
kMaxVramUtilizationFraction <= 1.0)` (vram_budget.hpp:45–47, confirmed). That was the prior review's #1
"reach 10/10" item, done. Not a M4.5-delta finding, but it materially supports the higher score. **No
action — noted as a closure.**

### Decomposition / single-responsibility vs §2

**DSR-1 [N/A] `DeviceConfig` stays single-responsibility with the new field.** `prefer_p2p_combine` is
correctly a member of `DeviceConfig` (resources/device selection), not a new struct — it is an
override-intent lever exactly like its banner siblings, and grouping the P2P levers under one banner is the
right cohesion. No decomposition warranted; the field count is small and the banner makes the two-knob
structure legible. Verified placement against the banner's own taxonomy (234–243). Good.

### Readability, naming, const-correctness, comment density vs §2

**R-1 [LOW] The `prefer_p2p_combine` doc-comment is ~26 lines for one `bool`, and re-derives the
intent/discovery split the banner 20 lines above already states.** Location: 262–287 vs the banner 229–253.
The field comment repeats the `enable_peer_access` vs `prefer_p2p_combine` distinction (264–272) the banner
just made (255–260), and restates "DISCOVERED state recorded in `Resources`/result metadata, NOT here (see
the override-knob banner above)" (283–284) which the banner also says. High-fidelity but redundant: the
same WHY is told twice within 30 lines. *Why it matters:* §2 readability; the M4.5 delta crosses into
restating the adjacent banner. *Concrete fix:* trim the field comment to (a) the one-line semantics, (b)
the *corrected* gate (per C-1), (c) the default rationale, and a cross-ref "see the override-knob banner
for the intent/discovery split" instead of re-deriving it. Severity: low. Effort: S. PARITY-SAFE: yes.
*Adversarial:* defensible as "the field should be self-documenting at its definition" — but it currently
*over*-documents to the point of restating the banner, which is the smell, not the cure.

**R-2 [N/A — good] Naming and polarity of `prefer_p2p_combine` are correct.** `prefer_*` correctly signals
"a preference applied WHEN the capability is present" (not a hard force), distinguishing it from
`enable_peer_access` (a permission). The boolean reads correctly at the call site
(`config.prefer_p2p_combine && can_access_peer`). The banner's `enable_*` / `prefer_*` / `enable_gds_*`
naming is internally consistent (permission vs preference). Good — keep.

**R-3 [N/A — good] `deterministic`/`prefer_p2p_combine` defaults are pinned by `static_assert` in the
test.** `test_config.cpp:82–87` `static_assert`s both the *type* (`bool`) and the *default* (`true`) of
`prefer_p2p_combine` and `enable_peer_access`, and 41–44 does the same for `deterministic`. This is the
desync-prevention the prior review (finding 10.1) wanted, applied to the new fields. The header carries no
in-file `static_assert` (none needed for `bool`s), but the contract is gated. Good.

### Layering / API / ABI vs §4

**L-1 [LOW] The `deterministic` and `prefer_p2p_combine` fields are declared but the §9 `build()` that
would *enforce* their documented semantics does not exist — the contract is expressible but not yet
validated.** Location: 287, 292–310; `grep` confirms **no source reads `.deterministic`** and the only
reader of `prefer_p2p_combine` is the runtime gate (which, per C-1, reads it incompletely). Consistent with
the whole config surface (M0–M4.5 have no `ConfigBuilder`); the fields are correctly *intent* (so they
belong here, not on `Resources`). The M4.5-specific note: the banner says "Every lever is parity-NEUTRAL …
so the host-staged fixed-order combine and the device-resident P2P combine are bit-identical on both
capability tiers" (240–243) — a claim that is *true* (proven by the parity test) but currently rests on the
*consumer* getting the gate right (C-1) since `build()` cannot enforce it. *Why it matters:* §4/§9 — the
API surface is ahead of its validation; the banner's parity-neutral promise is upheld by the consumer code
and the proven test, not by this header's (absent) validator. *Concrete fix:* none in this file beyond the
C-1/E-1 doc tidies; the real enforcement is the deferred `build()`. Severity: low. Effort: deferred (build).
PARITY-SAFE: yes.

**L-2 [N/A — verified] CUDA-free layering upheld by the delta.** Re-verified end-to-end: the new `bool` and
banner add no include, no CUDA type, no macro; the consumer (`core` TU) reads it as plain C++; the
device-resident combine reads only the resolved path. The §4 compiler-enforced CUDA-PRIVATE property is
intact. Exemplary, unchanged by the delta.

**Spec-1 [LOW — NEW this pass; spec/implementation drift on the M4.5 delta] The canonical §9 `DeviceConfig`
listing in `architecture.md` (the documented single source of truth) was never updated to include
`prefer_p2p_combine`.** Location: the header field (config.hpp:287) vs `architecture.md:563–581`. The spec's
`DeviceConfig` struct listing (arch:563–581) enumerates `devices`, `precision`, `mantissa_bits`,
`stream_count`, `search_streams`, `use_mem_pool`, `enable_peer_access`, `deterministic` — but **not**
`prefer_p2p_combine`. The M4.5 work added a *public-ABI* field to the implementation header without
reflecting it in the architecture doc that MEMORY.md designates the canonical spec. *Why it matters:* §2/§4
single-source-of-truth — the spec listing is supposed to be the authoritative field set; a public field
present in the shipped header but absent from the spec is exactly the drift the "single source of truth"
discipline exists to prevent, and it is on the M4.5 delta (unlike the speedup-figure nits below, which are
pre-M4.5). *Concrete fix:* add a `prefer_p2p_combine = true` line to the §9 `DeviceConfig` listing with the
"WHICH-PATH, parity-neutral, §11.4" annotation, mirroring the `enable_peer_access` line above it. (Not a
defect in `config.hpp`'s own text — it is the spec that lags — but it is the M4.5 delta's footprint and a
9.5→10 item.) Severity: low. Effort: S. PARITY-SAFE: yes (doc only).

### Capability-tier coherence (the probe + tagged-degrade + which-path recording)

**K-1 [N/A — verified, the delta's strongest property] The intent/discovery boundary is drawn correctly
and is real across the unit boundary; the tag is OFF the numeric payload.** Re-verified against the actual
layout:
- **Override intent on `DeviceConfig`**: `prefer_p2p_combine` (and `enable_peer_access`, `deterministic`,
  `devices`) are here (229–311). Correct.
- **Discovered capability on `Resources`/caps**: `can_access_peer` is on `gpus[g].caps`
  (`BackendCapabilities`, `backend.hpp:175`, read at `f2_blocks_multigpu.cpp:172`); free/total VRAM and
  emulation-honorable are runtime-probed there (`backend.hpp:163–184`) — NOT on `DeviceConfig`. Correct.
- **Which-path tag off the numeric payload**: `Resources::last_combine_path` is a `CombinePath` enum on
  `Resources` (`resources.hpp:110`), set by the entry point (`f2_blocks_multigpu.cpp:183,200`), NEVER on
  the `F2BlockTensor` it returns. Correct, matches the banner (248–252).
- **Non-throwing tagged degrade**: confirmed — `can_access_peer == false` does not throw; it takes the
  host-staged path and WARN-logs only on a genuine degrade (`f2_blocks_multigpu.cpp:196–199`). Matches the
  banner (251–252).
This is the single best-executed piece of the M4.5 capability-tier design, and `config.hpp`'s banner is
the authoritative cross-reference for it. **No defect — a Good Pattern to Keep.**

**K-2 [LOW — = C-1 at the tier level] The `enable_peer_access` MAY-WE veto is documented as a tier knob but
the tier gate does not consult it (and the path it admits violates it).** Same code fact as C-1 through the
capability-tier lens: the banner (255–260) and the field doc present `enable_peer_access` as the permission
lever, but the combine-path tier decision (`use_p2p`) ignores it and `p2p_combine.cu:265` calls
`cudaDeviceEnablePeerAccess` regardless, so the "two distinct knob types" promise is, for the *permission*
knob, not realized. *Concrete fix:* same as C-1. Severity: low (it is C-1 again; not double-counted in the
score). PARITY-SAFE: yes.

### Testability vs §13

**T-1 [LOW — verified the gap is real] The delta is exercised by a host-only unit test AND the locked
parity test, but the C-1 contradiction is not covered by any test.** `test_config.cpp` (a pure `.cpp`, no
GPU — itself the §4-layering proof) pins `prefer_p2p_combine`'s type and default via `static_assert` and a
runtime case (82–94), and `test_f2_multigpu_parity.cu` flips the field to drive both tiers
(`cfgG2h.prefer_p2p_combine=false` forces host-staged at line 338; `cfgG2p.prefer_p2p_combine=true` selects
P2P at 343) and asserts `resG2h.last_combine_path == HostStaged` for the false case (478). So the knob's
*contract* (type/default) and its *behavioral effect* (which combine path + tag) are both tested. **Good —
the rare config field whose effect, not just default, is covered.** The gap: I `grep`-confirmed there is
**no** test case in the tree setting `enable_peer_access = false` (the string `enable_peer_access` does not
appear in `test_f2_multigpu_parity.cu` at all). A lane setting `enable_peer_access=false,
prefer_p2p_combine=true` and asserting the path would have surfaced C-1. *Minor fix:* add that negative
lane. Severity: low (folds into C-1). Effort: S.

### Documentation accuracy (pre-M4.5 body — recorded, OUT of delta scope)

**Doc-1 [LOW — NEW this pass; pre-M4.5, in-file self-contradiction] The Ozaki speedup figure contradicts
itself within this file (and the spec).** Location: line 19 ("7–13× faster") vs line 161 ("MEASURED 7–17×
over native FP64"). These two figures are in the *same file* and disagree; `architecture.md` ADR-0010
(arch:711-region, line 711's ADR list) states a third value, "measured 8–17×". *Why it matters:* §2 DRY /
doc accuracy — a single source of truth should not state three different measured speedup ranges. This is
**pre-M4.5 body** (the `Precision` enum doc and header banner), out of the M4.5 delta, and is owned by the
prior `include-config.md` review (its finding 3.2). Recorded here only because it is an *in-file*
self-contradiction I encountered while verifying and the first pass dismissed it as "speedup-figure drift
… out of scope" without noting that lines 19 and 161 disagree *with each other*. Not scored against the
delta. Severity: low. Effort: S. PARITY-SAFE: yes.

**Doc-2 [N/A — verified, pre-M4.5] `mantissa_bits` is placed inside `Precision` here (config.hpp:195) but
listed at `DeviceConfig` level in the spec (arch:571).** A structural divergence between the shipped header
and the §9 listing. Pre-M4.5 (the `Precision` struct predates the delta) and the `Precision`-nested
placement is arguably *better* (it scopes `mantissa_bits` to the precision it modifies). Not an M4.5-delta
finding; recorded so the spec-vs-header field-placement drift is on the record alongside Spec-1. No action
on the delta.

---

## Considered & rejected (incl. rejected-for-parity)

- **REJECTED-FOR-PARITY — "Let `prefer_p2p_combine` also choose a faster combine *order* (e.g.
  tree/parallel fan-in across peers) on the capable tier."** Tempting as a perf lever, but ANY change to
  the combine *order* (away from the strict sequential `g=0..G-1` sum) breaks §12: FP addition is
  non-associative, so a tree/concurrent fan-in would produce a different bit pattern than the single-GPU
  reference and the host-staged baseline — and the locked `memcmp` two-tier-neutrality check
  (`test_f2_multigpu_parity.cu:453`) would fail. The whole point of the knob is that it changes only the
  *transport*, never the order. The doc correctly constrains it to "sums in the fixed `g=0..G-1` order
  on-device" (268–270). **Rejected; the knob must stay order-preserving.**

- **REJECTED-FOR-PARITY — "Default `prefer_p2p_combine` to `false` so the parity-proven host-staged
  baseline is the out-of-box behavior."** Unnecessary AND it would mis-signal the design: both paths are
  *proven* bit-identical (the locked `memcmp` gate covers both tiers), so default-`true` (capable-first,
  degrade-and-tag) is the documented §11.4 stance and is safe — on the budget 5090 it degrades to
  host-staged anyway via the `can_access_peer` gate. Changing the default would also break the
  `static_assert(... == true)` contract (`test_config.cpp:84–85`) that pins the §11.4 capable-first design.
  **Rejected — default `true` is correct; parity is not at risk because the gate degrades on the budget
  tier.**

- **[C-1 — is the DOC wrong or the CODE wrong?] Considered: maybe the doc is the bug and the code's
  two-term gate is the intended design (i.e. `enable_peer_access` should NOT gate the combine path).**
  Rejected as the resolution: the banner (255–260) is *explicit* that `enable_peer_access` is the permission
  to call `cudaDeviceEnablePeerAccess` "at all", and the P2P combine *does* call it (`p2p_combine.cu:265`) —
  so a `false` permission with a P2P path taken is a genuine contradiction the design intends to forbid. The
  intended design is the three-term gate; the code is the latent defect and the doc states the intent. C-1
  is filed as "header inaccurate about *current* runtime behavior" (med) with the fix preferentially in the
  consumer. Not rejected — but the *direction* of the fix (widen the gate, keep the doc) is the
  recommendation.

- **Rejected — "Move `prefer_p2p_combine` to `Resources` since it interacts with the discovered
  `can_access_peer`."** No: it is *intent* (what the user prefers), the textbook §(2).3 case for
  `DeviceConfig`. The interaction with `can_access_peer` happens at the *gate*, where intent meets discovery
  — that is the correct seam; putting intent on `Resources` would conflate the two. The banner makes this
  explicit and is right. **Rejected.**

- **Rejected — "Add `enable_gds_ingest` now so the banner's future lever is real."** The banner names it as
  "a future `enable_gds_ingest` (M5/M7)" (line 240) — correctly forward-referenced, not claimed present.
  Adding an unused knob now would be dead API surface (and an ABI commitment) before M5. The banner's
  "APPEND-ONLY as capability levers land" framing is the right posture. **Rejected — leave it as a
  documented future lever.** *Correction to the first pass:* the first pass claimed `enable_gds_ingest` "has
  no tracking home" — that is **wrong**; I confirmed it IS tracked in `docs/cleanup/00-overview.md:287` (the
  capability-lever list) and `docs/TODO.md:57,137` (the M5/M7 GDS items). So the forward-reference is
  consistent with the planning docs; there is no orphan. Not a finding.

- **Rejected — "The doc's `7–13×`/`7–17×` speedup-figure drift is part of this delta."** Out of scope:
  those figures live in the `Precision` enum doc / header banner (pre-M4.5 body), not in the M4.5
  `prefer_p2p_combine`/banner delta. Recorded under Doc-1 only because the two figures contradict *each
  other within the file*; the prior `include-config.md` review owns the fix (its 3.2). Not scored against
  the delta.

- **Rejected — "config.hpp should restate the full `prefer_p2p_combine && can_access_peer && G>=2` gate
  predicate for completeness."** No — that is the C-2/DRY-1 trap. The gate predicate is already restated in
  **five** doc/comment homes (`resources.hpp:41`, `resources.hpp:57`, `p2p_combine.hpp:66`,
  `f2_blocks_multigpu.cpp:27`, `f2_blocks_multigpu.cpp:158`) plus the one *actual code gate*
  (`f2_blocks_multigpu.cpp:172`). `config.hpp` should NOT become a sixth, especially since two of those
  copies state a `G>=2` term the code handles structurally (C-2). `config.hpp` should describe the knob's
  *meaning* and cross-ref the *one* authoritative gate home (the entry point), not duplicate the predicate.
  *Correction to the first pass:* it counted "four" restatements; the precise count of doc-homes is **five**
  (the first pass missed `f2_blocks_multigpu.cpp:27`). **Rejected — fewer copies, not more.**

- **Considered & dismissed — "The header omits the NCCL/`cudaMemcpy` broadcast-back step §11.4 describes."**
  Dismissed: the M4.5 combine (`combine_f2_partials_p2p`/`combine_f2_partials_host`) returns a *host*
  `F2BlockTensor` (the `p2p_combine.hpp` return contract); there is no broadcast in *this* M4.5 unit (the
  broadcast-to-every-GPU is the broader §11.4 vision for the fit phase). The header correctly describes the
  M4.5 reality and does not over-claim. The header's "NEVER NCCL AllReduce" is the *reduction* caveat, not a
  "never NCCL" claim — consistent with the spec's NCCL-for-broadcast-only design (arch:91). No defect.

- **Considered & dismissed — "The `deterministic` doc bullet about `stream_count` forced to 1 contradicts
  the unenforced state."** Dismissed: the doc (292–310) phrases the rules as what `build()` *will* enforce
  ("`build()` enforces the constraints"), which is forward-honest given `build()` is deferred; it does not
  claim the rules are enforced *today*. Covered as the L-1 declared-but-unvalidated note, not a separate
  defect.

## What it takes to reach 10/10

The delta is at 9.5. The half-point is one real contradiction plus doc/spec tidies:

1. **C-1 / K-2 (MED, S, PARITY-SAFE) — the substantive item.** Resolve the `enable_peer_access`-vs-gate
   contradiction. Preferred: widen the consumer gate to `prefer_p2p_combine && enable_peer_access &&
   can_access_peer` (the `device-resources` K2 recommendation), which makes this header's doc correct
   as-written AND stops `p2p_combine.cu:265` from calling `cudaDeviceEnablePeerAccess` under a `false`
   permission. If the gate is left as-is, correct the `prefer_p2p_combine` doc to stop claiming
   `enable_peer_access` gates the path and add a NOTE pointing at the gap. The header and the code MUST
   agree. Add the negative test lane (`enable_peer_access=false, prefer_p2p_combine=true`) that would have
   caught it (T-1).
2. **Spec-1 (LOW, S) — NEW.** Add `prefer_p2p_combine` to the `architecture.md` §9 `DeviceConfig` listing
   (arch:563–581) so the canonical spec field set matches the shipped public header (single-source-of-truth
   discipline; the field is public ABI).
3. **DRY-1 / C-2 (LOW, S).** Single-home the §11.4 P2P-gate predicate at the entry point and reduce the
   *five* doc restatements to cross-refs; do NOT let `config.hpp` restate the full predicate (it correctly
   does not today). Drop the over-stated `G>=2`-as-gate-term framing from the sibling docs (it is
   structural).
4. **Perf-2 (LOW, S).** Add one clause noting the device-resident path elides the host round-trip (the G
   D2H gathers + host placement sum) — the data-bounce saving that motivates the capable tier, staying
   parity-neutral.
5. **R-1 (LOW, S).** Trim the ~26-line `prefer_p2p_combine` comment so it stops re-deriving the banner 20
   lines above; keep semantics + corrected gate + default rationale + a banner cross-ref.
6. **E-1 / L-1 (LOW, deferred to `build()`).** When `ConfigBuilder::build()` lands, add `prefer_p2p_combine`
   (and enforce `deterministic`) in the §9 validation list with the cross-knob rule (normalize/WARN on
   `!enable_peer_access && prefer_p2p_combine`), enforcing the "two distinct knobs" contract instead of
   relying on the gate.

Item 1 is the only place the public contract currently tells a lie about runtime behavior — and the
adversarial trace this pass shows it is not merely a missing AND but an *actively violated veto*
(`p2p_combine.cu:265`). Doing 1 (the gate/doc agreement) is what moves this to a clean 10; item 2 (the spec
listing) is the cheapest remaining drift to close.

## Good patterns to keep

- **The intent/discovery split is textbook and REAL across the unit boundary.** Override intent on
  `DeviceConfig` (`prefer_p2p_combine`, `enable_peer_access`, `deterministic`, `devices`); discovered
  capability on `Resources`/`caps` (`can_access_peer`, VRAM, emulation-honorable); the which-path tag
  (`CombinePath`) on `Resources`, NEVER on the numeric `F2BlockTensor`. The cleanest realization of cleanup
  §(2).2/§(2).3 in the M4.5 set — verified against the actual `resources.hpp`/`backend.hpp` layout.
- **The knob is explicitly and CORRECTLY marked parity-NEUTRAL**, with the right §12 reasoning ("the
  transport only moves bytes; software fixes the order; NEVER NCCL AllReduce") and a *vendor-accurate*
  premise (`cudaMemcpyPeer` is a bit-preserving DMA byte copy with no arithmetic/re-rounding — confirmed
  against the CUDA Runtime API Memory-Management docs + Programming Guide §3.4). This is the §12 line the
  parity work protected; the delta is firmly on the safe side of it, and the locked `memcmp` two-tier check
  proves it empirically.
- **`enable_peer_access` (MAY-WE) vs `prefer_p2p_combine` (WHICH-PATH) is the right two-knob design** with
  the right naming polarity (`enable_*` permission, `prefer_*` preference) — keep the distinction (and make
  the *gate* honor it, per C-1).
- **The new fields are pinned by `static_assert` in a CUDA-free host test** (`test_config.cpp`), and the
  knob's *behavioral effect* (which combine path + the tag) is covered by the locked parity test — the rare
  config field whose effect, not just default, is tested.
- **`deterministic` (closed prior 9.1) correctly couples to the M4.5 fixed-order combine** in its doc, and
  `kMaxVramUtilizationFraction` (closed prior 5.1, the prior pass's #1 item) now exists with a
  `static_assert` range guard in `vram_budget.hpp` — the M4.5 work paid down the two highest-value items the
  9.0 review flagged.
- **Still CUDA-free, minimal includes, append-only banner posture** — the §4 layering promise and the
  "capability levers land as append-only override booleans" design are both upheld by the delta, and the
  forward-referenced `enable_gds_ingest` is genuinely tracked (00-overview §(2).3, TODO M5/M7), not an
  orphan.
