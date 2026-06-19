# M4.5 unit review (FINAL, adversarial 2nd pass) — `tests/reference/test_f2_multigpu_parity.cu` (the SPMG parity GATE)

Reviewer: senior CUDA/C++ reviewer, adversarial second pass over the first-pass draft.
Scope: the parity gate at `/home/suzunik/steppe/tests/reference/test_f2_multigpu_parity.cu`
(549 lines), read line-by-line, against the directly-related context it exercises:
`src/core/fstats/f2_blocks_multigpu.{hpp,cpp}`, `src/core/fstats/f2_combine.hpp`,
`src/device/{resources.hpp,p2p_combine.hpp,shard_plan.hpp}`, the probe in
`src/device/cuda/cuda_backend.cu` + `src/device/backend.hpp` (`BackendCapabilities`),
the contracts `include/steppe/{config,fstats}.hpp`, the sibling
`tests/reference/test_f2_blocks_equivalence.cu`, and the CMake wiring
(`tests/CMakeLists.txt:558–574`). Standard judged against: `docs/architecture.md`
§2/§4/§7/§9/§11.4/§12/§13.

Every CUDA-behavior claim is cited to the official NVIDIA docs (cuBLAS Results
Reproducibility 13.x; CUDA Runtime API `cudaMemcpyPeer`/`cudaGetDeviceProperties`),
fetched this pass; sources listed at the end. Where the first-pass draft asserted a CUDA
or sibling-test fact, I re-verified it against the actual file/doc and either CONFIRMED,
CORRECTED, or REJECTED it — flagged inline.

## Role & layering

This is a **self-checking `main()` test executable** (not a GoogleTest TU;
`tests/CMakeLists.txt:553` "Self-checking main()"). It increments a file-static
`g_failures` via `check()` and returns `EXIT_FAILURE`/`EXIT_SUCCESS`; CTest gates on the
exit code (`add_test(NAME f2_multigpu_parity …)`, line 574). It is wired with
`STEPPE_HAVE_EMU_TUNING=1` on the box build (line 571–573), so the EmulatedFp64 arm
exercises the fixed-slice Ozaki path the per-device `compute_f2_blocks` engages.

As a **test** it sits *outside* the production `app→api→core→device` layering (§4) — it
legitimately `#include <cuda_runtime.h>` and links `steppe::io + steppe::core +
steppe::device + steppe::core_internal + steppe::api`, and calls `cudaGetDeviceCount` /
`cudaGetDeviceProperties` directly. The §4 layering rule does **not** bind this file. The
binding standards are §12 (the bit-identity parity law it asserts), §13 (testing), §2
(decomposition/DRY/no-global-mutable-state *within the test*), §7 (only insofar as the
test touches the runtime API — two spots). I CONFIRM the first pass's role analysis.

Notably (and the first pass got this right): the test reaches honorability **only via the
CUDA-free `caps.emulated_fp64_honorable` probe field** (`backend.hpp:184`), and does *not*
include the device-private `device/cuda/f2_block_kernel.cuh` the sibling test pulls in for
`emulation_honorable(...)`. That is the correct seam — and it is the seam the F-COV-1 fix
must use (the first-pass fix sketch that named `emulation_honorable` would force a
device-private include into this TU; use the probe field instead).

**Central question — "could the gate pass while parity is actually broken?"** The
EmulatedFp64 bit-identity asserts (claims 1–4) are genuine `std::memcmp`; the
which-path-tag and tier asserts are sound; the native-FP64 oracle cell is *structurally*
honestly scoped. BUT there remain real coverage/scope holes a 9.5–10 gate must close —
the worst (F-COV-1, HIGH) is the silent-emulation-fallback blind spot: the gate can print
`[EmuFp64{40}] … bit-identical … PASS` while the EmulatedFp64 lane secretly ran native
Fp64. And the native-cell tolerance the first pass blessed as "mirrors the sibling" does
**not** mirror the sibling (corrected below, F-NUM-1 upgraded). Full detail follows.

## Score: 8.5/10 — a genuinely strong, honest bit-identity gate, with one HIGH coverage blind spot, one materially-wrong tolerance-provenance claim, and the usual decomposition/robustness gaps

The `memcmp` predicate is real and NaN/−0.0-safe; the which-path tag is read out-of-band
exactly as §11.4 mandates and the P2P arm is verified to *actually take* P2P (not silently
fall back) — the model the emulation check should copy; the native-FP64 cell is carved out
explicitly with shape asserted *exactly*. Held back from 9.5+ by: (a) **F-COV-1 (HIGH)** —
no assertion that EmulatedFp64 actually engaged, so the production fixed-slice path can be
silently un-exercised while green; (b) **F-NUM-1 (now med, corrected)** — the
`kNativeFp64ParityRelTol = 1e-9` "mirrors test_f2_blocks_equivalence's native-FP64 f2 tier"
claim is **false**: the sibling's native tier is `kTolNativeVsRef = 1e-8` in a *combined*
`|c−r|/(1e-9+|r|)` metric, while this test uses `1e-9` in a *different* (floor-at-1.0)
metric; (c) **F-DOC-1/F-DOC-2** — the native-Fp64 rationale states an undocumented
"batchCount" claim as cuBLAS fact and silently rests on an unstated "same SM count"
precondition; (d) **F-DEC-1** — `run_dataset` is ~220 lines doing six jobs; (e) several
smaller robustness/readability items. The thing it exists to prove, it proves — but these
gaps are exactly what a demanding reviewer must not wave through.

## Findings

### Performance (first-class this pass)

Framing CONFIRMED from the first pass: this is a **CI test**, not a hot path — datasets are
P=50 (fast) and P=768 (scale), run once. The production combine's perf (sequential
per-device fan-out; sequential `cudaMemcpyPeer` that could pipeline via
`cudaMemcpyPeerAsync` + events; pinned staging) lives in `f2_blocks_multigpu.cpp` /
`p2p_combine.cu`, **NOT in this file** — out of this unit's scope, pointer left for the
entry-point review (see "Considered & rejected"). I do not invent kernel-perf findings here.
But there are real, defensible *test-runtime* inefficiencies; none touches the combine math,
so all are parity-safe.

**F-PERF-1 (med, S, parity-safe: yes) — CONFIRMED, with the construction cost verified.**
`run_dataset:348–354`: inside `for (int p = 0; p < 2; ++p)` the loop rebuilds the reference
backend from scratch — `auto ref_backend = steppe::device::make_cuda_backend(0);` then
`compute_f2_blocks(*ref_backend, …)`. I verified via `cuda_backend.cu:79–106` that the
`CudaBackend` ctor does `cudaSetDevice(device_id)` then constructs `blas_` (a cuBLAS handle
— "a cuBLAS library context is tightly associated with the current device at the moment of
the `cublasCreate()` call", cuBLAS §2.1.2, cited in-source at `cuda_backend.cu:81–83`) and
allocates the determinism `workspace_` and sets it on the handle. So this is a fresh
`cudaSetDevice` + `cublasCreate` + workspace `cudaMalloc` + `set_workspace`, **twice per
dataset** (once per precision), on top of `resG1`'s device-0 backend already built once at
line 330. `cublasCreate` implicitly synchronizes the device. Fix: build one device-0
reference backend before the precision loop (it is the same device, same code → same bits).
Benefit: removes 2 (4 across both datasets) `cublasCreate`/`cublasDestroy`/workspace-alloc
round-trips. Risk: none — the test would catch any divergence. NB: this interacts with
F-COV-1/F-BUG-4 — keeping the reference backend a *distinct instance* from `resG1` is a
deliberate strength (F-BUG-4); hoisting it *out of the precision loop* (one ref backend for
both precisions) is the safe optimization and does not collapse it into `resG1`.

**F-PERF-2 (low → informational, parity-safe: yes) — CONFIRMED as NOT-a-defect.** In the
native lane both `candG2h` and `candG2p` are computed and each is load-bearing (G2h vs ref,
G2p vs ref when peer, and G2h vs G2p for claim 4). The four `compute_f2_blocks_multigpu`
calls per precision are each distinct-assertion-bearing. No redundant compute. Recorded so
the trail is explicit. (First pass agreed; I re-verified by tracing each candidate to its
assert.)

**F-PERF-3 (low, S, parity-safe: yes) — CONFIRMED.** EmulatedFp64 asserts build labels via
`std::string("[") + precNames[p] + "] …"` (lines 361–362, 390–391, 403–404, 454–455) — a
heap alloc per check — while the native lane uses stack `char lab[256]` + `snprintf` (lines
419, 432). Two label styles in one function; trivial runtime but a real inconsistency the
F-DEC-1 decomposition unifies. Fix: one `make_label(prec, claim)` helper.

**F-PERF-4 (informational, parity-safe: yes) — CONFIRMED as correctly-out-of-scope.** There
is no timing/throughput assertion on the P2P arm, so a P2P combine that silently became
slow would still PASS. That is *correct* for a parity gate (perf belongs in nvbench; §11.4
explicitly frames the P2P path as "architectural cleanliness, not a throughput lever").
N/A as a blocking finding.

**F-PERF-5 (NEW — low, S, parity-safe: yes): the four per-dataset device computes are run
on the implicit per-backend stream with a host round-trip between each, fully serialized —
acceptable for a test, but the gate also re-uploads the same Q/V/N column spans to device 0
multiple times per dataset (ref + G1 + G2h-shard0 + G2p-shard0), with no reuse.** Per
dataset, device 0 sees the full Q/V/N (ref, G1) twice and its shard-0 slice (G2h, G2p)
twice — four H2D uploads of overlapping host data, each inside a separately-constructed or
separately-driven backend, with a full host sync between computes. This is *data bouncing*
in the literal sense (the same host bytes re-DMA'd), and it is genuinely unavoidable
through the public seam **without changing what the gate proves** (each call must drive the
real entry point per config — see the rejected "cache the partials" item). So the fix is
limited to F-PERF-1's "don't also rebuild the backend each time"; the re-upload itself is
load-bearing (it is part of exercising the real `compute_f2_blocks_multigpu` per config).
Recorded as a real, measured inefficiency that is **correctly not removable** without
weakening the gate. Severity low.

**No grid-stride / casting-noise kernel findings against this file:** N/A because this TU
launches no kernels and does no device-side reduction — the kernels and the casts in the
sharding live in `f2_blocks_multigpu.cpp` (host) and `p2p_combine.cu` (device), reviewed in
their own units. The casts *in this file* (`static_cast<std::size_t>(P)*M` at line 155,
`static_cast<long>(snptab.count)` at line 303) are narrow, intentional, and width-correct
on a 64-bit platform (`long` ≥ `size_t`'s value range for realistic SNP counts) — verified,
no width bug (F-EDGE-3 confirms). The one place a *cleaner contract* would remove a cast is
the `int P; long M` split (F-BUG-3 ties in), but that mirrors the data fixture's
`shape.txt "P M"` and the sibling test verbatim — DRY-aligned, not noise.

### Correctness & bugs

**F-BUG-1 (med, S, parity-safe: yes) — CONFIRMED, reachability re-verified.**
`report_first_diff:251,259–261` computes `n = min(ref.f2.size(), cand.f2.size())` (f2
sizes) and reuses that same `n` to walk **vpair** (line 259). I confirmed via
`fstats.hpp:52–60` that a well-formed `F2BlockTensor` has `f2.size()==vpair.size()==P²·
n_block`, and `bit_equal` (lines 168–175) already gates equal f2 and vpair sizes — so in
every *passing* call this is safe. But `report_first_diff` is the **failure-path
diagnostic**: it runs precisely when something is already wrong. A regression that desynced
f2 and vpair lengths (a combine that placed f2 but not vpair, a partial with mismatched
slabs) would make the vpair loop index with the f2-derived `n` and read OOB on the shorter
vector. That is exactly the class of bug a parity gate exists to catch, so the diagnostic
must be robust on malformed input (§2 fail-fast). Fix: `const std::size_t nv =
std::min(ref.vpair.size(), cand.vpair.size());` for the vpair loop (or assert
`f2.size()==vpair.size()` at entry). Severity med (latent, failure-path-only).

**F-BUG-2 (med, S, parity-safe: yes) — CONFIRMED, severity raised low→med.** `load_qvn`
(lines 144–162) returns `false` for THREE distinct outcomes: absent `shape.txt` (intended
skip, line 148), malformed `shape.txt` (`fscanf != 2`, line 152, a real error), and
present-but-unreadable/wrong-size Q/V/N (line 159, a real error). The two error cases print
to stderr but still return `false`; the caller (`run_dataset:285–289`) prints `[skip] …
absent or unreadable — not run` and returns `false` (a logged skip, **not** a `++g_failures`).
Consequence: if `derived_acc` is present-and-valid and `derived_full` is
**present-but-truncated** (e.g. a half-written `V.f64`), `any_dataset_run` is true (acc ran),
`g_failures==0`, and the gate returns `EXIT_SUCCESS` having **silently skipped the scale
dataset it claims to cover**. That is a "gate green while a dataset it advertises never ran"
hole — and on a parity gate, the scale dataset (P=768) is precisely where a sharding bug is
most likely to surface. §2 fail-fast: present-but-malformed is a real error, distinct from
absent. Fix: `load_qvn` returns an enum `{Absent, Malformed, Ok}` (or an `out_present`
flag); the caller `++g_failures` on `Malformed`, skips only on `Absent`. I raise this to
**med** (the first pass had low): the silent-skip-of-the-scale-dataset is materially more
dangerous on a *parity* gate than the draft credited.

**F-BUG-3 (low, S, parity-safe: yes) — CONFIRMED.** `M` is parsed via `fscanf("%d %ld",
&P, &M)` (line 149) with no positivity/range check; `count = (size_t)P * (size_t)M` (line
155). A negative `M` wraps `count` to a huge `size_t` → `read_f64` `resize(count)` →
`std::bad_alloc` (caught at `main:530` → return 1). A zero `M` feeds an empty SNP table to
`assign_blocks`. Neither is a *silent* pass, so low severity — but `P=50 M=-1` aborts via
`bad_alloc` rather than a clean "malformed shape" diagnostic. Fix: validate `P>0 && M>0`
right after the `fscanf`. Ties into F-BUG-2 (the same `{Absent,Malformed,Ok}` return).

**F-BUG-4 (informational, parity-safe: yes) — CONFIRMED as a STRENGTH, not a bug.** The
reference uses a freshly-constructed device-0 backend (`make_cuda_backend(0)`, line 352)
while G==1 uses `resG1`'s device-0 backend (lines 357–358) — two *distinct* `CudaBackend`
instances (independent cuBLAS handles + workspaces) on the same physical device. This
*strengthens* claim (1): it proves bit-identity across independent backend instances, not
just "the same object twice." cuBLAS guarantees bit-wise reproducibility "on GPUs with the
same architecture and the same number of SMs" for a single stream/single toolkit version
[cuBLAS Results Reproducibility, re-verified this pass], which both instances satisfy. Good
property; recorded so "is the reference path sound?" has an explicit yes. (Note: F-PERF-1's
fix preserves this — hoist *one* distinct ref backend out of the precision loop; do not
fold it into `resG1`.)

**F-BUG-5 (NEW — low, parity-safe: yes): the native-lane assertions are SKIPPED entirely
when `kNativeFp64ParityRelTol` is met only because the metric is lenient — but more
importantly, the native-lane `else` branch (lines 411–440) has NO `report_first_diff` on
the bit-level *and* never asserts the two candidates are mutually bit-identical to each
other in the native lane via the strict path; the only strict native check is claim (4).**
Re-reading: claim (4) (lines 442–458) IS asserted strict-bit-identical for the native lane
too (`bit_equal(candG2h, candG2p)`), which is correct and load-bearing (the combine is
transport-only). So this is **not** a coverage hole — I CONFIRM the native lane's claim-(4)
is strict. I record this finding only to DOWNGRADE-TO-NONE my own initial suspicion: there
is no missing native-lane strict check. (Kept in the record per the "question your own
additions" mandate — see Considered & rejected.)

### Numerical / precision vs §12 — including the native-FP64 oracle-tol cell

**F-NUM-1 (med, S, parity-safe: yes) — CORRECTED & UPGRADED (first pass had this materially
wrong, low→med).** The first pass wrote that `kNativeFp64ParityRelTol = 1e-9` "does match
[the sibling] test's `native vs oracle < 1e-9`." **That is false.** I read
`test_f2_blocks_equivalence.cu:78–84`: the sibling's native tier is
`constexpr double kTolNativeVsRef = 1e-8;` applied in a **combined** metric
`comb = |c−r| / (kAtol + |r|)` with `kAtol = 1e-9` (`accuracy()`, lines 129–130). The `1e-9`
the first pass matched against is `kAtol` (the additive near-zero floor), **not** the native
threshold. So the parity test's own comment (lines 208–209) — "kNativeFp64ParityRelTol
mirrors test_f2_blocks_equivalence's native-FP64 f2 tier" — is **wrong on both axes**:
  - **Value:** sibling native tier = `1e-8`; this test = `1e-9` (an order of magnitude
    *tighter*, which happens to be safe here because the observed batched-GEMM noise is
    ~2e-13, but the *provenance claim is false*).
  - **Metric:** sibling = `|c−r|/(kAtol+|r|)` combined (atol=1e-9), over **off-diagonal
    pairs only** (`accuracy()` skips `i==j`); this test's `max_rel_dev` = `|r−c|/max(|r|,1.0)`
    over **all** entries including the diagonal, with a hard floor of 1.0. These are
    genuinely different tolerances measured differently.
Why it matters on a parity gate: the gate's value *is* the precision of its claims; a
comment that says "I mirror the sibling tier" when it does neither is the F-READ-1 drift
risk made concrete, and it misleads a maintainer who later "retunes the sibling tier to
match." Fix: (a) correct the comment to state the *actual* relationship ("an absolute-1e-9
bound under a floor-at-1.0 metric over all entries; this is *tighter* than the sibling's
off-diagonal combined `1e-8` tier and is justified by the ~2e-13 observed batched-GEMM
noise"), OR (b) genuinely single-source by reusing the sibling's `kTolNativeVsRef` + the
`accuracy()`-style combined metric (this is the cleaner fix and folds F-NUM-2 in).

**F-NUM-2 (low, parity-safe: yes) — CONFIRMED.** `max_rel_dev`'s denominator floor of `1.0`
(line 230) makes the "relative" deviation an **absolute** deviation for any `|f2| ≤ 1` —
which is most of the tensor (f2 entries are O(1e-2), the het-correction diagonal small).
So `kNativeFp64ParityRelTol` is effectively an absolute 1e-9 bound for the bulk. The comment
(lines 219–220) defends this as `atol+rtol·|b|`-spirit, which is defensible, but: (a) the
floor `1.0` is a **magic number** (§4) — name it `kF2RelDenomFloor` with the rationale
(f2 ∈ O(1)); (b) it diverges from `config.hpp`'s `kRelFloor` and from the sibling's combined
form (the F-NUM-1 metric divergence). Fix folds into F-NUM-1(b).

**F-NUM-3 (informational, parity-safe: yes) — CONFIRMED as the precision-side framing of
F-COV-1.** §12: EmulatedFp64 is honorable only when fixed-slice mantissa control engages
(`STEPPE_HAVE_EMU_TUNING` + the tuning API present). If it silently degrades to native Fp64,
the EmulatedFp64 lane *still* produces identical bits sharded-vs-not (native is
deterministic on same-SM-count), so claims (1)–(4) all still PASS — but the gate certified
the **wrong path**. See F-COV-1 for the fix.

**F-DOC-1 (med, S, parity-safe: yes) — CONFIRMED against the cuBLAS docs this pass.** The
native-FP64 rationale (lines 185–206, 370–374, 411–415) asserts (line 196): "cuBLAS native
FP64 (CUBLAS_COMPUTE_64F) is run-to-run reproducible at a FIXED configuration but is NOT
required to be invariant to the batchCount." I fetched the cuBLAS Results Reproducibility
section: it guarantees bit-wise identical results "when executed on GPUs with the **same
architecture and the same number of SMs**", for a **single CUDA stream** and within a
**single toolkit version**; the documented reproducibility-breakers are **multiple streams**
and **atomics/`cublasSetAtomicsMode`** (and fixed-point emulation workspace). The docs do
**not** mention `batchCount` as a reproducibility/algorithm-selection axis at all. So the
comment's causal claim is *plausible* (a different batchCount is a different API
configuration, and the guarantee is per-configuration — it does not promise *cross*-
configuration consistency) but it is presented as documented fact when the docs are silent
on the specific axis. Adversarial check on my own correction: is the comment *wrong*, or
just over-stated? It is over-stated, not wrong — the underlying inference (sharding changes
the call shape ⇒ a different cuBLAS configuration ⇒ possibly different last-ULP rounding) is
sound and is the *correct* reason native Fp64 is exempted. Fix: reword to the defensible
documented form — "the per-device size-grouped batched GEMMs issue a *different set of
`cublasGemmStridedBatchedEx` calls* (different batchCount per group) than the single-GPU
run; cuBLAS guarantees bit-identity only *per fixed configuration on a fixed SM count*, and a
different call shape is a different configuration, so native FP64 may round differently
sharded-vs-not — accumulation-shape sensitivity, not a documented batchCount knob." Keep the
`[UNCERTAIN]` framing. Severity med: a parity gate's documentation *is* its contract.

**F-DOC-2 (med, S, parity-safe: yes) — CONFIRMED.** The EmulatedFp64 bit-identity claim (the
gate's core) silently rests on the cuBLAS "**same number of SMs**" precondition, which holds
here only because both shards run on a homogeneous 2×5090 (or the homogeneous PRO box). The
gate neither states nor checks this. On a **heterogeneous** box (5090 + 4090; PRO 6000 +
anything) the per-shard GEMMs could land on devices with different SM counts, and even
fixed-slice Ozaki's underlying tensor-core GEMMs sum their slices via a cuBLAS GEMM whose
bit-reproducibility is SM-count-conditioned [cuBLAS Results Reproducibility — "same
architecture **and the same number of SMs**"]. The fixed-slice *decomposition* is
batchCount-deterministic (header lines 180–183, correct *about the slicing*), but the slices
are still GEMM-summed. Why it matters: the banner "bit-identical across G" is stated
unconditionally; its truth is **conditioned on device homogeneity** the gate does not
assert. Fix: (a) document the precondition; (b) assert it — `resG2*.gpus[0].caps`
vs `gpus[1].caps` `compute_major/minor` equal across g (the fields exist, `backend.hpp:156–
157`), and if a `caps.sm_count` field is added (it is *not* present today —
`BackendCapabilities` has `device_count`, `compute_major/minor`, `total/free_vram_bytes`,
`can_access_peer`, `emulated_fp64_honorable`, but **no SM count**), assert equal SM counts.
Severity med — the gate is the §12 contract and will outlive the current hardware.
Adversarial: real risk on the *current* boxes? No (homogeneous). But under-qualifying a
"proven" parity claim is the classic way it later turns out to have rested on an unstated
assumption.

### CUDA idioms / RAII / probe semantics vs §7, §11.4, §12

**F-COV-1 (HIGH, S, parity-safe: yes) — CONFIRMED as the single most important finding; fix
refined.** The gate never asserts that EmulatedFp64 *actually engaged the fixed-slice Ozaki
path* vs silently falling back to native Fp64. The sibling `test_f2_blocks_equivalence.cu`
explicitly guards this: under `STEPPE_HAVE_EMU_TUNING` it asserts the emu output **differs
bit-for-bit** from native (`emu_differs_from_native`, sibling lines 254–256; the FAIL
message at 339–342: "EmulatedFp64 honorable but emulation did NOT engage … silently fell
back … the §12 emulated-FP64 path is not running"). **This gate has no such check.** Its
EmulatedFp64 asserts (claims 1–4) compare *sharded EmulatedFp64* to *single-GPU
EmulatedFp64* — both via the same code, same precision enum. If `emulated_fp64_honorable`
were false (a build/toolkit/arch where the fixed mantissa-control API is absent — the
backend then degrades the EmulatedFp64 request to native Fp64 with a logged tag, per
`config.hpp` + §12), then **both sides** of every EmulatedFp64 comparison run native Fp64,
the bits match (native is deterministic on same SM count), and the gate prints
`[EmuFp64{40}] … bit-identical … PASS`. The gate would be **green while never having
exercised the production fixed-slice path it exists to certify** — the worst class of
"passes while broken" for *this* unit's stated purpose (lines 204–206: "EmulatedFp64{40} …
carries the strict bit-identity claim and PASSES it across G"). This is the precise
asymmetry F-CAP-1 highlights: the gate verifies the *combine transport* actually engaged
(`last_combine_path == P2pDeviceResident`) but not that the *precision mode* engaged.
**Fix (refined from the first pass — use the CUDA-free seam, do NOT include
`f2_block_kernel.cuh`):** the precision loop already computes both `ref` (native, p==0) and
`ref` (emu, p==1). Capture each into a named local (`natRef`, `emuRef`), then once both
exist assert, under `#ifdef STEPPE_HAVE_EMU_TUNING`: (1)
`resG1.gpus[0].caps.emulated_fp64_honorable == true` (the probe field, `backend.hpp:184` —
already CUDA-free, no device-private include); AND (2)
`std::memcmp(emuRef.f2.data(), natRef.f2.data(), …) != 0` (emu must DIFFER from native, or
emulation silently fell back). ~6 lines. PARITY-SAFE: yes — adds an assertion, changes no
math. Severity HIGH: it is the one finding that answers "any way the gate could pass while
parity is actually broken?" with a yes (parity *of the wrong path*).

**F-IDIOM-1 (low, parity-safe: yes) — CONFIRMED with one sub-claim DOWNGRADED for
honesty.** `device0_is_pro_tier:272–276`. (a) `cudaGetDeviceProperties(&prop, 0)` queries
device 0 by ordinal regardless of the current device, so no `cudaSetDevice` is needed —
correct (I could not get the docs to *explicitly* state the no-current-device requirement,
but the by-ordinal `device` parameter is in the signature and the call is the standard
spelling). (b) The first pass asserted "in CUDA 12+/13 `cudaGetDeviceProperties` is a macro
to `_v2`" as a docs-cited fact — **I could not confirm the macro mapping from the live runtime
API docs this pass** (the device-management page did not surface the `_v2` macro). I
therefore DOWNGRADE that sub-claim to "compiles correctly under CUDA 13 and is the
conventional spelling; the `_v2` macro detail is unconfirmed and irrelevant to the verdict"
— it remains a **no-defect** either way. The one genuine nit stands: `device0_is_pro_tier`
silently returns `false` if `cudaGetDeviceProperties` fails (line 274), which would
mis-classify a PRO box as non-PRO and *weaken* the strict peer assertion (claim 5). A probe
failure here is a real fault and should at least WARN. Severity low (and F-IDIOM-2's fix
removes the function entirely).

**F-IDIOM-2 (med, S, parity-safe: yes) — CONFIRMED, and the proposed fix re-verified
against the production gate.** The PRO-tier strict asserts key off `strstr(prop.name, "PRO")`
(lines 272–276, 481–488), a **magic marketing string** (§4) duplicated from
`test_backend_capabilities_probe.cu` (DRY miss, comment line 270). It is a fragile dispatch
key: "RTX PRO 6000" matches, but so would any future SKU with "PRO" in the name that lacks
stock-driver P2P → a **false FAIL**. The workflow brief says the PRO-specific tiering is "a
separate concern, not this audit," so I keep this **non-blocking**, but the cleaner
parity-neutral form is real: drop `device0_is_pro_tier` and assert the *invariant that
actually holds on every tier* — the **biconditional** "`resG2p.gpus[0].caps.can_access_peer`
⇔ `resG2p.last_combine_path == P2pDeviceResident`" (and `== HostStaged` otherwise). I
re-verified this is exactly the field the production fork reads
(`f2_blocks_multigpu.cpp:171–172`: `prefer_p2p_combine && gpus[0].caps.can_access_peer`),
so the test would gate on precisely the production condition — no hardware naming, no false
FAIL, valid on the 5090 and the PRO box alike. Severity med.

**F-IDIOM-3 (low, parity-safe: yes) — CONFIRMED.** No clean-error-state sweep at exit (§13's
`CudaTest` fixture "resets the device per suite and asserts a clean error state in
TearDown"). On the success path there is no `cudaGetLastError()` to confirm no async fault
went unnoticed before `RESULT: PASS`. Fix: `cudaDeviceSynchronize()` + `cudaGetLastError()`
(non-zero ⇒ fail) before the final PASS. Severity low (post-launch checks inside the
backends catch most of this; belt-and-suspenders at the test boundary).

**F-IDIOM-4 (informational — sound) — CONFIRMED.** `can_access_peer` is read from
`resG2p.gpus[0]` for the claim-(3) guard (line 401) and the tier sanity (line 467) — the
**same** field the production path gates on (`f2_blocks_multigpu.cpp:172`). So the test
cannot gate on P2P when production wouldn't take it, nor vice versa. Good. No fix.

**F-IDIOM-5 (NEW — low, parity-safe: yes): the gate builds `resG2h` and `resG2p` (two full
2-device `Resources`, each owning two `CudaBackend`s with their own cuBLAS handles +
workspaces) PLUS `resG1` PLUS a per-precision throwaway ref backend — i.e. up to 5 live
`CudaBackend` instances (and 5 cuBLAS contexts) concurrently per dataset.** `resG1` (1
backend), `resG2h` (2), `resG2p` (2) are all built before the precision loop (lines 330–345)
and held for the whole dataset, and F-PERF-1's ref backend adds a 6th transiently. This is
fine for VRAM at P=50/768 (the workspaces are small), but it is worth recording that the gate
holds **two independent 2-GPU resource bundles simultaneously** — `resG2h` and `resG2p` each
own a device-0 *and* device-1 backend, so device 0 hosts up to 3 cuBLAS contexts at once
(resG1, resG2h[0], resG2p[0]) plus the ref. No correctness or parity impact (each backend is
independently deterministic on the same device — F-BUG-4's property). Severity low; recorded
for the resource-footprint axis. Fix (optional): the two G2 bundles must be distinct (they
carry different `prefer_p2p_combine` config and the test reads each one's `last_combine_path`
independently), so they cannot be merged — this is **not removable** without weakening the
which-path-tag coverage. Like F-PERF-5, an inherent cost of driving the real seam per config.

### Decomposition / single-responsibility / function size vs §2

**F-DEC-1 (med, M, parity-safe: yes) — CONFIRMED.** `run_dataset` (lines 281–501) is ~220
lines doing six jobs: (1) load Q/V/N, (2) parse .snp + assign blocks, (3) build three
Resources bundles, (4) run the per-precision ref+candidate matrix with the strict/native
split, (5) the two-tier neutrality check, (6) the capability/which-path tag sanity. The
precision-loop body (348–463) alone is ~115 lines with a nested `if (have_two) {
if (is_emulated) {…} else {…} … }`. §2 (single responsibility, testability). Fix: extract
`assert_bit_identical(prefix, ref, cand, tag)` (wraps `bit_equal`+`check`+`report_first_diff`),
`assert_oracle_tol(prefix, ref, cand, tag)` (wraps `max_rel_dev`+`check`+`report_first_diff`),
`assert_tier_tags(resG2h, resG2p, peer0)`; the precision loop becomes a flat
`is_emulated ? assert_bit_identical : assert_oracle_tol` dispatch. This subsumes F-PERF-3
(one label helper) and is the natural home for the F-COV-1 honorability assert. Drops the
function well under 80 lines. Severity med — this is *the* file that is supposed to be the
most trustworthy in the multi-GPU surface, so its own readability matters.

**F-DEC-2 (low, parity-safe: yes) — CONFIRMED.** The strict/native dispatch is duplicated
host-staged vs P2P (claims (2) and (3): lines 386–410 are two near-identical
`bit_equal`+`check`+`report_first_diff` blocks; the native `else` 416–439 is two
near-identical `max_rel_dev`+`snprintf`+`check` blocks). The F-DEC-1 helpers collapse this
4-way duplication to two calls each.

### Readability, naming, const-correctness, comment density vs §7

**F-READ-1 (low) — CONFIRMED, and now PROVEN by F-DOC-1 + F-NUM-1.** The file is ~73 lines of
header banner + dense per-claim commentary. This is *mostly a virtue* for a §12 gate. The one
concrete improvement: several comments **restate the architecture doc at length inline**
(lines 177–213 reproduce a paragraph of §12 reasoning) where a one-line summary + `§12`
cross-reference would not drift. F-DOC-1 (the batchCount overstatement) and F-NUM-1 (the
false "mirrors the sibling" claim) are exactly the drift this predicts — two load-bearing
inline restatements that are *wrong*. Prefer cross-references over restatements for the
load-bearing claims. Severity low.

**F-READ-2 (low) — CONFIRMED.** The strict-vs-native branch keys off `prec.kind ==
EmulatedFp64` (line 385, robust to reordering `precs[]`), but the comment (line 374) hard-codes
"p==1 is the EmulatedFp64{40} lane; p==0 is native Fp64." Reordering `precs[]` would make the
comment lie while the code stays correct. Drop the index-mapping comment; let `is_emulated`
speak.

**F-READ-3 (low, const-correctness) — CONFIRMED.** `g_failures` (line 109) is a mutable
file-static — global mutable state, which §2 forbids ("No global mutable state"). The sibling
tests do the same (a test-harness convention), so it is arguably acceptable, but §2 is stated
absolutely and §18 says "no new global mutable state." A 9.5+ form threads a `TestResults&`
(or returns a count) through `check`/`run_dataset`. Severity low; flagged for completeness
per the no-skip mandate.

**F-READ-4 (NEW — low): `check()`'s `bool ok` parameter is evaluated eagerly at every call
site, so the P2P-arm `bit_equal(ref, candG2p)` (line 402) runs even when guarded by the
`if (resG2p.gpus[0].caps.can_access_peer)` — fine here (it IS guarded), but the native-lane
`max_rel_dev` (lines 417, 429) is a full O(P²·n_block) scan computed inline into the
`snprintf` label even on the pass path.** This is correct (the worst-rel value is wanted in
the label for observability — a documented good pattern), so it is **not a defect**;
recorded only to note the label intentionally carries the measured deviation. No fix.

### Magic numbers & hardcoded values vs §4

**F-MAG-1 (low) — CONFIRMED.** The "PRO" device-name substring (line 275) — see F-IDIOM-2;
fix removes the name dispatch.

**F-MAG-2 (low) — CONFIRMED.** `char lab[256]` (lines 419, 432) — fixed buffer, bare literal.
The `snprintf` is bounded (no overflow) but could truncate (won't at current lengths). The
F-DEC-1 label helper removes it.

**F-MAG-3 (low) — CONFIRMED.** `kDefaultDataRoot` / `kGenoBase` (lines 106–107) are hardcoded
box paths; both are appropriately-named `constexpr` (good) and the data root is `argv[1]`-
overridable. `kGenoBase` ("v66.p1_HO.aadr.patch.PUB") is **not** overridable; an AADR version
bump → `.snp` path 404 → `read_snp` throws → caught at line 530 → return 1 (clean fail, not a
silent pass). Low risk. Optional fix: `argv[2]` to override the geno base.

### Layering / API / ABI vs §4

**F-LAY-1 (informational — no defect) — CONFIRMED.** The which-path tag is read from
`Resources` (out-of-band), never from `F2BlockTensor` (lines 467, 471–472, 478–479, 487–488,
495–496) — exactly §11.4 / cleanup §(2).2. `bit_equal` compares only the numeric payload
(f2, vpair, block_sizes, P, n_block) — it does **not** touch `last_combine_path` (a different
object), so the tag cannot contaminate the bit-identity diff. Verified clean. Good pattern.

**F-LAY-2 (informational — no defect) — CONFIRMED and IMPORTANT for the F-COV-1 fix.** The
includes are minimal and correctly scoped to the CUDA-free seams plus the one
`<cuda_runtime.h>` it needs for `cudaGetDeviceCount`/`cudaGetDeviceProperties`. It does
**not** reach into device-private kernel headers (unlike the sibling, which pulls
`f2_block_kernel.cuh` for `emulation_honorable`). It gets honorability via the CUDA-free
`caps.emulated_fp64_honorable` field — the right seam. **The F-COV-1 fix must preserve this**:
use the probe field, not the device-private predicate, or it would regress this clean
layering. Good.

### Edge cases & failure modes

**F-EDGE-1 (low, parity-safe: yes) — CONFIRMED.** Single-GPU box (`visible == 1`): the G==1
floor runs, G==2 is skipped (logged), gate PASSES on claim (1) alone (guards at 332–345,
459–462, 466). On a 1-GPU box the gate proves only "the multi-GPU codepath with one device
== single-GPU" (a structural no-op — `f2_blocks_multigpu.cpp:88` special-cases G==1 to call
the backend directly, verified). Concern: on a 1-GPU box the gate gives **no** cross-device
evidence yet returns `EXIT_SUCCESS` — a CI that only ever runs it on a 1-GPU runner is
perpetually green while never testing the thing the gate exists for. Correct per the skip
policy, but "gate green on the wrong runner" is a coverage trap. Fix (process): pin the gate
to a ≥2-GPU runner; make the G==2-skip banner loud (currently a per-precision `[skip]`, easy
to miss). Severity low.

**F-EDGE-2 (low, parity-safe: yes) — CONFIRMED.** The empty-shard path (a device owning zero
blocks, `n_block < G`) is **never exercised** by this gate: derived_acc/derived_full have
~757 blocks ≫ G=2, so `f2_blocks_multigpu.cpp:126–135`'s empty-shard branch (M_local==0,
n_block_local==0) never fires here. That branch has its own coverage (`test_f2_empty_guard.cu`,
confirmed present in `tests/reference/`), so it is not this gate's job — but the *multi-GPU
combine of an empty partial* is not parity-tested end-to-end here. Recorded for completeness.

**F-EDGE-3 (informational — no defect) — CONFIRMED.** `static_cast<long>(snptab.count) != M`
(line 303): `count` is `size_t`, `M` is `long`; on a 64-bit platform `long` is 64-bit so the
cast is lossless for realistic SNP counts. Integer-width question closed: fine.

**F-EDGE-4 (NEW — low, parity-safe: yes): if exactly one of `derived_acc`/`derived_full` is
present and `run_snp` rows mismatch (`snptab.count != M`, line 303), `run_dataset`
`++g_failures` and returns `true` (a real failure, correctly counted) — but the same `.snp`
file is shared across BOTH datasets (`main:507`, one `snp` path), so a `.snp` whose row count
matches `derived_acc`'s M but not `derived_full`'s M would fail derived_full's row-count
guard.** This is **correct, defensive behavior** (the guard catches a SNP/Q-shape mismatch),
not a bug — recorded to note that the single shared `.snp` is validated independently against
each dataset's M, which is the right thing. No fix. (Adversarial self-check: is sharing one
`.snp` across two datasets with potentially different M a latent bug? No — both derived_acc
and derived_full are derived from the *same* AADR SNP prefix by construction, so M is the
same; the guard would catch it if a fixture violated that. Sound.)

### Testability vs §13

**F-TEST-1 (med, M, parity-safe: yes) — CONFIRMED.** No GoogleTest integration / no per-claim
CTest granularity — a single `add_test` whose pass/fail is the OR of ~14 checks across
2 datasets × 2 precisions. §13/§3 name GoogleTest + `gtest_discover_tests`; this file (like
its siblings, and as the CMake header at 553 documents) is a self-checking `main()` — an
accepted fallback, so not a hard violation. Consequence: one failing claim fails the whole
gate with no per-claim CTest reporting; mitigated by good `check()`/`report_first_diff()`
stdout. Optional fix: under `STEPPE_HAVE_GTEST`, compile the body as `TEST(...)` cases.
Severity med as a §13-conformance gap, low in practice.

**F-TEST-2 (low) — CONFIRMED.** The gate cannot run GPU-free (hard-requires ≥1 CUDA device,
line 514). Correct — it is a GPU parity gate — but the *fixed-order combine arithmetic* is
proven here only on GPU, while `combine_f2_partials_host` could *also* be unit-tested
host-only against hand-built partials (no GPU). Not this file's job; recorded.

### Capability-tier coherence vs §11.4 / §12

**F-CAP-1 (informational — sound, and the MODEL for F-COV-1) — CONFIRMED.** Lines 481–488:
on PRO tier the gate asserts both `can_access_peer == true` AND `resG2p.last_combine_path ==
P2pDeviceResident`. The second assert is the crucial one — it confirms claim (3) genuinely
ran the `cudaMemcpyPeer` combine, closing the "did P2P silently fall back?" hole for the
*transport*. This is exactly the design F-COV-1 asks for on the *precision* axis (assert the
path engaged, not just that the bits matched). The gate does this for the combine-transport
tag but **not** for the emulation-honorability tag — the F-COV-1 asymmetry.

**F-CAP-2 (low) — CONFIRMED.** `resG2h.last_combine_path == HostStaged` is asserted
unconditionally (lines 478–479, correct — `prefer_p2p_combine=false` forces it on every
tier), but the gate never asserts `resG1.last_combine_path`. Per `resources.hpp:46–50,110`
and `f2_blocks_multigpu.cpp:88–91` (G==1 returns before the §4 fork), `resG1`'s tag stays
`None` (the value-initialized default). A future bug that set the G==1 tag would go
unnoticed. Trivial to add: `check("resG1 tag == None (G==1 fast path sets no combine tag)",
resG1.last_combine_path == CombinePath::None);`. Severity low; tightens the tag contract.

## Considered & rejected (incl. rejected-for-parity)

- **REJECTED-FOR-PARITY: "use a tolerance for the EmulatedFp64 claims to be robust to
  cross-device GEMM noise."** Defeats the purpose — §12's law is bit-*identity* (`memcmp`),
  not tolerance, for the EmulatedFp64 production path. Fixed-slice Ozaki is batchCount-
  deterministic, so bit-identity is the *correct* assertion; any tolerance would mask a real
  sharding bug. **Rejected — would break the §12 bit-identity contract.** (The native-Fp64
  cell legitimately uses tolerance; that is the honestly-scoped exception — F-NUM-3.)

- **REJECTED: "cache the per-device partials and feed both the host-staged and P2P combine
  to avoid running the per-device GEMMs twice."** Tempting (G2h and G2p run the *same*
  per-device partials; only the combine differs). But the gate calls
  `compute_f2_blocks_multigpu` as a black box twice *precisely to prove the public entry
  point is deterministic per config* and that the two `prefer_p2p_combine` settings produce
  bit-identical results *through the real code path*. Caching partials would test a
  refactored harness, not the production seam — and the partials' bit-identity across the two
  calls is itself part of what makes claim (4) meaningful. **Rejected — preserves gate
  integrity.** (This is why F-PERF-5/F-IDIOM-5 are recorded as *not removable*.)

- **REJECTED: "parallelize the per-device fan-out / overlap the sequential `cudaMemcpyPeer`
  via `cudaMemcpyPeerAsync` + events to speed the gate."** Out of scope for *this file* —
  the fan-out is in `f2_blocks_multigpu.cpp:130` and the sequential P2P copies are in
  `p2p_combine.cu` (this TU launches neither). I confirmed via the docs that
  `cudaMemcpyPeer` is the *synchronous* byte copy and a `cudaMemcpyPeerAsync` exists — so the
  copy/compute overlap IS a real perf lever **in the entry-point/combine units**, where it is
  parity-relevant: the §11.4 fixed g=0..G-1 *combine order* must be preserved regardless of
  transport concurrency, and the entry-point header already defends sequential fan-out as
  parity-neutral and defers parallelism. **Rejected for this unit (wrong file); noted for the
  entry-point/p2p-combine review.**

- **REJECTED: flagging `cudaGetDeviceProperties` as a deprecated-API use.** It compiles
  correctly under CUDA 13 and is the conventional spelling. The first pass's "`_v2` macro"
  justification I could **not** confirm from the live docs this pass, but the conclusion is
  unchanged: **no defect.** (See F-IDIOM-1, sub-claim downgraded for honesty.)

- **REJECTED: flagging the high comment density as bloat.** For a §12 gate the dense
  per-claim commentary is a feature. I flag only the inline-restatement-can-drift risk
  (F-READ-1), now *proven* by F-DOC-1 and F-NUM-1.

- **REJECTED: flagging `bit_equal` using `==` for `block_sizes` while f2/vpair use
  `memcmp`.** Correct and deliberate: `block_sizes` is `vector<int>` (`==` is exact integer
  equality); f2/vpair are `vector<double>` where `memcmp` is the bit-exact predicate (`==`
  on doubles treats `+0.0 == −0.0` as equal and `NaN != NaN`, both wrong for a bit-identity
  gate). **No defect — good pattern to keep.**

- **REJECTED-FOR-PARITY: "the combine could sum partials in any order since blocks are
  disjoint (placement, not reduction)."** True for the *current* block-aligned disjoint shard
  plan, but §12 mandates the fixed g=0..G-1 *sum* form precisely so it stays parity-safe if
  shards ever overlap, and `x + 0.0 == x` keeps the disjoint case bit-exact. The gate must not
  assume disjointness. **Rejected — the fixed-order sum is the contract.**

- **REJECTED (my own NEW candidate, withdrawn): "the native lane is missing a strict
  candidate-vs-candidate check."** On re-read, claim (4) (`bit_equal(candG2h, candG2p)`, lines
  442–458) IS asserted strict-bit-identical for the native lane too — correctly, because the
  combine transport is byte-only and the per-device GEMM is identical in both candidates.
  **No defect** (F-BUG-5). Withdrawn.

- **REJECTED (first pass's F-NUM-1 *claim*, not the finding): "kNativeFp64ParityRelTol matches
  the sibling's 1e-9 native tier."** The sibling's native tier is `1e-8` (combined metric,
  atol 1e-9), not 1e-9. The finding stands (and is upgraded to med) but the *justification* was
  wrong and is corrected in F-NUM-1.

## What it takes to reach 10/10

Perf (test-runtime + coverage-honesty):
1. **F-PERF-1** Build the device-0 reference backend ONCE before the precision loop (keep it
   a distinct instance from `resG1` per F-BUG-4) — removes 2–4 `cublasCreate`/workspace
   round-trips. (S)
2. **F-PERF-3** Single label-formatting helper (folded into F-DEC-1). (S)

Correctness / coverage (load-bearing):
3. **F-COV-1 (HIGH)** Assert EmulatedFp64 actually engaged — under `STEPPE_HAVE_EMU_TUNING`,
   `resG1.gpus[0].caps.emulated_fp64_honorable == true` AND `memcmp(emuRef.f2, natRef.f2) !=
   0`, via the **CUDA-free probe field** (do NOT include `f2_block_kernel.cuh`; preserve
   F-LAY-2). Closes the silent-native-fallback blind spot. (S)
4. **F-BUG-1** Bound `report_first_diff`'s vpair loop by the vpair sizes (or assert
   `f2.size()==vpair.size()` at entry). (S)
5. **F-BUG-2 / F-BUG-3** Distinguish absent (skip) from malformed/invalid-shape (FAIL) in
   `load_qvn` (return `{Absent,Malformed,Ok}`); validate `P>0 && M>0`. Stops a truncated
   derived_full from being a silent skip on a parity gate. (S)
6. **F-IDIOM-2 / F-MAG-1** Replace the "PRO" device-name dispatch with the capability
   biconditional ("`can_access_peer` ⇔ `last_combine_path == P2pDeviceResident`"), valid on
   every tier; removes the marketing-string coupling and the false-FAIL risk. (S)

Numerics honesty:
7. **F-NUM-1 (corrected)** Either correct the false "mirrors the sibling tier" comment to the
   *actual* relationship (absolute-1e-9 under a floor-at-1.0 metric, tighter than the
   sibling's off-diagonal combined 1e-8), OR genuinely single-source the sibling's
   `kTolNativeVsRef` + combined metric. (S)
8. **F-DOC-1** Reword the native-Fp64 rationale to the documented form (accumulation-shape
   sensitivity at a different cuBLAS *configuration*, not a nonexistent documented
   "batchCount knob"); keep `[UNCERTAIN]`. (S)
9. **F-DOC-2** Document — and ideally assert (equal `compute_major/minor` across shards) —
   the "same architecture and SM count" precondition the EmulatedFp64 cross-device claim
   rests on. (S doc; M if a `caps.sm_count` field must first be added — it is absent today.)
10. **F-NUM-2** Name the `1.0` relative-denominator floor (`kF2RelDenomFloor`, f2 ∈ O(1));
    reconcile the metric with the sibling (folds into F-NUM-1b). (S)

Cleanliness / structure:
11. **F-DEC-1 / F-DEC-2** Decompose `run_dataset` into `assert_bit_identical`,
    `assert_oracle_tol`, `assert_tier_tags`; flatten the strict/native dispatch (drops it
    under 80 lines, kills the 4-way duplication, homes F-COV-1 + F-PERF-3). (M)
12. **F-IDIOM-3** Clean-error-state sweep (`cudaDeviceSynchronize` + `cudaGetLastError`)
    before `RESULT: PASS`. (S)
13. **F-IDIOM-1** WARN (don't silently return false) if `cudaGetDeviceProperties` fails in the
    tier probe. (S) — moot if F-IDIOM-2 removes the function.
14. **F-READ-3** Thread the failure count (or a `TestResults&`) instead of the file-static
    `g_failures`, to honor §2's "no global mutable state" to the letter. (S)
15. **F-CAP-2** Assert `resG1.last_combine_path == None` (G==1 sets no combine tag). (S)
16. **F-EDGE-1 (process)** Pin the gate to a ≥2-GPU CI runner; make the G==2-skip banner loud.

Optional §13 conformance:
17. **F-TEST-1** Compile the body as GoogleTest `TEST` cases under `STEPPE_HAVE_GTEST` for
    per-claim CTest granularity.

## Good patterns to keep

- **`memcmp` for doubles, `==` for the integer `block_sizes`** (`bit_equal`, lines 168–175) —
  the correct bit-identity predicate, NaN/−0.0-safe where it must be. The heart of the gate,
  done right.
- **The which-path tag read OUT-OF-BAND from `Resources`, never from `F2BlockTensor`**
  (F-LAY-1) — and the gate asserts the P2P arm *actually took P2P* (`last_combine_path ==
  P2pDeviceResident`, F-CAP-1), not merely that the bits matched. The model the
  emulation-honorability check (F-COV-1) should follow.
- **Honest *structure* of the native-Fp64 oracle-tolerance cell** — the gate does NOT pretend
  native Fp64 is bit-identical at G≥2; it carves out that one cell, asserts
  shape/`block_sizes` EXACTLY (a partition difference is always a bug, never GEMM noise), and
  reports the observed worst relative deviation so the noise level is visible. The *structure*
  of the exception is right; the *value/provenance* of its tolerance needs the F-NUM-1 fix and
  the *reasoning* needs the F-DOC-1 reword.
- **The two-tier neutrality claim (4) asserted as STRICT bit-identity for BOTH precisions**
  (lines 442–458) — correctly recognizing the combine transport only moves bytes and sums the
  same fixed order, so it is bit-identical regardless of per-device GEMM precision. Verified
  against `p2p_combine.hpp:24–31` (transport-only) — the reasoning is correct and load-bearing.
- **The G==1 floor as a structural no-op proof** — driving `compute_f2_blocks_multigpu` at
  G==1 and asserting bit-identity to the single-GPU reference proves the multi-GPU codepath
  adds nothing to the value path at G==1 (claim 1). Cheap; catches G==1-special-case
  regressions.
- **The reference uses a *distinct* device-0 backend instance from the G==1 candidate**
  (F-BUG-4) — proving bit-identity across independent cuBLAS handles/workspaces on the same
  device, a stronger property than reusing one object. (Preserve this when applying F-PERF-1.)
- **Skip policy: absent dataset → logged skip; missing 2nd GPU → logged skip of G==2; no
  device → return 1.** Sound — lets the fast gate run without the full matrix (modulo F-BUG-2's
  absent-vs-malformed conflation, which must be fixed).

---

### Sources (CUDA behavior claims, re-verified this pass)

- cuBLAS Results Reproducibility: bit-wise reproducibility holds "when executed on GPUs with
  **the same architecture and the same number of SMs**" (single CUDA stream, single toolkit
  version); the guarantee "no longer holds when multiple CUDA streams are active" and when
  atomics (`cublasSetAtomicsMode`) / fixed-point emulation are used; the docs do **not**
  mention `batchCount` as a reproducibility/algorithm-selection axis —
  https://docs.nvidia.com/cuda/archive/13.0.0/cublas/index.html (Results Reproducibility) and
  https://docs.nvidia.com/cuda/cublas/
- CUDA Runtime API `cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
  size_t count)` — "Copies memory between two devices"; a raw `count`-byte copy with no data
  conversion; a separate `cudaMemcpyPeerAsync` exists (so the base form is host-synchronous) —
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
- CUDA Runtime API `cudaGetDeviceProperties(cudaDeviceProp* prop, int device)` — queries by
  device ordinal (no `cudaSetDevice` required); compiles under CUDA 13 as the conventional
  spelling (the `_v2` macro mapping asserted by the first pass could NOT be confirmed from the
  live device-management page this pass and is irrelevant to the verdict) —
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
