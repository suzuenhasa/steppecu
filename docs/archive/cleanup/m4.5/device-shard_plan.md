# M4.5 unit review (ADVERSARIAL second pass) — `device/shard_plan.{hpp,cpp}`

The block-aligned, SNP-count-balanced shard planner: the host-pure, CUDA-free
planning step the single-node multi-GPU (SPMG) precompute drives against. Scope is
the ONE unit `plan_block_shards` + its `DeviceShard` value type.

Adversarial re-audit of the first-pass draft (which scored 9/10, 22 findings, 4
perf-findings). Every existing finding was re-verified line-by-line against the
actual source and, where it depends on documented behavior, against the C++ standard
draft (eel.is) and the steppe architecture. Read for context (the interfaces it
implements / is consumed by): `core/domain/block_partition_rule.hpp` (`BlockRange`,
the `block_ranges` inverse it is parallel to), the sole caller
`core/fstats/f2_blocks_multigpu.cpp`, both downstream consumers of
`DeviceShard.{b0,b1,s0,s1}` (`core/fstats/f2_combine.cpp`,
`device/cuda/p2p_combine.cu`), `device/resources.hpp` (`device_count()` →
`std::size_t`, `CombinePath`), the actual GEMM backend `device/cuda/cuda_backend.cu`
+ `device/cuda/f2_blocks_kernel.cu` (the size-bucketed strided-batched cost model),
and `core/internal/launch_config.hpp` (`cdiv`) + `core/internal/host_device.hpp`
(`STEPPE_ASSERT`) for the proposed DRY fixes. Standard judged against:
architecture.md §2, §4, §7, §8, §9, §11.3/§11.4, §12.

**Headline correction from this pass:** the previous draft's perf finding **P-2**
asserted the per-device GEMM cost is `n_block_local × P² × max_block_size_in_shard`
("pads every block in its shard to the shard's maximum block size `s_pad`"). That is
**factually wrong about the backend** and I am correcting it (see P-2): the backend
(`cuda_backend.cu:260-280`, `f2_blocks_kernel.cu:11-16`) does **size-bucketing** —
it buckets blocks by `ceil-pow2` (`kBlockGroupPadBase`) of size and pads each bucket
only to *that bucket's* width, one strided-batched call per bucket, with measured
1.43× pad waste *within a bucket* (the spike explicitly **rejected** the global-max
pad as 2.76× and slower). So the cost is `Σ_buckets (n_in_bucket × s_pad_bucket × P²)`,
not the shard's single max. The general thrust survives (SNP count is an imperfect
proxy) but the magnitude and the formula in the original P-2 were overstated.

## Role & layering

`plan_block_shards` is a **deterministic pure function** of `(block_sizes, ranges, G)`
that partitions the `n_block` jackknife blocks into `G` CONTIGUOUS *whole-block*
ranges balanced by SNP count, returning exactly `G` `DeviceShard` entries in the fixed
`g = 0..G-1` order. The whole-block-on-one-device invariant is the *structural* floor
the §12 PARITY LAW rests on (same SNP columns ⇒ same feeder bits ⇒ same `s_pad`
bucket ⇒ same strided-batched slab ⇒ bit-identical slab to single-GPU). The combine
layer (`f2_combine.cpp` host-staged, `p2p_combine.cu` device-resident) then sums the
per-device partials in fixed `g` order onto a zero tensor.

**Layering is correct and clean — re-verified.** CMake confirms `shard_plan.cpp` is a
plain `.cpp` (not `.cu`) in `steppe_device` (`device/CMakeLists.txt:30`, and the
comment at `:33` calls out the plain-`.cpp` intent). The header names only
`<cstddef>/<span>/<vector>` + `core/domain/block_partition_rule.hpp` (for
`core::BlockRange`), reaches no GPU, includes no `<cuda_runtime.h>`, and is consumed
unchanged by `steppe_core` (`f2_combine.cpp`, `f2_blocks_multigpu.cpp`) and by
`steppe_device` (`p2p_combine.cu`). This is the same CUDA-free-decl-in-device-layer
pattern as `backend.hpp` / `resources.hpp` / `p2p_combine.hpp`. The dependency
direction (`core` reaches *down* into a `device/` header for the value type;
`device` cannot link `steppe_core`) is respected. `BlockRange` is reached through the
header-only `steppe::core_internal` INTERFACE both layers share — verified
(`core/CMakeLists.txt:24` declares it INTERFACE, `device/CMakeLists.txt:48` links it
PUBLIC). **No layering defect.** A *placement* nuance (whether the value type should
live next to `BlockRange`) is discussed at L-1 — judgment call, not a violation.

The unit is small (118 lines `.cpp`, one function; 100 lines of richly-documented
header). The parity-critical contract (whole blocks, fixed order, contiguous tiling,
exactly-G entries) is upheld and the empirical tiling is exact across the degenerate
cases. The findings below are weighted toward **balance-quality (perf), cost-model
accuracy, casting hygiene, a couple of asserted-not-enforced contract edges, and a
missing unit test** — not correctness regressions against the locked parity gate.

## Score: 9/10 — near-exemplary host-pure planner; held off 9.5–10 by a real but narrow balance-quality blind spot (prefix-threshold greedy idles devices on skewed inputs), a scattered `int`/`long`/`size_t` cast triangle at the value-type seam that a `block_sizes`-parameter removal would mostly delete, and a missing GPU-free unit test for the one unit cheapest to test in isolation.

Rationale unchanged in magnitude from the first pass (9/10 stands), but the
*composition* of the deduction shifts after verification: the first draft's P-2
(cost model) was over-claimed and is downgraded to LOW; the balance blind spot (P-1)
is real but its *real-data* severity is milder than "HIGH" once you account for the
size-bucketing backend (a balanced SNP split does **not** balance GEMM cost anyway —
the bucketing is per-device and pad-waste-dominated), so P-1 is the genuine half-point
but it is a *narrow-input* throughput issue, not a general one. The other half is the
cast scatter + the missing unit test (the unit's own CUDA-free testability promise is
unrealized) + a handful of asserted-not-enforced edges. Layering, RAII (trivially),
determinism, single-home, doc-density, and `[[nodiscard]]`/`noexcept` are all clean.

---

## Findings

### Performance & balance quality (FIRST-CLASS this pass)

This is a host-pure planner — there are **no kernels, no copies, no streams, no syncs,
no allocations beyond the one result `std::vector(G)`** — so the classic perf hunt
(data bouncing / grid-stride / sequential-P2P / default-stream / hot-path allocation /
pinned memory) is **N/A *inside* this file**. I confirmed this exhaustively: one
`std::vector(G)` allocation (`:43`), two O(n_block) integer loops, zero inner
allocation, zero copy, zero recompute, no host round-trip. The function is trivially
cheap relative to the GEMMs. "Performance" here means **the quality of the output
partition** (how well the G GPUs are utilized in the precompute) and **cost-model
fidelity**, which is where the real findings are.

**P-1 — The greedy is a prefix-threshold split, not a max-load-balanced partition: a
single dominant block (especially trailing or standalone) idles GPUs. (CONFIRMED;
downgraded HIGH→MED-HIGH / effort M / PARITY-SAFE: yes-if-careful.)**
`shard_plan.cpp:81-109`. The greedy closes device `g` the first time running
`device_snps` crosses `target_per_device`, walking blocks left-to-right; the *last*
device absorbs everything remaining. This correctly tiles `[0,n_block)` but it is
order-dependent and blind to a heavy block that appears after the threshold logic has
committed the layout. The worst case is a giant *last* block: the `(b+1 < n_block)`
guard at `:89` (which exists for OOB safety — see C-4) prevents any close on the final
block, so a `{1,1,1,1,1000}` / G=3 partition puts all 1004 SNPs on device 0 and idles
devices 1 and 2. A giant *first* block `{1000,1,1}` / G=3 gives `dev0=1000, dev1=2,
dev2 idle`. *Why it matters:* §11.4 sells multi-GPU as "parallelism + speedup across
the G devices"; the GEMM compute IS the throughput lever (the combine is explicitly
NOT — §11.4 line 717: "off the critical path … architectural cleanliness, not a
throughput lever"). A pathological partition degrades GEMM throughput toward
single-GPU while still claiming G devices, *silently* (no warn, no error). The
header's own "proof" (`shard_plan.hpp:62-69`) proves only *correctness* (≤ G ranges,
every block placed), NOT balance.

*Concrete fix (parity-safe):* keep whole-block-aligned + fixed `g=0..G-1` placement
(do NOT touch the combine order — that is the parity line); replace the prefix
threshold with a **balanced contiguous partition that minimizes the max device load**
— the standard "linear partition into G contiguous parts minimizing the maximum part
sum" via binary search on the capacity + a feasibility greedy (`O(n_block·log(total))`),
still a *contiguous whole-block* split. Or, minimally, a forward-looking close that
considers the *next* block before sweeping a giant one into an already-full device.
Either keeps the shard contiguous and the combine fixed-order, so bit-identity is
untouched (a block is bit-identical no matter which device computes it).

*Adversarial check (this pass), which is why the severity drops from HIGH to MED-HIGH:*
(1) Real AADR interior blocks are near-uniform 5 cM; only the per-chromosome tail
blocks (`block_partition_rule` resets at each chromosome boundary) are short, and a
short *tail* block does not trigger the failure (it is the *giant* block that does,
and AADR has no single dominant block). So the pathology is largely synthetic /
merged-dataset, not the AADR fixture. (2) Crucially, **even a perfectly SNP-balanced
contiguous split does not balance the actual GEMM cost** — the backend buckets each
device's blocks by pow2 size and the cost is pad-waste-dominated *per device* (see
P-2), so "balance by max SNP load" is itself only an approximation. The right
objective, if pursued, is a cost model that mirrors the bucketing (P-2), not raw SNP
balance. So P-1's fix is worth doing for the *idle-device* pathology specifically (a
GPU getting **zero** work is unambiguously wrong on a skewed input), but its expected
benefit on real runs is modest. **MED-HIGH** (idle-GPU on skew is real; real-data
impact mild); **M** effort; parity-safe if contiguous + fixed-order preserved.

**P-2 — Cost model is SNP-count only; the real GEMM cost is the size-BUCKETED padded
work, NOT the shard-max pad the first draft claimed. (CORRECTED & DOWNGRADED:
HIGH→LOW / effort M / PARITY-SAFE: yes.)**
`shard_plan.cpp:57-68`. The balance target is `total_snps / G`. The first-pass draft
claimed the per-device GEMM "pads every block in its shard to the shard's *maximum*
block size `s_pad`, and pays `n_block_local × P² × s_pad`." **That is wrong** — I
verified the backend: `cuda_backend.cu:260-280` buckets blocks by `ceil-pow{
kBlockGroupPadBase}(size)` and issues **one strided-batched call per bucket, padded
only to that bucket's pow2 width** (`f2_blocks_kernel.cu:11-16`: "grouped: bucket
blocks by ceil-pow2 of size … padded only to the bucket width … 1.43× pad waste (vs
2.76× global)"). The global-max-pad cost the draft described is precisely the design
the spike **rejected**. So the true per-device cost is roughly
`Σ_buckets (n_in_bucket × s_pad_bucket × P²)` (plus per-bucket launch/setup), which is
*closer* to the SNP count than the shard-max model implies — the dominant non-linearity
is the per-bucket 1.43× pad factor, which is roughly uniform across devices for similar
size distributions. *Why it still matters (the surviving kernel of truth):* the SNP
count is not exactly the GEMM cost (two devices with equal SNP totals but different
size *distributions* hit different pow2 buckets and accrue different pad waste, and pay
different per-bucket fixed costs), so raw SNP balance can under/over-shoot real GEMM
time. But the gap is **second-order** (≤ ~1.43× pad spread, not the up-to-2× the
draft's max-pad model implied), so this is a documentation/tuning nit, not a perf bug.
*Concrete fix:* either (a) document in `shard_plan.hpp:59-65` that SNP count is a
*proxy* and the true cost is the per-device size-bucketed padded work (so a future
tuner knows where the slack is), or (b) if P-1 is implemented with a real cost model,
cost a candidate shard by the bucketed work, not raw SNP sum. Both parity-safe
(changes only *which whole blocks* go where). **LOW** (was HIGH; the magnitude was
over-claimed). **The header should NOT keep presenting SNP-count balance as the true
objective without the proxy caveat.**

**P-3 — Trailing empty shards on skewed `n_block ≥ G` inputs cost a `cudaSetDevice` +
empty-tensor round-trip per idle device. (CONFIRMED / effort S / PARITY-SAFE: yes.)**
When the greedy idles a device (P-1), the orchestrator still calls
`backend->compute_f2_blocks` with `n_block_local == 0` for it
(`f2_blocks_multigpu.cpp:130-153`), which early-returns — but it is still a
`cudaSetDevice` + an empty-tensor allocation + a backend round-trip per idle device.
Not this unit's bug to *fix* (the backend degenerate guard is correct and the empty
shard is a legitimate output), but the planner is the right place to *avoid creating*
needless empty shards when a balanced split would use all G. Folds into P-1's fix.
**LOW.**

**P-4 (confirmation, not a defect) — the single O(n_block) pass + the absence of any
allocation beyond the result `std::vector(G)` are exactly right.** Re-verified:
`:43` allocates the result once; the total-snps loop (`:58-60`) and the assignment
loop (`:81-100`) are O(n_block) with no inner allocation/copy/recompute; `block_ranges`
is passed in, not re-derived. The function is trivially cheap relative to the GEMMs.
**Keep.**

### Correctness & bugs

**C-1 — Element-wise parallel-array contract (`ranges[b].size() == block_sizes[b]`) is
documented as a precondition but only the *lengths* are checked. (CONFIRMED / effort
S / PARITY-SAFE: yes.)**
`shard_plan.hpp:84-86` states "MUST be parallel to `block_sizes`
(`ranges[b].size() == block_sizes[b]`)", but `shard_plan.cpp:35-40` validates only
`block_sizes.size() == ranges.size()`. The balance uses `block_sizes[b]` for the
target math (`:59`, `:82`) while the shard's `[s0,s1)` comes from `ranges` (`:94-95`,
`:108-109`). Mismatched-but-equal-length arrays (e.g. `block_sizes` from a different
partition) would compute the balance on one and the column spans on another — a
silently wrong shard, not a crash. The sole caller derives both from the same
`block_ranges` result (`f2_blocks_multigpu.cpp:104-108`: `block_sizes[b] =
ranges[b].size()`), so it cannot trigger today — but the function is the single home
and "owns the guard" by its own logic (`shard_plan.cpp:26-27`). *Why it matters:* §2
fail-fast at the contract boundary. *Concrete fix:* either (a) a debug `STEPPE_ASSERT`
that `ranges[b].size() == block_sizes[b]` for all b — `STEPPE_ASSERT` is reachable
from this CUDA-free `.cpp` via `core/internal/host_device.hpp` (verified CUDA-free,
linked PUBLIC through `steppe::core_internal`; `p2p_combine.cu` already includes it) —
or, cleaner, (b) **drop `block_sizes` entirely** and derive from `ranges[b].size()`
(D-1), removing the whole mismatch class. **MED** (latent silent-wrong-result), **S**.

**C-2 — `int` narrowing of `b0`/`b1`/`n_block` at every `DeviceShard` construction is
unchecked, and the header is silent on the `int` vs `long` width split. (CONFIRMED /
effort S / PARITY-SAFE: yes.)**
`shard_plan.cpp:92-95`, `:106-109`: `static_cast<int>(b0/b1/n_block)` where they are
`std::size_t`; `DeviceShard.b0/b1` are `int` (`shard_plan.hpp:39-40`). Safe in
practice — the tree-wide contract is that `n_block` is `int`-bounded (≈584k whole-genome
AADR, O(1e3) occupied; `BlockPartition::n_block` is `int`, `backend.hpp` takes
`int n_block`); the `block_partition_rule` review adjudicated this exact pattern as "a
conscious tradeoff." But it is an *unchecked* narrowing of a `size_t` from
`span::size()`, and the header gives no rationale at `:39-40` for why `b0/b1` are `int`
while `s0/s1` are `long`. *Concrete fix:* a one-line debug `STEPPE_ASSERT(n_block <=
static_cast<std::size_t>(std::numeric_limits<int>::max()))` at entry + a one-sentence
doc note on `DeviceShard` explaining the `int` (block ids, bounded) vs `long` (SNP
columns, M-scale) split — mirroring `BlockRange`'s own "Widths are `long` to match
`MatView::M`" note (`block_partition_rule.hpp:150`). **LOW.**

**C-3 — `target == 0` when `total_snps == 0` makes close-on-cross fire on every block;
tiling stays correct but "balance" is vacuous-by-accident. (CONFIRMED / effort S /
PARITY-SAFE: yes.)**
`shard_plan.cpp:67-68`: with `total_snps == 0` (all blocks empty — `BlockRange` permits
empty `[begin,end)` per `block_partition_rule.hpp:182-183`; `block_ranges` "does not
forbid it"), `target_per_device = (0+G-1)/G = 0`, so `reached_target = (device_snps >=
0)` is always true and the greedy closes after *every* block until devices run out;
the last device absorbs the tail. It works (the `(b+1<n_block)` and `(g+1<G)` guards
save it: `{0,0,0,0}` G=3 → `[0,1)[1,2)[2,4)` — exact tiling, one block per device then
the rest). `assign_blocks` never produces a zero-size block, but `block_ranges`'
contract explicitly allows hand-built/merged-dataset partitions with empty blocks, so
the single-source planner should be robust to its documented input domain. The header
comment "`>= 1 when total_snps >= 1`" (`:68`) acknowledges the `== 0` case exists but
not what balance *means* there. *Concrete fix:* one sentence ("with `total_snps == 0`
every block closes a device; tiling is still exact, the balance is vacuous"). **LOW**
— readability/contract-clarity nit.

**C-4 (verified-not-a-bug, with a STRONGER reason than the first draft gave) — the
final unconditional close at `:105-109` is always reachable, never double-writes, and
the `(b+1 < n_block)` guard is LOAD-BEARING for OOB safety, not cosmetic.** The first
draft noted the final close "always runs because the loop never closes the last
device." I traced *why the `(b+1 < n_block)` guard is required for memory safety*, not
just for leaving a trailing run: if the greedy were allowed to close on the very last
block (`b == n_block-1`), it would advance `g` and set `b0 = n_block`, and then the
final close at `:105-109` would read `ranges[b0] = ranges[n_block]` — **one past the
end** of the `ranges` span. The `(b+1 < n_block)` predicate is what guarantees
`b0 < n_block` at the final close, so `ranges[b0]` and `ranges[n_block-1]` are always
in-bounds. Concretely, without the guard, `{2}` / G=2 (target=1) would close plan[0]
on b=0, set `b0=1`, then read `ranges[1]` (OOB; only `ranges[0]` exists). With the
guard, b=0 does NOT close, the final close writes plan[0]={0,1}, and plan[1] stays
empty. So the guard is a correctness invariant. **No defect — and the header/`.cpp`
comment under-sells it** (it frames the guard as "leaves the trailing run for here",
not "prevents an OOB read at the final close"). A one-line comment upgrade is
warranted (folds into C-3's documentation nit). The `:104` comment is correct but
incomplete.

**C-5 (verified-not-a-bug) — `n_block < G` produces leading non-empty + trailing empty,
never a mid-range empty.** Re-traced `{10,10,10}` G=5 → `[0,1)[1,2)[2,3)[empty][empty]`.
The empties are exactly the trailing devices, matching the header contract
(`shard_plan.hpp:76-78`) and the orchestrator's empty-shard early-return. **No defect.**

### Edge cases & failure modes

**E-1 — No guard against a negative `block_sizes[b]`; a malformed element drives
`target_per_device` negative and collapses the split onto device 0. (CONFIRMED /
effort S / PARITY-SAFE: yes.)**
`shard_plan.cpp:57-60` accumulates `block_sizes[b]` (an `int`) into `long total_snps`.
The comment (`:54-56`) correctly argues whole-genome M ≈ 6e5 ≪ LONG_MAX, so the *sum*
cannot overflow for any realistic partition — TRUE, re-verified. But `block_sizes` is
`std::span<const int>` with no sign guard: a negative element would make `total_snps`
smaller than the true count, and a sufficiently negative total drives
`target_per_device` negative, after which `device_snps >= target` is true immediately
and the split collapses onto device 0. The sole caller derives `block_sizes` from
`ranges[b].size()` (= `end - begin ≥ 0` for valid `block_ranges` output, and
`block_ranges` *validates* non-decreasing ids so `end ≥ begin`), so unreachable today.
*Concrete fix:* fold into C-1's element-wise assert (`block_sizes[b] >= 0` alongside
`== ranges[b].size()`), or drop the parameter (D-1) since `BlockRange::size()` is
non-negative for a contiguous validated range. **LOW** (unreachable from the current
caller).

**E-2 — The `size_t G` ↔ `long` boundary at `:66` and the `g + 1 < G` unsigned
comparisons are safe ONLY because `G ≥ 1` is enforced first. (CONFIRMED /
informational.)**
`shard_plan.cpp:66` casts `G` to `long G_signed` for the ceiling division; `:87`
compares `g + 1 < G` (both `std::size_t`). After the `G == 0` throw (`:28`), `G ≥ 1`,
so `G_signed ≥ 1` (no div-by-zero) and `g + 1 < G` is well-defined (`g < G` always, so
`g+1` cannot wrap for any `G ≤ SIZE_MAX`; and `g < device_count() = gpus.size()`, a
handful of GPUs). **No defect** — but it is one more spot where the `size_t G` ↔ `long`
boundary is crossed ad hoc (see T-1/T-2). Informational.

**E-3 (verified-not-a-bug) — Empty input (`n_block == 0`) early-returns G empty shards;
the lone `plan(G)` allocation is the right shape.** `shard_plan.cpp:48-50` → `{0,0,0,0}`
× G. Downstream `validate_partials` accepts it (`covered == 0 == n_block_full`,
`f2_combine.cpp:67-72`). **No defect.**

### Type-casting noise (FIRST-CLASS this pass)

**T-1 — The `int`(block ids) ⇄ `std::size_t`(loop/index) ⇄ `long`(SNP columns)
triangle is re-cast at six sites and never centralized; the value type forces a cast
at every construction. (CONFIRMED / effort M / PARITY-SAFE: yes.)**
Enumerated and re-verified against the source:
- `:59` `static_cast<long>(block_sizes[b])` (int→long, accumulate total).
- `:66` `static_cast<long>(G)` (size_t→long, for ceil-div).
- `:82` `static_cast<long>(block_sizes[b])` (int→long, again — duplicate of `:59`).
- `:92-93` `static_cast<int>(b0)`, `static_cast<int>(b1)` (size_t→int, narrowing).
- `:106-107` `static_cast<int>(b0)`, `static_cast<int>(n_block)` (size_t→int, narrowing).
- `:94-95`, `:108-109` `ranges[b0].begin` / `ranges[b1-1].end` are `long` into `long
  s0/s1` (clean — no cast).
A single function juggles three integer widths and casts at six sites. The root cause:
`DeviceShard` mixes `int b0/b1` (to match the `int n_block` ABI of `backend.hpp` /
`F2BlockTensor`) with `long s0/s1` (to match `MatView::M` / `BlockRange`), while loop
indices are `std::size_t` (to match `span::size()`, which is `size_type = std::size_t`
— **verified against [span.overview]: `using size_type = size_t;`** via
eel.is/c++draft). Each axis is individually defensible, but the *scatter* of casts is
exactly the "casting noise" the prompt targets: the narrowing casts hide the unchecked
`int` bound (C-2), and the `int→long` of `block_sizes[b]` appears twice identically.
*Concrete fix:* (a) D-1 (drop `block_sizes`, derive from `ranges[b].size()` which is
already `long`) removes both `block_sizes` casts (`:59` and `:82`) outright; (b) a tiny
private helper `DeviceShard make_shard(std::size_t b0, std::size_t b1, std::span<const
BlockRange>)` doing the two narrowing casts in *one* audited place with the C-2 assert,
so the two construction sites (`:91`, `:105`) stop repeating the cast quartet. Keep the
`b0/b1` `int` vs `s0/s1` `long` choice (it mirrors `BlockRange`); the goal is to
*centralize* conversions, not change types. **MED** for the scatter; **M** effort.

**T-2 — `(total_snps + G_signed - 1) / G_signed` ceiling division is correct, but
`core::cdiv(long,long)` already homes this and is reachable. (CONFIRMED / effort S /
PARITY-SAFE: yes.)**
`shard_plan.cpp:67-68`. Verified the algebra: for `total ≥ 0`, `G ≥ 1`,
`(total + G - 1)/G == ceil(total/G)` in integer arithmetic with no overflow (`total ≤
~6e5`, `G` small) — standard and correct. **Verified `cdiv(long, long)` exists**
(`launch_config.hpp:89`) and is the architecture-named single home of ceiling division
(§8 table line 529: "`cdiv` (int + long)"). It is CUDA-free (`launch_config.hpp`
includes only `host_device.hpp` + `config.hpp`, no CUDA), in `steppe::core_internal`,
linked PUBLIC into `steppe_device`, and already included by `p2p_combine.cu` — so this
`.cpp` can include it legally. Using `core::cdiv(total_snps, G_signed)` removes the
hand-rolled ceiling-div *and documents intent* at the call site (it reads as "ceiling
divide", not as opaque arithmetic). The `size_t→long` cast of `G` is still needed
(`cdiv(long,long)` takes `long`), but that is one cast vs the inline expression. DRY
win, parity-safe (byte-identical result), layering-legal. **LOW** — nice-to-have.

### Decomposition / single-responsibility

**D-1 — `block_sizes` is a redundant parameter derivable from `ranges`; removing it
collapses C-1, E-1, half of T-1, AND a caller-side `long→int` narrowing. (CONFIRMED &
STRENGTHENED / effort S / PARITY-SAFE: yes.)**
`shard_plan.hpp:93-96` takes BOTH `std::span<const int> block_sizes` AND `std::span<
const BlockRange> ranges`, and the sole caller builds the former *from* the latter.
The planner uses `block_sizes` only for the SNP-count balance, `ranges` only for
`[s0,s1)`. But `BlockRange::size() == end - begin` (`block_partition_rule.hpp:156`)
**is** the SNP count — so `block_sizes` carries no information `ranges` lacks. Keeping
both: (i) forces the unchecked parallel-array contract (C-1); (ii) admits a sign hole
on a separate array (E-1); (iii) doubles the `int→long` cast (T-1); (iv) makes the
caller allocate+fill a redundant `std::vector<int>`. **New observation this pass that
strengthens the case:** building `block_sizes` *also forces a `long→int` narrowing in
the caller* — `f2_blocks_multigpu.cpp:105-106` does `block_sizes[b] = static_cast<int>(
ranges[b].size())`, narrowing the `long` `size()` to `int`. Dropping `block_sizes` and
using `ranges[b].size()` (a `long`) inside the planner removes that caller narrowing
too, so the SNP count stays `long` end-to-end (matching `total_snps`'s type and the
`MatView::M` width — and *removing* a latent truncation if a single block ever exceeded
`INT_MAX` columns, which is more plausible than `n_block` doing so). *Concrete fix:*
drop `block_sizes`; inside the planner use `ranges[b].size()`. One parameter, one
array, zero parallel-array contract, two fewer casts in the planner + one fewer in the
caller, one fewer caller allocation, and the balance math is byte-identical (`size()`
== the old `block_sizes[b]`). *Adversarial check:* any caller with `block_sizes` but
not `ranges`? No — `plan_block_shards` appears only in `f2_blocks_multigpu.cpp` (grep:
only the orchestrator + docs), and the header even *defines* `block_sizes` as "each
`BlockRange::size()`" (`shard_plan.hpp:80-81`). **MED** cleanliness/robustness, **S**,
fully parity-safe. The single best move in this review.

**D-2 (verified-not-a-bug) — Appropriately sized, single-responsibility; no split
needed.** ~75 lines of logic, one job, clearly sectioned (guards → degenerate → target
→ greedy → final close) with banner comments. **No defect.**

### Readability, naming, const-correctness, attributes

**R-1 — `DeviceShard::empty()` is `b0 >= b1` but the contract is `b0 == b1`; the `>=`
silently treats a malformed `b0 > b1` as "empty". (CONFIRMED / effort S / PARITY-SAFE:
yes.)** `shard_plan.hpp:46`. The planner only ever emits `b0 == b1` (value-init) or
`b0 < b1` (a real range), so `>=` and `==` are observationally identical for *its*
output. But `empty()` is a public predicate on a value type; `b0 >= b1` reports a
corrupt `b0 > b1` as "empty" rather than flagging it, and a downstream consumer
(`f2_combine.cpp:49`, `p2p_combine.cu` validate) computes `sh.b1 - sh.b0` as
`span_blocks` and would get a *negative* count from a `b0 > b1` shard, then
`part.n_block != span_blocks` throws a confusing message. *Concrete fix:* keep `>=`
and document "treats malformed `b0>b1` as empty", or assert `b0 <= b1` as an invariant.
**LOW** — purely value-type robustness; the planner never emits a malformed shard.

**R-2 (verified-not-a-bug) — Excellent comment density and `[[nodiscard]]`/`noexcept`.**
`plan_block_shards` is `[[nodiscard]]` (`:93`); `DeviceShard::empty()` is
`[[nodiscard]] … noexcept` (`:46`); the doc enumerates every edge case (G==0/1,
n_block==0, n_block<G) tied to a parity/arch section; the `.cpp` banners section every
phase. `plan_block_shards` is correctly NOT `noexcept` (it `throw`s). **Keep — the
doc-density bar for M4.5.**

**R-3 (verified-not-a-bug) — Variable naming is clear.** `target_per_device`, `g`,
`b0`, `device_snps`, and the named booleans `more_devices_left`/`reached_target`
(`:87-88`) make the close condition self-documenting. `G` is a bare capital but matches
the §11.4/§12 math notation and tree-wide convention. **Good.**

### Numerical / precision vs §12

**N-1 — N/A in the usual sense (no floating-point here), and that is the right design.**
Pure integer arithmetic; the parity floor it provides is *structural*
(whole-block-on-one-device), not numeric. The bit-identity is delivered by the combine
layer's fixed-order FP64 sum onto a zero tensor, not by this unit. Re-verified the
determinism: pure function of `(block_sizes, ranges, G)`, no RNG, no device-state
dependence, no FP — a block lands on the same device every run (the §12 "a block lands
on the same device every run, so its bits are stable" claim, `shard_plan.hpp:55-57`,
holds). The greedy's order-dependence (P-1) is on block *size*, not on anything
nondeterministic, so it does not threaten parity — it only affects balance. **No
precision defect.**

### Capability-tier coherence

**K-1 — N/A to this unit, correctly. The tier tag is off the numeric payload.**
Re-verified: the shard plan carries no combine-path tag; `CombinePath`
(`resources.hpp:46-57`) is recorded out-of-band on `Resources.last_combine_path`
(`f2_blocks_multigpu.cpp` sets it on each branch), never on `DeviceShard` or
`F2BlockTensor`. The SAME `shards` vector feeds both `combine_f2_partials_host` and
`combine_f2_partials_p2p` (`f2_blocks_multigpu.cpp`), which is what makes the two tiers
bit-identical. **No defect — and a good pattern (the plan is the shared parity floor
under both tiers).**

### Testability vs §13

**TST-1 — No dedicated GPU-free unit test for `plan_block_shards`; coverage is only
transitive through the GPU parity test, which cannot hit the skew/edge cases.
(CONFIRMED / effort S / PARITY-SAFE: yes.)**
Re-verified: `grep -rn plan_block_shards tests/` returns only
`tests/CMakeLists.txt:556` (a comment listing it as linked into the parity test's
`steppe::device` dependency) — there is **no `test_shard_plan.cpp`** and **no
`tests/unit/` shard test**. The only coverage is
`tests/reference/test_f2_multigpu_parity.cu`, which (a) requires a GPU and a real AADR
dataset, (b) tests the *whole* pipeline, and (c) therefore only ever hits the balanced
`n_block ≥ G` path on real near-uniform blocks — it would NOT catch P-1's skew
imbalance (no balance assertion), nor the `n_block<G` trailing-empty layout, nor the
target-0 all-empty case. The whole *point* of making this CUDA-free is GPU-free
testability (`shard_plan.hpp:16-18`: "equally exercisable host-only by the parity
test"). *Concrete fix:* add `tests/unit/test_shard_plan.cpp` (no GPU, no dataset)
asserting, for a table of `(block_sizes, G)`: exactly G shards; contiguous; cover ==
n_block; empties are trailing-only; the documented edge cases (G==1, n_block==0,
n_block<G, all-empty/target-0, giant-first/last block); and — if P-1 is fixed — a
max/min load-ratio bound. **MED** (a real coverage gap on the one unit cheapest to test
in isolation), **S** effort.

### Layering / API / placement

**L-1 — Placement nuance (debatable, NOT a violation): a pure block-partition *rule*
could live next to `block_partition_rule.hpp`. (LOW / effort M / PARITY-SAFE: yes.)**
`shard_plan.{hpp,cpp}` lives in `src/device/` and compiles into `steppe_device`, but
it is host-pure CUDA-free and is consumed *from `steppe_core`*
(`f2_blocks_multigpu.cpp`, `f2_combine.cpp`). The header justifies the placement:
"device-layer only by placement … internal SPMG orchestration plumbing … shared by
the device-layer multi-GPU orchestrator and the parity test" (`:13-18`). That is
coherent — it sits with `resources.hpp` / `p2p_combine.hpp` as the SPMG plumbing
cluster, and `DeviceShard` IS consumed by the device-side `p2p_combine.cu`. *Counter
(why moving it would be worse):* moving to `core/domain` would make `device/p2p_combine.cu`
depend *up* on `core/domain` — and the reason `BlockRange` lives in a header-only
`core_internal` INTERFACE both layers share is precisely that `device` cannot link
`steppe_core`. `DeviceShard` is consumed by BOTH layers, so a header-only home
reachable by both (like `block_partition_rule.hpp`'s INTERFACE) is the truly clean
spot; `device/shard_plan.hpp` already achieves "reachable by both." **Currently
defensible; not a violation.** **LOW**, debatable, **M** if pursued.

**L-2 (verified-not-a-bug, with a strict-IWYU nit) — duplicate includes are correct
IWYU style; the `.cpp` is missing `<span>` for strict IWYU.** The `.cpp`'s
`#include <cstddef>` and `#include "core/domain/block_partition_rule.hpp"` (`:12`,
`:17`) duplicate the `.hpp`'s, but IWYU *prefers* this (the `.cpp` uses `std::size_t`
and `BlockRange` directly) — correct style. The `.cpp` does NOT include `<span>` though
its signature names `std::span` (it gets it transitively via the `.hpp`); strict IWYU
would add `<span>` to the `.cpp`. **LOW / cosmetic.**

---

## Considered & rejected (incl. rejected-for-parity)

- **REJECTED-FOR-PARITY — "Sort blocks by size (LPT / longest-processing-time) for a
  better balance."** A classic load-balancer (sort descending, assign each to the
  least-loaded device) **breaks the parity floor**: it destroys the *contiguous*
  whole-block ranges, so `shards[g].s0/s1` would no longer be a single contiguous SNP
  span — the zero-copy sub-view `MatView{Q.data + P*s0, P, s1-s0}`
  (`f2_blocks_multigpu.cpp:130-143`) assumes contiguity, and a non-contiguous shard
  would need a gather (extra copy) AND would change which columns feed each device's
  GEMM, changing the `s_pad` bucket and the slab bits. P-1's fix is explicitly the
  *contiguous* variant to avoid this. **Rejected; P-1's fix stays contiguous.**
- **REJECTED-FOR-PARITY — "Combine in load-sorted or arrival order instead of
  g=0..G-1."** Any reorder of the combine sum breaks §12 (FP addition is
  non-associative; the fixed g=0..G-1 order is the parity law). The planner returns
  shards in `g` order and must continue to; both combine layers iterate `g=0..G-1`.
  **Rejected.**
- **REJECTED (correcting the FIRST DRAFT's P-2) — "cost a shard by `n_block_local ×
  max_block_size_in_shard` (the shard-max pad)."** This is the cost of the
  *global-max-pad* design the spike **rejected** (2.76× pad waste, slower —
  `f2_blocks_kernel.cu:14-16`). The actual backend buckets by pow2 size and pads
  per-bucket (`cuda_backend.cu:264-280`), so the shard-max formula over-states the cost
  by up to ~2× and would mis-direct a balancer. The corrected cost (if any cost model
  is added) is `Σ_buckets (n_in_bucket × s_pad_bucket × P²)`. See P-2.
- **REJECTED (not a defect) — "value-initialized" comment at `:43` is technically
  *default-inserted*.** **Verified against [vector.cons] via eel.is/c++draft:**
  "Constructs a vector with n **default-inserted** elements." For `DeviceShard`,
  default-insertion runs the in-class member initializers `{0,0,0,0}` — observationally
  identical to value-init. The comment's *intent* ("every entry is {0,0,0,0}") is
  correct; only the term is loose. Not worth a fix.
- **REJECTED — "use `std::accumulate` for `total_snps`."** The explicit loop (`:57-60`)
  is clear, and `std::accumulate` over `int` would need an explicit `0L` init to avoid
  `int` accumulator overflow anyway — the explicit loop with `long total` is *safer*
  and equally readable. Keep the loop.
- **REJECTED — "merge the total-snps loop and the assignment loop into one pass."**
  Impossible: `target_per_device` needs `total_snps` *before* the greedy can decide
  where to close. Two passes are required. (D-1's "derive sizes from ranges" still lets
  the first pass use `ranges[b].size()`, removing the `block_sizes` array.) Keep two
  passes.
- **REJECTED — "validate G against `resources.device_count()`."** No: the planner is
  correctly decoupled from `Resources` (it takes a plain `std::size_t G`), which is
  what makes it host-pure and unit-testable without a device bundle. The orchestrator
  passes `device_count()` (`f2_blocks_multigpu.cpp`). Correct separation.
- **REJECTED (parity-neutral, not a finding) — "P2P streams/overlap."** The
  sequential-P2P / stream-overlap hunt applies to `p2p_combine.cu`, NOT this planner
  (it issues no transfers). Out of scope for this unit; the `p2p_combine.cu` reviewer
  owns it. (For the record, `p2p_combine.cu` *does* serialize per-partial with a full
  `cudaDeviceSynchronize` (`p2p_combine.cu`), which is a legitimate perf finding —
  *there*, not here.)
- **REJECTED (not a defect) — "the two-pass O(n_block) is a perf concern."** It is
  trivially cheap vs the GEMMs (P-4); the only allocation is the result `vector(G)`.
  No data bouncing, no copy, no recompute. Confirmed.

---

## What it takes to reach 10/10

Performance / balance (the half-point):
1. **P-1 (MED-HIGH):** replace the prefix-threshold greedy with a *contiguous*
   max-load-balanced partition (binary-search-on-capacity feasibility, or at minimum a
   forward-looking close) so a single dominant/trailing block no longer idles GPUs.
   Keep whole-block-aligned + fixed `g` order ⇒ parity-safe. Add a balance assertion
   (max/min load ratio) to the new unit test.
2. **P-2 (LOW, doc-only unless P-1 adds a cost model):** correct the header's
   cost-model framing — SNP count is a *proxy*; the real per-device cost is the
   size-bucketed padded work `Σ_buckets (n_in_bucket × s_pad_bucket × P²)`
   (`cuda_backend.cu:264-280`), NOT the shard-max pad. If P-1 adds a cost objective,
   cost on the bucketed work, not raw SNP sum.

Cleanliness / robustness:
3. **D-1 (MED — the single best move):** drop the redundant `block_sizes` parameter;
   derive sizes from `ranges[b].size()` (already `long`). Collapses C-1, E-1, half of
   T-1, a caller `long→int` narrowing, and a caller allocation in one move.
4. **C-1/C-2/E-1 (if D-1 is NOT taken):** add the element-wise parallel-array assert
   (`ranges[b].size() == block_sizes[b]` and `>= 0`) and a `n_block ≤ INT_MAX` debug
   assert (`STEPPE_ASSERT` is reachable here); document the `int b0/b1` vs `long s0/s1`
   width split on `DeviceShard`.
5. **T-1/T-2:** centralize the narrowing casts in one audited `make_shard` helper; use
   the existing `core::cdiv(long,long)` (`launch_config.hpp:89`) for the ceiling
   division instead of the hand-rolled `G_signed` arithmetic (DRY, layering-legal).
6. **TST-1 (MED):** add `tests/unit/test_shard_plan.cpp` — GPU-free, dataset-free —
   asserting tiling/contiguity/cover + the documented edge cases (G==1, n_block==0,
   n_block<G, target-0/all-empty, giant-first/last block), realizing the unit's own
   CUDA-free testability promise and surfacing P-1 as a measurable load ratio.
7. **R-1/C-3/C-4:** tighten `empty()` to assert `b0 <= b1` (or document the `>=`); add
   one sentence on the `total_snps == 0` / `target == 0` behavior; and upgrade the
   `(b+1 < n_block)` comment to state its OOB-safety role (it prevents `ranges[b0]`
   reading past the end at the final close), not just "leaves the trailing run."

Doing P-1 + D-1 + TST-1 (the three substantive items) moves this to a clean 9.7–10:
the balance blind spot is the only thing keeping a correct, proven-parity,
well-layered, well-documented planner out of the top bracket.

## Good patterns to keep

- **Host-pure CUDA-free planner in the device layer, single-home for the block→device
  map.** One function, consumed by both the core orchestrator and the device combine,
  building into `steppe_device` as a plain `.cpp` with no CUDA include. Exactly the §8
  single-home + §4 layering discipline.
- **Deterministic pure function** of `(block_sizes, ranges, G)` — no Resources
  coupling, no RNG, no device-state dependence, no floating-point — which is what makes
  the §12 "same block, same device, stable bits" floor hold and what makes it trivially
  testable.
- **The whole-block-aligned + fixed g=0..G-1 + contiguous-tiling contract**, the
  structural parity floor under BOTH combine tiers (host-staged and P2P): the same
  `shards` vector feeds both, so the tiers are bit-identical by construction.
- **The `(b+1 < n_block)` close guard as an OOB invariant** — it is what keeps the
  final unconditional close's `ranges[b0]` index in-bounds. Correct and load-bearing
  (just under-documented).
- **Tier tag kept off the numeric payload** — the shard plan is tier-agnostic; the
  `CombinePath` tag lives on `Resources.last_combine_path`, never on `DeviceShard`.
- **Exemplary documentation**: every edge case enumerated and tied to an arch/parity
  section; `[[nodiscard]]`/`noexcept` correctly applied; the `int`(block ids) vs
  `long`(SNP columns) width choice mirrors `BlockRange`. The doc-density bar for M4.5.
- **Empty-shard-as-clean-no-op** (`b0 == b1` ⇒ `empty()`), with the orchestrator and
  both combine paths early-returning/placing-nothing — a tidy degenerate contract.
