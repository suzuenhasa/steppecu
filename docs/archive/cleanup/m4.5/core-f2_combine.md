# Unit review (adversarial 2nd pass) — `core/fstats/f2_combine` (the host-staged fixed-order f2 combine — the portable parity baseline)

Files audited line-by-line:
- `src/core/fstats/f2_combine.hpp` — the CUDA-free `combine_f2_partials_host` decl + the parity-law doc.
- `src/core/fstats/f2_combine.cpp` — `combine_f2_partials_host` + the `validate_partials` guard.

Context re-read in full to ground (and where needed to overturn) every claim:
- `include/steppe/fstats.hpp` — `F2BlockTensor` (layout `i + P·j + P·P·b`, FP64-storage-always, `vpair` "Integer-valued (carried as double)", `block_sizes`, `size()`).
- `src/device/shard_plan.hpp` — `DeviceShard{b0,b1,s0,s1}` (note: `s0/s1` are `long`, `b0/b1` are `int`), `empty()`, and `plan_block_shards` (the disjoint contiguous block-aligned tiling this combine assumes — the SOLE producer of the shards).
- `src/core/fstats/f2_blocks_multigpu.cpp` — the **sole caller** `compute_f2_blocks_multigpu`: where the G compact `partials[g]` come from (each per-device `compute_f2_blocks` over a zero-copy column sub-view, returned as **host** `F2BlockTensor`), the §4 P2P-vs-host gate, the `CombinePath` out-of-band tag, the G==1 structural fast-path, and the fixed g=0..G-1 order.
- `src/device/cuda/p2p_combine.cu` + `src/device/p2p_combine.hpp` — the sibling device-resident combine this unit must be bit-identical to (the same fixed-order placement-add onto a `cudaMemset(0)` accumulator; the **byte-for-byte duplicated** `validate_partials` differing only in the namespaced error strings + the one extra `device_ids.size()` check; the same dead `n_block_full < 0 ? 0 :` ternaries at lines 172/215; the same short-partial gap via `part_elems = slab·part.n_block`).
- `src/device/backend.hpp` (`ComputeBackend::compute_f2_blocks`, lines 237–266) — confirms the per-device return is a **compact** `[P × P × n_block_local]` HOST tensor (the device frees its buffers before returning), and that "the host-side fixed-order combine of their per-device … partials [is] orchestrated ABOVE this seam … NOT inside any one method."
- `src/device/resources.hpp` shape via the caller (`Resources`, `CombinePath`, `last_combine_path`, `caps.can_access_peer`) — the which-path tag is off the numeric payload.
- `src/core/CMakeLists.txt` (lines 58–77) — confirms `f2_combine.cpp` compiles into `steppe_core` STATIC with `steppe::device` linked **PRIVATE**, **no CUDA sources**, `CUDA_RESOLVE_DEVICE_SYMBOLS OFF`.
- `tests/unit/` — confirms there is **no** `test_f2_combine.cpp` (siblings DO have host unit tests: `test_block_ranges.cpp`, `test_f2_from_blocks.cpp`, `test_launch_config.cpp`, …), and the **only** coverage of this unit is the GPU-required end-to-end `tests/reference/test_f2_multigpu_parity.cu`.
- `tests/reference/test_f2_multigpu_parity.cu` — the locked gate. **Crucial for this pass:** the REFERENCE (line 354) is the single-GPU `compute_f2_blocks(*ref_backend, …)` which does **NOT** run a combine `+=`; the CANDIDATE (line 377) goes through `combine_f2_partials_host`'s `+= onto +0.0`. They are compared with `std::memcmp` (lines 173–174, 253, 260) for `EmulatedFp64{40}`. This fact is what makes the `-0.0` finding (N2) live rather than theoretical, and it overturns the first pass's framing of the P2 risk.
- `docs/architecture.md` §11.4 (line 717: "this combine is essentially free … off the bandwidth critical path"; "host-staged fixed-order combine remains the portable parity baseline … the only path on the budget consumer box"), §12 (line 741: "gathers the G partials to the host and sums them in fixed device order … free because `f2_blocks` is tiny and off the critical path"; the NEVER-NCCL-AllReduce law), and the §0 top-end `P=4266`/`B=757` ⇒ ≈220 GB resident-pair figure.

External behaviour verified against primary sources (NOT asserted from memory):
- `std::vector::assign(n, value)` "causes an automatic reallocation of the allocated storage space if -and only if- the new vector size surpasses the current vector capacity" and is "Linear on initial and final sizes (destructions, constructions)" — [cplusplus.com `vector::assign`](https://cplusplus.com/reference/vector/vector/assign/). Here `out` is default-constructed (capacity 0), so every `assign` reallocates **and value-constructs `total` elements** = a full streaming write.
- `std::vector::resize(count)` (no value) **value-initializes** new elements, which for non-class `double` is **zero-initialization to +0.0** — [cppreference `vector::resize` via search](https://www.geeksforgeeks.org/cpp/resize-vector-without-initializing-new-elements-in-cpp/), corroborated by [learncpp 16.10](https://www.learncpp.com/cpp-tutorial/stdvector-resizing-and-capacity/). **Consequence (corrects the first pass):** swapping `assign(total,0.0)` → `resize(total)` writes the **same** `total` zeros — it removes **no** write. Eliminating the redundant write requires NOT zeroing the owned region at all (see P1).
- IEEE-754 signed-zero addition: the rule is `x + (±0) = x` **"for x different from 0"**, and `(−0) + (+0) = +0` under round-to-nearest — [Signed zero (Wikipedia)](https://en.wikipedia.org/wiki/Signed_zero). So `(+0.0) += (−0.0)` yields **`+0.0`**, a **different bit pattern** from `−0.0`. This is the load-bearing exception the first pass's "`x + 0.0 == x` exactly for every finite x" elided — it is exactly where the `+=`-onto-zero combine can diverge bit-wise from the single-GPU reference (N2).

---

## Role & layering

`combine_f2_partials_host` is the **portable parity baseline** of the SPMG precompute (architecture §11.4 line 717; §12 line 741). It takes the G per-device **compact** `F2BlockTensor` partials (fixed g=0..G-1 = `DeviceConfig::devices` order), allocates one zero-initialized full-shape `[P × P × n_block_full]` tensor, and places + sums each device's compact partial at its block offset `shards[g].b0`, summing in the fixed g-order. Block-aligned sharding makes the shards disjoint, so each global slab is written exactly once (an add onto `+0.0`), but it is *implemented* as the general fixed-order `+=`, which is (a) literally the §12 host-side fixed-order sum, (b) the identical arithmetic the device P2P sibling performs, and (c) — with the one caveat in N2 — exact.

**Layering is exemplary and verified, not asserted.** CUDA-free, host-pure, in `steppe::core`; the header includes only `<span>`, the public CUDA-free `steppe/fstats.hpp`, and the CUDA-free `device/shard_plan.hpp`. `src/core/CMakeLists.txt` confirms it compiles into `steppe_core` STATIC with `steppe::device` PRIVATE and no CUDA TU. The combine **policy** (the §4 P2P-vs-host gate, the `CombinePath` tag, the degrade WARN) correctly lives one layer up in `compute_f2_blocks_multigpu`/`Resources`, **not** here — this unit takes no capability input and emits no tag. Decomposition (one guard + one combine), const-correctness, `[[nodiscard]]`, naming (`slab`/`total`/`out_base`/`in_base`/`covered`/`span_blocks`), and comment density are all at the senior bar.

**This 2nd pass confirms the first pass was substantially correct but (1) MISSED a real latent numerical hazard (N2, the `−0.0` flip) that its own "x + 0.0 == x exactly" footnote actively papered over, (2) MIS-STATED the mechanics of its headline P1 fix (`assign→resize` removes no write; only not-zeroing-the-owned-region does — and that is hard to do safely with `std::vector`), and (3) framed P2 as merely "bit-identical to the += form" when in fact `std::copy` is *strictly more faithful to the single-GPU reference* than the `+=` on `−0.0`, which both reverses the P2 risk note and makes P2 the principled fix for N2.** Several first-pass findings are downgraded to informational below; two are added; the net is a tighter, more honest review.

---

## Score: 9/10 — a correct, clean, parity-locked baseline; the deductions are a latent `−0.0` bit-flip hazard the formulation hides (currently dormant), a redundant full zero-then-overwrite write, a scalar elementwise loop that should be a bulk move, the byte-duplicated validator, and a short-partial guard gap — all parity-safe to fix and all on the (cold, §11.4) combine path

The unit does exactly what §11.4/§12 require and the locked `memcmp` test proves it bit-identical to single-GPU and to the P2P sibling for `EmulatedFp64{40}`. It is held at 9 (not raised, not lowered vs. the first pass) by: (a) a **latent** `−0.0`→`+0.0` flip in the `+=`-onto-`+0.0` formulation that *could* diverge from the single-GPU reference but does not on the tested AADR data (N2 — the first pass missed it and its IEEE footnote was wrong); (b) the function writes `2·P²·n_block_full` doubles to 0.0 and then overwrites every one on the disjoint path (P1); (c) the overwrite is a scalar `+=` triple loop, not the `memcpy`-grade bulk move the contiguous layout permits (P2); (d) the validator is duplicated byte-for-byte with the P2P sibling and both share a short-partial OOB gap (CL1/C1); (e) dead `< 0 ? 0 :` ternaries + a `long` cast scatter (C2/D1); (f) no host unit test despite the unit's whole point being host-testability (T1). None moves the headline wall-clock (the combine is explicitly "essentially free" per §11.4), but on a perf-first pass with parity locked, they are exactly the accreted warts to surface. Fix N2 (via P2), P1, CL1+C1, and add T1 → 9.5–10.

---

## Findings

### Performance (first-class this pass)

#### P1 — [MED] The full output is zero-initialized (`2·P²·n_block_full` doubles → 0.0) and then, on the only path it is ever called with (disjoint shards), every owned element is overwritten — a redundant full streaming write; and the first pass's `assign→resize` "fix" removes none of it
**Location:** lines 90–100 (three `out.*.assign(total / n_block_full, 0)`) vs. lines 110–129 (the loop that writes every owned slab).

**Issue.** `out.f2.assign(total, 0.0)` / `out.vpair.assign(total, 0.0)` each construct and zero all `total = P²·n_block_full` doubles (verified: `assign` on a capacity-0 vector reallocates and is "Linear on … final size … constructions" — [cplusplus.com](https://cplusplus.com/reference/vector/vector/assign/)). The combine then writes `out.f2[out_base+e] += …` over the **same** `total` elements — the block-aligned shards tile `[0, n_block_full)` exactly (`plan_block_shards` is the sole producer; `validate_partials`'s `covered == n_block_full` re-checks it). So every output double is written **twice**: once to 0.0, once to its value. At the §0 top end (`P=4266`, `B=757` ⇒ ≈220 GB resident pair, per §11.2 / architecture line 717) the redundant zero-write is real bandwidth, not noise.

**Why it matters.** Textbook redundant write / data-bouncing — the audit's first-class target. The zero-init exists only to make the *general overlapping-shard* `+=` correct on slabs no device owns; block-aligned shards have no such slabs, so the zero is dead on the real path.

**Correcting the first pass.** The draft's option (b) — "replace `.assign(total,0.0)` with `.resize(total)` (… same bits)" — is **mechanically inert for the write count**: `resize` also value-initializes every new `double` to `+0.0` (verified above), so it touches the same `total` elements. Swapping `assign→resize` changes nothing about the redundant write; it is *not* the fix. The redundant write disappears **only** if the owned region is never zeroed — i.e. each element is written **exactly once** by a placement. With a plain `std::vector<double>` you cannot obtain uninitialized storage without a custom no-init allocator (or `reserve`+placement into raw storage, which fights the vector's size invariant). So the honest options are:
- **(a) Conservative — keep the zero-init as the proven baseline.** Accept the redundant write as the price of the literal "sum onto zero" formulation. Then P1 is a *documented non-fix* and the score reflects the accepted, off-critical-path waste. Defensible given §11.4 "essentially free."
- **(b) Real fix — pair P2's placement with a no-init full buffer.** Allocate `f2`/`vpair` with a default-init / no-init allocator (or one resize-without-init helper) and write every owned element via P2's `std::copy`. The zero-init then survives **only** for a genuinely-unowned tail, which the disjoint contract proves cannot exist — so the double-write is gone. This is real engineering (a `no_init_allocator<double>` or equivalent) and changes the `F2BlockTensor` buffer's allocator type, so it must be weighed against §0 ABI stability (the public type's `std::vector<double>` member would need to stay default-allocator at the seam; a private staging buffer copied into the public vector would re-introduce a copy). Net: the clean version of (b) is **L effort** and only worth it if the ≈220 GB top-end is a real workload.

**Expected benefit.** Up to ~33% of the combine's memory traffic on the owned region (write-zero + read-partial + write-result → read-partial + write-result). **Risk.** (a): none. (b): med — relies on the disjoint-shard invariant `validate_partials` enforces; if a future caller ever passed overlapping shards (it does not; `plan_block_shards` tiles disjointly) the placement would drop the overlap sum. Re-gate by the locked `memcmp` test either way.

**Severity:** med. **Effort:** S for (a)-as-doc; L for (b)-done-properly. **Parity-safe?** yes (a); yes-if-careful (b) — and note (b) via P2 *also fixes N2*, see below.

#### P2 — [MED] The placement is a scalar `for (e) out[..] += in[..]` element loop, not a contiguous bulk move; the whole per-device partial is in fact ONE contiguous copy; and `std::copy` is *more faithful to single-GPU than the `+=`* (it preserves `−0.0` — see N2)
**Location:** lines 113–128 (the per-block loop + the inner `for (std::size_t e = 0; e < slab; ++e)` RMW).

**Issue.** For each owned block the code moves a `slab = P²`-element contiguous run via `+=` onto a known-`+0.0` destination. Because `out_base = slab·(b0+lb)` and `in_base = slab·lb`, as `lb` runs `0..n_block-1` the source spans `[0, slab·n_block)` contiguously and the destination spans `[slab·b0, slab·(b0+n_block))` contiguously — so the **entire per-device placement of `f2` (and of `vpair`) is one `std::copy` of `slab·part.n_block` elements**, and `block_sizes` is one `std::copy` of `part.n_block` ints at offset `b0`. The triple loop collapses to three `std::copy_n` per device. As written it is a scalar load-add-store the compiler must prove non-aliasing to vectorize (the four arrays are distinct vectors so they don't alias, but the `+=` still forces a load-of-zero + add + store where a copy is a single streaming store).

**Concrete fix (parity-safe, and the principled fix for N2):** per device g with `n_block_local > 0`:
```cpp
std::copy_n(part.f2.data(),          slab * part.n_block, out.f2.data()          + slab * b0);
std::copy_n(part.vpair.data(),       slab * part.n_block, out.vpair.data()       + slab * b0);
std::copy_n(part.block_sizes.data(), part.n_block,        out.block_sizes.data() + b0);
```
This lowers to `memmove`/`memcpy` and is the doc's own stated reduction ("With block-aligned sharding … this is in effect a PLACEMENT").

**Reversing the first pass's risk note.** The draft said `std::copy` is "bit-identical to `+= onto +0.0`." That is **false on `−0.0`**: `+0.0 += (−0.0)` → `+0.0` (different bits), whereas `std::copy` reproduces `−0.0` exactly. Since the single-GPU REFERENCE computes the slab directly (no add onto a zero — verified in the parity test, ref at line 354), **`std::copy` is bit-identical to the reference and the `+=` is the one that can diverge** (N2). So P2 is not merely an equal-bits speedup; it is *more correct* and is the recommended remedy for N2. If for the parity *argument* the `+`-onto-zero formulation must be preserved verbatim, `std::transform(first, last, out_it, std::plus<>{})` over the contiguous run keeps the `+` (and the `−0.0` flip with it — so it does NOT fix N2).

**Expected benefit.** Scalar RMW → `memcpy`-grade moves (≈2× per-element throughput vs. load-add-store, plus elimination of inner-loop index arithmetic). With P1(b) the combine becomes read-partial → bulk-copy-to-output — the minimum possible. **Risk.** Correct *because* shards are disjoint and the destination is the source-or-zero; the locked test re-validates (and would now also cover `−0.0` if the data ever produced it).

**Severity:** med. **Effort:** S. **Parity-safe?** yes — `std::copy` of the exact bytes is bit-identical to the single-GPU reference (strictly **safer** than the current `+=` on `−0.0`); re-gate with the `memcmp` test.

#### P3 — [LOW, INFORMATIONAL] No casting/precision bug on the `vpair` path — confirmed, must stay `double`
**Location:** lines 98, 121.

`vpair` holds integer-valued counts stored as `double` (`fstats.hpp`). The combine adds/places them as doubles onto `+0.0`; result is the exact integer-valued double the device produced (subject to the same `−0.0` caveat as `f2`, which for counts cannot arise — a count is `≥ 0` and never `−0.0`). No cast, no width risk. Converting to an integer storage type would break §12 FP64-storage-always and byte-identity with the P2P sibling (which `cudaMemset`s/copies doubles). **No change.** (Confirms the first pass.)

#### P4 — [MED, ARCHITECTURAL — N/A for THIS unit; home is the seam + the P2P sibling] The "gather G partials to host then sum" is the *intended* host-staged design, not an elidable bounce here
**Location:** not in `f2_combine.cpp` — in the `ComputeBackend::compute_f2_blocks` return contract (host storage, device buffers freed; `backend.hpp` + `p2p_combine.cu:18-25`).

**Issue + verification.** The audit's headline concern ("gathering G full-shape partials to host then summing — copies that could be elided"). **Architecture §12 (line 741) explicitly mandates this:** the parity path "gathers the G partials to the host and sums them in fixed device order … free because `f2_blocks` is tiny and off the critical path." For the **host-staged tier** (the budget 5090 box, no peer access) the D2H is **not elidable** — the host *is* where the sum must happen. The bounce is only wasteful on the **P2P tier**, where `p2p_combine.cu` re-uploads each host partial H2D and pulls it peer→root (a D2H→H2D round-trip), which the sibling itself flags as a "PERFORMANCE wart, not a parity wart … OUT OF SCOPE for M4.5." None of that is this unit's code.

**Concrete fix.** None in `f2_combine.cpp`. The architectural fix (deferred) is a device-resident-partial `compute_f2_blocks` variant so the P2P tier skips the round-trip; the host-staged tier still D2Hs once (it must), which it already does.

**Severity:** med (architectural) but **N/A for this unit's own code**. **Effort:** L (deferred). **Parity-safe?** yes — but not this unit's change. (Confirms the first pass; re-grounded against §12 line 741, which sanctions the host gather as the design, not a wart.)

#### P5 — [LOW, INFORMATIONAL] Three separate heap allocations (f2/vpair/block_sizes) — unavoidable given the public `F2BlockTensor` three-vector ABI; one-time, cold path
**Location:** lines 97–100. Fixed by the public type's shape (`fstats.hpp`); fusing would change the ABI. Negligible, cold. **No action.** (Confirms the first pass.) Note: if P1(b) lands, this is where a no-init allocator would attach.

### Correctness & bugs

#### C1 — [LOW] `validate_partials` checks the scalar `n_block`/`P` but NOT that `part.f2/vpair/block_sizes` are long enough — a short partial reads OOB; the guard's own comment over-states what it prevents
**Location:** `validate_partials` (lines 29–74) checks `part.n_block == span_blocks` (51) and `part.P == P` (60) but never `part.f2.size() == slab·part.n_block` (etc.). The loop at 119–122 then indexes `part.f2[in_base+e]` up to `slab·part.n_block - 1` (unchecked `operator[]`).

**Issue.** A malformed partial with correct `n_block`/`P` scalars but an under-sized `f2` vector would read out of bounds. The validator's comment (lines 23–28) *claims* it prevents "reading past a short partial," but it only validates the scalar, not the storage — the claim is slightly false. Cannot fire today (`compute_f2_blocks` sets `f2.size() == P²·n_block` by construction), so it is a defensive gap, not a live bug. The **P2P sibling has the identical gap** (`part_elems = slab·part.n_block` drives the DMA byte count and would over-read a short host partial) — closing it in one shared validator (CL1) fixes both.

**Concrete fix.** In the per-g loop, with `slab = (size_t)P·(size_t)P`: `if (part.n_block > 0 && (part.f2.size() != slab*part.n_block || part.vpair.size() != slab*part.n_block || part.block_sizes.size() != (size_t)part.n_block)) throw …`. Cheap O(G), off the critical path.

**Severity:** low (defensive). **Effort:** S. **Parity-safe?** yes (validation-only; rejects inputs that never occur on the parity path).

#### C2 — [LOW] Dead `n_block_full < 0 ? 0 : n_block_full` ternaries after the value is already proven `≥ 0`
**Location:** lines 96, 100 — but line 38 already throws on `n_block_full < 0`, so by line 96 the ternary can never take the `0` branch. Harmless dead defensive code (copy-pasted; the P2P sibling has the identical dead ternaries at `p2p_combine.cu:172, 215`). The `P==0`/`n_block_full==0` degenerate correctly yields an empty tensor (matches `plan_block_shards`'s `n_block==0 ⇒ G empty shards`).

**Concrete fix.** Drop both ternaries; use `static_cast<std::size_t>(n_block_full)`. Removes dead branches + a casting wart (D1). Single-home with CL1.

**Severity:** low. **Effort:** S. **Parity-safe?** yes (unreachable branch removed; behaviour unchanged).

### Edge cases & failure modes

#### E1 — [VERIFIED CLEAN] All-empty (G empty shards, `n_block_full == 0`)
`plan_block_shards` returns G `{0,0,0,0}` shards; every `n_block == 0`; `validate_partials` passes (`covered == 0`); `total == 0`; `assign(0, …)` allocates nothing; the loop never iterates. Returns `{P, 0, ∅, ∅, ∅}`. Matches the P2P `total == 0` guards. **No action.**

#### E2 — [VERIFIED CLEAN] Trailing/middle empty shards (`n_block < G`)
A device with `b0 == b1` ⇒ `part.n_block == 0`, `span_blocks == 0`; the P check is skipped (the `n_block > 0` guard, line 60); the loop body does not execute. Non-empty shards place at their `b0`. **No action.** (Confirms first pass E1/E2.)

#### E3 — [VERIFIED SAFE] `slab`/`total`/`out_base` overflow at the §0 top end
`slab = (size_t)P·(size_t)P` widens before multiply (`P=4266 ⇒ slab≈1.82e7`); `total = slab·(size_t)n_block_full ≈ 1.38e10` fits 64-bit; `out_base = slab·(size_t)b` casts `b` before multiply. `b = b0 + lb` is `int` but `b < n_block_full ≤ INT_MAX`, no `int` overflow. **Safe.** See D1 for cast tidying. (Confirms first pass E3.)

#### E4 — [NEW, LOW] `P` and `n_block_full` are validated `≥ 0` but their PRODUCT is not checked against `partials`/`shards` consistency beyond the per-g spans — confirmed non-issue
Considered: could a caller pass `P=0` with non-empty shards? `validate_partials` checks each `part.n_block == span_blocks` and `covered == n_block_full`; with `P=0`, `slab=0`, `total=0`, and the inner `for (e < 0)` never runs, so it is a clean (if vacuous) empty-tensor result. A non-empty partial with `P=0` would have `part.P=0==P` pass and `f2.size()==0`, consistent. No degenerate misbehaviour. **No action** — recorded so the `P==0` corner is explicitly cleared.

### Numerical / precision vs §12

#### N2 — [NEW, MED — the finding the first pass MISSED] The `+=`-onto-`+0.0` formulation flips `−0.0` to `+0.0`, which can diverge bit-wise from the single-GPU reference; currently DORMANT on the tested data but a real latent hazard, and it makes P2's `std::copy` the principled fix
**Location:** lines 119–122 (`out.f2[out_base+e] += part.f2[in_base+e]` onto a `+0.0` destination), and the doc's parity claim in `f2_combine.hpp:38-39` / `.cpp:108-109` ("`x + 0.0 == x` for all finite x").

**Issue.** The IEEE-754 identity is `x + (±0) = x` **only for `x ≠ 0`**; for `x = −0.0`, `(+0.0) + (−0.0) = +0.0` under round-to-nearest — a **different bit pattern** ([Signed zero, Wikipedia](https://en.wikipedia.org/wiki/Signed_zero)). The combine adds each partial element onto the `+0.0`-initialized accumulator, so **any `−0.0` in a partial's `f2`/`vpair` becomes `+0.0` in the combined tensor.** The single-GPU REFERENCE computes the slab directly with **no** add onto a zero (verified: parity test reference is `compute_f2_blocks(*ref_backend, …)`, line 354), so if its f2 ever equals `−0.0` for some `(i,j,b)`, the reference holds `−0.0` (`0x8000…0`) while the combine holds `+0.0` (`0x0000…0`) — and the gate is `std::memcmp` (lines 173–174), which **would catch this bit difference.**

**Why it is currently dormant (not a live failure).** The locked test passes bit-identical for `EmulatedFp64{40}`, which means **the production AADR f2/vpair values never land on exactly `−0.0`** on the tested `derived_acc`/`derived_full` inputs. f2 is `Σ(p_i−p_j)²`-shaped (a sum of squares, ≥ 0 in exact arithmetic) divided by counts, and the masked GEMM/divide on real data evidently never yields the `−0.0` bit pattern; `vpair` is a non-negative count. So the hazard does **not fire today**. But it is undocumented, the doc's "for all finite x" is *wrong as stated*, and a future input/precision/formula change that produced a `−0.0` (e.g. a `(a−b)` that rounds to `−0.0`, or a sign-carrying intermediate) would silently break the §12 bit-identity on the combine path while leaving single-GPU correct — exactly the kind of regression the parity gate exists to catch, but one the *formulation itself introduces*.

**Why it matters.** §12 bit-identity is the unit's contract. The current formulation is bit-identical to single-GPU **only under the empirical assumption that no partial element is `−0.0`** — an assumption the code neither documents nor enforces. This is a latent parity hazard hiding inside the very line the first pass cited as proof of exactness.

**Concrete fix.** Adopt **P2's `std::copy`/`copy_n` placement**: a copy reproduces `−0.0` byte-for-byte and is therefore bit-identical to the single-GPU reference *even if* a partial ever contains `−0.0` — i.e. P2 *removes* the hazard. Do **not** use the `std::transform(…, std::plus<>{})` variant for this purpose: it keeps the `+` and the flip. If the `+=` formulation must be retained for the parity narrative, then **document the `x ≠ 0` exception explicitly** and add a debug assertion / test that no partial element is `−0.0` (so the dormant assumption is recorded and guarded). The cleanest outcome is P2 (copy) + a one-line doc correction.

**Severity:** med (latent parity hazard + a factually wrong doc claim). **Effort:** S (P2 already fixes it). **Parity-safe?** yes — `std::copy` is *strictly safer* than the status quo on `−0.0`; it is bit-identical to the single-GPU reference unconditionally, where the current `+=` is bit-identical only under the unstated no-`−0.0` assumption. The locked test re-validates (and should add a `−0.0` synthetic partial to T1 to pin it).

#### N1 — [VERIFIED, REFRAMED] The zero-init + placement IS the §12 parity law — but only with the N2 caveat
The combine starts from the all-zero (= `+0.0`) bit pattern (`std::vector<double>(.., 0.0)`), and the P2P sibling `cudaMemset(0)`s the same `+0.0`, so the two tiers start identical. FP64 storage is mandated in every mode (`fstats.hpp`); `precision` never reaches the combine (it governs the upstream GEMMs only). The result is bit-identical to single-GPU **for every element except a potential `−0.0`** (N2). The first pass's N1 was correct *except* for the universal "for all finite x" claim, now corrected.

### CUDA idioms / RAII / async vs §7

**N/A — correctly.** Host-pure, CUDA-free by design (§4): no kernels, no streams/events, no device memory. The §7 idioms (RAII wrappers, post-launch checks, grid-stride, streams/events) apply to the P2P sibling, not here. The *absence* of CUDA is the correct state and is the unit's portability strength. Explicitly confirmed: no missing grid-stride / default-stream / sync issue, because there is no device code. (The P2P sibling's per-partial `cudaDeviceSynchronize` and sequential pulls are *its* perf surface — `cuda/p2p_combine.cu:316` — and are reviewed in `device-cuda-p2p_combine.md`, not here.)

### Magic numbers & hardcoded values vs §4

**N/A — clean.** No literals beyond `0`/`0.0` — and `0.0` IS the load-bearing parity constant (the value, not a tunable). No thread/block counts (those live in the sibling's `kPlaceAddBlockX`). Nothing to extract.

### Decomposition / SRP / function size vs §2

**Strong.** Two functions, one responsibility each, both short and readable. The only structural note is CL1 (the validator duplicated byte-for-byte with the sibling). No god-function, no mixed concerns.

### Readability, naming, const-correctness, `[[nodiscard]]`/`noexcept`, comments

#### R1 — [LOW] Correctly NOT `noexcept`; `[[nodiscard]]` on the decl only — at the bar
The decl has `[[nodiscard]]` (`f2_combine.hpp:66`); the definition needn't repeat it. Correctly **not** `noexcept` (throws via `validate_partials`; `assign` can `bad_alloc`). Naming is precise and self-documenting. **No action.**

#### R2 — [LOW] The doc's "`x + 0.0 == x` for all finite x" (`.hpp:38-39`, `.cpp:108-109`) is factually wrong on `−0.0`; and "PLACES + SUMS" should track P2 if it lands
The "for all finite x" exactness claim is the very text N2 corrects — it should read "for finite x ≠ ±0.0; a `−0.0` partial element would flip to `+0.0`, which does not occur on the production path (and P2's placement avoids it)." If P2 turns the body into `std::copy`, update the doc to say the implementation *is* the placement (fixed g-order = the loop order over devices). **Severity:** low (doc-vs-fact + doc-vs-code drift). **Parity-safe?** n/a (doc).

### Layering / API / ABI vs §4

**Exemplary — the unit's headline strength, verified against CMake.** `f2_combine.cpp` compiles into `steppe_core` STATIC with `steppe::device` PRIVATE, no CUDA TU (`src/core/CMakeLists.txt:58-77`). Header includes only `<span>` + the public CUDA-free `steppe/fstats.hpp` + the CUDA-free `device/shard_plan.hpp`. The combine policy (the §4 gate) is correctly NOT here. Dependency direction `core → CUDA-free DeviceShard` is legal (`shard_plan.hpp` is CUDA-free by contract, "device-layer only by placement," names only `core::BlockRange`). **No layering finding.**

### Testability vs §13

#### T1 — [MED] No dedicated host-only unit test; the unit is exercised ONLY through the GPU-required parity test, despite being designed to be host-testable
**Location:** `tests/unit/` has host tests for siblings (`test_block_ranges.cpp`, `test_f2_from_blocks.cpp`, …) but **no** `test_f2_combine.cpp`; the only coverage is `tests/reference/test_f2_multigpu_parity.cu` (a `.cu`, GPU-required, end-to-end).

**Issue.** The header asserts this unit is "unit-testable host-only and compiles into steppe_core without the device toolkit" (lines 13–14) — but that testability is never exercised. A GPU-free unit test could build synthetic `F2BlockTensor` partials + `DeviceShard`s and check: (a) exact placement; (b) the fixed-order sum equals a hand-computed reference; (c) `validate_partials` throws on each violation (count mismatch, P disagreement, `n_block != span`, non-tiling shards, negatives, and — per C1 — a short partial); (d) the degenerates (all-empty, trailing-empty); and **(e) a partial containing `−0.0`** to pin N2 (this would FAIL the current `+=` body and PASS the P2 `std::copy` body — making it the gate for the N2 fix). All in milliseconds, no GPU — exactly the §13 "exercisable GPU-free" property, and exactly where the budget box (no capable-tier GPU coverage) most needs cheap host coverage.

**Concrete fix.** Add `tests/unit/test_f2_combine.cpp` (host-only, like `test_block_ranges.cpp`), wire into CTest. Cheapest high-value addition; becomes the fast inner-loop gate for P1/P2/N2 before the slow GPU parity test.

**Severity:** med. **Effort:** S–M. **Parity-safe?** yes (test-only).

### Capability-tier coherence (probe / tagged-degrade / which-path)

**N/A for this unit — correctly, verified.** The tier gate (`prefer_p2p_combine && can_access_peer`), the `CombinePath` tag (`last_combine_path`), and the degrade WARN all live in `compute_f2_blocks_multigpu` / `Resources` (verified: `f2_blocks_multigpu.cpp:171-201`), **off** the numeric `F2BlockTensor`. This unit reads/writes no tag, takes no capability input, is the unconditional fallback and the only path on the budget box. The tag is observability, not payload (§12, cleanup §(2).2). Confirmed: combine policy is in the right layer; the tag is off the payload.

---

## Considered & rejected (incl. rejected-for-parity)

- **Parallelize the combine across threads/devices.** REJECTED. The fixed g=0..G-1 order *is* the §12 parity law; reordering adds would change bits for overlapping shards. For disjoint shards a parallel placement would be safe but the win is nil (off the critical path, §11.4). **rejected-for-parity (formulation) / not-worth-it (perf).**
- **NCCL AllReduce / a reduction tree.** REJECTED-FOR-PARITY. §12 (architecture line 741) is explicit: AllReduce order varies with G and is not bit-identical to a single-GPU sum. Non-negotiable — the unit exists to keep this reduction off NCCL.
- **`assign(total,0.0)` → `resize(total)` to skip the redundant write.** REJECTED as written (first pass's P1(b) mechanics). `resize` value-initializes every new `double` to `+0.0` (verified — [resize behaviour](https://www.geeksforgeeks.org/cpp/resize-vector-without-initializing-new-elements-in-cpp/)), so it writes the same `total` zeros; it removes no write. Eliminating the write needs a no-init allocator + placement (P1(b), L effort).
- **`std::transform(…, std::plus<>{})` as the P2 rewrite.** Considered, REJECTED for the N2 fix. It vectorizes the `+` but keeps the `+0.0 += −0.0 → +0.0` flip, so it does NOT remove the latent `−0.0` divergence; only the `std::copy` placement does. (`transform`+`plus` IS acceptable purely as a perf rewrite if the `+=` narrative must be kept *and* N2 is handled by a separate assertion/doc.)
- **Make `vpair` an integer type.** REJECTED-FOR-PARITY. `fstats.hpp` mandates FP64 storage; the P2P sibling copies doubles; an integer `vpair` breaks byte-identity with the device tier and the public ABI.
- **Drop the zero-init entirely (not just elide on the disjoint path).** REJECTED. If any slab were ever unowned (it isn't, but the *formulation* is "sum onto zero") an uninitialized destination is garbage. P1 only elides the owned-region write *paired with* P2's placement and gated by the disjoint validator.
- **Fuse the three output vectors into one allocation.** REJECTED (out of scope). The three-vector shape is the public `F2BlockTensor` ABI; fusing changes the public type. Cold-path allocs are negligible (P5).
- **`x + 0.0 == x` "for all finite x" is exact — leave the doc.** REJECTED — this is N2: the identity excludes `x = −0.0`. The doc claim is factually wrong and the formulation hides a latent bit-flip. Corrected in N2/R2.

---

## What it takes to reach 10/10

Numerical / parity-hardening (the new top priority this pass):
1. **N2 — eliminate the latent `−0.0` flip.** Adopt P2's `std::copy`/`copy_n` placement (bit-identical to single-GPU even on `−0.0`), and **correct the doc** ("for finite x ≠ ±0.0; placement avoids the `−0.0` flip"). Add the `−0.0` synthetic-partial case to T1 so the assumption is pinned, not merely empirically true on AADR.

Performance (all parity-safe; all re-gated by the locked `memcmp` test):
2. **P2 — collapse the scalar `+=` triple loop into per-device `std::copy_n`** of the contiguous owned runs (`f2`, `vpair`, `block_sizes`). Single highest-value change: `memcpy`-grade moves + removes inner-loop index arithmetic + fixes N2.
3. **P1 — eliminate the redundant zero-then-overwrite** on the owned region. Either (a) document it as the accepted off-critical-path price (S, zero risk), or (b) pair P2 with a no-init full buffer so each element is written once (L, only worth it for the ≈220 GB top-end). NOT `assign→resize` (that removes no write).

Correctness / robustness:
4. **C1 — validate `part.f2/vpair/block_sizes.size()` against `P²·n_block`** so the guard's own promise ("not reading past a short partial") is actually true (and the sibling inherits it via CL1).
5. **C2/D1 — drop the dead `< 0 ? 0 :` ternaries** (already proven `≥ 0`) and tidy the `long`/`size_t`/`int` cast scatter.

DRY / single-home:
6. **CL1 — single-home `validate_partials`.** The host validator (`f2_combine.cpp:29-74`) and the P2P validator (`cuda/p2p_combine.cu:100-141`) are byte-for-byte the same contract (modulo the namespaced strings + the `device_ids.size()` check). Hoist one CUDA-free `validate_f2_partials(partials, shards, P, n_block_full)` into a shared CUDA-free header (it names only `F2BlockTensor`/`DeviceShard`); both tiers call it (P2P adds the one extra `device_ids.size()` check). The sibling's comment already *requires* lock-step ("so the two tiers reject identically") but nothing enforces it — and C1 then fixes both at once. §8 single-home.

Testability:
7. **T1 — add `tests/unit/test_f2_combine.cpp`** (host-only, GPU-free): exact placement, fixed-order-sum-vs-reference, every `validate_partials` throw (incl. C1's short partial), the empty/trailing-empty degenerates, and the `−0.0` case (N2). The fast gate for 1–6 before the slow GPU parity test.

Land 1–7 and the unit is a 9.5–10: a placement that is unconditionally bit-identical to single-GPU (no `−0.0` hazard), the minimal-bandwidth combine, a guard that enforces its own contract, one validator shared with the P2P tier, and cheap host coverage that pins it all.

---

## Good patterns to keep

- **The CUDA-free, host-pure baseline in `steppe::core`.** Portable, compiles without the toolkit, the unconditional budget-box fallback. Layering verified against CMake (`steppe::device` PRIVATE, no CUDA TU). Do not let a CUDA dependency leak in.
- **Combine policy out of the mechanism.** The §4 tier gate + `CombinePath` tag live in the orchestrator/`Resources`; this unit takes no capability input and emits no tag. Clean mechanism-vs-policy split.
- **The fixed g=0..G-1 order as the explicit, documented parity law**, with the doc spelling out *why* it bit-matches single-GPU and the P2P sibling — impossible to misread, which is why a future maintainer won't "optimize" the order onto NCCL. (Tighten the `x + 0.0` exactness claim per N2, but the order law itself is the right invariant.)
- **Fail-fast `validate_partials` up front, O(G), off the critical path**, with rich context strings naming the offending `g`, its `n_block`, and the shard span. (Tighten per C1 + single-home per CL1, but the pattern is right.)
- **`std::span` parameters (not owning vectors).** The combine borrows; no input copies. P2's `std::copy` rewrite preserves this.
- **`size_t`-widened slab/total arithmetic** (cast-before-multiply) — correct at the §0 top-end scale where a naive `int` index overflows (E3 verifies).
- **Honest, dense comments tying each choice to its architecture section.** Keep the discipline; just make the one IEEE claim accurate (N2).
