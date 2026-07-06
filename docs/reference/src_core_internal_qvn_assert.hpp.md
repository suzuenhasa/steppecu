# `qvn_assert.hpp` reference

## 1. Purpose

`src/core/internal/qvn_assert.hpp` holds a single tiny helper,
`assert_qvn_consistent()`, that checks the one precondition every f2 calculation
depends on: that the three input arrays it is handed actually line up with each
other and are shaped sensibly.

Those three arrays are the **Q/V/N** trio that steppe's f2 code always consumes
together, each a `P`-rows-by-`M`-columns view (where `P` is the number of
populations and `M` is the number of SNPs in the block):

- **Q** is the reference-allele frequency at each population and SNP,
- **V** is a validity mask (1 where the genotype is present, 0 where it's
  missing), and
- **N** is the non-missing haploid count — how many alleles were actually
  observed there.

The whole point of these three is that they are read in lockstep, cell by cell:
the f2 math reaches into all three at the same `(population, SNP)` position at
once. If Q had a different number of populations than V, or N covered a different
number of SNPs than Q, that lockstep indexing would silently walk off into the
wrong memory and produce garbage — with no crash to tell you. So before any f2
entry point starts, it wants a cheap sanity check that the three views agree on
`P`, agree on `M`, and don't carry a nonsensical negative shape.

The reason this check lives in its own header rather than being written inline at
each entry point is the same "one source of truth" logic used elsewhere in the
codebase: there is more than one door into the f2 computation — the single-GPU
path in `f2_from_blocks.cpp` and the multi-GPU path in
`f2_blocks_multigpu.cpp` — and each of them calls this at the top. If each door
re-typed its own three asserts, they could drift: someone tightens the check in
one place and forgets the other, and now the two paths disagree about what a
valid input even is. Factoring the three asserts into one function means every
door enforces exactly the same contract, and there's a single spot to change it.

This is a **debug-only** guard. It uses `STEPPE_ASSERT`, which expands to a real
`assert` in debug builds and to nothing at all under `NDEBUG`, so in a release
build these checks cost zero — they're a development safety net that catches a
mis-wired caller during testing, not a runtime validation that stays in the hot
path. The `[[maybe_unused]]` markers on the parameters are there precisely so the
compiler doesn't warn about unused arguments once the asserts compile away.

The header is host-pure and CUDA-free: it only reads the `P` and `M` integers off
each `MatView`, never touches the underlying device data, and so it compiles with
the ordinary C++ compiler with no GPU involved.

---

## 2. The check itself (`assert_qvn_consistent`)

```
assert_qvn_consistent(const MatView& Q, const MatView& V, const MatView& N)
```

It runs three assertions in order:

1. **Same `P`** — `Q.P == V.P && V.P == N.P`. All three must describe the same
   number of populations. (Written as a chained equality so it also rules out the
   case where two match but the third differs.)
2. **Same `M`** — `Q.M == V.M && V.M == N.M`. All three must describe the same
   number of SNPs.
3. **Non-negative shape** — `Q.P >= 0 && Q.M >= 0`. A negative `P` or `M` means a
   `MatView` that was never properly filled in (its counts are still bogus), which
   this catches before the numbers get used as loop bounds or array strides. Only
   Q is tested here, because steps 1 and 2 have already established that V and N
   carry the same `P` and `M`, so checking one covers all three.

Each assertion carries a short message naming exactly what went wrong, so a
failure in a debug build points straight at the mismatch (disagreeing `P`,
disagreeing `M`, or an uninitialized view) rather than leaving you to guess.

There is no return value and no "recover" path — a failure here means the caller
wired up its inputs wrong, which is a programming bug to be fixed, not a data
condition to be handled at runtime.
