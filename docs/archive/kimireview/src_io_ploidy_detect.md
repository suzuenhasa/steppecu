I read this carefully. It is a tiny, focused leaf-node function, and **it is not slop** — but a senior reviewer would still have a few stylistic nits.

## What's genuinely good

- **The AT2 rule is implemented exactly right.** Initialize every sample to pseudo-haploid (ploidy 1), scan the capped prefix window, and bump to diploid (ploidy 2) on the first heterozygous call. The comment at lines 17–20 explains *why* the values are domain literals rather than pulling in `core` constants — that is thoughtful dependency hygiene.
- **Good defensive shape.** The early return on line 27 handles the empty-tile / zero-window cases cleanly, and the window cap on line 26 (`std::min(kPloidyDetectSnps, tile.n_snp)`) matches AT2's behavior when `ntest > nsnp`.
- **Clear loop intent.** The inner loop breaks immediately on the first het (line 40), which is the correct fast-path: once we see a het, the sample is definitively diploid.
- **Appropriate abstraction.** Using `code_in_byte(rec[byte_in_rec], pos_in_byte)` and `kHetCode` keeps the bit-packing details in `eigenstrat_format.hpp` where they belong.
- **Pure function, no I/O, no mutation.** It reads the same packed bytes the decode path will use, so there is no duplicated disk read and no side effects.

## What a senior developer would flag

**Redundant `static_cast<std::size_t>` around an `int` constant:**

```cpp
const std::size_t byte_in_rec =
    s / static_cast<std::size_t>(kCodesPerByte);          // line 34-35
const int pos_in_byte =
    static_cast<int>(s % static_cast<std::size_t>(kCodesPerByte));  // line 36-37
```

`kCodesPerByte` is `inline constexpr int kCodesPerByte = 4;`. In `s / kCodesPerByte`, the usual arithmetic conversions already promote the `int` to `std::size_t`, so the explicit cast on line 35 is noise. The inner cast on line 37 is similarly redundant for the modulo, and then you have to cast back to `int` because `code_in_byte` takes `int k`. A senior dev would likely write:

```cpp
const std::size_t byte_in_rec = s / kCodesPerByte;
const int pos_in_byte = static_cast<int>(s % kCodesPerByte);
```

or, if you really want explicit width:

```cpp
const auto pos_in_byte = static_cast<int>(s % kCodesPerByte);
```

It is not a bug, but it reads like someone who does not trust the conversion rules.

**The `int` return vector for ploidy:**

```cpp
std::vector<int> ploidy(tile.n_individuals, kPloidyPseudoHaploid);  // line 23
```

`int` is fine, but ploidy is conceptually a small enum or `std::uint8_t`. The header documents the contract well, so this is a minor API-grain complaint, not a real problem.

**Implicit trust in the tile's packed buffer:**

```cpp
const std::uint8_t* rec = tile.packed.data() + g * tile.bytes_per_record;  // line 30
```

There is no bounds check here. That is appropriate for a hot-path leaf function — the contract is enforced by `GenotypeTile` construction — but a senior reviewer will note that this function has no way to recover if `tile.packed` is shorter than `n_individuals * bytes_per_record`. A comment referencing that contract (or a `[[pre: ...]]` if the project uses contracts) would make the invariant explicit.

**Could be slightly more efficient, though clarity wins.**

Iterating per-SNP and recomputing `byte_in_rec` and `pos_in_byte` for every sample is simple and correct. A more performance-obsessive version might iterate packed bytes and test all four codes at once with a small lookup or bit trick. For a 1000-SNP window, the current code is almost certainly fast enough and much easier to audit, so this is not a real recommendation — just an observation that the author chose clarity over micro-optimization.

## The "slop" test

**Not slop.** There are no magic numbers without explanation, no copy-pasted drift, no stale comments, no missing error handling beyond the contract, and the algorithm is directly verifiable against the documented AT2 reference behavior.

## What it actually looks like

This looks like **competent, careful leaf-node utility code written by someone who understands the domain rule and values readability over cleverness.** It is the kind of small function you want in an `io` layer: single responsibility, well-commented, no hidden state, and easy to unit-test against a reference implementation. There is no CUDA here and no need for any; the author respected the architectural boundary described in the header.

A senior C++ reviewer would say: "Correct, clean, ship it — but drop the redundant casts." A senior genomics reviewer would say: "This matches AT2's `cpp_readgeno.cpp` ploidy logic; good."

**Verdict:** Solid A- production code. The only thing keeping it from a straight A is the noisy casts and the unspoken packed-buffer contract.
