# `index_cast.hpp` reference

## 1. Purpose

`src/core/internal/index_cast.hpp` holds a couple of tiny, boring conversion
helpers — but boring is exactly the point. Their whole job is to let the flat-array
index math in core's host-reference headers read as plain arithmetic instead of
being cluttered with spelled-out C++ casts.

Here is the problem they solve. steppe stores matrices as one long flat array, so to
reach element `(i, j)` of an `n`-column matrix the code computes a single offset like
`i + n * j`. In C++ those subscripts want to be `std::size_t` (the unsigned type array
indexing uses), but the loop counters and shape values that feed them are signed
(`int` / `long`, because they come from parity-matching CPU-reference code and can be
compared against negatives). Mixing signed and unsigned in one expression trips
compiler warnings and, worse, can silently wrap a stray negative into a gigantic
positive index. The safe-but-noisy fix is to write `static_cast<std::size_t>` around
every single term — which turns a clean `i + n * j` into a wall of casts that buries
the actual math.

So this header gives that cast **one canonical home**. `idx(i)` is that
`static_cast<std::size_t>` and nothing more, but now the index math reads as
`idx(i) + idx(n) * idx(j)` — the intent survives, the warning is silenced in exactly
one reviewed spot, and every subscript in core converges on the same helper instead
of each file open-coding its own cast. If the casting convention ever needs to change
(say, to add a bounds assertion in debug builds), it changes here once rather than in
a few hundred scattered subscripts.

Everything in this file is host-only and `constexpr`: pure compile-time-capable
scalar conversions with no GPU, no allocation, and no side effects. Each is `noexcept`
and `[[nodiscard]]` (the return value is the whole point, so throwing it away is
almost certainly a bug).

---

## 2. The index cast (`idx`)

```
idx(long i) -> std::size_t
```

The single signed→`size_t` cast. It takes a signed index or shape value and returns
the `std::size_t` the array subscript actually wants:

```
return static_cast<std::size_t>(i);
```

That is the entire function — a one-line, bit-faithful wrap of the explicit cast, so
swapping `static_cast<std::size_t>(i)` for `idx(i)` never changes a computed offset.
What it buys is readability: the flat-index expressions in core's reference headers
can be written the way you would write them on paper (`idx(i) + idx(n) * idx(j)`)
without a compiler warning about signed/unsigned arithmetic on every line.

Note the parameter is deliberately `long`, the widest signed index type core uses, so
a smaller `int` counter promotes into it cleanly and one overload covers every caller.

---

## 3. The long-double promotion (`ld`)

```
template <class T> ld(T x) -> long double
```

A sibling helper with the same de-noise motivation, but for a different cast. steppe's
CPU-oracle reference paths accumulate some sums in `long double` (extra precision, so
the oracle stays trustworthy even where the double-precision production path would lose
a few digits). Written out, each term wants a `static_cast<long double>` around it;
`ld(x)` is that cast given one name, so the accumulation reads as `ld(a) * ld(b)`
instead of a thicket of casts.

It is a template so it accepts every term type these sums mix — `double`, `int`,
`long`, `std::uint8_t` — and each of those converts to `long double`
value-preserving, meaning `ld` is a bit-identical stand-in for the explicit cast it
replaces. Like `idx`, it exists purely so the reference numerics read cleanly and the
cast lives in one reviewed place.

---

## 4. The non-negative count cast (`nonneg_count`)

```
nonneg_count(int  n) -> std::size_t
nonneg_count(long n) -> std::size_t
```

The same idea as `idx`, but for a value that is a **length or count** rather than an
index, with one extra safety step: it clamps a negative input up to `0` before
casting.

```
return idx(n < 0 ? 0 : n);
```

The distinction matters because of *how* a negative could sneak in. A shape field on a
freshly constructed, not-yet-populated object can legitimately sit at a sentinel
negative value ("no data loaded yet"). If that `-1` were cast straight through `idx`,
it would wrap to an astronomically large `std::size_t` — and a count that huge, used as
a loop bound or an allocation size, is a crash or a runaway waiting to happen. Clamping
to `0` first means an uninitialized shape reads as "zero elements", which is the safe,
do-nothing interpretation.

So the rule of thumb is: use `idx` when you have a genuine index you trust to be
in-range, and `nonneg_count` when you have a count/length that must never wrap even if
the object it came from is in an empty or half-built state. There are two overloads —
`int` and `long` — so callers get the clamp whether their count is stored narrow or
wide, without writing the cast (or the `n < 0 ? 0 : n` guard) themselves. And because
it is built on `idx`, the two share the single canonical cast rather than drifting into
two different conversions.

---
