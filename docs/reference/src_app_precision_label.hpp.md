# `precision_label.hpp` reference

## 1. Purpose

`src/app/precision_label.hpp` provides one small shared helper,
`precision_label`. It takes a resolved precision setting and returns a short
human-readable string — `"emu"`, `"tf32"`, or `"fp64"` — that names which flavor
of floating-point math the computation actually ran in.

That string is recorded in the small `meta.json` file that sits alongside a
directory of f2 blocks. It is the tag that says which precision was *engaged* for
a given cached f2 result, so anyone inspecting the cache later can see how it was
produced.

The helper is used by more than one command-line entry point (the extract-f2
command and the qpfstats command). Those two commands previously each carried a
byte-for-byte identical copy of the same little function. Moving it into this
single header removes that duplication, so there is now exactly one definition of
the label mapping instead of two copies that could drift apart.

---

## 2. The precision label mapping

`precision_label` maps each precision mode to a fixed short string:

| Precision mode | Label | What it means |
|---|---|---|
| `EmulatedFp64` | `"emu"` | Emulated double precision — the default, matrix-multiply-heavy mode. |
| `Tf32` | `"tf32"` | TF32 tensor-core arithmetic — the lower-precision, opt-in screening mode. |
| `Fp64` | `"fp64"` | Native double precision — the reference/fallback mode. |

The label describes the precision that was actually resolved and engaged along
the f2-directory writer path, not a mode the caller merely requested. It is
purely descriptive metadata — nothing downstream branches on the label, it is
written for a human reading the cache.

---

## 3. Exhaustive switch and the `fp64` fallback

The function is written as a `switch` over the precision mode rather than a chain
of `if`/ternary expressions. This is deliberate. If a new precision mode is ever
added to the underlying enumeration, the compiler can flag this exact switch as a
place that is missing a case, making the required update visible at build time. A
ternary chain would instead silently fall through to a default without warning.

After the switch there is a trailing `return "fp64"`. This makes native double
precision the fallback label for any value the switch did not match. Combined
with the exhaustive-switch style above, native `fp64` is both a real case and the
safety net.

The function is marked `inline` (it lives entirely in the header) and
`[[nodiscard]]` (its return value is the whole point, so discarding it is almost
certainly a mistake and the compiler will warn).

---

## 4. Relationship to `precision_str`

There is a similar-looking helper elsewhere in the code base — `precision_str` in
the result-emitting code — that also produces the same `emu`/`tf32`/`fp64`
vocabulary. The two are intentionally kept separate.

The difference is the input and the destination:

- `precision_str` takes a bare precision *mode* value and fills the `precision`
  column of a result row.
- `precision_label` takes the whole resolved precision *setting* that the
  f2-directory writer path produces, and fills the engaged-precision tag in an
  f2 `meta.json`.

Because their signatures differ (one takes the mode alone, the other takes the
full setting), they stay as two distinct helpers even though the strings they can
return are identical. They are not interchangeable.

---

## 5. CUDA-free by contract

This header is plain C++20 and belongs to the host-side application layer only.
It includes just the CUDA-free configuration header to get the precision type,
and it pulls in no CUDA headers of its own. Keeping it free of any GPU dependency
is an enforced contract, not an accident: it means this helper stays cheap to
include from host-only command code without dragging the GPU compilation
toolchain along with it. Anyone editing this file must preserve that — do not add
a CUDA include here.
