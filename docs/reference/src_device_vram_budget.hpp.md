# `vram_budget.hpp` reference

## 1. Purpose

`src/device/vram_budget.hpp` is the single home for the integer arithmetic that
decides how the batched f2 computation is sized against GPU memory. It answers two
questions with one shared body of code:

1. **Will the run fit?** — an up-front check, done before any GPU work starts, that
   the resident working set plus one temporary chunk can live inside free GPU memory.
2. **How big can each chunk be?** — the runtime decision, made inside the compute
   loop, of how many jackknife blocks of one bucket to pack into a single batched
   matrix-multiply before memory runs out.

The header exists so those two questions are answered by *exactly the same*
functions. Before this code was pulled into its own file, the budget math lived
inline in the GPU backend, and the up-front check and the runtime chunk sizing could
drift apart. Keeping one implementation means they never can.

Everything here is plain `std::size_t` integer arithmetic. There is no GPU call and
no CUDA type anywhere in the file. That is deliberate: it makes the budget logic
runnable — and unit-testable — on an ordinary machine with no GPU, and it lets both a
host-only test and the GPU backend include the same header.

---

## 2. Where the file lives and why it is CUDA-free

The file sits in `src/device/` rather than in the core library, because what it
describes is a *device fact*: how large the resident GPU tensors are, and how much
scratch memory the matrix-multiply library reserves. That is device-path policy, so
it belongs with the device code.

But it is intentionally **CUDA-free** — it includes no CUDA header and makes no
device call. Because `src/` is on the device library's public include path, a plain
C++ test reaches this header as `"device/vram_budget.hpp"` and exercises the budget
math directly, with no GPU present. The GPU backend includes the very same header,
so the tested code and the shipped code are identical.

The only two headers it pulls in are the launch-configuration header (for the
hardware grid-z limit) and the library configuration header (for the tuning
fractions, the workspace size, and the structural stack counts).

---

## 3. What the budget accounts for

The batched f2 device path keeps **two** equal-sized double-precision tensors
resident in GPU memory for the entire computation loop:

- the **f2 tensor** itself, and
- the **paired-variance tensor** (the weight the block jackknife needs).

Each of these is shaped `[P × P × n_block]`, where `P` is the number of populations
and `n_block` is the number of jackknife blocks. Each therefore occupies
`P² · n_block · 8` bytes (8 bytes per double). Because both are held at once, the
resident footprint of the pair is `2 · P² · n_block · 8` bytes.

An earlier version of the budget counted only *one* of these two tensors, which
under-reserved the resident working set by roughly a factor of two and could commit
more memory than actually existed. The functions here fix that by reserving **both**
tensors — plus the matrix-multiply library's fixed scratch workspace — *before*
applying the utilization fraction. The fraction is then scaling only the memory that
genuinely remains for one temporary chunk, never the gross free figure. This is the
correctness core of the file.

The count of resident tensors (`2`), the two chunk stack counts (`4` inputs and `4`
outputs, described in section 7), and the workspace size are all named constants in
the library configuration header, so this file and the memory-tier code derive the
same footprints from one source instead of re-typing the numbers.

---

## 4. The utilization-fraction compile-time check

A `static_assert` at the top of the file pins the legal range of the VRAM
utilization fraction to `(0, 1]` at compile time, right where the budget math
consumes it.

The reasoning is: a fraction of zero or less would budget *zero* bytes for a chunk —
a guaranteed clamp back up to a single block and a likely out-of-memory failure — and
a fraction above one would over-commit, handing out more than the free memory. The
whole point of the fraction is to *never* commit all of free VRAM, so a value outside
`(0, 1]` is a build error rather than a runtime surprise. The default value is
`0.80`.

---

## 5. Shared integer helpers: `nonneg` and `budget_bytes`

Two small building blocks are used by the rest of the file. Both are the *single
home* for a pattern that used to be spelled several different ways across the device
layer, so those sites can no longer drift apart.

### `nonneg(int v) -> std::size_t`

Clamps a possibly-negative `int` to zero, then widens it to `std::size_t`. Every
byte-count helper here calls this on its `int` inputs *before* multiplying, so a
stray negative value can never wrap around during the unsigned multiply and produce a
gigantic bogus size. The callers already guarantee their inputs are positive; this
keeps each helper well-defined even if that guarantee were ever violated.

### `budget_bytes(double fraction, std::size_t free) -> std::size_t`

Returns `floor(fraction · free)` — a fractional share of a free-byte figure. The
product is computed in `double` and truncated to whole bytes. `fraction` is a tunable
policy share in `(0, 1]`; `free` is always a runtime memory probe, never a hardcoded
number. This is the one place the "take a fraction of a free-byte figure" pattern
lives, shared by the chunk budget here and by the memory-tier thresholds elsewhere.
Changing a budget threshold moves no reported bits.

---

## 6. `resident_tensor_bytes(P, n_block)` — the co-resident tensor footprint

Returns the number of bytes held resident for the whole run:

```
2 · P² · n_block · sizeof(double)
```

This is the two-tensor term from section 3 — the f2 tensor and the paired-variance
tensor, each `[P × P × n_block]` doubles. All of the arithmetic is done in
`std::size_t` (each `int` is widened *before* being multiplied) so the product cannot
overflow a 32-bit count. If `P` or `n_block` is zero or negative, the function returns
`0`.

| Parameter | Meaning |
|---|---|
| `P` | number of populations (the edge length of the `P × P` slab) |
| `n_block` | number of jackknife blocks (the batch axis length) |

---

## 7. `per_block_chunk_elems` / `per_block_chunk_bytes` — one chunk's per-block footprint

A single batched chunk processes several jackknife blocks at once. These two
functions give the footprint of **one block within a chunk** — the first in
double-precision elements, the second in bytes.

### `per_block_chunk_elems(P, s_pad) -> std::size_t`

Returns the double-count of one block's share of a chunk:

```
kChunkInputStacks · P · s_pad  +  kChunkOutputStacks · P²
     (= 4 · P · s_pad)                (= 4 · P²)
```

The two coefficients are structural, not tunable, and are named once in the
configuration header. They break down as:

- **Four input stacks**, each `P · s_pad` doubles: the gathered genotype input, the
  gathered variance input, and the gathered scale input (which counts as two).
- **Four output stacks**, each `P²` doubles: the two matrix-multiply outputs (one
  each) and the result buffer (which counts as two).

Here `s_pad` is the bucket's padded SNP-block width (always at least 1 by
construction). This function is the single source of the chunk's per-block
coefficients; the byte version below and the memory-tier code both derive from it, so
the coefficients are maintained in exactly one place.

### `per_block_chunk_bytes(P, s_pad) -> std::size_t`

Simply `per_block_chunk_elems(P, s_pad) · sizeof(double)` — the same figure expressed
in bytes.

---

## 8. `chunk_budget_bytes(free_vram, P, n_block)` — VRAM left for one chunk

Returns how many bytes are available for a single chunk's temporary slabs, given the
current free GPU memory.

The calculation is:

1. Reserve the two resident tensors (`resident_tensor_bytes`, section 6) **and** the
   matrix-multiply scratch workspace.
2. Subtract that reserved amount from `free_vram`, saturating to `0` if free memory
   cannot even cover the resident set plus workspace.
3. Scale what remains by the utilization fraction (section 4), via `budget_bytes`.

Subtracting *both* resident tensors and the workspace *before* scaling is what makes
the budget sound no matter whether those resident allocations happen before or after
the free-memory probe was taken. If free memory is too small even for the resident
set, the function returns `0`; the caller then clamps up to a single block and may run
out of memory cleanly — a fail-fast, not silent corruption.

| Parameter | Meaning |
|---|---|
| `free_vram` | free device memory in bytes (e.g. the value a memory probe reports) |
| `P` | number of populations |
| `n_block` | number of jackknife blocks (for the resident-tensor reserve) |

---

## 9. `max_blocks_per_chunk` — the blocks-per-chunk gate and grid-z tiling

```
int max_blocks_per_chunk(free_vram, P, n_block, s_pad, nb_total)
```

This is the top-level decision the file exists to make: the largest number of blocks
of one bucket that fit in a single batched chunk. It divides the chunk budget
(section 8) by the per-block footprint (section 7), then clamps the result.

### The overflow-safe clamp

The quotient `budget / per_block` is computed and clamped entirely in `std::size_t`
**before** it is narrowed to `int`. This order matters: if the quotient were narrowed
first, a value above the `int` maximum would cast to a negative number and then clamp
catastrophically down to a single block. Working in `std::size_t` the whole way avoids
that.

The final result is:

```
min(quotient, nb_total, kMaxGridZ)   then floored at 1
```

so it is never more blocks than the bucket actually has, never more than the hardware
grid limit (below), and never fewer than one — a single block must always be
attempted, so that if the budget says zero the run out-of-memories cleanly mid-chunk
rather than silently producing nothing. The returned `int` is therefore always in
`[1, min(nb_total, kMaxGridZ)]` and can never overflow. If `nb_total` is zero or
negative (an empty bucket) the function returns `0` — there is nothing to chunk.

The divide is guarded against a zero per-block footprint even though the caller's
`P > 0` guarantee makes that impossible, so the helper is well-defined on any input.

### Grid-z tiling

Each block of a chunk becomes one slab on the third grid dimension (`gridDim.z`) of
the gather and scatter kernel launches. That dimension is capped by the hardware at
`kMaxGridZ` (65,535) on every GPU. On a small-memory GPU the memory quotient is far
below that limit, so it never matters. But on a large-memory GPU — where `n_block` is
larger and the per-block footprint smaller — the memory quotient can climb *above*
65,535, which would make the kernel launch fail on exactly that class of machine.

Capping the blocks-per-chunk at `kMaxGridZ` here makes the existing per-bucket chunk
loop tile the batch over the grid's third dimension with no new loop: a bucket with
more than 65,535 blocks is simply processed in chunks of at most 65,535 blocks. The
launch is then correct on both small- and large-memory GPUs.

| Parameter | Meaning |
|---|---|
| `free_vram` | free device memory in bytes |
| `P` | number of populations |
| `n_block` | number of jackknife blocks (for the resident-tensor reserve) |
| `s_pad` | the bucket's padded block width |
| `nb_total` | number of blocks in this bucket (the upper clamp) |
