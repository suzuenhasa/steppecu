# `launch_config.hpp` reference

## 1. Purpose

`src/core/internal/launch_config.hpp` is the single home for the math that turns a
problem size into a GPU launch grid. Before a GPU kernel runs, the caller must tell
it how many blocks of threads to launch along each axis. This header holds the small
set of helper functions and named constants that compute those numbers. Kernel files
never work out grid math on their own — they call into here.

The file holds four kinds of thing:

1. **The hardware grid-dimension limits** (`kMaxGridX`, `kMaxGridY`, `kMaxGridZ`) —
   the largest number of blocks CUDA will accept along each of the three grid axes.
2. **Ceiling division** (`cdiv`) — the one building block every grid calculation
   uses, in both an `int` and a `long` form.
3. **The grid-extent helpers** (`grid_for`, `grid_z_extent`, `grid_stride_extent`,
   `grid_for_x`) — the functions that compute how many blocks to launch, with the
   hardware limit checked or clamped automatically.
4. **The decode kernel's block shape** (`kDecodeBlockX`, `kDecodeBlockY`) — the one
   kernel whose thread block is deliberately not square, named here rather than
   re-picked inside the kernel.

Everything here is plain compile-time integer arithmetic. The functions are marked so
they can be called from host code and from device (GPU) code alike, and they fold to a
constant at compile time. The header pulls in no CUDA — only a tiny host/device macro
helper and the library's `config.hpp` — so it can be compiled by both the CPU-only
core library and the GPU kernels without forcing either to include the other's code.

---

## 2. Hardware grid-dimension limits

A CUDA launch grid has three axes: x, y, and z. Each axis has a hardware cap on how
many blocks it may span, and the caps are **not the same for every axis**. These
limits hold on every GPU generation steppe targets, including the newest Blackwell
(sm_120) cards.

| Constant | Value | What it caps |
|---|---|---|
| `kMaxGridX` | `2,147,483,647` (2^31 − 1) | The maximum number of blocks along the x axis. This is the **only** axis that reaches into the billions. |
| `kMaxGridY` | `65,535` | The maximum along the y axis. Far smaller than x. |
| `kMaxGridZ` | `65,535` | The maximum along the z axis — the same cap as y. |

Two consequences follow from x being the only large axis:

- **The big axis must ride x.** Whenever one dimension of a problem is huge — the
  number of genetic markers (SNPs), which can run into the hundreds of millions — that
  dimension is mapped onto the x axis, because it is the only one that can hold it. A
  smaller dimension (like the population count) rides y or z. This orientation rule is
  enforced by how the launch wrappers are written, not left to chance.
- **An over-limit axis fails loudly.** If a launch asks for more blocks than an axis
  allows, CUDA rejects it with a generic "invalid configuration" error that gives no
  clue which axis overflowed. To avoid that, the two helper functions below route every
  axis extent through a bounds check that fires with a precise message in debug builds,
  turning an opaque post-launch failure into an immediate, named one.

`kMaxGridZ` is deliberately defined as a copy of `kMaxGridY` rather than as its own
typed-out `65535`. The two axes share the same cap, so it is written once. If a future
GPU generation ever changed one axis's limit, only one line would need editing instead
of two that must be kept in step.

---

## 3. Ceiling division (`cdiv`)

`cdiv(n, block)` computes `ceil(n / block)` — how many blocks of `block` threads it
takes to cover `n` elements, rounding up so the last, partly-full block is still
counted. This is the fundamental building block of every grid calculation. It exists
so the rounding-up idiom is written and named in one place instead of being open-coded
as `(n + block - 1) / block` at each launch site.

Contract: `block` must be greater than 0 (it is a block dimension), and `n` must be
non-negative (it is a count or a dimension).

There are two overloads:

| Overload | Use it for |
|---|---|
| `cdiv(int, int)` | Ordinary dimensions that fit comfortably in a 32-bit `int` — for example the population count. |
| `cdiv(long, long)` | The SNP-count axis, which can exceed the ~2.1-billion limit of a 32-bit `int`. The SNP axis is stored as a `long` throughout the codebase, and its grid math **must** go through this wider overload. Using the `int` form on a SNP-scale count would overflow. |

---

## 4. Grid extent for square f2 launches (`grid_for`)

`grid_for(n, block, max_grid)` returns the number of blocks needed to cover `n`
elements along one axis — the same value `cdiv` gives, but with the hardware axis
limit checked at the same time. It is the helper the f2 (pairwise-distance) kernels use
to size each axis of their launch grid.

It has two defaults tuned for the common case:

- `block` defaults to `kCdivBlock` — the library's standard square block edge (a 16×16
  block of threads). The f2 output is a square P×P grid of population pairs, so a
  typical call is `grid_for(P, kCdivBlock)` for each of the x and y axes.
- `max_grid` defaults to `kMaxGridY` (65,535) — the conservative cap that is correct
  for any axis **other** than x. A caller mapping an extent onto the x axis passes
  `kMaxGridX` explicitly to get the larger allowance.

The bounds check is the point of the function. It asserts that the computed extent fits
within `max_grid`; if it does not, it fails immediately with a message saying the axis
overflowed and should be re-oriented onto x or tiled — rather than letting the launch
fall over later with an unhelpful CUDA error. In release builds the assert compiles
away entirely, so the check costs nothing in production while remaining load-bearing
during development.

**Caveat — square blocks only.** `grid_for` bakes in the square-block assumption via its
`kCdivBlock` default and is `int`-only. A kernel whose block is **not** square (the
decode kernel, below, uses a 32×8 block) must instead call `cdiv` directly with its own
per-axis block dimensions. And a kernel whose axis is the SNP count must use the
`cdiv(long, long)` overload, which `grid_for` cannot reach. The decode kernel still
routes its population (y) axis through `grid_for` so the y-axis cap check still applies
there.

---

## 5. Batch axis extent (`grid_z_extent`)

`grid_z_extent(n_in_group)` returns the z-axis extent for the batched f2
gather/scatter launches — the ones that process many genome blocks at once by putting
the batch count on the z axis (`gridDim.z = n_in_group`).

These launches set the z axis to the batch count **directly**, not through `grid_for`,
so they would slip past `grid_for`'s cap check. `grid_z_extent` is their dedicated
guard. It asserts the batch count is in the valid range `[1, kMaxGridZ]`:

- A count of **zero** is an invalid launch (a grid axis cannot be empty).
- A count **over** `kMaxGridZ` (65,535) is the failure that would only surface on a
  high-memory GPU, where a single size-bucket's block count can grow large enough to
  breach the z cap.

It then returns the count as an `unsigned`. The batched backend already tiles its work
into chunks of at most `kMaxGridZ` blocks, so this precondition always holds at the call
site in practice; the assert exists to pin that invariant so a future change can't
quietly violate it.

---

## 6. Decode-kernel block geometry

These two constants define the thread-block shape of the decode kernel — the kernel
that unpacks the compressed genotype records into per-population allele frequencies. Its
block is deliberately **not** square, so it does not reuse the standard `kCdivBlock`
edge; the two chosen dimensions are named here rather than typed as bare numbers inside
the kernel.

| Constant | Value | Why this value |
|---|---|---|
| `kDecodeBlockX` | `32` | The SNP-axis (x) edge of the block. 32 is exactly one warp — the group of 32 GPU threads that execute together. Putting a full warp along the SNP axis means the warp's 32 threads read 32 neighbouring SNPs of one record, so the packed-byte memory reads are contiguous and coalesce into efficient wide loads. Using the square block's edge of 16 (a half-warp) would halve that coalescing run and waste memory bandwidth, which is why the decode kernel picks its own edge. |
| `kDecodeBlockY` | `8` | The population-axis (y) edge. 8 keeps the block at 32 × 8 = 256 threads, a standard, occupancy-friendly block size for a memory-bandwidth-bound kernel, while spending the warp on the SNP axis where coalescing matters. |

---

## 7. 1-D grid-stride launch extent (`grid_stride_extent`)

`grid_stride_extent(total, block)` sizes a **grid-stride** 1-D launch: a kernel whose
threads loop over `total` elements, so the grid does not have to cover every element
one-to-one. It returns `cdiv(total, block)` blocks, but **clamped** to `kMaxGridX`
rather than asserted against it. `total` is a `long` because these launches flatten a
multi-axis problem (for example `nl × nr × nb × n_models`) into a single count that can
run past the 32-bit range; the ceiling division goes through the `long` overload and the
result is narrowed back to the `int` a launch expects (`kMaxGridX` is `2^31 − 1`, so a
clamped value always fits an `int`).

Clamping instead of asserting is correct here precisely because the kernel is
grid-stride: if the block count would exceed the x-axis cap, launching the cap's worth
of blocks is safe — each thread's stride loop still visits every element, it just does
more iterations. This is the helper the qpAdm fit and f-statistic launch wrappers use
for their flattened element-wise passes.

---

## 8. Single-axis grid extent, x-axis checked (`grid_for_x`)

`grid_for_x(n, block, what)` is the `grid_for` counterpart for a one-to-one 1-D launch
whose single axis rides x. Like `grid_for` it returns `cdiv(n, block)` with the
hardware limit checked by assert — but against `kMaxGridX` (the large x cap) rather than
`grid_for`'s conservative `kMaxGridY` default, and it takes `n` as a `long` so a
SNP-scale count cannot overflow the division. The `what` argument is a caller-supplied
message naming the axis, so an over-limit launch fails with a precise, per-call
diagnostic instead of a generic CUDA error. As with every helper here the assert
compiles away under `NDEBUG`, leaving a bare ceiling division in release builds.
