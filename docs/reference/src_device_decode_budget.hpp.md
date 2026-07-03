# `decode_budget.hpp` reference

## 1. Purpose

`src/device/decode_budget.hpp` is the single home for one decision: how many SNPs
the on-device genotype decode should process at a time. That count is called the
**tile width**. The header holds two named constants, one small helper, and two
budget functions that together pick the tile width from a live reading of how much
GPU memory is currently free.

The header is intentionally free of any CUDA code. It uses only plain host C++ and
`std::size_t` integer math, following the same pattern as the sibling VRAM-budget
header. That keeps it lightweight enough to include from the CUDA-free application
layer that drives the decode, and — just as important — it means every function
here can be unit-tested on a machine with no GPU at all.

---

## 2. Why the decode is tiled

The genotype decode step turns the packed, on-disk genotype data for a block of SNPs
into three full floating-point arrays in GPU memory — one for the allele-frequency
estimate, one for its variance, and one for the sample count — each sized
`populations × SNPs`. These three arrays are built for the *entire* block of SNPs
**before** any quality filter or memory-tier decision runs. The filters and tiers
only ever shrink the *result*; they are decided after the full decode has already
allocated its buffers.

That ordering is the problem. On a large panel the un-filtered decode buffers are
enormous: at 700 populations and 2.14 million SNPs the resident buffers come to
roughly 36 GB, which overflows a 32 GB GPU and crashes with an out-of-memory error.
Turning on a memory tier does not help, because the tier governs only the final
result and is chosen after the decode has already tried to allocate its full-size
buffers.

The fix is to decode a *slice* of SNPs at a time instead of the whole block. Peak GPU
memory then scales with `populations × tile_width` rather than
`populations × total_SNPs`, so a small enough tile always fits. This header is the
piece that sizes that tile against the current free-memory reading.

---

## 3. Staying parity-neutral, and the alignment rule

Tiling the decode changes **how many** slices the work is broken into — never the
numbers that come out. Three properties make the tiled path produce byte-for-byte the
same allele-frequency, variance, count, and kept-SNP arrays as the single-shot path,
for any legal tile width:

- Each SNP's keep-or-drop decision depends only on that SNP, so splitting the SNPs
  across tiles cannot change any decision.
- The per-tile scan that compacts the kept SNPs is monotone, so a tile boundary never
  reorders anything.
- The host appends each tile's kept results in file order, so the final arrays are
  assembled in exactly the original SNP order.

There is exactly **one** hard constraint the tile width must obey. The on-disk
genotypes are packed four codes to a byte (2 bits each). For a tile that starts at
SNP index `s_lo` to line up on a byte boundary of that packing, every tile start must
be a multiple of four. Since tile starts are multiples of the tile step, the step
itself must be a multiple of four. This header enforces that by rounding every tile
width it returns **down** to a multiple of four — both the computed budget and any
value forced through the environment override. The one exception is a single tile that
covers all SNPs: there the only start is `s_lo = 0`, which is already aligned, so the
width may equal the total SNP count even if that count is not a multiple of four.

---

## 4. Named constants

| Constant | Value | What it's for |
|---|---|---|
| `kDecodeTileVramFraction` | `0.6` | The fraction of currently-free GPU memory that one decode tile's working set is allowed to occupy. The decode holds about six `populations × tile_width` floating-point buffers plus per-SNP metadata plus the packed genotype slice, all at once. This share is deliberately lower than the fraction the main f2 stage uses, for two reasons: the decode buffers are short-lived, and leaving extra headroom lets the memory-tier stage that runs afterward have room to work. It only sets the tile count, so it is safe to tune and never changes a reported number. |
| `kDecodePerColMetaBytes` | `51` | The fixed per-SNP metadata footprint of the decode, in bytes, independent of how many populations or individuals there are. It is the sum of the small per-column fields the decode keeps for each SNP: reference and alternate allele (1 byte each), chromosome (4-byte integer), genetic and physical position (8 bytes each), a flags byte, an 8-byte keep-index, and the kept-SNP twins of chromosome (4 bytes), genetic position (8 bytes), and physical position (8 bytes). That is `1+1+4+8+8+1+8+4+8+8 = 51`. |

---

## 5. Per-column memory footprint — `decode_per_col_bytes`

```cpp
std::size_t decode_per_col_bytes(int P, std::size_t n_individuals) noexcept
```

Returns how many bytes of GPU memory the decode uses **per SNP column**, given the
population count `P` and the number of individuals. This is the denominator the budget
functions divide the free-memory allowance by to get a tile width. It has three parts:

- **The floating-point columns: `6 × P × 8` bytes.** Three resident decode columns
  (allele-frequency estimate, variance, count) plus three compacted output columns
  for the same three quantities. The compacted set is sized for the worst case in
  which every SNP in the tile is kept. Each value is an 8-byte double, and there are
  `P` of them per column, hence `6 × P × 8`.
- **The per-SNP metadata: `kDecodePerColMetaBytes` (51 bytes),** as described above.
- **The packed genotype slice: `ceil(n_individuals / 4)` bytes** — one column of the
  on-disk 2-bit packing sitting co-resident in GPU memory, four individuals per byte,
  rounded up.

A note on robustness: if `P` is passed as a negative number it is treated as zero, so
the function is well-defined on any input. Callers are still expected to guard that the
real population count is positive before they act on the result.

---

## 6. The pure tile-width budget — `decode_tile_snps_budget`

```cpp
long decode_tile_snps_budget(std::size_t free_vram, int P,
                             std::size_t n_individuals) noexcept
```

The largest tile width whose working set fits inside
`kDecodeTileVramFraction × free_vram`. It computes the allowance
(`0.6 × free_vram`), divides it by the per-column footprint from section 5 to get a
raw SNP count, then applies the two adjustments the alignment rule requires:

- Rounds the width **down** to a multiple of four, so every tile start stays aligned
  to the 2-bit packing (see section 3).
- Floors the width at four — at least one full packed byte's worth of SNPs — so a tiny
  free-memory reading can never drive the width to zero.

This function is `constexpr` and completely pure: it reads no environment variable and
makes no device call. It is the mathematical core that the two callers build on. The
environment override and the clamp against the real SNP count both live in the wrapper
in section 7, not here.

---

## 7. The final tile step — `decode_tile_snps`

```cpp
long decode_tile_snps(std::size_t free_vram, int P,
                      std::size_t n_individuals, long M) noexcept
```

The tile step the decode loop actually uses, where `M` is the total number of SNPs to
decode. It layers three things on top of the pure budget from section 6:

1. **Starts from the budget** for the given free memory, population count, and
   individual count.
2. **Applies an environment override.** If `STEPPE_DECODE_TILE_SNPS` is set to a
   positive number, that value *forces* the tile width, replacing the budget entirely.
   This is what lets a test drive many small tiles on purpose to exercise the tiling
   path. The forced value is still rounded down to a multiple of four and floored at
   four, so it cannot break the alignment rule.
3. **Clamps to `M`.** If the chosen width is larger than the total SNP count, it is cut
   down to `M` — a single tile covering everything. This is the one case where the
   returned width may not be a multiple of four, and it is safe: the only tile start is
   `s_lo = 0`, which is already aligned.

The function returns `0` when `M` is zero or negative, meaning there is nothing to
tile.

**What the returned step guarantees.** The value handed back is always one of two
things: either a multiple of four that is strictly less than `M` (so every tile start
`s_lo = k × step` lands on a 2-bit packing boundary), or exactly `M` (a single tile
starting at zero). Either way the alignment invariant from section 3 holds.
