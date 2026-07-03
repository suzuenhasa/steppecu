# `f2_batched_kernel.cuh` reference

## 1. Purpose

`src/device/cuda/f2_batched_kernel.cuh` declares three narrow launch-wrapper
functions for the *batched* way of computing the f2 statistics one genome block at
a time. It is a header of function signatures only — there is no kernel code in it.

The division of labor is deliberate. Host orchestration code (the backend) calls
these plain `launch_*` / `run_*` functions. The things that only the GPU compiler
should ever see — the actual kernel bodies, the `<<<...>>>` launch syntax, and the
calls into the batched matrix-multiply library — all live in the matching `.cu`
file. The host never includes a kernel body. This header is the seam between the
two.

Because these signatures name a CUDA type (a matrix-multiply library handle), the
header is private to the device layer of steppe. It is the internal boundary
between the backend and the kernels, *not* part of the CUDA-free public interface
that the rest of the library, the command-line tool, and the language bindings are
allowed to include.

The three functions correspond to the three stages of processing one *size group*
of genome blocks: gather the group's data into a padded layout, run the three
matrix multiplies, then assemble and scatter the results back. Section 3 describes
the data that flows between them; sections 4 through 6 cover each stage; section 7
covers the shared preconditions and failure behavior.

---

## 2. Why the batched f2 path is grouped by block size

steppe estimates uncertainty with a block jackknife, which splits the genome's SNPs
into many contiguous *blocks*. The f2 computation for each block is three matrix
multiplies. With hundreds of populations and hundreds of thousands of SNPs there
can be many hundreds of blocks, so *how* those per-block multiplies are issued
matters a great deal for speed and for memory.

Three designs were measured on real data (768 populations, 584,131 SNPs), and this
header implements the winner:

- **A loop of one matrix-multiply per block.** Correct, but dominated by the
  per-launch overhead of issuing that many small calls — measured at 591 ms.
- **One batched call sized to the largest block.** Every block is padded out to the
  single largest block's width and all of them run as one batched call. This is
  simple but wasteful: at 768 populations it needs 53.8 GB of GPU memory, which does
  not fit in 32 GB, so it is not viable at all. Its padding waste is 2.76x.
- **Size-grouped batched (the chosen design).** Blocks are bucketed by their SNP
  count *rounded up to the next power of two*. Every block in a bucket is padded to
  that bucket's width, and the whole bucket runs as *one* batched matrix-multiply
  call per stage. This was both the fastest — 317 ms — and the only memory-frugal
  batched design, because only one group is resident at a time. On the real-data
  panel, 768 populations produced 10 groups with only 1.43x padding waste.

Power-of-two buckets are what keep the padding waste bounded: within a bucket no
block can be more than a factor of two smaller than the bucket width, while the
number of batched calls stays small (it grows only with the logarithm of the
largest block). The base used for that rounding, `kBlockGroupPadBase`, is `2` and
lives in the shared config header.

The padding is free in the sense that matters: pad columns are written as zeros, so
they contribute nothing to any of the multiplies. Measured against native double
precision, the grouped path runs about 7.2x faster at the 40-bit emulated precision
and about 8.9x faster at 32-bit, preserving the speed win of doing the f2 work as
large matrix multiplies.

---

## 3. The data that flows between the three stages

The same quantities appear in every function, so they are defined once here.

**Sizes.** `P` is the number of populations. `M` is the total number of SNPs.
`n_in_group` is how many blocks are in the size group being processed. `s_pad` is
the padded width every block in the group is stretched to (the group's
power-of-two bucket width).

**The feeder inputs** (`dQ_all`, `dV_all`, `dS_all`). These are the decoded,
per-SNP arrays produced by the earlier "feeder" stage, laid out column-major and
covering *all* `M` SNPs at once, block by contiguous block. `dQ_all` and `dV_all`
are `P × M` (one row per population, one column per SNP). `dS_all` is `2P × M` — it
stacks two per-population quantities on top of each other. A block's SNPs occupy a
contiguous run of columns inside these arrays, described by that block's offset and
size.

**The gathered slabs** (`dQg`, `dVg`, `dSg`). After the gather stage, one group's
data sits in a padded, batched *slab* layout — one slab per block, all slabs the
same `s_pad` width so a single batched call can sweep them. `dQg` and `dVg` are
`P × s_pad × n_in_group`; `dSg` is `2P × s_pad × n_in_group`.

**The matrix-multiply outputs** (`dGg`, `dVpairg`, `dRg`). Per slab:

- `dGg` is `P × P` — the block's f2 numerator ingredients, from Q times Q transposed.
- `dVpairg` is `P × P` — the paired-variance term, from V times V transposed. It is
  *retained* (not just an intermediate) because a later model-fitting stage needs it
  as a weight.
- `dRg` is `2P × P` — from S times V transposed. Its top `P` rows and bottom `P`
  rows hold the two accumulated sums the final f2 formula divides together.

**The resident tensors** (`dF2_all`, `dVpair_all`). The final destinations, held in
GPU memory for the whole computation: the f2 tensor and its paired-variance tensor,
each `P × P × n_block`. The assemble stage writes each block's finished `P × P`
result into its own slab of these.

---

## 4. Stage one — gathering a group into padded slabs (`launch_gather_group`)

```
void launch_gather_group(const double* dQ_all, const double* dV_all, const double* dS_all,
                         const int* d_block_ids_in_group, const long* d_block_offsets,
                         const int* d_block_sizes,
                         int P, int s_pad, int n_in_group,
                         double* dQg, double* dVg, double* dSg,
                         cudaStream_t stream);
```

This copies one size group's SNP columns out of the block-contiguous feeder arrays
and into the padded batched slab layout that the batched matrix-multiply call
expects.

The group holds `n_in_group` blocks. Block `k` of the group is the global block
`d_block_ids_in_group[k]`, whose SNPs live in columns
`[d_block_offsets[id], d_block_offsets[id] + d_block_sizes[id])` of the feeder
arrays. Each block is copied into a slab that is padded out to `s_pad` columns.

**Padding is written as zero.** For any column at or beyond a block's real size, the
gather writes 0 into Q, V, and S. Because those pad columns are zero, they add
nothing to the later multiplies — a zero row or column drops straight out.

The gather runs entirely in native double precision. It is a memory-bound copy, so
reduced precision would buy nothing.

Preconditions and failure behavior are covered in section 7.

---

## 5. Stage two — the three f2 matrix multiplies (`run_f2_gemms_group`)

```
void run_f2_gemms_group(cublasHandle_t handle, const Precision& precision,
                        int P, int s_pad, int n_in_group,
                        const double* dQg, const double* dVg, const double* dSg,
                        double* dGg, double* dVpairg, double* dRg);
```

This runs the three f2 matrix multiplies for one size group, each as a single
batched call over the group's `n_in_group` padded slabs (each slab `s_pad` wide):

- `dGg` = Qg times Qg transposed, per slab.
- `dVpairg` = Vg times Vg transposed, per slab — kept for the later fitting stage.
- `dRg` = Sg times Vg transposed, per slab.

### Precision applies only to these multiplies

The `precision` argument governs **only** the three matrix multiplies here. When it
is `EmulatedFp64`, they run the fixed-slice emulated double-precision math at
`precision.mantissa_bits`; when it is `Fp64`, they run native double precision. The
numerically delicate divide that produces the final f2 value happens later, in the
assemble stage, and is always native double precision regardless of this setting.

### The handle and workspace contract (do not add a stream here)

This function takes a matrix-multiply library `handle` but deliberately takes **no
stream** and never calls the library's "set stream" function. That is not an
oversight — it is required for reproducibility.

The handle must already be fully set up by the caller, once: its stream bound, and
its explicit determinism workspace bound. Engaging the precision policy on the
handle is likewise done once by the caller. This routine only sets the per-call
compute type for its three multiplies.

If this function called "set stream" on every group, the library would reset its
workspace back to the default internal pool before each group's multiplies. That
would silently defeat the fixed-workspace requirement that makes the emulated
double-precision path produce the exact same bits every run. So the stream is bound
once, upstream, and never touched here.

Preconditions and failure behavior are covered in section 7.

---

## 6. Stage three — assembling and scattering results (`launch_assemble_blocks_group`)

```
void launch_assemble_blocks_group(const double* dGg, const double* dVpairg, const double* dRg,
                                  const int* d_block_ids_in_group,
                                  int P, int n_in_group,
                                  double* dF2_all, double* dVpair_all,
                                  cudaStream_t stream);
```

This takes one group's matrix-multiply outputs and writes the finished per-block f2
values into the resident tensors.

For each slab `k`, holding the group-local outputs for global block
`d_block_ids_in_group[k]`, it fuses two steps: it forms the f2 numerator from `dGg`
and `dRg`, then divides to produce the final value. That divide is the
catastrophic-cancellation-prone step of the whole computation, so it runs in
**native double precision** — even when the multiplies in stage two ran in the
faster emulated mode. It uses the same shared numerator-and-finalize primitives as
the non-batched f2 path, so the two paths agree exactly.

The result is scattered into `dF2_all` at that block's `P × P` slab (element
`i + P*j + P*P*block_id`), and `dVpairg` is carried through unchanged into
`dVpair_all`.

Unlike the earlier stages, this function has no `s_pad` parameter — its inputs and
outputs are all `P × P` slabs, with the padding already gone by this point.

Preconditions and failure behavior are covered in section 7.

---

## 7. Preconditions and failure behavior

All three functions check their size arguments with debug-mode assertions. These
are fail-fast guards against a corrupted call — for example an empty SNP shard
arriving from the sharded path — not conditions the normal backend can ever
violate. The backend always satisfies them by construction: bucket widths are never
smaller than the power-of-two base, and the batch is always tiled into chunks no
larger than the driver's grid limit before these functions are called.

`kMaxGridZ` referenced below is the CUDA driver's hard limit on how many slabs a
single launch may batch along its z-axis.

| Function | Precondition | Why |
|---|---|---|
| `launch_gather_group` | `s_pad >= 1` | `s_pad` becomes the launch grid's y-extent; a value of 0 is a zero-height grid the driver rejects. |
| `launch_gather_group` | `1 <= n_in_group <= kMaxGridZ` | `n_in_group` becomes the launch grid's z-extent (the batch count); 0 is an invalid launch and anything above the driver limit cannot be launched. |
| `run_f2_gemms_group` | `s_pad >= 1` | `s_pad` is the contraction length of the multiplies; a length of 0 would be a scale-only operation, not the intended product. |
| `run_f2_gemms_group` | `n_in_group >= 1` | It is the batch count for the batched multiply; 0 is a degenerate empty batch. This function issues no kernel launch of its own, so it asserts the bound directly rather than routing through the launch-geometry guard. |
| `launch_assemble_blocks_group` | `1 <= n_in_group <= kMaxGridZ` | Same z-extent batch-count reasoning as the gather stage. There is no `s_pad` check here because this stage's slabs are `P × P`. |

On an actual CUDA or matrix-multiply-library failure at runtime, all three
functions throw a typed error (a CUDA error or a matrix-multiply-library error)
rather than returning a status code.
