# `cuda_backend.cuh` reference

## 1. Purpose

`src/device/cuda/cuda_backend.cuh` declares `CudaBackend` — the class that runs
every steppe computation on an NVIDIA GPU. It is the GPU implementation of the
abstract `ComputeBackend` interface; the CPU implementation (`CpuBackend`) exists
only as a bit-exact reference oracle for tests, so `CudaBackend` is the real
product path.

The file is a *header* only: it declares the class shape (its nested types, its
methods, and its data members) but almost none of the method bodies. Those bodies
live in several separate `.cu` source files, one per subsystem (decode, f2, the
fit engine, and so on), which all compile into the same GPU library target. The
header was split out of one large source file so those subsystems could be
compiled apart while still presenting a single class to callers.

Two properties of the class are worth stating up front because the rest of the
file keeps relying on them:

- **One backend instance is bound to exactly one physical GPU.** The constructor
  takes a CUDA device number and pins the instance to that device for its whole
  life. A machine with two GPUs runs two `CudaBackend` objects. Every method
  re-selects its own device on entry so that interleaving calls to two backends
  from one process can never run work on the wrong GPU.
- **The class is move-only.** It owns GPU handles and memory buffers that cannot
  be copied, so (like its base class) it can be moved but not copied.

A recurring distinction in this file is between *host* memory (ordinary CPU RAM)
and *device* memory (GPU VRAM), and between the two directions of copying between
them: **H2D** is a host-to-device copy (CPU to GPU) and **D2H** is a device-to-host
copy (GPU to CPU). Much of the design effort captured here is about keeping large
results **device-resident** — left sitting in GPU memory — so the pipeline does not
pay to copy multi-gigabyte tensors back and forth. A value described as
**parity-neutral** changes only *where* bytes live or *how fast* something runs,
never a number steppe reports back.

---

## 2. How the class binds to one GPU

### The constructor

```
explicit CudaBackend(int device_id = 0);
```

`device_id` is the physical CUDA device number this instance owns. The default of
`0` keeps the ordinary single-GPU path (and every existing zero-argument call
site) bound to device 0.

The constructor has a subtle but load-bearing requirement: it must make its device
current *before* the cuBLAS handle and the workspace buffer are built. A cuBLAS
context is permanently tied to whichever CUDA device is current at the moment it is
created, and a GPU memory allocation lands on whichever device is current at the
moment it is allocated. So both must see `device_id` as the current device, not
whatever device happened to be current when the object was constructed.

C++ initializes data members in declaration order, and `device_id_` is declared
first. The class exploits this: while initializing `device_id_`, it runs a small
helper (`set_and_return_device`, section 18) that makes the device current as a
side effect. Because that happens before the cuBLAS handle and the workspace are
constructed later in the member list, both of them bind to the correct device.

### Selecting the device on every call

The wrapper types around cuBLAS and cuSOLVER never call `cudaSetDevice`
themselves — they only *record and assert* which device they were built on. The
backend is the one owner that legitimately *selects* the device. It does so on
every compute entry through `guard_device()` (section 18), so that a later change
of the ambient current device elsewhere in the process cannot cause this backend's
work to run on the wrong GPU.

---

## 3. Nested types

The class declares three small helper structs.

### `ResidentBlocks` (public)

The output of the shared f2 matrix-multiply body (`run_f2_blocks_resident`,
section 4). It carries the two large per-block tensors left sitting in GPU memory,
plus the host-side bookkeeping, and is returned by move so that no copy of the
tensors is made.

| Field | Type | Meaning |
|---|---|---|
| `f2` | `DeviceBuffer<double>` | The `P × P × n_block` f2 tensor, resident in this backend's GPU memory (or empty). |
| `vpair` | `DeviceBuffer<double>` | The paired-variance tensor of the same shape, resident (or empty). |
| `block_sizes` | `vector<int>` | The SNP count of each jackknife block, on the host. |
| `P` | `int` | Number of populations. |
| `n_block` | `int` | Number of jackknife blocks. |

It is declared before the methods that return it, because a member function's
return type must be a complete type at the point of its declaration.

### `SvdScratchSizes` (public)

The exact scratch sizes one large-path singular-value-decomposition (SVD) call
needs, so the caller can allocate one scratch arena once and reuse it across many
same-shape SVDs instead of allocating and freeing per call (each free is a
device-wide synchronization that would serialize the otherwise-asynchronous
stream). All sizes are element *counts*, not bytes, and are sized to whichever
orientation and routine the fixed dimensions select — only one branch ever runs, so
each field is exactly its branch's need with no over-allocation. See section 12 for
how they are used.

| Field | Meaning |
|---|---|
| `s` | Singular values: `min(nl, nr)`. |
| `u` | Economy left-vectors `U`: `nl * nr`. |
| `vt` | Right-vectors: `nl*nl` or `nr*nr`; read only by the Jacobi branch when `nr >= nl`. |
| `a2` | A non-const copy of the input matrix: `nl*nr` when `nl > nr` (the routines overwrite their input, and the caller's input must survive), otherwise `0`. |
| `info` | The routine's status integer: always `1`. |
| `lwork` | The cuSOLVER working-buffer size for the chosen routine. It depends only on the dimensions, never on the matrix values, so it is queried once per shape and reused. |

### `AssembleFlags` (private)

Two booleans passed to the per-model result assembler (section 13), named as a
struct so they cannot silently swap at the call site.

| Field | Meaning |
|---|---|
| `nonspd` | The covariance Cholesky failed, so this model's status is "covariance not positive-definite." |
| `se_computed` | A standard error was computed for this model, so its standard-error / z-score fields should be filled; otherwise they are left empty. |

---

## 4. The f2 block computation family

f2 is the foundational per-block statistic. For every pair of populations and every
jackknife block, steppe computes an f2 value and its paired variance, producing two
`P × P × n_block` tensors. Everything downstream (f3, f4, D, qpAdm, qpGraph) is
built from these.

### The shared matrix-multiply body

`run_f2_blocks_resident` is the single engine all the public entry points share. It
runs the f2 computation using a size-grouped, batched design: one fused pass
decodes and feeds all SNPs, then blocks are bucketed by size (rounded up to a power
of two), and each bucket runs as one batched set of three matrix multiplies
followed by a fused assemble step that scatters results into the resident f2 and
paired-variance tensors. Only one bucket's padded working slabs are in memory at a
time, which keeps GPU memory use modest. It returns the two tensors by move in a
`ResidentBlocks`, with no D2H copy and without freeing the buffers. Every public
entry point below produces **bit-identical per-block values** because they all call
this same body; they differ only in what they do with the result afterward.

### The entry points

| Method | What it does with the result |
|---|---|
| `compute_f2` | The simplest whole-matrix f2 (not per-block). |
| `compute_f2_blocks` | Runs the shared body, then materializes a host tensor via one D2H copy. A thin convenience wrapper; the hot fit path does not call it. |
| `compute_f2_blocks_device` | Runs the shared body and moves the resident tensors into a `DeviceF2Blocks` handle. No copy back — the full result escapes into a handle that stays in GPU memory. This is the production producer for the fit engine. |
| `compute_f2_blocks_resident` | Like the above but wraps the result in a `DevicePartial` (a per-GPU partial result for the multi-GPU combine). The buffers survive the worker thread's exit and are freed only after the combine has consumed them. |
| `compute_f2_blocks_into` | Runs the shared body, then D2H copies the compact result slabs into persistent per-backend pinned staging buffers, and a plain host copy places the bytes at the caller's block offset in a shared result. |
| `compute_f2_blocks_streamed` | The out-of-core path: spills each block to host RAM or to disk instead of holding the whole tensor in GPU memory. |

The `compute_f2_blocks_into` path exists to make two GPUs' copies-back run
concurrently. An earlier design pinned the caller's roughly-3-GB result slice on
every call, and the pin/unpin operations take a device-wide driver lock that
serialized the two workers' copies (a measured ~570 ms serial tail on the budget
box). The persistent per-backend pinned staging buffers (allocated once, reused)
let the two GPUs copy into their own buffers as concurrent DMAs, and the
staging-to-result host copy runs at CPU bandwidth with no driver lock.

### Streaming with SNP tiling

`stream_f2_blocks_impl` is the out-of-core mechanism. It reuses the shared body's
per-block gather / matrix-multiply / assemble steps verbatim, so the per-block
values are bit-identical to the resident path, but differs in two ways:

1. **SNP-tiled input.** Instead of decoding all SNP columns at once (which
   overflowed GPU memory on full-genome runs at large population counts), each
   chunk uploads and decodes only its own SNP-column tile from the host copy of the
   data. The GPU footprint becomes independent of the total number of SNPs; the full
   data stays in host RAM, owned by the caller.
2. **A small device ring buffer.** Instead of the full per-block tensors, it cycles
   a small ring of per-chunk buffers, computing each chunk into the next ring slot
   and spilling that chunk's blocks through a sink that triple-buffers the
   copy-back-and-write.

The resident path never routes here; it calls `compute_f2_blocks_device` directly.
`BlockSink` (the spill target) is forward-declared at the top of the header because
it is named only as a reference parameter.

---

## 5. Genotype decode and the format-reader front-end

Before any statistic can run, the packed 2-bit genotype data must be decoded into
the per-population arrays the math consumes. These methods handle decoding and the
on-device format conversion.

### The shared decode front-end

`decode_af_resident` uploads a packed tile (and, when supplied, per-sample ploidy)
and decodes it into caller-owned resident arrays. It does no copy-back; the caller
decides whether the result crosses to the host or stays in GPU memory. It supports
**SNP tiling**: `s_lo` is the starting SNP index into each individual's full packed
record, and the upload slices out a window of SNPs into a compact device buffer.
`s_lo` must be a multiple of 4 because of the 2-bit packing alignment; untiled
callers pass the default `0` and decode exactly the codes they always did.

| Method | What it decodes to |
|---|---|
| `decode_af` | Decodes and copies the full result to the host (the reference / oracle path). |
| `decode_af_resident` | Decodes into caller-owned resident arrays; no copy-back. |
| `decode_af_compact_autosome` | Decodes resident, then applies an on-device autosomes-only keep-mask and stream-compacts the kept columns, all in GPU memory. Only the small kept chromosome/position arrays cross to the host. |
| `decode_af_compact_filter` | The sibling that applies the full allele-frequency / coverage filter instead of the autosome-only mask, and additionally compacts the count array. Also SNP-tileable so peak memory scales with the tile width, not the whole genome. |

The two compacting decodes use CUB stream-compaction primitives (a flagged-select
plus an exclusive-scan and a keyed gather) so the survivors keep their original file
order, which is what the downstream block assignment depends on.

### Ploidy detection and transpose

- `detect_sample_ploidy_device` runs, on the GPU, the per-sample ploidy pre-pass
  that ADMIXTOOLS 2 does — one thread per individual over the same bytes the decode
  reads. It is a direct port of the host detector and is bit-identical by
  construction because it is all integer and bit operations.
- `transpose_to_canonical` performs, on the GPU, the transpose-plus-gather-plus-repack
  from a SNP-major source layout into steppe's canonical individual-major byte
  layout. One thread per output byte gathers its source row, encodes it, and repacks
  it. It is bit-exact to the host reference by construction. This is the engine
  behind reading the various on-disk genotype formats.

---

## 6. D-statistic block reduction

The D-statistic (and its close relatives) can be computed directly from decoded
genotypes rather than from an f2 cache. These methods run the normalized per-SNP,
per-block reduction on the GPU.

`dstat_block_reduce_device` is the core. It takes already-resident genotype arrays
(borrowed device pointers it does not free), a table of population quadruples, and
the per-block layout, runs one SNP-tiled kernel with one thread per
(quadruple, block) cell, and copies back only the tiny per-cell numerator,
denominator, and count sums.

Two public overloads share that core:

- `dstat_block_reduce(Q, V, ...)` — the host-pointer entry: it uploads the genotype
  arrays first, then runs the core. This owns the buffers it uploads.
- `dstat_block_reduce(DeviceDecodeResult, ...)` — reads genotype arrays that are
  *already* resident from a compacting decode, skipping the upload entirely. The
  math is byte-identical to the host-pointer overload.

---

## 7. Fused ratio-jackknife producers

These fuse a statistic's block reduction directly into its ratio jackknife, keeping
the intermediate blocks resident and dropping a copy-back that the two-step path
would pay. Only the small per-result estimate / standard-error / z-score (and,
where relevant, p-value) cross back.

| Method | Fuses | Notes |
|---|---|---|
| `f4ratio_blocks_jackknife(DeviceF2Blocks, ...)` | f4-ratio assemble into the ratio jackknife | Keeps the assembled f4 blocks and their leave-one-out variants resident and feeds them straight to the jackknife. |
| `dstat_blocks_jackknife(DeviceDecodeResult, ...)` | the D reduction into the ratio jackknife | Keeps the numerator / denominator / count resident; computes the two-sided p-value on the device. |

Each also has a **host-oracle overload** — `f4ratio_blocks_jackknife(F2BlockTensor, ...)`
and `dstat_blocks_jackknife(Q, V, ...)` — which is *not* the GPU path. Those forms
exist only as the door the CPU reference oracle uses; the GPU class implements the
resident forms.

---

## 8. f-statistic sweeps and survivor blocks

A sweep enumerates *every* population combination — every group of 4 (f4) or every
group of 3 (f3) out of P populations — and returns only the statistically
significant survivors. Because the number of combinations can reach billions, the
host must never enumerate them itself.

### The shared sweep core

`run_fstat_sweep_device` is shared by the f4 and f3 sweeps (parameterized by `k`,
the group size). The host drives only the outer chunk loop; every per-item step
runs on the device. For each chunk it:

1. **Unranks** — a kernel maps each thread to its own combination using the
   combinatorial number system and writes the device index list directly, replacing
   any host enumeration or upload.
2. **Gathers and computes** the statistic, its leave-one-out and total terms, and
   its diagonal jackknife variance, using the same kernels the explicit path uses.
3. **Filters** by absolute z-score, writing a survivor flag per item.
4. **Compacts** the survivors on the device with a CUB flagged-select, so no
   per-item filtering happens on the host.
5. **Copies back** only the small compacted survivor set.

Chunk size is derived from free GPU memory. This is single-GPU (multi-GPU work is
parked). The safety cap on the total combination count is enforced in the layer
above and re-checked here.

### The two public sweeps

- `f4_sweep` — every group of 4; delegates to the core with `k = 4`.
- `f3_sweep` — every group of 3; delegates with `k = 3`.

### Survivor blocks

`device_survivor_blocks` returns, for a resident f2, the ascending list of block
ids that survive missing-data removal (matching ADMIXTOOLS 2's read-with-remove-NA
behavior). It runs an on-device keep kernel over the paired-variance tensor and
returns the indices that pass. If the handle carries no paired-variance tensor (an
upload path that guarantees no missing blocks), every block survives and no kernel
runs. The result depends only on the loaded f2, not on any model, so callers
compute it once per assembly.

---

## 9. qpfstats smoothing

qpfstats jointly smooths all the f2 estimates by solving a batched least-squares
problem. The design deliberately has no per-block host loop (which is the
CPU-bound trap in the reference implementation); it is expressed as a few big GPU
operations that share one factorization.

`qpfstats_smooth` performs the smoothing solve directly: it forms one shared normal
matrix with a syrk (matrix-multiply-heavy, so it uses the emulated-double policy),
one right-hand-side matrix with a gemm, one Cholesky factorization (native double
precision), and then solves all columns with a pair of triangular solves.
Not-a-number (NaN) entries are handled by zeroing them: an all-NaN block's
right-hand-side column becomes zero, so the shared solve yields a zero result,
exactly matching the reference's all-NaN policy. The rare partial-NaN block is
re-solved on the host with a downdated matrix — but only for the few blocks that
are partial-NaN, which is zero host solves in the ordinary production case.

`qpfstats_blocks_smooth` is the fused end-to-end performance path: it runs the
genotype reduction, the per-combination block jackknife, the shared-factor smoothing
solve, and the per-pair recentering jackknife all in one GPU residency, reading each
stage's output straight from GPU memory. It eliminates the large intermediate
copy-back; only the small final coefficients cross back. The matrix-multiply
sub-steps honor the requested precision; the jackknives and the Cholesky/solve run
in native double precision.

It comes in the usual pair of overloads plus a shared core:

- `qpfstats_blocks_smooth(Q, V, ...)` — the host-pointer entry (uploads first).
- `qpfstats_blocks_smooth(DeviceDecodeResult, ...)` — skips the upload by reading
  genotypes already resident from a compacting decode.
- `qpfstats_blocks_smooth_device(...)` — the private shared core over borrowed
  resident device pointers that both overloads call.

---

## 10. The DATES weighted-LD engine

DATES estimates admixture dates from the decay of weighted linkage-disequilibrium
correlation with genetic distance. The whole per-sample pipeline runs on the GPU and
is flat in the number of SNPs because it never forms the ~10^12 SNP-pair object;
instead it computes the correlation at each lag as an inverse FFT of a power
spectrum (the standard FFT trick), using cuFFT.

| Method | What it does |
|---|---|
| `dates_curve` | The whole weighted-LD curve. Device-resident inputs; the host drives only the short per-sample loop and the one-time plan setup. Per sample it scatters onto a per-chromosome grid, runs a batched forward FFT over all chromosomes, forms the power / cross-power, runs a batched inverse FFT, and accumulates the lag moments, then re-bins them into per-(chromosome, bin) sufficient statistics. Native double-precision FFT, matched exactly to the reference by dividing out the unnormalized FFT length. |
| `dates_repack` | Repacks the target genotypes onto the kept SNP axis on the device (one thread per destination byte, race-free), replacing a host bit-shuffle hot loop. Integer- and bit-exact to the reference. |
| `dates_fit` | The exponential-decay fit, batched over the windowed correlation curves — one thread per curve runs the exact coarse-to-fine one-dimensional search (a grid pass, a ternary-refine pass, and an inner small normal-equation solve). The inner accumulators are native double precision (the device has no long double); the loose date tolerance is comfortably met. |

---

## 11. The qpAdm fit engine and its frozen contract

The qpAdm fit is the marquee second-phase computation. Its GPU implementation is
governed by a contract that is deliberately frozen so it keeps reproducing the
bit-exact double-precision reference results:

- The f2 tensor stays **resident** in GPU memory; the gather kernels read it in
  place, with no copy-back of the big tensor.
- Every step runs on the device — the gather, leave-one-out and centering kernels,
  the covariance matrix-multiply, the Cholesky inverse, the small on-device linear
  algebra for the SVD seed / alternating-least-squares / weight / chi-squared steps,
  and the batched leave-one-block-out re-fits.
- The only host transfers are the small fit intermediates (kilobyte-scale) that
  cross the boundary to the host-vector interfaces.
- It runs in **native double precision end-to-end**, which is the parity gate, so
  the GPU result matches the reference golden.

The single-model fit chain (the stages are conventionally numbered) is:

### Assembling the per-block f4 matrix

- `assemble_f4` builds the per-block f4 matrix from a resident f2 handle. The gather
  kernel reads the tensor in GPU memory; the leave-one-out / total / centering
  reduction runs on the device; only the small result matrix crosses the boundary.
- `assemble_f4_quartets` builds one f4 matrix whose rows are a caller-supplied list
  of population quartets (the standalone-f4 path). Each quartet column gathers its
  own four-slab combine, and the survivor-block drop reuses `device_survivor_blocks`.
- `assemble_f3_triples` is the three-slab sibling for f3 triples.

Each of these has a **host-oracle overload** taking a host `F2BlockTensor` that is
*not* the GPU path — it exists only as the reference oracle's door, because forcing
a host tensor onto the device only to read it back would defeat the resident design.

### Covariance

- `jackknife_cov` computes the weighted block-jackknife covariance matrix and its
  inverse: a centering kernel (native precision), a syrk for the covariance matrix
  (matrix-multiply-heavy, so it honors the requested precision with an automatic
  native fallback), a small diagonal nudge, and a Cholesky factor-and-invert (native
  precision). If the matrix is not positive-definite, that is returned as a value,
  not thrown.
- `jackknife_diag` computes only the diagonal variance, which is the shape needed
  for per-item standard errors in a sweep. It reuses the exact centering step of
  `jackknife_cov` and then reduces to the diagonal, avoiding the dense matrix, the
  syrk, and the Cholesky. This keeps the work and memory linear (no quadratic
  blow-up at sweep scale) while producing the identical arithmetic the dense
  diagonal would have, so it re-passes the existing goldens by construction.

### Rank test, weights, and standard error

- `rank_sweep` runs the qpWave / qpAdm rank test over ranks 0 through the maximum,
  mirroring the reference oracle operation-for-operation but running each per-rank
  fit on the device. It forms the nested rank-drop table, the smallest non-rejected
  rank, and (for observability only) the numerical rank of the covariance. It
  dispatches by model size: a model inside the small-model envelope runs the
  on-device small-linear-algebra path unchanged; a larger model runs the cuSOLVER
  large path (section 12).
- `gls_weights` computes the generalized-least-squares mixture weights via the
  reference's alternating-least-squares scheme: an on-device SVD seed, a fixed
  number of ALS iterations, a constrained weight solve, a normalization so the
  weights sum to one, and a chi-squared. All on the device, native precision.
- `gls_weights_loo_batched` runs all the leave-one-block-out weight re-fits in one
  batched device launch (not a host loop over blocks), reusing the covariance inverse
  unchanged, and returns the per-block weight matrix.
- `se_from_wmat` reduces those per-block weights to the standard error entirely on
  the device. Because standard error is linear in the weight scale, the reference's
  scale factor is reintroduced as a final multiply on the short output vector, so the
  variance reduction itself never leaves the device and no per-block weight matrix is
  copied back. The private `populate_loo_wmat_resident` is the producer that fills the
  resident per-block weight matrix that both this and the batched re-fit consume.

### Capability queries and instrumentation

- `provides_rank_sweep` and `provides_batched_fit` both return true, telling the host
  orchestrator that this backend genuinely implements the rank sweep and the batched
  fit on the GPU (replacing an older detect-by-exception scheme).
- `batched_dispatch_count` reports how many batched chunk-dispatches were issued, so
  a test can prove the model rotation actually ran batched rather than as a per-model
  loop.

---

## 12. Large-model path helpers (cuSOLVER SVD)

The small-model fit path is bit-parity-frozen and uses an on-device Jacobi SVD.
Models that exceed that small envelope (too many left or right populations, or too
high a rank) route instead to a large path built on cuSOLVER's dense SVD with
dynamically-sized scratch. This path runs in native double precision because the SVD
and the covariance quadratic form are ill-conditioned. The small path is left
untouched.

The dispatch decisions each come from a single shared predicate so the executed
routine and the observability report can never disagree:

- `model_fits_small_path(nl, nr, r)` — true iff the model fits the small-parity
  envelope. It delegates to the one shared source that also sizes the fit kernels'
  per-thread arrays, so this gate cannot drift wider than the arrays it routes into.
- `gesvdj_applicable(nl, nr)` — true iff both dimensions are at most 32, in which
  case the large path uses cuSOLVER's one-sided Jacobi SVD; otherwise it uses the
  QR-based SVD. 32 is the size limit of the batched Jacobi routine.

The scratch is sized and reused rather than reallocated per call (each device buffer
free is a device-wide synchronization that would serialize the stream):

- `large_svd_scratch_sizes(nl, nr)` returns a `SvdScratchSizes` (section 3). The
  caller allocates one buffer per field once and reuses it across many same-shape
  SVDs.
- `large_svd_V(...)` computes the leading right singular vectors of a matrix via
  cuSOLVER, native precision, deterministically. It orients the input so the taller
  dimension leads (cuSOLVER's dense SVD is deterministic only when rows are at least
  columns) and reads the result out of whichever factor corresponds. It comes in two
  overloads: a **scratch-taking** form where all the working buffers are caller-owned
  slices of reusable arenas (so a per-block sweep pays no per-call frees), and a
  **single-shot convenience** form that grows persistent per-backend scratch members
  and delegates to the first. The single-shot form is for one-off large fits; the
  inner block sweep hoists its own scratch and uses the scratch-taking form.
- `large_dbl_scratch`, `large_int_scratch`, and `large_loo_dbl_refit` size the
  double and integer scratch for the large-path alternating-least-squares, weight,
  and chi-squared kernels, and the per-refit stride for the parallel large
  leave-one-out kernel (each thread owns a self-contained slice so there is no
  cross-thread aliasing and the result is deterministic).
- `large_fit_one` runs a single large-model fit end-to-end on the device (SVD seed,
  ALS, constrained weight solve, chi-squared), mirroring the small-path chain with
  caller-supplied buffers.

---

## 13. The batched model-space rotation

`fit_models_batched` is the batched multi-model path. A qpAdm run often fits a whole
"rotation" of many models that share the same shape (a small number of left and
right populations and a low rank). This method fits an entire bucket of same-shape
small-path models in one batched dispatch rather than looping per model:

- the f4 gather / leave-one-out / centering runs over a (block, model) grid, reading
  the resident f2 with per-model index arenas (no copy-back of the tensor);
- the covariance for all models is formed with one strided-batched matrix-multiply
  (honoring the requested precision, with automatic native fallback);
- the covariance inverse for all models is done with batched cuSOLVER Cholesky;
  per-model failures are recorded as a not-positive-definite status rather than
  thrown;
- the rank sweep, weight solve, chi-squared, leave-one-out standard error, and
  population-drop all run in one model-batched kernel with one thread per model.

The host then assembles the final per-model result structures (tail p-values, the
nested rank-drop table, the population-drop pattern strings and feasibility) using
the same math the single-model path does. Only small-model-envelope models are
passed here; the larger tail is routed to the per-model default path.

Three private helpers support it:

- `fit_one_bucket` handles one same-shape bucket, sub-chunking it so each chunk's
  working arena fits in free GPU memory (the chunking only bites at very large model
  counts); each sub-chunk is still one batched dispatch.
- `fit_chunk` is the one genuine batched dispatch for a chunk of same-shape models,
  compacting the gather onto the survivor blocks when a drop set is present.
- `assemble_result` turns one model's batched outputs into a result structure, using
  `AssembleFlags` (section 3) to carry the two per-model outcome booleans safely.

---

## 14. qpGraph optimizer fleet

qpGraph fits admixture-graph topologies. The GPU path runs a whole "fleet" of
optimization restarts in one launch so the host never runs a per-restart or
per-candidate objective loop (the CPU trap in the reference optimizer).

- `qpgraph_fit_fleet` fits one topology. The resident f3 basis and the topology
  arenas upload to GPU memory, and a fleet kernel runs many restarts — one thread
  each, the whole multistart projected-Newton loop on the device — with only the
  per-restart result coming back. The best-of-restarts selection, the confidence
  bracket, and the final edge-length recovery are tiny host steps. The inner
  positive-definite edge solve and the generalized-least-squares form are native
  precision.
- `qpgraph_fit_fleet_batch` fits *every* candidate topology in a search in one
  launch. It packs all topologies' path-table arenas into one device buffer with a
  per-topology index table, sizes the per-thread scratch to the batch maximum, and
  fits them all against the same resident basis (which is bound to the population
  set, not the topology, so it is uploaded once). The host reduces the per-topology
  scores to a best and a spread — a reduction, not a per-candidate host fit.

---

## 15. Capability probing

`capabilities()` reports the capability tier of this backend's device (its
compute capability, memory, and whether direct GPU-to-GPU peer access is available).
Every field is parity-neutral — it is pure observability and data-movement
enablement, never on the statistics path, so it never changes a reported number.

Two properties matter:

- **It degrades without throwing on the peer-access probe.** A "no" answer to the
  peer-access query is *expected* on the budget consumer GPUs (where peer access is
  driver-disabled) and for a device asked about itself, so that answer routes through
  a non-throwing warning rather than an error. Genuine faults — failing to read the
  device count, properties, or memory info of the bound device — still throw, because
  a backend that cannot read its own device is a real error.
- **It is `const` and device-neutral.** It makes its device current only for the
  duration of the queries and restores the previously-current device, so calling it
  leaves no lingering device-selection side effect. That lets a resource manager
  probe each per-device backend in turn from any ambient context.

---

## 16. The two precision axes

steppe distinguishes two independent precision choices, and this class keeps them
separate.

1. **The matmul precision.** The `Precision` argument that flows into nearly every
   compute method governs the matrix-multiply-heavy stages (the f2 multiplies and the
   covariance syrk). Its default is emulated double precision, which is much faster
   than native double precision at essentially the same accuracy.
2. **The solve precision.** `set_solve_precision` records a *separate* request that
   governs only the cuSOLVER dense-solve steps (currently the covariance-inverse
   Cholesky). Its default is native double precision, which is the parity gate for
   the fit goldens. A caller may promote it to emulated double precision to route the
   ill-conditioned inverse through the emulated tensor-core path, validated stage by
   stage against the native oracle. This does not touch the matmul precision axis.

Setting the solve precision affects only where the solve math runs, never the matmul
stages, and vice versa.

---

## 17. Private data members and teardown order

The class holds seventeen private data members whose **declaration order is
load-bearing**. C++ destroys members in reverse declaration order, and several of
these have construction and teardown dependencies that the order encodes. Do not
reorder them.

The order and its reasons:

1. `device_id_` (`int`, default `0`) — the physical device this instance is bound
   to. Declared **first** so that its initializer makes the device current before
   any handle or buffer below is built, and so it is available for `guard_device()`
   on every call. A plain integer with no teardown concern.
2. `stream_` (`Stream`) — the one statistics stream per device, for run-to-run
   bit-stability (every launch and every asynchronous copy routes through it).
   Declared **before** the cuBLAS and cuSOLVER handles so it is destroyed **after**
   them: their contexts are bound to this stream, so the stream must outlive them.
3. `blas_` (`CublasHandle`) — the one cuBLAS handle, created once and reused. Because
   `device_id_`'s initializer already made the device current, its cuBLAS context
   binds to the correct device.
4. `solver_` (`CusolverDnHandle`) — the dense cuSOLVER handle for the fit's small
   linear algebra. Declared after `blas_` so it too constructs on the right device,
   and destroyed before `stream_` (the stream it uses must outlive it). It shares the
   one statistics stream; there is no second stream.
5. `workspace_` (`DeviceBuffer<std::byte>`, sized to the cuBLAS workspace bytes
   constant) — the fixed cuBLAS workspace. Declared after `blas_` so it frees first;
   `blas_` holds a non-owning pointer into it.
6. `solve_precision_` (`Precision`, default native double) — the solve-precision
   knob set by `set_solve_precision` (section 16).
7. `batched_dispatch_count_` (`size_t`, default `0`) — the observability counter for
   the batched rotation. Off the numeric path.
8. `tot_line_` (`vector<double>`) — a host cache of the centering line produced by
   `assemble_f4` and consumed by `jackknife_cov`. One model is fit at a time on a
   backend instance, so it is rebuilt per assemble.
9. `pinned_in_` (`PinnedRegistryCache`) — the amortized host-input pin registry.
   It page-locks the genotype H2D source pages once per (pointer, size) and reuses
   them across the many f2 calls a run issues, which is the precondition for two
   GPUs' uploads to run as concurrent DMAs. Declared **last** among the resource
   members (destroyed first) because it only unregisters host pages and has no
   dependency on the handles.
10. `stage_f2_`, `stage_vpair_` (`PinnedBuffer<double>`) — the persistent pinned
    copy-back staging buffers, allocated once and reused so the page-locking cost is
    paid once rather than per call (see section 4). Also declared late (freed early),
    since freeing pinned host memory has no dependency on the handles.
11. `solver_work_`, `svd_s_`, `svd_u_`, `svd_vt_`, `svd_a2_`, `svd_info_`
    (`DeviceBuffer`s) — the pooled per-backend cuSOLVER scratch (the Cholesky working
    buffer and the single-shot SVD output/scratch set). These grow monotonically to
    the largest size seen and never shrink, dropping the cuSOLVER allocate/free count
    to near-zero after warmup while leaving the bytes each routine sees unchanged
    (so results are bit-identical). They are safe without a lock because each device
    owns its own backend and uses one stream, so the scratch is used strictly
    sequentially on one thread. They are a distinct set from the f2 copy-back staging,
    so the two never alias. The inner large SVD sweep hoists its own scratch and does
    not touch these.

---

## 18. The two inline device-guard helpers

Two tiny functions are defined inline in the header (not out-of-line) because they
are on the hot path or needed during construction.

- `guard_device()` (private, `const`) makes this backend's device current at every
  compute entry. Because one backend is bound to one device and a single process may
  hold one backend per device and interleave their calls, each entry re-selects its
  own device rather than trusting the ambient current device. On the single-GPU path
  this re-selects device 0, which the runtime short-circuits as a no-op. It is
  parity-neutral: selecting a device moves no bits of the arithmetic.
- `set_and_return_device(device_id)` (private, `static`) makes the given device
  current and returns it. It is the member-initializer-list hook that selects the
  device before the cuBLAS and cuSOLVER handles construct (see section 2). It is
  `static` so it can run while the first member is still being initialized, when no
  `this` and no other member is available yet. It throws on an invalid device number
  — a backend cannot be bound to a device that does not exist.
