# `f4_quartets.hpp` reference

## 1. Purpose

`src/core/qpadm/f4_quartets.hpp` is a small dispatch seam. It exposes one
operation — assemble a per-block f4 statistic for a batch of population quartets —
and hands the actual work off to a compute backend.

The header itself contains no arithmetic and no GPU code. It declares two short
inline functions (both named `assemble_f4_quartets`) that immediately forward their
arguments to a `ComputeBackend` method of the same name. All the real computation —
reading the f2 data, applying the f4 formula, and running the jackknife bookkeeping —
lives in the backend implementations (a CPU reference backend and a CUDA GPU
backend), not here.

Keeping this as its own pure, GPU-free header lets any part of the codebase call the
f4-quartets operation without pulling in CUDA. The header is a direct sibling of
`f4_matrix.hpp`, which does the same thing for the qpAdm model f4 matrix; both reuse
the identical pattern of a thin host-side entry point that dispatches through the
backend.

---

## 2. The two overloads

There are two `assemble_f4_quartets` functions. They differ only in where the input
f2 data lives; the math they trigger is identical.

| Overload | f2 source | Role |
|---|---|---|
| `assemble_f4_quartets(be, const device::DeviceF2Blocks& f2, quartets, precision)` | f2 already resident in GPU memory | The primary, GPU-first path. Because the f2 tensor is already on the device, the GPU backend computes the result with no copy of the f2 data back from GPU to host. This is the path production runs take. |
| `assemble_f4_quartets(be, const F2BlockTensor& f2, quartets, precision)` | f2 held in a host-memory tensor | The reference / parity door. The CPU backend implements this and is used as the oracle that GPU results are validated against. The CUDA backend deliberately does **not** implement this overload — calling it on the GPU backend throws, because the GPU path is meant to read device-resident f2, not a host tensor. |

Both overloads take the same three remaining arguments: the compute backend to
dispatch through (`ComputeBackend& be`), the flattened quartet index list
(`std::span<const int> quartets`, described in section 3), and a precision policy
(`const Precision& precision`, described in section 5). Both return an `F4Blocks`
result (described in section 4). Both are marked `[[nodiscard]]`, so the returned
result cannot be accidentally ignored.

---

## 3. The `quartets` index layout

`quartets` is a single flat array of population indices, not a list of tuples. The
caller lays it out so that the four indices of each quartet sit next to each other.

- Its length is `4 * N`, where `N` is the number of quartets.
- Quartet number `k` occupies positions `4*k` through `4*k + 3`.
- Those four values, in order, are `{p1, p2, p3, p4}` — the four population indices
  along the population axis of the f2 data.

So a caller with three quartets stores twelve integers: the first four are quartet 0,
the next four are quartet 1, and the last four are quartet 2. This flat layout is
what lets the GPU path pass the whole batch to the device as one array.

Each quartet is read as the f4 statistic between the pair `(p1, p2)` on the left and
the pair `(p3, p4)` on the right.

---

## 4. What the functions compute and where the work happens

The operation assembles one `F4Blocks` result whose batch axis (`m`) is the `N`
input quartets — one f4 value per quartet, per genome block. For quartet
`k = (p1, p2, p3, p4)` and block `b`, the value is the four-term f2 combination:

```
X[k, b] = 0.5 * ( f2(p2, p3, b) + f2(p1, p4, b) - f2(p1, p3, b) - f2(p2, p4, b) )
```

This is the standard four-slab f4 identity, matching ADMIXTOOLS 2. Each quartet in
the batch can name a completely different set of four populations (the batch is
heterogeneous), which is why the input is a flat list of explicit index quads rather
than a single left/right grid.

None of this arithmetic lives in this header. The two inline functions do nothing but
call `be.assemble_f4_quartets(...)`. The backend does the reading, the four-term
combine, and the jackknife bookkeeping. The header exists only to give callers a
clean, CUDA-free entry point and to route to the right backend method.

The returned `F4Blocks` carries, for this batch of `N` quartets:

- the per-block f4 values,
- the jackknife point estimate (one value per quartet),
- the leave-one-out replicate values used later to compute uncertainty, and
- the per-block SNP counts used as jackknife weights.

Its shape fields are set so that the number of rows is `N` and the number of columns
is `1`, giving a batch size of `N` (one entry per quartet). This is the same result
shape the model-f4 assembler produces, so the downstream jackknife-covariance step
consumes the batch without any special handling.

---

## 5. Precision handling

A `Precision` policy is passed through to the backend, but for this operation the
core four-term difference is always computed in native double precision regardless of
what the policy asks for. The subtraction of nearly-equal f2 terms is numerically
delicate (it is prone to catastrophic cancellation), so it is carved out to run in
full double precision to protect accuracy. The `precision` argument is accepted and
acknowledged for interface consistency with the other assembly seams, but it does not
downgrade this particular four-slab difference to a faster, lower-precision mode.
