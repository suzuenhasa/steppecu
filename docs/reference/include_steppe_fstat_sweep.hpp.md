# `fstat_sweep.hpp` reference

## 1. Purpose

`include/steppe/fstat_sweep.hpp` is the public entry point for the
all-combinations f-statistic *sweep* — the production-scale sibling of the
ordinary `run_f4` / `run_f3` calls. It is deliberately free of any CUDA code so
it can be included anywhere (the core library, the command-line tool, the
language bindings) without pulling in the GPU layer.

The difference between an ordinary f-statistic call and a sweep is what the
caller supplies:

- An ordinary `run_f4` / `run_f3` takes an explicit list of quartets (or
  triples) the caller wants scored.
- A **sweep** takes no such list. Instead it enumerates *every* combination
  over a population set — every group of 4 populations for an f4 sweep, every
  group of 3 for an f3 sweep — scores all of them, and returns only the ones
  that pass a significance filter.

For each combination the sweep computes the f4 (or f3) point estimate, its
diagonal block-jackknife standard error, and the resulting z-score, then keeps
a combination only if it clears a `|z|` threshold or ranks among the top-K by
`|z|`. The full table of every combination's result is **never** built in host
memory — at sweep scale that table would be many terabytes. The filtering and
compaction happen on the GPU, so only the small surviving set is ever copied
back to the host.

The header exposes: a filter-mode enum (`SweepFilter`), an input request struct
(`SweepRequest`), an output struct (`SweepResult`), and two entry-point
functions (`run_f4_sweep`, `run_f3_sweep`). Multi-GPU is parked; a sweep runs on
a single GPU.

---

## 2. The GPU-only pipeline

The whole point of the sweep design is that the host never enumerates
combinations, never filters, and never loops over individual items. If it did,
the host would become the bottleneck — enumerating and shuttling billions of
combinations across the CPU/GPU boundary is exactly the disaster this design
avoids. The host only drives an outer loop over chunks of the combination index
range and receives the survivors.

Within each chunk, five stages run on the device:

1. **Unrank.** A GPU kernel maps each thread to the specific combination it is
   responsible for. It converts a linear index (thread number) into the actual
   set of population indices using the combinatorial number system — the
   standard bijection between a position in the lexicographic ordering of all
   combinations and the combination itself. This writes the device-side
   combination list, so no host enumeration is needed.
2. **Score.** The existing batched f-statistic machinery runs verbatim on that
   device combination list: the gather that assembles each combination's slabs,
   the leave-one-out and total accumulations, the cross-term step, and the
   diagonal-jackknife standard-error kernels. These are the same device kernels
   the ordinary f4/f3 path uses. The GPU backend kernels own precision
   selection — the default emulated double-precision math for the matrix-heavy
   work, with a native double-precision carve-out for the cancellation-sensitive
   combine step. The host driver only forwards the default fit precision.
3. **Filter.** A GPU kernel flags which items in the chunk survive (see section
   3 for the two filter modes).
4. **Compact.** A device stream-compaction pass (CUB's flagged select) packs the
   flagged survivors together on the GPU, discarding the rest without ever
   moving them.
5. **Copy back.** Only the compacted survivors are copied from the GPU to the
   host.

Because only survivors cross the CUDA-free boundary, a sweep over billions of
combinations returns a small result no matter how many combinations were
scored.

---

## 3. The maxcomb cap and the two filters

### The maxcomb safety cap

A sweep can ask for an astronomical number of combinations (every group of 4
out of several hundred populations runs into the billions). To stop a sweep
from silently running for hours, a cap is checked **before any compute
starts**. If the number of combinations the sweep would enumerate exceeds the
cap (`kFstatMaxComb`, defined in `config.hpp`), the sweep refuses: it returns
immediately with no GPU work done, `capped` set true, and an `InvalidConfig`
status.

The critical subtlety: this cap guards **compute time**, not just memory. Every
single combination is scored on the GPU in order to test it against the filter
— the filter limits how much *output* survives, never how much *work* is done.
So a billion-combination sweep is hours of GPU work even if only a handful of
results ultimately survive. The cap is the guard against that. A caller who
genuinely wants an over-cap sweep can lift the cap by setting `sure` on the
request (the analogue of the reference `sure` flag[^at2]).

### The two filters

- **Min-z (`SweepFilter::MinZ`).** The default. Keep every combination whose
  `|z|` is at least `min_z`. The flagging happens entirely on the device, so the
  full table is never materialized — this is the mode that stays safe at
  multi-terabyte scale. The default threshold of `3.0` matches the
  parity significance cut[^at2].
- **Top-K (`SweepFilter::TopK`).** Keep the K combinations with the largest
  `|z|`. The device keeps all items through compaction and the host ranks the
  compacted set down to K, returning them sorted by `|z|` descending.

---

## 4. SweepFilter

`SweepFilter` selects which survivor rule the sweep applies on the device.

| Value | Meaning |
|---|---|
| `MinZ` | Keep items with `|z| >= min_z`. The flag is computed on the device, which is what makes it safe at any scale — the full result table never has to exist. This is the default. |
| `TopK` | Keep the K items with the largest `|z|`. The device keeps everything through compaction; the host then ranks the compacted set down to K. |

---

## 5. SweepRequest

`SweepRequest` is the input to a sweep. One request describes one sweep. The
arity — whether it enumerates groups of 4 or groups of 3 — is *not* a field on
this struct; it is fixed by which entry point you call (`run_f4_sweep` means
groups of 4, `run_f3_sweep` means groups of 3).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `filter` | `SweepFilter` | `MinZ` | Which survivor rule to apply (see section 4). |
| `min_z` | `double` | `3.0` | The `|z|` threshold for the `MinZ` filter. The default of 3.0 matches the parity significance cut. Ignored when the filter is `TopK`. |
| `top_k` | `std::size_t` | `100` | How many top items to keep when the filter is `TopK`. Ignored when the filter is `MinZ`. |
| `pop_subset` | `vector<int>` | empty | An optional subset of the population axis to restrict the sweep to. These are indices into the f2 population axis. Empty means sweep the entire population set. A non-empty list means enumerate combinations only over those indices — useful for narrowing an otherwise over-cap sweep to a set of interest. |
| `sure` | `bool` | `false` | Lift the maxcomb cap. Left false, a sweep that would exceed `kFstatMaxComb` refuses before doing any work. Set true to force it to run anyway. This is the analogue of the reference `sure` flag. |

---

## 6. SweepResult

`SweepResult` carries back **only the survivors**, plus enough bookkeeping to
explain what the sweep did. It contains no ordering guarantee beyond this: for
`MinZ`, survivors come back in the per-chunk lexicographic order they were
enumerated in; for `TopK`, they come back sorted by `|z|` descending.

The per-survivor arrays are all parallel — index `r` refers to the same
survivor across every one of them.

| Field | Type | Meaning |
|---|---|---|
| `keys` | `vector<array<int,4>>` | The population-axis indices of each survivor combination. For an f4 sweep all four entries are used (`{p1,p2,p3,p4}`); for an f3 sweep only the first three are meaningful and the fourth is unused (left at 0). |
| `est` | `vector<double>` | The f4 or f3 point estimate for each survivor. |
| `se` | `vector<double>` | The diagonal block-jackknife standard error for each survivor. |
| `z` | `vector<double>` | The z-score for each survivor — `est / se`. |
| `p` | `vector<double>` | The two-sided p-value for each survivor, computed from the normal distribution as `2 * pnorm_upper(|z|)`. |

The remaining fields describe the sweep as a whole:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enumerated` | `std::size_t` | `0` | The total number of combinations the sweep *would* enumerate over the chosen population set. This is reported even when the sweep is capped and never runs, so a caller can see how far over the cap the request was. |
| `survivors` | `std::size_t` | `0` | How many rows survived the filter. Always equal to `keys.size()`. |
| `capped` | `bool` | `false` | True if the sweep refused because `enumerated` exceeded the cap and `sure` was not set. When true, no compute ran and the per-survivor arrays are empty. |
| `status` | `Status` | `Ok` | The outcome status. A capped sweep reports `InvalidConfig`. |
| `precision_tag` | `Precision::Kind` | `Fp64` | Which precision mode the compute actually used, recorded for the results metadata. |

---

## 7. Entry points

Both entry points take device-resident f2 data, the sweep request, and the GPU
resources handle. Each is marked `[[nodiscard]]` because the returned
`SweepResult` is the whole point of the call — ignoring it is always a mistake.
Both route through the CUDA backend's sweep implementation on the first GPU;
multi-GPU is parked, so a sweep is single-GPU.

- **`run_f4_sweep(f2, req, resources)`** — the four-population sweep. Enumerates
  every group of 4 over the population set (or over `pop_subset` if given),
  scores each with f4 plus its diagonal-jackknife standard error and z-score,
  filters and compacts on the device, and returns only the survivors.
- **`run_f3_sweep(f2, req, resources)`** — the three-population sibling.
  Identical in shape, but enumerates every group of 3 and scores each with f3.
  In its results the fourth key slot is unused.

---

[^at2]: **ADMIXTOOLS 2** — the reference implementation steppe reproduces for numerical parity. Maier R, Flegontov P, Flegontova O, Changmai P, Vyazov LA, Kim AKM, Reich D. *On the limits of fitting complex models of population history to f-statistics.* eLife 2023;12:e85492. <https://elifesciences.org/articles/85492>
