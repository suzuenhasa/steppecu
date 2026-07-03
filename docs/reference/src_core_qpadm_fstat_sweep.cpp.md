# `fstat_sweep.cpp` reference

## 1. Purpose

`src/core/qpadm/fstat_sweep.cpp` is the host-side driver for the all-combinations
f-statistic sweep. It implements the two public entry points, `run_f4_sweep` and
`run_f3_sweep`, declared in `steppe/fstat_sweep.hpp`.

A sweep asks steppe to score *every* combination of populations of a given size —
every group of 4 for an f4 sweep, every group of 3 for an f3 sweep — filter them by
statistical significance, and hand back only the ones that pass. The number of
combinations can run into the billions, so the whole enumerate-score-filter-compact
pipeline happens on the GPU; the full result table is never built on the host.

This file is deliberately **host-pure and CUDA-free**. It contains no GPU code. Its
job is orchestration: work out how many combinations a request would enumerate,
refuse the request up front if that count is dangerously large, forward the request
to the GPU backend as a device-side config, and then do a small amount of
finishing work on the handful of survivors the device returns. All of the actual
heavy computation lives behind a backend virtual call (`f4_sweep` / `f3_sweep`).

Only a single GPU is used — the first one in the resource list. Splitting the
combination range across multiple GPUs is a deferred follow-on and is not done by
this driver.

---

## 2. What runs on the host versus the device

The central design idea is that almost nothing runs on the host. Understanding the
split is the key to reading this file.

**The device does all the heavy work.** A single backend call enumerates every
combination, computes its f-statistic, tests it against the significance filter,
and compacts the survivors — all on the GPU. It returns only the survivors, already
ranked, never the full table.

**The host does not:**

- **Enumerate.** The host never loops over the combinations. A device kernel turns
  a flat index into the population tuple it stands for (an "unrank" step).
- **Filter item by item.** The host never inspects individual combinations. The
  device sets a keep/drop flag per item and a device-side compaction copies out only
  the keepers.
- **Hold the full table.** Only the survivors are ever copied back off the device.
  A sweep over billions of combinations still returns at most a bounded, small set,
  so host memory can never blow up.

**The host does only three small things**, each proportional to the survivor count,
never to the total combination count:

1. Compute the combination count and enforce the safety cap (section 4), before any
   device work.
2. Translate the public request into the backend's device-side config (section 6).
3. Finish the small survivor set: compute each survivor's p-value and move the
   result arrays into place (section 7).

---

## 3. The overflow-safe combination count

Before any GPU work, the driver needs to know how many combinations a request would
enumerate. That count is a binomial coefficient — C(P, k), read "P choose k", the
number of ways to pick k populations out of P. Here k is 3 (f3) or 4 (f4), and P can
be large, so the count itself can overflow a 64-bit integer.

`choose_saturating(P, k)` computes this count as an unsigned 64-bit value that
**saturates** — it clamps to the maximum 64-bit value instead of wrapping around —
on overflow. Saturation is the correct behavior here because the count is only ever
used to compare against the safety cap (section 4): a count that would overflow is
certainly larger than the cap, so pinning it to the maximum value keeps the
comparison correct without needing the true (uncomputable-in-64-bits) figure.

Details worth knowing:

- It is **pure host arithmetic with no allocation**. k is tiny (3 or 4), so the loop
  runs at most four times.
- It uses the multiplicative form of the binomial coefficient, building the product
  one factor at a time in an order chosen so that every partial result stays a whole
  number (no fractional intermediate that would lose precision).
- Each multiplication is **overflow-guarded**: before multiplying, it checks whether
  the multiply would exceed the 64-bit maximum, and if so returns the saturated
  maximum immediately.
- A request for k below 0, or for more populations than are available (P below k),
  returns 0 — there are no such combinations.

---

## 4. The maxcomb cap and the early-return paths

The driver can return in three ways: two early exits before any GPU work, and the
normal dispatch path.

The sweep range is either the whole population set (P) or a caller-supplied subset;
its size is what feeds the combination count. Two early exits are checked against
that range:

**Too few populations.** If the range is smaller than k, not even one combination
can be formed. This returns a clean, empty, successful result — status `Ok`, no
survivors.

**The maxcomb cap.** If the combination count exceeds `kFstatMaxComb` (100 million;
defined in `config.hpp`) and the caller has not set the `sure` override, the driver
**refuses the sweep up front** — status `InvalidConfig`, with the `capped` flag set —
before allocating anything or touching the GPU.

The reason this cap exists is subtle and important: **the significance filter bounds
the output, not the work.** Every single combination has to be computed on the GPU
in order to test it against the filter, even though only a few will survive. So a
billion-combination sweep is hours of GPU compute regardless of how tiny the survivor
set turns out to be. The cap therefore guards *compute time*, not just memory. One
hundred million combinations is a few minutes of GPU work at the measured rate;
beyond that, the driver makes the caller opt in explicitly with `sure`.

### The reported combination count on early returns

On both early-exit paths, the count reported back to the caller is the host's own
clamped estimate (the saturating count from section 3, clamped down to fit the
platform size type). This estimate is written into the result **only** on those two
early-return branches.

On the normal dispatch path it is *not* used: the backend knows the exact size of
the range it swept and echoes that exact count back, which overwrites the field
after the sweep returns. The host estimate is deliberately not written on the
dispatch path, so it is never a dead value that the compute path immediately
replaces.

---

## 5. Precision policy forwarding

The sweep runs with steppe's shared fit-precision policy — emulated double precision
with a 40-bit mantissa, obtained from `default_fit_precision()`. This driver only
**forwards** that policy to the backend sweep kernels; it does not choose it.

The distinction matters: *precision selection* — using the fast emulated mode for the
matrix-multiply-heavy parts while carving out native double precision for the
cancellation-sensitive combine step — is owned entirely by the device sweep kernels,
not by this host driver. The driver's only precision responsibility is to record,
in the result's `precision_tag`, the mode the backend will actually honor. That
honored tag can differ from the requested mode when the backend was built without the
emulation capability compiled in, in which case it quietly falls back to native
double precision; `honored_tag` reflects that reality rather than the request.

---

## 6. Mapping the request to the backend config

When a sweep does proceed, the public `SweepRequest` is translated into the backend's
device-side `SweepConfig`. Most fields map across directly: the arity k, the filter
mode (minimum-z versus top-K), the minimum-z threshold, the population subset, and the
`sure` override.

The one field that carries real logic is `top_k`, which is the size of the **device
reservoir** — a fixed-capacity buffer the GPU maintains as it sweeps, keeping only the
most significant results seen so far (largest absolute z-score), with a threshold that
rises as the buffer fills. Because this reservoir is bounded and lives on the device,
the sweep returns at most that many rows, already sorted by absolute z-score
descending, with no unbounded host vector and no host-side re-ranking.

How `top_k` is set depends on the filter mode:

- **Top-K filter.** The reservoir cap is the caller's requested K — that is the user's
  actual intent.
- **Minimum-z filter.** The requested K is meaningless in this mode (the public struct
  leaves it at a small default of 100), so it is *not* used. Instead the reservoir cap
  becomes a hard **safety ceiling** of `kFstatDefaultSweepTopK` (one million). This
  ceiling is large enough that a minimum-z sweep returns *all* of its qualifying
  survivors in the normal case; only if more than a million combinations clear the
  threshold does it retain the million most extreme ones. The ceiling exists purely so
  that even a billions-item minimum-z sweep can never exhaust host memory.
- **Zero or unset K.** A top-K request with K of zero also falls back to the bounded
  default, so the reservoir is always finite.

---

## 7. Finishing the survivor set

The backend call — `f4_sweep` for k of 4, `f3_sweep` for k of 3 — runs the entire
on-device pipeline and returns a `SweepSurvivors` bundle. Two kinds of outcome are
distinguished:

- A **device fault** (a genuine GPU failure) propagates out as an exception, which the
  application maps to a fault exit.
- A **domain outcome** (for example, an invalid configuration detected on the device)
  comes back as a status on the returned struct, which the driver records and returns
  without throwing.

On success, the driver's finishing work is small and bounded by the survivor count:

- It copies the backend's **exact** enumerated count and the `capped` flag into the
  result (this is the exact count that supersedes the host estimate from section 4).
- It **moves** the survivor arrays — the population-tuple keys, the estimates, the
  standard errors, and the z-scores — into the result rather than copying them.
- It computes each survivor's two-sided **p-value** from its z-score, using the same
  z-to-p convention as the rest of steppe (`f4_two_sided_p`). This is a per-survivor
  loop over the small survivor set — proportional to the number of survivors (at most
  the reservoir cap), never to the total combination count.
- It records the final survivor count.

No re-ranking happens here: the device already maintained the ranking in its
reservoir and returned the rows in sorted order.

---

## 8. The two entry points

Both public functions are thin wrappers over one shared implementation, differing only
in the arity they pass:

- `run_f4_sweep` sweeps every group of 4 populations (k = 4).
- `run_f3_sweep` sweeps every group of 3 populations (k = 3).

Each takes the device-resident f2 data, the sweep request, and the GPU resources, and
returns a `SweepResult`. Everything else — the cap check, the precision forwarding, the
config mapping, the dispatch, and the survivor finishing — is identical between them and
lives in the shared driver.
