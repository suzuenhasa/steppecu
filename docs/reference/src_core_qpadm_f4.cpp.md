# `f4.cpp` reference

## 1. Purpose

`src/core/qpadm/f4.cpp` implements `run_f4`, the standalone f4-statistic entry
point. Given a batch of population *quartets* — groups of four populations
`(p1, p2, p3, p4)` — it computes one f4 value for each quartet along with a
standard error, a z-score, and a two-sided p-value.

The important design fact is that f4 is a *sibling* of the qpWave routine, not a
cut-down copy of qpAdm. It reuses the exact same two building blocks those
routines use and adds no new math of its own:

1. **`assemble_f4_quartets`** — the routine that turns f2 blocks into the
   per-quartet f4 point estimate (the same four-term identity ADMIXTOOLS 2 uses).
2. The **block jackknife** — the routine that estimates uncertainty by repeatedly
   dropping one genome block at a time.

What f4 deliberately does *not* do is just as important. There is no iterative
model fit and no rank test. Those belong to qpAdm and qpWave. A bare f4 is simply
the jackknife point estimate for each quartet plus the jackknife standard error —
nothing more.

---

## 2. The processing pipeline

Every call, whichever entry point is used, runs the same shared body. The steps
are:

1. **Flatten the quartets.** The incoming quartets arrive as an array of
   four-element groups. Each group's four population indices are copied out onto
   the result table (so the emitted rows can be labelled `p1..p4`) and also
   flattened into one contiguous `4 × N` integer array, which is the shape the
   assemble routine consumes.
2. **Assemble once.** A single call to `assemble_f4_quartets` produces one batched
   result whose batch axis *is* the list of quartets. In the internal layout the
   left count is `N` and the right count is `1`, so the batch length is exactly
   `N` — one f4 per quartet. This is one call for the whole batch, not one call
   per quartet.
3. **Jackknife once.** A single block-jackknife pass over the whole batch produces
   the per-quartet variance (see section 3 for why only the diagonal is formed).
4. **Read out per quartet.** For each quartet `k`, the point estimate is taken
   straight from the assembled total, the standard error is the square root of the
   jackknife variance, the z-score is estimate ÷ standard error, and the p-value
   is the two-sided normal tail of the z-score (section 6).

Because there is exactly one assemble and exactly one jackknife for the whole
batch, the cost is amortised across all quartets rather than paid per quartet.

---

## 3. Why the standard error uses only the jackknife diagonal

A full block-jackknife normally produces a dense variance-covariance matrix that
is `m × m`, where `m` is the number of estimates in the batch. For f4, `m` is the
number of quartets, and in a large sweep that can be tens of thousands or more.
An `m × m` matrix at that scale is enormous — on the order of ten gigabytes or
more — and would exhaust GPU memory.

A bare f4 standard error never needs that full matrix. Each quartet's standard
error depends only on that quartet's *own* variance, which is the diagonal entry
of the covariance matrix. The off-diagonal terms (how one quartet co-varies with
another) are never used, because f4 does not solve a linear system across quartets
the way qpAdm's generalised least squares does.

So instead of building the dense matrix and reading its diagonal, the code calls
`jackknife_diag`, which forms the diagonal *directly*. It computes, for each
estimate `k`, the average over blocks of the squared per-block leave-one-out
deviation. That equals the diagonal of the full covariance by construction, but it
is produced in work proportional to `m × (number of blocks)` and memory
proportional to `m` — never the `m²` memory of the dense matrix. This is what lets
an f4 sweep over many thousands of quartets run without running out of memory.

---

## 4. Why f4 uses no ridge / fudge

qpAdm applies a tiny ridge term (a value of `1e-4`) when it inverts its
generalised-least-squares matrix, to keep a near-singular matrix invertible. f4
carries no such term: it always uses a fudge of zero.

The reason is that the ridge only ever matters when a matrix is *inverted*. f4
never inverts anything. Its standard error is the plain, unmodified diagonal
variance from the jackknife. Regularising it would change a reported number for no
mathematical reason.

The options struct that callers pass in does carry a `fudge` field, purely so the
f4 entry points have the same shape as the qpAdm and qpWave entry points. That
field is deliberately ignored inside f4. The code explicitly discards it (and marks
it discarded so the strict-warnings build stays quiet) rather than silently letting
it look consulted.

---

## 5. Precision policy

The batch assemble is numerically delicate: it subtracts nearly-equal quantities,
which is exactly the situation where emulated double-precision arithmetic would
lose accuracy to cancellation. For that reason the f4 assemble always runs in
native double precision, under the same cancellation carve-out that the other f2
subtraction stages use.

The shared body still passes the library's default precision policy (emulated
double precision) down into the assemble call. This is not a behaviour change — the
assemble routine overrides it to native precision internally. Passing the default
down is only there so every stage is handed one consistent policy value rather than
some stages being special-cased at the call site.

The precision actually honoured for the run is recorded on the result as a tag, so
a caller can see which arithmetic mode was really used.

---

## 6. The z-to-p conversion (`f4_two_sided_p`)

`f4_two_sided_p` turns a z-score into a two-sided normal-tail p-value, matching
ADMIXTOOLS 2's convention for f4. The value it returns is
`2 × (1 − Φ(|z|))`, where `Φ` is the standard normal cumulative distribution.

Rather than compute that through the normal CDF, it uses the identity that this
quantity equals `erfc(|z| / √2)`, and calls the standard library's native
double-precision complementary error function `erfc` directly. The `1/√2` factor is
computed once and cached.

The function is deliberately not defended against non-finite input, because the
non-finite results are meaningful. If a quartet is degenerate — for example the
standard error is zero, so the z-score is infinite or not-a-number — then `erfc`
naturally returns zero for an infinite input and not-a-number for a
not-a-number input. Those propagate through as honest sentinels: a genuinely
impossible p-value never gets dressed up as a real one.

---

## 7. Error handling and edge cases

f4 never throws for a data-shaped problem. Every outcome, good or bad, is reported
as a status value on the result plus per-row values. There are three cases:

- **Empty batch.** If the caller passes no quartets, the result is a clean, empty,
  successful result — no rows, no fault.
- **Degenerate assemble.** If every genome block is missing so the assemble
  produces nothing usable (a zero-length batch or zero blocks), the result is still
  reported as successful, but every row's estimate, standard error, z, and p is set
  to not-a-number. A caller that filters on status alone still sees a
  per-row not-a-number sentinel instead of a crash or a silently wrong number.
- **Normal success.** Once the assemble succeeds, the per-quartet diagonal standard
  error is always valid. Within the read-out loop, a non-positive variance for an
  individual quartet yields a not-a-number standard error (and hence a
  not-a-number z and p) for just that quartet, without failing the whole batch.

There is one failure mode f4 structurally *cannot* have: the "covariance is not
positive-definite" error that qpAdm can raise. That error only arises when a matrix
is inverted, and f4 never inverts. So the only real f4 failure is a degenerate
assemble, which is handled as not-a-number rows above.

---

## 8. The primary GPU and the two entry points

`kPrimaryGpu` is a small file-private constant fixed at `0`. It names the single
GPU this routine runs on. Any spreading of work across multiple GPUs happens at a
higher layer that batches many models and hands each GPU its own slice; within one
f4 call the work stays on the primary GPU. The constant mirrors the same convention
in the qpAdm fit code and is kept here as a local naming choice rather than a global
setting.

There are two public `run_f4` overloads, differing only in where the f2 data lives:

- One takes **device-resident** f2 blocks (already in GPU memory), so on the GPU
  path there is no copy back and forth from the host.
- One takes a **host-side** f2 tensor, which the CPU reference path reads directly.
  This overload exists for validation against the reference implementation.

Both overloads are thin forwarders into one shared, templated body. The body is
templated on the f2 source type so the two paths share a single implementation and
cannot drift apart — the same deduplication pattern the qpWave routine uses.

---
