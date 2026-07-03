# `model_search.cpp` reference

## 1. Purpose

`src/core/qpadm/model_search.cpp` is the orchestrator that fits a whole *list* of
candidate qpAdm models in one call. A model-space search hands it many candidate
models — each a choice of target, source ("left") populations, and reference
("right") populations — and this file drives every one of them through the qpAdm
fit, spreads the work across the available GPUs, and hands back the results in a
stable order.

It owns four things:

1. **The single per-model fit chain** — the one place a single model is turned into
   a result (assemble the f4 matrix, run the multi-step fit, tag the result with the
   model's identity). Everything else routes through this one definition.
2. **The default "fit a batch of models" body** — a plain per-model loop, used for
   the large/uncommon models and for the CPU reference backend.
3. **The per-GPU shard fitter** — the routine that takes one GPU's slice of the
   model list, splits it into the fast batched path and the fall-back per-model
   path, and returns results tagged with each model's identity.
4. **The two public search entry points** — one that reads f2 data already resident
   in GPU memory (the real path), and one that reads a host-side f2 tensor (the
   reference-oracle path). The device entry point also owns the multi-GPU fan-out:
   replicate the f2 data to every GPU once, shard the model list across GPUs, run
   one worker thread per GPU, and reassemble the results deterministically.

The file is deliberately free of CUDA code. It names only the GPU-free backend
interface (`ComputeBackend`) and the GPU-free resource handle (`Resources`), so all
the actual device work happens behind virtual calls on the backend. That keeps the
orchestration logic testable and lets the same code drive both the real GPU backend
and the CPU reference backend.

---

## 2. The single per-model fit chain (`fit_one_model`)

`fit_one_model` is the one and only place a single model becomes a result. Every
other path in the file — the device batched default, the large-model tail, the
per-GPU workers, and the host oracle — funnels through it, so the fit is defined
exactly once and cannot drift between paths.

It does three steps:

1. **Assemble the f4 matrix** (`assemble_f4`) from the f2 blocks, using the model's
   left-plus-target populations and its right populations.
2. **Run the multi-step fit** (`run_impl`): the generalised-least-squares solve, the
   rank test, and the source-drop / population-drop sub-models.
3. **Echo the model's stable identity** onto the result (`res.model_index = model.model_index`)
   so the caller can put the result back in the right place regardless of what order
   the fits actually finished in.

### Templated on where the f2 data lives

`fit_one_model` is a template on the f2 *source* type. The same body serves both the
device-resident f2 blocks (GPU memory, no copy back to the host) and the host-side
f2 tensor (the reference path). Both f2 types expose a `.block_sizes` member and both
work with an overloaded `assemble_f4`, so one `assemble → run → echo` definition
covers both. `fit_one_model_device` is just a thin, non-template forwarder onto this
body for the device-resident case.

### The survivor-block weighting

The block jackknife inside the fit must be weighted by the genome blocks that
actually survived assembly, not by every block that existed going in. If any
population pair had no data in a block, that block is dropped during `assemble_f4`,
and the fit reads the surviving block sizes from the assembled matrix (`X.block_sizes`)
rather than from the original f2 data. When nothing was dropped, the two are
byte-for-byte identical.

### The adjacency invariant (do not separate assemble from run)

`assemble_f4` caches a per-backend value (an internal total line count) that the
covariance step inside `run_impl` later reads back off the *same* backend object.
That makes it a hard rule that the assemble call and the `run_impl` call for one
model must happen back-to-back on the same backend instance, with no other model's
assemble in between. This file honours that by only ever fitting one model at a time
per backend, sequentially, within each GPU worker. Do not reorder or interleave the
assemble and run steps across models.

### Precision policy

The fit does not run in a single arithmetic mode. It follows the library-wide policy
of emulated double precision as the default with a native double-precision carve-out
for the numerically delicate steps:

- The f4 **assemble** stays in native double precision (it subtracts nearly-equal
  quantities, where emulated precision would lose accuracy to cancellation).
- The **covariance** step uses the emulated double-precision default (it is a large
  matrix multiply, where emulated precision is much faster at essentially the same
  accuracy), with an automatic fall-back to native precision.
- The **singular-value decomposition**, the **matrix inverse**, and the
  **chi-squared** step run in native double precision.

One consistent precision value (the library default) is passed down to both the
device path and the host path identically; the stages that need native precision
override it internally, so the call site never special-cases them.

---

## 3. The default batch body and the large-model tail (`fit_models_batched_default`)

`fit_models_batched_default` fits a list of models the simplest possible way: a plain
loop that calls the single per-model chain once per model, each fit running wholly
through the backend's own device virtual calls. It is used in exactly two situations:

- **The large / uncommon models** (the "tail" of the search). Dispatching these one
  model at a time is the correct shape for them — they do not fit the fixed-size
  device arrays the batched path relies on (see section 4).
- **The CPU reference backend.** The reference oracle has no batched override, so it
  routes every model — small or large — through this per-model loop. A host loop is
  the correct shape for the oracle; it is the bit-exact reference the batched GPU path
  is checked against, not the shipping fast path.

A data-shaped outcome (a model that does not fit, a degenerate solve, and so on) is
never thrown from here. It rides on the individual result's status field. Only a
genuine device or backend fault would throw.

---

## 4. The small-path partition gate (`model_in_small_path`)

`model_in_small_path` answers one yes/no question about a single model: is it small
enough to go through the fast, fixed-size batched GPU path? It is CUDA-free — a plain
host predicate on the model's left count, right count, and rank — so the search can
partition the model list before touching a GPU.

The important design point is that it does **not** invent its own size limits. It
delegates to a single shared source of truth (`model_fits_small_path`) that *also*
sizes the fixed per-thread arrays inside the GPU kernel and backs the backend's own
capability check. Because the same one function decides both "is this model
small-path?" and "how big are the kernel's fixed arrays?", the host partition gate
can never grow wider than the arrays it routes work into. If the gate could drift
wider, it would admit an oversized model that overflows the fixed device-local arrays
— undefined behaviour. Keeping the rule in one place makes that class of bug
impossible.

---

## 5. Fitting one GPU's shard (`fit_shard`)

`fit_shard` takes one GPU's slice of the model list and fits all of it on that GPU,
returning results in the same order as the input slice (each result also carries its
own model identity, so a later re-sort works regardless of order). It does this in
two stages.

### Partition, then route

It first walks the slice once and splits it into two sub-lists, remembering each
model's original position:

- **Small-path models** → the genuine batched primitive.
- **Large-path models** → the per-model default body (section 3).

For the small-path models, it makes an **explicit capability query** on the backend
(`provides_batched_fit()`) rather than a try/catch probe:

- If the backend advertises a real batched override (the GPU backend does), the small
  models go through it — a single batched dispatch per same-shape bucket, not a
  per-model loop. This is the actual throughput win of the whole design.
- If the backend has no batched override (the CPU oracle), the small models fall back
  to the per-model default too.

Using an explicit capability flag rather than catching an exception matters for
correctness: a real fault thrown from *inside* the batched override now propagates as
the fault it is, instead of being silently swallowed and misread as "no override
available."

### Scatter the results back (`scatter_by_pos`)

Each sub-list's results are scattered back into the output at the positions the
partition recorded (`scatter_by_pos`), so the returned vector lines up with the input
slice. That scatter routine is defined once, with a defensive bounds check, and is
shared by both the small-path and large-path scatter so the two cannot diverge.

---

## 6. The deterministic re-sort by pre-sized slots (`scatter_into_slots`)

The search must return results in a stable, reproducible order no matter how many
GPUs ran, how the model list was sharded, or which fits finished first. This file
gets that for free by **pre-sizing the result vector to `n` and writing each result
into the slot named by its own model identity**, rather than sorting after the fact.

`scatter_into_slots` is that write. Each model's identity (`model_index`) is a value
in `[0, n)` that the search stamped onto the input, so writing `results[model_index]`
puts every result in its canonical place. Because each identity is written exactly
once, the pre-sized-slot write *is* the re-sort — there is no separate sort step, and
the answer is bit-identical for any number of GPUs.

The routine **fails fast** if a model identity is out of range, because the whole
determinism guarantee rests on every index being a valid, unique slot. That check is
defined once here so the single-GPU fast path and every multi-GPU worker enforce it
identically. (The host-oracle entry point in section 9 deliberately does *not* fail
fast — its semantics differ, so it keeps its own inline fall-back instead of calling
this routine.)

---

## 7. The device search entry point (`run_qpadm_search`, device f2)

This is the real entry point: it takes f2 blocks already resident in GPU memory, a
list of models, the options, and the set of GPUs to use. It pre-sizes the result
vector to `n` slots (section 6) and then splits on how many GPUs are available.

### The single-GPU fast path

When there is exactly one GPU, there is no sharding, no threads, and no f2
replication. The one backend fits all `n` models on its device through `fit_shard`,
and the results scatter into their pre-sized slots. This is the supported, default
path.

### The multi-GPU fan-out

When there are two or more GPUs:

1. **Replicate the f2 data to every GPU** once, up front (section 8).
2. **Shard the model list** into contiguous, non-overlapping ranges — one per GPU.
3. **Fan out one worker thread per GPU** (`std::jthread`). Each worker fits its own
   contiguous sub-range wholly on its own GPU (using that GPU's own f2 replica) and
   writes its results into the shared, pre-sized result vector.
4. **Join** all workers at a barrier (the thread destructors), then surface any fault.

The concurrent writes are **race-free without a lock**. Each shard is a contiguous,
non-overlapping range of model identities, and each result is written into the slot
named by its identity, so no two workers ever touch the same slot. An empty shard
(more GPUs than models) simply does nothing.

### The exception model

There are two distinct kinds of "something went wrong," handled differently:

- **Data-shaped outcomes** (a model that does not fit, a degenerate solve) are never
  thrown. They ride on each result's status field.
- **Genuine device or backend faults** are real exceptions. Each worker catches its
  own exception and stashes it; after the join barrier, the entry point rethrows the
  fault from the **lowest-numbered GPU** that failed. Picking the lowest index makes
  the reported failure deterministic rather than a race between workers.

---

## 8. Multi-GPU f2 replication and why multi-GPU is deferred

The multi-GPU path needs every GPU to have its own copy of the f2 data before the
workers start. `replicate_f2` does that broadcast once, up front. The input f2 lives
on exactly one GPU; for that GPU the existing handle is reused with no copy, and for
every *other* GPU the data is materialised to host once and uploaded. The uploaded
replicas are owned and freed automatically at the end of the search (they are held in
a vector whose destructors release them); the per-GPU pointers the workers use are
borrowed views into either the original handle or an uploaded replica.

The residency rule — "this GPU needs an upload if its device id differs from the one
the f2 already lives on" — is evaluated in a single pre-pass that both records the
per-GPU decision and counts the uploads. The host copy is materialised only if at
least one GPU actually needs it. Consolidating the rule into one expression stops the
three separate loops (classify, materialise, upload) from drifting apart if only one
were edited.

### Why multi-GPU is deferred

This replication is correct and deterministic — a two-GPU run produces bit-identical
results to a one-GPU run — but it is a throughput bottleneck on the current hardware.
The consumer-grade GPUs in use cannot talk to each other directly (no peer-to-peer),
so the f2 data has to make a large round trip through host memory: roughly several
gigabytes and a few seconds. That host bounce caps the multi-GPU speed-up at only
about 1.2× on real data, so it is not worth splitting across GPUs today. Until a
future fix eliminates the bounce (by precomputing each GPU's data in place rather than
copying, or a direct device-to-device combine), the single-GPU path is the supported
one — prefer one GPU.

---

## 9. The host-oracle search entry point (`run_qpadm_search`, host f2)

The second public entry point takes a host-side f2 tensor instead of device-resident
blocks. It routes **every** model through the single per-model fit chain
(`fit_one_model`) on the first backend, in a plain host loop. A host loop is the
correct shape for the reference oracle: it is the bit-exact reference the device
batched path is checked against, not the shipping deliverable. It reuses the exact
same per-model chain the device path uses (just instantiated on the host f2 type), so
the reference and the real path can never fit a model two different ways.

One deliberate difference from the device path: the oracle does **not** fail fast on a
bad model identity. Where the device path (section 6) throws if a result's identity is
out of range, the oracle instead falls back to writing the result at its plain loop
position. The oracle's job is to produce a reference answer for every model it was
given, so it prefers a sensible fall-back slot over aborting the whole run — its
semantics genuinely differ from the strict device re-sort, which is why this scatter
stays inline here instead of calling the shared fail-fast routine.
