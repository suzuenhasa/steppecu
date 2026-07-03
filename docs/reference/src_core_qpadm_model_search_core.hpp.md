# `model_search_core.hpp` reference

## 1. Purpose

`src/core/qpadm/model_search_core.hpp` answers one narrow question: when steppe
has a large batch of models to fit and several GPUs to fit them on, *which models
go to which GPU?* It defines a small struct that describes one GPU's slice of the
work and a single function that produces the full assignment.

The whole file is deliberately **host-only and GPU-free**. It contains no CUDA
code, makes no calls into the GPU runtime, and touches no device memory — the
assignment is pure integer index math over two numbers: how many models there are
and how many GPUs are available. That matters for testing: because nothing here
needs a real GPU, the planner can be unit-tested on an ordinary machine against a
stand-in backend, which is the same testing approach already proven for the
multi-GPU f2 code.

The context is a large model-search workload (thousands of candidate models fit in
one sweep). Splitting that batch across GPUs is exactly the kind of thing that is
easy to get subtly wrong — off-by-one ranges, uneven splits, or a
finish-order-dependent result. Isolating the split into this tiny, testable,
math-only unit is what keeps it correct and reproducible.

---

## 2. The shard plan and the `ModelShard` struct

A *shard* is one GPU's assigned slice of the model batch. `plan_model_shards`
returns a list of them — one per GPU — and each is a `ModelShard`:

| Field | Type | Meaning |
|---|---|---|
| `g` | `int` | The device index, from `0` to `G-1`. |
| `lo` | `std::size_t` | The first model this device owns (inclusive) — an index into the caller's `models[]` array. |
| `hi` | `std::size_t` | One past the last model this device owns (exclusive). |

The range is **half-open**, `[lo, hi)`: device `g` owns the models at positions
`lo`, `lo+1`, …, `hi-1`. This is the usual C++ begin/end convention, so the number
of models a device owns is simply `hi - lo`, and an empty range is just `lo == hi`.

### Contiguous ranges, not round-robin

Each device is given a **dense, contiguous block** of models — device 0 gets the
first block, device 1 the next block, and so on. It is deliberately *not*
round-robin (where device 0 would get models 0, G, 2G, … interleaved with the
others).

The reason is how each GPU then processes its slice. A device fits its models in
one *batched* dispatch, packing them together into a single work buffer. A
contiguous run of model indices packs cleanly into that buffer; an interleaved,
strided set of indices would not. Contiguous ranges are what make the downstream
batched dispatch simple and efficient.

---

## 3. How the models are balanced across devices

The split is **balanced by count** so no GPU is left doing far more work than the
others. With `n` models and `G` devices:

- The first `n % G` devices each get `ceil(n / G)` models.
- The remaining devices each get `floor(n / G)` models.

In words: divide as evenly as possible, and hand the leftover models (there are
`n % G` of them, always fewer than `G`) to the lowest-numbered devices, one extra
each. For example, 10 models across 3 devices gives ranges `[0, 4)`, `[4, 7)`,
`[7, 10)` — sizes 4, 3, 3.

---

## 4. Invariants and guarantees

The returned plan holds to a set of guarantees that callers can rely on:

- **Full coverage, no gaps, no overlaps.** The shards together cover exactly
  `[0, n_models)`. Every model is assigned to exactly one device; none is dropped
  and none is assigned twice.
- **Fixed device order.** The shards come back in device order, `g = 0, 1, …,
  G-1`. This is the *same* fixed order the multi-GPU f2 combine step uses, which is
  what keeps a multi-GPU run reproducible: results are always merged in one
  predictable order regardless of which device happens to finish first.
- **Empty shards are valid.** When there are fewer models than devices (`n < G`),
  the devices that drew nothing get an empty shard where `lo == hi`. That is a
  legitimate no-op slot, not an error — the loser device simply has no work.
- **`G` must be at least 1.** There must be at least one device. A device count of
  zero is not a valid input.

---

## 5. The implicit re-sort — why there is no sorting code

A subtle design point: results must come back in a stable, input order even though
several GPUs finish their slices at unpredictable times. You might expect a sorting
or reordering step to fix that up afterward. There isn't one, and there doesn't
need to be one.

The trick is that ordering is handled by *where each result is written*, not by
sorting after the fact. Before any worker thread starts, the caller allocates a
results vector already sized to hold all `n_models` entries. Each worker then
writes its result for model `i` into the slot at that model's own original index —
`results[models[i].model_index]` — rather than appending to a shared list.

Because every result lands in its own pre-assigned slot, the final vector is
already in the correct order the moment the last worker finishes. The finish order
of the threads is irrelevant. This "write into your own pre-sized slot" discipline
*is* the re-sort — it is implicit in the write pattern, which is why this file
needs no reordering logic at all. It is the same technique the multi-GPU f2 combine
uses, where each device writes into its own pre-sized partial-results slot.
