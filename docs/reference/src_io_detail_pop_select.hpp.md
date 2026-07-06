# `pop_select.hpp` reference

## 1. Purpose

`src/io/detail/pop_select.hpp` holds the one shared function that decides **which
populations a genotype file's sample table actually keeps**, and turns the survivors
into the tidy per-population row groups the rest of steppe consumes.

A little background. Genotype datasets ship two companion files: a big matrix of the
genotype calls, and a small text sidecar that names, one line per sample, which
population each sample belongs to. EIGENSTRAT calls that sidecar the `.ind` file;
PLINK calls it the `.fam` file. They have different columns and different formatting,
but once you have parsed either one you are left with the exact same thing: a list of
populations, each with the set of sample rows that belong to it. From that list you
usually do not want *every* population — you want a chosen subset (say, "just these
five", or "the twenty biggest", or "everything with at least four samples"). Picking
that subset is identical work no matter which sidecar format you started from.

That is what this header is: the shared **selection tail**. The `.ind` reader and the
`.fam` reader each do their own format-specific parsing, then both hand their parsed
groups to `select_populations()` here to apply the subset rule. The reason it lives in
one place is the same reason the f2 numerics do — the two readers used to carry
byte-identical copy-pasted copies of this logic, and two copies of a rule can quietly
drift apart when someone fixes a bug in one and forgets the other. With a single
source of truth, an `.ind` file and a `.fam` file describing the same cohort are
guaranteed to select the same populations, in the same order, every time.

Everything here is plain host C++20 — no GPU, no CUDA. It runs once at load time,
long before any genotypes reach the device.

---

## 2. `RawGroup` — one parsed-but-not-yet-chosen population

```
struct RawGroup {
    std::string label;                 // the population name
    std::size_t first_seen = 0;        // its rank in first-appearance order
    std::vector<std::size_t> rows;     // the sample row indices in this population
};
```

This is what a reader produces for each distinct population *before* selection has
happened — every population in the file, whether or not it will survive the filter.

`label` is the population name as it appeared in the sidecar. `rows` are the row
indices of the samples belonging to it; because a sidecar row's position **is** that
sample's index into the genotype matrix, these indices point straight at the right
columns of genotype data later on. `first_seen` records the order in which the
population first showed up while scanning the file, and it exists purely so the
"top-K" mode below can break ties deterministically — same file in, same choice out,
run after run.

`RawGroup` is the reader-internal input type. The output type, `PopGroup` (defined in
`ind_reader.hpp`), is the trimmed public version: just `label` and `rows`, no
`first_seen`, because once selection is done the appearance order no longer matters.

---

## 3. `select_populations()` — apply the subset rule

```
template <class ThrowFn>
std::vector<PopGroup> select_populations(
    const std::vector<RawGroup>& groups,
    const PopSelection& sel,
    const ThrowFn& throw_io_error);
```

Takes every parsed population (`groups`), the caller's selection request (`sel`), and
a callback for reporting an error (`throw_io_error`), and returns the chosen
populations as `PopGroup`s sorted by name.

`sel` is a `PopSelection` and its `mode` field picks one of three strategies, each
handled by a `case` in the `switch`. The function first collects pointers to the
surviving `RawGroup`s into `selected` (pointers, so nothing is copied while
filtering), then finishes with a shared tail (section 7) that all three modes run
through.

The small `filter_into` lambda is a shared convenience used by two of the three
modes: it walks every group and keeps the ones a predicate accepts. Writing it once
keeps the "Explicit" and "MinN" cases down to a single line each.

### Why `throw_io_error` is a callback

This is the neat trick that lets one function serve two readers with honest error
messages. When selection ends up empty, the user deserves an error that says which
file and which reader failed — the `.ind` reader wants a message prefixed
`io::read_ind:` and the `.fam` reader wants `io::read_fam:`, each followed by the
offending path. Rather than teach this shared helper about both formats, each reader
passes in its **own** throw callback that already knows its prefix and path. This
header just calls `throw_io_error("population selection is empty for ")` and the
reader's callback stitches on the format-specific context. So the selection logic
stays shared while the error text stays per-format — the copies cannot drift, but the
messages still read correctly for each caller.

`ThrowFn` is a template parameter (rather than, say, a `std::function`) so the whole
thing inlines with no call overhead, and the callback is expected to actually throw —
`select_populations` keeps going as if it will not return, which is why there is no
early `return` after the throw.

---

## 4. Mode `Explicit` — keep exactly this named list

```
case Explicit:  keep every group whose label is in sel.labels
```

The caller supplied `sel.labels`, a list of population names they want. This builds an
`unordered_set` of those wanted names (so each membership test is a fast hash lookup
rather than a scan) and keeps any group whose `label` is in the set.

A couple of consequences worth knowing. Names the caller asked for that are **not** in
the file are simply absent from the result — this mode does not complain about a
missing request, it only keeps what genuinely exists. And the output order does not
follow the order the caller listed the names in; like every mode, the result comes out
sorted by label in the shared tail (section 7).

---

## 5. Mode `AutoTopK` — keep the K largest populations

```
case AutoTopK:  keep the sel.k populations with the most samples
```

This picks the `k` populations that have the most sample rows — a common default when
you just want the best-represented groups without naming them. It copies pointers to
all the groups into `by_count`, then `stable_sort`s them by descending `rows.size()`,
breaking ties by ascending `first_seen`, and takes the first `k`.

Two details make this deterministic and safe:

- The tie-break on `first_seen` means two populations with an equal sample count are
  ordered by which appeared first in the file — so a file with ties always yields the
  same K, not a coin flip. (`stable_sort` reinforces this by never reordering equal
  elements.)
- `k` is clamped with `min(sel.k, by_count.size())` before slicing, so asking for more
  populations than the file contains just returns all of them rather than reading past
  the end.

---

## 6. Mode `MinN` — keep every population with enough samples

```
case MinN:  keep every group with rows.size() >= sel.min_n
```

The simplest rule: keep any population that has at least `sel.min_n` samples, drop the
rest. A one-line `filter_into` on the sample count. This is the "give me everything
that is not too thin to be useful" option, and unlike top-K it does not cap how many
populations come back — it keeps as many as clear the bar.

---

## 7. The shared tail — empty check, sort, and convert

After the mode-specific `switch`, all three paths funnel through the same closing
steps:

1. **Empty guard.** If nothing survived selection, call `throw_io_error(...)`
   (section 3) so the reader reports its format-specific "population selection is
   empty" error. An empty cohort can never produce statistics, so this is caught
   right at the door rather than failing mysteriously downstream.
2. **Sort by label.** The surviving pointers are sorted by `label`. This is what makes
   the output order **stable and format-independent** — the same set of populations
   comes back in the same alphabetical order whether it came from an `.ind` or a `.fam`
   file, and regardless of which mode chose it. Downstream code (and goldens) can rely
   on that ordering.
3. **Convert to `PopGroup`.** Finally each chosen `RawGroup` is copied into a public
   `PopGroup` — label plus rows, dropping the now-irrelevant `first_seen` — and the
   vector is returned. The result is `reserve`d up front so the conversion does not
   reallocate.

The upshot: no matter which sidecar format you started from or which selection rule
you asked for, you get back the same shape of answer — a label-sorted list of
populations with their genotype row indices — produced by exactly one piece of code.
