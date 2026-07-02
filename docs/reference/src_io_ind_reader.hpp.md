# `ind_reader.hpp` reference

## 1. Purpose

`src/io/ind_reader.hpp` declares the reader for the EIGENSTRAT `.ind` file — the
small text file that names the individuals (samples) in a genotype dataset and
says which population each one belongs to. Parsing it produces two things the rest
of steppe needs before it can compute anything:

1. **A per-population membership map** — for each population that survives
   selection, the list of genotype records (individuals) that belong to it.
2. **A total individual count** — how many rows the `.ind` file has, which is the
   size of the individual axis of the genotype matrix.

The reason this reader matters is that a population is defined entirely by *which
individuals it owns*, and the `.ind` file is the only place that mapping lives. The
row order of the `.ind` file is the individual axis of the genotype matrix, so the
row index of an `.ind` line is exactly the genotype record index for that
individual. This reader is what binds a population label to the set of genotype
records it covers.

The header is deliberately lightweight. It is pure host C++ and only uses the
standard library (`std::string`, `std::vector`, `std::size_t`). It does not depend
on any GPU code or any of steppe's core numeric types, so it is cheap to include and
easy to test on its own.

---

## 2. The `.ind` file and row ordering

An EIGENSTRAT `.ind` file has one whitespace-separated record per individual, with
three columns:

```
<sample-id>   <sex>   <population-label>
```

This reader only cares about the **third column**, the population label. It reads
the label of every row and groups individuals by it.

The single most important fact about the file is its ordering. The order of rows in
the `.ind` file *is* the order of individuals in the genotype matrix (the TGENO
record order). Because of that, **the zero-based row index of an `.ind` line is the
genotype record index for that individual.** Row 0 of the `.ind` describes the
individual stored as record 0 in the genotype data, and so on. Every index this
reader hands back is a genotype-record index in that sense — it can be used
directly to look up that individual's genotypes.

---

## 3. Choosing which populations to keep (`PopSelection`)

A real dataset can contain hundreds of populations, but a given analysis usually
wants a specific subset of them to become the rows of the computed matrices.
`PopSelection` is the small struct that says *which* populations to keep. It offers
three selection modes, chosen by its `mode` field.

### The three modes (`PopSelection::Mode`)

| Mode | Meaning |
|---|---|
| `Explicit` | Keep exactly the labels the caller listed in `labels` that are actually present in the file. Labels that were requested but not found are simply skipped, not an error. |
| `AutoTopK` | Keep the `k` populations with the **most individuals**. Ranking is by individual count, largest first; ties between populations with the same count are broken by which one appears *first* in `.ind` row order. This exactly mirrors the reference tool's "most common K" behavior. |
| `MinN` | Keep every population that has at least `min_n` individuals. |

`Mode::AutoTopK` is the default mode.

### The ordering invariant that ties them together

Whichever mode is used, **the final set of selected populations is sorted in
ascending order by label** before it is returned. This is not a cosmetic detail:
that sorted order fixes the population (row) ordering of the matrices steppe
computes downstream. Two runs that select the same populations — even via different
modes — will always place those populations in the same row order, because the
last step is always the same ascending sort by label.

This behavior is matched bit-for-bit against the reference tool that generated
steppe's validation matrices (`build_tgeno_matrix.py`), whose final selection step
is likewise a plain ascending sort of the chosen labels. Reproducing the selection
*and* its ordering exactly is what lets the decoder reproduce the reference
allele-frequency matrices bit-for-bit. The standing validation configuration is
`AutoTopK` with `k = 50` on the real v66 `.ind` file.

### `PopSelection` fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `Mode` | `AutoTopK` | Which of the three selection strategies to apply. |
| `k` | `std::size_t` | `0` | For `AutoTopK`: how many of the largest populations to keep. |
| `min_n` | `std::size_t` | `1` | For `MinN`: keep populations with at least this many individuals. |
| `labels` | `std::vector<std::string>` | empty | For `Explicit`: the exact labels to keep. The present subset of these is kept and then sorted. |

Only the field relevant to the chosen `mode` is consulted; the others are ignored.

---

## 4. One selected population (`PopGroup`)

`PopGroup` describes a single population that survived selection: its label, and the
individuals that belong to it.

| Field | Type | Meaning |
|---|---|---|
| `label` | `std::string` | The population label from column 3 of the `.ind` file. |
| `rows` | `std::vector<std::size_t>` | The individual-record indices that belong to this population, in ascending order (which is file order). Each index is an `.ind` row index, which — per section 2 — is the same as the genotype record index. |

The `rows` list is what lets a later stage gather exactly the genotype columns for
this population out of the full genotype matrix.

---

## 5. The parsed result (`IndPartition`)

`IndPartition` is the whole return value of parsing an `.ind` file: the selected
populations plus the total individual count.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `groups` | `std::vector<PopGroup>` | empty | The selected populations, already in the final row order of the computed matrices — ascending by label, as described in section 3. |
| `n_individuals_total` | `std::size_t` | `0` | The total number of `.ind` rows, i.e. the full length of the genotype individual axis. This is the count of *all* individuals in the file, independent of which populations were selected. |

Note the distinction: `groups` only holds the populations that passed selection,
but `n_individuals_total` always reflects the entire file. The total is the size of
the genotype matrix's individual axis, which is needed even for individuals that
were not selected into any group.

---

## 6. Parsing the file (`read_ind`)

```cpp
[[nodiscard]] IndPartition read_ind(const std::string& path,
                                    const PopSelection& sel,
                                    std::size_t n_records_present);
```

`read_ind` opens the `.ind` file at `path`, reads the population label (column 3)
of every row, groups the row indices by label, applies the selection described by
`sel`, and orders the result for the matrix row axis. The return type is marked
`[[nodiscard]]` because the parsed partition is the whole point of calling it —
ignoring it is almost certainly a mistake.

### The `n_records_present` cap

`n_records_present` caps the individual axis to the number of records that are
actually present in the genotype (`.geno`) file. It exists to handle a genotype
file that is shorter than its `.ind` — a partial file. Any `.ind` row whose index
is greater than or equal to `n_records_present` is ignored: that individual has no
genotype data behind it, so it must not be grouped into any population.

- For a normal TGENO dataset, pass the genotype header's record count
  (`n_records`).
- To use every `.ind` row with no cap at all, pass `SIZE_MAX`.

This mirrors the reference tool, which applies the same "first `n_records` rows
only" cap, so the two stay in agreement on which individuals count.

### Error handling

`read_ind` throws `std::runtime_error` in two cases:

- The file at `path` is missing or cannot be read.
- The selection produces an **empty** set of populations (nothing matched).

Surfacing I/O and selection failures as exceptions is the deliberate contract of
this reader: it reports failures to its caller as thrown exceptions rather than
depending on any of steppe's core or device error types. Callers that want to
tolerate a missing file or an empty selection must catch the exception themselves.
