# `individual_partition.cpp` reference

## 1. Purpose

`src/io/individual_partition.cpp` reads an EIGENSTRAT `.ind` (or PLINK `.fam`)
sidecar and turns it into a **per-individual singleton partition**: one group per
retained sample, each labelled by its Genetic ID, in genotype-record order.

That "one group per sample" shape is the whole point. The sibling reader
`read_ind` (in `ind_reader.cpp`) keys the partition on the GroupID / population
column and *collapses* every individual sharing a label into a single `PopGroup`.
READv2 needs the opposite: each sample has to stay its own index so the per-pair
all-pairs sweep can enumerate every individual against every other. This reader
gives it exactly that — a partition whose group list is the sample list, each
group holding a single genotype row.

It is a pure host, CUDA-free io leaf. It opens one text file, tokenizes lines,
and returns a struct. It launches no kernel and reads no genotype payload — the
`.geno` matrix is read later, by the geno reader, using the row indices this
partition hands it.

---

## 2. What it produces

The return type is `IndPartition` (defined in `ind_reader.hpp`), shared with the
pop-collapsing reader:

| Field | Meaning here |
|---|---|
| `groups` | One `PopGroup` per **retained** sample, in genotype-record order. Each is `PopGroup{ label = Genetic ID, rows = { this_row } }` — a single-element `rows`. |
| `n_individuals_total` | The number of genotype records the file actually contributed (see section 6) — the width of the genotype matrix, not the count of retained samples. |

Because every group's `rows` is a singleton, the group index and the sample are
one and the same. That is the index space READ­v2's sweep walks: group `k` is the
`k`-th selected individual, and its label is that individual's Genetic ID.

---

## 3. Where the Genetic ID comes from (`.ind` vs `.fam`)

The Genetic ID is the sample's identity — the name the `--samples` restriction and
the downstream name→index resolver key on. Which column carries it depends on the
on-disk format:

- **EIGENSTRAT family** (`.ind`, the `GenoFormat::Eigenstrat` / `Tgeno` / `Geno` /
  `Ancestrymap` layouts) — column 1 (the first whitespace-token). That column *is*
  the sample id in the EIGENSTRAT convention; column 2 is sex and column 3 is the
  population/GroupID.
- **PLINK** (`.fam`, `GenoFormat::Plink`) — column 2, the within-family individual
  id (IID). PLINK's column 1 is the family id (FID), so the sample identity lives
  in the second field. The reader falls back to column 1 only if the line somehow
  has fewer than two tokens, so a malformed one-column `.fam` line still yields
  *something* rather than crashing.

The format is passed in by the caller (which already knows it from the genotype
triple), so this file never sniffs or guesses — it just picks the right column.

---

## 4. The optional `--samples` restriction

The `samples` argument is an `optional<vector<string>>`:

- **`nullopt`** — keep every individual in the file (subject to the record cap of
  section 6). This is the default "all samples" case.
- **A list** — keep only the samples whose Genetic ID is in the list. The list is
  loaded into an `unordered_set` (`want`) for O(1) membership tests, and any row
  whose id isn't wanted is skipped before it ever becomes a group.

When a restriction is in force the reader also tracks, in a `matched` set, which
of the *requested* ids it actually saw. That is what powers the
not-found check in section 5 — a requested id that never turns up is a hard error,
not a silent omission.

Note the asymmetry the set handling makes explicit: `want` is a set, so passing
the same id twice on `--samples` is harmless (it collapses to one), but two
different rows carrying the same id in the *file* is fatal (section 5).

---

## 5. Fail-fast validation

The reader would rather stop than hand back a subtly wrong partition, so three
conditions throw `std::runtime_error` (all prefixed
`io::read_individual_partition:`):

1. **Unreadable file.** If the `.ind`/`.fam` won't open, it fails immediately with
   the path.
2. **Duplicate Genetic ID among the retained samples.** A `seen` map records every
   id already turned into a group. A second row with the same id is fatal, because
   the downstream name→index resolver would have no way to say which row a name
   refers to — the mapping would be ambiguous. The message tells the user how to
   recover: restrict `--samples` to unique ids, or de-duplicate the `.ind`. The
   check is scoped to the *retained* set, so duplicates that were filtered out by
   `--samples` never trip it.
3. **A requested `--samples` id that isn't in the file.** After the scan, if the
   count of matched ids doesn't equal the count of requested ids, the reader walks
   the request list to find the first id it never matched and names it in the
   error. Silently dropping a requested sample would quietly change what gets
   modeled, so this is a hard stop.

There is also a final guard: if the partition came out **empty** — no rows
survived — that's an error too (`no individuals selected from …`). An empty
partition is never a valid result to return.

---

## 6. The record cap and `n_individuals_total`

The reader takes `n_records_present`, the number of genotype records that actually
exist in the companion `.geno` matrix, and it will not read past that many rows of
the sidecar. The loop breaks once `row` reaches `n_records_present`. This protects
against a sidecar that carries *more* lines than the genotype file has columns —
only the rows that have a matching genotype column are honored, and any trailing
sidecar rows are ignored rather than producing dangling indices.

`n_individuals_total` is then set to `row`: the count of records the file
contributed, capped at `n_records_present`. Two things about this number are worth
being precise on:

- It counts **every** non-empty line consumed, retained or not — the `row` counter
  advances for a filtered-out sample just as for a kept one. So under a `--samples`
  restriction, `n_individuals_total` is the size of the *full* individual set
  (the genotype matrix width), while `groups.size()` is the *selected* subset. The
  two are equal only when there's no restriction.
- The row index stored in each singleton group (`this_row`) is the index into that
  full record space, so the geno reader can address the right column of the
  genotype matrix even for a sparsely-selected subset.

Blank lines don't count. A line that tokenizes to nothing is skipped before `row`
is touched, so whitespace-only padding in the sidecar neither consumes a record
slot nor shifts the row indices.

---

## 7. Contracts and invariants

- **Order is preserved.** Groups appear in genotype-record order, so
  `groups[k].rows[0]` is strictly increasing across `k`. The index space matches
  the on-disk row order the geno reader expects.
- **Every group is a singleton.** `rows.size() == 1` for every group, always. That
  is the structural difference from `read_ind` and the property the READv2 sweep
  relies on.
- **Labels are unique.** After a successful return, no two groups share a label —
  guaranteed by the duplicate check (section 5). This is what lets a later stage
  build an unambiguous name→index map.
- **`groups.size() >= 1`.** A returned partition always has at least one sample;
  the empty case throws.
- **Under a restriction, the request is fully honored.** A successful return means
  every id on `--samples` was found (and, being a set membership test, each was
  found exactly once because a file duplicate would have thrown first).

---

## 8. Edge cases

- **A short `.fam` line** (fewer than two tokens) falls back to column 1 for the id
  rather than reading out of bounds.
- **Extra whitespace / blank lines** are tolerated — tokenization ignores runs of
  whitespace and empty lines are skipped without consuming a record slot.
- **A sidecar longer than the genotype matrix** is truncated at
  `n_records_present`; the extra tail is silently ignored (section 6).
- **Repeated ids on `--samples`** are harmless (they dedupe in the `want` set); a
  repeated id *in the file* among retained rows is fatal (section 5).
- **All requested samples filtered down to nothing** can't happen quietly: either a
  requested id was missing (a named error) or the result is empty (the
  `no individuals selected` error).
