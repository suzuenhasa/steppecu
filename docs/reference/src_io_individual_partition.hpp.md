# `individual_partition.hpp` reference

## 1. Purpose

`src/io/individual_partition.hpp` declares a single free function,
`read_individual_partition`, and its whole job is to answer one question: given a
genotype dataset's `.ind` (or PLINK `.fam`) sidecar, what are the *individuals*,
in genotype-record order, one label each?

That sounds like the job `read_ind` already does, but there's a deliberate
difference. `read_ind` (in `ind_reader.hpp`) keys the partition on the population
column — it *collapses* every individual that shares a GroupID into one `PopGroup`,
because the f-statistic machinery works on populations. READv2 doesn't want that.
READv2 is a per-*individual* all-pairs sweep: it forms every pair of samples and
scores them, so it needs each sample to be its own thing, its own index, its own
label. So this reader emits one **singleton** `PopGroup` per retained sample —
`{ label = the sample's Genetic ID, rows = { that one row } }` — in the order the
rows appear on disk.

The result reuses the exact same `IndPartition` / `PopGroup` structs as `read_ind`
(section 4), which is what lets everything downstream — the genotype tile readers,
the canonical-tile path, the sweep enumerator — treat a per-individual partition and
a per-population partition identically. The only difference the caller sees is that
here every group has exactly one row and its label is a sample identity rather than a
population name. This is a pure host, C++20, io-leaf translation unit: no CUDA, no
device code, just file parsing.

---

## 2. The signature and what each argument means

```
IndPartition read_individual_partition(
    GenoFormat format,
    const std::string& path,
    const std::optional<std::vector<std::string>>& samples,
    std::size_t n_records_present);
```

| Argument | What it's for |
|---|---|
| `format` | The on-disk genotype format. Its only job here is to pick which column carries the Genetic ID: PLINK `.fam` uses column 2 (the within-family IID), everything in the EIGENSTRAT family uses column 1 (section 3). |
| `path` | The sidecar to read — a `.ind` for the EIGENSTRAT family, a `.fam` for PLINK. |
| `samples` | An optional allow-list of Genetic IDs to keep. `std::nullopt` means "take every individual present". A present-but-empty vector is a legitimately-empty restriction (and would select nothing, tripping the "no individuals selected" guard). |
| `n_records_present` | How many genotype records actually exist. Rows in the sidecar past this count are ignored (section 6). This comes from the genotype reader's header, not from counting `.ind` lines, so a sidecar with trailing junk rows can't outrun the real data. |

It's marked `[[nodiscard]]` — the returned partition *is* the whole point of calling
it, so dropping it on the floor is almost certainly a bug.

---

## 3. Which column is the Genetic ID

The one format-dependent decision is where the sample's identity lives on each line.
The reader tokenizes a line on whitespace, then:

- **PLINK `.fam`** (`format == GenoFormat::Plink`, and the line has at least 2
  tokens): the Genetic ID is token 2 — the within-family individual ID (IID). A
  `.fam` line is `FID IID PID MID SEX PHENO`, and it's the IID, not the family ID,
  that identifies the sample.
- **Everything else** (the EIGENSTRAT `.ind` family): the Genetic ID is token 1 —
  the first column, which in an `.ind` is the sample identity.

Note the guard on the PLINK branch: it only reaches for token 2 when the line
actually has two tokens. A malformed one-token PLINK line falls back to token 1
rather than reading out of bounds.

---

## 4. What comes back: `IndPartition` and `PopGroup`

Both types are declared in `ind_reader.hpp` and shared verbatim:

- `PopGroup { std::string label; std::vector<std::size_t> rows; }` — one selected
  group. From this reader, `label` is a Genetic ID and `rows` is a single-element
  vector holding that individual's genotype-record index.
- `IndPartition { std::vector<PopGroup> groups; std::size_t n_individuals_total; }` —
  the whole partition, plus the total count of individuals the sidecar presented.

The subtle contract worth stating plainly: **the row index stored in a group is the
individual's position in the genotype records**, because the `.ind`/`.fam` row order
*is* the genotype individual axis. That's why the reader records `this_row` (the
pre-filter row counter) into the group, not the group's own sequence number — the
genotype tile readers use that index to pull the right column out of the packed
genotype matrix. If a `--samples` filter drops rows 0 and 2, the surviving groups
still carry rows 1, 3, 4… — the original disk positions, never renumbered.

`groups` is in **row order**: the singleton index space the READv2 all-pairs sweep
enumerates is exactly `[0, groups.size())`, and pair `(i, j)` means "the i-th and
j-th retained samples in file order". Keeping the output in file order is what makes
that index space stable and predictable.

---

## 5. `n_individuals_total`

`n_individuals_total` is set to the number of genotype records the reader *examined*
— every non-empty row it walked, up to the `n_records_present` cap — **not** the
number of samples it retained. So with a `--samples` restriction, `groups.size()` is
the selected count while `n_individuals_total` is the population the selection was
drawn from. Blank lines are skipped before the counter advances, so they don't
inflate the total. The distinction matters to callers that want to report "kept N of
M individuals".

---

## 6. The parse loop, step by step

The reader opens the file (an unopenable path is an immediate throw, section 7) and
walks it line by line. For each line:

1. **Stop at the record cap.** If the running row counter has reached
   `n_records_present`, the loop breaks — trailing sidecar rows beyond the real
   genotype data are never considered.
2. **Tokenize on whitespace.** A completely blank line is skipped without consuming
   a row index.
3. **Pick the Genetic ID** per the format rule of section 3, and capture the current
   row index (`this_row`) before advancing the counter.
4. **Apply the `--samples` filter.** If a restriction is in force and this ID isn't
   in the allow-list, skip it. Otherwise mark the ID as matched (section 8 uses this
   to detect requested-but-absent IDs).
5. **Reject a duplicate** retained Genetic ID (section 7).
6. **Emit the singleton group** `{ gid, { this_row } }` and remember the ID in a
   `seen` map so step 5 can catch a later collision.

After the loop, `n_individuals_total` is finalized, the requested-ID completeness
check runs (section 8), and an empty result is rejected (section 7).

It's a single streaming pass — no need to hold the whole file in memory, and the
row counter is advanced exactly once per non-empty in-range line, which is what keeps
the stored indices aligned to the genotype axis.

---

## 7. The three fail-fast errors

Every failure throws `std::runtime_error` with an `io::read_individual_partition: `
prefix, so a caller (the READv2 command) can catch it, print the message, and exit
with an invalid-configuration code. There is no partial or best-effort result — the
function either returns a fully valid partition or throws.

1. **Unreadable file** — the sidecar can't be opened.
2. **A duplicate Genetic ID among the retained samples.** This is the load-bearing
   one. READv2 builds a name→index resolver over the labels, and two samples sharing
   an ID would make that lookup ambiguous — a silent correctness hazard where a pair
   involving "the sample named X" is undefined. So the reader refuses up front, with
   a message that tells the user how to fix it: restrict `--samples` to unique IDs,
   or de-duplicate the `.ind`. Note the check is over *retained* rows only — a
   duplicate that a `--samples` filter excludes is harmless and doesn't trip it,
   because it can never reach the resolver.
3. **No individuals selected.** If the walk produced zero groups — an empty file, an
   all-blank file, or a `--samples` set that matched nothing survivable — that's an
   error, not an empty success. READv2 needs at least two individuals to form a pair,
   so an empty partition is never useful.

---

## 8. The `--samples` completeness check

When a restriction is in force, the reader doesn't just silently keep whatever
matched — it insists that **every** requested ID was actually found. It tracks
matched IDs in a `matched` set as it goes; if, at the end, `matched.size()` doesn't
equal the number of distinct requested IDs, it re-scans the request list and throws
on the first ID that never turned up, naming it and the file.

This is a deliberate strictness. A typo'd or stale sample name in a `--samples` file
should be a loud, named error, not a quietly smaller sweep — otherwise a user could
ask for 50 samples, get 49 because one name was misspelled, and never notice the pair
count was off. Failing fast on the missing name is the honest behavior. (The check
compares against the *distinct* requested IDs, so a `--samples` file that lists the
same ID twice isn't itself an error as long as that ID is present.)

---

## 9. Edge cases worth knowing

- **Blank lines** anywhere in the sidecar are skipped and cost nothing — they don't
  consume a row index or inflate `n_individuals_total`.
- **Trailing rows past `n_records_present`** are ignored, so a sidecar that's longer
  than the genotype data (extra descriptive rows, say) can't produce individuals
  that don't correspond to real genotype columns.
- **A one-token PLINK line** falls back to column 1 rather than reading past the end
  of the token list (section 3).
- **A `--samples` file listing the same ID twice** is fine as long as the ID exists;
  the completeness check works over distinct requested IDs. But two *different* disk
  rows carrying the same Genetic ID is the section-7 duplicate error.
- **Row indices are never renumbered** after filtering — a retained sample keeps its
  original genotype-record position, which is exactly what the genotype tile readers
  need to index the packed matrix.
