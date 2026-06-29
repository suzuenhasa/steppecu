I read through `plink_reader.cpp` against its header and the existing `snp_reader.cpp` / `ind_reader.cpp` twins. It is clearly the work of someone who understands PLINK/AT2 semantics, but a senior reviewer would flag consistency and duplication issues.

## What's genuinely good

- **The byte-exact format contract is well documented** (lines 3–20, 190–193). Pinning the canonical-polarity decision `ref := A1` and citing `format-readers.md` / AT2 `mcio.c` shows real domain care.
- **Locale-free, throw-free numeric parsing** with `std::from_chars` (lines 55–60) and an explicit non-finite guard (lines 98–106) mirrors the `.snp` reader's correctness arguments.
- **Single-homed constants** for chromosome codes and missing-allele mapping (line 23, lines 118–119) avoid magic literals.
- **Error handling is uniform**: every failure path throws `std::runtime_error` with a line number (lines 101–103, 170–172, 180–183), matching the `io`-leaf contract.
- **No C-style regressions**: no `FILE*`, no `printf`, no raw pointers; pure C++20 RAII.

## What a senior developer would flag

**The `n_records_present` cap in `read_fam` is inconsistent with `read_ind` and with the header contract.** Lines 242–245:

```cpp
if (row >= n_records_present) {
    ++row;
    continue;  // beyond the present .bed records (partial-file cap)
}
```

`read_ind` handles the same cap but explicitly **does not** increment `row` there (`ind_reader.cpp:64-72`), because counting past-cap rows would inflate `n_individuals_total` past `n_records_present`. Here it *does* increment, so if the `.fam` is longer than the `.bed`, `part.n_individuals_total = row;` (line 270) ends up larger than the `.bed` axis. The header (lines 90–92) promises "the SAME partial-file cap read_ind takes" — it isn't. That's a real bug for partial-file workflows.

**Heavy duplication with `snp_reader.cpp` and `ind_reader.cpp`.** `parse_full`, `chrom_code`, `split_ws`, `parse_genpos`, `RawGroup`, the selection switch, `filter_into`, and the final label sort are all copied nearly verbatim. The comment at lines 44–50 admits this ("replicated rather than shared because snp_reader's helpers are file-local"). For a one-off reader that's defensible, but in a showcase it looks like the author couldn't decide between "don't touch the golden file" and proper factoring. A senior dev would ask for a private `io/detail/format_read_common.hpp` rather than three copies.

**`.bim` field count is only a lower bound.** Line 179:

```cpp
if (fields.size() < kBimFields) { ... }
```

PLINK `.bim` is *exactly* six columns; extra fields are silently ignored. A malformed seven-column line would pass by taking the first six and hiding the problem. The `.snp` reader has a reason to accept `>=3` (optional alleles); `.bim` doesn't.

**`bim_allele` silently truncates multi-character alleles.** Lines 143–147:

```cpp
if (fields[col].empty()) return kCanonMissingAllele;
const char a = fields[col][0];
return a == kPlinkMissingAllele ? kCanonMissingAllele : a;
```

PLINK alleles are normally single bases, but a malformed `DEL` becomes `D` with no diagnostic. A `fields[col].size() == 1` check (or at least an assertion) would make the contract explicit.

**No `default` in the selection switch.** Line 284:

```cpp
switch (sel.mode) {
    case PopSelection::Mode::Explicit: ...
```

Exhaustive today, but if someone adds a mode to `PopSelection`, this silently does nothing and returns an empty selection. A `default: throw std::logic_error(...)` is cheap insurance.

**Performance / ergonomia nits:**
- `std::istringstream` per line allocates and depends on locale; a `std::string_view`-based split would be faster and avoid `<sstream>`.
- `SnpTable` vectors are grown without reserve; for a large `.bim` that's a lot of reallocations. A two-pass parse or `reserve` from file size would help.

## The "slop" test

**Not slop.** The comments are dense but accurate, the format conventions are single-homed, error handling is consistent, and the AT2 provenance is documented. The biggest issue (the cap bug) is the kind of subtle copy-paste drift that duplication causes, not lazy, unthinking slop.

## What it actually looks like

Competent, careful research code written by someone who knows the genomics file formats and the reference implementation cold. It reads like a deliberate twin of the existing EIGENSTRAT readers rather than an independently designed parser — mostly good, but that duplication leads to the one real cap-semantics drift. The comment-to-code ratio is high even by this project's standards; the comments explain *why*, but a senior reader might still wish some of the "EXACTLY the same as snp_reader" boilerplate were replaced by shared code.

## Verdict

**B.** Ship after fixing the `n_records_present` row-count bug in `read_fam` and tightening `.bim` validation. A follow-up refactor to share the common helpers with `snp_reader` / `ind_reader` would move this to a solid **B+**.