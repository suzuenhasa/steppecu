I read through `src/io/geno_reader.cpp` carefully. This is **not slop** — it’s clearly written by someone who understands the genomics formats and cares about safety. But a senior reviewer would flag the maintainability debt that comes from being *too* careful in the same file over and over.

## What's genuinely good

- **Defensive arithmetic everywhere.** The checked multiply before `resize` (lines 441–445, 595–599, etc.) and the translation of `std::bad_alloc`/`std::length_error` into `std::runtime_error` (lines 463–473) are exactly the kind of thing that prevents silent heap-buffer-overflow writes. The static_assert on `std::streamoff` at line 76 is a nice touch.
- **Fail-fast on stale partitions.** Bounding individual rows by `records_present_` in `read_tile` (lines 411–418) and by `header_.n_ind` in the SNP-major readers (e.g. lines 572–578) catches a real class of “wrong dataset” bugs.
- **Format detection is layered and testable.** TGENO/GENO magic → EIGENSTRAT probe → PLINK `.bed` → ANCESTRYMAP (lines 244–330), with explicit magic constants and mode-byte rejection for individual-major `.bed`.
- **CRLF tolerance and partial-file handling** are thought through, not bolted on.
- **No CUDA in this TU, and it owns that.** The layering comment at the top is accurate; this is a clean host-side leaf.

## What a senior developer would flag

**Massive comment-to-code ratio, and a lot of it repeats the code.** Comments like this are common:

```cpp
// A blank line: count it as a record ONLY if more non-blank content follows
// (an interior blank is a malformed file the reader will reject; here we just
// do not let a trailing blank inflate n). Peek: at EOF -> trailing blank, stop.
if (in.peek() == std::char_traits<char>::eof()) break;
```

That’s four lines of comment for one line of code that already says `peek() == eof`. A senior dev would trust the code more and trim the prose.

**Copy-pasted selection/reorder construction across four readers.** The block that builds `sel_rows`, `pop_offsets`, `pop_labels`, and validates `row >= header_.n_ind` appears almost verbatim in:

- `read_snp_major_tile` (lines 563–584)
- `read_eigenstrat_snp_major_tile` (lines 684–705)
- `read_plink_snp_major_tile` (lines 837–858)
- `read_ancestrymap_snp_major_tile` (lines 981–1002)

This is classic copy-paste drift. One day someone adds a validation to two of them and forgets the other two. It should be one helper: `build_snp_major_selection(part, n_ind) -> SnpMajorTile`.

**Hand-rolled whitespace tokenization in two places, inconsistently.** `parse_ancestrymap_geometry` (lines 208–216) uses a `std::vector<std::string_view>` loop, while `read_ancestrymap_snp_major_tile` (lines 1082–1099) uses a fixed `std::array<std::string_view, kAncestrymapFields>` loop. Both do the same thing. Centralize it.

**CRLF stripping is duplicated.** `count_text_records` does its own `pop_back()` at line 42, and the ASCII parsers repeat the same `if (!line.empty() && line.back() == '\r')` incantation. There’s already a `strip_cr` helper at line 85 — use it.

**The API naming is a bit coy.** `read_tile` is the generic-sounding name, but it only handles TGENO individual-major files; the SNP-major path is `read_snp_major_tile`. The header explains it, but the call sites will still be confusing.

**Per-individual seek in `read_tile`.** Lines 493–504 seek and read for every selected individual. That’s fine for M1’s small demo, but it’s O(n_selected) seeks; for larger tiles a sequential gather or mmap would be worth considering.

**The header pulls in four other headers.** `geno_reader.hpp` includes `eigenstrat_format.hpp`, `genotype_tile.hpp`, `ind_reader.hpp`, and `snp_major_tile.hpp`. Several of those could be forward-declared to reduce compile coupling, though for a project this size it’s not a crisis.

**No CUDA-specific footguns here** — this file is pure host C++ as advertised, so the CUDA checklist doesn’t apply to this TU.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted *wrong* code, missing error checks, and algorithms that only pass because the test data is small. This file has none of that. The duplication is a *refactoring* problem, not a slop problem.

## What it actually looks like

This looks like **competent, defensive research engineering by someone who knows the domain and prioritizes correctness over elegance.** The author has clearly been bitten by silent size_t wrap, stale partitions, and format-misclassification before, and they wrote the file so it can’t happen again. The downside is that every lesson got pasted into every reader function instead of being factored out. A senior teammate would say: “Solid, safe code — now let’s DRY it up and cut the comment noise.”

## Verdict

**B+ — ship after deduplicating the selection/reorder boilerplate and centralizing the whitespace tokenizer.** If this is part of a job-application showcase, keep the safety discipline (it’s genuinely impressive), but refactor the repeated blocks so the reviewer sees engineering taste, not just engineering caution.