# src__io__plink_reader
Files: /home/suzunik/steppe/src/io/plink_reader.cpp, /home/suzunik/steppe/src/io/plink_reader.hpp
Subsystem: io

## Findings

### G5
- [G5.io][LOW] plink_reader.cpp:245 — the phenotype "ignore" sentinels `"9"`, `"-9"`, `"0"` are bare inline string literals, whereas the Control/Case sentinels are named constants (`kFamControlPheno`/`kFamCasePheno`, lines 133-134). The same AT2 mcio.c:1180-1205 convention is thus half-named, half-inline; the three ignore literals are an unnamed duplicated set (also re-spelled in the header prose and the cpp banner comment). Suggested: surface the ignore set as a named constant (e.g. `kFamIgnorePhenos`) beside the case/control constants for consistency and single-sourcing.

### G6
- [G6.io][LOW] plink_reader.cpp:229,262 — the variable `row` doubles as both the running .bed individual-record index AND, after the loop, the total individual count stored into `part.n_individuals_total` (line 262). The name reads as an index, not a count, at the assignment site. Minor; read_ind has the identical pattern, so this is a cross-reader convention rather than a local inconsistency. Suggested: none required (matches read_ind); optionally a one-word note that post-loop `row == total rows seen`.

### G8
- [G8.io][LOW] plink_reader.cpp:223-235 — the block comment (lines 223-228) asserts the universal PLINK invariant ".fam row i == .bed individual i, 1:1" and that "`row` increments for every well-formed line", but the very next code path (line 235) `continue`s on a short (<6-field) line WITHOUT incrementing `row`, silently breaking that 1:1 mapping for a malformed line. This contradicts read_bim's sibling contract in the SAME unit, which fail-fasts on a short record (cpp:173-178) precisely to protect axis alignment. The comment rationale ("a real .fam always has 6 columns") concedes the skip is a tolerance, but the strongly-stated 1:1 invariant above it is then not actually enforced. Suggested: reconcile the comment with the behavior (either note the short-line skip as an explicit tolerance that assumes no interior malformed lines, or align the malformed-line handling with read_bim's fail-fast).

### G9
- [G9.io][LOW] plink_reader.cpp:248-249 — the `pheno == kFamControlPheno -> "Control"` / `kFamCasePheno -> "Case"` remap hardcodes the output label strings `"Control"`/`"Case"` inline. These are config-like format conventions (they must match what AT2 emits for col6 "1"/"2") but are bare string literals at the logic site. Suggested: name them (e.g. `kControlLabel`/`kCaseLabel`) alongside the pheno-token constants so the token→label pairing is visible in one place.
