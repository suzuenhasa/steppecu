# src__app__result_emit
Files: /home/suzunik/steppe/src/app/result_emit.cpp, /home/suzunik/steppe/src/app/result_emit.hpp
Subsystem: app-cli

## Findings

### G6
- [G6.src__app__result_emit][MED] result_emit.cpp:282,298 — `emit_rotation_csv` writes the column header `"f4rank"` but the data value written for that column is `r.est_rank` (line 298), not `r.f4rank`. The column name is misleading: it claims f4rank but carries est_rank. (The non-rotation `emit_csv` summary writes f4rank and est_rank as two distinct columns, 122-126, so this is an inconsistency within the file.) Suggested: rename the rotation column to `"est_rank"` (or write `r.f4rank` if f4rank is truly intended), and align the JSON field below.
- [G6.src__app__result_emit][MED] result_emit.cpp:339 — same in `emit_rotation_json`: the JSON field key is `"f4rank"` but the emitted value is `r.est_rank`. Misleading key vs value; should match the CSV decision above. Suggested: rename the key to `est_rank` or emit `r.f4rank` consistently.

### G7
- [G7.src__app__result_emit][MED] result_emit.cpp:129-131, 200-203, 388-390, 449-452, 503-506, 552-555, 605-608 — the precision-tag mapping ternary (`EmulatedFp64 ? "emu" : Tf32 ? "tf32" : "fp64"`) is copy-pasted verbatim in 7 places. A drift bug if a new Precision::Kind is added (one site updated, others not). Suggested: extract a file-static `const char* precision_str(Precision::Kind)` helper (sibling to status_str) and call it everywhere.
- [G7.src__app__result_emit][MED] result_emit.cpp:209-224 vs 409-424 — the three JSON array lambdas (`emit_int_arr`, `emit_dbl_arr`, `emit_dofdiff_arr`) are defined identically inside both `emit_json` and `emit_qpwave_json`. Suggested: lift to file-static free functions taking `std::ostream&` so both emitters share one definition (the file comment at 351-354 already claims this reuse — currently the lambdas are duplicated, not shared).
- [G7.src__app__result_emit][MED] result_emit.cpp:138-144 vs 363-369 — the rankdrop CSV loop body (the 7-column f4rank/dof/chisq/p/dofdiff/chisqdiff/p_nested write) is duplicated byte-for-byte between `emit_csv` and `emit_qpwave_csv`. Suggested: extract `emit_rankdrop_csv(os, r.rankdrop_*, sep)` and call from both.
- [G7.src__app__result_emit][LOW] result_emit.cpp:472-474, 488-490, 522-524, 538-540, 573-575, 590-592 — the `at` index-guarded label-accessor lambda is re-defined identically in all six standalone-stat emitters (f4/f3/f4ratio, csv+json). Suggested: a single file-static `label_at(const std::vector<std::string>&, std::size_t)` helper.
- [G7.src__app__result_emit][LOW] result_emit.cpp:166-189 vs 324-335 — the JSON weights/se/z parallel-array emission loop (the `(i ? ", " : "")` + `have_se && i < r.se.size() ? json_double : "null"` pattern) is duplicated between `emit_json` and `emit_rotation_json`. Minor; the surrounding object shapes differ, so extraction is optional. Suggested: optionally factor a `emit_se_z_arrays` helper if churn warrants.

### G8
- [G8.src__app__result_emit][LOW] result_emit.cpp:382 — comment cites "cli-bindings.md §357" as the schema source for the qpwave summary columns; the other section comments cite "§4.4"/"§4.1". A section number 357 is suspicious (likely a stale/typo cross-reference, not a real spec section). Suggested: verify and correct the §-reference.

No issues found for groups: G2, G3, G4, G5, G9, G10.
