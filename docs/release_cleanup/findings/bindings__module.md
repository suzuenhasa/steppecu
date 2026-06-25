# bindings__module
Files: /home/suzunik/steppe/bindings/module.cpp
Subsystem: bindings

## Findings

### G5
- [G5.bindings__module][LOW] module.cpp:856,939 — the Morgans->centimorgans conversion `blgsize * 100.0` is duplicated as a bare literal `100.0` in `run_extract_f2_py` and `run_qpfstats_py`. Same magic constant in two places = drift risk if the unit convention ever changes. Suggested: a single named helper/constexpr `morgans_to_cm(blgsize)` (or `kCmPerMorgan`) reused at both meta-fill sites.

### G7
- [G7.bindings__module][MED] module.cpp:237-245,266-274,294-302,322-330 — the indices->names lambda is copy-pasted VERBATIM four times across `f4_to_dict`, `dstat_to_dict`, `f3_to_dict`, `f4ratio_to_dict` (identical bounds-checked `pops[i]`-or-empty resolve). Suggested: one free function `names_of(const std::vector<int>&, const std::vector<std::string>& pops)` called by all four emitters.
- [G7.bindings__module][MED] module.cpp:570-580,611-621,971-981,1010-1020,572-578 — the "resolve a fixed-arity tuple of pop names to a `std::array<int,N>`, throwing key_error on miss" loop is repeated for N=4 (run_f4_py, run_qpdstat_py), N=3 (run_f3_py), N=5 (run_f4ratio_py), and N=4-via-quadruples (run_dstat_py 685-693). Each is the same body differing only by the arity constant. Suggested: a templated `resolve_tuple<N>(resolver, q, what)` helper.
- [G7.bindings__module][MED] module.cpp:390-398,419-427,472-479,504-511,584-592,625-633,985-993,1024-1032,1074-1082 — the device-fit prologue/epilogue (`ensure_resources` -> `gpus.front().device_id` -> `upload_f2_blocks_to_device(h.tensor, device_id)` -> run-the-seam -> `catch(std::exception) raise_value("device error: ")`) is hand-rolled identically in ~9 fit entries. Pure boilerplate around the one differing `run_*` call. Suggested: a small `with_device_f2(h, lambda(dev_f2, resources))` wrapper that owns the try/catch + upload.
- [G7.bindings__module][LOW] module.cpp:803-816,902-910 — the precision-string parse (`"fp64"/"native"`->Fp64, `"emulated_fp64"/"emu"`->EmulatedFp64, `"tf32"`->Tf32, else raise) is duplicated in `run_extract_f2_py` and `run_qpfstats_py` with only the error-message tool name differing. Suggested: a `parse_precision(std::optional<std::string>, const char* tool)` helper returning `steppe::Precision`.
- [G7.bindings__module][LOW] module.cpp:700-704,734-738,837-841,919-922 — the "no CUDA device available (steppe is a GPU product; a CUDA-capable GPU is required)" fail-fast message is hand-duplicated in four entries (and `ensure_resources` 104-108 has its own copy). Easy to let these drift. Suggested: a single `raise_no_device()` helper / shared constant.

### G8
- [G8.bindings__module][LOW] module.cpp:1117 — module docstring still says "GPU qpAdm/qpWave bindings (M(py-1); marshalling only)", but the TU now also binds qpgraph, qpgraph-search, f3, f4, f4ratio, qpdstat, dstat, dates, extract-f2, qpfstats (M(py-2)+). Stale scope description. Suggested: broaden the doc text to the actual surface.

### G3
- [G3.bindings__module][LOW] module.cpp:1096-1099 — `f2_to_numpy` recomputes the flat element count inline (`P * P * nb_count`) instead of reusing the existing `F2BlockTensor::size()` accessor (fstats.hpp:73-77) which performs the identical widened multiply; the function also does not assert `src.size() == n` before the copy loop (1103) even though the contract assumes it. Both are internally controlled so no live bug, but the inline recompute is a maintenance smell. Suggested: use `h.tensor.size()` and (optionally) a debug-time size check.
