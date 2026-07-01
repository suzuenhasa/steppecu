# src__app__cmd_qpwave
Files: /home/suzunik/steppe/src/app/cmd_qpwave.cpp, /home/suzunik/steppe/src/app/cmd_qpwave.hpp
Subsystem: app-cli

## Findings

### G4
- [G4.src__app__cmd_qpwave][LOW] cmd_qpwave.cpp:131 — `right_n = static_cast<int>(right_idx.size()) - 1` can underflow to a negative count if `right_idx` is empty. In this path `config.right().empty()` is rejected at line 75 so `resolve_all` yields >=1 index, making this safe in practice; but the subtraction has no local guard and relies on an invariant established 50+ lines earlier. Suggested: assert/comment that `right_idx` is non-empty here (or compute from the already-validated count) so the R0 convention is locally self-evident.

### G8
- [G8.src__app__cmd_qpwave][LOW] cmd_qpwave.cpp:91 — comment cites a fixed source line `test_qpwave_parity.cu:332` for the CudaBackend block; a hardcoded line number in a cross-file reference is a classic stale-comment vector (any edit to that test silently invalidates it). Suggested: drop the `:332` and reference the file/test-name only.

### G9
- [G9.src__app__cmd_qpwave][LOW] cmd_qpwave.cpp:118 — `OutputFormat fmt = OutputFormat::Csv;` is declared non-const then immediately passed by reference to `parse_output_format(..., fmt)` which overwrites it (line 119); correct as-is (it is a true out-param sink), noted only because the `= Csv` initializer is dead (always overwritten on the success path, and the failure path returns). Suggested: leave as-is or drop the initializer; not a real defect.
