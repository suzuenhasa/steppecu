# src__app__cmd_qpgraph
Files: /home/suzunik/steppe/src/app/cmd_qpgraph.cpp, /home/suzunik/steppe/src/app/cmd_qpgraph.hpp
Subsystem: app-cli

## Findings

### G4
- [G4.type][LOW] cmd_qpgraph.cpp:67-68 — `std::tolower(c)` is called with a plain `char c`; on signed-char platforms a byte > 127 sign-extends to a negative `int`, which is UB for the `<cctype>` functions (only EOF and `unsigned char` values are valid). Population labels are ASCII in practice so this is latent, but the header-row lowercasing is fed un-validated file content. Suggested: cast through `unsigned char` before `std::tolower` (`std::tolower(static_cast<unsigned char>(c))`).

### G7
- [G7.dup][LOW] cmd_qpgraph.cpp:163-179 vs 278-292 — the device-fit dispatch block (build_resources -> empty-gpus guard -> `gpus.front().device_id` -> `upload_f2_blocks_to_device` -> run, all inside one `try`/`catch (const std::exception&)`) is duplicated across `run_qpgraph_command` and `run_qpgraph_search_command`, differing only by the command-name prefix in the error strings and the `run_qpgraph` vs `run_qpgraph_search` call. Suggested: factor a small helper that yields `(Resources, device_id, DeviceF2Blocks)` or takes the run-callable, parameterized on the command-name prefix.
- [G7.dup][LOW] cmd_qpgraph.cpp:190-208 vs 300-317 — the output-tail boilerplate (`parse_output_format` guard, `out_file().empty()` branch, ofstream-open guard, `emit`/`emit_search` to `std::cout` or the file) is copy-pasted between the two commands, differing only by the emit function, the command-name prefix, and the result type. Suggested: extract a templated/lambda-driven `emit_to_destination(config, prefix, emitFn)` helper.

### G8
- [G8.comment][LOW] cmd_qpgraph.cpp:56 — the comment "replace commas with spaces so split is uniform" restates the immediately-following one-line loop verbatim; it adds no rationale beyond the code. Minor; harmless. Suggested: drop or replace with the *why* (admixtools R `write.csv` emits comma-separated edge lists).
