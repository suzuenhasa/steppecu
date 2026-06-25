# src__device__resources
Files: /home/suzunik/steppe/src/device/resources.hpp, /home/suzunik/steppe/src/device/resources.cpp
Subsystem: backend

## Findings

### G7
- [G7.src__device__resources][LOW] resources.cpp:56,80,87,106 — the fail-fast prefix string `"steppe::device::build_resources: "` is copy-pasted across all four `std::runtime_error` throws (two in `validate_device_order`, one each in `resolve_device_order` and `build_resources`); a rename of the function leaves the messages drifting from the symbol. Suggested: a single `static constexpr const char* kBuildErrPrefix` (or a small message helper) concatenated into each throw.

### G8
- [G8.src__device__resources][LOW] resources.cpp:55-59 — the comment block above the `visible < 1` guard in `resolve_device_order` describes the auto-enumerate rationale, but the message it throws ("auto-enumeration found no visible CUDA device") is only reachable on the EMPTY-`config.devices` path; the explicit-list path returns at line 50 before this guard, and an explicit list on a zero-visible box is instead caught by `validate_device_order` at line 79. The behavior is correct, but the comment ("no <cuda_runtime.h> here ... single CUDA-free count query") restates surrounding design rather than flagging that this guard only covers the auto path. Minor — Suggested: note that the explicit-list-on-zero-device case is handled downstream by validate_device_order, not here.
