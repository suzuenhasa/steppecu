# src__device__backend_factory
Files: /home/suzunik/steppe/src/device/backend_factory.hpp
Subsystem: backend

## Findings

### G8
- [G8.comments][LOW] backend_factory.hpp:17-21 — The header comment states "`make_cpu_backend` and the `CpuBackend` class were MOVED here from `steppe::core`" and "the two implementations of ONE interface now share ONE namespace", but this header declares only the three free functions — `CpuBackend` is NOT declared in this file. The prose describing a class living "here" is stale/misleading relative to the file's actual contents (only `make_cpu_backend()` is declared; the class lives in cpu_backend's TU). Suggested: reword to say the factory is declared here while `CpuBackend` itself remains a private implementation detail of its TU, or remove the class-placement claim.
- [G8.comments][LOW] backend_factory.hpp:19 — Orphan/inline "TODO §A" reference embedded in prose ("the old `steppe::core` placement was a namespace/layer mismatch — TODO §A") with no actionable description of what §A is or what remains to be done. Suggested: resolve or convert to a concrete tracked TODO, or drop the marker if the move is complete.
