# src__core__internal__host_device
Files: /home/suzunik/steppe/src/core/internal/host_device.hpp
Subsystem: core-stats

## Findings

### G8
- [G8.task][LOW] host_device.hpp:81 — The NDEBUG branch of `STEPPE_ASSERT(cond, msg)` expands to `((void)0)`, which does NOT reference `cond` or `msg`. The header comment (lines 76-78) markets the macro as the carrier for "enforce the documented precondition on hot paths," i.e. a call site is likely to be the ONLY use of a given variable. In a release build that variable then becomes unused and can warn (or fail under `-Werror`), which is surprising given the comment's framing. Standard `assert` itself has the same property, but a project-owned wrapper can do better. Suggested: expand the NDEBUG branch to `((void)sizeof(cond), (void)sizeof(msg), (void)0)` (or document the unused-variable caveat alongside the side-effect caveat at lines 76-78 so the contract is explicit).
