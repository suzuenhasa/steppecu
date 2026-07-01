# src__core__qpadm__model_search
Files: /home/suzunik/steppe/src/core/qpadm/model_search.cpp, /home/suzunik/steppe/src/core/qpadm/model_search.hpp
Subsystem: core-qpadm

## Findings

### G3 (dead/unused includes)
- [G3.src__core__qpadm__model_search][LOW] model_search.cpp:20 — `#include "steppe/error.hpp" // Status` is included with the rationale comment "Status", but `Status` (and any other symbol from error.hpp) is never referenced anywhere in the translation unit. Likely a leftover include; either it is genuinely unused or it is only transitively required, in which case the "// Status" annotation is wrong. Suggested: drop the include (and its comment) if compilation still succeeds, or correct the annotation to name the symbol that actually needs it.

### G8 (comments)
- [G8.src__core__qpadm__model_search][LOW] model_search.cpp:20 — the `// Status` end-of-line comment is stale/misleading: it documents a dependency the code does not exercise (no `Status` usage in the file). The include comments on the surrounding lines (Precision @41/117, F2BlockTensor @235/326/340) are accurate, so this one stands out as drift. Suggested: remove or fix the annotation alongside the include decision above.

(Note: the long TODO(multigpu-host-bounce) block at model_search.cpp:168-192 and :283-286 is a deliberate, accurate, dated deferral record consistent with the parked multi-GPU policy — NOT an orphan TODO; not flagged.)
