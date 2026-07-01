# src__core__internal__log
Files: /home/suzunik/steppe/src/core/internal/log.hpp
Subsystem: core-stats

## Findings

### G8
- [G8.comment][LOW] log.hpp:13-16 — Header comment references implementation detail "one NDEBUG branch evaluating its arg and two not" about the three macros this facade replaced; this is historical/migration context for code that no longer exists in this unit. Harmless rationale but at risk of becoming stale once the replaced macros are fully gone from the tree. Suggested: trim to the behavioral contract and the §2 DRY rationale, dropping the per-placeholder eval detail.
