# include__steppe__f4ratio
Files: /home/suzunik/steppe/include/steppe/f4ratio.hpp, /home/suzunik/steppe/src/core/qpadm/f4ratio.cpp
Subsystem: core-qpadm

## Findings

### G8
- [G8.include__steppe__f4ratio][MED] src/core/qpadm/f4ratio.cpp:13-38 — STALE file-header comment. The PIPELINE block (lines 13-38), especially step 2 (lines 22-31), describes a HOST-PURE per-tuple ratio jackknife loop ("for each tuple k ... var = Σ(xtau_b^2)/nb; ... write the ratio xtau explicitly") as if it lives in this .cpp. It no longer does: the body (lines 134-143) delegates the entire ratio jackknife to `be.f4ratio_blocks_jackknife(...)`, and the in-function comment at lines 71-77 explicitly states "is NO LONGER a host loop here." The header still documents the removed implementation, contradicting the code below it. Suggested: trim the step-2 math derivation in the header to a one-line pointer to the `f4ratio_blocks_jackknife`/`RatioBlockJackknife` backend seam (move the detailed jack_mat_stats derivation to where the kernel/oracle actually computes it), so the header describes current behavior.
- [G8.include__steppe__f4ratio][LOW] src/core/qpadm/f4ratio.cpp:32-33 — the "Near-zero denom (AT2 setmiss thresh=1e-6) ... if abs(x_blocks[den_k+m*b])<1e-6 treat that block as MISSING (skip ...)" comment describes a per-block skip performed inside the now-delegated backend seam, not in this TU; reads as describing local code that is gone. Suggested: fold into the same header trim as above, attributing the setmiss skip to the seam.
