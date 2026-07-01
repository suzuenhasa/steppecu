# src__device__cpu__cpu_backend
Files: /home/suzunik/steppe/src/device/cpu/cpu_backend.cpp
Subsystem: device-cpu

## Findings

### G3 (dead/unused/computed-but-unread)
- [G3][MED] cpu_backend.cpp:1843-1855 — `xmat_from_loo_block` is a private static helper that is defined but never called anywhere in the TU; `gls_weights_loo_batched` (the only S7 LOO consumer) builds `rep.x_total` inline (lines 1718-1721) instead. Dead code. Suggested: delete the helper (and its doc comment) or wire it into `gls_weights_loo_batched` if it was meant to replace the inline copy.
- [G3][MED] cpu_backend.cpp:615 — in `dates_curve`, the per-sample scratch vector `res` is allocated (sized `M`) but never written or read; line 665 reads a local `const double r`, not `res`. The allocation is wasted (M doubles per call). Suggested: drop `res` from the declaration.
- [G3][LOW] cpu_backend.cpp:622,635 — in `dates_curve`, `pl` is read from `target_ploidy` (line 622) then immediately discarded with `(void)pl` (line 635); the dosage is `g/2.0` flat regardless of ploidy. The parameter `target_ploidy` is therefore effectively unused in the body. Rationale is given (lines 634-635: DATES uses /2.0 flat), so this is documented-but-dead. Suggested: if ploidy genuinely never affects the DATES dosage, drop the local read; keep the param only if the interface contract requires it.

### G8 (comments)
- [G8][LOW] cpu_backend.cpp:646 — stale working-note comment ("stash w1,w2 in res temporarily? No — recompute below; store src idx.") referencing the now-dead `res` vector; it reads as a leftover authoring deliberation, not documentation. Suggested: remove with the `res` allocation (G3 above).

No issues found for: G2 (no CUDA APIs — pure host TU), G4 (index widening to std::size_t is applied consistently at the P²·M / slab / m_sz seams; `quadruples.size()/4` and similar are bounded by the f-stat sweep cap, no countdown loops), G5 (literals are named where they matter — kPairwiseBaseCase, kMinJackknifeBlocks, kGesvdjMaxDim is single-sourced in config.hpp; the splitmix/optimizer constants 1e-4/0.5/8/1e-30 are local algorithm tuning), G6 (naming is consistent; m_sz vs M aliasing is explicitly documented at the model-vs-SNP-axis seams), G7 (duplication already extracted — f2_pair_over_range, als_ridge_solve, ridge_diagonal, als_xvec, survivor_blocks, row_major_at are the single-source bodies the near-identical paths share), G9 (config knobs surfaced; no positional-boolean opacity — the tot_mode/compute_p calls use /*name=*/ comments), G10 (locals initialized at declaration; output vectors zero-assigned before the fill loops).
