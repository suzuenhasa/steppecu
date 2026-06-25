# src__device__cuda__decode_af_kernel
Files: /home/suzunik/steppe/src/device/cuda/decode_af_kernel.cu, /home/suzunik/steppe/src/device/cuda/decode_af_kernel.cuh
Subsystem: device-cuda

## Findings

### G3
- [G3.dead][LOW] decode_af_kernel.cu:60 — `using core::finalize_af;` is imported but never referenced in the TU; only `finalize_af_counts` (line 142) is used (the per-sample-ploidy finalize). Dead using-declaration. Suggested: drop the `using core::finalize_af;` line.

### G8
- [G8.stale][LOW] decode_af_kernel.cu:52 — the include comment names the primitives as "`accumulate_genotype, finalize_af`", but the code actually calls the per-sample-ploidy variants `accumulate_genotype_ploidy` (line 140) and `finalize_af_counts` (line 142); `finalize_af` is not called. Comment lists the wrong (non-suffixed) symbols. Suggested: update the trailing comment to `accumulate_genotype_ploidy, finalize_af_counts`.
- [G8.stale][LOW] decode_af_kernel.cu:73 — the kernel doc-comment says "writing Q/V/N via the SHARED finalize" and "via the SHARED accumulate_genotype", referring to the base names; the implementation uses the `_ploidy`/`_counts` variants (lines 140, 142). Minor naming drift in the rationale comment. Suggested: name the actual `_ploidy`/`_counts` primitives so the comment tracks the code.
