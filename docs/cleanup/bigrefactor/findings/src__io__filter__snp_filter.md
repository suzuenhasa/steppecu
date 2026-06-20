# Review findings — src__io__filter__snp_filter

Files: /home/suzunik/steppe/src/io/filter/snp_filter.cpp, /home/suzunik/steppe/src/io/filter/snp_filter.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (why clean), for the record:
- 4.2/4.6 (index width / overflow): the only global index into the P*M arrays is
  snp_filter.cpp:87-88 `off = (size_t)p + (size_t)P * (size_t)s` — every operand is
  widened to std::size_t BEFORE the multiply and add, so P*s and the sum are 64-bit.
  At scale (P=2500, M=584131 ⇒ P*M ≈ 1.5e9, larger tiles can exceed 2^31) this is
  correctly computed in size_t. SNP loop counters `s`/`M` are `long`; `total_indiv`,
  `Mu`, `si` are `std::size_t`. No 32-bit index/offset into a >2^31 array.
- 4.1 (float-vs-double): all arithmetic is FP64 by design (pooled AF, missing_frac,
  npn/ploidy_d) — intentional, parity-load-bearing; no wrong narrowing.
- 4.3 (allocation sizing): only std::vector element-count constructors
  (snp_filter.cpp:21, 113) — no cudaMalloc/new, no byte-vs-element confusion.
- 4.4 (unsigned countdown): no reverse loops.
- 4.5 (signed/unsigned compares): size_t-vs-size_t comparisons use explicit casts
  (snp_filter.cpp:51, 126, 132, 137); loop bounds are same-signedness (long<long,
  int<int).
- 4.7 (host/device pointer typing): raw `const double*` q/v/n (snp_filter.hpp:42-44)
  are host memory by contract in this host-pure `io`-leaf TU (is_cuda=false); no
  device memory is touched, so no host/device space-confusion risk here.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (why clean), for the record:
- This is a host-pure `io`-leaf TU (is_cuda=false): plain C++20, no CUDA. No
  __global__/__device__ kernels, no nvcc-compiled code. The only "__host__
  __device__" / "cuda" / "geno" string hits (snp_filter.hpp:106, snp_filter.cpp:48,61)
  are in comments/prose (a reference to the decode_af device primitive pattern,
  and the substrings in "geno"/"surface"), not code.
- 2.1 (dropped archs Maxwell/Pascal/Volta, sm_50/60/70): no arch flags, no CMake
  arch lists, no compute_*/sm_* in this unit (CMake is out of scope of these files).
- 2.2 (texture/surface references removed in CUDA 12): no texture<...>, surface<...>,
  cudaBindTexture*, or texture-object API anywhere in the unit.
- 2.3 (non-_sync warp intrinsics): no __shfl/__ballot/__any/__all/__activemask —
  no warp-level code at all (host TU).
- 2.4 (cudaThreadSynchronize -> cudaDeviceSynchronize): no CUDA runtime sync calls
  (no cudaThreadSynchronize, no cudaDeviceSynchronize) — no CUDA runtime use here.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (why clean), for the record:
- 3.1 (commented-out blocks "just in case"): none. Every comment in both files is
  documentation/rationale prose (file headers snp_filter.cpp:1-6 / .hpp:1-21, the
  cleanup-reference rationale at .cpp:24-67/120-141, the doc comments .hpp:35-170) —
  no commented-out statements or disabled code.
- 3.2 (unreachable code): no `#if 0`. No code after a return/break/throw. The early
  returns (snp_filter.cpp:22 `if (P<=0||M<=0) return out;`, :114 `if (M<=0) return
  keep;`) are guards; every `return false`/`throw` is on a conditional guard path,
  all reachable.
- 3.3 (unused symbols): all includes are used — .cpp `<cstddef>` (std::size_t),
  `<stdexcept>` (std::invalid_argument), `<string>` (std::to_string),
  filter_decision.hpp (folded_maf); .hpp `<cstddef>`, `<vector>`,
  filter_decision.hpp (predicates in snp_keep_decision), include_exclude.hpp
  (SnpMembership), snp_reader.hpp (SnpTable), config.hpp (FilterConfig). No unused
  params/locals/helpers. The struct field DecodedTileSummaryInput::v (.hpp:43) is
  documented as intentionally unread (N==0 ⇔ V==0 ⇔ missing by the decode contract,
  .hpp:98-99, .cpp:62-63) — it is a public INPUT-CONTRACT field mirroring
  DecodeResult.q/.v/.n, not a dead local; intentional, not flagged.
- 3.4 (computed but unread): every assignment is read. total_indiv -> total_indiv_d
  -> missing_frac; ploidy_d -> numerator; pooled_ref_count/pooled_allele_count/
  nonmissing_indiv all feed sm.* outputs; mem_noop read twice (:137,:154); chrom &
  membership_ok feed snp_keep_decision (:157). No write-only variable.
-->
