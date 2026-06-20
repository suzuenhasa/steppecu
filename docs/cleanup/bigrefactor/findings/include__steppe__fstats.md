# Review findings — include__steppe__fstats

Files: /home/suzunik/steppe/include/steppe/fstats.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

<!--
Notes (not findings):
- 4.1: `f2`/`vpair` are std::vector<double> (lines 52, 60) — FP64-by-design, correct. N/A.
- 4.2/4.6: The flat index `i + P·j + P·P·b` (lines 36-38, 48, 56) is documentation only; no
  int index arithmetic is computed in this header. The sole computed quantity, size() (lines
  74-77), casts P, P, n_block to std::size_t BEFORE the multiply, so P·P·n_block (~10^10) does
  not overflow int. This is the correct widening pattern; the overflow-prone flat-index math
  lives in consumers (src/device, src/core), not here.
- 4.3: No cudaMalloc/new/raw allocation — std::vector handles `* sizeof(T)`. N/A.
- 4.4/4.5: No loops in this header. N/A.
- 4.7: No raw pointers; intentionally CUDA-free host storage (std::vector). N/A.
- P, n_block as int (lines 68, 71) and block_sizes as std::vector<int> (line 65) fit comfortably
  at scale (P<=2500, n_block<=757); not a finding.
-->

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

<!--
Notes (not findings):
- This is the installed, intentionally CUDA-FREE public header (lines 12-15, "No CUDA header
  here"). It includes only <cstddef> and <vector> (lines 27-28) and defines a plain host-only
  POD struct (F2BlockTensor, lines 47-78). No device code, no CUDA runtime/driver API.
- 2.1 Dropped archs: no build flags, no CMake, no SASS/PTX arch lists in this header. N/A.
- 2.2 Texture/surface references: no texture<...>/surface<...>/cudaBindTexture*; no CUDA at all. N/A.
- 2.3 Non-_sync warp intrinsics: no warp intrinsics / no device code. N/A.
- 2.4 cudaThreadSynchronize: no CUDA runtime sync calls. N/A.
-->

## Group 3 — Dead / commented-out code

No Group 3 issues found.

<!--
Notes (not findings):
- 3.1: All comments in this header (lines 1-23 file header, 32-46 struct doc, 48-77 member docs)
  are architecture/doc comments, NOT commented-out code "kept just in case". None are dead.
- 3.2: No #if 0, no code after return/break, no unreachable code. The single function size()
  (lines 74-77) is one unconditional return. N/A.
- 3.3: Both includes are used — <cstddef> (line 27) by std::size_t in size() (lines 74-76);
  <vector> (line 28) by std::vector<double>/<int> (lines 52, 60, 65). No unused vars/params/
  helpers; all struct members (f2, vpair, block_sizes, P, n_block) are the M4 public-API
  deliverable. vpair is explicitly RETAINED (lines 54-58) as the S4 jackknife weight, so it is
  not dead even though the diagonal of f2 is documented as never consumed downstream.
- 3.4: Nothing is computed-but-unread in this header; it is a POD struct + one convenience
  accessor. The "computed but unread" diagonal note (lines 44-46) concerns downstream consumers,
  not storage here. N/A.
-->

