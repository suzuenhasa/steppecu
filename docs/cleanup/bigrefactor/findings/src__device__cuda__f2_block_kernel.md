# Review findings — src__device__cuda__f2_block_kernel

Files: /home/suzunik/steppe/src/device/cuda/f2_block_kernel.cu, /home/suzunik/steppe/src/device/cuda/f2_block_kernel.cuh

## Group 4 — Type & numeric

- [4.7][LOW] f2_block_kernel.cuh:38-106 / f2_block_kernel.cu:102-108,151-155,307-309,375-376 — All kernel params and launch-wrapper signatures use raw `const double*`/`double*` with no host-vs-device space distinction; a caller could pass a host pointer where a device pointer is required and the compiler would not catch it (the wrappers all take device pointers `dQ_raw`/`dG`/`dR`/`dF2`). Suggested: this is the project-wide device seam pattern, not a local defect — only a candidate for a thin device-pointer wrapper type if the codebase adopts one; no action needed in this TU.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.
