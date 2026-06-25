# src__device__cuda__detect_ploidy_kernel
Files: /home/suzunik/steppe/src/device/cuda/detect_ploidy_kernel.cu, /home/suzunik/steppe/src/device/cuda/detect_ploidy_kernel.cuh
Subsystem: device-cuda

## Findings

### G3
- [G3.src__device__cuda__detect_ploidy_kernel][LOW] detect_ploidy_kernel.cu:87-91 — `if (window == 0) { ... }` has an empty body (comment only); the kernel is launched unconditionally on the next line regardless, so the `if` statement itself is dead/no-op code. The explanatory comment is correct (window=0 is handled by the kernel's loop never executing), but the surrounding `if` adds no behavior. Suggested: drop the empty `if` and keep the explanation as a plain comment above the launch.
