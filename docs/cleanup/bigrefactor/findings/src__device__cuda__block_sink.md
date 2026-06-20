# Review findings — src__device__cuda__block_sink

Files: /home/suzunik/steppe/src/device/cuda/block_sink.cu, /home/suzunik/steppe/src/device/cuda/block_sink.cuh

## Group 4 — Type & numeric

- [4.7][LOW] block_sink.cuh:59-60, block_sink.cu:107-108,268-269 — `spill_block` takes raw `const double* f2_dev`/`vpair_dev` (DEVICE pointers) while the sink internally manipulates `s.f2.data()` (HOST pinned pointers) of the identical raw `double*` type; nothing at the type level prevents passing a host pointer where a device one is expected (or vice versa). The cudaMemcpyAsync `kind` arg (cudaMemcpyDeviceToHost) is the only guard. Suggested: optionally introduce a thin DevicePtr<double>/HostPtr<double> tag-wrapper at the seam so the wrong space can't compile; non-blocking hygiene only.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

- [3.3][LOW] block_sink.cu:29 — `#include <utility>` is unused in this TU. The only `std::move` is in block_sink.cuh:119 (`DiskSink` ctor), and that header already has `std::move` available transitively via pinned_buffer.cuh (which includes `<utility>` at its line 45); nothing in block_sink.cu itself uses any `<utility>` symbol. Suggested: drop the `<utility>` include from the .cu, and (separately/IWYU) add `#include <utility>` directly to block_sink.cuh which uses `std::move` but only gets it transitively.
