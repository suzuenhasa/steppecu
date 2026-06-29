# Synthesized Code Review Report — Wave 1

## 1. Executive Summary

This is **solid research-engineering code, not slop**. Across the 16 files the author clearly understands the genomics algorithms, the CUDA porting pitfalls, and the value of architectural seams. The good parts are genuinely good: RAII ownership of CUDA handles and buffers, consistent `STEPPE_CUDA_CHECK` error wrapping, a bit-exact parity obsession that ties GPU kernels back to the CPU oracle, and a clean CUDA-free boundary between `core`/`app` and the device backend. Most files land in the **B+** range — correct on the happy path and maintainable by someone who reads the comments.

The most common senior-dev flags are **integer-type sloppiness**, **over-commenting**, and **weak error-handling taxonomy**. Several CUDA files mix `long`, `size_t`, `int`, and `long long` in index math and cuBLAS calls in ways that pass on Linux LP64 but are latent portability bugs. Comments are often dense, ALL-CAPS, and salted with cleanup-ticket references that will age poorly. I/O and config error paths frequently misuse `Status::InvalidConfig` for filesystem failures, silently drop errors, or route warnings only through debug sinks. Before showing this to senior developers, you want to tighten the portability and failure-mode warts and give the prose a serious edit.

## 2. Cross-Cutting Themes

- **Integer-type inconsistency.** This is the single most repeated issue. CUDA files mix `long`, `size_t`, `int`, and `long long` without a project-wide width policy. In `f2_blocks_kernel.cu` the gather kernel uses `long` for strides (`const long Pl = static_cast<long>(P);` line 89), the assemble kernel uses `size_t` (`const size_t Pz = static_cast<size_t>(P);` line 168), and `run_f2_gemms_group` uses `long` for cuBLAS strides (`const long Psp = static_cast<long>(P) * s_pad;` line 268) even though `cublasGemmStridedBatchedEx` takes `long long`. `qpfstats_kernel.cu` casts a `long` block count to `unsigned` for the launch grid (lines 56–57). `fstats.hpp` uses `int P`/`int n_block` and `std::vector<int> block_sizes`. `config.hpp` uses `unsigned long long`, `std::size_t`, `unsigned`, and `int` for counts in adjacent constants. Pick a width (prefer `std::int64_t` for signed indices, `std::size_t` for sizes) and use it everywhere.

- **Comment bloat and stale/ticket-laden prose.** Many files have more comment lines than code lines, and the comments cite internal roadmap/cleanup tickets that mean nothing to an outside reviewer. `f2_block_kernel.cu` opens with a 30-line "PRECISION POLICY (MEASURED on real AADR... this is the law)" manifesto (lines 17–50). `config.hpp` has "PRECISION POLICY IS THE LAW (MEASURED... never on synthetic data)" (line 16) and references like "cleanup X-6/B2". `cli_parse.cpp` still carries a stale scaffolding comment that `qpwave` "remains a scaffold no-op" (lines 354–356) even though it is fully wired a few pages later. A senior reviewer will assume dense prose is compensating for unclear code.

- **Error-handling pattern inconsistencies.** The project has a typed `Status` but uses it sloppily. `f2_dir_writer.cpp` returns `Status::InvalidConfig` for directory-creation and file-write failures (lines 407–409). `result_emit.cpp` never checks `std::ostream` state after writes. `config_builder.cpp` declares `build() const` but mutates `error_message_` inside it (lines 209–212), making the `const` contract a lie and the class thread-unsafe. `f2_block_kernel.cu` makes a capability-downgrade warning "observable" but routes it through a debug-only `STEPPE_LOG_WARN` sink, so release builds drop it silently. Decide whether an error is "invalid config," "I/O error," or "warning," and make the contract honest.

- **Hand-rolled serialization and duplicated output logic.** `result_emit.cpp` and `f2_dir_writer.cpp` both build JSON/CSV by concatenating strings manually. `result_emit.cpp` has no validation that parallel arrays (`r.se`, `r.z`, `r.p`, etc.) are the same length; it just indexes them. `f2_dir_writer.cpp` has an incomplete JSON escaper that only handles `\n`, `\r`, `\t` and leaves other control characters unescaped (lines 334–348), and it writes `meta.json` in text mode on Windows so embedded `\n` string values will get CRLF-mangled. This is exactly the kind of thing that breaks golden-file tests when someone adds a field with a special character.

- **CUDA seam fragility.** Several files rely on "the caller already set the stream" without runtime guards (`f2_blocks_kernel.cu` lines 255–263). `DeviceBuffer` allocations in `qpgraph_fit_kernels.cu` are constructed without a stream argument (lines 485–486), which can force synchronous default-stream allocation and defeat async intent. `qpfstats_kernel.cu` uses a global `atomicAdd` for NaN counting (lines 34–36) that will serialize warps when NaNs cluster. `qpadm_fit_kernels.cu` launches single-thread `<<<1,1>>>` "kernels" for scalar work and allocates large compile-time local arrays that silently scale register/local-memory pressure. These are correct but not performance-review-ready.

- **Duplication and copy-paste drift.** `cuda_backend.cu` is a 5,679-line monolith that repeats the CUB scan/select/gather choreography in `decode_af_compact_autosome` and `decode_af_compact_filter`. `device_f2_blocks.cu` defines a `DeviceGuard` inline twice (lines 56–63 and 92–99) and claims it matches `p2p_combine.cu`'s guard, but it does not — the latter has deleted copies/moves and `STEPPE_CUDA_WARN` in the destructor. `f2_blocks_kernel.cu` repeats integer-width choices across kernels instead of using one stride type. Centralize shared helpers.

- **Numerical sentinels and edge cases.** `qpgraph_fit_kernels.cu` overloads `1e30` for "NNLS failed," "LU singular," and "topology too big" (lines 279, 281, 333). `qpadm_fit_kernels.cu` uses `(0.0 / 0.0)` as a NaN sentinel in multiple places (lines 1654, 1657, etc.). `qpgraph_fit_kernels.cu` tests for singular LU with an exact-zero pivot (`if (best == 0.0) return false;` line 57). `solve_constrained_weights` in `qpadm_fit_kernels.cu` normalizes by a sum that could be zero (lines 470–471). These are silent-wrong-result bugs waiting for bad data.

## 3. Per-File Verdict Table

| File | Grade | Key concern |
|------|-------|-------------|
| `src/device/cuda/f2_blocks_kernel.cu` | B+ | Mixed `long`/`size_t`/`long long` in cuBLAS index math; fix before Windows/large-P builds. |
| `src/device/cuda/f2_block_kernel.cu` | B+ | Verbose manifesto comments; deprecated `ATOMIC_FLAG_INIT`; NDEBUG-silent downgrade warning. |
| `src/device/cuda/qpadm_fit_kernels.cu` | B+ | 2,209-line monolith; single-thread kernels; `f2_block_keep_kernel` is a block-serial footgun. |
| `src/device/cuda/qpgraph_fit_kernels.cu` | B | `1e30` overloaded sentinel; `DeviceBuffer` lacks stream; suboptimal 64-thread launch config. |
| `src/device/cuda/qpfstats_kernel.cu` | B+ | Global `atomicAdd` serializes; magic block sizes; `long`→`unsigned` grid cast unchecked. |
| `src/device/cuda/cuda_backend.cu` | B+ | 5,679-line monolith; duplicated decode compaction; latent pinned-input use-after-free. |
| `src/device/cuda/device_f2_blocks.cu` | B | Inline duplicated `DeviceGuard` that does *not* match `p2p_combine.cu`; synchronous copies after pinning. |
| `src/core/fstats/f2_blocks_multigpu.cpp` | B+ | Extreme comment-to-code ratio; `finish_streamed_tier` duck-types tier handles by field name. |
| `src/core/fstats/f2_from_blocks.cpp` | B+ | `compute_f2_block` dispatches to `backend.compute_f2` naming mismatch; over-commented seam. |
| `src/app/result_emit.cpp` | B | No parallel-array length validation; no stream error checks; hand-rolled JSON/CSV. |
| `src/app/f2_dir_writer.cpp` | B+ | `Status::InvalidConfig` for I/O failures; `sha256_file` fails silently; incomplete JSON escaping. |
| `src/app/cli_parse.cpp` | B+ | `std::exit` soup; `build_config` prints to stderr; stale "scaffold no-op" comment. |
| `src/core/config/config_builder.cpp` | B+ | `build() const` mutates `error_message_`; `--ploidy` not range-validated; massive `merge_cli` boilerplate. |
| `src/io/eigenstrat_format.cpp` | B+ | `noexcept` function heap-allocates; locale-dependent `isdigit`; scanner accepts embedded garbage digits. |
| `include/steppe/fstats.hpp` | B+ | `int` dimensions and `double vpair` counts; no invariant enforcement; oversold as "public API." |
| `include/steppe/config.hpp` | B+ | Numeric-type inconsistency; comment bloat; `FilterConfig` uses vectors for set membership. |

## 4. Top 5 Concrete Fixes to Make First

1. **Fix integer-width portability in CUDA and cuBLAS call paths.**
   Replace `long` and `size_t` soup with `std::int64_t` for signed strides and `std::size_t` for buffer sizes. Most importantly, pass `long long`/`int64_t` to `cublasGemmStridedBatchedEx` everywhere. In `f2_blocks_kernel.cu`:
   ```cuda
   const long Psp = static_cast<long>(P) * s_pad;  // line 268
   ```
   should be `int64_t`/`long long`. Make `twoP` consistently 64-bit (`const int twoP = kF2StackedBlocks * P;` at line 267 vs `long twoP` in the gather kernel). In `qpfstats_kernel.cu`, guard the grid launch:
   ```cuda
   const long blocks = (total + kZeroNanThreads - 1) / kZeroNanThreads;  // line 56
   ```
   with `STEPPE_ASSERT(blocks <= UINT_MAX)` or use a helper that returns `unsigned`. Also change `F2BlockTensor` dimensions from `int` to `std::int64_t` in `fstats.hpp`.

2. **Refactor the CUDA backend monolith and unify `DeviceGuard`.**
   `cuda_backend.cu` is 5,679 lines and owns the whole GPU backend. Split it into focused TUs (`cuda_backend_f2.cu`, `cuda_backend_decode.cu`, `cuda_backend_qpadm.cu`, etc.). Deduplicate `decode_af_compact_autosome` and `decode_af_compact_filter` into one helper parameterized by keep-mask and N-gather. In `device_f2_blocks.cu`, delete the two inline `DeviceGuard` definitions and extract a single `device/cuda/detail/device_guard.hpp` that matches the `p2p_combine.cu` version (deleted copy/move, `STEPPE_CUDA_WARN` in destructor). Then actually use `cudaMemcpyAsync` on the stream you pin for, or stop claiming the transfers are async.

3. **Harden error taxonomy and propagation.**
   Stop returning `Status::InvalidConfig` for filesystem failures in `f2_dir_writer.cpp` — use `Status::IOError`. Make `sha256_file` return `std::expected<std::string, Status>` instead of an empty string that looks like success. In `result_emit.cpp`, check `if (!os) return Status::IoError;` after each major emit block. In `config_builder.cpp`, make `build()` non-const and return the error message inside `BuildResult` instead of mutating `error_message_` through a `const` lie. Range-validate `--ploidy` so `--ploidy 3` cannot silently propagate.

4. **Replace hand-rolled JSON/CSV with a small internal builder and add length contracts.**
   In `result_emit.cpp`, assert at the top of each emitter that the parallel result arrays have equal length (`r.se.size() == r.est.size()`, etc.) and that optional vectors are either empty or match. In `f2_dir_writer.cpp`, escape the full JSON control-character range (`U+0000–U+001F`), write `meta.json` in binary mode (`std::ios::binary`), and use `std::string_view` for `kDefaultDiskCachePath` and string-view returns where possible. If you won't pull in a JSON library, build a tiny RAII object/array helper so you cannot forget commas or brackets.

5. **Editorial pass: prune comments by half and scrub stale/ticket references.**
   Remove the ALL-CAPS manifestos ("PRECISION POLICY IS THE LAW"), the cleanup/roadmap ticket citations (`[7.1]`, `X6`, `M4.5`, `cleanup X-13/B26`), and stale scaffolding comments (the `qpwave` "scaffold no-op" in `cli_parse.cpp`). Replace the deprecated `ATOMIC_FLAG_INIT` in `f2_block_kernel.cu` with default construction. Rename `compute_f2_block` in `f2_from_blocks.cpp` to match `backend.compute_f2`, or vice versa. In `f2_blocks_multigpu.cpp`, replace the template duck-typing of `finish_streamed_tier` with a small trait/concept so renaming a tier field fails at compile time with a readable error.

Do those five and the codebase moves from "competent research code with warts" to "credible production port."
