# Review findings — src__device__cpu__cpu_backend

Files: /home/suzunik/steppe/src/device/cpu/cpu_backend.cpp

## Group 4 — Type & numeric

No Group 4 issues found.

Notes (why clean, for the record — not findings):
- 4.1 (float/double): all FP math is `double`/`long double` by design (FP64 oracle, §12). The `long double` accumulators (pairwise_sum, jackknife/loo sums) are intentional and parity-load-bearing; no wrong narrowing. The only `static_cast<double>` of a `long double` numerator (lines 190, 290, 695, 712, 726, 474, 993) is the deliberate "native-FP64 numerator/divide" step mirrored from the GPU path — not a precision-losing temp in a parity-critical accumulation.
- 4.2/4.6 (index width at scale): every global index into the P*P, P*P*n_block (f2 tensor) and P*M (decode) arrays is widened to `std::size_t` BEFORE the multiply (lines 128-129, 193-196, 232-233, 270, 291-294, 321-322, 349-350; assemble_f4 399-401, 407-408, 412). The hot-path element access goes through `core::MatView::element` (views.hpp:66-68), which widens via `static_cast<long>(P) * s` — 64-bit. The packed-tile read (line 343) uses `g * tile.bytes_per_record + byte_in_rec`, all `std::size_t` (backend.hpp:208-211). No `int` product of large dims feeds a wider index.
- 4.3 (allocation sizing): all buffers are `std::vector<T>::assign(count, val)` (element-count API, `sizeof(T)` handled internally) — no raw cudaMalloc/new with element/byte confusion.
- 4.4 (unsigned countdown): no descending unsigned loop; pairwise_sum recurses with `std::size_t` halving (no underflow — base case guards n<=128 and n==0).
- 4.5 (signed/unsigned compares): loop bounds are type-consistent (`int i < P`, `long s < M`, `std::size_t k < n`); the fit-engine `nl*nr`, `nl*r`, `nl*nl` products are over small model dims (sources/rights, not P/M/n_block) so no int-overflow risk.
- 4.7 (host/device pointer typing): pure host TU, no CUDA, no device pointers; raw `const double*`/`const std::uint8_t*` are host-only views. N/A.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (why clean, for the record — not findings):
- This is a pure host C++20 TU: the file header (lines 40-46) pins "NO CUDA header is included here"; the include set (lines 48-68) is std lib + host-pure steppe headers only. No CUDA toolkit token appears anywhere.
- 2.1 (dropped archs): no architecture flags, no CMake content, no `__CUDA_ARCH__`/sm_* usage in this `.cpp`. N/A.
- 2.2 (texture/surface references): no `texture<...>`/`surface<...>`/`cudaBindTexture*`/texture-object code. N/A.
- 2.3 (non-`_sync` warp intrinsics): scalar host loops only; no `__shfl*`/`__ballot*`/`__any`/`__all`/`__activemask`. N/A.
- 2.4 (`cudaThreadSynchronize`): no CUDA runtime calls of any kind. N/A.

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/cpu/cpu_backend.cpp:745 — `xmat_from_loo_block(const F4Blocks&, int b)` is a private static helper defined here but never called anywhere (grep across src/include/tests finds only this definition). It is superseded by the S7 path in `gls_weights_loo_batched` (lines 638-651), which builds a one-block `F4Blocks` whose `x_total` is the LOO slice and reads it via `xmat_from_total` — so the LOO-block xmat builder is dead. Suggested: delete `xmat_from_loo_block` (and its doc comment, lines 743-744), or if intentionally kept for a future direct-LOO path, mark `[[maybe_unused]]` with a TODO.

Notes (why otherwise clean — not findings):
- 3.1 (commented-out blocks): no commented-out code kept "just in case"; all `//` blocks are explanatory parity/design rationale (file header, AT2 math derivations). The "No: crossprod(x)=t(x)·x" aside (lines 930-931) is a derivation note, not commented-out code.
- 3.2 (unreachable code): no `#if 0`, no code after `return`/`break`. Every early `return out;` (lines 131, 204, 236, 327, 393, 439) is a guard with reachable follow-on for the non-degenerate path.
- 3.3 (other unused symbols): all private helpers (`pairwise_sum`, `compute_loo_and_total`, `xmat_from_total`, `seed_AB`, `opt_A`, `opt_B`, `als_weights`, `chisq_of`) and the `tot_line_` member are referenced internally (verified by grep). `rank_test` (line 493) looks unused inside this TU but is a virtual override consumed via core/qpadm/gls_solve.hpp:20 — not dead. All includes (lines 48-68) resolve to used symbols (`<climits>`→INT_MIN@569, `<cmath>`→std::sqrt@454, `<cstdint>`→int64/uint8@338-342, `<limits>`→numeric_limits@570-588, `<memory>`→unique_ptr@1004, `<span>`→std::span@246/376/430). The `(void)precision;` casts (lines 117, 223, 379, 433, 497, 523, 615) are intentional unused-param acknowledgments under -Werror, not dead code.
- 3.4 (computed but unread): no assigned-never-read locals. The `diffsum / nb * nb` shape (lines 719-720) is a deliberate AT2 `mean(...)*nb` parity transcription, not a dead no-op.

