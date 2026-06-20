# Review findings — src__device__cuda__f2_blocks_out

Files: /home/suzunik/steppe/src/device/cuda/f2_blocks_out.cu, /home/suzunik/steppe/src/device/f2_blocks_out.hpp

## Group 4 — Type & numeric

No Group 4 issues found.

Notes (clean, but verified at scale — not findings):
- 4.2/4.6 The block-major index into the resident/host tensor (the up-to-~10^10-element f2/vpair buffers) is computed 64-bit: `slab = (size_t)P * (size_t)P` then `off = slab * (size_t)b` (.cu:79,92,100). `slab` is already `size_t`, so the `* b` multiply is 64-bit — no `int` overflow before widening.
- 4.6 The Disk-tier region/total sizes widen every operand before multiplying: `region = (uint64_t)P * (uint64_t)P * (uint64_t)n_block * sizeof(double)` (.cu:68-70) and `total = (size_t)P * (size_t)P * (size_t)n_block` (.cu:144-146). The offset helpers in f2_disk_format.hpp (f2_block_offset/vpair_block_offset) also widen to uint64_t before the `* b` multiply.
- 4.3 Allocation sizing correct: `vector<double>::assign(total, 0.0)` is element-count (.cu:147-148); `pread_all(..., total * sizeof(double), ...)` and `bytes = slab * sizeof(double)` are byte-counts (.cu:80,153,155). No element/byte confusion.
- 4.1 All math is `double` (FP64-by-design); no float narrowing in a parity-critical path.
- 4.4/4.5 No loops in either file — no unsigned countdown / signed-unsigned bound risk.
- 4.7 The D2H accessor takes raw `double*` host out-pointers and copies from `const double*` device pointers (.cu:78,85-96); this is the established backend-seam idiom and not a unit-local defect. The 64-bit file-offset assumption is pinned by `static_assert(sizeof(long) >= 8)` (.cu:53).

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

Notes (verified, not findings):
- 2.1 No SM-arch flags or CMake arch lists in either unit file (build flags live in CMake, out of unit scope); no sm_50/60/70-gated code.
- 2.2 No texture/surface references (no `texture<...>`, `cudaBindTexture*`, `surface<...>`); this TU has no kernels — it is host-side D2H/memcpy/file I/O only.
- 2.3 No warp intrinsics of any kind (no `__shfl*`/`__ballot`/`__any`/`__all` legacy or `_sync` variants); no `__syncthreads`. No kernels in the TU.
- 2.4 No `cudaThreadSynchronize`. The only CUDA runtime calls are current/supported: `cudaGetDevice`/`cudaSetDevice` (.cu:89,90,91), `cudaMemcpy` with `cudaMemcpyDeviceToHost` (.cu:93-96). `cudaMemcpy` is implicitly synchronizing here (no explicit device-sync call needed/used).

## Group 3 — Dead / commented-out code

- [3.3][LOW] src/device/cuda/f2_blocks_out.cu:19 — `#include "device/cuda/pinned_buffer.cuh"` is unused; no `RegisteredHostRegion`/`PinnedBuffer` symbol is referenced anywhere in the TU (the Resident D2H at .cu:93-96 is a plain `cudaMemcpy` into an unpinned `double*`). The "pin the D2H; graceful degrade" intent in the comment was never realized. Suggested: drop the include (and its stale comment), or actually pin the host out-buffer if the perf intent stands.
- [3.3][LOW] src/device/cuda/f2_blocks_out.cu:21 — `#include "core/internal/log.hpp"` is unused; no `STEPPE_LOG_WARN`/`STEPPE_LOG*` macro is invoked. The dtor/move-assign `std::fclose` (.cu:28,36) ignore the return without logging, so the "STEPPE_LOG_WARN (teardown)" comment describes a never-written log call. Suggested: drop the include, or add the intended teardown warn-on-close-failure log.
- [3.3][LOW] src/device/cuda/f2_blocks_out.cu:17 — `#include "device/cuda/device_buffer.cuh"` is a redundant direct include; `DeviceBuffer` is not named in this TU and is transitively provided by `device_f2_blocks_impl.cuh` (the include's own comment says "via device_f2_blocks_impl"). Suggested: rely on the transitive include and remove the direct one (or keep only if an IWYU policy mandates direct includes).
- [3.2][LOW] src/device/cuda/f2_blocks_out.cu:160 — `return F2BlockTensor{};` after the `switch(tier)` is unreachable (all three `OutputTier` enumerators return in their `case`, as the inline comment notes). It is the intentional fall-through to silence the "control reaches end of non-void function" warning given the switch has no `default`. Not a defect — defensive; noted for completeness only. Suggested: leave as-is.

## Group 5 — Hardcoded values / magic numbers

- [5.3][LOW] src/device/cuda/f2_blocks_out.cu:79, .cu:144-146; src/device/f2_blocks_out.hpp:88-89 — the per-block slab element count `(size_t)P * (size_t)P` is open-coded in three places: `read_block_to_host` (`slab` at .cu:79), `to_host` Disk-tier (`total = P*P*n_block` at .cu:144-146), and `F2BlocksOut::size()` (.hpp:88-89). The same `P²` (and `P²·n_block`) shape is also re-derived in `disk_header` (`region` at .cu:68-70) and again in `f2_disk_format.hpp`'s `f2_block_offset`/`vpair_block_offset`. All five sites are currently consistent (each correctly widens before multiplying), but the repetition is a DRIFT surface: a future edit to one (e.g. a layout/stride change) can silently diverge the others. Suggested: factor a single `slab_elems(P)` / `block_offset(P, b)` helper (or have `read_block_to_host`/`to_host` reuse `size()` and the `f2_disk_format.hpp` offset helpers) so the block-major shape is defined once.
- [5.1][LOW] src/device/cuda/f2_blocks_out.cu:56 — `std::fread(buf, 1, bytes, f)` uses element-size literal `1` (byte-granular read so the `got != bytes` short-read check is exact). This is the idiomatic fread-as-byte-read pattern, not a magic number; documenting as a deliberate non-finding. Suggested: leave as-is.

Notes (clean, not findings):
- 5.1 No unnamed numeric literals of concern: the only bare numbers are `0`/`nullptr` (member/handle init, .cu:28,33,41 and .hpp), the `assign(total, 0.0)` zero-fill (.cu:147-148, a natural value-init not a tunable), and the `fread` element-size `1` (noted above). No `1024`/`0.5`/`0.001f`/tolerances/block-dims anywhere.
- 5.2 No hardcoded sizes/bounds: the header size is taken via `sizeof(F2DiskHeader)` (.cu:67) and byte counts via `sizeof(double)` (.cu:70,80,153,155) — derived, not literal `64`/`8`. (The literal `64` lives only in the dependency `f2_disk_format.hpp`, out of this unit's scope, and is pinned there by `static_assert(sizeof(F2DiskHeader)==64)`.)
- 5.4 No hardcoded paths/IDs/device ids: the D2H targets `resident.device_id` (.cu:91), not a literal device; the path comes from the `DiskF2Blocks` descriptor (.cu:31,37). `SEEK_SET` (.cu:54) is a standard-library constant.
- 5.5 No `32` literal and no warp-size usage anywhere — this TU is host-side D2H/memcpy/file I/O with no kernels.

## Group 6 — Naming

- [6.1][LOW] src/device/cuda/f2_blocks_out.cu:90 — the device-restore RAII guard is declared as `struct G { int d; ~G() { (void)cudaSetDevice(d); } } restore{prev};`. The type name `G` and its member `d` are both single-letter and opaque outside that one line (a type/member, not a tight-loop counter); a reader must reverse-engineer that `G` is the cudaSetDevice scope-guard and `d` the device id to restore. The instance `restore` is fine. Suggested: name the type (e.g. `struct DeviceGuard { int device; ... } restore{prev};`).
- [6.1][LOW] src/device/cuda/f2_blocks_out.cu:48-49 — `pread_all`'s label parameter `const char* what` is a mildly opaque name for "the region tag used in the error message" (callers pass `"f2"`, `"vpair"`, `"f2-region"`); `what` reads like a generic placeholder rather than its role. Low impact (purely diagnostic), one-liner. Suggested: rename to `region`/`label`/`what_region`.
- [6.3][LOW] src/device/cuda/f2_blocks_out.cu:84-85 — within the same two-line Resident block the two device pointers abbreviate inconsistently: `f2_dev` keeps the full `f2` stat name while `vp_dev` contracts `vpair`→`vp`. Everywhere else in the unit the second stat is spelled `vpair` in full (params `vpair_slab_out` .cu:78/.hpp:85, helper `vpair_block_offset` .cu:111, members `vpair`/`vpair_offset`). The lone `vp` abbreviation breaks the file's own `vpair` convention. Suggested: rename `vp_dev` → `vpair_dev` to match `f2_dev` and the rest of the file.

Notes (clean, not findings):
- 6.1 The remaining short names are idiomatic or are tight local scalars, not cryptic: `o` for the moved-from `other` in the move ctor/assign (.cu:30,34) is the canonical C++ special-member idiom; `d` as the `disk_header(const DiskF2Blocks& d)` param (.cu:63) and `h` as the `F2DiskHeader h{}` local (.cu:64,109,152) are conventional single-use locals; `f`/`buf`/`bytes`/`offset` in `pread_all` (.cu:48) are clear; `b` is the block index in `read_block_to_host(int b,...)` (.cu:78) — an index/loop-style counter, in scope; `slab`/`bytes`/`off`/`region`/`total`/`got`/`prev` are all descriptive.
- 6.2 No misleading names: `block_sizes` is a per-block size list, `n_block` is a count, `*_offset` are byte offsets, `read_handle` is a FILE*, `tier` is the OutputTier — every name matches its role. `size()` returns an element count (P²·n_block) as documented (.hpp:87-90), not a byte size.
- 6.3 Conventions are otherwise consistent within each file: snake_case for functions/locals/members, the `n_block`/`block_sizes` naming, and `*_offset`/`*_dev`/`*_slab_out` suffix families are uniform (the lone `vp_dev` exception is flagged above).
- 6.4 No nonstandard abbreviations: `f2`/`vpair` are the project's established stat names, `dev`/`vp`/`prev`/`off`/`buf` are common and unambiguous in context; no invented or domain-obscuring contractions.

## Group 7 — Duplication

- [7.2][LOW] src/device/cuda/f2_blocks_out.cu:70, .cu:129, .cu:142; src/device/f2_blocks_out.hpp:89 — the `n_block < 0 ? 0 : n_block` clamp-to-nonnegative is hand-inlined in four places: `disk_header`'s `region` (.cu:70), `to_host` HostRam `out.n_block` (.cu:129), `to_host` Disk `out.n_block` (.cu:142), and `size()` (.hpp:89). This is distinct from the existing 5.3 finding (which is about the `P*P` product) — it is the repeated negative-guard expression. A future change to the clamp policy (e.g. asserting instead of silently zeroing) must touch all four. Suggested: a single `inline int clamped_n_block(int n) noexcept { return n < 0 ? 0 : n; }` (or store the already-clamped value once) and call it at each site.
- [7.4][LOW] src/device/cuda/f2_blocks_out.cu:106-108, .cu:150-151 — the Disk-tier read paths each open with a near-identical `if (!disk.read_handle) throw std::runtime_error("F2BlocksOut::<fn>: Disk tier has no open read handle")` guard, differing only by the function-name prefix in the message. Both then immediately do `const F2DiskHeader h = disk_header(disk);` (.cu:109, .cu:152). Suggested: a small private helper (e.g. `const F2DiskHeader& require_disk_header()` that checks the handle and returns/builds the header) to fold the guard + header-reconstruct boilerplate into one place.
- [7.1][LOW] src/device/cuda/f2_blocks_out.cu:93-96, .cu:101-102, .cu:110-111, .cu:153-156 — each read-back arm performs the SAME copy twice in a row, the second call differing only by the f2-vs-vpair pointer pair: two `cudaMemcpy(...DeviceToHost)` (.cu:93-96), two `std::memcpy` (.cu:101-102), two `pread_all` (.cu:110-111), and two `pread_all` for the whole regions (.cu:153-156). It is the same "do this for f2, then identically for vpair" shape repeated across all three tiers. Minor (each pair is 1-2 lines and the asymmetry is just the operand), but it is a real copy-paste-differing-by-a-constant pattern. Suggested: optional — a 3-arg lambda/helper `copy_one(dst, src_or_off, tag)` invoked once per stat per tier; leave as-is if the explicit two-call form is judged clearer.

Notes (clean / deferred to other groups, not new findings):
- 7.2 The other major repeated shape — the `(size_t)P * (size_t)P` (and `·n_block`) product re-derived in `read_block_to_host` (.cu:79), `to_host` Disk (.cu:144-146), `size()` (.hpp:88-89), `disk_header`'s `region` (.cu:68-70), and `f2_disk_format.hpp`'s offset helpers — is already filed as [5.3][LOW]; not re-counted here.
- 7.2 Genuine loop-invariant hoisting is already done well: `slab` and `bytes` are each computed once at the top of `read_block_to_host` (.cu:79-80) and reused; the partial-read loop is already extracted into the `pread_all` helper (.cu:48-59) rather than copy-pasted into `read_block_to_host`/`to_host`.
- 7.3 The repeated `sizeof(double)` byte-multiplier (.cu:70,80,153,155) and the `static_cast<uint64_t>`/`static_cast<size_t>` widenings are idiomatic per-site widening, not a foldable duplication — hoisting them into a helper would obscure the parity-load-bearing intent without removing real risk.
- 7.4 The DiskF2Blocks move ctor (.cu:30-33) and move-assign (.cu:34-44) share the member-move list; this is the canonical hand-written move special-member pair (assign also needs the self-check + fclose-old), not collapsible boilerplate — extracting would not simplify.

## Group 8 — Comments

- [8.2][LOW] src/device/cuda/f2_blocks_out.cu:19 — the include comment `// RegisteredHostRegion (pin the D2H; graceful degrade)` is STALE: no `RegisteredHostRegion`/pinning is used anywhere — the Resident D2H at .cu:93-96 is a plain `cudaMemcpy` into an unpinned `double*`. The comment describes a pin-the-D2H behavior the code does not have. Suggested: drop the comment with the unused include (already filed as [3.3] for the include itself), or implement the described pinning.
- [8.2][LOW] src/device/cuda/f2_blocks_out.cu:21 — the include comment `// STEPPE_LOG_WARN (teardown)` is STALE: no `STEPPE_LOG_WARN`/`STEPPE_LOG*` is ever invoked; the dtor/move-assign `std::fclose` (.cu:28,36) silently discard their return with no teardown warn. The comment promises a teardown log that was never written. Suggested: remove the comment with the unused include (already filed as [3.3]), or add the intended close-failure warn.

Notes (clean, not findings):
- 8.1 No restating-the-code comments. Every comment explains WHY / parity intent, not the mechanics: the file/header banners (.cu:1-6, .hpp:1-15) state the §12 bit-identical contract; .cu:46-47 (`pread the whole slab... loops over partial reads`), .cu:61-62 (`reconstruct the disk header from the descriptor's shape... we do not re-parse`), .cu:117/.cu:120-123/.cu:136-139 (tier read-back rationale) all justify rather than narrate. No `x++; // increment x` style lines.
- 8.3 Non-obvious choices ARE documented with rationale: the 64-bit-file-offset assumption is explained AND pinned (`static_assert(sizeof(long) >= 8)`, .cu:52-53); the `disk_header` reconstruction (vs re-parsing the on-disk 64-byte header) is justified (.cu:61-62); the unreachable `return F2BlockTensor{}` is annotated `// unreachable (all enumerators handled)` (.cu:160); the §12 bit-identical-across-tiers invariant is stated at every read-back arm (.cu:117,120-123,125-126,136-139) and in the header (.hpp:11-15,75-85). The lone undocumented spot — the `n_block < 0 ? 0 : n_block` defensive clamp (.cu:70,129,142; .hpp:89) gives no reason WHY n_block could be negative — is low value and already filed for de-duplication under [7.2]; not re-raised as a rationale gap.
- 8.4 No orphan TODO/FIXME/HACK/XXX/TBD/WIP markers anywhere in either file (grep-verified).

## Group 9 — Constants & configuration

No Group 9 issues found.

Notes (clean, not findings):
- 9.1 No should-be-const/constexpr left mutable: every local in the .cu is already `const` (`slab`/`bytes` .cu:79-80, `off` .cu:92,100, `region` .cu:68, `total` .cu:144, `got` .cu:56, `h` .cu:64,109,152, the `f2_dev`/`vp_dev` device pointers .cu:84-85). The RAII guard member `int d` (.cu:90) is write-once-then-read but is a guard-captured value, not a config constant. The public data members of `DiskF2Blocks`/`F2BlocksOut` (P, n_block, tier, block_sizes, the tier payloads — .hpp:44-49, .hpp:63-71) are intentionally mutable: the result is populated by the precompute sink, then read back; they are not compile-time constants. `size()` is `[[nodiscard]] noexcept` (.hpp:87) — appropriately qualified.
- 9.2 No tangled/buried config knobs: this TU has no tunable parameters (block dims, tile sizes, thresholds, tolerances) — all "constants" are derived from the descriptor shape (`sizeof(F2DiskHeader)` .cu:67, `sizeof(double)` .cu:70,80,153,155, `P²·n_block` products) or are stdlib constants (`SEEK_SET` .cu:54, the `fread` byte element-size `1` .cu:56). The only behavioral policy — the `n_block < 0 ? 0 : n_block` clamp (.cu:70,129,142; .hpp:89) — is already filed for de-duplication under [7.2]; it is not a buried tunable.
- 9.3 No positional booleans: there are NO `bool` parameters, `true`/`false` literals, or `foo(true,false)`-style calls anywhere in either file (grep-verified). The tier dispatch already uses the named `OutputTier` enum (.hpp:63, switched at .cu:83,119) rather than boolean flags, and the copy direction uses the named `cudaMemcpyDeviceToHost` enumerator (.cu:94,96) — exactly the named-flag discipline 9.3 asks for.

## Group 10 — Initialization

No Group 10 issues found.

Notes (clean, not findings):
- 10.1 No late/distant or uninitialized-then-assigned locals. In the Resident arm `int prev = 0;` (.cu:88) is explicitly value-initialized BEFORE `cudaGetDevice(&prev)` writes it (.cu:89) — not an uninitialized-then-assigned hole. `slab`/`bytes` are declared with their initializers at the top of `read_block_to_host` (.cu:79-80) right where their value is known; `off` (.cu:92,100), `region` (.cu:68), `total` (.cu:144), `got` (.cu:56) are each declared at first use with initializer. The RAII guard's `int d` (.cu:90) is brace-initialized via `restore{prev}` — copied from the already-set `prev`, never read uninitialized.
- 10.2 No unsound zero-init reliance. `F2DiskHeader h{}` (.cu:64,109,152) value-initializes the whole struct, then every field `disk_header` actually consumes (P, n_block, the three computed offsets) is explicitly assigned (.cu:65-72); the header is used only for local offset math, never serialized, so any field left at the `{}`-zero default is intentionally inert. The DiskF2Blocks/F2BlocksOut data members all carry in-class initializers (.hpp:45-48: `P=0`, `n_block=0`, `read_handle=nullptr`; .hpp:63-66: `tier=Resident`, `P=0`, `n_block=0`), and the hand-written move ctor (.cu:30-33) sets every member in its init list while move-assign (.cu:34-44) assigns every member — no member depends on implicit zero-init. The Disk-tier `out.f2.assign(total,0.0)`/`out.vpair.assign(total,0.0)` (.cu:147-148) is an explicit value-fill (not a zero-init assumption) and is immediately overwritten by the `pread_all` region reads (.cu:153-156).
