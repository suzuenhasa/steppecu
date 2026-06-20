# Review findings — src__device__backend

Files: /home/suzunik/steppe/src/device/backend.hpp

## Group 4 — Type & numeric

- [4.7][LOW] backend.hpp:485-494 — `compute_f2_blocks_into` takes raw `double* dst_f2`, `double* dst_vpair`, `int* block_sizes_dst` with no host-vs-device type distinction; the doc-comment contract pins them as caller-provided PINNED HOST destinations, but the bare pointer type can't enforce that a device pointer isn't passed by mistake. Suggested: this is the intended CUDA-free seam shape (device residency crosses via opaque DevicePartial/DeviceF2Blocks handles, not raw pointers), so leave as-is or add a thin host-pointer tag only if a wrapper already exists elsewhere; not worth introducing here. Note: no narrowing/overflow — `slab_off = (size_t)P*P*b0` (line 467) is documented with the correct size_t cast, and all index math lives in the .cu/.cpp, not this header.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.
<!-- 3.1 commented-out blocks: every `//` is documentation (Doxygen contracts, AT2/§12 parity rationale, design notes); no code commented "just in case". 3.2 unreachable: only preprocessor is the include guard (25-26 / 763-764); no #if 0; the inline default virtual bodies (e.g. 430-435, 452-460, 485-494, 512-516, 578-582, 592-595, 608-611, 622-625, 645-648, 660-663, 679-682, 721-725) end at the throw with nothing after it. 3.3 unused symbols: all 7 system includes are live (cstddef→size_t 208-211/284-285/759; cstdint→uint8_t 207; span→std::span 320/576-577/...; stdexcept→runtime_error 432+; string→pat 181; vector→64/73/...) and all 8 steppe includes are used by declared symbols (config→Precision 383; error→Status 125/137; fstats→F2BlockTensor 589 param; qpadm→QpAdmOptions/Model/Result 318-320/717-721; views→core::MatView 380; device_partial→DevicePartial 452; device_f2_blocks→DeviceF2Blocks 319/428/575/718; stream_f2_blocks→StreamTarget 512); the fwd `class ComputeBackend;` (308) is used by the free-fn decl (319). 3.4 computed-but-unread: declarations/POD defaults/inline throw bodies only; the `(void)param;` discards are the intentional unused-parameter idiom for default-throwing virtuals whose signatures are the frozen contract used by overrides. -->

