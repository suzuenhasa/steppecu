# Review findings — src__device__cuda__decode_af_kernel

Files: /home/suzunik/steppe/src/device/cuda/decode_af_kernel.cu, /home/suzunik/steppe/src/device/cuda/decode_af_kernel.cuh

## Group 4 — Type & numeric

No Group 4 issues found.

## Group 2 — Deprecated / removed APIs & platform support

No Group 2 issues found.

## Group 3 — Dead / commented-out code

No Group 3 issues found.

## Group 5 — Hardcoded values / magic numbers

- [5.3][MED] src/device/cuda/decode_af_kernel.cu:69 — `byte_in_rec = s / 4u` re-picks the 2-bit-packing radix (4 genotype codes per byte) as a bare literal. The project already single-homes this as `io::kCodesPerByte = 4` (src/io/eigenstrat_format.hpp:54), and mind_prepass.cpp routes its identical byte-index math through it (`s / kPerByte`). The kernel here and the CPU oracle (src/device/cpu/cpu_backend.cpp:336) hold a duplicated copy of the radix that can DRIFT from the io single-home; the byte index MUST match the packing the reader used. Note: decode_af.hpp deliberately avoids depending on `io` (lines 53-57), so the fix is a named packing-radix constant in the core decode primitive home (decode_af.hpp), not a direct `io::kCodesPerByte` use. Suggested: name the radix once in decode_af.hpp (e.g. `kCodesPerByte`, pinned equal to `io::kCodesPerByte` by the existing equivalence test) and have the kernel + CPU backend divide by it.
- [5.3][MED] src/device/cuda/decode_af_kernel.cu:70 — `pos_in_byte = s & 3` re-picks the same 2-bit-packing fact (`& (codes_per_byte - 1)`) as the bare mask `3`. This duplicates the in-byte-position convention that decode_af.hpp::genotype_code already embeds (`k & 3`, line 67) and that io::code_in_byte derives from `kCodesPerByte` (eigenstrat_format.hpp:105). The kernel computes the position itself then passes it to genotype_code, so the mask lives in two places with no shared name; drift between the `/4u` byte index and the `&3` position would silently mis-decode. Suggested: derive the position mask from the same named radix (`pos = s % kCodesPerByte` or `& (kCodesPerByte-1)`) used for the byte index, so both come from one constant.

## Group 6 — Naming

- [6.4][LOW] src/device/cuda/decode_af_kernel.cu:69 — `byte_in_rec` abbreviates "record" to "rec" while the very next-cited parameter in the same TU spells it out as `bytes_per_record` (line 57/91); an inconsistent abbreviation of the same word ("record") within one file. The local also reads as plausibly "bytes in record" rather than "byte index within the record". Suggested: spell it consistently — e.g. `byte_in_record` (and optionally clarify intent as `byte_index_in_record`) to match `bytes_per_record`.

## Group 7 — Duplication

No Group 7 issues found.

## Group 8 — Comments

- [8.1][LOW] src/device/cuda/decode_af_kernel.cu:46 — the inline comment on `using core::kDecodeBlockY;` reads `// population axis (32*8 = 256 threads/block)`, but `kDecodeBlockY` is 8 (launch_config.hpp:167); the `32*8 = 256` is the whole block's thread-count product (kDecodeBlockX·kDecodeBlockY), attached to the Y-dim alias line. A reader skimming can misread it as kDecodeBlockY = 256, and the "256 threads/block" arithmetic restates a value derivable from the two block edges. Contrast the sibling line 45 comment, which correctly annotates kDecodeBlockX's own value (`32 = one warp`). Suggested: state kDecodeBlockY's own value/role on this line (e.g. `// population axis (8; block is 32×8 = 256 threads)`) so the inline comment describes the symbol it annotates.

## Group 9 — Constants & configuration

No Group 9 issues found.

## Group 10 — Initialization

No Group 10 issues found.
