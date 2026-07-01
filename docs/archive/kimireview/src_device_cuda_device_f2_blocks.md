I read `device_f2_blocks.cu` and its related headers carefully. The short version: **not slop**, but it has a couple of "wait, that's not what the comment says" moments that would make a senior reviewer pause.

## What's genuinely good

- **Clean CUDA-free seam.** The pimpl split between `DeviceF2Blocks` (CUDA-free opaque handle), `device_f2_blocks_impl.cuh` (the `Impl` with `DeviceBuffer<double>`), and this `.cu` file is the right way to keep `core`/`api`/Python from dragging in `<cuda_runtime.h>`.
- **RAII ownership is correct.** Device memory is owned by `DeviceBuffer<double>`, host pinning by `RegisteredHostRegion`, and the handle is move-only. No raw `cudaMalloc`/`cudaFree` pairs leaking out.
- **Fail-fast error handling is consistent with the project.** Every CUDA runtime call routes through `STEPPE_CUDA_CHECK` and throws a typed `CudaError` with `std::source_location` — no `std::exit`, no silent ignores.
- **The cross-device free story is documented.** The long comment at lines 21–27 explaining why `~DeviceF2Blocks()` can be device-agnostic is correct: `cudaFree` is pointer-device-aware, and the escape contract in the header is honest about moved-from husks.
- **Bit-identical transfer intent.** Using raw byte copies (`cudaMemcpyDeviceToHost`/`HostToDevice`) rather than reinterpreting the data preserves `-0.0`/NaN payloads exactly, which matters for parity claims.

## What a senior developer would flag

**The `DeviceGuard` in this file is *not* the same as p2p_combine's, despite the comment saying it matches.**

At lines 56–63 and 92–99:

```cuda
struct DeviceGuard { int dev; ~DeviceGuard() { (void)cudaSetDevice(dev); } } restore{prev};
```

Compare to `p2p_combine.cu:163–178`: that version has an explicit `noexcept` constructor, explicitly deleted copy/move, and uses `STEPPE_CUDA_WARN` in the destructor. This version is an aggregate with no deleted special members and silently swallows `cudaSetDevice` errors with a bare `(void)`. The comment at line 60 even says "Named guard matching p2p_combine.cu's DeviceGuard" — it doesn't. A senior dev would call this **copy-paste drift with false documentation**.

**Worse, the same guard is defined inline twice in one file.** Lines 56–63 and 92–99 are nearly identical. If this is a reusable pattern, it belongs in a `detail` header (`device/cuda/device_guard.hpp` or similar). Defining it locally in two functions invites the exact drift that already happened.

**Synchronous `cudaMemcpy` while pinning for async.**

Lines 67–72 and 113–118 register host pages with `RegisteredHostRegion`, which exists to make `cudaMemcpyAsync` a true DMA. But the copies themselves use blocking `cudaMemcpy` — no stream, no async, no overlap. `p2p_combine.cu` actually uses `cudaMemcpyAsync` + `cudaStreamSynchronize` (lines 231–235). So the comment "EXACTLY like p2p_combine.cu:185–186" is accurate about *pinning* but misleading about the overall pattern. A senior would ask: why pin at all if you're going to block the host anyway?

**No input validation.**

`upload_f2_blocks_to_device` takes `device_id` and copies `host.P`, `host.n_block`, `host.block_sizes` without checking:
- Is `device_id` a valid CUDA ordinal?
- Does `host.block_sizes.size() == host.n_block`?
- Is `host.f2.size() == host.size()`?

If a caller passes an inconsistent `F2BlockTensor`, you'll get a silent mis-copy or a much-later out-of-bounds read downstream. `to_host()` has the same issue in reverse: `impl` could be non-null while `P`/`n_block` disagree with the buffer size.

**Repeated negative-`n_block` handling and allocation-before-early-return.**

Lines 49 and 86 both do `(n_block < 0 ? 0 : n_block)`. The `F2BlockTensor::size()` already handles this, but `DeviceF2Blocks::size()` in the header also does it. It's not wrong, but it's noise.

In `upload_f2_blocks_to_device`, the `out.impl` allocation happens at lines 102–104 *after* the device guard is set. That's correct, but the function could be tightened: the `if (total == 0) return out;` at line 90 returns an empty handle before any device-switch, which is good; however, `to_host()` resizes `out.f2`/`out.vpair` at lines 52–53 *before* the early return at line 54. Minor, but inconsistent ordering.

**Comment density is high even by this codebase's standards.**

Lines 21–27, 39–45, 76–82, etc. are accurate but defensive. Comments like "Defaulted dtor -> ~unique_ptr<Impl> -> ~DeviceBuffer<double>'s bare cudaFree" explain implementation mechanics that a senior reader already knows. The file reads like someone justifying every decision rather than stating the contract.

## The "slop" test

**Not slop.** There's no magic-number soup, no obviously wrong algorithm, no unchecked allocations, no `printf` debugging, no `// TODO` left in this file. The RAII story is sound, the error path is exception-safe, and the cross-device lifetime reasoning is correct. The comments are verbose but not *stale* — except for the p2p_combine "matching" claim, which is stale in substance even if recent in date.

## What it actually looks like

This looks like **competent, careful C++/CUDA written by someone who understands the project architecture but hasn't fully internalized the "don't repeat yourself" discipline yet.** The hard parts — ownership, CUDA-free seams, typed exceptions, pointer-device-aware free — are done right. The easy parts — extracting a shared helper, keeping comments honest, validating inputs, and using async copies consistently with the pinning infrastructure — are where it sags.

A senior reviewer would say: "The design is solid, but tighten the duplication and either use async copies or stop pretending this is the async path."

## Verdict

**B, ship after two fixes:** (1) extract a real shared `DeviceGuard` that matches the p2p_combine version (or unify them), and (2) decide whether these transfers are intentionally synchronous; if so, drop the pinning-or-call-it-what-it-is, if not, move to `cudaMemcpyAsync` on a stream. Add a little input validation and trim the defensive comments, and it's A- work.