I read through this carefully. This is **not slop** — it's clearly written by someone who understands ownership, 64-bit arithmetic, and the project's parity doctrine. But a senior developer would flag some concrete issues, especially around the Disk I/O path, where the implementation doesn't quite match its own name and comments.

## What's genuinely good

- **Move-only RAII for the disk read handle.** `FileCloser` is a stateless custom deleter, `DiskF2Blocks` has all special members `=default`, and the comment explicitly explains the empty-base optimization (lines 26–41). This is exactly the right way to own a `std::FILE*` in modern C++.
- **Tier-agnostic read-back surface.** `read_block_to_host` and `to_host` branch on `tier` once, keep the numeric read-back identical across Resident/HostRam/Disk, and handle all enumerators (lines 78–183). The unreachable `return F2BlockTensor{}` at line 182 is defensive against future enum additions.
- **64-bit offset arithmetic is done carefully.** `disk_header` widens each factor to `std::uint64_t` before multiplying (lines 68–70), and `pread_all` `static_assert`s that `long` is at least 64 bits before casting `std::uint64_t` offsets into `std::fseek` (line 50). Someone was thinking about LP64 and multi-GB files.
- **CUDA device guard in the Resident arm.** The local `DeviceGuard` restores the previous device on scope exit, routes the restore through the non-throwing `STEPPE_CUDA_WARN`, and `(void)`-discards the `[[nodiscard]]` result (line 101). That's the safe pattern for a destructor.
- **Parity commentary is explicit.** The file-level comment (lines 1–8) and the per-section comments spell out the byte-identical contract and why each data-movement lever is parity-neutral. That kind of documentation is rare and valuable.

## What a senior developer would flag

**`pread_all` doesn't pread, and it doesn't loop:**

```cuda
// pread the whole slab at offset, looping over partial reads. Throws on short read.
void pread_all(std::FILE* f, void* buf, std::size_t bytes, std::uint64_t offset,
               const char* region) {
    ...
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0)
        throw std::runtime_error(...);
    const std::size_t got = std::fread(buf, 1, bytes, f);
    if (got != bytes)
        throw std::runtime_error(...);
}
```

The function name and the comment promise a POSIX `pread`-style, loop-until-done reader. The implementation is a single `fseek` + `fread` with a short-read throw (lines 44–56). That's not `pread` (which wouldn't move the file offset and would be thread-safe), and it doesn't loop over partial reads. A senior reviewer will notice the mismatch immediately.

**The Disk arm is not thread-safe.** Because it uses `std::fseek`/`std::fread` on a shared `std::FILE*` (`disk.read_handle`), concurrent calls to `read_block_to_host` from the FIT's tile reader would race on the file offset and corrupt reads. If the caller is parallel over blocks — and "FIT's tile reader" sounds like it could be — this is a real bug. Either use POSIX `pread` (which matches the helper's name) or add explicit synchronization around the seek/read pair.

**Silent null return in the Resident arm:**

```cuda
const double* f2_dev = resident.f2_device();
const double* vpair_dev = resident.vpair_device();
if (!f2_dev || !vpair_dev) return;
```

At lines 91–92, missing device pointers cause a silent no-op. Hiding a misconfigured `F2BlocksOut` this way makes debugging harder; it should probably throw or at least log/assert.

**Reconstructed disk header bypasses file validation.** `disk_header` rebuilds offsets from `P` and `n_block` instead of reading the actual 64-byte `F2DiskHeader` from disk (lines 58–74). The comment defends this because the layout is deterministic, but it also means magic/version/dtype checks and detection of a stale or truncated cache file are skipped. A cache-format reader normally validates the header before trusting offsets.

**Per-block `cudaHostRegister` is a footgun.** Lines 114–115 register the caller's output buffers for every block read. That's correct for parity, but it's expensive and assumes the caller passed pageable heap memory. A stack buffer, already-registered memory, or a non-pageable allocation will fail at runtime. The comment calls it a "data-movement lever," which is true, but it couples the API to the caller's allocation properties.

**Stale line-number references in comments.** Line 106 references `device_f2_blocks.cu:55-56`. Line numbers drift; a symbol or section reference would stay accurate.

**C-style string concatenation for error messages.** Lines 52, 55, 130, and 173 build `std::runtime_error(std::string("...") + region)`. It works, but it duplicates the `"F2BlocksOut(Disk): "` prefix and reads like C-with-strings. A small formatting helper or `std::format` (if the project can use it) would be cleaner.

## The "slop" test

**Not slop.** Slop is unexplained magic numbers, copy-pasted drift, missing error checks, and obviously wrong algorithms. This file has none of that. The ownership model is correct, the parity contract is explicit, and the comments explain *why*, not just *what*. The issues are real, but they're the kind of rough edges that come from careful code that got slightly ahead of itself — not from carelessness.

## What it actually looks like

This looks like **solid systems-integration code written by a developer who knows C++ ownership and the project's parity requirements.** The CUDA side is conservative and correct, the RAII around the `FILE*` is exactly right, and the 64-bit arithmetic is careful. The weak spot is the Disk I/O helper: the name and comment promise a robust, thread-safe `pread` loop, but the implementation is a single `fseek`/`fread` pair that is neither `pread` nor reentrant. That's the kind of mismatch that makes a senior reviewer pause and wonder what else was renamed without being fully reimplemented.

**Verdict:** B — competent and largely production-ready, but fix `pread_all` (use real `pread` or add synchronization, loop honestly, and rename if it stays `fseek`/`fread`) and consider validating the on-disk header before showing this as showcase code. With those fixes it climbs to A-.
