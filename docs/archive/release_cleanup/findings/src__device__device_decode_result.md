# src__device__device_decode_result
Files: /home/suzunik/steppe/src/device/device_decode_result.hpp, /home/suzunik/steppe/src/device/cuda/device_decode_result.cu, /home/suzunik/steppe/src/device/cuda/device_decode_result_impl.cuh
Subsystem: device-cuda

## Findings

### G13
- [G13.error handling][MED] device_decode_result.cu:52-57 — `to_host_qvn` unconditionally `cudaMemcpy`s from `impl->n.data()`, but the header (lines 70-77) documents the precondition "Requires n_device() non-null (a regime-B result)". For a regime-A result (`impl` non-null but `impl->n` empty), `impl->n.data()` is `nullptr` while `pmk > 0`, so the third `cudaMemcpy` is `cudaMemcpy(dst, nullptr, pmk*8, D2H)`. `STEPPE_CUDA_CHECK` will then throw a runtime error rather than the contract being enforced at the call boundary. The `empty()` short-circuit at line 36 only tests `P`/`M_kept`, not `n` residency, so it does not catch this misuse. Suggested: assert/guard `impl->n.data() != nullptr` (regime-B precondition) before the N copy, or skip the N copy and clear `n_host` when `n` is empty.

### G18
- [G18.correctness traps][MED] device_decode_result.cu:56-57 — same root cause as the G13 finding viewed as a correctness trap: a regime-A handle passed to `to_host_qvn` reads a null source pointer for N (the q/v copies at 52-55 are well-formed; only the N copy is the trap). Suggested: enforce the documented regime-B precondition before the N copy.

(groups checked: G2-G10, G11-G22; no issues found in the remaining groups. The PIMPL handle is move-only with deleted copies and `unique_ptr<Impl>` ownership; the `pmk` size product is correctly `size_t`-widened (cu:42-43) and the `* sizeof(double)` byte products are overflow-safe; no raw device indexing, no kernels, no launches, and no alloc/free outside the `DeviceBuffer` RAII owner live in this unit.)
