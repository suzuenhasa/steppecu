# `exit_code_for_caught.hpp` reference

## 1. Purpose

`src/app/exit_code_for_caught.hpp` is a one-function bridge that turns a caught
`std::exception` into the process exit code the command-line tool should return to
the shell. It exists to answer one narrow question at the top of every command: was
this exception a genuine device out-of-memory fault, or something else?

Its whole job is a single reclassification. Without it, a command's top-level
`catch (const std::exception& e)` would have to `return kExitRuntimeError` (exit
code `5`, the catch-all) for *every* exception, lumping a real GPU out-of-memory
condition in with unrelated runtime errors. This function separates the two: a real
device allocation failure exits with `kExitDeviceOom` (exit code `3`) instead, so a
calling script can tell "the GPU ran out of memory" apart from "something else went
wrong" and branch on it. Everything else keeps the catch-all `5`.

The header is deliberately free of any CUDA code. The whole `src/app` layer is
CUDA-free, and this file has to stay that way while still producing a GPU-specific
exit code. Sections 2 and 3 explain how it manages that.

---

## 2. The layering rule — why this file cannot classify the fault itself

The exceptions that actually carry GPU-error information — the device library's
typed exceptions `CudaError`, `CublasError`, and `CusolverError`, and the low-level
status codes they wrap (`cudaError_t`, `cublasStatus_t`, `cusolverStatus_t`) — are
**private to the device library**. They are declared in a CUDA header (`.cuh`) that
the app layer never includes, and a build-time check enforces that the app stays
CUDA-free. So the app physically cannot name those types or inspect those status
codes.

That means the actual "is this an out-of-memory fault?" test cannot live in this
file. It is delegated to a separate function, `device::device_fault_status`, which:

- is **declared** in `device/resources.hpp` with a CUDA-free signature (it takes a
  plain `const std::exception&` and returns a plain `std::optional<Status>`), so the
  app can call it without seeing any CUDA type; and
- is **defined** inside the device library's CUDA translation unit, where it is
  allowed to `dynamic_cast` the exception to the typed device exceptions and read
  their status codes.

The app already links the device library, so the symbol resolves at link time. The
result is a clean seam: the CUDA-specific classification happens on the device side
of a CUDA-free interface, and this app-side function only consumes its verdict.

---

## 3. What the function does

`exit_code_for_caught(const std::exception& e)` is pure dispatch — three lines with
no branches of its own beyond one `if`:

1. Call `device::device_fault_status(e)`. This returns a `std::optional<Status>`:
   a `Status` value if the exception was recognized as a device fault it can
   classify, or empty (`std::nullopt`) if it was not.
2. If a status came back, map it to an exit code with
   `config::exit_code_for(*status)` and return that.
3. If nothing came back, return `config::kExitRuntimeError` (the catch-all, `5`).

`device_fault_status` recognizes a fault by attempting a `dynamic_cast` to each of
the three device exception types in turn and, on a match, checking whether the
wrapped status code is specifically the *allocation-failure* code for that library
(`cudaErrorMemoryAllocation`, `CUBLAS_STATUS_ALLOC_FAILED`, or
`CUSOLVER_STATUS_ALLOC_FAILED`). Only those allocation failures produce a status
(`Status::DeviceOom`); any other device error, and any exception that is not one of
those three types at all, produces empty.

So in practice the only non-empty result flowing through step 2 is
`Status::DeviceOom`, which `exit_code_for` maps to `kExitDeviceOom` (`3`). The code
is nonetheless written generically — it maps whatever status comes back through the
shared `exit_code_for` mapping rather than hard-coding `3` — so it stays correct if
the device classifier ever learns to report additional statuses.

---

## 4. The reclassification contract — which faults become exit code 3

The precise contract is: **only a genuine device (VRAM) allocation failure is
reclassified from `5` to `3`. Everything else keeps `5`.** Two boundary cases are
worth stating explicitly, because both are easy to get wrong:

- A **host** `std::bad_alloc` — running out of ordinary system RAM, not GPU
  memory — is *not* a device allocation failure. It is not one of the three device
  exception types, so `device_fault_status` returns empty and the exit code stays
  the catch-all `5`. That is correct: it was not the GPU that ran out of memory.
- Every **non-allocation** device fault — any `CudaError`, `CublasError`, or
  `CusolverError` whose wrapped status is something other than the allocation-failure
  code — also returns empty and keeps `5`. Only the allocation-failure status codes
  are treated as out-of-memory.

The net effect on exit codes is a single, targeted `5 → 3` reclassification for real
GPU out-of-memory conditions, and no change for anything else.

---

## 5. Guarantees and where it is called

The function is marked `noexcept` and does no allocation of its own — it is safe to
call from an exception handler, which is exactly where it runs. It is `[[nodiscard]]`
so a caller cannot accidentally ignore the exit code it computes.

Every command's top-level `catch (const std::exception& e)`, and the tool's
`main()`, calls this in place of a hard-coded `return kExitRuntimeError;`. The
pattern is: catch the exception, pass it here, and return whatever exit code comes
back. That single substitution is what gives the whole command-line tool an honest
fault taxonomy — a distinct exit code for GPU out-of-memory — while keeping the app
layer entirely free of CUDA.
