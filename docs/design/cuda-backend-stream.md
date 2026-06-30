# The CudaBackend statistic `Stream` (non-blocking, per-device; teardown ordering)

> Source: `src/device/cuda/cuda_backend.cu` — the `Stream stream_{}` member of `CudaBackend`.
> Cross-refs: architecture.md §7 (RAII teardown ordering), §11.4 (multi-GPU execution), §12
> (single-stream-per-device determinism). The load-bearing teardown / construction order
> invariant is retained inline at the member; the rationale lives here.

The ONE statistic stream PER DEVICE for bit-stability (architecture.md §12
single-stream-per-device determinism rule). An OWNING, non-blocking RAII `Stream`
(stream.hpp; `cudaStreamNonBlocking`), created in the ctor AFTER `device_id_`'s
initializer made this device current — so the stream is associated with `device_id_` (CUDA
Runtime API: a stream binds to the device current at create time). Every launch,
`cudaMemcpyAsync`, and the trailing `cudaStreamSynchronize` route through `stream_.get()`.
The handle is bound to this stream + the workspace ONCE in the ctor and is never
re-`cublasSetStream`'d (cleanup X-1/B1), so the emulated-FP64 determinism workspace persists
for every GEMM (cuBLAS §2.4.7).

## WHY NON-BLOCKING (M4.5 SPMG, perf-discovery.md P2/F1)

The prior member was the NULL *legacy default* stream and the build is NOT compiled
`--default-stream per-thread`, so under the §11.4 multi-GPU fan-out the two per-device
worker threads' launches implicitly serialized against the single process-wide legacy
default stream (CUDA C Programming Guide §3.2.8.5.2 "Default Stream") — the measured 18%
kernel overlap. A `cudaStreamNonBlocking` stream does NOT implicitly synchronize with the
legacy default stream (CUDA Runtime API, `cudaStreamCreateWithFlags`), so each device's
backend now issues on its own independent lane and the two devices' GEMMs can overlap. This
is a pure scheduling change: stream choice moves no arithmetic bits, so §12 parity is
unaffected. §12 mandates ONE stream PER DEVICE on the statistic path — this single
per-device non-blocking stream satisfies it (we do NOT add a second statistic stream).

## Declaration order is load-bearing at teardown (reverse-order destruction)

1. `workspace_` declared AFTER `blas_` so it is freed FIRST — `blas_` holds a NON-owning
   pointer into it; `cublasDestroy` only synchronizes (it does not read the workspace), so
   freeing the workspace VRAM before the handle is destroyed is safe (architecture.md §7).
2. `stream_` declared BEFORE `blas_` so it is destroyed AFTER it — the handle's cuBLAS
   context is bound to this stream, so the stream must outlive `cublasDestroy` (which
   synchronizes the bound stream); destroying the stream only after the handle is gone
   avoids tearing down a stream the handle still references (architecture.md §7 RAII
   teardown ordering).

Construction order (declaration order) is also load-bearing: `device_id_` is declared
FIRST, and its initializer makes the device current, so both `stream_` (created on the
current device) and `blas_`'s cuBLAS context (cuBLAS §2.1.2) bind to `device_id_`.
