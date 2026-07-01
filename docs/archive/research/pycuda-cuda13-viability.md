# PyCUDA viability for steppe's Python access layer (step 2 / M(py-1)) — adversarial verdict

**Synthesized from 4 lenses (binding-mechanism, fit-vs-nanobind, interop/zero-copy, cost-benefit), with every load-bearing claim re-verified against primary sources on 2026-06-21.**

---

## Headline verdict

**(B) nanobind-only for the binding + a DLPack / `__cuda_array_interface__` interop seam on the nanobind module. No PyCUDA.**

PyCUDA is a CUDA *driver-API scripting* library. steppe launches no kernels from Python — all compute (f2 GEMMs, qpAdm/qpWave/rotation kernels, cuBLAS/cuSOLVER) is compiled C++/CUDA in `steppe_device`, orchestrated in C++. The binding's job is to call the finished public C++ API (`run_qpadm` / `run_qpwave` / `run_qpadm_search` in `include/steppe/qpadm.hpp`) and marshal numpy. PyCUDA cannot do that (it is not a C++ FFI), and the one thing it *can* contribute — a device-array handoff — is done better and with zero new dependencies by the very library steppe already plans to use. PyCUDA is, at best, redundant; at worst, a context-ownership liability.

---

## 1. What PyCUDA is, and its CUDA-13 viability

**What it is.** PyCUDA (Andreas Klöckner / inducer) exposes the **CUDA driver API** to Python: device allocation (`pycuda.driver.mem_alloc`, `GPUArray`), runtime kernel compilation (`SourceModule` → nvcc/NVRTC on source *strings*), module loading (`module_from_file` for cubin/PTX), kernel launch, streams, and **Python-owned CUDA contexts** (`pycuda.autoinit`, `make_default_context`). Its entire reason to exist is *writing, compiling, and launching CUDA kernels from Python.*

**It is maintained.** Latest release **2026.1, dated 2026-01-15** (PyPI; GitHub `v2026.1`, "15 Jan"). Recent tags: `v2025.1.3` (18 Feb), `v2025.1.2` (09 Sep 2025), `v2025.1.1` (08 Jun 2025), `v2025.1` (07 Feb 2025). Single maintainer (inducer77 / Andreas Klöckner). PyPI metadata still pins `Requires-Python ~=3.8` and gives only a generic `Programming Language :: Python :: 3` classifier (no per-version matrix). [pypi.org/project/pycuda, github.com/inducer/pycuda/releases]

**CUDA 13: CONFIRMED compatible (this corrects three of the four lenses).** The GitHub *release notes* — not the docs changelog — are the authoritative recency source, and they show two explicit CUDA-13 fixes:
- **v2025.1.2 (09 Sep 2025): "Add `#if` for cuCtxCreate compat break in CUDA 13"**
- **v2026.1 (15 Jan 2026): "Fix dll dir in CUDA 13.x"**

These address the real, ecosystem-wide CUDA-13.0 ABI break where `cuCtxCreate` was superseded by **`cuCtxCreate_v4`** taking a `CUctxCreateParams` struct (confirmed independently by cuda-api-wrappers #746, perftest #354, and NVIDIA dev-forum reports). So the earlier-lens claim "no primary source documents CUDA-13 support" was an artifact of reading the *stale published docs changelog* (which stops at the CUDA-5/6 era) and the PyPI page (which lists no CUDA versions at all). **Corrected finding: PyCUDA builds against and is patched for CUDA 13.**

**Residual CUDA-13 uncertainties (flagged, not fabricated):**
1. No primary source directly confirms a **PyCUDA-on-Blackwell sm_120** run. It rides a CUDA-13 toolkit, and CUDA 13 *does* target sm_120, so it is *expected* to work — but this is "expected, not directly verified." (The same gap applies to NVIDIA's own cuda.core.) steppe's Blackwell target is unaffected by CUDA 13.0's drop of pre-7.5 architectures.
2. **PyPI wheel coverage** for arbitrary CUDA-13 × Python combinations is not guaranteed; source builds need the toolkit + a C++ compiler. Ecosystem build churn around CUDA 13 / Python 3.13 is documented.
3. A version/date wrinkle: tag `v2025.1.3` shows "18 Feb" while `v2025.1.2` shows "09 Sep 2025" — the 2025.1.3 date is almost certainly Feb 2026. Not load-bearing, flagged for honesty.

---

## 2. Does it fit steppe's binding need? — No (the honest answer)

**The binding mechanism question is decisive and not close.**

- **PyCUDA cannot call a host C++ function.** It is a driver wrapper, not a C++/Python binding generator. It can load kernels only as `extern "C"`-named cubin/PTX modules or recompile source strings at runtime. steppe's compute is **internal, name-mangled C++** behind a host-side API (`run_qpadm`, `run_qpwave`, `run_qpadm_search`), itself wrapped around cuBLAS/cuSOLVER. There is **no PyCUDA mechanism** to reach that API. To bind steppe you would *still* need nanobind — so PyCUDA contributes nothing to the binding and merely adds a second moving part.

- **nanobind (ADR-0002) is the purpose-built, correct mechanism.** It calls C++ in-process and marshals numpy ↔ results — exactly the requirement. This is settled; PyCUDA does not compete for this role.

- **The two-context-owner problem (a real, serious conflict — reinforces rejection).** `pycuda.autoinit` / `make_default_context()` creates its **own** CUDA context and makes it current on the Python thread. steppe's C++ already owns its context, streams, and cuBLAS/cuSOLVER handles. CUDA device pointers and handles are **context-scoped**; a foreign current context on the calling thread redirects runtime-API calls and yields invalid-handle / "no current context" failures (a well-known bug class; cf. TensorFlow `cuCtxSetCurrent` #25220). NVIDIA's own guidance discourages multiple contexts per device. PyCUDA's GC-tied context teardown adds destruction-order hazards against C++ RAII.
  - The *only* safe coexistence mode is `pycuda.autoprimaryctx` (retain the runtime **primary** context, create nothing) — but that **neuters PyCUDA's sole value-add** (its own context + kernel launch). You would hand-coordinate two context-lifecycle regimes for **zero** functional benefit. Negative value.

**Bottom line for §2:** PyCUDA solves a problem steppe does not have (launch kernels from Python) and *fails* at the problem steppe does have (call compiled C++). nanobind-calling-C++ is correct.

---

## 3. The interop question — zero-copy GPU-array handoff: real need, but PyCUDA is the wrong vehicle

**Is the handoff worth a seam?** There is a *legitimate, architecturally-aligned* use case: a user with GPU-resident data (CuPy / PyTorch / numba / RAPIDS) feeding into steppe's precompute, or getting the f2 tensor back as a live GPU array — consistent with steppe's "keep f2 device-resident" principle (`DeviceF2Blocks` in VRAM). **But scope it honestly:** in the precompute-once / fit-many flow the **f2 directory is the interchange** and fits already read f2 from VRAM in C++, so there is *no host round-trip to eliminate* on the core path. The handoff helps a **narrower persona** (pre-existing GPU data → precompute; or keep resident f2 as a Python GPU tensor across interleaved fits). Reasonable to **defer it to post-M(py-1)** and build it on nanobind, not block the first binding on it.

**Is PyCUDA the right vehicle for the handoff? No.** The handoff is defined by **protocols**, not by any one library:
- **DLPack** (preferred) — its Python spec defines explicit producer/consumer **stream synchronization** semantics.
- **`__cuda_array_interface__` (CAI) v3** (fallback) — leaves lifetime/sync to the user; CuPy docs call CAI misuse "undefined behavior."

**nanobind already implements both, natively.** Per nanobind's own docs: `nb::ndarray` does **zero-copy** exchange with **NumPy, PyTorch, TensorFlow, JAX, CuPy, and MLX** via DLPack (the GPU-compatible generalization of the buffer protocol); supports the **`nb::device::cuda`** constraint; and can return framework-typed device tensors (`nb::cupy`, `nb::pytorch`). GPU device support landed in nanobind **2.2.0 (2024-10-03)**; versioned DLPack / `nb::array_api` in **2.10.1 (2025-12-08)**. Crucially, nanobind has **no build-time dependency on CUDA** — so steppe gets the interop seam **from the binding it already uses, with zero new dependencies.**

PyCUDA's only contribution here would be `GPUArray.__cuda_array_interface__` (a v3 dict, `stream: None`, **no `__dlpack__`**) — a strict **subset** of nanobind's capability, that *also* drags in a second context. As an interop hub it is strictly less capable.

**The one genuine engineering item (independent of PyCUDA).** nanobind treats the device as **"mere metadata" and performs NO automatic stream synchronization** at the DLPack/CAI boundary (its words: it "knows neither the stream that produced the data nor the runtime needed to act on it"). So steppe's marshalling layer must coordinate the caller's stream with steppe's own (e.g. `cudaStreamWaitEvent` / event sync) when wrapping or handing out device pointers. **That — not a Python CUDA stack — is where the interop validation effort belongs.** Recommend a small spike to verify the "wrap external VRAM pointer + owner-capsule deleter → return as CuPy/PyTorch" recipe before relying on it (nanobind's changelog implies but does not explicitly document wrapping an *externally-allocated* device pointer with a custom-deleter capsule — flagged).

---

## 4. Verdict and what to add to the M(py-1) plan

### VERDICT: (B) nanobind-only binding + a DLPack/CAI interop seam on the nanobind module. **No PyCUDA, in-process, ever.**

| Lens | Question | Verdict |
|---|---|---|
| (a) Binding mechanism | Can PyCUDA bind steppe's C++ API? | **NO — wrong tool.** Driver wrapper, not a C++ FFI; can't call `run_qpadm`. nanobind (ADR-0002) is correct. |
| (b) Interop / zero-copy handoff | Is a device-array seam worth it, and is PyCUDA the vehicle? | Seam = **real but secondary** (defer post-M(py-1)). Vehicle = **DLPack/CAI via nanobind**, not PyCUDA (subset, no DLPack). |
| (c) CUDA-13 + context ownership | Is PyCUDA viable / safe alongside steppe? | CUDA-13 **confirmed compatible**, but **context-owner conflict** = liability for zero upside. |

**What to add to the M(py-1) plan:**
1. **Confirm nanobind for the binding** (ADR-0002 stands). The binding calls the public C++ API and marshals numpy. No PyCUDA dependency.
2. **Reserve a DLPack/CAI interop seam on the nanobind module, prefer DLPack** (CAI as fallback). Realize via `nb::ndarray` with `nb::device::cuda`; no new build-time deps.
3. **Defer the seam to post-M(py-1)** unless the GPU-data-in / resident-f2-tensor-out persona is an early requirement. The core precompute→fit path has no host round-trip to remove.
4. **Make stream synchronization the explicit validation target** of the seam: nanobind does no auto-sync, so the marshalling layer must coordinate streams/events. Spike the "wrap external VRAM pointer + owner capsule" recipe first.
5. **If raw CUDA-from-Python is ever genuinely needed later (it is NOT for M(py-1))**, the maintenance-correct choice is **NVIDIA's first-party cuda-python / `cuda.core` / `cuda.bindings`** (latest **13.3.1, 2026-05-29**; stable **CUDA Python 1.0** shipped with CUDA 13.3), which NVIDIA explicitly built to let third parties stop maintaining their own wrappers ("access … could only be accomplished by means of third-party software such as Numba, CuPy, Scikit-CUDA, RAPIDS, **PyCUDA**, PyTorch, or TensorFlow"). The legacy third-party PyCUDA is the path NVIDIA is routing *around* — not the choice to standardize on.

---

## 5. Sources (accessed 2026-06-21)

- PyCUDA on PyPI (latest 2026.1, 2026-01-15; `Requires-Python ~=3.8`): https://pypi.org/project/pycuda/
- PyCUDA GitHub releases (CUDA-13 fix commits: v2025.1.2 "cuCtxCreate compat break in CUDA 13"; v2026.1 "Fix dll dir in CUDA 13.x"): https://github.com/inducer/pycuda/releases
- PyCUDA docs (driver API, SourceModule, GPUArray, context model — note: published CUDA changelog is stale): https://documen.tician.de/pycuda/
- CUDA 13.0 `cuCtxCreate` → `cuCtxCreate_v4` ABI break (corroboration): https://github.com/eyalroz/cuda-api-wrappers/issues/746 ; https://github.com/linux-rdma/perftest/issues/354
- nanobind `nb::ndarray` docs (DLPack + CAI zero-copy with NumPy/PyTorch/TF/JAX/CuPy/MLX; `nb::device::cuda`; `nb::cupy`/`nb::pytorch`; no build-time CUDA dep; **no auto stream sync — device is "mere metadata"**): https://nanobind.readthedocs.io/en/latest/ndarray.html
- CuPy interoperability (CAI vs DLPack; CAI misuse "undefined behavior"): https://docs.cupy.dev/en/stable/user_guide/interoperability.html
- NVIDIA "Unifying the CUDA Python Ecosystem" (official first-party stack; names PyCUDA as third-party software to route around): https://developer.nvidia.com/blog/unifying-the-cuda-python-ecosystem/
- NVIDIA cuda-python (cuda.core / cuda.bindings; latest 13.3.1, 2026-05-29): https://pypi.org/project/cuda-python/ ; https://github.com/NVIDIA/cuda-python/releases ; https://nvidia.github.io/cuda-python/
- CUDA Python 1.0 stable shipped with CUDA 13.3 (2026): https://www.phoronix.com/news/NVIDIA-CUDA-13.3-Released
