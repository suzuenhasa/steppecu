I read `qpgraph_fit_kernels.cu` carefully. This is **not slop** — it's clearly written by someone who understands the qpGraph math and the host oracle they're porting. But a senior C++/CUDA reviewer would flag a handful of real issues.

## What's genuinely good

- **Bit-exact parity intent is explicit and credible.** The constants are single-sourced from `core/qpadm/qpgraph_opt_constants.hpp` (line 30, 44), and the splitmix init is lifted directly into `d_init_theta` (lines 302–310). The comments repeatedly cite host parity, which is the right obsession for this kind of accelerator port.
- **Native FP64 inner solve.** The ridge + AT2 scaling + NNLS/LU path in `d_qpgraph_score` (lines 256–283) mirrors the host reference and avoids the "approximate the GLS on the GPU" trap.
- **Heterogeneous-topology batch fleet is architecturally thoughtful.** Reconstructing `QpGraphDeviceTopo` per-thread from packed arena offsets (lines 449–456) is exactly how you'd avoid rebuilding kernels per topology during a graph search.
- **RAII scratch via `DeviceBuffer`** in the host launchers (lines 485–486, 506–507) instead of raw `cudaMalloc`/`cudaFree` pairs.
- **Defensive `kMaxThetaDev` guard on both sides:** host throws before launch (lines 477–482), device returns a sentinel (line 333). That's correct layered defense.

## What a senior developer would flag

**The `1e30` score sentinel is slop-adjacent:**

```cuda
279: if (!d_nnls(ccs, q1, ne, bl, ...)) return 1e30;
281: if (!d_solve(ccs, ne, q1, bl, ...)) return 1e30;
333: if (D > kMaxThetaDev) return 1e30;
```

A single magic double (`1e30`) is overloaded for "NNLS failed," "LU singular," and "topology too big." The host maps it to `NonSpdCovariance`, but the kernel can't distinguish the cases. Worse, it's in the same range as legitimate but bad scores for large data. An enum/flag field in the output, or at least a named constant like `kQpGraphScoreInvalid`, would make this not embarrassing in a showcase.

**Heavy per-thread stack arrays inside the optimization hot loop:**

```cuda
343: double thp[kMaxThetaDev], thm[kMaxThetaDev];
...
359: double thn[kMaxThetaDev];
```

These are allocated per-dimension, per-iteration, per-thread, and the code copies all `D` elements each time (lines 344, 360). For small `D` it's fine, but it bloats register pressure and local memory right where the kernel is supposed to be tight. A senior reviewer would ask why `thp`/`thm`/`thn` aren't just two/three temporaries mutated in place rather than full `kMaxThetaDev`-sized arrays copied on every score eval.

**Three full GLS evaluations per dimension per Newton step:**

Lines 335–346 do center + two finite-difference probes; then line 362 does a probe step; then the backtracking loop (364–367) can call `d_qpgraph_score` up to `kMaxBacktrack` more times. That's potentially 4+ full `(fill_pwts → ppwts → Wm → cc → ridge → solve → residual)` pipelines per dimension per iteration. The comments frame this as "the PRODUCTIZED IDEA-1" design, which is fair, but the code doesn't expose any cheaper gradient path or reuse partials between `s`, `sp`, and `sm`. For a job-application showcase, I'd expect at least a TODO or a note about the asymptotic cost.

**LU factorization uses an exact-zero pivot test:**

```cuda
57: if (best == 0.0) return false;
```

In FP64 this is numerically naive. Partial pivoting is already implemented, but the singular check should be `best <= eps * norm` or similar, not bitwise zero. Same file that worries about AT2 ridge conditioning should worry about this.

**`DeviceBuffer` allocations are likely synchronous on the default stream:**

```cuda
485: DeviceBuffer<double> g_dbl(...);
```

The launch wrappers accept a `cudaStream_t`, but `DeviceBuffer` here is constructed without a stream argument. If it's backed by `cudaMalloc` (not an async pool allocator), these allocations serialize on the default stream and quietly defeat the async intent of passing `stream` in. A senior reviewer would want to see `DeviceBuffer` constructed with the same stream, or a comment explaining why it's safe.

**The batch launcher manually mutates `ScratchLayout` fields:**

```cuda
528: ScratchLayout Lmax{};
529: Lmax.dbl_total = dbl_per_thread;
530: Lmax.int_total = int_per_thread;
```

`ScratchLayout` apparently has no constructor that takes totals, or the author didn't use it. This is a small API smell — the caller already computed `dbl_per_thread` via `make_layout`, then passed raw ints to be re-stuffed into a struct. Why not pass the `ScratchLayout` directly?

**Launch configuration is hardcoded and minimally tuned:**

```cuda
487: const int TPB = 64;
...
532: const int TPB = 64;
```

64 threads per block for a kernel that is entirely serial within the thread and FP64-heavy is almost certainly suboptimal. Occupancy will be limited by registers and local memory, not by parallelism. There's no mention of occupancy exploration, and the fleet kernel has no shared-memory use that would justify 64. A senior CUDA reviewer would ask for occupancy numbers or at least a sweep over 64/128/256.

**The `d_fill_pwts_centered` unique-cell scan is O(n_pe²):**

```cuda
185–206: for each a, scan all b < a, then scan all b >= a ...
```

The comment admits the table is path-ordered and not cell-grouped, then implements a brute-force duplicate check. For small `n_pe` it passes, but it's the kind of quadratic micro-inefficiency that makes another senior dev say "ugh" when the rest of the file is supposed to be productionized.

**Over-commented and ALL-CAPS in places.** Lines 2–28 read like a design doc, not a file header. Comments like "fail LOUDLY" (line 331) and "the SAME device body" (line 407) are fine informally, but in a showcase file they make the code look like it was written to impress rather than to be maintained. Some comments also state the obvious, e.g. line 392–393 explaining why `Dw` clamps.

## The "slop" test

**Not slop.** The math is right, the host-parity story is consistent, memory is RAII-managed, error paths exist, and there are no obviously stale copy-paste comments. The rough edges are real, but they're the kind you get from a competent developer porting a complex host algorithm to CUDA, not from slapdash coding.

## What it actually looks like

This looks like **solid research-engineering code by a domain expert who knows the qpGraph algorithm cold and knows enough CUDA to get it on the GPU correctly.** The design choices (one thread per restart, native FP64 inner solve, batch topology reconstruction) are reasonable for the problem. What it doesn't look like is code that's been through a GPU performance pass or a strict C++ API review: register pressure, allocation streams, launch tuning, and numerical edge cases are all taken for granted.

A senior CUDA specialist would say: "Correct direction, but before this is the hot path I'd profile register usage, try larger blocks, and see if I can cut the per-dimension score evaluations." A senior C++ person would say: "Stop using `1e30` as your universal error value, give `ScratchLayout` a real constructor, and calm the comments down."

**Verdict:** B. Ship after replacing the `1e30` sentinel, auditing `DeviceBuffer` stream usage, and running an occupancy/launch-config sweep.