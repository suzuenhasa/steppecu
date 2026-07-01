# Group 15 — Memory: transfers — ROLLUP SUMMARY

Scope: `src/device` (TIER 0/1 device layer). Tasks reviewed:
- 15.1 — `cudaMemcpy` (H<->D) inside a loop that should be hoisted/batched/kept-resident.
- 15.2 — Direction enum not matching the actual transfer.
- 15.3 — Pageable host memory for frequent transfers where pinned ~doubles bandwidth.

## 1. Coverage

| | Count |
|---|---|
| Units in scope (scope=device) | 25 |
| Units reviewed (`## Group 15` present) | 25 (100%) |
| Clean (no findings) | 22 |
| With findings | 3 |

Clean units (22): backend, backend_factory, cpu/cpu_backend, cuda/block_sink, cuda/check, cuda/decode_af_kernel, cuda/device_buffer, cuda/device_partial, cuda/f2_block_kernel, cuda/f2_blocks_kernel, cuda/handles, cuda/p2p_combine, cuda/pinned_buffer, cuda/qpadm_fit_kernels, cuda/stream, f2_disk_format, host_ram, resources, shard_plan, stream_f2_blocks, tier_select, vram_budget.

Most clean units are CUDA-free headers (interfaces, POD descriptors, host-pure policy) or device-resident kernel TUs that own zero H<->D transfers — by design, all transfer code is concentrated in `cuda_backend.cu` and the f2-blocks transport TUs.

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | Total |
|---|---|---|---|---|
| 15.1 (in-loop memcpy / not resident) | 0 | 2 | 0 | 2 |
| 15.2 (direction enum mismatch) | 0 | 0 | 0 | 0 |
| 15.3 (pageable for frequent transfers) | 0 | 1 | 2 | 3 |
| **Total** | **0** | **3** | **2** | **5** |

15.2 came back fully clean across all units: every transfer's direction enum matches its actual src/dst space (full-file audits in cuda_backend, f2_blocks_out, device_f2_blocks confirmed no transposed H2D/D2H/D2D, no `cudaMemcpyDefault` ambiguity).

## 3. Top findings (HIGH first)

No HIGH findings.

### MED
- [15.1][MED] `src/device/cuda/cuda_backend.cu:2270-2280` — In `fit_chunk` the column-by-column batched-inverse loop runs up to `m` (=nl·nr) iterations and EACH iteration does a fresh per-column H2D `cudaMemcpyAsync` of the B per-model RHS pointers (`:2273`). The pointers are fully deterministic, so all `m` pointer arrays could be uploaded with ONE H2D before the loop. The `:2267` comment ("precompute all m, H2D once each lazily") over-promises vs. the per-iteration upload. Parity-neutral; a scale/latency cost on the S8 per-chunk hot path. Suggested: hoist to a single `DeviceBuffer<double*> dBptrAll(m*B)` filled once and uploaded once, index `+c*B` per column (mirrors the single `dAptr` upload at `:2254`).
- [15.1][MED] `src/device/cuda/f2_blocks_out.cu:78,93-96` — `read_block_to_host` is the fit's per-block tile reader: each call does TWO synchronous D2H copies (f2 + vpair) for ONE block, and the fit consumes block-by-block (n_block up to ~757) — the "memcpy in a loop, should be kept-resident/batched" anti-pattern, and the exact f2_blocks D2H bounce flagged as the multi-GPU speed wall. Suggested: provide a resident-consuming fit path reading `resident.f2_device()`/`vpair_device()` in place (no D2H), or expose a batched multi-block read that issues one larger/strided D2H.
- [15.3][MED] `src/device/cuda/f2_blocks_out.cu:93-96` (and `:19`) — the Resident-arm per-block D2H copies into PAGEABLE caller buffers (`:78`), at ~half pinned bandwidth, multiplied by 15.1's per-block frequency. The TU even includes `pinned_buffer.cuh` with a "pin the D2H; graceful degrade" comment (`:19`) but never uses it. Suggested: stage the D2H through a pinned/`RegisteredHostRegion` region (graceful-degrade), or drop the dead include/comment if the perf path is abandoned.

### LOW
- [15.3][LOW] `src/device/cuda/cuda_backend.cu:1962-1967, 2031-2033` (and other fit-path H2D/D2H sites) — qpAdm-fit intermediates cross the seam via PAGEABLE `std::vector`s. Correctly left pageable for KB-scale small fits, BUT `dLoo`/`x_loo` (S7) and the S8 `dLoo`/`dQinv` arenas scale as m·nb (nb up to ~757) × model batch B, growing into MB-to-low-GB at the rotation/large-model envelope while riding pageable host memory (no overlap under §11.4 fan-out). Latent / scale-dependent. Suggested: if profiling at the S8 envelope shows these on the critical path, amortize-pin the x_loo/Qinv source like the f2 `pinned_in_`/`stage_*_` seam; otherwise leave pageable.
- [15.3][LOW] `src/device/cuda/device_f2_blocks.cu:86-89` — asymmetry: `upload_f2_blocks_to_device`'s H2D reads from PAGEABLE host source while the mirror `to_host` PINS its D2H destinations (`:55-56`). In `replicate_f2` the same host tensor is the stable, reused base pointer across all G-1 upload iterations (the case the PinnedRegistryCache rationale favors pinning). Suggested: optionally wrap the H2D source in a `RegisteredHostRegion` (mirror `:55-56`) — but this whole no-peer host-bounce is already DEFERRED (real fix = per-device precompute removing the transfer), and a single-shot register tax may not amortize, so documented-tradeoff LOW, not a bug.

## 4. Cross-cutting pattern

The single recurring theme is the **fit / f2-blocks transfer seam at scale**, not correctness:

1. **The f2_blocks D2H bounce is the load-bearing issue.** Both MED 15.1 findings (cuda_backend `fit_chunk` per-column pointer uploads; f2_blocks_out `read_block_to_host` per-block D2H) and one LOW (device_f2_blocks upload asymmetry) all converge on the same architectural wart already in MEMORY: the real cure is a fully resident/device-resident fit reading f2_blocks from VRAM with zero per-block D2H, rather than tuning the transfers. The reviews confirm the hot path is *mostly* already resident (M0/M4 amortized `pinned_in_`, resident f2/Vpair moved out, fit uploads hoisted above per-rank/per-block loops) — these findings are the remaining per-block/per-column fragments.
2. **Pinned/pageable split is sound where amortizable.** Reviewers repeatedly validated the intentional design: stable reused H2D inputs (Q/V/N) are amortized-pinned via `PinnedRegistryCache`, large D2H stages through persistent `PinnedBuffer`, and fresh per-call result vectors are deliberately pageable (measured page-lock tax with zero amortization). The 15.3 findings are exactly the spots where this discipline is *not yet* applied to scale-sensitive paths (fit x_loo/Qinv arenas; the f2_blocks upload source) — all latent/scale-dependent, none a present bug.
3. **`pinned_buffer.cuh` included-but-unused in f2_blocks_out** — a recognized-but-unrealized perf intent (also filed under hygiene/dead-include in other groups).

No HIGH-severity transfer bug, no direction-enum mismatch, no UB/race in the transfer layer.

---

## Headline numbers

- **Units in scope / reviewed:** 25 / 25 (22 clean, 3 with findings)
- **Total findings:** 5 (0 HIGH, 3 MED, 2 LOW)
- **#HIGH:** 0
