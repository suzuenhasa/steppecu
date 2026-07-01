# Group 20 — Performance: memory access — ROLLUP

Scope = kernel. Tasks: 20.1 uncoalesced global access · 20.2 shared-memory bank conflicts · 20.3 re-reading the same global value instead of caching.

FP64/§12 context applied: native + Ozaki emulated FP64, single statistic stream, deterministic reductions are intentional and were NOT flagged. Findings are layout/access-efficiency only — none touch math or parity.

## 1. Coverage

13 units in scope. 4 with findings, 9 clean.

| Unit | Result |
|---|---|
| src__device__cuda__decode_af_kernel | 2 MED |
| src__device__cuda__f2_block_kernel | 1 MED / 2 LOW |
| src__device__cuda__f2_blocks_kernel | 1 MED / 2 LOW |
| src__device__cuda__qpadm_fit_kernels | 2 MED / 1 LOW |
| src__device__cuda__block_sink | clean |
| src__device__cuda__check | clean |
| src__device__cuda__cuda_backend | clean |
| src__device__cuda__device_buffer | clean |
| src__device__cuda__device_f2_blocks | clean |
| src__device__cuda__device_partial | clean |
| src__device__cuda__f2_blocks_out | clean |
| src__device__cuda__p2p_combine | clean |
| src__device__cuda__pinned_buffer | clean |

The 9 clean units are host-side orchestration / RAII / transport / declaration-only TUs with no `__global__` kernels, no `__shared__`, and no thread-strided global access — all three tasks are structurally out of scope (vacuous). The 4 with-findings units are exactly the four kernel-bearing TUs.

## 2. Counts by task + severity

| Task | HIGH | MED | LOW | total |
|---|---|---|---|---|
| 20.1 uncoalesced global | 0 | 3 | 3 | 6 |
| 20.2 shared bank conflict | 0 | 0 | 0 | 0 |
| 20.3 re-read global | 0 | 2 | 2 | 4 |
| **total** | **0** | **5** | **5** | **10** |

- 0 HIGH, 5 MED, 5 LOW = 10 findings.
- 20.2 produced ZERO findings across all units: no multi-element `__shared__` array exists anywhere; the only shared decls are single broadcast scalars (qpadm_fit_kernels `s_tr` :1057), which have no bank-conflict surface.

## 3. Top findings (HIGH first)

No HIGH. The MED findings, ranked by scale impact:

1. [20.1][MED] f2_block_kernel.cu:112-133 (`f2_feeder_kernel`) — SNP axis `s` on `threadIdx.x` but the `[P×M]` buffers are pop-contiguous (`idx = i + Pl*s`), so adjacent lanes stride by P doubles ⇒ fully uncoalesced loads AND stores on the bandwidth-bound feeder (the kernel where coalescing actually costs throughput). Orientation is forced by the grid.y 65535-cap fix (M must ride x), so the trade is real. Suggested: row-major / `[M×P]` feeder buffers, or a 16×16 shared-memory tile-and-transpose on store.

2. [20.1][MED] qpadm_fit_kernels.cu:1104-1188 (`qpadm_fit_models_kernel`) — one-thread-per-MODEL full-fit kernel; every load/store is model-major so adjacent lanes stride by a full per-model slice (`total[j+nr*i]` stride m, `qinv` stride m·m, all outputs stride nl/(rmax+1)/(nl+1)). At n_models in the millions (S8 rotation) this is the heaviest kernel reading/writing the largest arenas with the worst per-lane access efficiency. Suggested: struct-of-arrays / element-major-across-models (transposed) arena, or block-per-model mapping. Layout-only, no parity change.

3. [20.1][MED] decode_af_kernel.cu:81-85 — Q/V/N output writes are strided: column-major `[P×M]` with `off = i + P·s`, warp rides `s`, so the 3 stores scatter 32 lanes across 32 cache lines — triples store traffic at P·M scale. Reads coalesce on the SNP axis but writes anti-coalesce on it. Suggested: row-major-over-warp-axis output or staged coalesced burst; else document the column-major store stride as an accepted f2-contract cost.

4. [20.1][MED] f2_blocks_kernel.cu:152,154 (`assemble_blocks_group_kernel`) — transposed Rg reads (`sj + si*twoP`) stride by 2P (~5000 doubles ≈ 40 KB at P=2500) per lane ⇒ uncoalesced; mathematically required (symmetric f2 needs both i-row and j-row column sums). Suggested: stage a padded Rg-row tile in shared memory (block already reads a square tile), or accept if assemble is off the GEMM-dominated critical path.

5. [20.3][MED] qpadm_fit_kernels.cu:996-1020 / 429-456 (`f4_loo_total*`) — each thread writes `dLoo[...]` then re-reads that just-written cell twice more across the tot_line and est passes (3 global touches per loo cell). Register-caching is NOT viable (nb up to 757 doubles would spill; dLoo is a required output). Read-after-write hits L2 not DRAM, so cost is L2 bandwidth. Suggested: optionally fuse the three passes so each loo[k,b] is consumed where produced — must verify FP64 bit-parity (§12), fused order may differ.

LOW (hygiene, near-certain compiler CSE / broadcast already absorbs):
- [20.1][LOW] f2_block_kernel.cu:168,170 — two transposed R reads in `assemble_f2_kernel`, but small `[2P×P]` buffer off the dominant cost.
- [20.3][LOW] f2_block_kernel.cu:120,124,128 — `Q_raw[idx]` read twice (`__restrict__` ⇒ likely single LDG).
- [20.3][LOW] f2_blocks_kernel.cu:89-90,98 — per-slab scalars re-read by every thread (block-broadcast, L1-absorbed).
- [20.3][LOW] f2_blocks_kernel.cu:149-150,161-162 — index expr `si+sj*Pp` recomputed (register CSE, not a global re-read).
- [20.1][LOW] qpadm_fit_kernels.cu:1212-1214 (`qpadm_loo_models_kernel`) — model/block-major strided loo gather, bounded by SMALL bucket (m<=50), low stakes.

## 4. Cross-cutting pattern

The single dominant pattern is **column-major `[P×M]` (or `[2P×P]`) f2/decode arenas read by a warp riding the SNP/model axis** — i.e. the warp's fast axis is NOT the unit-stride (population) dimension, so every load/store strides by P (or 2P, or a per-model slice). It recurs in decode_af, the f2 feeder, the f2_blocks assemble transpose, and the qpAdm model-fit kernels. In two cases the orientation is FORCED and load-bearing (the grid.y 65535-cap fix makes M ride the x axis), so it is a deliberate, documented bandwidth trade, not a free win — the consistent recommended remedy is a shared-memory tile-and-transpose rather than reorienting the grid. None are correctness bugs; all are scale-dependent throughput on the largest arenas (P up to 2500, M up to ~584k, n_models in the millions). Secondary pattern: read-after-write re-reads of just-written `dLoo` cells in the fit, gated to L2 — fusable only if FP64 accumulation order is bit-validated. Notably, 20.2 (bank conflicts) is empty project-wide: these kernels use per-thread registers / per-thread VRAM-arena slices rather than multi-element shared tiles, so there is no shared-memory serialization surface.

## Headline
13 units (4 with findings, 9 clean) · 10 findings (0 HIGH / 5 MED / 5 LOW) · 0 HIGH.
