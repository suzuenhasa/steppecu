# steppe — Next Work (post-AT2-parity)

*Created 2026-06-21, after the AT2 end-to-end parity arc closed. `main @ ae80628`.*

## Where we are
- **Full AT2 end-to-end parity** on real AADR v66 (raw genotypes → decode → f2 → qpAdm), diploid + pseudo-haploid, to the emulated-FP64 floor. The 4-fix chain: `--blgsize` = Morgans + monomorphic-drop → AT2 SNP-anchored block partition → per-sample pseudo-haploid `adjust_pseudohaploid`. (See `docs/research/{tgeno-at2-support,block-partition-at2,f2-estimator-at2}.md`.)
- **All 5 goldens correct** (regenerated from the convertf-PA), steppe reproduces each at tier, **THOROUGH ctest 45/45**.
- **Haak 2015 reproduced** (`docs/studies/haak2015.md`).
- **CLI works**: `steppe extract-f2` (genotypes → f2 dir) + `steppe qpadm` (the study flow). Built with `-DSTEPPE_BUILD_CLI=ON`.

Correctness is settled. What remains is **scale/perf, productization, and breadth** — not parity.

---

## 0. Scale / perf — wall-clock sweep on 1240K **(DOING FIRST — user priority)**
Now that parity is rock-solid, characterize the throughput envelope on the **large** panel (the real target; the HO golden is tiny by comparison — memory `design-for-scale-not-smallest-model`).
- Data: `/workspace/data/aadr/1240k/v66.p1_1240K.aadr.patch.PUB` (TGENO, ~1.2M SNPs, ~23k ind, 6.7 GB geno) — **on the box, complete**.
- **Wall-clock sweep for f2 and the other stages**: time `extract-f2` (decode → f2 blocks) across increasing pop-set sizes, then the qpAdm fit + the S8 rotation. Report a per-stage × size table, where the time goes (decode vs f2 GEMM vs fit), VRAM, and scaling.
- **MUST be the Release build** (`build-rel`, `-DSTEPPE_BUILD_CLI=ON`) — the debug per-kernel `cudaDeviceSynchronize` voids all timing (memory `perf-bench-release-build`). **REAL data only** (memory `real-data-only-all-results`).
- Workflow: `agentscripts/sweep-1240k-wallclock.js`. Results → `docs/perf/1240k-sweep.md`.
- Then: the **2M** panel (download pending), and the multi-GPU crossover (memory `m45-*`).

## 1. Finish Step 2 — Productization (CLI + bindings)
The engines exist and are golden-gated; this is wiring + packaging.
- **M(cli-2) `qpwave` CLI** — small; mirror the working `qpadm` subcommand (`run_qpwave` exists, golden-gated).
- **M(cli-3) `qpadm-rotate` CLI** — small; `run_qpadm_search` exists.
- **M(py-1) Python bindings** — bigger. **nanobind** (NOT PyCUDA — `docs/research/pycuda-cuda13-viability.md`) + a **DLPack / `__cuda_array_interface__`** interop seam. MUST = results→pandas + a Q/V/N array entry; flagship = the msprime→steppe→qpAdm power-analysis loop (`docs/research/interop-usecases.md`). **GPU-only wheel** (no CPU runtime — memory `cpu-is-test-only`).
- Contract: `docs/design/cli-bindings.md`.

## 2. I/O formats (USER ASK)
- **Older `.GENO` (classic PACKEDANCESTRYMAP, "GENO" magic, SNP-major) + EIGENSTRAT readers** — steppe is currently **TGENO-only** (`io::GenoReader::read_tile`). Add format-detect on the magic + dispatch so older AADR releases / non-v66 datasets load.
- **CI guard** — reject any golden built by AT2-R directly on a raw TGENO `.geno` (record format in golden metadata; fail if TGENO+AT2-R), so the corruption trap can't recur.

## 3. Step 3 — standalone f-stats, each WITH its own CLI/bindings (NOT built)
The f4 math is internal to the fit — there are no standalone entry points yet.
- **f4 / f3 / D-stat / f4-ratio / qpDstat** — each shipped with its own CLI + bindings.
- **Non-negative constrained weights** (deferred from the fit-engine-finish F5).
- Then **qpfstats / DATES / qpGraph**.

---

### Sequencing note (memory `build-sequence-backend-first`)
Backend finish (done) → finish the accessible surface for what exists (CLI + bindings, §1) → then new standalone stats each WITH its CLI/bindings (§3). The perf sweep (§0) and I/O formats (§2) are cross-cutting and can interleave. Per the user, **§0 (1240K wall-clock sweep) goes first**.
