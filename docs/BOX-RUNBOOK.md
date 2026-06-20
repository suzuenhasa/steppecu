# steppe — GPU box runbook (set up / update / verify a remote instance)

The repeatable procedure for standing up (or refreshing) a remote GPU box to build &
test `steppe`. **Read this when a new instance is provisioned (e.g. a fresh rtxbox) so
you don't re-derive the setup.** Rationale for each gotcha lives in the `[[rtxbox]]` /
`[[steppebox5090]]` / `[[steppe-dev-process]]` memories. Automated version:
`scripts/box_bringup.sh <ssh-alias>`. P2P check: `scripts/p2p_probe.cu`.

Nothing builds locally — author locally → rsync → build/test on the box.

---

## 0. SSH alias (do this first; instances are EPHEMERAL)
vast/Verda rentals get a new IP/port each time. Add (or update) an alias in `~/.ssh/config`:
```
Host rtxbox            # or box5090
    HostName  <ip>
    User      root
    Port      <port>   # vast: e.g. 43215; Verda: usually 22
    IdentityFile ~/.ssh/id_vastai      # the key that works for both providers
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ConnectTimeout 15
```
Confirm: `ssh <alias> 'echo OK $(hostname)'`.

## 1. VERIFY the box (reject early — a wrong box costs hours)
```
ssh <alias> 'nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv'
ssh <alias> 'nvidia-smi | head -4'                       # CUDA version (top-right)
ssh <alias> 'ls -d /usr/local/cuda-13* && /usr/local/cuda/bin/nvcc --version | tail -2'
ssh <alias> 'df -h /'                                     # disk
ssh <alias> 'nvidia-smi topo -p2p r'                      # P2P capability (OK vs NS)
ssh <alias> 'systemd-detect-virt; [ -f /.dockerenv ] && echo DOCKER || echo "no docker"'
ssh <alias> 'nproc; free -h | head -2; ldconfig -p | grep -c nccl'
```
**REJECT if:** CUDA < 13.0 / driver < 580 (the fixed-slice Ozaki FP64 emulation — the core IP — needs CUDA 13.0 U2+; rentals can't upgrade the driver), or disk < ~80 GB (full AADR ≈30 GB + build + tensors), or compute_cap ≠ 12.0 (sm_120). **ACCEPT** = 2× sm_120, CUDA 13.0.x, driver ≥580, ≥80 GB.
**P2P note:** `topo -p2p r` = OK is necessary but not sufficient — verify the real DMA (step 6). Consumer GeForce 5090 = `NS` (driver-disabled, host-staged only); RTX PRO 6000 = OK on stock driver.

## 2. Install tooling (idempotent)
```
ssh <alias> 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq cmake ninja-build python3-numpy >/dev/null 2>&1; cmake --version | head -1; ninja --version'
# rsync sometimes ships as a 0-byte stub — fix if `rsync --version` fails:
ssh <alias> 'rsync --version | head -1 || apt-get install -y --reinstall rsync'
```

## 3. ⚠ THE nvcc-PATH GOTCHA (causes a bogus "Failed to detect a default CUDA architecture")
`nvcc` is often NOT on PATH. Prefix EVERY build/test command:
```
export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0
```
(`ulimit -c 0` — fail-fast/assert tests `abort()`; the kernel can dump a ~404 MB core per abort and fill the disk. Reclaim a flooded box with `rm -f /var/lib/vastai_kaalia/data/core-*`.)

## 4. Sync the repo (from local)
```
rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh /home/suzunik/steppe/ <alias>:/workspace/steppe/
```

## 5. Get the data → `/workspace/data/aadr/`
The data is `raw/` (the v66 TGENO triple, ~3.8 GB) + `derived_*` Q/V/N matrices. **The vast↔box network is SLOW (~1 MB/s) — do NOT bulk-pull `derived_*` between boxes.** Instead:
- **`raw`**: rsync from LOCAL (`~/steppe/aadr/v66.p1_HO.aadr.patch.PUB.{geno,snp,ind}` → `<alias>:/workspace/data/aadr/raw/`), ~10 MB/s.
- **`derived_*`**: REGENERATE on the box from raw (faster than the slow download; the M4.5 parity test is internal so byte-matching another box is unnecessary). Needs `python3-numpy` (step 2) + the generator:
```
scp /home/suzunik/steppe/aadr/build_tgeno_matrix.py <alias>:/workspace/data/aadr/
ssh <alias> 'cd /workspace/data/aadr && python3 build_tgeno_matrix.py --geno raw/v66.p1_HO.aadr.patch.PUB.geno --ind raw/v66.p1_HO.aadr.patch.PUB.ind --out derived_acc  --auto-top 50  --snp-cap 100000'   # P=50  M=100k
ssh <alias> 'cd /workspace/data/aadr && python3 build_tgeno_matrix.py --geno raw/v66.p1_HO.aadr.patch.PUB.geno --ind raw/v66.p1_HO.aadr.patch.PUB.ind --out derived_full --auto-top 768'                    # P=768 M=584131
```
(Alt: box-to-box rsync via SSH agent forwarding — no key left on the box — `eval $(ssh-agent); ssh-add ~/.ssh/id_vastai; ssh -A <alias> "rsync -a -e 'ssh -p <vastport>' root@<vastip>:/workspace/data/aadr/derived_full /workspace/data/aadr/"` — only worth it if the source box is fast.)

## 6. Build + sanity test
```
ssh <alias> 'cd /workspace/steppe && export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && cmake -S . -B build -GNinja && cmake --build build && ctest --test-dir build --output-on-failure'
```
Expect: clean build (warnings-as-errors) + all ctest green (36/36 on `m4.5-multigpu`). On `box5090` ctest ≈ 178 s (the CPU `filter_oracle` ≈ 88 s — slow box CPU, not a regression). The **qpadm portion is now FAST by default** (`qpadm_parity` ≈ 0.7 s, `qpadm_rotation` ≈ 3.5 s) — see the FAST/THOROUGH note below.

Default `ctest` is the **FAST** dev loop: it validates the qpAdm fit/rotation **GPU-vs-AT2-golden ONLY** (`qpadm_parity` = the 9-pop GPU full fit incl. SE vs `golden_fit0` — the SE-math validation; `qpadm_rotation` = the 84-model GPU rotation vs `golden_rot` + the G1==G2 determinism gate + NRBIG-via-`jackknife=None` vs `golden_fit1` for f4rank/rankdrop/popdrop with the LOO SE skipped). The slow CpuBackend oracle re-derivation + the NRBIG GPU full LOO-SE + the synthetic-throughput sweep are **opt-in**: run `STEPPE_THOROUGH=1 ctest --test-dir build -R 'qpadm' --output-on-failure` for the full proof (the CpuBackend diff oracle + the GPU-vs-CPU localizers + the NRBIG full SE + the at-scale throughput). The CpuBackend also runs automatically in FAST mode when **no GPU is visible** (`CUDA_VISIBLE_DEVICES=""`) — the CI-without-GPU acceptance gate. Target: plain `ctest` qpadm tests < 90 s. No golden tolerance is weakened and the SE math is unchanged; coverage is moved (FAST vs THOROUGH), not lost.

**⚠ PERF / BENCHMARKING — ALWAYS use a RELEASE build, NOT the raw `cmake -GNinja` above.** The plain `cmake -GNinja` (no `CMAKE_BUILD_TYPE`) is a **debug build (no `NDEBUG`)**, so `STEPPE_CUDA_CHECK_KERNEL` fires a `cudaDeviceSynchronize` **after every kernel launch** (check.cuh, gated by `STEPPE_DEBUG_ONLY`) — which serializes streams and made nsys traces unusable (~42% of API time). For any timing/bench, build **Release**: `cmake --preset release` (or `--preset ci` for tests too; or `-DCMAKE_BUILD_TYPE=Release` in a separate dir) — `NDEBUG` drops the per-kernel sync, "no forced sync in the hot path" (check.cuh doc). Correctness/ctest is fine on the debug build; **PERF NUMBERS ARE ONLY MEANINGFUL ON RELEASE.** (Honest perf story: on the precompute, multi-GPU is a **modest throughput layer** — the within-M4.5 device-resident combine `867a4bf` made the 2-GPU path 1.10× @ P=768 / 1.22× @ P=400 on rtxbox, but that was then superseded: the REAL precompute wins were getting the result **off the CPU** — device-resident output `1f80c0c`, ~4.3× @ P=512 — and **M5 streaming** `176a07d`/`c65179f`. Multi-GPU's proper home is the Phase-2 fit/rotation. See `docs/cleanup/m5/00-results.md` + `docs/cleanup/m4.5/why-multigpu-slow.md`.)

## 7. ⚠ LONG-JOB DETACHED RUN (box5090's network drops long ssh connections)
A >~3 min ssh (clean build + ctest, a big bench) can be silently cut. Run detached on the box + poll a logfile:
```
ssh <alias> 'cd /workspace/steppe && export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0 && setsid bash -c "ctest --test-dir build --output-on-failure > /tmp/run.log 2>&1; echo ===DONE=== >> /tmp/run.log" </dev/null >/dev/null 2>&1 & echo launched'
# poll (short reads survive drops):
ssh <alias> 'for i in $(seq 1 40); do grep -q ===DONE=== /tmp/run.log && break; sleep 5; done; tail -40 /tmp/run.log'
```

## 8. Verify real P2P (the capable-path gate — PRO 6000 / datacenter only)
`topo -p2p r` = OK is not proof; measure an actual byte-exact `cudaMemcpyPeer`:
```
scp scripts/p2p_probe.cu <alias>:/tmp/ && ssh <alias> 'export PATH=/usr/local/cuda/bin:$PATH && cd /tmp && nvcc -arch=sm_120 -O2 p2p_probe.cu -o p2p_probe && ./p2p_probe'
```
Expect on a PRO 6000: `cudaDeviceCanAccessPeer 0->1=1 1->0=1`, `byte-exact = YES`, ~55 GB/s. On a consumer 5090: `canAccessPeer=0` (expected — host-staged only).

## 9. Spin-down
Reclaim core dumps if the disk flooded: `ssh <alias> 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'`. rtxbox is $3.78/h — spin down when idle (its work: AT2 goldens, GDS, `ncu`/nsys re-profiling, the deferred device-resident final-D2H pinning; the precompute perf rabbit-hole is closed — the win was off-CPU + streaming, not multi-GPU).

---

## Per-box cheat-sheet
| | rtxbox (RTX PRO 6000) | box5090 (vast 2× 5090) |
|---|---|---|
| tier | CAPABLE (96 GB, P2P OK stock driver) | BUDGET/consumer (32 GB, P2P disabled) |
| `can_access_peer` | **true** → P2P device-combine | **false** → host-staged combine |
| runs | AT2 goldens · GDS · `ncu`/nsys re-profiling · deferred D2H pinning | the M5 single-GPU streamed sweep (P=2500=51.5s) · graceful-degrade · daily dev |
| qpAdm fit / S8 rotation | multi-GPU viable (P2P avoids the host bounce) | **RUN SINGLE-GPU** — no P2P ⇒ f2 replication host-bounces (~3.8s/8.72GB, only ~1.21× @ 9086 real models); multi-GPU rotation deferred (`TODO(multigpu-host-bounce)`) |
| P_max single-GPU | ~P=768+ in-core; full-autosome P=2500 via M5 streaming | in-core ~P=700; **full-autosome P=2500 completes via M5 streaming (peak ~26 GB)** |
| gotchas | ephemeral (update alias) · nvcc PATH · $$$ | flaky net (detached runs) · slow CPU · nvcc PATH |
