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
Expect: clean build (warnings-as-errors) + all ctest green (36/36 on `m4.5-multigpu`). On `box5090` ctest ≈ 178 s (the CPU `filter_oracle` ≈ 88 s — slow box CPU, not a regression).

**⚠ PERF / BENCHMARKING — ALWAYS use a RELEASE build, NOT the raw `cmake -GNinja` above.** The plain `cmake -GNinja` (no `CMAKE_BUILD_TYPE`) is a **debug build (no `NDEBUG`)**, so `STEPPE_CUDA_CHECK_KERNEL` fires a `cudaDeviceSynchronize` **after every kernel launch** (check.cuh, gated by `STEPPE_DEBUG_ONLY`) — which serializes streams and made nsys traces unusable (~42% of API time). For any timing/bench, build **Release**: `cmake --preset release` (or `--preset ci` for tests too; or `-DCMAKE_BUILD_TYPE=Release` in a separate dir) — `NDEBUG` drops the per-kernel sync, "no forced sync in the hot path" (check.cuh doc). Correctness/ctest is fine on the debug build; **PERF NUMBERS ARE ONLY MEANINGFUL ON RELEASE.** (On Release, multi-GPU is now FASTER than single-GPU — measured 1.10x @ P=768, 1.22x @ P=400 on rtxbox; see 867a4bf / `docs/cleanup/m4.5/why-multigpu-slow.md`.)

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
Reclaim core dumps if the disk flooded: `ssh <alias> 'rm -f /var/lib/vastai_kaalia/data/core-* 2>/dev/null'`. rtxbox is $3.78/h — spin down when idle (its work: AT2 goldens, GDS, `ncu`/nsys re-profiling; the multi-GPU speedup is already measured + proven — 1.10x @ P=768).

---

## Per-box cheat-sheet
| | rtxbox (RTX PRO 6000) | box5090 (vast 2× 5090) |
|---|---|---|
| tier | CAPABLE (96 GB, P2P OK stock driver) | BUDGET/consumer (32 GB, P2P disabled) |
| `can_access_peer` | **true** → P2P device-combine | **false** → host-staged combine |
| runs | AT2 goldens · GDS · `ncu`/nsys re-profiling (speedup already proven 1.10x) | the fix-passes · graceful-degrade target · daily dev |
| P_max single-GPU | ~P=768+ fits | P=768 OOMs (sharding required ≥~P=700) |
| gotchas | ephemeral (update alias) · nvcc PATH · $$$ | flaky net (detached runs) · slow CPU · nvcc PATH |
