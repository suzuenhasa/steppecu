#!/usr/bin/env bash
# box_bringup.sh <ssh-alias> [--build] — verify + set up a steppe GPU box (idempotent).
#   no flag : probe hardware/CUDA/disk/P2P/virt + install tooling (cmake/ninja/numpy).
#   --build : also rsync the repo, build + ctest (detached, drop-proof), and P2P-probe.
# Data setup is MANUAL (judgement call) — see docs/BOX-RUNBOOK.md §5.
# The SSH alias must already exist in ~/.ssh/config (RUNBOOK §0). Nothing builds locally.
set -uo pipefail
ALIAS="${1:?usage: box_bringup.sh <ssh-alias> [--build]}"
MODE="${2:-}"
LOCAL="${STEPPE_LOCAL:-/home/suzunik/steppe}"
CUDA='export PATH=/usr/local/cuda/bin:$PATH && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && ulimit -c 0'
say(){ printf '\n=== %s ===\n' "$*"; }

say "1. reachability + hardware ($ALIAS)"
ssh "$ALIAS" 'echo OK $(hostname); nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv | head -5' \
  || { echo "UNREACHABLE — fix the ~/.ssh/config alias (instances are ephemeral; RUNBOOK §0)"; exit 1; }

say "2. cuda / nvcc / disk / p2p / virt  (REJECT if CUDA<13.0, driver<580, disk<80G, cc!=12.0)"
ssh "$ALIAS" 'ls -d /usr/local/cuda-13* 2>/dev/null; /usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2; echo "--- disk ---"; df -h / | tail -1; echo "--- p2p (OK=>capable, NS=>consumer/host-staged) ---"; nvidia-smi topo -p2p r 2>/dev/null | sed -n "1,5p"; echo "--- virt ---"; systemd-detect-virt 2>/dev/null; ([ -f /.dockerenv ] && echo DOCKER || echo no-docker); echo "--- nccl ---"; ldconfig -p | grep -c nccl'

say "3. install tooling (idempotent)"
ssh "$ALIAS" 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq cmake ninja-build python3-numpy >/dev/null 2>&1; cmake --version | head -1; ninja --version; (rsync --version >/dev/null 2>&1 || apt-get install -y --reinstall rsync >/dev/null 2>&1); rsync --version | head -1'

if [ "$MODE" = "--build" ]; then
  say "4. rsync repo $LOCAL -> $ALIAS:/workspace/steppe"
  ssh "$ALIAS" 'mkdir -p /workspace/steppe'
  rsync -az --delete-after --exclude .git --exclude build --exclude aadr -e ssh "$LOCAL/" "$ALIAS:/workspace/steppe/" && echo "rsync ok"

  say "5. build + ctest (DETACHED on box + poll — drop-proof; needs data at /workspace/data/aadr)"
  ssh "$ALIAS" "cd /workspace/steppe && $CUDA && setsid bash -c \"cd /workspace/steppe && $CUDA && cmake -S . -B build -GNinja >/tmp/bringup.log 2>&1 && cmake --build build >>/tmp/bringup.log 2>&1 && echo ===CTEST=== >>/tmp/bringup.log && ctest --test-dir build >>/tmp/bringup.log 2>&1; echo ===DONE=== >>/tmp/bringup.log\" </dev/null >/dev/null 2>&1 & echo launched detached"
  echo "polling /tmp/bringup.log (up to ~5 min)..."
  ssh "$ALIAS" 'for i in $(seq 1 60); do grep -q ===DONE=== /tmp/bringup.log 2>/dev/null && break; sleep 5; done; echo "--- result ---"; grep -E "tests passed|tests failed|FAILED|[Ee]rror" /tmp/bringup.log | tail -10'

  say "6. P2P probe (real cudaMemcpyPeer)"
  scp -q "$LOCAL/scripts/p2p_probe.cu" "$ALIAS:/tmp/p2p_probe.cu" \
    && ssh "$ALIAS" "$CUDA && cd /tmp && nvcc -arch=sm_120 -O2 p2p_probe.cu -o p2p_probe 2>&1 && ./p2p_probe"
fi

say "done — data setup is manual (RUNBOOK §5: raw from local, derived_* regen via build_tgeno_matrix.py). Pass --build to rsync+build+test+p2p."
