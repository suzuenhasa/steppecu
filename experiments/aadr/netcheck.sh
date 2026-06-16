#!/usr/bin/env bash
# Network diagnostic: distinguish slow box egress vs Dataverse per-connection throttling.
set -u
GENO_URL="https://dataverse.harvard.edu/api/access/datafile/13994808"
RAW=/workspace/data/aadr/raw
mkdir -p "$RAW"

human() { awk -v b="$1" 'BEGIN{ split("B KB MB GB",u); s=b; i=1; while(s>=1024&&i<4){s/=1024;i++}; printf "%.1f %s/s\n", s, u[i] }'; }

echo "=== [1] box egress vs Cloudflare (curl --max-time 12) ==="
sp=$(curl -s -o /dev/null --max-time 12 -w '%{speed_download}' "https://speed.cloudflare.com/__down?bytes=200000000" 2>/dev/null)
echo "cloudflare: ${sp:-0} B/s -> $(human "${sp:-0}")"

echo "=== [2] Dataverse single connection (curl --max-time 12) ==="
sp=$(curl -sL -o /dev/null --max-time 12 -w '%{speed_download}' "$GENO_URL" 2>/dev/null)
echo "dataverse 1-conn: ${sp:-0} B/s -> $(human "${sp:-0}")"

echo "=== [3] Dataverse aria2 16-connection (25s sample) ==="
cd "$RAW"
rm -f probe.geno probe.geno.aria2
timeout 25 aria2c -x16 -s16 -k1M --console-log-level=warn --summary-interval=5 \
  -o probe.geno "$GENO_URL" >/tmp/aria.log 2>&1
got=$(stat -c%s probe.geno 2>/dev/null || echo 0)
echo "aria2 16-conn: downloaded ${got} bytes in ~25s -> $(human $(( got / 25 )))"
echo "--- aria2 log tail ---"; tail -4 /tmp/aria.log
rm -f probe.geno probe.geno.aria2

echo "=== [4] verdict hint ==="
echo "if [3] >> [2], Dataverse throttles per-connection -> use aria2 -x16 for the real download."
