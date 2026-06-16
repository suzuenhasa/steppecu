#!/usr/bin/env bash
# Grab a CONTIGUOUS prefix (header + first KSNP records) of the AADR HO packed .geno,
# via NCONN parallel byte-range curls (the box egress is ~650 KB/s aggregate; ranges
# just keep it saturated). The packed format is header(rlen) + nsnp records of rlen
# bytes each, so a [0, rlen*(KSNP+1)) prefix is a valid file for the first KSNP SNPs.
# Usage: prefix_download.sh [KSNP=130000] [NCONN=16]
set -euo pipefail
RAW=/workspace/data/aadr/raw
KSNP=${1:-130000}; NCONN=${2:-16}
URL="https://dataverse.harvard.edu/api/access/datafile/13994808"   # HO .geno, AADR v66.p1
cd "$RAW"

echo "==> stopping any full download"
pkill -f aria2c 2>/dev/null || true; sleep 1
rm -f v66.p1_HO.aadr.patch.PUB.geno v66.p1_HO.aadr.patch.PUB.geno.aria2 || true   # reclaim the abandoned preallocated full file

nind=$(grep -cve '^[[:space:]]*$' v66.p1_HO.aadr.patch.PUB.ind)
rlen=$(( (nind + 3) / 4 )); (( rlen < 48 )) && rlen=48
bytes=$(( rlen * (KSNP + 1) ))      # header record + KSNP SNP records
echo "nind=$nind rlen=$rlen KSNP=$KSNP prefix_bytes=$bytes (~$(( bytes/1024/1024 )) MiB)"

rm -f part_* prefix.geno
chunk=$(( (bytes + NCONN - 1) / NCONN ))
echo "==> $NCONN parallel ranged curls, chunk=$chunk"
pids=()
for ((c=0;c<NCONN;c++)); do
  s=$(( c*chunk )); e=$(( s+chunk-1 )); (( e >= bytes )) && e=$(( bytes-1 ))
  (( s > e )) && continue
  curl -sL --fail -r "${s}-${e}" "$URL" -o "part_$(printf %03d "$c")" &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && { echo "ERROR: a ranged curl failed"; exit 1; }

cat part_* > prefix.geno
rm -f part_*
got=$(stat -c%s prefix.geno)
echo "prefix.geno: $got bytes (want $bytes)"
[ "$got" -eq "$bytes" ] || { echo "ERROR: size mismatch"; exit 1; }
head -c4 prefix.geno | grep -q '^GENO' || { echo "ERROR: bad GENO magic"; exit 1; }

# Present a triple for build_matrix --prefix HOpref  (geno=prefix, snp/ind=full)
ln -sf prefix.geno                       HOpref.geno
ln -sf v66.p1_HO.aadr.patch.PUB.snp      HOpref.snp
ln -sf v66.p1_HO.aadr.patch.PUB.ind      HOpref.ind
echo "PREFIX_READY  prefix=$RAW/HOpref  snps_available=$KSNP  nind=$nind"
df -h /workspace | tail -1
