#!/usr/bin/env bash
# download-aadr.sh — fetch an AADR v66 panel from Harvard Dataverse (curl, resumable, MD5-checked).
#
#   ./download-aadr.sh [PANEL] [OUTDIR]
#     PANEL   one of: 1240K | HO | 2M      (default: 1240K)
#     OUTDIR  where to put the files       (default: ./aadr_<PANEL>)
#
# Resolves fileIds from the AADR dataset API at RUNTIME (so it keeps working when the AADR
# publishes a new version), downloads the panel's .geno/.snp/.ind/.anno + the md5 manifest,
# and verifies each file. Sequential + a small pause between files (be kind to the server).
# The genotype is the big one (1240K ~7GB, HO ~4GB, 2M ~12GB) — resumable via `curl -C -`.
set -euo pipefail

PANEL="${1:-1240K}"
OUT="${2:-./aadr_${PANEL}}"
DOI="doi:10.7910/DVN/FFIDCW"                 # The Allen Ancient DNA Resource (AADR)
BASE="https://dataverse.harvard.edu"
UA="steppe-aadr-dl/1.0 (research; polite)"

case "$PANEL" in 1240K|HO|2M) ;; *) echo "usage: $0 [1240K|HO|2M] [outdir]" >&2; exit 2 ;; esac
command -v python3 >/dev/null || { echo "need python3 (to parse the dataset JSON)" >&2; exit 1; }
mkdir -p "$OUT"; cd "$OUT"

echo ">> resolving AADR v66 '$PANEL' files ($DOI) ..."
# fileId<TAB>label for this panel's data files + the md5 manifest.
# Match 'v66.p1_<PANEL>.aadr' and EXCLUDE the *_compatibility* variants.
MANIFEST="$(curl -sfL --max-time 90 -A "$UA" "$BASE/api/datasets/:persistentId/?persistentId=$DOI" \
  | PANEL="$PANEL" python3 -c '
import sys, os, json
panel = os.environ["PANEL"]
want = "v66.p1_%s.aadr" % panel
v = json.load(sys.stdin)["data"]["latestVersion"]
for f in v["files"]:
    lab = f.get("label", ""); fid = f["dataFile"]["id"]
    if (want in lab and "_compatibility" not in lab) or lab == "v66.p1__files.md5sum":
        print("%s\t%s" % (fid, lab))
')"
[ -n "$MANIFEST" ] || { echo "no files matched panel '$PANEL'" >&2; exit 1; }
echo "$MANIFEST" | sed 's/^/   /'

echo ">> downloading (resumable) ..."
while IFS=$'\t' read -r fid lab; do
  [ -z "$fid" ] && continue
  echo "   -> $lab"
  curl -fL -C - --retry 3 -A "$UA" -o "$lab" "$BASE/api/access/datafile/$fid"
  sleep 2
done <<< "$MANIFEST"

if [ -f v66.p1__files.md5sum ]; then
  echo ">> verifying MD5s ..."
  ok=1
  for f in v66.p1_${PANEL}.aadr*; do
    [ -e "$f" ] || continue
    exp="$(grep -m1 "  $f\$" v66.p1__files.md5sum 2>/dev/null | awk '{print $1}')"
    [ -n "$exp" ] || exp="$(grep -m1 " $f\$" v66.p1__files.md5sum 2>/dev/null | awk '{print $1}')"
    if [ -z "$exp" ]; then echo "   $f: (no entry in manifest — skipped)"; continue; fi
    got="$(md5sum "$f" | awk '{print $1}')"
    if [ "$exp" = "$got" ]; then echo "   $f: OK"; else echo "   $f: MD5 MISMATCH (exp $exp got $got)"; ok=0; fi
  done
  [ "$ok" = 1 ] && echo ">> all MD5s verified." || { echo ">> MD5 mismatch — re-run to resume/repair." >&2; exit 1; }
fi

echo ">> done. AADR '$PANEL' in: $OUT"
echo "   feed it to steppe:  steppe extract-f2 --prefix $OUT/v66.p1_${PANEL}.aadr.patch.PUB --auto-top-k 200 --maxmiss 0.5 --device 0 --out-dir f2_dir"
